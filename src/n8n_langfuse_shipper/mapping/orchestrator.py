"""Central mapping orchestration for n8n execution to Langfuse trace conversion.

This module coordinates the complete transformation pipeline, iterating through node
runs in chronological order and delegating specialized tasks to helper modules. It
maintains mapping state via MappingContext and assembles the final LangfuseTrace.

Core orchestration function `_map_execution` performs these steps:
    1. Create trace with deterministic ID from execution ID
    2. Build root span representing entire execution
    3. Construct reverse edge graph and agent hierarchy maps
    4. Sort all node runs by startTime for chronological span ordering
    5. For each node run:
        - Resolve parent span via precedence rules
        - Detect generation spans and extract usage
        - Normalize I/O structures and strip system prompts
        - Collect binary assets (when enabled)
        - Serialize and optionally truncate I/O
        - Extract model metadata
        - Detect Gemini empty output anomaly
        - Create span with all metadata
    6. Return complete trace and collected assets

Helper Functions:
    _serialize_and_truncate: JSON serialization with unconditional binary stripping
    _flatten_runs: Sort node runs chronologically across all nodes
    _prepare_io_and_output: System prompt strip, normalize, serialize, truncate
    _extract_model_and_metadata: Model name extraction with parameter fallback
    _detect_gemini_empty_output_anomaly: Gemini bug detection with tool_calls suppression
    _collect_binary_assets: Extract binary data and insert pending placeholders
    _discover_additional_binary_assets: Scan for data URLs and file-like structures
    _apply_ai_filter: AI-only filtering with context window preservation

Design Notes:
    - Pure functions only; no network or database I/O
    - Deterministic IDs via UUIDv5 with stable namespace
    - Binary stripping precedes truncation (independent operations)
    - Input propagation uses parent output when inputOverride absent
    - Size guard on cached outputs when truncation enabled
"""
from __future__ import annotations

import base64 as _b64
import copy
import hashlib as _hash
import json
import logging
import re
from datetime import timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid5

from ..models.langfuse import LangfuseSpan, LangfuseTrace
from ..models.n8n import N8nExecutionRecord, NodeRun, WorkflowNode
from ..observation_mapper import is_ai_node, map_node_to_observation_type
from .binary_sanitizer import (
    contains_binary_marker as _contains_binary_marker,
)
from .binary_sanitizer import (
    strip_binary_payload as _strip_binary_payload,
)
from .generation import (
    detect_generation as _detect_generation,
)
from .generation import (
    extract_concise_output as _extract_concise_output,
)
from .generation import (
    extract_usage as _extract_usage,
)
from .id_utils import SPAN_NAMESPACE
from .io_normalizer import (
    extract_generation_input_and_params as _extract_generation_input_and_params,
)
from .io_normalizer import (
    normalize_node_io as _normalize_node_io,
)
from .io_normalizer import (
    strip_system_prompt_from_langchain_lmchat as _strip_system_prompt_from_langchain_lmchat,
)
from .mapping_context import MappingContext
from .model_extractor import (
    extract_model_from_parameters as _extract_model_from_parameters,
)
from .model_extractor import (
    extract_model_value as _extract_model_value,
)
from .model_extractor import (
    looks_like_model_param_key as _looks_like_model_param_key,
)
from .node_extractor import (
    extract_nodes_data as _extract_nodes_data,
)
from .parent_resolution import (
    build_child_agent_map as _build_child_agent_map,
)
from .parent_resolution import (
    build_reverse_edges as _build_reverse_edges,
)
from .parent_resolution import (
    resolve_parent as _resolve_parent,
)
from .prompt_detection import (
    build_prompt_registry as _build_prompt_registry,
)
from .prompt_resolution import (
    resolve_prompt_for_generation as _resolve_prompt_for_generation,
)
from .prompt_version_resolver import (
    create_version_resolver_from_env as _create_version_resolver_from_env,
)
from .time_utils import epoch_ms_to_dt as _epoch_ms_to_dt

logger = logging.getLogger(__name__)


def _serialize_and_truncate(
    obj: Any, limit: Optional[int]
) -> Tuple[Optional[str], bool]:
    """Serialize object to JSON with unconditional binary stripping and optional truncation.

    Binary stripping always applies regardless of truncation setting to prevent
    large base64 payloads in output. Truncation applies only when limit > 0.

    Args:
        obj: Data structure to serialize (typically normalized node I/O)
        limit: Maximum characters before truncation; None or <=0 disables truncation

    Returns:
        Tuple of (serialized_string, truncated_flag)
        - serialized_string: JSON or string representation, None if input None
        - truncated_flag: True if output exceeded limit and was shortened

    Note:
        Binary detection errors fail open (continue without stripping). JSON
        serialization failures fall back to str() representation.
    """
    if obj is None:
        return None, False
    try:
        if _contains_binary_marker(obj):
            obj = _strip_binary_payload(obj)
    except Exception:
        pass
    try:
        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        raw = str(obj)
    truncated = False
    if isinstance(limit, int) and limit > 0 and len(raw) > limit:
        truncated = True
        raw = raw[:limit] + f"...<truncated {len(raw)-limit} chars>"
    return raw, truncated


def _flatten_runs(
    run_data: Dict[str, List[NodeRun]]
) -> List[Tuple[int, str, int, NodeRun]]:
    """Flatten and sort all node runs by startTime for chronological span ordering.

    Extracts runs from nested runData structure and sorts by execution start timestamp
    to ensure parent spans are emitted before their children in the final trace.

    Args:
        run_data: Mapping of node_name to list of run instances

    Returns:
        List of tuples (startTime, node_name, run_index, NodeRun) sorted chronologically

    Note:
        Run index within tuple represents position in the original node's run list,
        used for deterministic span ID generation.
    """
    flattened: List[Tuple[int, str, int, NodeRun]] = []
    for node_name, runs in run_data.items():
        for idx, run in enumerate(runs):
            flattened.append((run.startTime, node_name, idx, run))
    flattened.sort(key=lambda x: x[0])
    return flattened


def _extract_model_and_metadata(
    *,
    run: NodeRun,
    node_name: str,
    node_type: str,
    is_generation: bool,
    wf_node_obj: Dict[str, WorkflowNode],
    raw_input_obj: Any,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Extract AI model name and diagnostic metadata from node run data.

    Performs multi-pass search for model identifiers:
    1. Top-level model field in run.data
    2. Breadth-first nested search for model/model_name/modelId/model_id keys
    3. Fallback to static parameters in workflow definition (generation spans only)

    Args:
        run: NodeRun execution instance with runtime data
        node_name: Node identifier for workflow lookup
        node_type: Node type string for generation detection
        is_generation: Whether span classified as LLM generation
        wf_node_obj: Workflow node definition map for parameter fallback
        raw_input_obj: Propagated or explicit input for extended search

    Returns:
        Tuple of (model_value, metadata_dict)
        - model_value: Extracted model string or None if not found
        - metadata_dict: Diagnostic flags like n8n.model.missing, n8n.model.from_parameters

    Note:
        Parameter fallback triggers opportunistically when model-like keys detected
        in static parameters, even for non-generation spans.
    """
    metadata: Dict[str, Any] = {}
    model_val: Optional[str] = None
    try:
        if isinstance(run.data, dict):
            search_root: Dict[str, Any] = dict(run.data)
            if isinstance(run.inputOverride, dict):
                search_root["_inputOverride"] = run.inputOverride
            if raw_input_obj and isinstance(raw_input_obj, dict):
                search_root["_inferredInput"] = raw_input_obj
            top_model = run.data.get("model")
            if isinstance(top_model, str) and top_model:
                model_val = top_model
            else:
                model_val = _extract_model_value(search_root)
    except Exception:
        pass
    if model_val is None:
        attempt_param_fallback = is_generation
        node_static = wf_node_obj.get(node_name)
        try:
            if (
                not attempt_param_fallback
                and node_static
                and isinstance(getattr(node_static, "parameters", None), dict)
            ):
                params_dict = node_static.parameters or {}
                for pk in params_dict.keys():
                    if _looks_like_model_param_key(pk):
                        attempt_param_fallback = True
                        break
                if not attempt_param_fallback:
                    for _sk, sv in params_dict.items():
                        if isinstance(sv, dict):
                            for pk in sv.keys():
                                if _looks_like_model_param_key(pk):
                                    attempt_param_fallback = True
                                    break
                        if attempt_param_fallback:
                            break
        except Exception:
            pass
        if attempt_param_fallback and node_static is not None:
            try:
                fallback = _extract_model_from_parameters(node_static)
            except Exception:
                fallback = None
            if fallback:
                model_val, key_path, extra_meta = fallback
                metadata["n8n.model.from_parameters"] = True
                metadata["n8n.model.parameter_key"] = key_path
                for k, v in extra_meta.items():
                    metadata[k] = v
    if model_val is None and is_generation:
        metadata["n8n.model.missing"] = True
        try:
            if isinstance(run.data, dict):
                metadata["n8n.model.search_keys"] = list(run.data.keys())[:12]
        except Exception:
            pass
    return model_val, metadata


def _detect_gemini_empty_output_anomaly(
    *,
    is_generation: bool,
    norm_output_obj: Any,
    run: NodeRun,
    node_name: str,
    next_observation_type: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Detect Gemini/Vertex chat empty output bug with tool_calls suppression logic.

    Identifies anomalous empty text response with non-zero prompt tokens, distinguishing
    genuine bugs from intentional tool invocation transitions (similar to OpenAI
    finish_reason=tool_calls).

    Anomaly conditions (all must be true):
    1. span is generation
    2. generations[0][0].text == ""
    3. promptTokens > 0
    4. totalTokens >= promptTokens
    5. completionTokens == 0 or absent

    Suppression path when next span is tool observation:
    - Status remains original (not forced to error)
    - Metadata flags n8n.gen.tool_calls_pending=true
    - Token counters preserved for analytics

    Error enforcement path otherwise:
    - Status forced to "error"
    - Synthetic error message inserted when original error absent
    - Metadata flags n8n.gen.empty_output_bug=true
    - Structured log entry emitted

    Args:
        is_generation: Whether span classified as LLM generation
        norm_output_obj: Normalized output structure for text extraction
        run: NodeRun containing tokenUsage and error data
        node_name: Node identifier for logging
        next_observation_type: Lookahead observation type for suppression logic

    Returns:
        Tuple of (status_override, metadata_dict)
        - status_override: "error" when anomaly detected and not suppressed, else None
        - metadata_dict: Diagnostic flags and token counters
    """
    if not is_generation:
        return None, {}
    meta: Dict[str, Any] = {}
    try:
        search_struct: Any = norm_output_obj if norm_output_obj is not None else run.data
        tu_source = search_struct if isinstance(search_struct, dict) else run.data
        tu = (
            (
                tu_source.get("tokenUsage")
                if isinstance(tu_source, dict)
                else None
            )
            or (
                tu_source.get("tokenUsageEstimate")
                if isinstance(tu_source, dict)
                else None
            )
            or {}
        )
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if isinstance(tu, dict):
            prompt_tokens = (
                tu.get("promptTokens")
                or tu.get("inputTokens")
                or tu.get("input_token_count")
            )
            completion_tokens = (
                tu.get("completionTokens")
                or tu.get("outputTokens")
                or tu.get("output_token_count")
            )
            total_tokens = tu.get("totalTokens") or tu.get("tokenCount")
        empty_text = False
        gen_block = search_struct
        if (
            isinstance(search_struct, dict)
            and "response" in search_struct
            and isinstance(search_struct["response"], dict)
        ):
            gen_block = search_struct["response"]
        try:
            gens = gen_block.get("generations") if isinstance(gen_block, dict) else None
            if isinstance(gens, list) and gens:
                first_layer = gens[0]
                if isinstance(first_layer, list) and first_layer:
                    inner = first_layer[0]
                    if isinstance(inner, dict):
                        txt = inner.get("text")
                        gen_info = inner.get("generationInfo")
                        if isinstance(txt, str) and txt == "":
                            empty_text = True
                        if isinstance(gen_info, dict) and len(gen_info) == 0:
                            meta["n8n.gen.empty_generation_info"] = True
        except Exception:
            pass
        anomaly = (
            empty_text
            and isinstance(prompt_tokens, int)
            and prompt_tokens > 0
            and (isinstance(total_tokens, int) and total_tokens >= prompt_tokens)
            and (completion_tokens in (0, None))
        )
        if anomaly:
            # Suppress error status when immediately followed by a tool span.
            # This represents an intentional "tool_calls" transition (similar
            # to Azure OpenAI finish_reason=tool_calls) rather than an empty
            # output bug. Metadata indicates suppression for downstream
            # analytics while preserving token context.
            if next_observation_type == "tool":
                meta["n8n.gen.tool_calls_pending"] = True
                meta["n8n.gen.prompt_tokens"] = prompt_tokens
                if total_tokens is not None:
                    meta["n8n.gen.total_tokens"] = total_tokens
                meta["n8n.gen.completion_tokens"] = completion_tokens or 0
                return None, meta
            meta["n8n.gen.empty_output_bug"] = True
            meta["n8n.gen.prompt_tokens"] = prompt_tokens
            if total_tokens is not None:
                meta["n8n.gen.total_tokens"] = total_tokens
            meta["n8n.gen.completion_tokens"] = completion_tokens or 0
            if not run.error:
                run.error = {"message": "Gemini empty output anomaly detected"}
            try:
                logger.warning(
                    (
                        "gemini_empty_output_anomaly span=%s "
                        "prompt_tokens=%s total_tokens=%s completion_tokens=%s"
                    ),
                    node_name,
                    prompt_tokens,
                    total_tokens,
                    completion_tokens,
                )
            except Exception:
                pass
            return "error", meta
    except Exception:
        return None, {}
    return None, meta


def _prepare_io_and_output(
    *,
    raw_input_obj: Any,
    raw_output_obj: Any,
    is_generation: bool,
    node_type: str,
    truncate_limit: Optional[int],
) -> Tuple[Any, Any, Optional[str], bool, Optional[str], bool, Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
    """Normalize, serialize, and truncate node I/O with generation-specific handling.

    Orchestrates the complete I/O transformation pipeline:
    1. Strip system prompts from LangChain LMChat inputs (generation spans only)
    2. Extract LLM parameters from generation input (messages[0] as input, rest as metadata)
    3. Unwrap AI channel and generic JSON wrapper structures
    4. Merge back binary blocks if lost during unwrapping
    5. Serialize to JSON with binary stripping
    6. Extract concise output text for generation spans
    7. Apply optional truncation when limit > 0

    Args:
        raw_input_obj: Original input data (inputOverride or propagated parent output)
        raw_output_obj: Original output data from run.data
        is_generation: Whether span classified as LLM generation
        node_type: Node type string for generation-specific transformations
        truncate_limit: Max characters before truncation; None or <=0 disables

    Returns:
        9-tuple containing:
        - norm_input_obj: Normalized input structure
        - norm_output_obj: Normalized output structure
        - input_str: Serialized input string (None if input None)
        - input_trunc: True if input exceeded limit
        - output_str: Serialized output string (concise for generation, JSON otherwise)
        - output_trunc: True if output exceeded limit
        - input_flags: Transformation flags (unwrapped_ai_channel, unwrapped_json_root)
        - output_flags: Transformation flags including promoted_binary_wrapper
        - llm_params_metadata: Dict with n8n.llm.* keys for generation config params
    """
    llm_params_metadata: Dict[str, Any] = {}

    if is_generation:
        raw_input_obj = _strip_system_prompt_from_langchain_lmchat(
            raw_input_obj, node_type
        )
        # Extract LLM parameters and clean input (messages[0])
        raw_input_obj, llm_params_metadata = _extract_generation_input_and_params(
            raw_input_obj, is_generation
        )
    norm_input_obj, input_flags = _normalize_node_io(raw_input_obj)
    norm_output_obj, output_flags = _normalize_node_io(raw_output_obj)
    try:
        if (
            isinstance(raw_output_obj, dict)
            and "binary" in raw_output_obj
            and isinstance(raw_output_obj["binary"], dict)
            and isinstance(norm_output_obj, list)
        ):
            norm_output_obj = {"binary": raw_output_obj["binary"], "_items": norm_output_obj}
            output_flags["promoted_binary_wrapper"] = True
    except Exception:
        pass
    input_str, input_trunc = _serialize_and_truncate(norm_input_obj, truncate_limit)
    output_str: Optional[str] = None
    output_trunc = False
    if is_generation:
        extracted_text = _extract_concise_output(norm_output_obj, node_type)
        if extracted_text is not None:
            output_str = extracted_text
    if output_str is None:
        output_str, output_trunc = _serialize_and_truncate(norm_output_obj, truncate_limit)
    return (
        norm_input_obj,
        norm_output_obj,
        input_str,
        input_trunc,
        output_str,
        output_trunc,
        input_flags,
        output_flags,
        llm_params_metadata,
    )


_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$")


def _discover_additional_binary_assets(
    obj: Any,
    *,
    execution_id: int,
    node_name: str,
    run_index: int,
    max_assets: int,
) -> tuple[list[Any], bool]:
    """Scan for non-canonical binary assets like data URLs and file-like dicts.

    Extends binary discovery beyond the standard run.data.binary block to find:
    - Data URLs matching data:<mime>;base64,<payload>
    - File-like dicts with mimeType/fileName + data field
    - Long base64 strings (≥64 chars) in contexts with file-indicative sibling keys

    Discovery bounded by max_assets limit to prevent excessive scanning on
    malformed or adversarial inputs.

    Args:
        obj: Data structure to scan (typically normalized output)
        execution_id: Execution ID for asset metadata
        node_name: Node name for asset metadata
        run_index: Run index for asset metadata
        max_assets: Maximum assets to collect; <=0 disables extended scan

    Returns:
        Tuple of (discovered_assets, limit_hit_flag)
        - discovered_assets: List of BinaryAsset instances
        - limit_hit_flag: True if scan stopped due to max_assets limit
    """
    from ..media_api import BinaryAsset
    found: list[BinaryAsset] = []
    limit_hit = False
    seen_sha: set[str] = set()
    def _add(base64_str: str, mime: str | None, fname: str | None, path: str) -> None:
        nonlocal limit_hit
        if max_assets > 0 and len(found) >= max_assets:
            limit_hit = True
            return
        try:
            if len(base64_str) < 64:
                return
            full_bytes = _b64.b64decode(base64_str, validate=False)
            h = _hash.sha256(full_bytes).hexdigest()
            if h in seen_sha:
                return
            seen_sha.add(h)
            clean_path = re.sub(r"\[[0-9]+\]", "", path)
            path_segments = [seg for seg in clean_path.split(".") if seg and seg != "<root>"]
            found.append(
                BinaryAsset(
                    execution_id=execution_id,
                    node_name=node_name,
                    run_index=run_index,
                    field_path=path,
                    original_path_segments=path_segments,
                    mime_type=mime if isinstance(mime, str) else None,
                    filename=fname if isinstance(fname, str) else None,
                    size_bytes=len(full_bytes),
                    sha256=h,
                    base64_len=len(base64_str),
                    content_b64=base64_str,
                )
            )
        except Exception:
            return
    def _walk(o: Any, path: str, parent: Any | None) -> None:  # noqa: C901
        nonlocal limit_hit
        if max_assets > 0 and len(found) >= max_assets:
            limit_hit = True
            return
        if isinstance(o, dict):
            mime = o.get("mimeType") or o.get("mime_type") or o.get("fileType")
            fname = o.get("fileName") or o.get("name")
            data_field = o.get("data")
            if isinstance(data_field, str):
                m = _DATA_URL_RE.match(data_field)
                if m and len(m.group(2)) >= 64:
                    _add(m.group(2), m.group(1), fname, f"{path}.data" if path else "data")
                else:
                    if (mime or fname) and len(data_field) >= 64:
                        _add(data_field, mime, fname, f"{path}.data" if path else "data")
            for k, v in o.items():
                if isinstance(v, str) and len(v) >= 64:
                    sib_keys = {sk.lower() for sk in o.keys()}
                    if any(h in sib_keys for h in ("mimetype", "filename", "filetype")):
                        _add(v, mime, fname, f"{path}.{k}" if path else k)
            for k, v in list(o.items())[:50]:
                _walk(v, f"{path}.{k}" if path else k, o)
        elif isinstance(o, list):
            for idx, item in enumerate(o[:100]):
                _walk(item, f"{path}[{idx}]" if path else f"[{idx}]", parent)
        elif isinstance(o, str):
            m = _DATA_URL_RE.match(o)
            if m and len(m.group(2)) >= 64:
                _add(m.group(2), m.group(1), None, path or "<root>")
    _walk(obj, "", None)
    return found, limit_hit


def _collect_binary_assets(
    run_data_obj: Any,
    *,
    execution_id: int,
    node_name: str,
    run_index: int,
) -> tuple[Any, list[Any], bool, bool]:
    """Extract binary assets from node run data and insert pending placeholders.

    Processes canonical run.data.binary block and delegates to extended discovery for
    non-standard binary embeddings. Promotes item-level binary blocks to top level when
    original lacks binary key. Replaces discovered binary data with temporary placeholders
    containing SHA256 hash and size metadata for subsequent media token exchange.

    Args:
        run_data_obj: Node run output data structure
        execution_id: Execution ID for asset metadata
        node_name: Node name for asset metadata
        run_index: Run index for asset metadata

    Returns:
        4-tuple containing:
        - cloned: Deep copy of run_data_obj with placeholders substituted
        - assets: List of discovered BinaryAsset instances
        - limit_hit: True if extended scan hit EXTENDED_MEDIA_SCAN_MAX_ASSETS limit
        - promoted_item_binary: True if binary block promoted from nested items
    """
    from ..config import get_settings
    from ..media_api import BinaryAsset
    assets: list[BinaryAsset] = []
    limit_hit = False
    promoted_item_binary = False
    if not isinstance(run_data_obj, dict):
        return run_data_obj, assets, limit_hit, promoted_item_binary
    cloned = copy.deepcopy(run_data_obj)
    if "binary" not in cloned:
        aggregated: dict[str, Any] = {}
        for _k, v in list(cloned.items())[:20]:
            if not isinstance(v, list):
                continue
            stack: list[tuple[Any, int]] = [(v, 0)]
            while stack:
                cur, depth = stack.pop()
                if depth > 3:
                    continue
                if isinstance(cur, list):
                    for itm in cur[:100]:
                        stack.append((itm, depth + 1))
                elif isinstance(cur, dict):
                    bin_block = cur.get("binary")
                    if isinstance(bin_block, dict):
                        for slot, slot_val in bin_block.items():
                            if slot not in aggregated and isinstance(slot_val, dict):
                                aggregated[slot] = copy.deepcopy(slot_val)
        if aggregated:
            cloned["binary"] = aggregated
            promoted_item_binary = True
    bin_section = cloned.get("binary") if isinstance(cloned, dict) else None
    if isinstance(bin_section, dict):
        for slot, slot_val in bin_section.items():
            if not isinstance(slot_val, dict):
                continue
            raw_data = slot_val.get("data")
            mime = (
                slot_val.get("mimeType")
                or slot_val.get("mime_type")
                or slot_val.get("fileType")
            )
            fname = slot_val.get("fileName") or slot_val.get("name")
            base64_str: str | None = None
            if isinstance(raw_data, str) and len(raw_data) > 100:
                base64_str = raw_data
            elif isinstance(raw_data, dict) and isinstance(raw_data.get("data"), str):
                inner = raw_data.get("data")
                if isinstance(inner, str) and len(inner) > 100:
                    base64_str = inner
            if not base64_str:
                continue
            try:
                _ = _b64.b64decode(base64_str[:4000], validate=False)
                full_bytes = _b64.b64decode(base64_str, validate=False)
                h = _hash.sha256(full_bytes).hexdigest()
                size_bytes = len(full_bytes)
            except Exception:
                continue
            slot_val["data"] = {
                "_media_pending": True,
                "sha256": h,
                "bytes": size_bytes,
                "base64_len": len(base64_str),
                "slot": slot,
            }
            assets.append(
                BinaryAsset(
                    execution_id=execution_id,
                    node_name=node_name,
                    run_index=run_index,
                    field_path=f"binary.{slot}",
                    original_path_segments=["binary", slot],
                    mime_type=mime if isinstance(mime, str) else None,
                    filename=fname if isinstance(fname, str) else None,
                    size_bytes=size_bytes,
                    sha256=h,
                    base64_len=len(base64_str),
                    content_b64=base64_str,
                )
            )
    settings = get_settings()
    extra_assets, extra_limit_hit = _discover_additional_binary_assets(
        cloned,
        execution_id=execution_id,
        node_name=node_name,
        run_index=run_index,
        max_assets=settings.EXTENDED_MEDIA_SCAN_MAX_ASSETS,
    )
    if extra_assets:
        for asset in extra_assets:
            try:
                raw_path = asset.field_path
                cleaned = re.sub(r"\[[0-9]+\]", "", raw_path)
                segments = [seg for seg in cleaned.split(".") if seg]
                if segments and segments[0] == "<root>":
                    segments = segments[1:]
                if not segments:
                    continue
                cursor: Optional[Dict[str, Any]] = cloned if isinstance(cloned, dict) else None
                for seg in segments[:-1]:
                    if cursor is not None and isinstance(cursor, dict):
                        nxt = cursor.get(seg)
                        cursor = nxt if isinstance(nxt, dict) else None
                    else:
                        cursor = None
                        break
                if not isinstance(cursor, dict):
                    continue
                leaf = segments[-1]
                if leaf in cursor and not isinstance(cursor.get(leaf), dict):
                    cursor[leaf] = {
                        "_media_pending": True,
                        "sha256": asset.sha256,
                        "bytes": asset.size_bytes,
                        "base64_len": asset.base64_len,
                        "slot": asset.field_path,
                    }
            except Exception:
                continue
        assets.extend(extra_assets)
    if extra_limit_hit:
        limit_hit = True
    return cloned, assets, limit_hit, promoted_item_binary


def _apply_ai_filter(
    trace: LangfuseTrace,
    record: N8nExecutionRecord,
    run_data: Dict[str, List[NodeRun]],
    extraction_nodes: list[str],
    include_patterns: list[str],
    exclude_patterns: list[str],
    max_value_len: int,
    *,
    child_agent_map: Optional[
        Dict[str, Tuple[str, str]]
    ] = None,
    filter_mode: str = "contextual",
) -> None:
    """Apply AI-only filtering with context window preservation (mutates trace in place).

    Retains only:
    - Root span (always preserved)
    - All AI node spans (detected via observation_mapper.is_ai_node)
    - Up to 2 spans immediately before first AI span (pre-context)
    - Up to 2 spans immediately after last AI span (post-context)
    - Spans on parent chain between AI spans (chain connectors)

    Optionally extracts data from specified nodes before filtering and attaches to root
    span metadata under n8n.extracted_nodes for visibility when excluded nodes contain
    critical configuration or business context.

    Sets metadata on root span:
    - n8n.filter.ai_only=true
    - n8n.filter.excluded_node_count: Number of discarded spans
    - n8n.filter.window_start_span, window_end_span: AI window boundaries
    - n8n.filter.pre_context_count, post_context_count, chain_context_count
    - n8n.extracted_nodes: Extracted node data (when extraction_nodes specified)

    Special case when no AI spans found:
    - Retains only root span
    - Sets n8n.filter.no_ai_spans=true on all excluded spans

    Args:
        trace: LangfuseTrace to filter (modified in place)
        record: N8nExecutionRecord for node type lookups
        run_data: Complete runData dict for node data extraction
        extraction_nodes: List of node names to extract data from
        include_patterns: Wildcard patterns for keys to include in extraction
        exclude_patterns: Wildcard patterns for keys to exclude from extraction
        max_value_len: Maximum string length per extracted value
        child_agent_map: Optional AI hierarchy map; nodes present as
            keys are classified as AI regardless of type/category
    """
    extracted_data = None  # Initialize outside try for scope
    try:
        # Extract node data BEFORE filtering removes spans
        if extraction_nodes:
            extracted_data = _extract_nodes_data(
                run_data=run_data,
                extraction_nodes=extraction_nodes,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                max_value_len=max_value_len,
            )
            if extracted_data:
                logger.debug(
                    f"Extracted data from {extracted_data.get('_meta', {}).get('extracted_count', 0)} nodes"
                )

        if not trace.spans:
            return
        root_span = next(
            (s for s in trace.spans if s.parent_id is None or "n8n.execution.id" in s.metadata),
            trace.spans[0],
        )
        original_order = list(trace.spans)
        spans_by_id: Dict[str, LangfuseSpan] = {s.id: s for s in original_order}
        ai_span_ids: set[str] = set()
        ai_indices: List[int] = []
        node_lookup: Dict[str, Tuple[Optional[str], Optional[str]]] = {
            n.name: (n.type, n.category) for n in record.workflowData.nodes
        }
        # Build set of node names connected to agents via ai_*
        # connections for graph-aware classification.
        graph_ai_names: set[str] = set()
        if child_agent_map:
            graph_ai_names = set(child_agent_map.keys())
            # Also include agent node names themselves.
            for _child, (agent, _lt) in child_agent_map.items():
                graph_ai_names.add(agent)
        for idx, span in enumerate(original_order):
            if span.id == root_span.id:
                continue
            bare_name = span.metadata.get(
                "n8n.node.name", span.name
            )
            node_type, node_cat = node_lookup.get(
                bare_name, (None, None)
            )
            is_ai = is_ai_node(node_type, node_cat)
            if not is_ai and bare_name in graph_ai_names:
                is_ai = True
            if is_ai:
                ai_span_ids.add(span.id)
                ai_indices.append(idx)
        keep_ids: set[str] = {root_span.id}
        normalized_mode = (filter_mode or "contextual").strip().lower()
        if not ai_indices:
            for s in original_order:
                if s.id != root_span.id:
                    s.metadata.setdefault("n8n.filter.no_ai_spans", True)
            trace.spans = [root_span]
            root_span.metadata["n8n.filter.ai_only"] = True
            root_span.metadata["n8n.filter.mode"] = normalized_mode
            root_span.metadata.setdefault("n8n.filter.excluded_node_count", len(original_order) - 1)
            root_span.metadata["n8n.filter.no_ai_spans"] = True
            # Attach extracted data even when no AI spans
            if extracted_data:
                root_span.metadata["n8n.extracted_nodes"] = extracted_data
            return
        first_ai_idx = ai_indices[0]
        last_ai_idx = ai_indices[-1]
        keep_ids.update(ai_span_ids)
        if normalized_mode == "strict":
            # Keep AI spans plus required ancestor closure up to root.
            for ai_id in ai_span_ids:
                cur = spans_by_id.get(ai_id)
                hops = 0
                while cur and cur.parent_id and hops < 300:
                    hops += 1
                    parent_id = cur.parent_id
                    if parent_id == root_span.id:
                        break
                    parent_span = spans_by_id.get(parent_id)
                    if parent_span is None:
                        break
                    keep_ids.add(parent_id)
                    cur = parent_span
        else:
            # Contextual mode: AI spans + pre/post window + chain connectors.
            if first_ai_idx - 1 >= 1:
                keep_ids.add(original_order[first_ai_idx - 1].id)
            if first_ai_idx - 2 >= 1:
                keep_ids.add(original_order[first_ai_idx - 2].id)

            def _is_on_chain(span: LangfuseSpan) -> bool:
                cur = span
                hops = 0
                while cur.parent_id and hops < 200:
                    hops += 1
                    if cur.parent_id in ai_span_ids:
                        return True
                    cur = spans_by_id.get(cur.parent_id) or cur
                    if cur.id == root_span.id:
                        break
                return False

            for i in range(first_ai_idx + 1, last_ai_idx):
                s = original_order[i]
                if s.id == root_span.id:
                    continue
                if _is_on_chain(s):
                    keep_ids.add(s.id)
            if last_ai_idx + 1 < len(original_order):
                keep_ids.add(original_order[last_ai_idx + 1].id)
            if last_ai_idx + 2 < len(original_order):
                keep_ids.add(original_order[last_ai_idx + 2].id)
        new_spans: List[LangfuseSpan] = []
        excluded = 0
        for s in original_order:
            if s.id in keep_ids:
                new_spans.append(s)
            else:
                excluded += 1
        root_span.metadata["n8n.filter.ai_only"] = True
        root_span.metadata["n8n.filter.mode"] = normalized_mode
        root_span.metadata["n8n.filter.excluded_node_count"] = excluded
        # Window metadata reconstruction
        window_start_span = original_order[first_ai_idx].id
        window_end_span = original_order[last_ai_idx].id
        root_span.metadata["n8n.filter.window_start_span"] = window_start_span
        root_span.metadata["n8n.filter.window_end_span"] = window_end_span
        # Compute counts from retained spans for clarity & test parity
        retained_non_root = []
        for s in new_spans:
            if s.id != root_span.id:
                retained_non_root.append(s)
        retained_ai_indices = [i for i, s in enumerate(retained_non_root) if s.id in ai_span_ids]
        pre_context_count = 0
        post_context_count = 0
        chain_context_count = 0
        if retained_ai_indices:
            first_retained_ai = retained_ai_indices[0]
            last_retained_ai = retained_ai_indices[-1]
            pre_context_count = first_retained_ai
            # Chain spans: between first and last AI excluding AI spans
            for i in range(first_retained_ai + 1, last_retained_ai):
                mid_span = retained_non_root[i]
                if mid_span.id not in ai_span_ids:
                    chain_context_count += 1
            post_context_count = len(retained_non_root) - (last_retained_ai + 1)
            if pre_context_count > 2:
                pre_context_count = 2
            if post_context_count > 2:
                post_context_count = 2
        if normalized_mode == "strict":
            pre_context_count = 0
            post_context_count = 0
            chain_context_count = 0
        root_span.metadata["n8n.filter.pre_context_count"] = pre_context_count
        root_span.metadata["n8n.filter.post_context_count"] = post_context_count
        root_span.metadata["n8n.filter.chain_context_count"] = chain_context_count

        # Attach extracted node data to root span metadata
        if extracted_data:
            root_span.metadata["n8n.extracted_nodes"] = extracted_data

        trace.spans = new_spans
    except Exception as e:  # pragma: no cover
        try:
            logger.warning("ai_filter_failed error=%s", e)
        except Exception:
            pass


def _fixup_agent_parents(
    trace: LangfuseTrace,
    ctx: MappingContext,
) -> None:
    """Correct Tier-1 (Agent Hierarchy) parent assignments after all spans exist.

    During chronological mapping, tool span start_time may precede its agent's
    start_time (n8n records tool invocations from earlier loop iterations first).
    This means last_span_for_node[agent] does not yet exist when the tool is
    processed, causing resolve_parent to fall back to root.

    This function performs a single post-loop pass over all spans:
    1. Collects all spans that belong to known agent names (from child_agent_map)
       and sorts them by start_time.
    2. For each span whose metadata indicates an agent parent (n8n.agent.parent),
       finds the *latest* agent span whose start_time <= tool span's start_time.
       If none found (all agent spans start after the tool), uses the earliest
       agent span.
    3. If the resolved parent differs from the current parent_id, updates it and
       emits n8n.agent.parent_fixup=True metadata.

    This is deterministic: same input yields same fixup outcome.

    Args:
        trace: LangfuseTrace with all spans created (mutated in place)
        ctx: MappingContext containing child_agent_map and root_span_id
    """
    if not ctx.child_agent_map:
        return

    # Collect agent names (values of child_agent_map).
    agent_names: set[str] = set()
    for _child, (agent, _lt) in ctx.child_agent_map.items():
        agent_names.add(agent)

    # Build agent_name -> sorted list of (start_time, span_id).
    agent_spans: Dict[str, List[Tuple[Any, str]]] = {}
    for span in trace.spans:
        bare_name = span.metadata.get(
            "n8n.node.name", span.name
        )
        if bare_name in agent_names:
            agent_spans.setdefault(bare_name, []).append(
                (span.start_time, span.id)
            )
    for entries in agent_spans.values():
        entries.sort(key=lambda e: e[0])

    # Fixup pass: re-resolve parent for agent-hierarchy spans.
    for span in trace.spans:
        agent_parent = span.metadata.get("n8n.agent.parent")
        if not agent_parent:
            continue
        candidates = agent_spans.get(agent_parent)
        if not candidates:
            continue

        # Find latest agent span starting at or before this tool.
        best_id: Optional[str] = None
        for a_start, a_id in candidates:
            if a_start <= span.start_time:
                best_id = a_id
            else:
                break
        # If no agent span starts before the tool, use earliest.
        if best_id is None:
            best_id = candidates[0][1]

        if best_id != span.parent_id:
            span.parent_id = best_id
            span.metadata["n8n.agent.parent_fixup"] = True


def _expand_agent_envelopes(
    trace: LangfuseTrace,
    ctx: MappingContext,
) -> None:
    """Expand agent span timing to encompass all child spans.

    n8n records agent executionTime covering only the final
    loop iteration while child LLM/tool runs span all iterations.
    This creates children whose timing extends beyond their
    parent agent's boundary, producing confusing Langfuse timelines.

    For each agent span, computes min(start_time) and max(end_time)
    across all direct children. If any child extends beyond the
    agent's boundary, expands the agent's timing and stores the
    original values in metadata for debugging.

    Args:
        trace: LangfuseTrace with fully resolved parents
        ctx: MappingContext containing child_agent_map
    """
    if not ctx.child_agent_map:
        return

    # Index spans by id for parent lookup.
    spans_by_id: Dict[str, LangfuseSpan] = {
        s.id: s for s in trace.spans
    }

    # Collect children grouped by parent agent span id.
    agent_children: Dict[str, List[LangfuseSpan]] = {}
    agent_names: set[str] = set()
    for _child, (agent, _lt) in ctx.child_agent_map.items():
        agent_names.add(agent)

    agent_span_ids: set[str] = set()
    for span in trace.spans:
        bare = span.metadata.get("n8n.node.name", span.name)
        if bare in agent_names:
            agent_span_ids.add(span.id)

    for span in trace.spans:
        if (
            span.parent_id
            and span.parent_id in agent_span_ids
        ):
            agent_children.setdefault(
                span.parent_id, []
            ).append(span)

    # Expand each agent span to envelope its children.
    for agent_id, children in agent_children.items():
        agent_span = spans_by_id.get(agent_id)
        if agent_span is None or not children:
            continue

        child_min_start = min(c.start_time for c in children)
        child_max_end = max(c.end_time for c in children)

        expanded = False
        original_start = agent_span.start_time
        original_end = agent_span.end_time

        if child_min_start < agent_span.start_time:
            agent_span.start_time = child_min_start
            expanded = True
        if child_max_end > agent_span.end_time:
            agent_span.end_time = child_max_end
            expanded = True

        if expanded:
            agent_span.metadata[
                "n8n.agent.envelope_expanded"
            ] = True
            agent_span.metadata[
                "n8n.agent.original_start_time"
            ] = original_start.isoformat()
            orig_ms = int(
                (original_end - original_start).total_seconds()
                * 1000
            )
            agent_span.metadata[
                "n8n.agent.original_execution_time_ms"
            ] = orig_ms
            logger.debug(
                "agent_envelope_expanded agent=%s "
                "original_ms=%s expanded_start=%s "
                "expanded_end=%s children=%d",
                agent_span.metadata.get(
                    "n8n.node.name", agent_span.name
                ),
                orig_ms,
                agent_span.start_time.isoformat(),
                agent_span.end_time.isoformat(),
                len(children),
            )


def _assign_agent_iterations(
    trace: LangfuseTrace,
    ctx: MappingContext,
) -> None:
    """Assign agent loop iteration index to child spans.

    For each agent span, identifies child spans belonging to
    successive iterations. Iteration boundaries are inferred from
    generation (LLM) child spans: each generation child marks the
    start of a new iteration. Non-generation children (tools,
    memory) between two generation children belong to the earlier
    iteration.

    Sets n8n.agent.iteration (0-based int) on each child span's
    metadata.

    Args:
        trace: LangfuseTrace with resolved parents and envelopes
        ctx: MappingContext containing child_agent_map
    """
    if not ctx.child_agent_map:
        return

    agent_names: set[str] = set()
    for _child, (agent, _lt) in ctx.child_agent_map.items():
        agent_names.add(agent)

    agent_span_ids: set[str] = set()
    for span in trace.spans:
        bare = span.metadata.get("n8n.node.name", span.name)
        if bare in agent_names:
            agent_span_ids.add(span.id)

    # Group children by parent agent span.
    agent_children: Dict[str, List[LangfuseSpan]] = {}
    for span in trace.spans:
        if (
            span.parent_id
            and span.parent_id in agent_span_ids
        ):
            agent_children.setdefault(
                span.parent_id, []
            ).append(span)

    for _agent_id, children in agent_children.items():
        # Sort children by start_time for iteration inference.
        sorted_children = sorted(
            children, key=lambda s: s.start_time
        )
        # Find generation spans (iteration boundaries).
        gen_indices: List[int] = []
        for i, child in enumerate(sorted_children):
            if child.observation_type == "generation":
                gen_indices.append(i)

        if not gen_indices:
            # No generations: assign iteration 0 to all.
            for child in sorted_children:
                child.metadata["n8n.agent.iteration"] = 0
            continue

        # Assign iterations: each generation starts a new one.
        # Children before the first generation get iteration 0.
        iteration = 0
        gen_set = set(gen_indices)
        next_gen_ptr = 0
        for i, child in enumerate(sorted_children):
            if i in gen_set and next_gen_ptr < len(gen_indices):
                iteration = next_gen_ptr
                next_gen_ptr += 1
            child.metadata["n8n.agent.iteration"] = iteration


def _map_execution(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    *,
    collect_binaries: bool = False,
) -> tuple[LangfuseTrace, list[Any]]:
    """Central orchestration function mapping n8n execution to Langfuse trace structure.

    Performs complete transformation pipeline:
    1. Create trace with deterministic ID from execution.id
    2. Build root span representing entire execution
    3. Construct static graph structures (reverse edges, agent hierarchy)
    4. Initialize mapping context for state tracking
    5. Flatten and sort all node runs chronologically
    6. For each node run:
        - Resolve parent span via precedence rules
        - Detect generation spans and extract token usage
        - Normalize and truncate I/O structures
        - Collect binary assets (when enabled)
        - Extract model metadata with parameter fallback
        - Detect Gemini empty output anomaly
        - Create span with all attributes and metadata
    7. Return complete trace and discovered assets

    Args:
        record: N8nExecutionRecord with workflow data and node run results
        truncate_limit: Max characters for I/O before truncation; None or <=0 disables
        collect_binaries: When True, extract binary assets and insert placeholders

    Returns:
        Tuple of (trace, assets)
        - trace: Complete LangfuseTrace with deterministic IDs and hierarchical spans
        - assets: List of discovered BinaryAsset instances (empty if collect_binaries=False)

    Note:
        Binary stripping always applies regardless of truncation or collection settings.
        Collected assets contain placeholders in trace output; actual token exchange occurs
        in post-map patch phase.
    """
    trace_id = str(record.id)
    started_at = (
        record.startedAt if record.startedAt.tzinfo is not None else record.startedAt.replace(tzinfo=timezone.utc)
    )
    stopped_raw = record.stoppedAt
    if stopped_raw is not None and stopped_raw.tzinfo is None:
        stopped_raw = stopped_raw.replace(tzinfo=timezone.utc)
    base_name = record.workflowData.name or "execution"
    trace = LangfuseTrace(
        id=trace_id,
        name=base_name,
        timestamp=started_at,
        metadata={"workflowId": record.workflowId, "status": record.status},
    )
    root_span_id = str(uuid5(SPAN_NAMESPACE, f"{trace_id}:root"))
    root_span = LangfuseSpan(
        id=root_span_id,
        trace_id=trace_id,
        name=base_name,
        start_time=started_at,
        end_time=stopped_raw or (started_at + timedelta(milliseconds=1)),
        observation_type="span",
        metadata={"n8n.execution.id": record.id},
    )
    trace.spans.append(root_span)
    wf_node_lookup: Dict[str, Dict[str, Optional[str]]] = {
        n.name: {"type": n.type, "category": n.category} for n in record.workflowData.nodes
    }
    wf_node_obj: Dict[str, WorkflowNode] = {n.name: n for n in record.workflowData.nodes}
    ctx = MappingContext(
        trace_id=trace_id,
        root_span_id=root_span_id,
        wf_node_lookup=wf_node_lookup,
        wf_node_obj=wf_node_obj,
        reverse_edges=_build_reverse_edges(record.workflowData),
        child_agent_map=_build_child_agent_map(record.workflowData),
        truncate_limit=truncate_limit,
    )
    run_data = record.data.executionData.resultData.runData

    # Build prompt source registry (Phase 1: Detection)
    try:
        prompt_registry = _build_prompt_registry(
            run_data, record.workflowData.nodes
        )
        logger.debug(
            f"Built prompt registry with {len(prompt_registry)} entries"
        )
    except Exception as e:
        logger.warning(f"Failed to build prompt registry: {e}")
        prompt_registry = {}

    # Initialize version resolver for cross-environment support (Phase 2: API Fallback)
    version_resolver = None
    try:
        version_resolver = _create_version_resolver_from_env()
    except Exception as e:
        logger.warning(f"Failed to create version resolver: {e}")

    flattened = _flatten_runs(run_data)
    collected_assets: list[Any] = []
    # Root span I/O surfacing (configured via env). We resolve settings lazily
    # here to avoid importing config at module import time (determinism & tests).
    from ..config import get_settings  # local import to keep orchestrator pure until call
    target_input_name: Optional[str] = None
    target_output_name: Optional[str] = None
    try:
        _settings = get_settings()
        target_input_name = _settings.ROOT_SPAN_INPUT_NODE
        target_output_name = _settings.ROOT_SPAN_OUTPUT_NODE
        # Pre-normalize case-insensitive targets (lower) for matching speed.
        target_input = (
            target_input_name.lower()
            if target_input_name
            else None
        )
        target_output = (
            target_output_name.lower()
            if target_output_name
            else None
        )
    except Exception:  # pragma: no cover - defensive failure
        target_input = None
        target_output = None
    # Track last matched run index & serialized input/output (post-normalization)
    root_input_val: Any = None
    root_input_run_index: Optional[int] = None
    root_output_val: Any = None
    root_output_run_index: Optional[int] = None
    for _start_ts_raw, node_name, idx, run in flattened:
        wf_meta = ctx.wf_node_lookup.get(node_name, {})
        node_type = wf_meta.get("type") or node_name
        category = wf_meta.get("category")
        obs_type_guess = map_node_to_observation_type(node_type, category)
        span_id = str(uuid5(SPAN_NAMESPACE, f"{ctx.trace_id}:{node_name}:{idx}"))
        start_time = _epoch_ms_to_dt(run.startTime)
        end_time = start_time + timedelta(milliseconds=run.executionTime or 0)
        parent_id, prev_node, prev_node_run = _resolve_parent(
            node_name=node_name,
            run=run,
            trace_id=ctx.trace_id,
            child_agent_map=ctx.child_agent_map,
            last_span_for_node=ctx.last_span_for_node,
            reverse_edges=ctx.reverse_edges,
            root_span_id=ctx.root_span_id,
        )
        is_generation = _detect_generation(node_type, run)
        observation_type = obs_type_guess or ("generation" if is_generation else "span")
        raw_input_obj: Any = None
        if run.inputOverride is not None:
            raw_input_obj = run.inputOverride
        elif prev_node and prev_node in ctx.last_output_data:
            raw_input_obj = {"inferredFrom": prev_node, "data": ctx.last_output_data[prev_node]}
        limit_hit = False
        promoted_item_binary = False
        if collect_binaries:
            mutated_output, assets, limit_hit, promoted_item_binary = _collect_binary_assets(
                run.data,
                execution_id=record.id,
                node_name=node_name,
                run_index=idx,
            )
            if assets:
                collected_assets.extend(assets)
            raw_output_obj = mutated_output
        else:
            raw_output_obj = run.data
        status_norm = (run.executionStatus or "").lower()
        if run.error:
            status_norm = "error"
        display_name = f"{node_name} #{idx}"
        metadata: Dict[str, Any] = {
            "n8n.node.name": node_name,
            "n8n.node.type": node_type,
            "n8n.node.category": category,
            "n8n.node.run_index": idx,
            "n8n.node.execution_time_ms": run.executionTime,
            "n8n.node.execution_status": status_norm,
        }
        (
            norm_input_obj,
            norm_output_obj,
            input_str,
            input_trunc,
            output_str,
            output_trunc,
            input_flags,
            output_flags,
            llm_params_metadata,
        ) = _prepare_io_and_output(
            raw_input_obj=raw_input_obj,
            raw_output_obj=raw_output_obj,
            is_generation=is_generation,
            node_type=node_type,
            truncate_limit=ctx.truncate_limit,
        )
        # Merge LLM parameters into metadata
        if llm_params_metadata:
            metadata.update(llm_params_metadata)
        if input_flags.get("unwrapped_ai_channel") or output_flags.get("unwrapped_ai_channel"):
            metadata["n8n.io.unwrapped_ai_channel"] = True
        if input_flags.get("unwrapped_json_root") or output_flags.get("unwrapped_json_root"):
            metadata["n8n.io.unwrapped_json_root"] = True
        if collect_binaries and promoted_item_binary:
            metadata["n8n.io.promoted_item_binary"] = True
        if node_name in ctx.child_agent_map:
            agent_name, link_type = ctx.child_agent_map[node_name]
            metadata["n8n.agent.parent"] = agent_name
            metadata["n8n.agent.link_type"] = link_type
        if prev_node:
            metadata["n8n.node.previous_node"] = prev_node
        if prev_node_run is not None:
            metadata["n8n.node.previous_node_run"] = prev_node_run
        elif prev_node and node_name in ctx.reverse_edges and node_name not in ctx.child_agent_map:
            metadata["n8n.graph.inferred_parent"] = True
        if input_trunc:
            metadata["n8n.truncated.input"] = True
        if output_trunc:
            metadata["n8n.truncated.output"] = True
        model_val, model_meta = _extract_model_and_metadata(
            run=run,
            node_name=node_name,
            node_type=node_type,
            is_generation=is_generation,
            wf_node_obj=ctx.wf_node_obj,
            raw_input_obj=raw_input_obj,
        )
        if model_meta:
            metadata.update(model_meta)
        # Look ahead to next run's observation type for tool_calls suppression.
        next_observation_type: Optional[str] = None
        try:
            # Flattened list iteration order preserved; we can peek ahead using
            # Python list semantics without additional passes.
            # (Performance: O(1) per iteration.)
            # Determine index of current tuple by simple search of parent span
            # id mapping not yet stored; we reuse loop variables.
            # We rely on 'flattened' scope; safe inside same function.
            # Acquire next tuple if available.
            # Note: using enumerate at top would require structural change; we
            # keep local search for minimal diff & clarity (list sizes are
            # small enough for shipper throughput constraints).
            # Optimize early exit when first element mismatch.
            # This light linear scan stays bounded by node count (< few
            # hundred) and only executes for generation spans.
            if is_generation:
                for pos, tup in enumerate(flattened):
                    if tup[1] == node_name and tup[2] == idx and tup[3] is run:
                        if pos + 1 < len(flattened):
                            _nxt_ts, nxt_name, _nxt_idx, nxt_run = flattened[pos + 1]
                            nxt_meta = ctx.wf_node_lookup.get(nxt_name, {})
                            nxt_type = nxt_meta.get("type") or nxt_name
                            nxt_cat = nxt_meta.get("category")
                            nxt_guess = map_node_to_observation_type(
                                nxt_type, nxt_cat
                            )
                            # Generation detection applies only for classification
                            # fallback; we preserve explicit tool classification.
                            next_observation_type = nxt_guess
                        break
        except Exception:
            next_observation_type = None

        # Prompt resolution for generation spans
        prompt_name: Optional[str] = None
        prompt_version: Optional[int] = None
        if is_generation and prompt_registry:
            try:
                # Get agent parent for disambiguation (if generation is under agent)
                agent_parent = None
                if node_name in ctx.child_agent_map:
                    agent_parent, _ = ctx.child_agent_map[node_name]

                prompt_result = _resolve_prompt_for_generation(
                    node_name=node_name,
                    run_index=idx,
                    run_data=run_data,
                    prompt_registry=prompt_registry,
                    agent_input=raw_input_obj,
                    agent_parent=agent_parent,
                    child_agent_map=ctx.child_agent_map,
                )

                # Store original version for metadata transparency
                original_version = prompt_result.original_version

                # Apply cross-environment version resolution if needed
                if (
                    prompt_result.prompt_name
                    and prompt_result.prompt_version
                    and version_resolver
                ):
                    resolved_version, resolution_source = (
                        version_resolver.resolve_version(
                            prompt_result.prompt_name,
                            prompt_result.prompt_version,
                        )
                    )
                    prompt_name = prompt_result.prompt_name
                    prompt_version = resolved_version

                    # Emit resolution metadata
                    metadata["n8n.prompt.version.original"] = original_version
                    if resolved_version != original_version:
                        metadata["n8n.prompt.version.mapped"] = resolved_version
                        metadata[
                            "n8n.prompt.version.mapping_source"
                        ] = resolution_source
                else:
                    # No version resolver or no prompt metadata
                    prompt_name = prompt_result.prompt_name
                    prompt_version = prompt_result.prompt_version

                # Emit prompt resolution debug metadata
                if prompt_result.resolution_method != "none":
                    metadata[
                        "n8n.prompt.resolution_method"
                    ] = prompt_result.resolution_method
                    metadata["n8n.prompt.confidence"] = prompt_result.confidence
                    if prompt_result.ancestor_distance is not None:
                        metadata[
                            "n8n.prompt.ancestor_distance"
                        ] = prompt_result.ancestor_distance
                    if prompt_result.candidate_count > 0:
                        metadata[
                            "n8n.prompt.candidate_count"
                        ] = prompt_result.candidate_count
                    if prompt_result.ambiguous:
                        metadata["n8n.prompt.ambiguous"] = True

                    if prompt_name and prompt_version:
                        logger.info(
                            f"Resolved prompt for {node_name}[{idx}]: "
                            f"name='{prompt_name}', version={prompt_version}, "
                            f"method={prompt_result.resolution_method}, "
                            f"confidence={prompt_result.confidence}"
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to resolve prompt for {node_name}[{idx}]: {e}"
                )

        status_override, anomaly_meta = _detect_gemini_empty_output_anomaly(
            is_generation=is_generation,
            norm_output_obj=norm_output_obj,
            run=run,
            node_name=node_name,
            next_observation_type=next_observation_type,
        )
        if anomaly_meta:
            metadata.update(anomaly_meta)
        if status_override:
            status_norm = status_override
        if limit_hit:
            metadata["n8n.media.upload_failed"] = True
            metadata.setdefault("n8n.media.error_codes", []).append("scan_asset_limit")
        span = LangfuseSpan(
            id=span_id,
            trace_id=ctx.trace_id,
            parent_id=parent_id,
            name=display_name,
            start_time=start_time,
            end_time=end_time,
            observation_type=observation_type,
            input=input_str,
            output=output_str,
            metadata=metadata,
            error=run.error,
            model=model_val,
            usage=_extract_usage(run),
            status=status_norm,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )
        trace.spans.append(span)
        # Capture last run input/output for configured nodes (case-insensitive).
        lowered_name = node_name.lower()
        if target_input and lowered_name == target_input:
            root_input_val = input_str  # Already normalized/truncated
            root_input_run_index = idx
        if target_output and lowered_name == target_output:
            root_output_val = output_str
            root_output_run_index = idx
        ctx.last_span_for_node[node_name] = span_id
        try:
            size_guard_ok = True
            if (
                ctx.truncate_limit is not None
                and isinstance(ctx.truncate_limit, int)
                and ctx.truncate_limit > 0
            ):
                size_guard_ok = len(str(raw_output_obj)) < ctx.truncate_limit * 2
            if isinstance(raw_output_obj, dict) and size_guard_ok:
                ctx.last_output_data[node_name] = raw_output_obj
        except Exception:
            pass
    # ── Post-loop agent-hierarchy parent fixup ──────────────────────
    # Tool node runs may start *before* their agent's run in n8n's
    # execution model. During the chronological loop above, the agent
    # span does not exist yet when we process such a tool, so
    # resolve_parent falls back to root.  Now that ALL spans exist we
    # can correct Tier-1 (Agent Hierarchy) parent assignments.
    _fixup_agent_parents(trace, ctx)
    # ── Agent envelope expansion ────────────────────────────────────
    # n8n records agent executionTime covering only the final loop
    # iteration, while child LLM/tool runs span all iterations. This
    # causes children to overflow the agent's time boundary, producing
    # confusing Langfuse timelines. Expand agent spans to encompass all
    # children, preserving original timing in metadata.
    _expand_agent_envelopes(trace, ctx)
    # ── Agent iteration assignment ──────────────────────────────────
    # Assign n8n.agent.iteration metadata to child spans within each
    # agent cluster, enabling iteration-level grouping in Langfuse UI.
    _assign_agent_iterations(trace, ctx)

    # Populate root span input/output if configured and captured. We avoid any
    # mutation earlier to keep mapping deterministic; this simple assignment is
    # transparent and preserves original sanitization/truncation semantics.
    try:
        if target_input:
            if root_input_run_index is not None:
                root_span.input = root_input_val
                # Emit traceability metadata keys (do not duplicate execution id invariant).
                root_span.metadata["n8n.root.input_node"] = target_input_name
                root_span.metadata["n8n.root.input_run_index"] = root_input_run_index
            else:
                root_span.metadata["n8n.root.input_node_not_found"] = True
        if target_output:
            if root_output_run_index is not None:
                root_span.output = root_output_val
                root_span.metadata["n8n.root.output_node"] = target_output_name
                root_span.metadata["n8n.root.output_run_index"] = root_output_run_index
            else:
                root_span.metadata["n8n.root.output_node_not_found"] = True
    except Exception:  # pragma: no cover - defensive; failure should not abort mapping
        try:
            root_span.metadata.setdefault("n8n.root.population_error", True)
        except Exception:
            pass
    return trace, collected_assets


__all__ = [
    "_serialize_and_truncate",
    "_flatten_runs",
    "_prepare_io_and_output",
    "_extract_model_and_metadata",
    "_detect_gemini_empty_output_anomaly",
    "_collect_binary_assets",
    "_discover_additional_binary_assets",
    "_apply_ai_filter",
    "_map_execution",
]
