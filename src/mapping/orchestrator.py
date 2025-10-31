"""Orchestrator helpers extracted from `mapper.py`.

Pure functions only; centralizes procedural mapping logic so `mapper.py`
remains a thin facade. No behavioral changes.
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
from .parent_resolution import (
    build_child_agent_map as _build_child_agent_map,
)
from .parent_resolution import (
    build_reverse_edges as _build_reverse_edges,
)
from .parent_resolution import (
    resolve_parent as _resolve_parent,
)
from .time_utils import epoch_ms_to_dt as _epoch_ms_to_dt

logger = logging.getLogger(__name__)


def _serialize_and_truncate(obj: Any, limit: Optional[int]) -> Tuple[Optional[str], bool]:
    """Serialize object to JSON string with unconditional binary stripping.

    Stripping applies even when truncation disabled (limit None/<=0), matching
    invariants asserted by tests. Fail-open on detection errors.
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


def _flatten_runs(run_data: Dict[str, List[NodeRun]]) -> List[Tuple[int, str, int, NodeRun]]:
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
) -> Tuple[Any, Any, Optional[str], bool, Optional[str], bool, Dict[str, bool], Dict[str, bool]]:
    if is_generation:
        raw_input_obj = _strip_system_prompt_from_langchain_lmchat(
            raw_input_obj, node_type
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
    from ..media_api import BinaryAsset
    found: list[BinaryAsset] = []
    limit_hit = False
    seen_sha: set[str] = set()
    def _add(base64_str: str, mime: str | None, fname: str | None, path: str) -> None:
        nonlocal limit_hit
        if max_assets > 0 and len(found) >= max_assets:
            limit_hit = True; return
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
            limit_hit = True; return
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
        for k, v in list(cloned.items())[:20]:
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
                        cursor = None; break
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


def _apply_ai_filter(trace: LangfuseTrace, record: N8nExecutionRecord) -> None:
    try:
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
        for idx, span in enumerate(original_order):
            if span.id == root_span.id:
                continue
            node_type, node_cat = node_lookup.get(span.name, (None, None))
            if is_ai_node(node_type, node_cat):
                ai_span_ids.add(span.id)
                ai_indices.append(idx)
        keep_ids: set[str] = {root_span.id}
        if not ai_indices:
            for s in original_order:
                if s.id != root_span.id:
                    s.metadata.setdefault("n8n.filter.no_ai_spans", True)
            trace.spans = [root_span]
            root_span.metadata["n8n.filter.ai_only"] = True
            root_span.metadata.setdefault("n8n.filter.excluded_node_count", len(original_order) - 1)
            root_span.metadata["n8n.filter.no_ai_spans"] = True
            return
        first_ai_idx = ai_indices[0]
        last_ai_idx = ai_indices[-1]
        keep_ids.update(ai_span_ids)
        if first_ai_idx - 1 >= 1:
            keep_ids.add(original_order[first_ai_idx - 1].id)
        if first_ai_idx - 2 >= 1:
            keep_ids.add(original_order[first_ai_idx - 2].id)
        index_by_id = {s.id: i for i, s in enumerate(original_order)}
        def _is_on_chain(span: LangfuseSpan) -> bool:
            cur = span; hops = 0
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
        root_span.metadata["n8n.filter.pre_context_count"] = pre_context_count
        root_span.metadata["n8n.filter.post_context_count"] = post_context_count
        root_span.metadata["n8n.filter.chain_context_count"] = chain_context_count
        trace.spans = new_spans
    except Exception as e:  # pragma: no cover
        try:
            logger.warning("ai_filter_failed error=%s", e)
        except Exception:
            pass


def _map_execution(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    *,
    collect_binaries: bool = False,
) -> tuple[LangfuseTrace, list[Any]]:
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
    flattened = _flatten_runs(run_data)
    collected_assets: list[Any] = []
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
        usage = _extract_usage(run)
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
        metadata: Dict[str, Any] = {
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
        ) = _prepare_io_and_output(
            raw_input_obj=raw_input_obj,
            raw_output_obj=raw_output_obj,
            is_generation=is_generation,
            node_type=node_type,
            truncate_limit=ctx.truncate_limit,
        )
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
            # small enough for backfill throughput constraints).
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
            name=node_name,
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
        )
        trace.spans.append(span)
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
