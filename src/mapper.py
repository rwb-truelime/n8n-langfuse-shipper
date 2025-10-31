"""Core mapping logic from n8n execution records to Langfuse internal models.

This module is the "T" (Transform) in the ETL pipeline. It takes raw n8n
execution records fetched from PostgreSQL and converts them into a structured
LangfuseTrace object, which is a serializable representation of a Langfuse trace
and its nested spans.

Key responsibilities include:
- Mapping one n8n execution to one Langfuse trace.
- Mapping each n8n node run to a Langfuse span.
- Resolving parent-child relationships between spans using a multi-tiered
  precedence logic (agent hierarchy, runtime pointers, static graph analysis).
- Detecting and classifying "generation" spans (LLM calls).
- Extracting and normalizing token usage and model names for generations.
- Sanitizing and preparing I/O data by stripping binary content and applying
  optional truncation.
- Propagating outputs from parent spans as inputs to child spans when no
  explicit input override is present.
- Generating deterministic, repeatable IDs for traces and spans.
"""
from __future__ import annotations

import logging
import re
from .mapping.time_utils import epoch_ms_to_dt as _epoch_ms_to_dt
from .mapping.id_utils import SPAN_NAMESPACE
from .mapping.binary_sanitizer import (
    likely_binary_b64 as _likely_binary_b64,
    contains_binary_marker as _contains_binary_marker,
    strip_binary_payload as _strip_binary_payload,
)
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple
from uuid import NAMESPACE_DNS, uuid5
from dataclasses import dataclass, field

from .models.langfuse import LangfuseSpan, LangfuseTrace, LangfuseUsage
from .media_api import BinaryAsset, MappedTraceWithAssets
from .models.n8n import N8nExecutionRecord, NodeRun, WorkflowNode
from .observation_mapper import map_node_to_observation_type, is_ai_node

SPAN_NAMESPACE = uuid5(NAMESPACE_DNS, "n8n-langfuse-shipper-span")
logger = logging.getLogger(__name__)


from .mapping.io_normalizer import (
    unwrap_ai_channel as _unwrap_ai_channel,
    unwrap_generic_json as _unwrap_generic_json,
    normalize_node_io as _normalize_node_io,
    strip_system_prompt_from_langchain_lmchat as _strip_system_prompt_from_langchain_lmchat,
)
 # Binary helper functions now sourced from mapping.binary_sanitizer. Original
 # inline definitions removed to reduce mapper.py size; imports above preserve
 # previous names for callers inside this module.


def _strip_system_prompt_from_langchain_lmchat(
    input_obj: Any, node_type: str
) -> Any:
    """Strip System prompt from LangChain LMChat node inputs.

    LangChain LMChat nodes (@n8n/n8n-nodes-langchain.lmChat*) often combine
    System and User prompts into one large message blob. This function detects
    the split marker between them and strips everything before the User part.

    Split marker: '\\n\\n## START PROCESSING\\n\\nHuman: ##'
    - System part ends after: '## START PROCESSING\\n\\n'
    - User part starts at: 'Human: ##'

    Args:
        input_obj: The raw input object (can be deeply nested structure).
        node_type: The node type string to check for lmChat pattern.

    Returns:
        Modified input object with System prompt stripped if applicable,
        otherwise original input_obj (fail-open).
    """
    # Only process lmChat nodes
    if not isinstance(node_type, str):
        return input_obj
    node_type_lower = node_type.lower()
    if "lmchat" not in node_type_lower:
        return input_obj

    # Split marker pattern
    split_marker = "\n\n## START PROCESSING\n\nHuman: ##"
    user_start = "Human: ##"

    try:
        # Work on a deep copy to avoid mutating original
        import copy

        modified = copy.deepcopy(input_obj)
        modified_any = False

        # Recursively search for 'messages' arrays at any depth
        def _process_messages_recursive(
            obj: Any, depth: int = 0
        ) -> bool:
            """Recursively find and process messages arrays.

            Returns True if any modification was made.
            """
            nonlocal modified_any

            if depth > 25:  # Safety limit
                return False

            if isinstance(obj, dict):
                # Check if this dict has a 'messages' key
                if "messages" in obj and isinstance(obj["messages"], list):
                    messages = obj["messages"]
                    for i, msg in enumerate(messages):
                        # Case 1: message is a dict with string values
                        if isinstance(msg, dict):
                            for key, value in list(msg.items()):
                                if (
                                    isinstance(value, str)
                                    and split_marker in value
                                ):
                                    split_idx = value.find(user_start)
                                    if split_idx != -1:
                                        msg[key] = value[split_idx:]
                                        modified_any = True
                                        logger.debug(
                                            "Stripped system prompt "
                                            f"(dict depth={depth}); "
                                            f"node_type={node_type}, "
                                            f"removed_chars={split_idx}"
                                        )
                        # Case 2: message is a string directly
                        elif isinstance(msg, str) and split_marker in msg:
                            split_idx = msg.find(user_start)
                            if split_idx != -1:
                                messages[i] = msg[split_idx:]
                                modified_any = True
                                logger.debug(
                                    "Stripped system prompt "
                                    f"(str depth={depth}); "
                                    f"node_type={node_type}, "
                                    f"removed_chars={split_idx}"
                                )

                # Recurse into all dict values
                for value in obj.values():
                    _process_messages_recursive(value, depth + 1)

            elif isinstance(obj, list):
                # Recurse into all list items
                for item in obj[:100]:  # Limit list traversal
                    _process_messages_recursive(item, depth + 1)

            return modified_any

        _process_messages_recursive(modified)
        return modified if modified_any else input_obj

    except Exception as e:
        # Fail-open: log and return original
        logger.debug(
            f"Failed to strip system prompt from lmChat input: {e}",
            exc_info=True,
        )
        return input_obj


def _serialize_and_truncate(obj: Any, limit: Optional[int]) -> Tuple[Optional[str], bool]:
    """Serialize an object to a compact JSON string and optionally truncate it.

    This function first strips any binary content from the object, then
    serializes it to a JSON string. If a `limit` is provided and the string
    exceeds it, the string is truncated.

    Args:
        obj: The object to serialize.
        limit: The maximum length of the output string. If None or <= 0,
            truncation is disabled (but binary stripping still occurs).

    Returns:
        A tuple containing:
        - The serialized (and possibly truncated) string, or None if input is None.
        - A boolean flag indicating whether truncation occurred.
    """
    if obj is None:
        return None, False
    # Binary detection & stripping before serialization
    try:
        if _contains_binary_marker(obj):
            obj = _strip_binary_payload(obj)
    except Exception:
        pass
    try:
        import json
        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        raw = str(obj)
    truncated = False
    if isinstance(limit, int) and limit > 0 and len(raw) > limit:
        truncated = True
        raw = raw[:limit] + f"...<truncated {len(raw)-limit} chars>"
    return raw, truncated


def _unwrap_ai_channel(container: Any) -> Any:
    """Unwrap an ai_* channel wrapper returning inner json payload(s).

    Returns original object on mismatch or failure. Depth / breadth guarded.
    """
    if not isinstance(container, dict) or len(container) != 1:
        return container
    try:
        (only_key, value) = next(iter(container.items()))
    except Exception:
        return container
    if not (isinstance(only_key, str) and only_key.startswith("ai_")):
        return container
    collected: List[Dict[str, Any]] = []

    def _walk(v: Any, depth: int = 0) -> None:
        if depth > 25:
            return
        if isinstance(v, list):
            for item in v[:100]:
                _walk(item, depth + 1)
        elif isinstance(v, dict):
            j = v.get("json") if isinstance(v.get("json"), dict) else None
            if j:
                collected.append(j)
            else:
                if any(
                    k in v
                    for k in (
                        "model",
                        "model_name",
                        "modelId",
                        "model_id",
                        "tokenUsage",
                        "tokenUsageEstimate",
                    )
                ):
                    collected.append(v)

    try:
        _walk(value)
    except Exception:
        return container
    if not collected:
        return container
    if len(collected) == 1:
        return collected[0]
    import json
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for c in collected:
        try:
            sig = json.dumps(c, sort_keys=True)[:4000]
        except Exception:
            sig = str(id(c))
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq


def _unwrap_generic_json(container: Any) -> Any:
    """Promote deeply nested list/main/json wrappers to direct json payload(s)."""
    if not isinstance(container, dict):
        return container
    collected: List[Dict[str, Any]] = []

    def _walk(o: Any, depth: int = 0) -> None:
        if depth > 25 or len(collected) >= 150:
            return
        if isinstance(o, dict):
            j = o.get("json") if isinstance(o.get("json"), dict) else None
            if j:
                collected.append(j)
            else:
                for v in o.values():
                    _walk(v, depth + 1)
        elif isinstance(o, list):
            for item in o[:100]:
                _walk(item, depth + 1)

    try:
        _walk(container)
    except Exception:
        return container
    if not collected:
        return container
    if len(collected) == 1:
        return collected[0]
    import json
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for c in collected:
        try:
            sig = json.dumps(c, sort_keys=True)[:4000]
        except Exception:
            sig = str(id(c))
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq


def _normalize_node_io(obj: Any) -> Tuple[Any, Dict[str, bool]]:
    """Normalize node input/output returning (cleaned_obj, flags).

    Passes (order matters):
      1. AI channel unwrapping (single ai_* key).
      2. Generic json promotion (unwrap list/main wrappers down to json).

    Flags:
      unwrapped_ai_channel: AI channel wrapper removed.
      unwrapped_json_root: generic json wrapper removed or promoted.
    """
    flags: Dict[str, bool] = {}
    base = obj
    if isinstance(base, dict):
        # Preserve reference to any top-level binary block so it can be merged
        # back after wrapper unwrapping (preventing loss of placeholders or
        # media token structures when the generic json promotion discards
        # sibling keys).
        binary_block = base.get("binary") if isinstance(base.get("binary"), dict) else None
        after_ai = _unwrap_ai_channel(base)
        if after_ai is not base:
            flags["unwrapped_ai_channel"] = True
        after_json = _unwrap_generic_json(after_ai)
        if after_json is not after_ai:
            flags["unwrapped_json_root"] = True
        if binary_block and isinstance(after_json, dict) and "binary" not in after_json:
            # Merge; prefer unwrapped json keys, append binary at end.
            merged = dict(after_json)
            merged["binary"] = binary_block
            after_json = merged
        return after_json, flags
    return base, flags


_GENERATION_PROVIDER_MARKERS = [
    # Core providers
    "openai", "anthropic", "gemini", "mistral", "groq",
    # Chat wrappers / legacy internal labels
    "lmchat", "lmopenai",
    # Newly added / upstream supported vendors
    "cohere", "deepseek", "ollama", "openrouter", "bedrock", "vertex", "huggingface", "xai",
    # Custom provider marker for Limescape Docs node
    "limescape",
]


def _detect_generation(node_type: str, node_run: NodeRun) -> bool:
    """Determine if a node run should be classified as a 'generation'.

    A node run is considered a generation if it meets one of the following
    criteria, in order:
    1.  A `tokenUsage` object is found anywhere in its `run.data`.
    2.  The `node_type` contains a known LLM provider marker (e.g., 'openai')
        and does not contain an exclusion marker (e.g., 'embedding').

    Args:
        node_type: The static type of the n8n node.
        node_run: The runtime data for the node execution.

    Returns:
        True if the node run is classified as a generation, False otherwise.
    """
    # Fast path: top-level tokenUsage
    if node_run.data and any(k in node_run.data for k in ("tokenUsage", "tokenUsageEstimate")):
        return True
    # Nested search (common n8n shape: {"ai_languageModel": [[{"json": {"tokenUsage": {...}}}]]})
    if any(
        _find_nested_key(node_run.data, k) is not None
        for k in ("tokenUsage", "tokenUsageEstimate")
    ):
        return True
    lowered = node_type.lower()
    if any(excl in lowered for excl in ("embedding", "embeddings", "reranker")):
        return False
    return any(marker in lowered for marker in _GENERATION_PROVIDER_MARKERS)


def _extract_usage(node_run: NodeRun) -> Optional[LangfuseUsage]:
    """Extract and normalize token usage data from a node run.

    Supports both 'tokenUsage' and 'tokenUsageEstimate' (precedence: tokenUsage).
    Normalizes various key formats (promptTokens, completion, etc.) into
    LangfuseUsage(input, output, total). Synthesizes total when possible.
    """
    data = node_run.data or {}
    tu: Any = None

    # 1. Top-level fast path (precedence order)
    for key in ("tokenUsage", "tokenUsageEstimate"):
        if isinstance(data, dict) and isinstance(data.get(key), dict):
            tu = data.get(key)
            break

    # 2. Nested search if not found top-level
    if tu is None:
        for key in ("tokenUsage", "tokenUsageEstimate"):
            container = _find_nested_key(data, key)
            if container and isinstance(container.get(key), dict):
                tu = container[key]
                break

    if not isinstance(tu, dict):
        # Limescape Docs custom usage keys may live directly in flattened data structure.
        # Look for totalInputTokens / totalOutputTokens / totalTokens at any nesting level.
        # We'll search top-level first, then one nested level to keep cost low.
        def _scan_custom(obj: Any, depth: int = 25) -> Optional[Dict[str, Any]]:
            if depth < 0:
                return None
            if isinstance(obj, dict):
                keys = obj.keys()
                if any(k in keys for k in ("totalInputTokens", "totalOutputTokens", "totalTokens")):
                    return obj
                for v in list(obj.values())[:150]:  # shallow guard
                    found = _scan_custom(v, depth - 1)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj[:150]:
                    found = _scan_custom(item, depth - 1)
                    if found:
                        return found
            return None
        tu = _scan_custom(data)
        if not isinstance(tu, dict):
            return None

    # Gather candidates
    input_val = tu.get("input")
    output_val = tu.get("output")
    total_val = tu.get("total")

    # Legacy verbose keys
    p_tokens = tu.get("promptTokens")
    c_tokens = tu.get("completionTokens")
    t_tokens = (
        tu.get("totalTokens")
        or tu.get("totalInputTokens")
        or tu.get("totalOutputTokens")
    )
    # Limescape custom flattened keys (treat totalInputTokens /
    # totalOutputTokens as input/output if canonical missing)
    if input_val is None:
        input_val = tu.get("totalInputTokens", input_val)
    if output_val is None:
        output_val = tu.get("totalOutputTokens", output_val)

    # Alternate short legacy keys
    p_short = tu.get("prompt")
    c_short = tu.get("completion")
    # total already in total_val (or t_tokens)

    # Reconcile precedence
    if input_val is None:
        if p_tokens is not None:
            input_val = p_tokens
        elif p_short is not None:
            input_val = p_short
    if output_val is None:
        if c_tokens is not None:
            output_val = c_tokens
        elif c_short is not None:
            output_val = c_short
    if total_val is None:
        if t_tokens is not None:
            total_val = t_tokens
        elif input_val is not None and output_val is not None:
            try:
                total_val = int(input_val) + int(output_val)
            except Exception:
                pass

    if input_val is None and output_val is None and total_val is None:
        return None

    return LangfuseUsage(input=input_val, output=output_val, total=total_val)


def _find_nested_key(
    data: Any, target: str, max_depth: int = 150
) -> Optional[Dict[str, Any]]:
    """Perform a depth-limited search for a dictionary containing a specific key.

    This helper traverses a nested structure of dictionaries and lists to find
    the first dictionary that has `target` as a key. It is used to locate
    data like `tokenUsage`, `tokenUsageEstimate`, or `model` which may be deeply nested.

    Args:
        data: The object to search.
        target: The dictionary key to find.
        max_depth: The maximum recursion depth.

    Returns:
        The first dictionary found containing the target key, or None.
    """
    try:
        if max_depth < 0:
            return None
        if isinstance(data, dict):
            if target in data:
                return data
            for v in data.values():
                found = _find_nested_key(v, target, max_depth - 1)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for item in data[:150]:  # guard large lists
                found = _find_nested_key(item, target, max_depth - 1)
                if found is not None:
                    return found
    except Exception:
        return None
    return None


def _extract_model_value(data: Any) -> Optional[str]:
    """Attempt to find a model name within a complex, nested node output object.

    This function uses a series of heuristics to locate a model identifier,
    searching for common keys like 'model', 'model_name', etc. It performs a
    breadth-first search and can handle various n8n data structures, including
    JSON-encoded strings.

    Args:
        data: The object to search for a model name, typically `run.data`.

    Returns:
        The model name string if found, otherwise None.
    """
    if not data:
        return None
    keys = ("model", "model_name", "modelId", "model_id")

    # 1. Targeted channel fast-path
    try:
        if isinstance(data, dict):
            for chan_key, chan_val in data.items():
                if (
                    isinstance(chan_key, str)
                    and chan_key.startswith("ai_")
                    and isinstance(chan_val, list)
                ):
                    for outer in chan_val:
                        if not isinstance(outer, list):
                            continue
                        for item in outer:
                            if not isinstance(item, dict):
                                continue
                            j = item.get("json") if isinstance(item.get("json"), dict) else None
                            if not j:
                                continue
                            # Direct model keys
                            for k in keys:
                                val = j.get(k)
                                if isinstance(val, str) and val:
                                    return val
                            opts = j.get("options") if isinstance(j.get("options"), dict) else None
                            if opts:
                                for k in keys:
                                    val = opts.get(k)
                                    if isinstance(val, str) and val:
                                        return val
    except Exception:
        pass

    # Helper: inspect a mapping for immediate model-like keys or inside an 'options' object
    def _direct_scan(obj: Dict[str, Any]) -> Optional[str]:
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v:
                return v
        opts = obj.get("options") if isinstance(obj.get("options"), dict) else None
        if opts:
            for k in keys:
                v = opts.get(k)
                if isinstance(v, str) and v:
                    return v
        return None

    # 2 & 3. Immediate / json-wrapper level scan
    if isinstance(data, dict):
        direct = _direct_scan(data)
        if direct:
            return direct
        j = data.get("json") if isinstance(data.get("json"), dict) else None
        if j:
            direct = _direct_scan(j)
            if direct:
                return direct

    # 4. Prepare BFS queue; include parsed JSON string expansions on-demand
    queue: Deque[Tuple[Any, int]] = deque([(data, 0)])
    visited = 0
    max_nodes = 800  # slightly higher to account for json-string expansions
    max_depth = 25
    while queue and visited < max_nodes:
        current, depth = queue.popleft()
        visited += 1
        try:
            if isinstance(current, dict):
                direct = _direct_scan(current)
                if direct:
                    return direct
                if depth < max_depth:
                    for v in current.values():
                        queue.append((v, depth + 1))
            elif isinstance(current, list):
                if depth < max_depth:
                    for item in current[:150]:  # cap list breadth
                        queue.append((item, depth + 1))
            elif isinstance(current, str):
                # Light heuristic: only attempt parse if plausible JSON object containing 'model'
                s = current.strip()
                if 10 < len(s) < 50_000 and s.startswith('{') and 'model' in s:
                    # Avoid expensive parsing repeatedly for huge unrelated strings.
                    import json
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, dict):
                            queue.append((parsed, depth + 1))
                    except Exception:
                        pass
        except Exception:
            continue
    return None


def _looks_like_model_param_key(key: str) -> bool:
    lk = key.lower()
    if lk == "mode":  # exclude common false positive
        return False
    # Exclude known non-model keys containing 'model'
    if lk in {"modelprovider", "model_type", "modeltype", "modelprovidername"}:
        return False
    return ("model" in lk) or ("deployment" in lk)


def _extract_model_from_parameters(node: WorkflowNode) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Attempt to extract a model/deployment name from static workflow node parameters.

    Returns a tuple ``(model_value, key_path, extra_meta)`` or ``None``. Previous
    implementation returned early inside loops which triggered a mypy
    ``unreachable`` warning for subsequent code. This refactor preserves first
    match semantics while using an accumulator to avoid early ``return`` inside
    the BFS loop (eliminating the unreachable diagnostic).
    """
    params = getattr(node, "parameters", None)
    if not isinstance(params, dict):
        return None
    from collections import deque as _dq

    queue = _dq([(params, "parameters", 0)])
    seen = 0
    max_nodes = 120
    priority_keys = [
        "model",
        "customModel",
        "deploymentName",
        "deployment",
        "modelName",
        "model_id",
        "modelId",
    ]
    result: Optional[Tuple[str, str, Dict[str, Any]]] = None
    azure_flag = "azure" in node.type.lower()

    while queue and seen < max_nodes:
        current, path, depth = queue.popleft()
        seen += 1
        if result is not None:
            break
        # 1. Priority exact key pass
        for pk in priority_keys:
            if pk in current:
                v = current.get(pk)
                if isinstance(v, str) and v and not v.startswith("={{"):
                    meta_pk: Dict[str, Any] = {}
                    if azure_flag:
                        meta_pk["n8n.model.is_deployment"] = True
                    result = (v, f"{path}.{pk}", meta_pk)
                    break
        # 2. Generic fallback (excluding modelProvider handled in helper)
        for k, v in current.items():
            if (
                isinstance(v, str)
                and v
                and not v.startswith("={{")
                and _looks_like_model_param_key(k)
            ):
                meta_k: Dict[str, Any] = {}
                if azure_flag:
                    meta_k["n8n.model.is_deployment"] = True
                result = (v, f"{path}.{k}", meta_k)
                break
        # Recurse breadth-first
        if depth < 25:
            for k, v in current.items():
                if isinstance(v, dict):
                    queue.append((v, f"{path}.{k}", depth + 1))
    return result


def _build_reverse_edges(workflow_data: Any) -> Dict[str, List[str]]:
    """Construct a reverse edge map: node -> list of predecessor node names.

    Mirrors the inline logic previously inside `map_execution_to_langfuse`.
    Failures are swallowed (best-effort) to preserve existing lenient behavior.
    """
    reverse: Dict[str, List[str]] = {}
    try:
        connections = getattr(workflow_data, "connections", {}) or {}
        for from_node, conn_types in connections.items():
            if not isinstance(conn_types, dict):
                continue
            for _conn_type, outputs in conn_types.items():
                if not isinstance(outputs, list):
                    continue
                for output_list in outputs:
                    if not isinstance(output_list, list):
                        continue
                    for edge in output_list:
                        to_node = edge.get("node") if isinstance(edge, dict) else None
                        if to_node:
                            reverse.setdefault(to_node, []).append(from_node)
    except Exception:
        return {}
    return reverse


def _build_child_agent_map(workflow_data: Any) -> Dict[str, Tuple[str, str]]:
    """Build mapping child_node -> (agent_node, connection_type) for ai_* edges.

    Reproduces the original pattern-based hierarchy detection logic.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    try:
        connections = getattr(workflow_data, "connections", {}) or {}
        for from_node, conn_types in connections.items():
            if not isinstance(conn_types, dict):
                continue
            for conn_type, outputs in conn_types.items():
                if not (isinstance(conn_type, str) and conn_type.startswith("ai_")):
                    continue
                if not isinstance(outputs, list):
                    continue
                for output_list in outputs:
                    if not isinstance(output_list, list):
                        continue
                    for edge in output_list:
                        if isinstance(edge, dict):
                            agent = edge.get("node")
                            if agent:
                                mapping[from_node] = (agent, conn_type)
    except Exception:
        return {}
    return mapping


def _flatten_runs(run_data: Dict[str, List[NodeRun]]) -> List[Tuple[int, str, int, NodeRun]]:
    """Flatten runData into a chronologically sorted list for processing.

    Returns list of tuples: (startTime_ms, node_name, run_index, NodeRun).
    """
    flattened: List[Tuple[int, str, int, NodeRun]] = []
    for node_name, runs in run_data.items():
        for idx, run in enumerate(runs):
            flattened.append((run.startTime, node_name, idx, run))
    flattened.sort(key=lambda x: x[0])
    return flattened


def _resolve_parent(
    *,
    node_name: str,
    run: NodeRun,
    trace_id: str,
    child_agent_map: Dict[str, Tuple[str, str]],
    last_span_for_node: Dict[str, str],
    reverse_edges: Dict[str, List[str]],
    root_span_id: str,
) -> Tuple[str, Optional[str], Optional[int]]:
    """Resolve parent span id and previous node info for a node run.

    Implements precedence: agent hierarchy -> runtime exact -> runtime last ->
    static reverse graph -> root.
    Returns (parent_id, prev_node, prev_node_run_index).
    """
    parent_id: str = root_span_id
    prev_node: Optional[str] = None
    prev_node_run: Optional[int] = None

    # 1. Agent hierarchy
    if node_name in child_agent_map:
        agent_name, _link_type = child_agent_map[node_name]
        if agent_name in last_span_for_node:
            parent_id = last_span_for_node[agent_name]
        else:
            parent_id = root_span_id
        prev_node = agent_name
        return parent_id, prev_node, prev_node_run

    # 2. Runtime sequential (exact run id if provided)
    if run.source and len(run.source) > 0 and run.source[0].previousNode:
        prev_node = run.source[0].previousNode
        prev_node_run = run.source[0].previousNodeRun
        if prev_node_run is not None:
            parent_id = str(uuid5(SPAN_NAMESPACE, f"{trace_id}:{prev_node}:{prev_node_run}"))
        else:
            parent_id = last_span_for_node.get(prev_node, root_span_id)
        return parent_id, prev_node, prev_node_run

    # 3. Static reverse graph fallback
    if node_name in reverse_edges and reverse_edges[node_name]:
        graph_preds = reverse_edges[node_name]
        chosen_pred = None
        for cand in graph_preds:
            if cand in last_span_for_node:
                chosen_pred = cand
                break
        if not chosen_pred:
            chosen_pred = graph_preds[0]
        prev_node = chosen_pred
        parent_id = last_span_for_node.get(chosen_pred, parent_id)
        return parent_id, prev_node, prev_node_run

    # 4. Root fallback
    return parent_id, prev_node, prev_node_run


def _prepare_io_and_output(
    *,
    raw_input_obj: Any,
    raw_output_obj: Any,
    is_generation: bool,
    node_type: str,
    truncate_limit: Optional[int],
) -> Tuple[
    Any,  # norm_input_obj
    Any,  # norm_output_obj
    Optional[str],  # input_str
    bool,  # input_trunc
    Optional[str],  # output_str (possibly extracted text)
    bool,  # output_trunc
    Dict[str, bool],  # input_flags
    Dict[str, bool],  # output_flags
]:
    """Normalize and serialize input/output plus generation text extraction.

    This consolidates prior inline logic to reduce complexity in the main loop.
    The behavior is intentionally identical to the pre-refactor implementation:
      1. Normalize (unwrap AI channel then generic json) for input & output.
      2. Serialize & truncate (with binary stripping) input first.
      3. For generation spans attempt concise textual extraction (Gemini / Limescape
         markdown / ai_ wrapper text) using the normalized output object.
      4. If extraction succeeds, use extracted text (never truncated); else serialize
         normalized output with truncation.
    """
    # Strip System prompt from LangChain LMChat generation inputs BEFORE normalization
    # to preserve the original structure with messages array
    if is_generation:
        raw_input_obj = _strip_system_prompt_from_langchain_lmchat(
            raw_input_obj, node_type
        )

    norm_input_obj, input_flags = _normalize_node_io(raw_input_obj)
    norm_output_obj, output_flags = _normalize_node_io(raw_output_obj)

    # Reattach promoted top-level binary if normalization unwrapped list root,
    # ensuring binary slots persist. This occurs when earlier promotion added
    # run.data['binary'] and _normalize_node_io unwrapped list content.
    try:
        if (
            isinstance(raw_output_obj, dict)
            and "binary" in raw_output_obj
            and isinstance(raw_output_obj["binary"], dict)
            and isinstance(norm_output_obj, list)
        ):
            norm_output_obj = {
                "binary": raw_output_obj["binary"],
                "_items": norm_output_obj,
            }
            output_flags["promoted_binary_wrapper"] = True
    except Exception:
        pass
    input_str, input_trunc = _serialize_and_truncate(norm_input_obj, truncate_limit)
    output_str: Optional[str] = None
    output_trunc = False
    if is_generation:
        try:
            candidate = norm_output_obj
            extracted_text: Optional[str] = None
            # Gemini / Vertex structure
            if isinstance(candidate, dict):
                resp = (
                    candidate.get("response")
                    if isinstance(candidate.get("response"), dict)
                    else candidate
                )
                gens = resp.get("generations") if isinstance(resp, dict) else None
                if isinstance(gens, list) and gens:
                    first_layer = gens[0]
                    if isinstance(first_layer, list) and first_layer:
                        inner = first_layer[0]
                        if isinstance(inner, dict):
                            txt = inner.get("text")
                            if isinstance(txt, str) and txt.strip():
                                extracted_text = txt
            # Limescape Docs markdown preference
            if (
                extracted_text is None
                and isinstance(candidate, dict)
                and "limescape" in node_type.lower()
            ):
                md_val = candidate.get("markdown")
                if isinstance(md_val, str) and md_val.strip():
                    extracted_text = md_val
            # Fallback nested ai_* channel text search
            if extracted_text is None and isinstance(candidate, dict):
                for k, v in candidate.items():
                    if k.startswith("ai_") and isinstance(v, list) and v:
                        for layer in v[:3]:
                            if isinstance(layer, list) and layer:
                                first = layer[0]
                                if isinstance(first, dict):
                                    txt = first.get("text")
                                    if isinstance(txt, str) and txt.strip():
                                        extracted_text = txt
                                        break
                        if extracted_text:
                            break
            if extracted_text is not None:
                output_str = extracted_text
                output_trunc = False
        except Exception:
            # Best-effort; fall back to full serialization
            pass
    if output_str is None:
        output_str, output_trunc = _serialize_and_truncate(
            norm_output_obj, truncate_limit
        )
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


# --------------------------- Stage 3 Refactor Helpers ---------------------------

def _extract_model_and_metadata(
    *,
    run: NodeRun,
    node_name: str,
    node_type: str,
    is_generation: bool,
    wf_node_obj: Dict[str, WorkflowNode],
    raw_input_obj: Any,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Extract model value plus related metadata flags.

    Mirrors previous inline logic without altering behavior. Returns
    (model_val, metadata_fragment).
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
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Detect Gemini empty-output anomaly returning (status_override, metadata_fragment).

    Side-effect: may mutate run.error to inject synthetic error object (matching prior behavior).
    Returns status_override ('error') when anomaly detected else None.
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


# --------------------------- Stage 4 Refactor: MappingContext ---------------------------

@dataclass
class MappingContext:
    """Aggregate shared mapping state (static + mutable) for clarity.

    Pure organizational refactor; behavior must remain unchanged.
    """
    trace_id: str
    root_span_id: str
    wf_node_lookup: Dict[str, Dict[str, Optional[str]]]
    wf_node_obj: Dict[str, WorkflowNode]
    reverse_edges: Dict[str, List[str]]
    child_agent_map: Dict[str, Tuple[str, str]]
    truncate_limit: Optional[int]
    last_span_for_node: Dict[str, str] = field(default_factory=dict)
    last_output_data: Dict[str, Any] = field(default_factory=dict)


def _map_execution(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    *,
    collect_binaries: bool = False,
) -> tuple[LangfuseTrace, list[BinaryAsset]]:
    """Internal mapping core returning (trace, assets).

    When collect_binaries=True, binary assets are extracted during the primary
    node loop (no second traversal). The original raw run.data remains
    unchanged; a cloned version with provisional placeholders feeds span I/O
    serialization.
    """
    # Trace id is now exactly the n8n execution id (string). Simplicity &
    # direct searchability in Langfuse UI.
    # NOTE: This shipper supports ONE n8n instance per Langfuse project. If
    # you aggregate multiple instances into a single Langfuse project, raw
    # numeric execution ids may collide. Use separate Langfuse projects per
    # instance.
    trace_id = str(record.id)
    # Normalize potentially naive datetimes to UTC (defensive: DB drivers /
    # external inputs may yield naive objects)
    started_at = (
        record.startedAt
        if record.startedAt.tzinfo is not None
        else record.startedAt.replace(tzinfo=timezone.utc)
    )
    stopped_raw = record.stoppedAt
    if stopped_raw is not None and stopped_raw.tzinfo is None:
        stopped_raw = stopped_raw.replace(tzinfo=timezone.utc)
    # New naming convention: trace name is just the workflow name (fallback 'execution').
    base_name = record.workflowData.name or "execution"
    trace = LangfuseTrace(
        id=trace_id,
        name=base_name,
        timestamp=started_at,
        metadata={
            "workflowId": record.workflowId,
            "status": record.status,
            # executionId removed to avoid duplication; use root span metadata
            # key n8n.execution.id instead
        },
    )

    # Build lookup of workflow node -> (type, category)
    wf_node_lookup: Dict[str, Dict[str, Optional[str]]] = {
        n.name: {"type": n.type, "category": n.category} for n in record.workflowData.nodes
    }
    wf_node_obj: Dict[str, WorkflowNode] = {n.name: n for n in record.workflowData.nodes}

    # Root span representing the execution as a whole
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
    collected_assets: list[BinaryAsset] = []
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
    # Binary collection (if enabled) operates on a clone to avoid mutating
    # original run.data object used for input propagation caching. The
    # placeholders inserted here are later replaced by Langfuse media token
    # strings (see media_api.patch_and_upload_media). This is a breaking
    # change from the prior Azure JSON reference object approach.
        limit_hit = False
        promoted_item_binary = False
        if collect_binaries:
            (
                mutated_output,
                assets,
                limit_hit,
                promoted_item_binary,
            ) = _collect_binary_assets(
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
        status_override, anomaly_meta = _detect_gemini_empty_output_anomaly(
            is_generation=is_generation,
            norm_output_obj=norm_output_obj,
            run=run,
            node_name=node_name,
        )
        if anomaly_meta:
            metadata.update(anomaly_meta)
        if status_override:
            status_norm = status_override
        # Media scan cap: if extended discovery exceeded configured max we attach
        # scan_asset_limit error code (fail-open; remaining assets ignored).
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
            usage=usage,
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
        # generation classification lives on span; no separate collection
    return trace, collected_assets


def map_execution_to_langfuse(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    *,
    filter_ai_only: bool = False,
) -> LangfuseTrace:
    trace, _assets = _map_execution(
        record, truncate_limit=truncate_limit, collect_binaries=False
    )
    if filter_ai_only:
        _apply_ai_filter(trace, record)
    return trace


def _collect_binary_assets(
    run_data_obj: Any, *, execution_id: int, node_name: str, run_index: int
) -> tuple[Any, list[BinaryAsset], bool, bool]:
    """Traverse a run.data dict, extracting n8n-style binary objects.

    Returns (cloned_structure_with_placeholders, assets).
    We only target the canonical n8n pattern under run.data['binary'] entries.
    """
    assets: list[BinaryAsset] = []
    limit_hit = False
    promoted_item_binary = False
    if not isinstance(run_data_obj, dict):
        return run_data_obj, assets, limit_hit, promoted_item_binary
    import copy
    cloned = copy.deepcopy(run_data_obj)
    # ---------------- Item-level binary promotion -----------------
    # Some nodes emit only wrapper lists (e.g. main -> [[[ { json, binary } ]]])
    # without a top-level 'binary' key. We scan shallow wrapper keys (current
    # unwrapping patterns) for list structures containing dict items with a
    # 'binary' key and merge those slot maps into a new top-level 'binary'.
    # First occurrence of a slot wins to preserve deterministic behavior.
    if "binary" not in cloned:
        aggregated: dict[str, Any] = {}
        # Candidate wrapper keys we commonly unwrap; we also consider any
        # list-valued top-level key to avoid missing unconventional wrappers.
        for k, v in list(cloned.items())[:20]:  # bound to avoid pathological cases
            if not isinstance(v, list):
                continue
            # Traverse up to depth 3 (mirrors existing unwrap depth) collecting
            # item dicts that contain a 'binary' dict.
            stack: list[tuple[Any, int]] = [(v, 0)]
            while stack:
                cur, depth = stack.pop()
                if depth > 3:
                    continue
                if isinstance(cur, list):
                    for itm in cur[:100]:  # cap iteration for safety
                        stack.append((itm, depth + 1))
                elif isinstance(cur, dict):
                    bin_block = cur.get("binary")
                    if isinstance(bin_block, dict):
                        for slot, slot_val in bin_block.items():
                            if slot not in aggregated and isinstance(slot_val, dict):
                                # Shallow copy sufficient (will be deep-copied in
                                # canonical processing below)
                                aggregated[slot] = copy.deepcopy(slot_val)
                    # Do not dive into nested dicts beyond locating 'binary'
        if aggregated:
            cloned["binary"] = aggregated
            promoted_item_binary = True
    # Canonical n8n binary block processing (if present). We do NOT early-return
    # when absent so that extended discovery can still run for nodes that only
    # expose ad-hoc file-like dicts or data URLs.
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
                import base64 as _b64, hashlib as _hash
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
    # Discover additional assets outside canonical blocks (extended scan)
    from .config import get_settings  # lazy import to avoid cycle in tests
    settings = get_settings()
    extra_assets, extra_limit_hit = _discover_additional_binary_assets(
        cloned,
        execution_id=execution_id,
        node_name=node_name,
        run_index=run_index,
        max_assets=settings.EXTENDED_MEDIA_SCAN_MAX_ASSETS,
    )
    if extra_assets:
        # Insert provisional placeholders at the first path segment location
        for asset in extra_assets:
            # Patch the exact leaf indicated by asset.field_path. We support simple dot paths
            # and ignore list indices (future enhancement) while still capturing dict leaves.
            try:
                raw_path = asset.field_path
                # Remove any list index notation like foo[0][1]
                cleaned = re.sub(r"\[[0-9]+\]", "", raw_path)
                segments = [seg for seg in cleaned.split(".") if seg]
                if segments and segments[0] == "<root>":  # normalize root sentinel
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


_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$")


def _discover_additional_binary_assets(
    obj: Any,
    *,
    execution_id: int,
    node_name: str,
    run_index: int,
    max_assets: int,
) -> tuple[list[BinaryAsset], bool]:
    """Discover binary assets outside canonical run.data['binary'] blocks.

    Supported (Option 4 scope):
      * data URLs: data:<mime>;base64,<payload>
      * file-like dicts with keys (mimeType|fileType) and (fileName|name) and base64 'data'
      * standalone file-like dict with (mimeType + data) even if no filename
      * long base64 strings (>=64 chars) when adjacent dict has mimeType or fileName

    Excludes (staged for later): Buffer objects, pure hex blobs.
    """
    found: list[BinaryAsset] = []
    limit_hit = False
    seen_sha: set[str] = set()

    def _add(base64_str: str, mime: str | None, fname: str | None, path: str) -> None:
        nonlocal limit_hit
        if max_assets > 0 and len(found) >= max_assets:
            limit_hit = True
            return
        try:
            import base64 as _b64, hashlib as _hash
            if len(base64_str) < 64:
                return
            full_bytes = _b64.b64decode(base64_str, validate=False)
            h = _hash.sha256(full_bytes).hexdigest()
            if h in seen_sha:
                return
            seen_sha.add(h)
            # Compute path segments without array indices
            clean_path = re.sub(r"\[[0-9]+\]", "", path)
            path_segments = [
                seg for seg in clean_path.split(".")
                if seg and seg != "<root>"
            ]
            found.append(
                BinaryAsset(
                    execution_id=execution_id,
                    node_name=node_name,
                    run_index=run_index,
                    field_path=path,
                    original_path_segments=[seg for seg in re.sub(r"\[[0-9]+\]", "", path).split(".") if seg and seg != "<root>"],
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

    def _walk(o: Any, path: str, parent: Any | None) -> None:  # noqa: C901 - complexity acceptable, bounded
        nonlocal limit_hit
        if max_assets > 0 and len(found) >= max_assets:
            limit_hit = True
            return
        if isinstance(o, dict):
            # File-like dict pattern
            mime = o.get("mimeType") or o.get("mime_type") or o.get("fileType")
            fname = o.get("fileName") or o.get("name")
            data_field = o.get("data")
            if isinstance(data_field, str):
                # data URL pattern
                m = _DATA_URL_RE.match(data_field)
                if m and len(m.group(2)) >= 64:
                    # Replace original 'data' key value (not a new dataUrl key)
                    _add(m.group(2), m.group(1), fname, f"{path}.data")
                else:
                    # Plain base64 with supporting keys (relaxed: allow >=64 chars without requiring
                    # global long-base64 regex; avoids missing moderately sized assets).
                    if (mime or fname) and len(data_field) >= 64:
                        _add(data_field, mime, fname, f"{path}.data")
            # Long base64 value sitting in a dict with contextual keys
            for k, v in o.items():
                if isinstance(v, str) and len(v) >= 64:
                    sib_keys = {sk.lower() for sk in o.keys()}
                    if any(h in sib_keys for h in ("mimetype", "filename", "filetype")):
                        _add(v, mime, fname, f"{path}.{k}")
            for k, v in list(o.items())[:50]:
                _walk(v, f"{path}.{k}" if path else k, o)
        elif isinstance(o, list):
            for idx, item in enumerate(o[:100]):
                _walk(item, f"{path}[{idx}]", parent)
        elif isinstance(o, str):
            # Standalone data URL string
            m = _DATA_URL_RE.match(o)
            if m and len(m.group(2)) >= 64:
                # For a standalone data URL string, we cannot reliably patch unless
                # parent key known; keep existing behavior (no replacement) – future enhancement.
                _add(m.group(2), m.group(1), None, path or "<root>")

    _walk(obj, "", None)
    return found, limit_hit


def map_execution_with_assets(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    collect_binaries: bool = False,
    *,
    filter_ai_only: bool = False,
) -> MappedTraceWithAssets:
    # collect_binaries kept for backwards compatibility in case external code
    # passed False explicitly.
    trace, assets = _map_execution(
        record, truncate_limit=truncate_limit, collect_binaries=collect_binaries
    )
    if filter_ai_only:
        _apply_ai_filter(trace, record)
    return MappedTraceWithAssets(trace=trace, assets=assets if collect_binaries else [])


def _apply_ai_filter(trace: LangfuseTrace, record: N8nExecutionRecord) -> None:
    """Apply windowed AI filtering (lean context retention).

    Steps:
      1. Identify AI spans.
      2. If none → root-only fallback.
            3. Include root, all AI spans, strictly up to 2 immediate chronological
                 spans preceding earliest AI (excluding root which is always kept),
                 logical in-between spans whose ancestor chain touches an AI span,
                 and strictly up to 2 immediate chronological spans after latest AI.
      4. Record window start/end metadata.
    """
    try:
        if not trace.spans:
            return
        root_span = next(
            (s for s in trace.spans if s.parent_id is None or "n8n.execution.id" in s.metadata),
            trace.spans[0],
        )
        wf_lookup: Dict[str, Tuple[Optional[str], Optional[str]]] = {
            n.name: (n.type, n.category) for n in record.workflowData.nodes
        }
        original_order = list(trace.spans)
        spans_by_id: Dict[str, LangfuseSpan] = {s.id: s for s in original_order}
        ai_span_ids: set[str] = set()
        ai_indices: List[int] = []
        for idx, span in enumerate(original_order):
            if span is root_span:
                continue
            n_type, n_cat = wf_lookup.get(span.name, (None, None))
            if is_ai_node(n_type, n_cat):
                ai_span_ids.add(span.id)
                ai_indices.append(idx)
        keep_ids: set[str] = {root_span.id}
        if not ai_indices:
            root_span.metadata["n8n.filter.ai_only"] = True
            root_span.metadata["n8n.filter.excluded_node_count"] = len(original_order) - 1
            root_span.metadata["n8n.filter.no_ai_spans"] = True
            trace.spans = [root_span]
            return
        first_ai_idx = ai_indices[0]
        last_ai_idx = ai_indices[-1]
        keep_ids.update(ai_span_ids)
        # Pre-context: two spans immediately before first AI (indices first_ai_idx-1, first_ai_idx-2)
        pre_context_count = 0
        if first_ai_idx - 1 >= 1:  # index 0 is root already included
            keep_ids.add(original_order[first_ai_idx - 1].id)
            pre_context_count += 1
        if first_ai_idx - 2 >= 1:
            keep_ids.add(original_order[first_ai_idx - 2].id)
            pre_context_count += 1
        index_by_id = {s.id: i for i, s in enumerate(original_order)}

        def _is_on_chain(span: LangfuseSpan) -> bool:
            idx = index_by_id.get(span.id, -1)
            if idx < first_ai_idx or idx > last_ai_idx:
                return False
            guard = 0
            cur = span
            while cur and guard < 1000 and cur.parent_id:
                guard += 1
                if cur.parent_id in ai_span_ids:
                    return True
                cur = spans_by_id.get(cur.parent_id)
            return False

        chain_context_count = 0
        for i in range(first_ai_idx + 1, last_ai_idx):
            span = original_order[i]
            if span.id in keep_ids:
                continue
            if _is_on_chain(span):
                keep_ids.add(span.id)
                chain_context_count += 1
        # Post-context: two spans immediately after last AI (indices last_ai_idx+1, last_ai_idx+2)
        post_context_count = 0
        if last_ai_idx + 1 < len(original_order):
            keep_ids.add(original_order[last_ai_idx + 1].id)
            post_context_count += 1
        if last_ai_idx + 2 < len(original_order):
            keep_ids.add(original_order[last_ai_idx + 2].id)
            post_context_count += 1
        filtered_spans = [s for s in original_order if s.id in keep_ids]
        excluded_count = len(original_order) - len(filtered_spans)
        root_span.metadata["n8n.filter.window_start_span"] = original_order[first_ai_idx].name
        root_span.metadata["n8n.filter.window_end_span"] = original_order[last_ai_idx].name
        root_span.metadata["n8n.filter.ai_only"] = True
        root_span.metadata["n8n.filter.excluded_node_count"] = excluded_count
        root_span.metadata["n8n.filter.pre_context_count"] = pre_context_count
        root_span.metadata["n8n.filter.post_context_count"] = post_context_count
        root_span.metadata["n8n.filter.chain_context_count"] = chain_context_count
        trace.spans = filtered_spans
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                (
                    "AI window filter summary: execution_id=%s total_spans=%d kept=%d excluded=%d "
                    "ai_spans=%d pre_context=%d chain_context=%d post_context=%d window_start=%s window_end=%s"
                ),
                record.id,
                len(original_order),
                len(filtered_spans),
                excluded_count,
                len(ai_span_ids),
                pre_context_count,
                chain_context_count,
                post_context_count,
                original_order[first_ai_idx].name,
                original_order[last_ai_idx].name,
            )
    except Exception as e:  # pragma: no cover
        logger.warning("AI filtering failed (fail-open): %s", e)
