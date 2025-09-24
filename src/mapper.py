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

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Deque
from collections import deque
import re
from uuid import uuid5, UUID, NAMESPACE_DNS

from .models.n8n import N8nExecutionRecord, NodeRun
from .models.langfuse import (
    LangfuseTrace,
    LangfuseSpan,
    LangfuseUsage,
)
from .observation_mapper import map_node_to_observation_type

SPAN_NAMESPACE = uuid5(NAMESPACE_DNS, "n8n-langfuse-shipper-span")


def _epoch_ms_to_dt(ms: int) -> datetime:
    """Convert an epoch timestamp (ms or s) to a timezone-aware UTC datetime.

    n8n can produce timestamps in either milliseconds or seconds. This function
    uses a heuristic to guess the unit: values less than 10^12 are assumed to be
    seconds.

    Args:
        ms: The epoch timestamp.

    Returns:
        A timezone-aware datetime object in UTC.
    """
    # Some n8n exports may already be seconds; if value looks like seconds (10 digits) treat accordingly
    # Heuristic: if ms < 10^12 assume seconds.
    if ms < 1_000_000_000_000:  # seconds
        return datetime.fromtimestamp(ms, tz=timezone.utc)
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{200,}={0,2}$")  # long pure base64 chunk


def _likely_binary_b64(s: str, *, context_key: str | None = None, in_binary_block: bool = False) -> bool:
    """Heuristically detect if a string is likely a base64-encoded binary.

    This function aims to avoid false positives on long text fields that happen
    to be valid base64. It requires a minimum length, character diversity, and
    can use contextual hints like the dictionary key it's associated with.

    Args:
        s: The string to check.
        context_key: The dictionary key associated with the string, if any.
        in_binary_block: Flag indicating if we are already inside a known
            binary data structure.

    Returns:
        True if the string is likely a base64-encoded binary, False otherwise.
    """
    if len(s) < 200:
        return False
    if not _BASE64_RE.match(s) and not s.startswith("/9j/"):  # jpeg magic check retained
        return False
    # Uniform strings (e.g. 'AAAAA...') are still valid base64 but often indicate padding or test data.
    # We previously filtered them out; restore stripping when:
    #  - they appear under a key hinting at base64/binary (data, rawBase64, file, binary)
    #  - or they are inside an explicit binary block
    diversity = len(set(s[:120]))
    if diversity < 4 and not in_binary_block:
        lowered = (context_key or "").lower()
        if not any(h in lowered for h in ("data", "base64", "file", "binary")):
            return False
    return True


def _contains_binary_marker(d: Any, depth: int = 0) -> bool:
    """Recursively scan an object for markers of n8n binary data structures.

    This function detects the standard n8n binary object format (a dictionary
    with 'mimeType' and 'data' keys) or any large base64-encoded string values.
    The search is depth-limited to prevent excessive recursion on complex objects.

    Args:
        d: The object to scan (dict, list, or primitive).
        depth: The current recursion depth.

    Returns:
        True if a binary marker is found, False otherwise.
    """
    if depth > 4:
        return False
    try:
        if isinstance(d, dict):
            if "binary" in d and isinstance(d["binary"], dict):
                for v in d["binary"].values():
                    if isinstance(v, dict):
                        data_section = v.get("data") if isinstance(v.get("data"), dict) else v
                        if isinstance(data_section, dict):
                            mime = data_section.get("mimeType") or data_section.get("mime_type")
                            b64 = data_section.get("data")
                            if mime and isinstance(b64, str) and (len(b64) > 200 and (_BASE64_RE.match(b64) or b64.startswith("/9j/"))):
                                return True
            for val in d.values():
                if _contains_binary_marker(val, depth + 1):
                    return True
        elif isinstance(d, list):
            for item in d[:25]:
                if _contains_binary_marker(item, depth + 1):
                    return True
        elif isinstance(d, str):
            if _likely_binary_b64(d):
                return True
    except Exception:
        return False
    return False


def _strip_binary_payload(obj: Any) -> Any:
    """Recursively replace binary data and large base64 strings with placeholders.

    This function traverses a nested object and replaces any detected binary
    content with a placeholder dictionary, e.g.,
    `{"_binary": True, "note": "binary omitted", "_omitted_len": ...}`.
    This sanitizes the data before serialization, preventing huge payloads.

    Args:
        obj: The object to sanitize.

    Returns:
        A new object with binary content replaced by placeholders.
    """
    placeholder_text = "binary omitted"
    try:
        if isinstance(obj, dict):
            new_d = {}
            for k, v in obj.items():
                if k == "binary" and isinstance(v, dict):
                    # replace inner binary data parts
                    new_bin = {}
                    for bk, bv in v.items():
                        if isinstance(bv, dict):
                            bv2 = dict(bv)
                            data_section = bv2.get("data")
                            # Case 1: legacy assumption where data_section is a dict containing its own 'data'
                            if isinstance(data_section, dict) and isinstance(data_section.get("data"), str):
                                ds = data_section.copy()
                                raw_b64 = ds.get("data", "")
                                if len(raw_b64) > 64:
                                    ds["data"] = placeholder_text
                                    ds["_omitted_len"] = len(raw_b64)
                                    bv2["data"] = ds
                            # Case 2: common n8n shape where bv2['data'] is directly the base64 string alongside mimeType/fileType
                            elif isinstance(data_section, str) and (len(data_section) > 64 and (_BASE64_RE.match(data_section) or data_section.startswith("/9j/"))):
                                bv2["_omitted_len"] = len(data_section)
                                bv2["data"] = placeholder_text
                            new_bin[bk] = bv2
                        else:
                            new_bin[bk] = bv
                    new_d[k] = new_bin
                elif isinstance(v, str) and _likely_binary_b64(v, context_key=k):
                    new_d[k] = {"_binary": True, "note": placeholder_text, "_omitted_len": len(v)}
                else:
                    new_d[k] = _strip_binary_payload(v)
            return new_d
        if isinstance(obj, list):
            out_list = []
            for x in obj:
                if isinstance(x, str) and _likely_binary_b64(x):
                    out_list.append({"_binary": True, "note": placeholder_text, "_omitted_len": len(x)})
                else:
                    out_list.append(_strip_binary_payload(x))
            return out_list
    except Exception:
        return obj
    return obj


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
    """Heuristically unwrap the noisy list nesting of n8n AI node outputs.

    n8n AI nodes often wrap their actual output in a deeply nested structure
    like `{"ai_languageModel": [[{"json": {...}}]]}`. This function attempts
    to extract the core `json` object(s) to improve readability in Langfuse.

    It looks for a dictionary with a single 'ai_*' key and flattens the
    nested lists to return the core payload.

    Args:
        container: The object to unwrap, typically a dictionary from a node's
            input or output.

    Returns:
        The unwrapped object if the pattern is matched, otherwise the original
        container.
    """
    try:
        if not isinstance(container, dict) or len(container) != 1:
            return container
        (only_key, value), = container.items()  # type: ignore[misc]
        if not (isinstance(only_key, str) and only_key.startswith("ai_")):
            return container
        # Flatten nested list layers while collecting dict leaves
        collected: List[Dict[str, Any]] = []

        def _walk(v: Any, depth: int = 0):
            if depth > 6:
                return
            if isinstance(v, list):
                for item in v[:100]:
                    _walk(item, depth + 1)
            elif isinstance(v, dict):
                # Prefer inner 'json' dict if present
                j = v.get("json") if isinstance(v.get("json"), dict) else None
                if j:
                    collected.append(j)  # type: ignore[arg-type]
                else:
                    # If dict itself seems like a payload (has model/usage markers) collect directly
                    if any(k in v for k in ("model", "model_name", "modelId", "model_id", "tokenUsage")):
                        collected.append(v)  # type: ignore[arg-type]
            # Other primitive types ignored

        _walk(value)
        if not collected:
            return container
        if len(collected) == 1:
            return collected[0]
        # Deduplicate by json dump signature (lightweight)
        import json
        seen = set()
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
    except Exception:
        return container


def _unwrap_generic_json(container: Any) -> Any:
    """Promote nested list/channel wrappers exposing inner {"json": {...}} payloads to root.

    Patterns addressed:
      {"main": [[[{"json": {...}}]]]} -> {...}
      Mixed multi-json -> list of json dicts (deduped)
    Conservative traversal limits to avoid pathological shapes.
    """
    try:
        if not isinstance(container, dict):
            return container
        collected: List[Dict[str, Any]] = []

        def _walk(o: Any, depth: int = 0):
            if depth > 6 or len(collected) >= 50:
                return
            if isinstance(o, dict):
                if "json" in o and isinstance(o.get("json"), dict):
                    collected.append(o["json"])  # type: ignore[arg-type]
                else:
                    for v in o.values():
                        _walk(v, depth + 1)
            elif isinstance(o, list):
                for item in o[:100]:
                    _walk(item, depth + 1)

        _walk(container)
        if not collected:
            return container
        if len(collected) == 1:
            return collected[0]
        import json
        seen = set()
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
    except Exception:
        return container


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
        after_ai = _unwrap_ai_channel(base)
        if after_ai is not base:
            flags["unwrapped_ai_channel"] = True
        after_json = _unwrap_generic_json(after_ai)
        if after_json is not after_ai:
            flags["unwrapped_json_root"] = True
        return after_json, flags
    return base, flags


_GENERATION_PROVIDER_MARKERS = [
    # Core providers
    "openai", "anthropic", "gemini", "mistral", "groq",
    # Chat wrappers / legacy internal labels
    "lmchat", "lmopenai",
    # Newly added / upstream supported vendors
    "cohere", "deepseek", "ollama", "openrouter", "bedrock", "vertex", "huggingface", "xai",
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
    if node_run.data and "tokenUsage" in node_run.data:
        return True
    # Nested search (common n8n shape: {"ai_languageModel": [[{"json": {"tokenUsage": {...}}}]]})
    if _find_nested_key(node_run.data, "tokenUsage") is not None:
        return True
    lowered = node_type.lower()
    if any(excl in lowered for excl in ("embedding", "embeddings", "reranker")):
        return False
    return any(marker in lowered for marker in _GENERATION_PROVIDER_MARKERS)


def _extract_usage(node_run: NodeRun) -> Optional[LangfuseUsage]:
    """Extract and normalize token usage data from a node run.

    This function searches for a `tokenUsage` object within the `node_run.data`
    and normalizes various key formats (e.g., `promptTokens`, `completion`)
    into a canonical `LangfuseUsage` object with `input`, `output`, and `total`
    fields. It can also synthesize `total` if `input` and `output` are present.

    Args:
        node_run: The node run data to process.

    Returns:
        A `LangfuseUsage` object if token usage data is found, otherwise None.
    """
    tu = node_run.data.get("tokenUsage") if node_run.data else None
    if tu is None and node_run.data:
        container = _find_nested_key(node_run.data, "tokenUsage")
        if container and isinstance(container.get("tokenUsage"), dict):  # type: ignore[arg-type]
            tu = container["tokenUsage"]
    if not isinstance(tu, dict):
        return None
    # Gather candidates
    input_val = tu.get("input")
    output_val = tu.get("output")
    total_val = tu.get("total")
    # Legacy verbose keys
    p_tokens = tu.get("promptTokens")
    c_tokens = tu.get("completionTokens")
    t_tokens = tu.get("totalTokens")
    # Alternate short legacy keys
    p_short = tu.get("prompt")
    c_short = tu.get("completion")
    t_short = tu.get("total")  # already captured

    # Reconcile precedence (normalized first)
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
                total_val = int(input_val) + int(output_val)  # type: ignore[arg-type]
            except Exception:
                pass
    if input_val is None and output_val is None and total_val is None:
        return None
    return LangfuseUsage(input=input_val, output=output_val, total=total_val)


def _find_nested_key(data: Any, target: str, max_depth: int = 6) -> Optional[Dict[str, Any]]:
    """Perform a depth-limited search for a dictionary containing a specific key.

    This helper traverses a nested structure of dictionaries and lists to find
    the first dictionary that has `target` as a key. It is used to locate
    data like `tokenUsage` or `model` which may be deeply nested.

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
                return data  # type: ignore[return-value]
            for v in data.values():
                found = _find_nested_key(v, target, max_depth - 1)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for item in data[:25]:  # guard large lists
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
                if isinstance(chan_key, str) and chan_key.startswith("ai_") and isinstance(chan_val, list):
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
            direct = _direct_scan(j)  # type: ignore[arg-type]
            if direct:
                return direct

    # 4. Prepare BFS queue; include parsed JSON string expansions on-demand
    queue: Deque[Tuple[Any, int]] = deque([(data, 0)])
    visited = 0
    max_nodes = 800  # slightly higher to account for json-string expansions
    max_depth = 8
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
                    for item in current[:60]:  # cap list breadth
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


def map_execution_to_langfuse(record: N8nExecutionRecord, truncate_limit: Optional[int] = 4000) -> LangfuseTrace:
    """Map a single n8n execution record to a Langfuse trace object.

    This is the main transformation function. It orchestrates the conversion of
    an `N8nExecutionRecord` into a `LangfuseTrace` containing a root span and a
    span for each node run.

    The process involves:
    1.  Creating a trace and a root span for the entire execution.
    2.  Flattening all node runs and sorting them chronologically by start time.
    3.  Iterating through each node run to create a corresponding `LangfuseSpan`.
    4.  For each span, resolving its parent, processing its I/O, detecting if it's
        a generation, and extracting relevant metadata.
    5.  Building the final `LangfuseTrace` object containing all spans.

    Args:
        record: The n8n execution record from the database.
        truncate_limit: The character limit for serializing I/O fields.
            A value of 0 or None disables truncation.

    Returns:
        A `LangfuseTrace` object ready for export.
    """
    # Trace id is now exactly the n8n execution id (string). Simplicity & direct searchability in Langfuse UI.
    # NOTE: This shipper supports ONE n8n instance per Langfuse project. If you aggregate multiple instances
    # into a single Langfuse project, raw numeric execution ids may collide. Use separate Langfuse projects per instance.
    trace_id = str(record.id)
    # Normalize potentially naive datetimes to UTC (defensive: DB drivers / external inputs may yield naive objects)
    started_at = record.startedAt if record.startedAt.tzinfo is not None else record.startedAt.replace(tzinfo=timezone.utc)
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
            # executionId removed to avoid duplication; use root span metadata key n8n.execution.id instead
        },
    )

    # Build lookup of workflow node -> (type, category)
    wf_node_lookup: Dict[str, Dict[str, Optional[str]]] = {
        n.name: {"type": n.type, "category": n.category} for n in record.workflowData.nodes
    }

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

    # Track last span id per node for fallback parent resolution (runtime execution order)
    last_span_for_node: Dict[str, str] = {}

    # Build a reverse graph from workflow connections to infer predecessors when runtime source info is absent.
    reverse_edges: Dict[str, List[str]] = {}
    try:
        connections = getattr(record.workflowData, "connections", {}) or {}
        for from_node, conn_types in connections.items():
            if not isinstance(conn_types, dict):
                continue
            for _conn_type, outputs in conn_types.items():
                if not isinstance(outputs, list):
                    continue
                for output_list in outputs:  # each output_list is a list of connection dicts
                    if not isinstance(output_list, list):
                        continue
                    for edge in output_list:
                        to_node = edge.get("node") if isinstance(edge, dict) else None
                        if to_node:
                            reverse_edges.setdefault(to_node, []).append(from_node)
    except Exception:
        # Fail silently; graph fallback is best-effort
        pass

    run_data = record.data.executionData.resultData.runData

    # Build child->agent mapping for AI hierarchy connections.
    # Pattern-based: ANY connection type starting with 'ai_' (and not equal to literal 'main')
    # is treated as hierarchical: from_node (child) -> edge.node (agent/parent).
    # This future‑proofs against new n8n AI surface additions (ai_outputParser, ai_retriever, etc.).
    child_agent_map: Dict[str, Tuple[str, str]] = {}  # child -> (agent, connection_type)
    try:
        connections = getattr(record.workflowData, "connections", {}) or {}
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
                                child_agent_map[from_node] = (agent, conn_type)
    except Exception:
        pass

    # Flatten all runs to process in chronological order, giving agent spans chance to be created before children
    flattened: List[Tuple[int, str, int, NodeRun]] = []
    for node_name, runs in run_data.items():
        for idx, run in enumerate(runs):
            flattened.append((run.startTime, node_name, idx, run))
    flattened.sort(key=lambda x: x[0])

    # Maintain last output object (raw dict) per node for input propagation.
    last_output_data: Dict[str, Any] = {}
    for start_ts_raw, node_name, idx, run in flattened:
        wf_meta = wf_node_lookup.get(node_name, {})
        node_type = wf_meta.get("type") or node_name
        category = wf_meta.get("category")
        obs_type_guess = map_node_to_observation_type(node_type, category)
        span_id = str(uuid5(SPAN_NAMESPACE, f"{trace_id}:{node_name}:{idx}"))
        start_time = _epoch_ms_to_dt(run.startTime)
        end_time = start_time + timedelta(milliseconds=run.executionTime or 0)
        parent_id: Optional[str] = root_span_id
        prev_node = None
        prev_node_run = None

        # Hierarchical agent first
        if node_name in child_agent_map:
            agent_name, link_type = child_agent_map[node_name]
            if agent_name in last_span_for_node:
                parent_id = last_span_for_node[agent_name]
            else:
                parent_id = root_span_id  # fallback until agent processed
            prev_node = agent_name  # treat agent as previous for input inference
        else:
            # Sequential runtime or graph fallback
            if run.source and len(run.source) > 0 and run.source[0].previousNode:
                prev_node = run.source[0].previousNode
                prev_node_run = run.source[0].previousNodeRun
                if prev_node_run is not None:
                    parent_id = str(
                        uuid5(SPAN_NAMESPACE, f"{trace_id}:{prev_node}:{prev_node_run}")
                    )
                else:
                    parent_id = last_span_for_node.get(prev_node, root_span_id)
            elif node_name in reverse_edges and reverse_edges[node_name]:
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

        usage = _extract_usage(run)
        is_generation = _detect_generation(node_type, run)
        observation_type = obs_type_guess or ("generation" if is_generation else "span")
        raw_input_obj: Any = None
        if run.inputOverride is not None:
            raw_input_obj = run.inputOverride
        else:
            if prev_node and prev_node in last_output_data:
                raw_input_obj = {"inferredFrom": prev_node, "data": last_output_data[prev_node]}

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
        # Normalize / unwrap noisy AI & generic json wrappers
        norm_input_obj, input_flags = _normalize_node_io(raw_input_obj)
        norm_output_obj, output_flags = _normalize_node_io(raw_output_obj)
        if input_flags.get("unwrapped_ai_channel") or output_flags.get("unwrapped_ai_channel"):
            metadata["n8n.io.unwrapped_ai_channel"] = True
        if input_flags.get("unwrapped_json_root") or output_flags.get("unwrapped_json_root"):
            metadata["n8n.io.unwrapped_json_root"] = True
        input_str, input_trunc = _serialize_and_truncate(norm_input_obj, truncate_limit)
        output_str, output_trunc = _serialize_and_truncate(norm_output_obj, truncate_limit)
        if node_name in child_agent_map:
            agent_name, link_type = child_agent_map[node_name]
            metadata["n8n.agent.parent"] = agent_name
            metadata["n8n.agent.link_type"] = link_type
        if prev_node:
            metadata["n8n.node.previous_node"] = prev_node
        if prev_node_run is not None:
            metadata["n8n.node.previous_node_run"] = prev_node_run
        elif prev_node and node_name in reverse_edges and node_name not in child_agent_map:
            metadata["n8n.graph.inferred_parent"] = True
        if input_trunc:
            metadata["n8n.truncated.input"] = True
        if output_trunc:
            metadata["n8n.truncated.output"] = True
        # Model extraction: BFS variant key search (model, model_name, modelId, model_id)
        model_val = None
        # Build a richer search root: primary run.data plus supplemental contexts that may hold the embedded model.
        if isinstance(run.data, dict):
            search_root: Dict[str, Any] = dict(run.data)
            if isinstance(run.inputOverride, dict):
                search_root["_inputOverride"] = run.inputOverride
            # If we inferred parent output as input, include it (raw_input_obj already built above)
            if raw_input_obj and isinstance(raw_input_obj, dict):
                search_root["_inferredInput"] = raw_input_obj
            top_model = run.data.get("model")
            if isinstance(top_model, str) and top_model:
                model_val = top_model
            else:
                model_val = _extract_model_value(search_root)
        if model_val is None and is_generation:
            # Attach diagnostic metadata to help identify why extraction failed (list a few top-level keys only)
            metadata["n8n.model.missing"] = True
            try:
                if isinstance(run.data, dict):
                    metadata["n8n.model.search_keys"] = list(run.data.keys())[:12]
            except Exception:
                pass
        span = LangfuseSpan(
            id=span_id,
            trace_id=trace_id,
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
        last_span_for_node[node_name] = span_id
        try:
            size_guard_ok = True
            # Only enforce size guard when truncation is active (>0). When disabled (<=0 or None), always cache.
            if truncate_limit is not None and isinstance(truncate_limit, int) and truncate_limit > 0:
                size_guard_ok = len(str(raw_output_obj)) < truncate_limit * 2
            if isinstance(raw_output_obj, dict) and size_guard_ok:
                last_output_data[node_name] = raw_output_data = raw_output_obj
        except Exception:
            pass
        # No separate generations list; generation classification lives on the span itself.
    return trace


__all__ = ["map_execution_to_langfuse"]
