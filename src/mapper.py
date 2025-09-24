"""Core mapping logic from n8n execution records to Langfuse internal models.

Enhancements (Iteration 3):
 - Structured JSON serialization for input/output with truncation flags.
 - Normalized status + error propagation.
 - Improved parent resolution using `previousNodeRun` when present.
 - Richer metadata fields to aid downstream analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
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
    # Some n8n exports may already be seconds; if value looks like seconds (10 digits) treat accordingly
    # Heuristic: if ms < 10^12 assume seconds.
    if ms < 1_000_000_000_000:  # seconds
        return datetime.fromtimestamp(ms, tz=timezone.utc)
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{200,}={0,2}$")  # long pure base64 chunk


def _likely_binary_b64(s: str, *, context_key: str | None = None, in_binary_block: bool = False) -> bool:
    """Heuristic to decide if a long string is really base64 (binary) vs just long text.

    Prior implementation flagged *any* long base64-like string (including 'x'*500) which caused
    benign large textual fields to be stripped as binary, preventing truncation flags from firing
    (test expectation). We now require a minimum diversity of characters to reduce false positives.
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
    """Detect probable binary base64 blobs.
    Signals true when:
      - Standard n8n binary structure with mimeType + long data field.
      - Any string value looks like a large base64 block (>200 chars, matches regex) or starts with /9j/ (jpeg magic in base64).
    Depth limited to avoid excessive recursion.
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
    """Replace nested binary sections & large base64 strings with placeholder references.
    Preserves structure but inserts marker objects: {"_binary": true, "note": "omitted", ...meta}
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
    """Serialize obj to compact JSON (fallback to str) and truncate if needed.

    If limit is None, truncation is disabled (except binary sections replaced by placeholder).
    Returns (serialized_value_or_None, truncated_flag).
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


_GENERATION_PROVIDER_MARKERS = [
    # Core providers
    "openai", "anthropic", "gemini", "mistral", "groq",
    # Chat wrappers / legacy internal labels
    "lmchat", "lmopenai",
    # Newly added / upstream supported vendors
    "cohere", "deepseek", "ollama", "openrouter", "bedrock", "vertex", "huggingface", "xai",
]


def _detect_generation(node_type: str, node_run: NodeRun) -> bool:
    """Decide whether a node should be classified as a generation.

    Rules (ordered):
      1. Explicit: presence of tokenUsage object in run.data.
      2. Heuristic: node_type contains any provider marker in _GENERATION_PROVIDER_MARKERS (case-insensitive)
         while NOT containing 'embedding' / 'embeddings' / 'reranker' (to avoid misclassifying embeddings/rerankers).
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
    """Normalize various tokenUsage shapes to canonical input/output/total.

    Accepted legacy keys inside run.data.tokenUsage:
      - promptTokens / completionTokens / totalTokens
      - prompt / completion / total
      - input / output / total (already normalized)
    Synthesis:
      - If total missing but input & output present -> total = input + output
      - If only legacy prompt/completion present -> map directly.
    No negative or contradictory reconciliation; if conflicting representations differ,
    preference order: input/output/total > promptTokens/completionTokens/totalTokens > prompt/completion/total.
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
    """Depth-limited search returning the first dict containing target as a key.

    Supports typical n8n execution output nesting patterns: channel -> list[ list[ { json: {...} } ] ]
    We purposely stop after encountering the first match to avoid large traversals.
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


def map_execution_to_langfuse(record: N8nExecutionRecord, truncate_limit: Optional[int] = 4000) -> LangfuseTrace:
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
        input_str, input_trunc = _serialize_and_truncate(raw_input_obj, truncate_limit)
        output_str, output_trunc = _serialize_and_truncate(raw_output_obj, truncate_limit)
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
            model=run.data.get("model") if isinstance(run.data, dict) else None,
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
