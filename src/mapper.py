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
from uuid import uuid5, UUID, NAMESPACE_DNS

from .models.n8n import N8nExecutionRecord, NodeRun
from .models.langfuse import (
    LangfuseTrace,
    LangfuseSpan,
    LangfuseGeneration,
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


def _contains_binary_marker(d: Any) -> bool:
    """Heuristic: detect presence of binary base64 data in typical n8n node I/O structures.
    Looks for nested dict path containing keys like 'binary' -> node -> data -> 'data' (base64) with mimeType.
    """
    try:
        if isinstance(d, dict):
            if "binary" in d and isinstance(d["binary"], dict):
                # If any nested value under binary has a dict containing 'data' and mimeType
                for v in d["binary"].values():
                    if isinstance(v, dict):
                        data_section = v.get("data") if isinstance(v.get("data"), dict) else v
                        if isinstance(data_section, dict):
                            # Look for mimeType and long 'data' field
                            mime = data_section.get("mimeType") or data_section.get("mime_type")
                            b64 = data_section.get("data")
                            if mime and isinstance(b64, str) and len(b64) > 256:
                                return True
            # Recurse a limited depth
            for val in list(d.values())[:10]:
                if _contains_binary_marker(val):
                    return True
        elif isinstance(d, list):
            for item in d[:10]:  # limit breadth
                if _contains_binary_marker(item):
                    return True
    except Exception:
        return False
    return False


def _strip_binary_payload(obj: Any) -> Any:
    """Replace large binary base64 payload sections with a placeholder string.
    Returns a shallow-copied structure with replacements.
    """
    placeholder = "Binary data omitted"
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
                            if isinstance(data_section, dict) and isinstance(data_section.get("data"), str):
                                ds = data_section.copy()
                                if len(ds.get("data", "")) > 64:  # assume large
                                    ds["data"] = placeholder
                                    bv2["data"] = ds
                            new_bin[bk] = bv2
                        else:
                            new_bin[bk] = bv
                    new_d[k] = new_bin
                else:
                    new_d[k] = _strip_binary_payload(v)
            return new_d
        if isinstance(obj, list):
            return [_strip_binary_payload(x) for x in obj]
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


def _detect_generation(node_type: str, node_run: NodeRun) -> bool:
    if node_run.data and "tokenUsage" in node_run.data:
        return True
    # Heuristic on type
    llm_markers = ["OpenAi", "Anthropic", "Gemini", "Mistral", "Groq", "LmChat", "LmOpenAi"]
    return any(marker.lower() in node_type.lower() for marker in llm_markers)


def _extract_usage(node_run: NodeRun) -> Optional[LangfuseUsage]:
    tu = node_run.data.get("tokenUsage") if node_run.data else None
    if not isinstance(tu, dict):
        return None
    return LangfuseUsage(
        promptTokens=tu.get("promptTokens"),
        completionTokens=tu.get("completionTokens"),
        totalTokens=tu.get("totalTokens"),
    )


def map_execution_to_langfuse(record: N8nExecutionRecord, truncate_limit: Optional[int] = 4000) -> LangfuseTrace:
    # Keep deterministic trace id for idempotency; retain existing pattern to avoid breaking historical IDs.
    trace_id = f"n8n-exec-{record.id}"
    # New naming convention: trace name is just the workflow name (fallback 'execution').
    base_name = record.workflowData.name or "execution"
    trace = LangfuseTrace(
        id=trace_id,
        name=base_name,
        timestamp=record.startedAt,
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
        start_time=record.startedAt,
        end_time=record.stoppedAt or (record.startedAt + timedelta(milliseconds=1)),
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

    # Build child->agent mapping for special AI hierarchy connections
    child_agent_map: Dict[str, Tuple[str, str]] = {}  # child -> (agent, connection_type)
    special_conn_types = {"ai_tool", "ai_languageModel", "ai_memory"}
    try:
        connections = getattr(record.workflowData, "connections", {}) or {}
        for from_node, conn_types in connections.items():
            if not isinstance(conn_types, dict):
                continue
            for conn_type, outputs in conn_types.items():
                if conn_type not in special_conn_types:
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
                                # from_node is the tool/model/memory, agent is parent
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
            token_usage=usage,
            status=status_norm,
        )
        trace.spans.append(span)
        last_span_for_node[node_name] = span_id
        try:
            size_guard_ok = True
            if truncate_limit is not None and isinstance(truncate_limit, int):
                size_guard_ok = len(str(raw_output_obj)) < truncate_limit * 2
            if isinstance(raw_output_obj, dict) and size_guard_ok:
                last_output_data[node_name] = raw_output_obj
        except Exception:
            pass
        if is_generation:
            trace.generations.append(
                LangfuseGeneration(
                    span_id=span_id,
                    model=span.model,
                    usage=usage,
                    input=span.input,
                    output=span.output,
                )
            )
    return trace


__all__ = ["map_execution_to_langfuse"]
