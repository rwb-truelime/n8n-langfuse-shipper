"""Core mapping logic from n8n execution records to Langfuse internal models.

Enhancements (Iteration 3):
 - Structured JSON serialization for input/output with truncation flags.
 - Normalized status + error propagation.
 - Improved parent resolution using `previousNodeRun` when present.
 - Richer metadata fields to aid downstream analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
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


def _serialize_and_truncate(obj: Any, limit: int) -> Tuple[Optional[str], bool]:
    """Serialize obj to compact JSON (fallback to str) and truncate if needed.

    Returns (serialized_value_or_None, truncated_flag).
    """
    if obj is None:
        return None, False
    try:
        import json

        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        raw = str(obj)
    truncated = False
    if len(raw) > limit:
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


def map_execution_to_langfuse(record: N8nExecutionRecord, truncate_limit: int = 4000) -> LangfuseTrace:
    trace_id = f"n8n-exec-{record.id}"
    trace = LangfuseTrace(
        id=trace_id,
        name=record.workflowData.name,
        timestamp=record.startedAt,
        metadata={
            "workflowId": record.workflowId,
            "status": record.status,
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
        name=record.workflowData.name or f"execution-{record.id}",
        start_time=record.startedAt,
        end_time=record.stoppedAt or (record.startedAt + timedelta(milliseconds=1)),
        observation_type="span",
        metadata={"n8n.execution.id": record.id},
    )
    trace.spans.append(root_span)

    # Track last span id per node for fallback parent resolution (runtime execution order)
    last_span_for_node: Dict[str, str] = {}

    # Build connection helper structures:
    #  - reverse_edges: existing predecessor inference (execution flow / main connections)
    #  - agent_parent_for_node: mapping child (model/tool/memory) -> agent node name for ai_* edges
    reverse_edges: Dict[str, List[str]] = {}
    agent_parent_for_node: Dict[str, str] = {}
    agent_nodes: set[str] = set(
        n.name for n in record.workflowData.nodes if (n.type or "").lower().endswith(".agent") or "agent" in (n.type or "").lower()
    )
    try:
        connections = getattr(record.workflowData, "connections", {}) or {}
        for from_node, conn_types in connections.items():
            if not isinstance(conn_types, dict):
                continue
            for conn_type, outputs in conn_types.items():
                if not isinstance(outputs, list):
                    continue
                for output_list in outputs:  # each output_list is a list of edge dicts
                    if not isinstance(output_list, list):
                        continue
                    for edge in output_list:
                        to_node = edge.get("node") if isinstance(edge, dict) else None
                        if not to_node:
                            continue
                        # Standard predecessor capture (for fallback parent inference)
                        reverse_edges.setdefault(to_node, []).append(from_node)
                        # Agent cluster parenting: if connection type is ai_* and target is an agent node,
                        # treat the agent as logical parent of the from_node.
                        if conn_type.startswith("ai_") and to_node in agent_nodes:
                            agent_parent_for_node[from_node] = to_node
    except Exception:
        # Fail silently; graph + agent parenting is best-effort
        pass

    run_data = record.data.executionData.resultData.runData

    # Maintain last output object (raw dict) per node for input propagation.
    last_output_data: Dict[str, Any] = {}
    for node_name, runs in run_data.items():
        wf_meta = wf_node_lookup.get(node_name, {})
        node_type = wf_meta.get("type") or node_name
        category = wf_meta.get("category")
        obs_type_guess = map_node_to_observation_type(node_type, category)
        for idx, run in enumerate(runs):
            span_id = str(uuid5(SPAN_NAMESPACE, f"{trace_id}:{node_name}:{idx}"))
            start_time = _epoch_ms_to_dt(run.startTime)
            end_time = start_time + timedelta(milliseconds=run.executionTime or 0)
            parent_id: Optional[str] = root_span_id
            # Agent cluster override: if this node is wired via ai_* connection to an agent node
            # then parent directly to that agent's first run span (deterministic id) unless a more
            # precise runtime previousNodeRun explicitly points elsewhere.
            if node_name in agent_parent_for_node:
                agent_name = agent_parent_for_node[node_name]
                agent_span_id = str(uuid5(SPAN_NAMESPACE, f"{trace_id}:{agent_name}:0"))
                parent_id = agent_span_id
            prev_node = None
            prev_node_run = None
            if run.source and len(run.source) > 0 and run.source[0].previousNode:
                prev_node = run.source[0].previousNode
                prev_node_run = run.source[0].previousNodeRun
                if prev_node_run is not None:
                    parent_id = str(uuid5(SPAN_NAMESPACE, f"{trace_id}:{prev_node}:{prev_node_run}"))
                else:
                    parent_id = last_span_for_node.get(prev_node, root_span_id)
            elif node_name in reverse_edges and reverse_edges[node_name]:
                # Choose the last seen predecessor among graph parents if any have executed; else first static predecessor.
                graph_preds = reverse_edges[node_name]
                # Prefer one that already has a span to maintain execution chain feel.
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
            # Observation type selection with memory override preference -> retriever per user guidance.
            if obs_type_guess:
                observation_type = obs_type_guess
            else:
                observation_type = "generation" if is_generation else "span"
            if "memory" in (node_type or "").lower():
                observation_type = "retriever"
            # Determine logical input: prefer explicit inputOverride, else prior parent's raw output if available.
            raw_input_obj: Any = None
            if run.inputOverride is not None:
                raw_input_obj = run.inputOverride
            else:
                # Use parent node output if exists
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
            if node_name in agent_parent_for_node:
                metadata["n8n.agent.child"] = True
                metadata["n8n.agent.parent"] = agent_parent_for_node[node_name]
            if prev_node:
                metadata["n8n.node.previous_node"] = prev_node
            if prev_node_run is not None:
                metadata["n8n.node.previous_node_run"] = prev_node_run
            elif prev_node and node_name in reverse_edges:
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
            # Store raw output for possible downstream propagation (limit size in memory by simple heuristic)
            try:
                if isinstance(raw_output_obj, dict) and len(str(raw_output_obj)) < truncate_limit * 2:
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
