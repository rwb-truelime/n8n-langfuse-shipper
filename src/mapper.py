"""Core mapping logic from n8n execution records to Langfuse internal models."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
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


def _truncate(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + f"...<truncated {len(value)-limit} chars>"


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

    # Track last span id per node for parent resolution
    last_span_for_node: Dict[str, str] = {}

    run_data = record.data.executionData.resultData.runData
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
            if run.source and run.source[0].previousNode:
                prev_node = run.source[0].previousNode
                parent_id = last_span_for_node.get(prev_node, root_span_id)

            usage = _extract_usage(run)
            is_generation = _detect_generation(node_type, run)
            observation_type = obs_type_guess or ("generation" if is_generation else "span")
            span = LangfuseSpan(
                id=span_id,
                trace_id=trace_id,
                parent_id=parent_id,
                name=node_name,
                start_time=start_time,
                end_time=end_time,
                observation_type=observation_type,
                input=_truncate(str(run.inputOverride) if run.inputOverride else None, truncate_limit),
                output=_truncate(str(run.data) if run.data else None, truncate_limit),
                metadata={
                    "n8n.node.type": node_type,
                    "n8n.node.category": category,
                },
                error=run.error,
                model=run.data.get("model") if isinstance(run.data, dict) else None,
                token_usage=usage,
            )
            trace.spans.append(span)
            last_span_for_node[node_name] = span_id
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
