"""Parent span resolution with fixed precedence order.

This module implements the authoritative parent-child relationship logic for spans,
applying a strict 5-tier precedence strategy to determine each span's parent:

1. Agent Hierarchy: Non-main AI connections (ai_tool, ai_languageModel, etc.)
   make the agent span the parent of the component span
2. Runtime Sequential (Exact Run): previousNode + previousNodeRun points to
   specific run index
3. Runtime Sequential (Last Seen): previousNode without run index links to
   last emitted span for that node
4. Static Reverse Graph Fallback: Workflow connections infer likely parent when
   runtime data absent
5. Root Fallback: No parent determined → link to root execution span

Functions construct supporting data structures (reverse edges, agent map) and apply
the resolution strategy during mapping iteration. Metadata signals document which
tier resolved the relationship for downstream analytics.

Public Functions:
    build_reverse_edges: Parse workflow connections into child→parents map
    build_child_agent_map: Extract AI hierarchy from non-main ai_* connections
    resolve_parent: Apply precedence rules to determine parent span ID

Design Note:
    Changing precedence order or adding tiers requires updates to tests, README,
    and .github/copilot-instructions.md in the same PR.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
from uuid import uuid5

from .id_utils import SPAN_NAMESPACE

logger = logging.getLogger(__name__)

__all__ = [
    "build_reverse_edges",
    "build_child_agent_map",
    "resolve_parent",
]


def build_reverse_edges(workflow_data: Any) -> Dict[str, List[str]]:
    """Build child→parents mapping from workflow connections for static graph fallback.

    Parses workflow connections to create reverse edge map enabling parent inference
    when runtime source data absent. Only processes "main" type connections; non-main
    AI connections handled separately by agent hierarchy.

    Args:
        workflow_data: Workflow data object containing connections dict

    Returns:
        Dict mapping child node name to list of parent node names

    Note:
        Fail-open on parse errors; returns empty dict if connections malformed.
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


def build_child_agent_map(
    workflow_data: Any
) -> Dict[str, Tuple[str, str]]:
    """Extract AI agent hierarchy from non-main ai_* connections.

    Identifies child-agent relationships where non-main connections with type prefix
    "ai_" (e.g., ai_tool, ai_languageModel, ai_memory) indicate hierarchical parent.
    Agent spans become parents of their component/tool spans in precedence tier 1.

    Args:
        workflow_data: Workflow data object containing connections dict

    Returns:
        Dict mapping child node name to tuple of (agent_node_name, connection_type)

    Note:
        Only first agent link per child preserved when multiple exist. Fail-open on
        parse errors returns empty dict.
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


def resolve_parent(
    *,
    node_name: str,
    run: Any,
    trace_id: str,
    child_agent_map: Dict[str, Tuple[str, str]],
    last_span_for_node: Dict[str, str],
    reverse_edges: Dict[str, List[str]],
    root_span_id: str,
) -> Tuple[str, str | None, int | None]:
    """Determine parent span ID using fixed 5-tier precedence strategy.

    Applies authoritative precedence order:
    1. Agent Hierarchy: Check child_agent_map for AI relationship
    2. Runtime Exact: run.source previousNode + previousNodeRun → exact span
    3. Runtime Last: run.source previousNode only → last span for that node
    4. Static Graph: reverse_edges suggests likely parent from workflow connections
    5. Root Fallback: No parent determined → root span

    Args:
        node_name: Current node being processed
        run: NodeRun with optional source chain data
        trace_id: Trace ID for deterministic span ID construction
        child_agent_map: AI hierarchy map from build_child_agent_map
        last_span_for_node: Mapping of node name to most recent span ID
        reverse_edges: Static reverse graph from build_reverse_edges
        root_span_id: Root span ID for fallback

    Returns:
        Tuple of (parent_span_id, prev_node, prev_node_run)
        - parent_span_id: Resolved parent span ID
        - prev_node: Previous node name if determined (for metadata)
        - prev_node_run: Previous run index if exact match (for metadata)

    Note:
        Changing precedence order breaks test contracts; update docs + tests together.
    """
    parent_id: str = root_span_id
    prev_node: str | None = None
    prev_node_run: int | None = None
    if node_name in child_agent_map:
        agent_name, _link_type = child_agent_map[node_name]
        parent_id = last_span_for_node.get(agent_name, root_span_id)
        prev_node = agent_name
        return parent_id, prev_node, prev_node_run
    if (
        getattr(run, "source", None)
        and len(run.source) > 0
        and run.source[0].previousNode
    ):
        prev_node = run.source[0].previousNode
        prev_node_run = run.source[0].previousNodeRun
        if prev_node_run is not None:
            parent_id = str(
                uuid5(
                    SPAN_NAMESPACE,
                    f"{trace_id}:{prev_node}:{prev_node_run}",
                )
            )
        elif prev_node is not None:
            parent_id = last_span_for_node.get(prev_node, root_span_id)
        return parent_id, prev_node, prev_node_run
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
        if chosen_pred is not None:
            parent_id = last_span_for_node.get(chosen_pred, parent_id)
        return parent_id, prev_node, prev_node_run
    return parent_id, prev_node, prev_node_run
