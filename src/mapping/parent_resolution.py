"""Parent resolution utilities extracted from mapper.

Implements precedence: agent hierarchy -> runtime exact -> runtime last -> static reverse graph -> root.
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


def build_child_agent_map(workflow_data: Any) -> Dict[str, Tuple[str, str]]:
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
    run: Any,  # NodeRun-like
    trace_id: str,
    child_agent_map: Dict[str, Tuple[str, str]],
    last_span_for_node: Dict[str, str],
    reverse_edges: Dict[str, List[str]],
    root_span_id: str,
) -> Tuple[str, str | None, int | None]:
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
