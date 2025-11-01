"""Mapping iteration state container for orchestrator loop.

This module defines the MappingContext dataclass that holds mutable state during
the execution-to-trace mapping iteration. Using a dedicated context object improves
readability and testability by making state dependencies explicit.

State Fields:
    trace_id: Deterministic trace ID (str(execution.id))
    root_span_id: UUID of root execution span
    wf_node_lookup: Node name → {type, category} for quick metadata access
    wf_node_obj: Node name → WorkflowNode for parameter fallback
    reverse_edges: Static graph child→parents map
    child_agent_map: AI hierarchy child→(agent, link_type) map
    truncate_limit: Optional character limit for I/O truncation
    last_span_for_node: Mutable map of node name → most recent span ID
    last_output_data: Mutable cache of node name → raw output (for input propagation)

Usage Pattern:
    1. Orchestrator initializes context before node iteration
    2. Each iteration reads static fields and updates mutable caches
    3. Parent resolution and input propagation query cached state
    4. Size guard prevents unbounded cache growth when truncation enabled

Design Note:
    Context mutability isolated to orchestrator scope; helper functions remain pure
    by receiving required state as explicit parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models.n8n import WorkflowNode

__all__ = ["MappingContext"]


@dataclass
class MappingContext:
    trace_id: str
    root_span_id: str
    wf_node_lookup: Dict[str, Dict[str, Optional[str]]]
    wf_node_obj: Dict[str, WorkflowNode]
    reverse_edges: Dict[str, List[str]]
    child_agent_map: Dict[str, Tuple[str, str]]
    truncate_limit: Optional[int]
    last_span_for_node: Dict[str, str] = field(default_factory=dict)
    last_output_data: Dict[str, Any] = field(default_factory=dict)
