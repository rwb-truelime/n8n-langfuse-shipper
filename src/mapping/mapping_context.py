"""MappingContext dataclass extracted from mapper.

Holds mutable state during execution mapping.
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
