"""Pydantic models for representing raw n8n execution data.

These models provide a typed, validated structure for the JSON data retrieved
from the `n8n_execution_entity` and `n8n_execution_data` tables in the n8n
PostgreSQL database. They are used to safely access nested fields during the
transformation process in the `mapper` module.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NodeRunSource(BaseModel):
    """Represents the source of a node run, indicating its predecessor."""

    previousNode: Optional[str] = None
    previousNodeRun: Optional[int] = None


class NodeRun(BaseModel):
    """Represents a single runtime execution of a node within a workflow."""

    startTime: int
    executionTime: int
    executionStatus: str
    data: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[List[NodeRunSource]] = None
    inputOverride: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class ResultData(BaseModel):
    """Contains the `runData`, which maps node names to their execution runs."""

    runData: Dict[str, List[NodeRun]] = Field(default_factory=dict)


class ExecutionDataDetails(BaseModel):
    """A nested container for the `ResultData`."""

    resultData: ResultData


class ExecutionData(BaseModel):
    """The top-level container for the detailed execution data from the `data` column."""

    executionData: ExecutionDataDetails


class WorkflowNode(BaseModel):
    """Represents a static node as defined in the workflow graph."""

    name: str
    type: str
    category: Optional[str] = None
    # Raw parameters dict as stored in workflow definition; used for model fallback extraction
    parameters: Optional[Dict[str, Any]] = None


class WorkflowData(BaseModel):
    """Represents the static definition of a workflow, including its nodes and connections."""

    id: str
    name: str
    nodes: List[WorkflowNode] = Field(default_factory=list)
    # Raw workflow connections graph. Structure (simplified):
    # {
    #   "NodeA": {
    #       "main": [  # list indexed by output index
    #           [ {"index": 0, "node": "NodeB", "type": "main"}, ...],  # connections from output 0
    #           [ ... ]  # output 1
    #       ],
    #       "ai_languageModel": [ ... ],
    #   },
    #   ...
    # }
    connections: Dict[str, Any] = Field(default_factory=dict)


class N8nExecutionRecord(BaseModel):
    """The primary model representing a complete n8n execution record.

    This model combines the metadata from the `execution_entity` table with the
    parsed `workflowData` and `data` JSON blobs into a single, validated
    structure.
    """

    id: int
    workflowId: str
    status: str
    startedAt: datetime
    stoppedAt: datetime
    workflowData: WorkflowData
    data: ExecutionData

    # Additional fields sometimes present in n8n execution rows can be added here later as needed.
