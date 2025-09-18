from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NodeRunSource(BaseModel):
    previousNode: Optional[str] = None
    previousNodeRun: Optional[int] = None


class NodeRun(BaseModel):
    startTime: int
    executionTime: int
    executionStatus: str
    data: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[List[NodeRunSource]] = None
    inputOverride: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class ResultData(BaseModel):
    runData: Dict[str, List[NodeRun]] = Field(default_factory=dict)


class ExecutionDataDetails(BaseModel):
    resultData: ResultData


class ExecutionData(BaseModel):
    executionData: ExecutionDataDetails


class WorkflowNode(BaseModel):
    name: str
    type: str
    category: Optional[str] = None


class WorkflowData(BaseModel):
    id: str
    name: str
    nodes: List[WorkflowNode] = Field(default_factory=list)


class N8nExecutionRecord(BaseModel):
    id: int
    workflowId: str
    status: str
    startedAt: datetime
    stoppedAt: datetime
    workflowData: WorkflowData
    data: ExecutionData

    # Additional fields sometimes present in n8n execution rows can be added here later as needed.
