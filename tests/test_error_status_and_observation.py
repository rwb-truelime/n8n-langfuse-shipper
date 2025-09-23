from __future__ import annotations

from datetime import datetime, timezone
from src.mapper import map_execution_to_langfuse
from src.models.n8n import (
    N8nExecutionRecord,
    WorkflowData,
    WorkflowNode,
    ExecutionData,
    ExecutionDataDetails,
    ResultData,
    NodeRun,
)


def test_error_status_normalization():
    now = datetime.now(timezone.utc)
    runData = {
        "ErrNode": [
            NodeRun(
                startTime=int(now.timestamp() * 1000),
                executionTime=5,
                executionStatus="success",  # original status
                data={"value": 1},
                error={"message": "boom"},
            )
        ]
    }
    rec = N8nExecutionRecord(
        id=700,
        workflowId="wf-error",
        status="failed",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(id="wf-error", name="Error WF", nodes=[WorkflowNode(name="ErrNode", type="ToolWorkflow")]),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "ErrNode")
    assert span.status == "error", "Error status not normalized"
    assert span.error is not None

