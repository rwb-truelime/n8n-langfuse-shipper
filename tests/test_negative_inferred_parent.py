from __future__ import annotations

from datetime import datetime, timezone

from n8n_langfuse_shipper.mapper import map_execution_to_langfuse
from n8n_langfuse_shipper.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    NodeRunSource,
    ResultData,
    WorkflowData,
    WorkflowNode,
)


def test_no_inferred_parent_when_runtime_source_present():
    now = datetime.now(timezone.utc)
    runData = {
        "A": [
            NodeRun(
                startTime=int(now.timestamp() * 1000),
                executionTime=5,
                executionStatus="success",
                data={},
            )
        ],
        "B": [
            NodeRun(
                startTime=int(now.timestamp() * 1000) + 10,
                executionTime=5,
                executionStatus="success",
                data={},
                source=[NodeRunSource(previousNode="A", previousNodeRun=0)],
            )
        ],
    }
    rec = N8nExecutionRecord(
        id=8800,
        workflowId="wf-inf",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-inf",
            name="Inferred Negative WF",
            nodes=[WorkflowNode(name="A", type="ToolWorkflow"), WorkflowNode(name="B", type="ToolWorkflow")],
        ),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    b_span = next(s for s in trace.spans if s.name == "B")
    assert "n8n.graph.inferred_parent" not in b_span.metadata
