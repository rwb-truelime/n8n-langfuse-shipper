from __future__ import annotations

from datetime import datetime, timezone

from src.mapper import map_execution_to_langfuse
from src.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    NodeRunSource,
    ResultData,
    WorkflowData,
    WorkflowNode,
)


def _base(now, run_data):
    return N8nExecutionRecord(
        id=777,
        workflowId="wf-prop",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-prop",
            name="Propagation WF",
            nodes=[
                WorkflowNode(name="Parent", type="ToolWorkflow"),
                WorkflowNode(name="Child", type="ToolWorkflow"),
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData=run_data))
        ),
    )


def test_input_propagation_basic():
    now = datetime.now(timezone.utc)
    run_data = {
        "Parent": [
            NodeRun(
                startTime=int(now.timestamp() * 1000),
                executionTime=5,
                executionStatus="success",
                data={"value": 42},
            )
        ],
        "Child": [
            NodeRun(
                startTime=int(now.timestamp() * 1000) + 10,
                executionTime=5,
                executionStatus="success",
                data={"ok": True},
                source=[NodeRunSource(previousNode="Parent", previousNodeRun=0)],
            )
        ],
    }
    rec = _base(now, run_data)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    child_span = next(s for s in trace.spans if s.name == "Child")
    assert child_span.input is not None and "inferredFrom" in child_span.input


def test_input_propagation_size_guard_blocks_large_parent():
    now = datetime.now(timezone.utc)
    large_obj = {"data": "x" * 1000}
    run_data = {
        "Parent": [
            NodeRun(
                startTime=int(now.timestamp() * 1000),
                executionTime=5,
                executionStatus="success",
                data=large_obj,
            )
        ],
        "Child": [
            NodeRun(
                startTime=int(now.timestamp() * 1000) + 10,
                executionTime=5,
                executionStatus="success",
                data={"ok": True},
                source=[NodeRunSource(previousNode="Parent", previousNodeRun=0)],
            )
        ],
    }
    rec = _base(now, run_data)
    trace = map_execution_to_langfuse(rec, truncate_limit=100)  # activates size guard path
    child_span = next(s for s in trace.spans if s.name == "Child")
    # Input should be None or not contain inferredFrom due to size guard
    assert not child_span.input or "inferredFrom" not in child_span.input
