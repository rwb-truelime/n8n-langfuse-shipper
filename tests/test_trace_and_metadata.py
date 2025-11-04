from __future__ import annotations

from datetime import datetime, timezone

from n8n_langfuse_shipper.mapper import map_execution_to_langfuse
from n8n_langfuse_shipper.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)


def _simple_record(exec_id: int = 999):
    now = datetime.now(timezone.utc)
    return N8nExecutionRecord(
        id=exec_id,
        workflowId="wf-meta",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-meta",
            name="Meta WF",
            nodes=[WorkflowNode(name="Only", type="ToolWorkflow")],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "Only": [
                            NodeRun(
                                startTime=int(now.timestamp() * 1000),
                                executionTime=1,
                                executionStatus="success",
                                data={"ok": True},
                            )
                        ]
                    }
                )
            )
        ),
    )


def test_root_metadata_contains_execution_id_only():
    rec = _simple_record(5001)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    root = trace.spans[0]
    assert root.metadata.get("n8n.execution.id") == 5001
    # Ensure it is not duplicated in trace.metadata
    assert "n8n.execution.id" not in trace.metadata


def test_deterministic_across_truncation_modes():
    rec = _simple_record(5002)
    trace_a = map_execution_to_langfuse(rec, truncate_limit=None)
    trace_b = map_execution_to_langfuse(rec, truncate_limit=50)
    ids_a = [s.id for s in trace_a.spans]
    ids_b = [s.id for s in trace_b.spans]
    assert ids_a == ids_b, "Span IDs changed between truncation settings"
