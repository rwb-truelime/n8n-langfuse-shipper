from datetime import datetime, timezone

from src.mapper import map_execution_to_langfuse, map_execution_with_assets
from src.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)


def _sample_record():
    rd = {
        "MyNode": [
            NodeRun(
                startTime=1710000000000,
                executionTime=10,
                executionStatus="success",
                data={"value": 1},
                source=None,
                inputOverride=None,
                error=None,
            )
        ]
    }
    return N8nExecutionRecord(
        id=123,
        workflowId="wf1",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wf1",
            name="wf",
            nodes=[WorkflowNode(name="MyNode", type="test.node")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData=rd))
        ),
    )


def test_media_disabled_equivalence():
    rec = _sample_record()
    trace_plain = map_execution_to_langfuse(rec)
    mapped_assets = map_execution_with_assets(rec, collect_binaries=False)
    assert mapped_assets.assets == []
    assert len(trace_plain.spans) == len(mapped_assets.trace.spans)
    assert trace_plain.spans[0].id == mapped_assets.trace.spans[0].id
