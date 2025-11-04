from __future__ import annotations

from datetime import datetime, timezone

from n8n_langfuse_shipper.__main__ import _decode_compact_pointer_execution
from n8n_langfuse_shipper.mapper import map_execution_to_langfuse
from n8n_langfuse_shipper.models.n8n import (
    N8nExecutionRecord,
    WorkflowData,
    WorkflowNode,
)


def test_pointer_compact_decoding_basic():
    # pool layout (simplified):
    # 0 => root { resultData: <1> }
    # 1 => { runData: <2> }
    # 2 => { NodeA: <3>, NodeB: <4> }
    # 3 => list of run objects for NodeA
    # 4 => list of run objects for NodeB
    now_ms = 1_700_000_000
    pool = [
        {"resultData": "1"},
        {"runData": "2"},
        {"NodeA": "3", "NodeB": "4"},
        [
            {"startTime": now_ms, "executionTime": 10, "executionStatus": "success", "data": {"x":1}},
        ],
        [
            {"startTime": now_ms + 5, "executionTime": 5, "executionStatus": "success", "data": {"y":2}},
        ],
    ]
    decoded = _decode_compact_pointer_execution(pool, debug=True, execution_id=55)
    assert decoded is not None
    run_data = decoded.executionData.resultData.runData
    assert set(run_data.keys()) == {"NodeA", "NodeB"}

    record = N8nExecutionRecord(
        id=55,
        workflowId="wf-pointer",
        status="success",
        startedAt=datetime.now(timezone.utc),
        stoppedAt=datetime.now(timezone.utc),
        workflowData=WorkflowData(id="wf-pointer", name="Pointer WF", nodes=[WorkflowNode(name="NodeA", type="ToolWorkflow"), WorkflowNode(name="NodeB", type="ToolWorkflow")], connections={}),
        data=decoded,
    )
    trace = map_execution_to_langfuse(record, truncate_limit=None)
    names = [s.name for s in trace.spans]
    assert "NodeA" in names and "NodeB" in names
