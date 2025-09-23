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
    NodeRunSource,
)

# Tests generation detection when tokenUsage absent but provider substring present

def test_generation_detection_by_type_only():
    now = datetime.now(timezone.utc)
    runData = {
        "Starter": [
            NodeRun(
                startTime=int(now.timestamp() * 1000),
                executionTime=5,
                executionStatus="success",
                data={"x": 1},
            )
        ],
        "AnthropicModel": [
            NodeRun(
                startTime=int(now.timestamp() * 1000) + 5,
                executionTime=7,
                executionStatus="success",
                data={"model": "claude"},
                source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
            )
        ],
    }
    rec = N8nExecutionRecord(
        id=601,
        workflowId="wf-gen-type",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-gen-type",
            name="Gen Type WF",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(name="AnthropicModel", type="AnthropicChat"),
            ],
        ),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "AnthropicModel")
    # Should classify as generation despite no explicit tokenUsage
    # The mapper currently sets observation_type to 'generation' only when _detect_generation True and obs_type guess none.
    # For safety, assert either generation span OR generation entry present.
    assert any(g.span_id == span.id for g in trace.generations), "Expected generation entry for provider heuristic"

