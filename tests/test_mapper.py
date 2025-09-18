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
from src.checkpoint import load_checkpoint, store_checkpoint
import tempfile
import os


def _base_record(run_data):
    return N8nExecutionRecord(
        id=100,
        workflowId="wf-1",
        status="success",
        startedAt=datetime.now(timezone.utc),
        stoppedAt=datetime.now(timezone.utc),
        workflowData=WorkflowData(
            id="wf-1",
            name="Sample Workflow",
            nodes=[
                WorkflowNode(name="First", type="ToolWorkflow"),
                WorkflowNode(name="LLM", type="OpenAi"),
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_data)
            )
        ),
    )


def test_deterministic_ids_and_generation_detection():
    run_data = {
        "First": [
            NodeRun(
                startTime=1_700_000_000,
                executionTime=50,
                executionStatus="success",
                data={"result": 1},
            )
        ],
        "LLM": [
            NodeRun(
                startTime=1_700_000_100,
                executionTime=120,
                executionStatus="success",
                data={"tokenUsage": {"promptTokens": 5, "completionTokens": 7, "totalTokens": 12}},
                source=[NodeRunSource(previousNode="First", previousNodeRun=0)],
            )
        ],
    }
    rec = _base_record(run_data)
    trace1 = map_execution_to_langfuse(rec, truncate_limit=200)
    trace2 = map_execution_to_langfuse(rec, truncate_limit=200)
    ids1 = [s.id for s in trace1.spans]
    ids2 = [s.id for s in trace2.spans]
    assert ids1 == ids2, "Deterministic span IDs changed between runs"
    # Generation detection for LLM node
    llm_span = next(s for s in trace1.spans if s.name == "LLM")
    assert llm_span.token_usage is not None
    assert any(g.span_id == llm_span.id for g in trace1.generations)


def test_parent_resolution_with_previous_node_run():
    run_data = {
        "A": [
            NodeRun(
                startTime=1_700_000_000,
                executionTime=10,
                executionStatus="success",
                data={},
            )
        ],
        "B": [
            NodeRun(
                startTime=1_700_000_020,
                executionTime=5,
                executionStatus="success",
                data={},
                source=[NodeRunSource(previousNode="A", previousNodeRun=0)],
            )
        ],
    }
    rec = _base_record(run_data)
    trace = map_execution_to_langfuse(rec, truncate_limit=100)
    span_a = next(s for s in trace.spans if s.name == "A")
    span_b = next(s for s in trace.spans if s.name == "B")
    assert span_b.parent_id == span_a.id


def test_truncation_flags():
    large_text = "x" * 500
    run_data = {
        "Node": [
            NodeRun(
                startTime=1_700_000_000,
                executionTime=5,
                executionStatus="success",
                data={"big": large_text},
            )
        ]
    }
    rec = _base_record(run_data)
    trace = map_execution_to_langfuse(rec, truncate_limit=100)
    span = next(s for s in trace.spans if s.name == "Node")
    assert span.metadata.get("n8n.truncated.output") is True


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cp.txt")
        assert load_checkpoint(path) is None
        store_checkpoint(path, 123)
        assert load_checkpoint(path) == 123
