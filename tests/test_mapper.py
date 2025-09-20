from __future__ import annotations

from datetime import datetime, timezone, timedelta
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


def test_graph_fallback_parent_inference():
    # Build workflow with connections but omit source info in NodeRun for child
    run_data = {
        "Parent": [
            NodeRun(
                startTime=1_700_000_000,
                executionTime=10,
                executionStatus="success",
                data={},
            )
        ],
        "Child": [
            NodeRun(
                startTime=1_700_000_020,
                executionTime=5,
                executionStatus="success",
                data={},
            )
        ],
    }
    rec = _base_record(run_data)
    # Inject a simple connection graph: Parent -> Child
    rec.workflowData.connections = {
        "Parent": {"main": [[{"index": 0, "node": "Child", "type": "main"}]]}
    }
    trace = map_execution_to_langfuse(rec, truncate_limit=200)
    parent_span = next(s for s in trace.spans if s.name == "Parent")
    child_span = next(s for s in trace.spans if s.name == "Child")
    assert child_span.parent_id == parent_span.id, "Graph fallback failed to assign parent span"
    assert child_span.metadata.get("n8n.graph.inferred_parent") is True


def _agent_hierarchy_execution():
    now = datetime.utcnow()
    # Connections: Tool1 -> HAL9000 (ai_tool), LLM1 -> HAL9000 (ai_languageModel), Memory1 -> HAL9000 (ai_memory)
    connections = {
        "Tool1": {"ai_tool": [[{"node": "HAL9000"}]]},
        "LLM1": {"ai_languageModel": [[{"node": "HAL9000"}]]},
        "Memory1": {"ai_memory": [[{"node": "HAL9000"}]]},
    }
    runData = {
        "HAL9000": [
            NodeRun(
                startTime=int(now.timestamp() * 1000),
                executionTime=50,
                executionStatus="success",
                data={"agent": True},
            )
        ],
        "Tool1": [
            NodeRun(
                startTime=int(now.timestamp() * 1000) + 10,
                executionTime=10,
                executionStatus="success",
                data={"toolResult": 42},
            )
        ],
        "LLM1": [
            NodeRun(
                startTime=int(now.timestamp() * 1000) + 20,
                executionTime=30,
                executionStatus="success",
                data={"model": "gpt-4", "tokenUsage": {"completion": 5, "prompt": 10, "total": 15}},
            )
        ],
        "Memory1": [
            NodeRun(
                startTime=int(now.timestamp() * 1000) + 15,
                executionTime=5,
                executionStatus="success",
                data={"memory": "state"},
            )
        ],
    }
    record = N8nExecutionRecord(
        id=333,
        workflowId="wf3",
        status="success",
        startedAt=now,
        stoppedAt=now + timedelta(seconds=1),
        workflowData=WorkflowData(
            id="wf3",
            name="Agent Workflow",
            nodes=[
                WorkflowNode(name="HAL9000", type="n8n-ai.agent"),
                WorkflowNode(name="Tool1", type="n8n-ai.tool"),
                WorkflowNode(name="LLM1", type="n8n-ai.languageModel"),
                WorkflowNode(name="Memory1", type="n8n-ai.memory"),
            ],
            connections=connections,
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))
        ),
    )
    return record


def test_agent_hierarchy_parenting():
    trace = map_execution_to_langfuse(_agent_hierarchy_execution(), truncate_limit=200)
    spans_by_name = {s.name: s for s in trace.spans}
    root = next(s for s in trace.spans if s.parent_id is None)
    agent = spans_by_name["HAL9000"]
    assert agent.parent_id == root.id
    for child in ["Tool1", "LLM1", "Memory1"]:
        assert spans_by_name[child].parent_id == agent.id, f"{child} not parented to agent"
        assert spans_by_name[child].metadata.get("n8n.agent.parent") == "HAL9000"
    llm_span = spans_by_name["LLM1"]
    gen_ids = {g.span_id for g in trace.generations}
    assert llm_span.id in gen_ids
