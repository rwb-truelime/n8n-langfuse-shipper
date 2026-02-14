"""Tests for agent envelope expansion.

When agent span timing covers only the final loop iteration,
_expand_agent_envelopes() stretches it to encompass ALL children.
Original timing stored in metadata.
"""
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


def _ms(offset: int) -> int:
    """Epoch ms relative to a base of 1_700_000_000_000."""
    return 1_700_000_000_000 + offset


def _make_agent_record(
    agent_start_ms: int,
    agent_exec_ms: int,
    child_runs: dict[str, list[NodeRun]],
    child_agent_connections: dict[str, str] | None = None,
    agent_node_type: str = (
        "@n8n/n8n-nodes-langchain.agent"
    ),
    extra_nodes: list[WorkflowNode] | None = None,
) -> N8nExecutionRecord:
    """Build record with an agent and AI-connected children.

    child_agent_connections maps child node name → link type
    (e.g. 'ai_tool', 'ai_languageModel').
    """
    if child_agent_connections is None:
        child_agent_connections = {}

    nodes = [
        WorkflowNode(
            name="MyAgent",
            type=agent_node_type,
        ),
    ]
    for name in child_runs:
        nodes.append(
            WorkflowNode(
                name=name,
                type="@n8n/n8n-nodes-langchain.lmChatOpenAi",
            )
        )
    if extra_nodes:
        nodes.extend(extra_nodes)

    connections: dict = {}
    for child_name, link_type in (
        child_agent_connections.items()
    ):
        connections.setdefault(child_name, {})
        connections[child_name][link_type] = [
            [{"node": "MyAgent", "type": link_type, "index": 0}]
        ]

    all_runs: dict = {
        "MyAgent": [
            NodeRun(
                startTime=agent_start_ms,
                executionTime=agent_exec_ms,
                executionStatus="success",
            )
        ],
    }
    all_runs.update(child_runs)

    return N8nExecutionRecord(
        id=800,
        workflowId="wf-env",
        status="success",
        startedAt=datetime(
            2026, 1, 1, tzinfo=timezone.utc
        ),
        stoppedAt=datetime(
            2026, 1, 1, 0, 5, tzinfo=timezone.utc
        ),
        workflowData=WorkflowData(
            id="wf-env",
            name="EnvelopeTest",
            nodes=nodes,
            connections=connections,
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=all_runs)
            )
        ),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_no_expansion_when_children_inside_agent():
    """Agent already encompasses all children → no expansion."""
    agent_start = _ms(0)
    agent_exec = 5000  # 5 seconds

    child_runs = {
        "LLM": [
            NodeRun(
                startTime=_ms(500),
                executionTime=1000,
                executionStatus="success",
            )
        ]
    }
    rec = _make_agent_record(
        agent_start_ms=agent_start,
        agent_exec_ms=agent_exec,
        child_runs=child_runs,
        child_agent_connections={
            "LLM": "ai_languageModel"
        },
    )
    trace = map_execution_to_langfuse(rec)
    agent = next(
        s
        for s in trace.spans
        if s.name == "MyAgent #0"
    )
    assert (
        "n8n.agent.envelope_expanded"
        not in agent.metadata
    )


def test_expansion_when_child_before_agent():
    """Child starts before agent → agent start expanded."""
    # Agent starts at 10s, child at 2s
    agent_start = _ms(10_000)
    agent_exec = 3000

    child_runs = {
        "LLM": [
            NodeRun(
                startTime=_ms(2000),
                executionTime=1500,
                executionStatus="success",
            )
        ]
    }
    rec = _make_agent_record(
        agent_start_ms=agent_start,
        agent_exec_ms=agent_exec,
        child_runs=child_runs,
        child_agent_connections={
            "LLM": "ai_languageModel"
        },
    )
    trace = map_execution_to_langfuse(rec)
    agent = next(
        s
        for s in trace.spans
        if s.name == "MyAgent #0"
    )
    child = next(
        s
        for s in trace.spans
        if s.name == "LLM #0"
    )
    # Agent start should now be <= child start
    assert agent.start_time <= child.start_time
    assert agent.metadata.get(
        "n8n.agent.envelope_expanded"
    )
    assert "n8n.agent.original_start_time" in agent.metadata
    assert (
        "n8n.agent.original_execution_time_ms"
        in agent.metadata
    )


def test_expansion_when_child_ends_after_agent():
    """Child ends after agent → agent end expanded."""
    agent_start = _ms(0)
    agent_exec = 2000  # ends at 2s

    child_runs = {
        "LLM": [
            NodeRun(
                startTime=_ms(500),
                executionTime=5000,  # ends at 5.5s
                executionStatus="success",
            )
        ]
    }
    rec = _make_agent_record(
        agent_start_ms=agent_start,
        agent_exec_ms=agent_exec,
        child_runs=child_runs,
        child_agent_connections={
            "LLM": "ai_languageModel"
        },
    )
    trace = map_execution_to_langfuse(rec)
    agent = next(
        s
        for s in trace.spans
        if s.name == "MyAgent #0"
    )
    child = next(
        s
        for s in trace.spans
        if s.name == "LLM #0"
    )
    assert agent.end_time >= child.end_time
    assert agent.metadata.get(
        "n8n.agent.envelope_expanded"
    )


def test_original_timing_preserved_in_metadata():
    """Expanded agent stores original start and exec time."""
    agent_start = _ms(10_000)
    agent_exec = 2000

    child_runs = {
        "LLM": [
            NodeRun(
                startTime=_ms(1000),
                executionTime=500,
                executionStatus="success",
            )
        ]
    }
    rec = _make_agent_record(
        agent_start_ms=agent_start,
        agent_exec_ms=agent_exec,
        child_runs=child_runs,
        child_agent_connections={
            "LLM": "ai_languageModel"
        },
    )
    trace = map_execution_to_langfuse(rec)
    agent = next(
        s
        for s in trace.spans
        if s.name == "MyAgent #0"
    )
    orig_ms = agent.metadata.get(
        "n8n.agent.original_execution_time_ms"
    )
    assert orig_ms == agent_exec


def test_multi_child_min_max_envelope():
    """Multiple children → agent covers min(start) to max(end)."""
    agent_start = _ms(5000)
    agent_exec = 1000

    child_runs = {
        "ToolA": [
            NodeRun(
                startTime=_ms(1000),
                executionTime=2000,
                executionStatus="success",
            )
        ],
        "ToolB": [
            NodeRun(
                startTime=_ms(3000),
                executionTime=5000,  # ends at 8s
                executionStatus="success",
            )
        ],
    }
    rec = _make_agent_record(
        agent_start_ms=agent_start,
        agent_exec_ms=agent_exec,
        child_runs=child_runs,
        child_agent_connections={
            "ToolA": "ai_tool",
            "ToolB": "ai_tool",
        },
    )
    trace = map_execution_to_langfuse(rec)
    agent = next(
        s
        for s in trace.spans
        if s.name == "MyAgent #0"
    )
    tool_a = next(
        s
        for s in trace.spans
        if s.name == "ToolA #0"
    )
    tool_b = next(
        s
        for s in trace.spans
        if s.name == "ToolB #0"
    )
    # Agent must encompass both children
    assert agent.start_time <= tool_a.start_time
    assert agent.end_time >= tool_b.end_time
    assert agent.metadata.get(
        "n8n.agent.envelope_expanded"
    )


def test_no_agent_hierarchy_no_expansion():
    """Nodes without ai_* connections → no envelope logic."""
    run_data = {
        "Step1": [
            NodeRun(
                startTime=_ms(0),
                executionTime=100,
                executionStatus="success",
            )
        ],
        "Step2": [
            NodeRun(
                startTime=_ms(200),
                executionTime=100,
                executionStatus="success",
                source=[
                    NodeRunSource(previousNode="Step1")
                ],
            )
        ],
    }
    nodes = [
        WorkflowNode(
            name="Step1", type="n8n-nodes-base.set"
        ),
        WorkflowNode(
            name="Step2", type="n8n-nodes-base.set"
        ),
    ]
    rec = N8nExecutionRecord(
        id=801,
        workflowId="wf-noagent",
        status="success",
        startedAt=datetime(
            2026, 1, 1, tzinfo=timezone.utc
        ),
        stoppedAt=datetime(
            2026, 1, 1, 0, 1, tzinfo=timezone.utc
        ),
        workflowData=WorkflowData(
            id="wf-noagent",
            name="NoAgent",
            nodes=nodes,
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_data)
            )
        ),
    )
    trace = map_execution_to_langfuse(rec)
    for span in trace.spans:
        assert (
            "n8n.agent.envelope_expanded"
            not in span.metadata
        )
