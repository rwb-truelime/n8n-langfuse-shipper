"""Tests for agent iteration metadata assignment.

_assign_agent_iterations() groups child spans of each agent into
iteration buckets. Iteration boundaries are inferred from generation
(LLM) spans: each generation marks a new iteration.
"""
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


def _ms(offset: int) -> int:
    return 1_700_000_000_000 + offset


def _make_agent_record(
    agent_start: int,
    agent_exec: int,
    child_nodes: list[tuple[str, str, int, int]],
    child_agent_connections: dict[str, str],
) -> N8nExecutionRecord:
    """Build an agent + children execution.

    child_nodes: list of (name, type, startTime, execTime).
    child_agent_connections: name → link_type.
    """
    nodes = [
        WorkflowNode(
            name="Agent",
            type="@n8n/n8n-nodes-langchain.agent",
        ),
    ]
    run_data: dict = {
        "Agent": [
            NodeRun(
                startTime=agent_start,
                executionTime=agent_exec,
                executionStatus="success",
            )
        ]
    }
    connections: dict = {}
    seen_nodes: set[str] = set()
    for name, ntype, start, ex in child_nodes:
        if name not in seen_nodes:
            nodes.append(
                WorkflowNode(name=name, type=ntype)
            )
            seen_nodes.add(name)
        run_data.setdefault(name, []).append(
            NodeRun(
                startTime=start,
                executionTime=ex,
                executionStatus="success",
            )
        )
        lt = child_agent_connections.get(
            name, "ai_tool"
        )
        if name not in connections:
            connections[name] = {}
            connections[name][lt] = [
                [{"node": "Agent", "type": lt, "index": 0}]
            ]

    return N8nExecutionRecord(
        id=900,
        workflowId="wf-iter",
        status="success",
        startedAt=datetime(
            2026, 1, 1, tzinfo=timezone.utc
        ),
        stoppedAt=datetime(
            2026, 1, 1, 0, 5, tzinfo=timezone.utc
        ),
        workflowData=WorkflowData(
            id="wf-iter",
            name="IterTest",
            nodes=nodes,
            connections=connections,
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_data)
            )
        ),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

LM_TYPE = (
    "@n8n/n8n-nodes-langchain.lmChatOpenAi"
)
TOOL_TYPE = (
    "@n8n/n8n-nodes-langchain.toolHttpRequest"
)


def _get_child_spans(trace, agent_name="Agent #0"):
    agent = next(
        s for s in trace.spans if s.name == agent_name
    )
    return [
        s
        for s in trace.spans
        if s.parent_id == agent.id
    ]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_no_generations_all_iteration_zero():
    """No generation children → all get iteration 0."""
    rec = _make_agent_record(
        agent_start=_ms(0),
        agent_exec=10_000,
        child_nodes=[
            ("ToolA", TOOL_TYPE, _ms(1000), 500),
            ("ToolB", TOOL_TYPE, _ms(2000), 500),
        ],
        child_agent_connections={
            "ToolA": "ai_tool",
            "ToolB": "ai_tool",
        },
    )
    trace = map_execution_to_langfuse(rec)
    children = _get_child_spans(trace)
    for c in children:
        assert c.metadata.get(
            "n8n.agent.iteration"
        ) == 0, (
            f"{c.name} should have iteration 0"
        )


def test_single_generation_single_iteration():
    """One generation + one tool → both iteration 0."""
    rec = _make_agent_record(
        agent_start=_ms(0),
        agent_exec=10_000,
        child_nodes=[
            ("LLM", LM_TYPE, _ms(1000), 2000),
            ("Tool", TOOL_TYPE, _ms(4000), 500),
        ],
        child_agent_connections={
            "LLM": "ai_languageModel",
            "Tool": "ai_tool",
        },
    )
    trace = map_execution_to_langfuse(rec)
    children = _get_child_spans(trace)
    llm = next(
        c for c in children if "LLM" in c.name
    )
    tool = next(
        c for c in children if "Tool" in c.name
    )
    # First generation = iteration 0
    assert llm.metadata["n8n.agent.iteration"] == 0
    # Tool after first (only) gen = iteration 0
    assert tool.metadata["n8n.agent.iteration"] == 0


def test_two_generations_two_iterations():
    """Two LLM runs create iteration 0 and 1."""
    rec = _make_agent_record(
        agent_start=_ms(0),
        agent_exec=20_000,
        child_nodes=[
            ("LLM", LM_TYPE, _ms(1000), 2000),
            ("Tool", TOOL_TYPE, _ms(4000), 500),
            ("LLM", LM_TYPE, _ms(6000), 2000),
        ],
        child_agent_connections={
            "LLM": "ai_languageModel",
            "Tool": "ai_tool",
        },
    )
    trace = map_execution_to_langfuse(rec)
    children = _get_child_spans(trace)
    children_sorted = sorted(
        children, key=lambda s: s.start_time
    )
    # First LLM → iteration 0
    assert (
        children_sorted[0].metadata[
            "n8n.agent.iteration"
        ]
        == 0
    )
    # Tool between gens → iteration 0
    assert (
        children_sorted[1].metadata[
            "n8n.agent.iteration"
        ]
        == 0
    )
    # Second LLM → iteration 1
    assert (
        children_sorted[2].metadata[
            "n8n.agent.iteration"
        ]
        == 1
    )


def test_tool_before_first_gen_gets_iter_zero():
    """Non-generation child before first generation → iter 0."""
    rec = _make_agent_record(
        agent_start=_ms(0),
        agent_exec=15_000,
        child_nodes=[
            ("Memory", TOOL_TYPE, _ms(500), 200),
            ("LLM", LM_TYPE, _ms(1000), 2000),
        ],
        child_agent_connections={
            "Memory": "ai_memory",
            "LLM": "ai_languageModel",
        },
    )
    trace = map_execution_to_langfuse(rec)
    children = _get_child_spans(trace)
    mem = next(
        c for c in children if "Memory" in c.name
    )
    assert mem.metadata["n8n.agent.iteration"] == 0


def test_all_children_have_iteration_metadata():
    """Every child must have n8n.agent.iteration set."""
    rec = _make_agent_record(
        agent_start=_ms(0),
        agent_exec=30_000,
        child_nodes=[
            ("LLM", LM_TYPE, _ms(1000), 2000),
            ("ToolA", TOOL_TYPE, _ms(4000), 500),
            ("LLM", LM_TYPE, _ms(6000), 2000),
            ("ToolB", TOOL_TYPE, _ms(9000), 500),
            ("LLM", LM_TYPE, _ms(11000), 2000),
        ],
        child_agent_connections={
            "LLM": "ai_languageModel",
            "ToolA": "ai_tool",
            "ToolB": "ai_tool",
        },
    )
    trace = map_execution_to_langfuse(rec)
    children = _get_child_spans(trace)
    assert len(children) >= 5
    for c in children:
        assert "n8n.agent.iteration" in c.metadata, (
            f"{c.name} missing iteration metadata"
        )


def test_non_agent_children_no_iteration():
    """Spans not parented to agents don't get iteration."""
    from n8n_langfuse_shipper.models.n8n import (
        NodeRunSource,
    )

    nodes = [
        WorkflowNode(
            name="Start",
            type="n8n-nodes-base.manualTrigger",
        ),
        WorkflowNode(
            name="Step",
            type="n8n-nodes-base.set",
        ),
    ]
    run_data = {
        "Start": [
            NodeRun(
                startTime=_ms(0),
                executionTime=10,
                executionStatus="success",
            )
        ],
        "Step": [
            NodeRun(
                startTime=_ms(100),
                executionTime=50,
                executionStatus="success",
                source=[
                    NodeRunSource(
                        previousNode="Start"
                    )
                ],
            )
        ],
    }
    rec = N8nExecutionRecord(
        id=901,
        workflowId="wf-noiter",
        status="success",
        startedAt=datetime(
            2026, 1, 1, tzinfo=timezone.utc
        ),
        stoppedAt=datetime(
            2026, 1, 1, 0, 1, tzinfo=timezone.utc
        ),
        workflowData=WorkflowData(
            id="wf-noiter",
            name="NoIter",
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
            "n8n.agent.iteration" not in span.metadata
        ), f"{span.name} should not have iteration"
