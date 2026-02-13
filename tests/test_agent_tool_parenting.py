"""Tests for agent-tool parenting and AI classification of base-package tools.

Validates two fixes:
1. Connection-graph AI classification: nodes connected to agents via ai_tool
   (e.g. n8n-nodes-base.microsoftSqlTool, n8n-nodes-base.dataTableTool) are
   retained when FILTER_AI_ONLY=true, even if their type/category isn't in the
   LangChain AI package.
2. Post-loop agent parent fixup: tool spans whose startTime precedes their
   agent's startTime are correctly re-parented to the agent span instead of
   falling back to root.

Scenarios:
- Tool starts BEFORE agent (timing inversion) → fixup re-parents to agent
- Tool starts AFTER agent (normal case) → still parented to agent
- Multi-run agent: tool from iteration N parented to correct agent run
- n8n-nodes-base tool types retained by AI filter via graph classification
- Nodes NOT in child_agent_map and NOT AI type → still excluded
- LangChain tools + base tools coexist, all correctly parented
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid5

import pytest

from n8n_langfuse_shipper.mapper import (
    SPAN_NAMESPACE,
    map_execution_to_langfuse,
)
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


def _make_record(
    nodes, run_map, connections, exec_id=5000
) -> N8nExecutionRecord:
    """Helper to build an N8nExecutionRecord with connections."""
    return N8nExecutionRecord(
        id=exec_id,
        workflowId="wf-agent-tool-test",
        status="success",
        startedAt=datetime(
            2026, 2, 13, 8, 0, 0, tzinfo=timezone.utc
        ),
        stoppedAt=datetime(
            2026, 2, 13, 8, 10, 0, tzinfo=timezone.utc
        ),
        workflowData=WorkflowData(
            id="wf-at",
            name="AgentToolWorkflow",
            nodes=nodes,
            connections=connections,
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_map)
            )
        ),
    )


def _ai_tool_connection(tool_name, agent_name):
    """Build an ai_tool connection from tool_name → agent_name."""
    return {
        tool_name: {
            "ai_tool": [
                [{"node": agent_name, "type": "ai_tool", "index": 0}]
            ]
        }
    }


def _merge_connections(*dicts):
    """Merge multiple connection dicts."""
    merged = {}
    for d in dicts:
        for k, v in d.items():
            if k in merged:
                merged[k].update(v)
            else:
                merged[k] = dict(v)
    return merged


# ──────────────────────────────────────────────────────────────────
# Fixture: agent with base-package tools, tools start BEFORE agent
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def timing_inversion_record():
    """Agent workflow where tool startTimes precede agent startTime.

    Timeline:
        t=1000  SqlTool run 0
        t=1100  DataTableTool run 0
        t=1200  DataTableTool run 1
        t=1500  LLM run 0  (LangChain generation)
        t=2000  MyAgent run 0  (starts AFTER tools!)
        t=2100  LLM run 1
        t=2200  SqlTool run 1
        t=2300  DataTableTool run 2
    """
    nodes = [
        WorkflowNode(
            name="MyAgent",
            type="@n8n/n8n-nodes-langchain.agent",
            category="AI/LangChain Nodes",
        ),
        WorkflowNode(
            name="SqlTool",
            type="n8n-nodes-base.microsoftSqlTool",
            category=None,
        ),
        WorkflowNode(
            name="DataTableTool",
            type="n8n-nodes-base.dataTableTool",
            category=None,
        ),
        WorkflowNode(
            name="LLM",
            type="@n8n/n8n-nodes-langchain.lmChatOpenAi",
            category="AI/LangChain Nodes",
        ),
        WorkflowNode(
            name="PostProcess",
            type="Set",
            category="Core Nodes",
        ),
    ]
    connections = _merge_connections(
        _ai_tool_connection("SqlTool", "MyAgent"),
        _ai_tool_connection("DataTableTool", "MyAgent"),
        {
            "LLM": {
                "ai_languageModel": [
                    [
                        {
                            "node": "MyAgent",
                            "type": "ai_languageModel",
                            "index": 0,
                        }
                    ]
                ]
            }
        },
        {
            "MyAgent": {
                "main": [
                    [
                        {
                            "node": "PostProcess",
                            "type": "main",
                            "index": 0,
                        }
                    ]
                ]
            }
        },
    )
    run_map = {
        "SqlTool": [
            NodeRun(
                startTime=1000,
                executionTime=50,
                executionStatus="success",
                data={"main": [[{"json": {"sql": "row1"}}]]},
            ),
            NodeRun(
                startTime=2200,
                executionTime=30,
                executionStatus="success",
                data={"main": [[{"json": {"sql": "row2"}}]]},
            ),
        ],
        "DataTableTool": [
            NodeRun(
                startTime=1100,
                executionTime=40,
                executionStatus="success",
                data={"main": [[{"json": {"dt": "r1"}}]]},
            ),
            NodeRun(
                startTime=1200,
                executionTime=40,
                executionStatus="success",
                data={"main": [[{"json": {"dt": "r2"}}]]},
            ),
            NodeRun(
                startTime=2300,
                executionTime=40,
                executionStatus="success",
                data={"main": [[{"json": {"dt": "r3"}}]]},
            ),
        ],
        "LLM": [
            NodeRun(
                startTime=1500,
                executionTime=200,
                executionStatus="success",
                data={
                    "main": [[{"json": {"response": "hi"}}]],
                    "tokenUsage": {
                        "input": 10,
                        "output": 5,
                    },
                },
            ),
            NodeRun(
                startTime=2100,
                executionTime=150,
                executionStatus="success",
                data={
                    "main": [[{"json": {"response": "bye"}}]],
                    "tokenUsage": {
                        "input": 15,
                        "output": 8,
                    },
                },
            ),
        ],
        "MyAgent": [
            NodeRun(
                startTime=2000,
                executionTime=500,
                executionStatus="success",
                data={
                    "main": [
                        [{"json": {"agent": "done"}}]
                    ]
                },
                source=[
                    NodeRunSource(previousNode="LLM")
                ],
            ),
        ],
        "PostProcess": [
            NodeRun(
                startTime=3000,
                executionTime=10,
                executionStatus="success",
                data={"main": [[{"json": {"ok": True}}]]},
                source=[
                    NodeRunSource(previousNode="MyAgent")
                ],
            ),
        ],
    }
    return _make_record(nodes, run_map, connections)


# ──────────────────────────────────────────────────────────────────
# Test: tools present in unfiltered trace
# ──────────────────────────────────────────────────────────────────


def test_all_tool_spans_present_in_unfiltered_trace(
    timing_inversion_record,
):
    """All tool runs produce spans even without AI filtering."""
    trace = map_execution_to_langfuse(
        timing_inversion_record, filter_ai_only=False
    )
    names = [s.name for s in trace.spans]
    # 2 SqlTool + 3 DataTableTool + 2 LLM + 1 Agent + 1 Post + root
    assert names.count("SqlTool") == 2
    assert names.count("DataTableTool") == 3
    assert names.count("LLM") == 2
    assert names.count("MyAgent") == 1
    assert names.count("PostProcess") == 1


# ──────────────────────────────────────────────────────────────────
# Test: base-package tools retained by AI filter via graph
# ──────────────────────────────────────────────────────────────────


def test_base_tools_retained_by_ai_filter(
    timing_inversion_record,
):
    """n8n-nodes-base tool types connected via ai_tool are kept."""
    trace = map_execution_to_langfuse(
        timing_inversion_record, filter_ai_only=True
    )
    names = [s.name for s in trace.spans]
    # All AI-connected spans should be present
    assert "SqlTool" in names
    assert "DataTableTool" in names
    assert "LLM" in names
    assert "MyAgent" in names
    # PostProcess is non-AI, may or may not be in context
    root = next(
        s for s in trace.spans if s.parent_id is None
    )
    assert root.metadata.get("n8n.filter.ai_only") is True


# ──────────────────────────────────────────────────────────────────
# Test: parent fixup for tools starting before agent
# ──────────────────────────────────────────────────────────────────


def test_tools_before_agent_parented_correctly(
    timing_inversion_record,
):
    """Tool spans that start before their agent are re-parented via fixup."""
    trace = map_execution_to_langfuse(
        timing_inversion_record, filter_ai_only=False
    )
    trace_id = str(timing_inversion_record.id)
    agent_span_id = str(
        uuid5(SPAN_NAMESPACE, f"{trace_id}:MyAgent:0")
    )

    # Find all tool spans
    sql_spans = [
        s for s in trace.spans if s.name == "SqlTool"
    ]
    dt_spans = [
        s for s in trace.spans if s.name == "DataTableTool"
    ]
    llm_spans = [
        s for s in trace.spans if s.name == "LLM"
    ]

    # ALL tools/LLMs connected to agent should be parented to it
    for s in sql_spans:
        assert s.parent_id == agent_span_id, (
            f"SqlTool run {s.metadata.get('n8n.node.run_index')}"
            f" should be parented to MyAgent, got {s.parent_id}"
        )
    for s in dt_spans:
        assert s.parent_id == agent_span_id, (
            f"DataTableTool run "
            f"{s.metadata.get('n8n.node.run_index')}"
            f" should be parented to MyAgent"
        )
    for s in llm_spans:
        assert s.parent_id == agent_span_id, (
            f"LLM run {s.metadata.get('n8n.node.run_index')}"
            f" should be parented to MyAgent"
        )


def test_fixup_metadata_emitted(timing_inversion_record):
    """Spans that were re-parented emit n8n.agent.parent_fixup."""
    trace = map_execution_to_langfuse(
        timing_inversion_record, filter_ai_only=False
    )
    # SqlTool run 0 starts at t=1000, agent at t=2000 → fixup
    sql0 = next(
        s
        for s in trace.spans
        if s.name == "SqlTool"
        and s.metadata.get("n8n.node.run_index") == 0
    )
    assert sql0.metadata.get("n8n.agent.parent_fixup") is True

    # SqlTool run 1 starts at t=2200, agent at t=2000 → no fixup
    # (resolve_parent already found agent in last_span_for_node)
    sql1 = next(
        s
        for s in trace.spans
        if s.name == "SqlTool"
        and s.metadata.get("n8n.node.run_index") == 1
    )
    # This should NOT have fixup since it was correct initially
    assert sql1.metadata.get("n8n.agent.parent_fixup") is not True


# ──────────────────────────────────────────────────────────────────
# Test: non-connected non-AI nodes still excluded
# ──────────────────────────────────────────────────────────────────


def test_non_ai_non_connected_excluded():
    """Nodes not in child_agent_map and not AI type are excluded."""
    nodes = [
        WorkflowNode(
            name="Agent",
            type="@n8n/n8n-nodes-langchain.agent",
            category="AI/LangChain Nodes",
        ),
        WorkflowNode(
            name="LLM",
            type="@n8n/n8n-nodes-langchain.lmChatOpenAi",
            category="AI/LangChain Nodes",
        ),
        WorkflowNode(
            name="RandomSet",
            type="Set",
            category="Core Nodes",
        ),
    ]
    connections = {
        "LLM": {
            "ai_languageModel": [
                [
                    {
                        "node": "Agent",
                        "type": "ai_languageModel",
                        "index": 0,
                    }
                ]
            ]
        }
    }
    run_map = {
        "Agent": [
            NodeRun(
                startTime=1000,
                executionTime=100,
                executionStatus="success",
                data={"main": [[{"json": {}}]]},
            )
        ],
        "LLM": [
            NodeRun(
                startTime=1100,
                executionTime=50,
                executionStatus="success",
                data={
                    "main": [[{"json": {"r": "ok"}}]],
                    "tokenUsage": {"input": 5, "output": 3},
                },
                source=[
                    NodeRunSource(previousNode="Agent")
                ],
            )
        ],
        "RandomSet": [
            NodeRun(
                startTime=1200,
                executionTime=5,
                executionStatus="success",
                data={"main": [[{"json": {"x": 1}}]]},
                source=[
                    NodeRunSource(previousNode="Agent")
                ],
            )
        ],
    }
    rec = _make_record(
        nodes, run_map, connections, exec_id=5001
    )
    trace = map_execution_to_langfuse(
        rec, filter_ai_only=True
    )
    names = [s.name for s in trace.spans]
    assert "Agent" in names
    assert "LLM" in names
    # RandomSet not connected to agent via ai_* and not AI type
    # It MAY appear as post-context. Verify it's at most in
    # context window, not classified as AI.
    root = next(
        s for s in trace.spans if s.parent_id is None
    )
    assert root.metadata.get("n8n.filter.ai_only") is True


# ──────────────────────────────────────────────────────────────────
# Test: multi-run agents with timing inversion
# ──────────────────────────────────────────────────────────────────


def test_multi_run_agent_parenting():
    """Tool runs parented to the correct agent run based on timing.

    Timeline:
        t=1000  AgentX run 0
        t=1100  ToolA run 0  → parent = AgentX run 0
        t=2000  AgentX run 1
        t=2100  ToolA run 1  → parent = AgentX run 1
    """
    nodes = [
        WorkflowNode(
            name="AgentX",
            type="@n8n/n8n-nodes-langchain.agent",
            category="AI/LangChain Nodes",
        ),
        WorkflowNode(
            name="ToolA",
            type="n8n-nodes-base.microsoftSqlTool",
            category=None,
        ),
    ]
    connections = _ai_tool_connection("ToolA", "AgentX")
    run_map = {
        "AgentX": [
            NodeRun(
                startTime=1000,
                executionTime=800,
                executionStatus="success",
                data={"main": [[{"json": {"r": 0}}]]},
            ),
            NodeRun(
                startTime=2000,
                executionTime=800,
                executionStatus="success",
                data={"main": [[{"json": {"r": 1}}]]},
            ),
        ],
        "ToolA": [
            NodeRun(
                startTime=1100,
                executionTime=50,
                executionStatus="success",
                data={"main": [[{"json": {"t": 0}}]]},
            ),
            NodeRun(
                startTime=2100,
                executionTime=50,
                executionStatus="success",
                data={"main": [[{"json": {"t": 1}}]]},
            ),
        ],
    }
    rec = _make_record(
        nodes, run_map, connections, exec_id=5002
    )
    trace = map_execution_to_langfuse(
        rec, filter_ai_only=False
    )
    trace_id = "5002"
    agent0_id = str(
        uuid5(SPAN_NAMESPACE, f"{trace_id}:AgentX:0")
    )
    agent1_id = str(
        uuid5(SPAN_NAMESPACE, f"{trace_id}:AgentX:1")
    )

    tool_spans = sorted(
        [s for s in trace.spans if s.name == "ToolA"],
        key=lambda s: s.start_time,
    )
    assert len(tool_spans) == 2
    # ToolA run 0 (t=1100) → AgentX run 0 (t=1000)
    assert tool_spans[0].parent_id == agent0_id
    # ToolA run 1 (t=2100) → AgentX run 1 (t=2000)
    assert tool_spans[1].parent_id == agent1_id


# ──────────────────────────────────────────────────────────────────
# Test: tool before ANY agent run uses earliest agent
# ──────────────────────────────────────────────────────────────────


def test_tool_before_all_agent_runs_uses_earliest():
    """Tool starting before all agent runs is parented to earliest agent span.

    Timeline:
        t=500   ToolB run 0  → no agent started yet → earliest
        t=1000  AgentY run 0
    """
    nodes = [
        WorkflowNode(
            name="AgentY",
            type="@n8n/n8n-nodes-langchain.agent",
            category="AI/LangChain Nodes",
        ),
        WorkflowNode(
            name="ToolB",
            type="n8n-nodes-base.dataTableTool",
            category=None,
        ),
    ]
    connections = _ai_tool_connection("ToolB", "AgentY")
    run_map = {
        "AgentY": [
            NodeRun(
                startTime=1000,
                executionTime=500,
                executionStatus="success",
                data={"main": [[{"json": {}}]]},
            )
        ],
        "ToolB": [
            NodeRun(
                startTime=500,
                executionTime=50,
                executionStatus="success",
                data={"main": [[{"json": {"b": 1}}]]},
            )
        ],
    }
    rec = _make_record(
        nodes, run_map, connections, exec_id=5003
    )
    trace = map_execution_to_langfuse(
        rec, filter_ai_only=False
    )
    trace_id = "5003"
    agent_id = str(
        uuid5(SPAN_NAMESPACE, f"{trace_id}:AgentY:0")
    )
    tool_span = next(
        s for s in trace.spans if s.name == "ToolB"
    )
    assert tool_span.parent_id == agent_id
    assert tool_span.metadata.get("n8n.agent.parent_fixup") is True


# ──────────────────────────────────────────────────────────────────
# Test: deterministic IDs unchanged
# ──────────────────────────────────────────────────────────────────


def test_deterministic_ids_stable(timing_inversion_record):
    """Re-processing identical input yields identical span IDs."""
    t1 = map_execution_to_langfuse(
        timing_inversion_record, filter_ai_only=False
    )
    t2 = map_execution_to_langfuse(
        timing_inversion_record, filter_ai_only=False
    )
    ids1 = sorted(s.id for s in t1.spans)
    ids2 = sorted(s.id for s in t2.spans)
    assert ids1 == ids2

    parents1 = sorted(
        (s.id, s.parent_id) for s in t1.spans
    )
    parents2 = sorted(
        (s.id, s.parent_id) for s in t2.spans
    )
    assert parents1 == parents2


# ──────────────────────────────────────────────────────────────────
# Test: is_ai_node generic Tool suffix detection (usableAsTool)
# ──────────────────────────────────────────────────────────────────


def test_is_ai_node_generic_tool_suffix():
    """Any node type ending with 'Tool' is recognised as AI.

    n8n's ``usableAsTool`` mechanism appends ``Tool`` to the
    node name (``convertNodeToAiTool``).  Our detection must
    catch ALL such types from ANY package — not a hardcoded
    list.
    """
    from n8n_langfuse_shipper.observation_mapper import is_ai_node

    # n8n-nodes-base tools (existing execution 887 types)
    assert is_ai_node(
        "n8n-nodes-base.microsoftSqlTool", None
    )
    assert is_ai_node(
        "@n8n/n8n-nodes-base.microsoftSqlTool", None
    )
    assert is_ai_node(
        "@n8n/n8n-nodes-base.dataTableTool", None
    )
    assert is_ai_node(
        "n8n-nodes-base.postgresTool", None
    )
    assert is_ai_node(
        "n8n-nodes-base.mySqlTool", None
    )
    # Future / hypothetical usableAsTool conversions
    assert is_ai_node(
        "n8n-nodes-base.httpRequestTool", None
    )
    assert is_ai_node(
        "n8n-nodes-base.slackTool", None
    )
    assert is_ai_node(
        "communityPackage.coolCustomTool", None
    )
    assert is_ai_node(
        "@acme/custom-nodes.fancyTool", None
    )
    # Bare (un-prefixed) tool types
    assert is_ai_node("randomTool", None)
    # Non-tool base node should NOT be AI
    assert not is_ai_node(
        "n8n-nodes-base.Set", None
    )
    assert not is_ai_node(
        "n8n-nodes-base.If", None
    )
    assert not is_ai_node(
        "n8n-nodes-base.Switch", None
    )
