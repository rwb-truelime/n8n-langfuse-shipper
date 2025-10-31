"""Tests for AI-only span filtering feature.

Updated for windowed AI filtering semantics:
    * Retains root span always.
    * Retains all AI spans.
    * Retains at most two chronological spans immediately before first AI span
        (excluding root which is unconditional) and two chronological spans
        immediately after last AI span.
    * Retains only parent-chain bridging spans strictly between earliest and
        latest AI spans whose ancestor chain links to an AI span (no unrelated
        siblings outside window).
    * Root-only fallback when no AI spans exist.

Validates:
    * Classification via is_ai_node (category vs type).
    * Window pre/post context counts capped at 2.
    * Presence of window metadata keys.
    * Root-only behavior when no AI spans.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.mapper import map_execution_to_langfuse
from src.observation_mapper import is_ai_node
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


def _base_record(nodes, run_map, exec_id=999) -> N8nExecutionRecord:
    return N8nExecutionRecord(
        id=exec_id,
        workflowId="wf-ai-filter-test",
        status="success",
        startedAt=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        stoppedAt=datetime(2025, 1, 1, 0, 1, 0, tzinfo=timezone.utc),
        workflowData=WorkflowData(
            id="wf",
            name="workflow",
            nodes=nodes,
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_map)
            )
        ),
    )


def test_is_ai_node_category_fast_path():
    assert is_ai_node("Whatever", "AI/LangChain Nodes") is True


def test_is_ai_node_type_match():
    # Known AI type without category
    assert is_ai_node("LmChatOpenAi", None) is True
    assert is_ai_node("Agent", None) is True


def test_is_ai_node_limescape_docs_prefixed():
    # Prefixed custom docs node types should normalize to AI membership
    assert is_ai_node("n8n-nodes-limescape-docs.limescapeDocs", None) is True
    assert is_ai_node("@n8n/n8n-nodes-limescape-docs.limescapeDocs", None) is True


def test_is_ai_node_negative():
    assert is_ai_node("Set", "Core Nodes") is False
    assert is_ai_node("If", "Core Nodes") is False
    assert is_ai_node(None, None) is False  # type: ignore[arg-type]


def test_filter_keeps_root_and_ai_only():
    nodes = [
        WorkflowNode(name="Start", type="ManualTrigger", category="Trigger Nodes"),
        WorkflowNode(name="Prep", type="Set", category="Core Nodes"),
        WorkflowNode(name="LLM", type="LmChatOpenAi", category="AI/LangChain Nodes"),
        WorkflowNode(name="Vector", type="VectorStorePinecone", category="AI/LangChain Nodes"),
    ]
    run_map = {
        "Start": [
            NodeRun(startTime=1000, executionTime=10, executionStatus="success", data={"main": [[{"json": {}}]]})
        ],
        "Prep": [
            NodeRun(
                startTime=1100,
                executionTime=20,
                executionStatus="success",
                data={"main": [[{"json": {"prepared": True}}]]},
                source=[NodeRunSource(previousNode="Start")],
            )
        ],
        "LLM": [
            NodeRun(
                startTime=1200,
                executionTime=500,
                executionStatus="success",
                data={
                    "main": [[{"json": {"response": "Hi"}}]],
                    "tokenUsage": {"input": 10, "output": 20},
                },
                source=[NodeRunSource(previousNode="Prep")],
            )
        ],
        "Vector": [
            NodeRun(
                startTime=1700,
                executionTime=100,
                executionStatus="success",
                data={"main": [[{"json": {"vector": [0.1, 0.2]}}]]},
                source=[NodeRunSource(previousNode="LLM")],
            )
        ],
    }
    rec = _base_record(nodes, run_map, exec_id=1001)
    trace = map_execution_to_langfuse(rec, filter_ai_only=True)
    # Window semantics: root + AI spans (LLM, Vector) + up to two pre-context
    # spans before first AI (Prep, Start) and up to two post-context after last
    # AI (none here). Total expected = 1(root)+2(pre)+2(ai)=5.
    assert len(trace.spans) == 5
    root = next(s for s in trace.spans if s.parent_id is None)
    assert root.metadata.get("n8n.filter.ai_only") is True
    # Only non-AI spans excluded would be none here (all part of ancestor chain)
    # Excluded node count should reflect any nodes dropped (none dropped here).
    assert root.metadata.get("n8n.filter.excluded_node_count") == 0
    names = [s.name for s in trace.spans]
    assert {"Start", "Prep", "LLM", "Vector"}.issubset(set(names))
@pytest.fixture
def sample_execution_record_ai() -> N8nExecutionRecord:
    """Representative execution containing:
    - Several non-AI spans before first AI span
    - Multiple AI spans (agent + generations) forming window
    - Trailing non-AI spans after last AI span (to test post-context retention)
    """
    nodes = [
        WorkflowNode(name="Trigger", type="ManualTrigger", category="Trigger Nodes"),
        WorkflowNode(name="Prep1", type="Set", category="Core Nodes"),
        WorkflowNode(name="Prep2", type="Set", category="Core Nodes"),
        WorkflowNode(name="AgentA", type="Agent", category="AI/LangChain Nodes"),
        WorkflowNode(name="LLM1", type="LmChatOpenAi", category="AI/LangChain Nodes"),
        WorkflowNode(name="ToolX", type="Tool", category="AI/LangChain Nodes"),
        WorkflowNode(name="LLM2", type="LmChatOpenAi", category="AI/LangChain Nodes"),
        WorkflowNode(name="Post1", type="Merge", category="Core Nodes"),
        WorkflowNode(name="Post2", type="Set", category="Core Nodes"),
        WorkflowNode(name="TailIgnored", type="Set", category="Core Nodes"),
    ]
    # Build run data with chronological ordering using startTime increments.
    run_map = {
        "Trigger": [
            NodeRun(startTime=1000, executionTime=10, executionStatus="success", data={"main": [[{"json": {}}]]})
        ],
        "Prep1": [
            NodeRun(
                startTime=1100,
                executionTime=10,
                executionStatus="success",
                data={"main": [[{"json": {"p": 1}}]]},
                source=[NodeRunSource(previousNode="Trigger")],
            )
        ],
        "Prep2": [
            NodeRun(
                startTime=1200,
                executionTime=10,
                executionStatus="success",
                data={"main": [[{"json": {"p": 2}}]]},
                source=[NodeRunSource(previousNode="Prep1")],
            )
        ],
        "AgentA": [
            NodeRun(
                startTime=1300,
                executionTime=30,
                executionStatus="success",
                data={"main": [[{"json": {"agent": "start"}}]]},
                source=[NodeRunSource(previousNode="Prep2")],
            )
        ],
        "LLM1": [
            NodeRun(
                startTime=1400,
                executionTime=50,
                executionStatus="success",
                data={"main": [[{"json": {"response": "hi"}}]], "tokenUsage": {"input": 50, "output": 10}},
                source=[NodeRunSource(previousNode="AgentA")],
            )
        ],
        "ToolX": [
            NodeRun(
                startTime=1450,
                executionTime=20,
                executionStatus="success",
                data={"main": [[{"json": {"tool": "x"}}]]},
                source=[NodeRunSource(previousNode="LLM1")],
            )
        ],
        "LLM2": [
            NodeRun(
                startTime=1500,
                executionTime=40,
                executionStatus="success",
                data={"main": [[{"json": {"response": "bye"}}]], "tokenUsage": {"input": 40, "output": 5}},
                source=[NodeRunSource(previousNode="ToolX")],
            )
        ],
        "Post1": [
            NodeRun(
                startTime=1600,
                executionTime=10,
                executionStatus="success",
                data={"main": [[{"json": {"post": 1}}]]},
                source=[NodeRunSource(previousNode="LLM2")],
            )
        ],
        "Post2": [
            NodeRun(
                startTime=1700,
                executionTime=10,
                executionStatus="success",
                data={"main": [[{"json": {"post": 2}}]]},
                source=[NodeRunSource(previousNode="Post1")],
            )
        ],
        "TailIgnored": [
            NodeRun(
                startTime=1800,
                executionTime=10,
                executionStatus="success",
                data={"main": [[{"json": {"tail": True}}]]},
                source=[NodeRunSource(previousNode="Post2")],
            )
        ],
    }
    return _base_record(nodes, run_map, exec_id=2001)


def test_window_pre_post_counts(sample_execution_record_ai):
    """Validate pre/post context counts do not exceed 2 and are recorded."""
    trace = map_execution_to_langfuse(sample_execution_record_ai, filter_ai_only=True)
    root = next(s for s in trace.spans if "n8n.execution.id" in s.metadata)
    pre_count = root.metadata.get("n8n.filter.pre_context_count")
    post_count = root.metadata.get("n8n.filter.post_context_count")
    chain_count = root.metadata.get("n8n.filter.chain_context_count")
    assert pre_count is not None and 0 <= pre_count <= 2
    assert post_count is not None and 0 <= post_count <= 2
    assert chain_count is not None and chain_count >= 0


def test_no_excessive_pre_context(sample_execution_record_ai):
    """Ensure we do not retain large blocks of spans before the first AI span."""
    full_trace = map_execution_to_langfuse(sample_execution_record_ai, filter_ai_only=False)
    filtered = map_execution_to_langfuse(sample_execution_record_ai, filter_ai_only=True)
    # Determine first AI span in original (skip root at index 0)
    first_ai_full_idx = None
    for idx, span in enumerate(full_trace.spans[1:], start=1):
        # Use workflow classification via span name mapping from the record
        # (span.name matches WorkflowNode.name)
        node = next((n for n in sample_execution_record_ai.workflowData.nodes if n.name == span.name), None)
        if node and is_ai_node(node.type, node.category):
            first_ai_full_idx = idx
            break
    assert first_ai_full_idx is not None, "Fixture must contain AI spans"
    # Find first non-root span in filtered trace that is AI (window start)
    first_ai_filtered_idx = None
    for idx, span in enumerate(filtered.spans[1:], start=1):
        node = next((n for n in sample_execution_record_ai.workflowData.nodes if n.name == span.name), None)
        if node and is_ai_node(node.type, node.category):
            first_ai_filtered_idx = idx
            break
    assert first_ai_filtered_idx is not None
    root = next(s for s in filtered.spans if "n8n.execution.id" in s.metadata)
    pre_ct = root.metadata.get("n8n.filter.pre_context_count")
    assert pre_ct is not None and pre_ct <= 2
    # Ensure number of non-root spans preceding first AI in filtered <= 2
    preceding_filtered = [s for s in filtered.spans[1:first_ai_filtered_idx]]
    assert len(preceding_filtered) == pre_ct


def test_window_metadata_keys_present(sample_execution_record_ai):
    trace = map_execution_to_langfuse(sample_execution_record_ai, filter_ai_only=True)
    root = next(s for s in trace.spans if "n8n.execution.id" in s.metadata)
    for key in (
        "n8n.filter.window_start_span",
        "n8n.filter.window_end_span",
        "n8n.filter.excluded_node_count",
        "n8n.filter.pre_context_count",
        "n8n.filter.post_context_count",
        "n8n.filter.chain_context_count",
    ):
        assert key in root.metadata


def test_filter_root_only_when_no_ai():
    nodes = [
        WorkflowNode(name="Set", type="Set", category="Core Nodes"),
        WorkflowNode(name="If", type="If", category="Core Nodes"),
    ]
    run_map = {
        "Set": [NodeRun(startTime=1000, executionTime=5, executionStatus="success", data={"main": [[{"json": {}}]]})],
        "If": [
            NodeRun(
                startTime=1100,
                executionTime=5,
                executionStatus="success",
                data={"main": [[{"json": {}}]]},
                source=[NodeRunSource(previousNode="Set")],
            )
        ],
    }
    rec = _base_record(nodes, run_map, exec_id=1002)
    trace = map_execution_to_langfuse(rec, filter_ai_only=True)
    assert len(trace.spans) == 1
    root = trace.spans[0]
    assert root.metadata.get("n8n.filter.ai_only") is True
    assert root.metadata.get("n8n.filter.no_ai_spans") is True
    assert root.metadata.get("n8n.filter.excluded_node_count") == 2


def test_filter_preserves_ancestor_chain():
    nodes = [
        WorkflowNode(name="Start", type="ManualTrigger", category="Trigger Nodes"),
        WorkflowNode(name="Prep", type="Set", category="Core Nodes"),
        WorkflowNode(name="AgentNode", type="Agent", category="AI/LangChain Nodes"),
    ]
    run_map = {
        "Start": [NodeRun(startTime=1000, executionTime=5, executionStatus="success", data={"main": [[{"json": {}}]]})],
        "Prep": [
            NodeRun(
                startTime=1100,
                executionTime=5,
                executionStatus="success",
                data={"main": [[{"json": {"prep": 1}}]]},
                source=[NodeRunSource(previousNode="Start")],
            )
        ],
        "AgentNode": [
            NodeRun(
                startTime=1200,
                executionTime=15,
                executionStatus="success",
                data={"main": [[{"json": {"result": "x"}}]]},
                source=[NodeRunSource(previousNode="Prep")],
            )
        ],
    }
    rec = _base_record(nodes, run_map, exec_id=1003)
    trace = map_execution_to_langfuse(rec, filter_ai_only=True)
    names = [s.name for s in trace.spans]
    # Root + Start + Prep + AgentNode retained (Start ancestor)
    assert {trace.spans[0].name, "Start", "Prep", "AgentNode"} == set(names)


def test_filter_disabled_no_changes():
    nodes = [
        WorkflowNode(name="Set", type="Set", category="Core Nodes"),
        WorkflowNode(name="LLM", type="LmChatOpenAi", category="AI/LangChain Nodes"),
    ]
    run_map = {
        "Set": [NodeRun(startTime=1000, executionTime=5, executionStatus="success", data={"main": [[{"json": {}}]]})],
        "LLM": [
            NodeRun(
                startTime=1100,
                executionTime=5,
                executionStatus="success",
                data={"main": [[{"json": {"t": 1}}]]},
                source=[NodeRunSource(previousNode="Set")],
            )
        ],
    }
    rec = _base_record(nodes, run_map, exec_id=1004)
    full_trace = map_execution_to_langfuse(rec, filter_ai_only=False)
    filtered_trace = map_execution_to_langfuse(rec, filter_ai_only=True)
    # Filtering retains Set because it's an ancestor of an AI node; counts equal
    assert len(full_trace.spans) == len(filtered_trace.spans) == 3
    assert any(s.name == "Set" for s in full_trace.spans)
    assert any(s.name == "Set" for s in filtered_trace.spans)
    root_filtered = next(s for s in filtered_trace.spans if s.parent_id is None)
    assert root_filtered.metadata.get("n8n.filter.ai_only") is True
    assert any(s.metadata.get("n8n.filter.ai_only") for s in filtered_trace.spans if s.parent_id is None)
