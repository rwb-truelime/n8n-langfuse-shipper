"""Tests for span name run-index suffix (#N) on all node spans.

Every node span gets a display name with format 'NodeName #N' where N
is the zero-based run index. Root span name remains unchanged. The
deterministic span ID still uses the original node_name (no suffix)
ensuring ID stability.
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


def _make_record(
    run_data: dict,
    nodes: list[WorkflowNode] | None = None,
) -> N8nExecutionRecord:
    if nodes is None:
        nodes = [
            WorkflowNode(
                name=n, type="n8n-nodes-base.set"
            )
            for n in run_data
        ]
    return N8nExecutionRecord(
        id=700,
        workflowId="wf-idx",
        status="success",
        startedAt=datetime(
            2026, 1, 1, tzinfo=timezone.utc
        ),
        stoppedAt=datetime(
            2026, 1, 1, 0, 1, tzinfo=timezone.utc
        ),
        workflowData=WorkflowData(
            id="wf-idx",
            name="IndexTest",
            nodes=nodes,
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_data)
            )
        ),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_single_run_node_gets_hash_zero():
    """A node with one run gets '#0' suffix."""
    run_data = {
        "Alpha": [
            NodeRun(
                startTime=1000,
                executionTime=100,
                executionStatus="success",
            )
        ]
    }
    trace = map_execution_to_langfuse(_make_record(run_data))
    non_root = [
        s for s in trace.spans if s.parent_id is not None
    ]
    assert len(non_root) == 1
    assert non_root[0].name == "Alpha #0"


def test_multi_run_node_sequential_indices():
    """Multiple runs get '#0', '#1', '#2' etc."""
    run_data = {
        "Beta": [
            NodeRun(
                startTime=1000 + i * 100,
                executionTime=50,
                executionStatus="success",
            )
            for i in range(4)
        ]
    }
    trace = map_execution_to_langfuse(_make_record(run_data))
    non_root = [
        s
        for s in trace.spans
        if s.parent_id is not None
    ]
    names = sorted(s.name for s in non_root)
    assert names == [
        "Beta #0",
        "Beta #1",
        "Beta #2",
        "Beta #3",
    ]


def test_root_span_name_unchanged():
    """Root span keeps the workflow name without suffix."""
    run_data = {
        "X": [
            NodeRun(
                startTime=1000,
                executionTime=10,
                executionStatus="success",
            )
        ]
    }
    trace = map_execution_to_langfuse(_make_record(run_data))
    root = next(
        s
        for s in trace.spans
        if s.parent_id is None
    )
    assert root.name == "IndexTest"
    assert "#" not in root.name


def test_bare_node_name_in_metadata():
    """Metadata n8n.node.name stores original name without suffix."""
    run_data = {
        "Gamma": [
            NodeRun(
                startTime=1000,
                executionTime=10,
                executionStatus="success",
            )
        ]
    }
    trace = map_execution_to_langfuse(_make_record(run_data))
    span = next(
        s for s in trace.spans if s.name == "Gamma #0"
    )
    assert span.metadata["n8n.node.name"] == "Gamma"


def test_span_id_uses_original_name():
    """Span ID is deterministic based on original node name.

    Changing the display name must NOT alter the UUIDv5 span ID.
    """
    from uuid import uuid5

    from n8n_langfuse_shipper.mapping.id_utils import (
        SPAN_NAMESPACE,
    )

    run_data = {
        "Delta": [
            NodeRun(
                startTime=1000,
                executionTime=10,
                executionStatus="success",
            )
        ]
    }
    trace = map_execution_to_langfuse(_make_record(run_data))
    span = next(
        s for s in trace.spans if s.name == "Delta #0"
    )
    expected_id = str(
        uuid5(SPAN_NAMESPACE, f"{trace.id}:Delta:0")
    )
    assert span.id == expected_id
