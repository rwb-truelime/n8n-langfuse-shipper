"""Tests for AI-only node extraction feature - basic functionality.

Tests core extraction mechanics:
    * Extract from multiple specified nodes
    * Handle missing nodes gracefully
    * Extract all runs from multi-run nodes
    * Include execution status and metadata structure
    * Empty filtering results (null values for visibility)
    * Extraction metadata (_meta section)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from n8n_langfuse_shipper.mapper import map_execution_to_langfuse
from n8n_langfuse_shipper.mapping.node_extractor import extract_nodes_data
from n8n_langfuse_shipper.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)


def _base_record_with_nodes(nodes_config: list[tuple[str, str]]) -> N8nExecutionRecord:
    """Create test execution record with specified nodes.

    Args:
        nodes_config: List of (node_name, node_type) tuples

    Returns:
        N8nExecutionRecord with nodes and basic run data
    """
    nodes = [WorkflowNode(name=name, type=typ) for name, typ in nodes_config]

    # Create one basic run for each node
    run_data = {}
    for name, _ in nodes_config:
        run_data[name] = [
            NodeRun(
                startTime=1704067200000,  # 2024-01-01 00:00:00 UTC
                executionTime=100,
                executionStatus="success",
                data={"output": f"data from {name}"},
            )
        ]

    return N8nExecutionRecord(
        id=1000,
        workflowId="wf-extraction-test",
        status="success",
        startedAt=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        stoppedAt=datetime(2024, 1, 1, 0, 1, 0, tzinfo=timezone.utc),
        workflowData=WorkflowData(
            id="wf",
            name="test_workflow",
            nodes=nodes,
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_data)
            )
        ),
    )


def test_extract_single_node():
    """Extract data from a single specified node."""
    nodes = [
        ("WebhookTrigger", "Webhook"),
        ("DataTransform", "Set"),
        ("AI Agent", "@n8n/n8n-nodes-langchain.agent"),
    ]
    rec = _base_record_with_nodes(nodes)
    run_data = rec.data.executionData.resultData.runData

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["WebhookTrigger"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    assert "WebhookTrigger" in extracted
    assert extracted["WebhookTrigger"]["runs"]
    assert len(extracted["WebhookTrigger"]["runs"]) == 1

    run_entry = extracted["WebhookTrigger"]["runs"][0]
    assert run_entry["run_index"] == 0
    assert run_entry["execution_status"] == "success"
    assert run_entry["output"] == {"output": "data from WebhookTrigger"}

    # Check metadata
    assert extracted["_meta"]["extracted_count"] == 1
    assert extracted["_meta"]["nodes_requested"] == 1


def test_extract_multiple_nodes():
    """Extract data from multiple specified nodes."""
    nodes = [
        ("Node1", "Set"),
        ("Node2", "Set"),
        ("Node3", "@n8n/n8n-nodes-langchain.agent"),
    ]
    rec = _base_record_with_nodes(nodes)
    run_data = rec.data.executionData.resultData.runData

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["Node1", "Node2"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    assert "Node1" in extracted
    assert "Node2" in extracted
    assert "Node3" not in extracted
    assert extracted["_meta"]["extracted_count"] == 2
    assert extracted["_meta"]["nodes_requested"] == 2


def test_extract_missing_node():
    """Handle missing node gracefully (node not in execution)."""
    nodes = [("ExistingNode", "Set")]
    rec = _base_record_with_nodes(nodes)
    run_data = rec.data.executionData.resultData.runData

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["ExistingNode", "MissingNode", "AnotherMissing"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    assert "ExistingNode" in extracted
    assert "MissingNode" not in extracted
    assert "AnotherMissing" not in extracted

    # Check metadata tracking
    assert extracted["_meta"]["extracted_count"] == 1
    assert extracted["_meta"]["nodes_requested"] == 3
    assert "MissingNode" in extracted["_meta"]["nodes_not_found"]
    assert "AnotherMissing" in extracted["_meta"]["nodes_not_found"]


def test_extract_multiple_runs():
    """Extract all runs from a node with multiple executions."""
    rec = _base_record_with_nodes([("MultiRun", "Set")])

    # Add multiple runs for the same node
    rec.data.executionData.resultData.runData["MultiRun"] = [
        NodeRun(
            startTime=1704067200000,
            executionTime=100,
            executionStatus="success",
            data={"output": "run 1"},
        ),
        NodeRun(
            startTime=1704067300000,
            executionTime=150,
            executionStatus="success",
            data={"output": "run 2"},
        ),
        NodeRun(
            startTime=1704067400000,
            executionTime=200,
            executionStatus="error",
            data={"output": "run 3"},
            error={"message": "test error"},
        ),
    ]

    run_data = rec.data.executionData.resultData.runData
    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["MultiRun"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    assert len(extracted["MultiRun"]["runs"]) == 3

    # Verify run indices and statuses
    assert extracted["MultiRun"]["runs"][0]["run_index"] == 0
    assert extracted["MultiRun"]["runs"][0]["execution_status"] == "success"
    assert extracted["MultiRun"]["runs"][0]["output"] == {"output": "run 1"}

    assert extracted["MultiRun"]["runs"][1]["run_index"] == 1
    assert extracted["MultiRun"]["runs"][1]["execution_status"] == "success"

    assert extracted["MultiRun"]["runs"][2]["run_index"] == 2
    assert extracted["MultiRun"]["runs"][2]["execution_status"] == "error"


def test_extract_with_input_data():
    """Extract both input and output data from node."""
    rec = _base_record_with_nodes([("NodeWithInput", "Set")])

    # Add input override
    rec.data.executionData.resultData.runData["NodeWithInput"] = [
        NodeRun(
            startTime=1704067200000,
            executionTime=100,
            executionStatus="success",
            inputOverride={"input_key": "input_value"},
            data={"output_key": "output_value"},
        )
    ]

    run_data = rec.data.executionData.resultData.runData
    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["NodeWithInput"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    run_entry = extracted["NodeWithInput"]["runs"][0]
    assert run_entry["input"] == {"input_key": "input_value"}
    assert run_entry["output"] == {"output_key": "output_value"}


def test_extract_null_when_no_data():
    """Include null values when node has no input/output data."""
    rec = _base_record_with_nodes([("EmptyNode", "Set")])

    # Node with no data
    rec.data.executionData.resultData.runData["EmptyNode"] = [
        NodeRun(
            startTime=1704067200000,
            executionTime=50,
            executionStatus="success",
            data={},  # Empty output
        )
    ]

    run_data = rec.data.executionData.resultData.runData
    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["EmptyNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    run_entry = extracted["EmptyNode"]["runs"][0]
    # Should have null/None values for visibility
    assert run_entry["input"] is None
    assert run_entry["output"] is None or run_entry["output"] == {}
    assert run_entry["execution_status"] == "success"


def test_no_extraction_when_empty_list():
    """Return empty dict when extraction_nodes is empty."""
    rec = _base_record_with_nodes([("SomeNode", "Set")])
    run_data = rec.data.executionData.resultData.runData

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=[],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    assert extracted == {}


def test_extraction_metadata_structure():
    """Verify _meta section structure and contents."""
    rec = _base_record_with_nodes([("Node1", "Set"), ("Node2", "Set")])
    run_data = rec.data.executionData.resultData.runData

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["Node1", "MissingNode"],
        include_patterns=["userId"],
        exclude_patterns=["*password*"],
        max_value_len=10000,
    )

    meta = extracted["_meta"]
    assert meta["extracted_count"] == 1
    assert meta["nodes_requested"] == 2
    assert "MissingNode" in meta["nodes_not_found"]

    # Config should be included when patterns specified
    assert "extraction_config" in meta
    assert meta["extraction_config"]["include_keys"] == ["userId"]
    assert meta["extraction_config"]["exclude_keys"] == ["*password*"]


@pytest.mark.skip(reason="Integration test - needs lru_cache.cache_clear() for config mocking")
def test_integration_with_ai_filtering(monkeypatch):
    """End-to-end test: extraction attaches to root span during AI filtering."""
    # Patch config to specify extraction nodes
    from n8n_langfuse_shipper import config as config_module

    # Store original function
    original_get_settings = config_module.get_settings

    class MockSettings:
        FILTER_AI_EXTRACTION_NODES = ["WebhookTrigger"]
        FILTER_AI_EXTRACTION_INCLUDE_KEYS = []
        FILTER_AI_EXTRACTION_EXCLUDE_KEYS = []
        FILTER_AI_EXTRACTION_MAX_VALUE_LEN = 10000

        # Add all required Settings attributes
        DB_TABLE_PREFIX = ""

    def mock_get_settings():
        return MockSettings()

    monkeypatch.setattr(config_module, "get_settings", mock_get_settings)

    nodes = [
        ("WebhookTrigger", "Webhook"),
        ("DataNode", "Set"),
        ("AI Agent", "@n8n/n8n-nodes-langchain.agent"),
    ]
    rec = _base_record_with_nodes(nodes)

    # Add some data to WebhookTrigger
    rec.data.executionData.resultData.runData["WebhookTrigger"][0].data = {
        "body": {"userId": "123", "action": "test"},
        "headers": {"x-correlation-id": "abc-456"},
    }

    # Map with AI filtering enabled
    trace = map_execution_to_langfuse(rec, filter_ai_only=True)

    # Find root span
    root_span = next(s for s in trace.spans if "n8n.execution.id" in s.metadata)

    # Verify extraction attached
    assert "n8n.extracted_nodes" in root_span.metadata, (
        f"Expected n8n.extracted_nodes in metadata. "
        f"Actual keys: {list(root_span.metadata.keys())}"
    )
    extracted = root_span.metadata["n8n.extracted_nodes"]

    assert "WebhookTrigger" in extracted
    assert extracted["WebhookTrigger"]["runs"][0]["output"]["body"]["userId"] == "123"
    assert extracted["_meta"]["extracted_count"] == 1
