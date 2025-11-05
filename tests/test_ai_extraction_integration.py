"""
Integration tests for AI-only filtering with node extraction feature.

These tests verify end-to-end behavior:
- Extraction occurs before filtering
- Extracted data attached to root span metadata
- Integration with mapper public API
- Metadata structure correctness
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

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


@pytest.fixture
def ai_execution_with_chat_and_tools():
    """
    Execution with AI chat node and tool nodes.
    Tool nodes will be filtered out but should be extractable.
    """
    return N8nExecutionRecord(
        id=9999,
        workflowId="wf-ai-tools",
        status="success",
        startedAt=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        stoppedAt=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        workflowData=WorkflowData(
            id="wf-ai-tools",
            name="AI Tools Workflow",
            nodes=[
                WorkflowNode(name="AIChatNode", type="@n8n/langchain.lmchat"),
                WorkflowNode(name="WebScraperTool", type="n8n-nodes-base.httpRequest"),
                WorkflowNode(
                    name="DatabaseQueryTool",
                    type="n8n-nodes-base.postgres",
                ),
                WorkflowNode(name="TriggerNode", type="n8n-nodes-base.manualTrigger"),
            ],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "TriggerNode": [
                            NodeRun(
                                startTime=1704110400000,
                                executionTime=10,
                                executionStatus="success",
                                data={
                                    "main": [
                                        [{"json": {"query": "weather in Tokyo"}}]
                                    ]
                                },
                            )
                        ],
                        "AIChatNode": [
                            NodeRun(
                                startTime=1704110401000,
                                executionTime=2000,
                                executionStatus="success",
                                data={
                                    "main": [
                                        [
                                            {
                                                "json": {
                                                    "response": "Tokyo weather is sunny",
                                                    "model": "gpt-4",
                                                }
                                            }
                                        ]
                                    ],
                                    "ai_languageModel": [
                                        [
                                            {
                                                "json": {
                                                    "tokenUsage": {
                                                        "input": 50,
                                                        "output": 20,
                                                        "total": 70,
                                                    }
                                                }
                                            }
                                        ]
                                    ],
                                },
                            )
                        ],
                        "WebScraperTool": [
                            NodeRun(
                                startTime=1704110403000,
                                executionTime=800,
                                executionStatus="success",
                                data={
                                    "main": [
                                        [
                                            {
                                                "json": {
                                                    "url": "https://weather.example.com",
                                                    "html": "<html>Weather data...</html>",
                                                    "temperature": "25C",
                                                    "condition": "sunny",
                                                }
                                            }
                                        ]
                                    ]
                                },
                            )
                        ],
                        "DatabaseQueryTool": [
                            NodeRun(
                                startTime=1704110404000,
                                executionTime=300,
                                executionStatus="success",
                                data={
                                    "main": [
                                        [
                                            {
                                                "json": {
                                                    "query": "SELECT * FROM weather_history",
                                                    "rows": [
                                                        {
                                                            "date": "2024-01-01",
                                                            "temp": "23C",
                                                        }
                                                    ],
                                                    "secret_connection_string": "postgresql://user:pass@host/db",
                                                }
                                            }
                                        ]
                                    ]
                                },
                            )
                        ],
                    }
                )
            )
        ),
    )


def test_extraction_with_ai_filtering_enabled(
    monkeypatch, ai_execution_with_chat_and_tools
):
    """
    When AI filtering enabled with extraction config:
    - Tool nodes excluded from spans
    - Tool nodes extracted to root metadata
    - Root span contains n8n.extracted_nodes
    """
    monkeypatch.setenv(
        "FILTER_AI_EXTRACTION_NODES", "WebScraperTool,DatabaseQueryTool"
    )
    monkeypatch.setenv("FILTER_AI_EXTRACTION_INCLUDE_KEYS", "")
    monkeypatch.setenv("FILTER_AI_EXTRACTION_EXCLUDE_KEYS", "secret*")
    monkeypatch.setenv("FILTER_AI_EXTRACTION_MAX_VALUE_LEN", "10000")

    # Need to clear lru_cache on get_settings
    from n8n_langfuse_shipper.config import get_settings
    get_settings.cache_clear()

    # Pass filter_ai_only as function parameter (not env var)
    trace = map_execution_to_langfuse(
        ai_execution_with_chat_and_tools, filter_ai_only=True
    )

    # Should have root span + TriggerNode + AIChatNode (tools filtered out)
    assert len(trace.spans) == 3
    span_names = {s.name for s in trace.spans}
    assert "TriggerNode" in span_names
    assert "AIChatNode" in span_names
    assert "WebScraperTool" not in span_names
    assert "DatabaseQueryTool" not in span_names

    # Find root span (parent_id is None)
    root_span = next(s for s in trace.spans if s.parent_id is None)

    # Root span should have extraction metadata
    assert "n8n.extracted_nodes" in root_span.metadata
    extracted = root_span.metadata["n8n.extracted_nodes"]

    # Should have both extracted nodes
    assert "WebScraperTool" in extracted
    assert "DatabaseQueryTool" in extracted

    # WebScraperTool should have run data
    web_scraper = extracted["WebScraperTool"]
    assert "runs" in web_scraper
    assert len(web_scraper["runs"]) == 1
    run = web_scraper["runs"][0]
    assert run["run_index"] == 0
    assert "output" in run
    assert "url" in run["output"]
    assert "temperature" in run["output"]

    # DatabaseQueryTool should have run data with secret filtered out
    db_query = extracted["DatabaseQueryTool"]
    assert "runs" in db_query
    assert len(db_query["runs"]) == 1
    run = db_query["runs"][0]
    assert "output" in run
    assert "query" in run["output"]
    assert "rows" in run["output"]
    assert "secret_connection_string" not in run["output"]  # excluded

    # Should have metadata section
    assert "_meta" in extracted
    meta = extracted["_meta"]
    assert meta["extracted_count"] == 2
    assert meta["filter_ai_only"] is True


def test_extraction_without_ai_filtering(monkeypatch, ai_execution_with_chat_and_tools):
    """
    When AI filtering disabled:
    - Extraction config ignored
    - All nodes present as spans
    - No n8n.extracted_nodes metadata
    """
    monkeypatch.setenv("FILTER_AI_EXTRACTION_NODES", "WebScraperTool")

    from n8n_langfuse_shipper.config import get_settings
    get_settings.cache_clear()

    # Pass filter_ai_only=False explicitly
    trace = map_execution_to_langfuse(
        ai_execution_with_chat_and_tools, filter_ai_only=False
    )

    # Should have root span + all 4 nodes
    assert len(trace.spans) == 5
    span_names = {s.name for s in trace.spans}
    assert "WebScraperTool" in span_names  # Not filtered
    assert "DatabaseQueryTool" in span_names

    # Root span should NOT have extraction metadata
    root_span = next(s for s in trace.spans if s.parent_id is None)
    assert "n8n.extracted_nodes" not in root_span.metadata


def test_extraction_with_no_matching_nodes(monkeypatch, ai_execution_with_chat_and_tools):
    """
    When extraction config specifies nodes not in execution:
    - Empty n8n.extracted_nodes with _meta showing 0 count
    """
    monkeypatch.setenv("FILTER_AI_EXTRACTION_NODES", "NonExistentNode")

    from n8n_langfuse_shipper.config import get_settings
    get_settings.cache_clear()

    trace = map_execution_to_langfuse(
        ai_execution_with_chat_and_tools, filter_ai_only=True
    )

    root_span = next(s for s in trace.spans if s.parent_id is None)
    assert "n8n.extracted_nodes" in root_span.metadata
    extracted = root_span.metadata["n8n.extracted_nodes"]

    # Only _meta present
    assert "_meta" in extracted
    assert extracted["_meta"]["extracted_count"] == 0
    assert len(extracted) == 1  # Just _meta


def test_extraction_preserves_multiple_runs(monkeypatch):
    """
    When node executed multiple times:
    - All runs extracted with correct indices
    """
    execution = N8nExecutionRecord(
        id=10000,
        workflowId="wf-multi-run",
        status="success",
        startedAt=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        stoppedAt=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        workflowData=WorkflowData(
            id="wf-multi-run",
            name="Multi Run Workflow",
            nodes=[
                WorkflowNode(name="AIChatNode", type="@n8n/langchain.lmchat"),
                WorkflowNode(name="LoopNode", type="n8n-nodes-base.function"),
            ],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "AIChatNode": [
                            NodeRun(
                                startTime=1704110400000,
                                executionTime=100,
                                executionStatus="success",
                                data={
                                    "main": [[{"json": {"response": "first"}}]],
                                    "ai_languageModel": [
                                        [{"json": {"tokenUsage": {"total": 10}}}]
                                    ],
                                },
                            )
                        ],
                        "LoopNode": [
                            NodeRun(
                                startTime=1704110401000,
                                executionTime=50,
                                executionStatus="success",
                                data={"main": [[{"json": {"iteration": 1}}]]},
                            ),
                            NodeRun(
                                startTime=1704110402000,
                                executionTime=50,
                                executionStatus="success",
                                data={"main": [[{"json": {"iteration": 2}}]]},
                            ),
                            NodeRun(
                                startTime=1704110403000,
                                executionTime=50,
                                executionStatus="success",
                                data={"main": [[{"json": {"iteration": 3}}]]},
                            ),
                        ],
                    }
                )
            )
        ),
    )

    monkeypatch.setenv("FILTER_AI_EXTRACTION_NODES", "LoopNode")

    from n8n_langfuse_shipper.config import get_settings
    get_settings.cache_clear()

    trace = map_execution_to_langfuse(execution, filter_ai_only=True)

    root_span = next(s for s in trace.spans if s.parent_id is None)
    extracted = root_span.metadata["n8n.extracted_nodes"]

    loop_data = extracted["LoopNode"]
    assert len(loop_data["runs"]) == 3

    # Check run indices and iteration values
    for i, run in enumerate(loop_data["runs"]):
        assert run["run_index"] == i
        assert run["output"]["iteration"] == i + 1


def test_extraction_metadata_structure(monkeypatch, ai_execution_with_chat_and_tools):
    """
    Verify complete metadata structure:
    - _meta section with counts and config
    - Each node with runs array and timestamps
    """
    monkeypatch.setenv("FILTER_AI_EXTRACTION_NODES", "WebScraperTool")
    monkeypatch.setenv("FILTER_AI_EXTRACTION_INCLUDE_KEYS", "url,temp*")
    monkeypatch.setenv("FILTER_AI_EXTRACTION_EXCLUDE_KEYS", "html")
    monkeypatch.setenv("FILTER_AI_EXTRACTION_MAX_VALUE_LEN", "5000")

    from n8n_langfuse_shipper.config import get_settings
    get_settings.cache_clear()

    trace = map_execution_to_langfuse(
        ai_execution_with_chat_and_tools, filter_ai_only=True
    )

    root_span = next(s for s in trace.spans if s.parent_id is None)
    extracted = root_span.metadata["n8n.extracted_nodes"]

    # Verify _meta structure
    meta = extracted["_meta"]
    assert meta["extracted_count"] == 1
    assert meta["filter_ai_only"] is True
    assert set(meta["extraction_nodes"]) == {"WebScraperTool"}
    assert set(meta["include_patterns"]) == {"url", "temp*"}
    assert set(meta["exclude_patterns"]) == {"html"}
    assert meta["max_value_len"] == 5000

    # Verify node structure
    node_data = extracted["WebScraperTool"]
    assert "runs" in node_data
    run = node_data["runs"][0]

    # Run should have required fields
    assert "run_index" in run
    assert "start_time" in run
    assert "execution_time_ms" in run
    assert "execution_status" in run
    assert "output" in run

    # Output filtered correctly
    assert "url" in run["output"]
    assert "temperature" in run["output"]
    assert "html" not in run["output"]  # excluded
    assert "condition" not in run["output"]  # not included


def test_extraction_with_empty_extraction_nodes_list(
    monkeypatch, ai_execution_with_chat_and_tools
):
    """
    When FILTER_AI_EXTRACTION_NODES is empty:
    - No extraction occurs
    - No n8n.extracted_nodes metadata
    """
    monkeypatch.setenv("FILTER_AI_EXTRACTION_NODES", "")

    from n8n_langfuse_shipper.config import get_settings
    get_settings.cache_clear()

    trace = map_execution_to_langfuse(
        ai_execution_with_chat_and_tools, filter_ai_only=True
    )

    root_span = next(s for s in trace.spans if s.parent_id is None)
    assert "n8n.extracted_nodes" not in root_span.metadata


def test_extraction_with_wildcard_node_patterns(monkeypatch):
    """
    Wildcard patterns in extraction_nodes not supported.
    Only exact node name matching.
    """
    execution = N8nExecutionRecord(
        id=10001,
        workflowId="wf-wildcard",
        status="success",
        startedAt=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        stoppedAt=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        workflowData=WorkflowData(
            id="wf-wildcard",
            name="Wildcard Test Workflow",
            nodes=[
                WorkflowNode(name="AIChatNode", type="@n8n/langchain.lmchat"),
                WorkflowNode(name="Tool1", type="n8n-nodes-base.httpRequest"),
                WorkflowNode(name="Tool2", type="n8n-nodes-base.httpRequest"),
            ],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "AIChatNode": [
                            NodeRun(
                                startTime=1704110400000,
                                executionTime=100,
                                executionStatus="success",
                                data={
                                    "main": [[{"json": {"response": "ok"}}]],
                                    "ai_languageModel": [
                                        [{"json": {"tokenUsage": {"total": 10}}}]
                                    ],
                                },
                            )
                        ],
                        "Tool1": [
                            NodeRun(
                                startTime=1704110401000,
                                executionTime=50,
                                executionStatus="success",
                                data={"main": [[{"json": {"data": "from tool1"}}]]},
                            )
                        ],
                        "Tool2": [
                            NodeRun(
                                startTime=1704110402000,
                                executionTime=50,
                                executionStatus="success",
                                data={"main": [[{"json": {"data": "from tool2"}}]]},
                            )
                        ],
                    }
                )
            )
        ),
    )

    # Try wildcard pattern (not supported - only exact names)
    monkeypatch.setenv("FILTER_AI_EXTRACTION_NODES", "Tool*")

    from n8n_langfuse_shipper.config import get_settings
    get_settings.cache_clear()

    trace = map_execution_to_langfuse(execution, filter_ai_only=True)

    root_span = next(s for s in trace.spans if s.parent_id is None)
    extracted = root_span.metadata.get("n8n.extracted_nodes", {})

    # No nodes extracted (wildcard not supported)
    assert extracted.get("_meta", {}).get("extracted_count", 0) == 0
    assert "Tool1" not in extracted
    assert "Tool2" not in extracted
