"""Test prompt detection patterns for HTTP and Langfuse nodes.

Tests detection of nodes that fetch Langfuse prompts via:
1. HTTP Request nodes (GET /api/public/prompts/{name})
2. @n8n/n8n-nodes-langchain.promptLangfuse official nodes

Validates metadata extraction from various output schemas and edge cases.
"""
from __future__ import annotations

import pytest

from n8n_langfuse_shipper.mapping.prompt_detection import (
    PromptMetadata,
    PromptSourceInfo,
    detect_prompt_fetch_node,
    build_prompt_registry,
)


def test_detect_http_request_prompt_fetch():
    """Verify HTTP Request node detection via API pattern and output."""
    output_data = {
        "main": [[{
            "json": {
                "id": "cm3abc123",
                "name": "Sales Agent Orchestrator",
                "version": 58,
                "type": "chat",
                "labels": ["sales", "production"],
                "createdAt": "2024-01-15T10:00:00.000Z",
            }
        }]]
    }

    parameters = {
        "url": "https://cloud.langfuse.com/api/public/prompts/sales-agent",
        "method": "GET",
    }

    result = detect_prompt_fetch_node(
        node_name="Fetch Agent Prompt",
        node_type="n8n-nodes-base.httpRequest",
        node_parameters=parameters,
        run_index=0,
        run_output=output_data,
    )

    assert result is not None
    assert result.prompt_metadata.name == "Sales Agent Orchestrator"
    assert result.prompt_metadata.version == 58
    assert result.prompt_metadata.type == "chat"
    assert result.prompt_metadata.labels == ["sales", "production"]
    assert result.detection_method == "http_api"


def test_detect_langfuse_official_node():
    """Verify official Langfuse prompt node detection."""
    output_data = {
        "main": [[{
            "json": {
                "name": "Marketing Campaign Generator",
                "version": 15,
                "type": "text",
                "labels": ["marketing", "dev"],
            }
        }]]
    }

    result = detect_prompt_fetch_node(
        node_name="Load System Prompt",
        node_type="@n8n/n8n-nodes-langchain.promptLangfuse",
        node_parameters=None,
        run_index=0,
        run_output=output_data,
    )

    assert result is not None
    assert result.prompt_metadata.name == "Marketing Campaign Generator"
    assert result.prompt_metadata.version == 15
    assert result.prompt_metadata.type == "text"
    assert result.prompt_metadata.labels == ["marketing", "dev"]
    assert result.detection_method == "node_type"


def test_detect_non_prompt_http_request():
    """Verify non-prompt HTTP requests are not detected."""
    output_data = {
        "main": [[{
            "json": {
                "userId": 123,
                "email": "user@example.com",
                "name": "John Doe",  # name field but not prompt schema
            }
        }]]
    }

    parameters = {
        "url": "https://api.example.com/users",
        "method": "GET",
    }

    result = detect_prompt_fetch_node(
        node_name="Fetch User Data",
        node_type="n8n-nodes-base.httpRequest",
        node_parameters=parameters,
        run_index=0,
        run_output=output_data,
    )

    assert result is None


def test_detect_missing_required_fields():
    """Verify detection fails gracefully when required fields missing."""
    output_data = {
        "main": [[{
            "json": {
                # Missing name and version
                "type": "chat",
                "labels": ["test"],
            }
        }]]
    }

    result = detect_prompt_fetch_node(
        node_name="Broken Prompt Fetch",
        node_type="@n8n/n8n-nodes-langchain.promptLangfuse",
        node_parameters=None,
        run_index=0,
        run_output=output_data,
    )

    assert result is None


def test_detect_empty_output():
    """Verify detection handles empty node outputs."""
    result = detect_prompt_fetch_node(
        node_name="Empty Response",
        node_type="n8n-nodes-base.httpRequest",
        node_parameters={"url": "https://cloud.langfuse.com/api/public/prompts/test"},
        run_index=0,
        run_output={},
    )

    assert result is None


def test_detect_nested_json_output():
    """Verify detection handles nested JSON wrappers."""
    output_data = {
        "main": [[{
            "json": {
                "response": {
                    "data": {
                        "name": "Deeply Nested Prompt",
                        "version": 42,
                        "type": "chat",
                    }
                }
            }
        }]]
    }

    # Should NOT find because extraction only checks direct json wrapper
    result = detect_prompt_fetch_node(
        node_name="Nested Prompt Fetch",
        node_type="n8n-nodes-base.httpRequest",
        node_parameters={"url": "https://cloud.langfuse.com/api/public/prompts/nested"},
        run_index=0,
        run_output=output_data,
    )

    # Expected to return None since extraction is not deeply recursive
    assert result is None


def test_detect_optional_labels_absent():
    """Verify detection succeeds when optional labels field absent."""
    output_data = {
        "main": [[{
            "json": {
                "name": "Simple Prompt",
                "version": 1,
                # No type or labels
            }
        }]]
    }

    result = detect_prompt_fetch_node(
        node_name="Minimal Prompt",
        node_type="@n8n/n8n-nodes-langchain.promptLangfuse",
        node_parameters=None,
        run_index=0,
        run_output=output_data,
    )

    assert result is not None
    assert result.prompt_metadata.name == "Simple Prompt"
    assert result.prompt_metadata.version == 1
    assert result.prompt_metadata.type is None
    assert result.prompt_metadata.labels == []


def test_detect_version_as_string():
    """Verify version coercion when provided as string."""
    output_data = {
        "main": [[{
            "json": {
                "name": "Test Prompt",
                "version": "25",  # String instead of int
                "type": "chat",
            }
        }]]
    }

    result = detect_prompt_fetch_node(
        node_name="String Version",
        node_type="@n8n/n8n-nodes-langchain.promptLangfuse",
        node_parameters=None,
        run_index=0,
        run_output=output_data,
    )

    assert result is not None
    assert result.prompt_metadata.name == "Test Prompt"
    assert result.prompt_metadata.version == 25  # Should be coerced to int


def test_build_prompt_registry_single_prompt():
    """Verify registry building with single prompt fetch."""
    run_data = {
        "Fetch Prompt": [{
            "startTime": 1000,
            "executionTime": 50,
            "executionStatus": "success",
            "data": {
                "main": [[{
                    "json": {
                        "name": "Test Prompt",
                        "version": 10,
                        "type": "chat",
                    }
                }]]
            },
        }],
        "LLM Agent": [{
            "startTime": 2000,
            "executionTime": 500,
            "executionStatus": "success",
            "data": {"main": [[{"json": {"output": "result"}}]]},
        }],
    }

    workflow_nodes = [
        {
            "name": "Fetch Prompt",
            "type": "@n8n/n8n-nodes-langchain.promptLangfuse",
            "parameters": {},
        },
        {
            "name": "LLM Agent",
            "type": "@n8n/n8n-nodes-langchain.agent",
            "parameters": {},
        },
    ]

    registry = build_prompt_registry(run_data, workflow_nodes)

    assert len(registry) == 1
    assert ("Fetch Prompt", 0) in registry
    assert registry[("Fetch Prompt", 0)].name == "Test Prompt"
    assert registry[("Fetch Prompt", 0)].version == 10


def test_build_prompt_registry_no_prompts():
    """Verify empty registry when no prompt fetches detected."""
    run_data = {
        "Start": [{
            "startTime": 1000,
            "executionTime": 10,
            "executionStatus": "success",
            "data": {},
        }],
        "Agent": [{
            "startTime": 2000,
            "executionTime": 500,
            "executionStatus": "success",
            "data": {"main": [[{"json": {"output": "response"}}]]},
        }],
    }

    workflow_nodes = [
        {"name": "Start", "type": "n8n-nodes-base.start", "parameters": {}},
        {"name": "Agent", "type": "@n8n/n8n-nodes-langchain.agent", "parameters": {}},
    ]

    registry = build_prompt_registry(run_data, workflow_nodes)

    assert len(registry) == 0


def test_detect_output_schema_fallback():
    """Verify detection via output schema when node type not recognized."""
    output_data = {
        "main": [[{
            "json": {
                "name": "Fallback Detected Prompt",
                "version": 99,
                "type": "chat",
                "labels": ["test"],
            }
        }]]
    }

    # Unknown node type but valid prompt output
    result = detect_prompt_fetch_node(
        node_name="Custom Prompt Fetch",
        node_type="n8n-nodes-custom.promptFetcher",
        node_parameters={},
        run_index=0,
        run_output=output_data,
    )

    assert result is not None
    assert result.prompt_metadata.name == "Fallback Detected Prompt"
    assert result.prompt_metadata.version == 99
    assert result.detection_method == "output_schema"
