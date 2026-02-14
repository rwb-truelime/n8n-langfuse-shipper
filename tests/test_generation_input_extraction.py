"""Tests for extracting clean input and LLM parameters from generation nodes.

This module verifies that generation spans with messages arrays correctly extract:
1. Clean input (messages[0] only) for the Langfuse input field
2. LLM configuration parameters as metadata (n8n.llm.* keys)
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


def test_generation_input_extraction_with_llm_params():
    """Verify generation input extracts messages array and LLM params to metadata."""
    now = datetime.now(timezone.utc)

    # Typical lmChat input with messages + LLM configuration
    user_message = {"content": "What is the weather like?"}

    lmchat_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=150,
        executionStatus="success",
        data={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "response": {
                                "generations": [[{"text": "It's sunny today."}]]
                            },
                            "tokenUsage": {
                                "promptTokens": 10,
                                "completionTokens": 5,
                                "totalTokens": 15,
                            },
                        }
                    }
                ]
            ]
        },
        inputOverride={
            "messages": [user_message],
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 60000,
            "max_retries": 3,
            "configuration": {"retry_on_429": True},
            "model_kwargs": {"top_p": 0.9},
            "response_format": {"type": "json_object"},
        },
    )

    rec = N8nExecutionRecord(
        id=12345,
        workflowId="wf-gen-extract",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-gen-extract",
            name="Generation Input Extraction Test",
            nodes=[
                WorkflowNode(
                    name="OpenAIChat",
                    type="@n8n/n8n-nodes-langchain.lmChatOpenAI",
                )
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={"OpenAIChat": [lmchat_run]}
                )
            )
        ),
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    # Find the generation span
    gen_span = next(
        s for s in trace.spans if s.name == "OpenAIChat #0"
    )

    # Verify it's classified as generation
    assert gen_span.observation_type == "generation"

    # Verify input contains the messages array (serialized)
    assert gen_span.input is not None
    # Input should be serialized messages array
    assert "What is the weather like?" in gen_span.input

    # Verify input does NOT contain LLM configuration parameters at the same level
    assert "max_tokens" not in gen_span.input
    assert "temperature" not in gen_span.input
    assert "timeout" not in gen_span.input
    assert "max_retries" not in gen_span.input
    # These would appear if the full object was serialized:
    assert '"max_tokens": 1000' not in gen_span.input
    assert '"temperature": 0.7' not in gen_span.input

    # Verify LLM parameters are in metadata with n8n.llm.* prefix
    assert "n8n.llm.max_tokens" in gen_span.metadata
    assert gen_span.metadata["n8n.llm.max_tokens"] == 1000

    assert "n8n.llm.temperature" in gen_span.metadata
    assert gen_span.metadata["n8n.llm.temperature"] == 0.7

    assert "n8n.llm.timeout" in gen_span.metadata
    assert gen_span.metadata["n8n.llm.timeout"] == 60000

    assert "n8n.llm.max_retries" in gen_span.metadata
    assert gen_span.metadata["n8n.llm.max_retries"] == 3

    assert "n8n.llm.configuration" in gen_span.metadata
    assert gen_span.metadata["n8n.llm.configuration"] == {"retry_on_429": True}

    assert "n8n.llm.model_kwargs" in gen_span.metadata
    assert gen_span.metadata["n8n.llm.model_kwargs"] == {"top_p": 0.9}

    assert "n8n.llm.response_format" in gen_span.metadata
    assert gen_span.metadata["n8n.llm.response_format"] == {"type": "json_object"}


def test_generation_without_messages_array_unchanged():
    """Verify nodes without messages array are unaffected."""
    now = datetime.now(timezone.utc)

    # Generation node without messages array structure
    other_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=100,
        executionStatus="success",
        data={
            "tokenUsage": {"total": 20}
        },
        inputOverride={
            "prompt": "What is 2+2?",
            "max_tokens": 100,
        },
    )

    rec = N8nExecutionRecord(
        id=12346,
        workflowId="wf-no-messages",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-no-messages",
            name="No Messages Array Test",
            nodes=[
                WorkflowNode(
                    name="CustomLLM",
                    type="CustomLLMNode",
                )
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={"CustomLLM": [other_run]}
                )
            )
        ),
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    # Find the span
    span = next(
        s for s in trace.spans if s.name == "CustomLLM #0"
    )

    # Verify it's classified as generation
    assert span.observation_type == "generation"

    # Verify input contains the original structure (unchanged)
    assert span.input is not None
    assert "prompt" in span.input
    assert "What is 2+2?" in span.input
    assert "max_tokens" in span.input

    # No LLM parameter extraction should have occurred
    assert "n8n.llm.prompt" not in span.metadata
    assert "n8n.llm.max_tokens" not in span.metadata


def test_non_generation_node_unaffected():
    """Verify non-generation nodes don't trigger LLM param extraction."""
    now = datetime.now(timezone.utc)

    # Non-generation node with messages-like structure
    other_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=50,
        executionStatus="success",
        data={"result": "ok"},
        inputOverride={
            "messages": [{"content": "Some data"}],
            "timeout": 5000,
        },
    )

    rec = N8nExecutionRecord(
        id=12347,
        workflowId="wf-non-gen",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-non-gen",
            name="Non-Generation Test",
            nodes=[
                WorkflowNode(
                    name="HTTPRequest",
                    type="n8n-nodes-base.httpRequest",
                )
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={"HTTPRequest": [other_run]}
                )
            )
        ),
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    # Find the span
    span = next(
        s for s in trace.spans if s.name == "HTTPRequest #0"
    )

    # Verify it's NOT classified as generation
    assert span.observation_type != "generation"


    # Verify input contains the original structure (no extraction)
    assert span.input is not None
    assert "messages" in span.input
    assert "timeout" in span.input

    # No LLM parameters in metadata
    assert "n8n.llm.timeout" not in span.metadata


def test_recursive_search_with_nested_structure():
    """Verify recursive search finds messages dict at depth 4 with options flattening."""
    now = datetime.now(timezone.utc)

    # Realistic production structure: ai_languageModel wrapping at multiple levels
    user_message = {"content": "What is Python?"}

    lmchat_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=200,
        executionStatus="success",
        data={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "response": {
                                "generations": [[{"text": "Python is a language."}]]
                            },
                            "tokenUsage": {
                                "promptTokens": 15,
                                "completionTokens": 8,
                                "totalTokens": 23,
                            },
                        }
                    }
                ]
            ]
        },
        inputOverride={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "messages": [user_message],
                            "estimatedTokens": 16327,
                            "options": {
                                "max_tokens": 4000,
                                "temperature": 0.7,
                                "timeout": 60000,
                                "max_retries": 2,
                                "configuration": {},
                                "model_kwargs": {
                                    "response_format": {"type": "json_object"}
                                },
                            },
                        }
                    }
                ]
            ]
        },
    )

    # Build execution
    workflow_data = WorkflowData(
        id="test-wf-recursive",
        name="Recursive Search Test",
        nodes=[
            WorkflowNode(
                name="LMChat",
                type="n8n-nodes-langchain.lmChatAzureOpenAi",
            )
        ],
        connections={},
    )

    execution = N8nExecutionRecord(
        id=999,
        workflowId="test-wf",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=workflow_data,
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"LMChat": [lmchat_run]})
            )
        ),
    )

    # Map to Langfuse
    trace = map_execution_to_langfuse(execution)

    # Find LMChat span
    span = next((s for s in trace.spans if s.name == "LMChat #0"), None)
    assert span is not None
    assert span.observation_type == "generation"

    # Verify input contains ONLY messages array (clean extraction)
    assert span.input is not None
    assert isinstance(span.input, str)
    assert "What is Python?" in span.input

    # Verify LLM params are NOT in the input string
    assert "max_tokens" not in span.input
    assert "temperature" not in span.input
    assert "timeout" not in span.input
    assert "estimatedTokens" not in span.input
    assert "options" not in span.input

    # Verify all LLM parameters in metadata with flattened options
    assert span.metadata["n8n.llm.estimatedTokens"] == 16327
    assert span.metadata["n8n.llm.max_tokens"] == 4000
    assert span.metadata["n8n.llm.temperature"] == 0.7
    assert span.metadata["n8n.llm.timeout"] == 60000
    assert span.metadata["n8n.llm.max_retries"] == 2
    assert span.metadata["n8n.llm.configuration"] == {}
    assert span.metadata["n8n.llm.model_kwargs"] == {
        "response_format": {"type": "json_object"}
    }

    # Verify options dict was NOT stored as nested key
    assert "n8n.llm.options" not in span.metadata
