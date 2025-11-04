"""Tests for stripping System prompts from LangChain LMChat node inputs.

This module verifies that large message blobs containing both System and User
prompts are correctly split, with only the User portion being retained in the
Langfuse span input field.
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


def test_lmchat_strips_system_prompt_from_input():
    """Verify System prompt is stripped from lmChat generation input."""
    now = datetime.now(timezone.utc)

    # Simulate a typical lmChat input with System + User prompts combined
    system_part = (
        "System: # Role and Objective\n\n"
        "You are the **Primary Transport Management System Order Entry "
        "Specialist** for Kennis Transport & Logistics (KTL)."
    )
    user_part = (
        "Human: ## Order placed by Transpas Relational:\n\n"
        "Name: John Doe\nEmail: john@example.com\nPhone: 123-456-7890"
    )
    combined_message = (
        system_part + "\n\n## START PROCESSING\n\n" + user_part
    )

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
                                "generations": [[{"text": "Order processed"}]]
                            },
                            "tokenUsage": {
                                "promptTokens": 100,
                                "completionTokens": 20,
                                "totalTokens": 120,
                            },
                        }
                    }
                ]
            ]
        },
        inputOverride={
            "messages": [
                {"content": combined_message}
            ]
        },
    )

    rec = N8nExecutionRecord(
        id=12345,
        workflowId="wf-lmchat-test",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-lmchat-test",
            name="LMChat System Strip Test",
            nodes=[
                WorkflowNode(
                    name="GoogleVertexChat",
                    type="@n8n/n8n-nodes-langchain.lmChatGoogleVertex",
                )
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={"GoogleVertexChat": [lmchat_run]}
                )
            )
        ),
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    # Find the lmChat span
    lmchat_span = next(
        s for s in trace.spans if s.name == "GoogleVertexChat"
    )

    # Verify it's classified as generation
    assert lmchat_span.observation_type == "generation"

    # Verify input does NOT contain System prompt
    assert lmchat_span.input is not None
    assert "System: # Role and Objective" not in lmchat_span.input
    assert "Primary Transport Management System" not in lmchat_span.input

    # Verify input DOES contain User prompt WITHOUT "Human: " prefix
    assert "## Order placed by Transpas Relational" in lmchat_span.input
    assert "Name: John Doe" in lmchat_span.input

    # Verify "Human:" prefix was stripped
    assert "Human:" not in lmchat_span.input
    assert "John Doe" in lmchat_span.input


def test_lmchat_handles_missing_split_marker():
    """Verify graceful handling when split marker is absent."""
    now = datetime.now(timezone.utc)

    # Input without the split marker
    simple_message = "Just a simple user message without system prompt"

    lmchat_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=50,
        executionStatus="success",
        data={"result": "ok"},
        inputOverride={
            "messages": [
                {"content": simple_message}
            ]
        },
    )

    rec = N8nExecutionRecord(
        id=12346,
        workflowId="wf-lmchat-simple",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-lmchat-simple",
            name="LMChat Simple Test",
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

    # Find the lmChat span
    lmchat_span = next(
        s for s in trace.spans if s.name == "OpenAIChat"
    )

    # Verify input is unchanged (original message preserved)
    assert lmchat_span.input is not None
    assert "Just a simple user message" in lmchat_span.input


def test_non_lmchat_node_unaffected():
    """Verify non-lmChat nodes are not processed by strip function."""
    now = datetime.now(timezone.utc)

    # Use a split marker in a non-lmChat node type
    message_with_marker = (
        "Some data\n\n## START PROCESSING\n\nHuman: ## More data"
    )

    other_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=10,
        executionStatus="success",
        data={"output": "test"},
        inputOverride={
            "messages": [
                {"content": message_with_marker}
            ]
        },
    )

    rec = N8nExecutionRecord(
        id=12347,
        workflowId="wf-other",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-other",
            name="Other Node Test",
            nodes=[
                WorkflowNode(
                    name="OtherNode",
                    type="@n8n/n8n-nodes-base.someOtherNode",
                )
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={"OtherNode": [other_run]}
                )
            )
        ),
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    # Find the other span
    other_span = next(
        s for s in trace.spans if s.name == "OtherNode"
    )

    # Verify input is completely unchanged (includes full original text)
    assert other_span.input is not None
    assert "Some data" in other_span.input
    assert "Human: ## More data" in other_span.input


def test_lmchat_multiple_messages_with_marker():
    """Verify handling of multiple messages in array, some with markers."""
    now = datetime.now(timezone.utc)

    msg1 = "First message without marker"
    msg2_sys = "System: You are a helpful assistant"
    msg2_user = "Human: ## What is 2+2?"
    msg2_combined = msg2_sys + "\n\n## START PROCESSING\n\n" + msg2_user
    msg3 = "Third message without marker"

    lmchat_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=80,
        executionStatus="success",
        data={
            "tokenUsage": {"total": 50}
        },
        inputOverride={
            "messages": [
                {"content": msg1},
                {"content": msg2_combined},
                {"content": msg3},
            ]
        },
    )

    rec = N8nExecutionRecord(
        id=12348,
        workflowId="wf-lmchat-multi",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-lmchat-multi",
            name="LMChat Multi Message Test",
            nodes=[
                WorkflowNode(
                    name="AnthropicChat",
                    type="@n8n/n8n-nodes-langchain.lmChatAnthropic",
                )
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={"AnthropicChat": [lmchat_run]}
                )
            )
        ),
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    # Find the lmChat span
    lmchat_span = next(
        s for s in trace.spans if s.name == "AnthropicChat"
    )

    # Verify input exists
    assert lmchat_span.input is not None

    # Verify first and third messages unchanged
    assert "First message without marker" in lmchat_span.input
    assert "Third message without marker" in lmchat_span.input

    # Verify second message has System stripped and "Human:" prefix removed
    assert "System: You are a helpful assistant" not in lmchat_span.input
    assert "## What is 2+2?" in lmchat_span.input
    assert "Human:" not in lmchat_span.input


def test_lmchat_strips_system_from_string_messages():
    """Verify System prompt stripping when messages is list of strings."""
    now = datetime.now(timezone.utc)

    # Simulate messages as list of strings (not dicts with content key)
    # This matches the actual structure seen in Langfuse screenshot
    system_part = (
        "System: # Role and Objective\n\n"
        "You are the **Primary Transport Management System Order Entry "
        "Specialist** for Kennis Transport & Logistics (KTL)."
    )
    user_part = (
        "Human: ## Order placed by Transpas Relational:\n\n"
        "Name: John Doe\nEmail: john@example.com"
    )
    combined_message = (
        system_part + "\n\n## START PROCESSING\n\n" + user_part
    )

    lmchat_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=80,
        executionStatus="success",
        data={
            "tokenUsage": {"total": 120}
        },
        inputOverride={
            "messages": [combined_message]  # String directly, not dict
        },
    )

    rec = N8nExecutionRecord(
        id=12349,
        workflowId="wf-lmchat-str-msg",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-lmchat-str-msg",
            name="LMChat String Messages Test",
            nodes=[
                WorkflowNode(
                    name="GeminiChat",
                    type="@n8n/n8n-nodes-langchain.lmChatGemini",
                )
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={"GeminiChat": [lmchat_run]}
                )
            )
        ),
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    # Find the lmChat span
    lmchat_span = next(
        s for s in trace.spans if s.name == "GeminiChat"
    )

    # Verify input exists
    assert lmchat_span.input is not None

    # Verify System prompt is stripped
    assert "System: # Role and Objective" not in lmchat_span.input
    assert "Primary Transport Management System" not in lmchat_span.input

    # Verify User prompt is retained WITHOUT "Human: " prefix
    assert "## Order placed by Transpas Relational" in lmchat_span.input
    assert "Name: John Doe" in lmchat_span.input
    assert "Human:" not in lmchat_span.input
    assert "John Doe" in lmchat_span.input


def test_lmchat_strips_system_from_deeply_nested_structure():
    """Verify System prompt stripping in deeply nested structures.

    Real n8n data has messages deeply nested like:
    ai_languageModel[0][0]['json']['messages'][0]
    """
    now = datetime.now(timezone.utc)

    # Simulate deeply nested structure matching actual n8n data
    system_part = (
        "System: # Role and Objective\n\n"
        "You are the **Primary Transport Management System Order Entry "
        "Specialist** for Kennis Transport & Logistics (KTL)."
    )
    user_part = (
        "Human: ## Order placed by Transpas Relational:\n\n"
        "Name: John Doe\nEmail: john@example.com"
    )
    combined_message = (
        system_part + "\n\n## START PROCESSING\n\n" + user_part
    )

    lmchat_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=7240,
        executionStatus="success",
        data={
            "tokenUsage": {
                "promptTokens": 36415,
                "completionTokens": 569,
            }
        },
        inputOverride={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "messages": [combined_message],
                            "estimatedTokens": 34374,
                            "options": {
                                "auth_options": {"ic": 1, "type": "secret"}
                            },
                        }
                    }
                ]
            ]
        },
    )

    rec = N8nExecutionRecord(
        id=340345,
        workflowId="wf-deep-nested",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-deep-nested",
            name="Deep Nested LMChat Test",
            nodes=[
                WorkflowNode(
                    name="Google Vertex Chat Model",
                    type="@n8n/n8n-nodes-langchain.lmChatGoogleVertex",
                )
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={"Google Vertex Chat Model": [lmchat_run]}
                )
            )
        ),
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    # Find the lmChat span
    lmchat_span = next(
        s for s in trace.spans if s.name == "Google Vertex Chat Model"
    )

    # Verify input exists
    assert lmchat_span.input is not None

    # Verify System prompt is stripped (even though deeply nested)
    assert "System: # Role and Objective" not in lmchat_span.input
    assert "Primary Transport Management System" not in lmchat_span.input

    # Verify User prompt is retained WITHOUT "Human: " prefix
    assert "## Order placed by Transpas Relational" in lmchat_span.input
    assert "Name: John Doe" in lmchat_span.input
    assert "Human:" not in lmchat_span.input
    assert "John Doe" in lmchat_span.input
