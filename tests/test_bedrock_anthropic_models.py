"""Comprehensive tests for AWS Bedrock Anthropic model support.

Tests generation detection, model extraction, and token usage for:
- Claude 3.5 Sonnet v2 (anthropic.claude-3-5-sonnet-20241022-v2:0)
- Claude Sonnet 4.5 (anthropic.claude-sonnet-4-5-20250929-v1:0)
- Claude Opus 4.6 (anthropic.claude-opus-4-6-v1)

Validates that the shipper correctly identifies and processes Anthropic models
running on AWS Bedrock infrastructure.
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


def _build_bedrock_record(
    node_name: str,
    node_type: str,
    model_id: str,
    include_token_usage: bool = True,
    include_output: bool = True,
    parameters: dict | None = None,
):
    """Build test execution record for Bedrock Anthropic node.
    
    Args:
        node_name: Name of the test node
        node_type: Node type identifier (e.g., @n8n/n8n-nodes-langchain.lmChatAwsBedrock)
        model_id: Bedrock model identifier (e.g., anthropic.claude-3-5-sonnet-20241022-v2:0)
        include_token_usage: Whether to include tokenUsage in output
        include_output: Whether to include response output
        parameters: Optional static parameters dict for the workflow node
    
    Returns:
        N8nExecutionRecord with simulated Bedrock Anthropic execution
    """
    now = datetime.now(timezone.utc)
    starter_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=5,
        executionStatus="success",
        data={"main": [[{"json": {"input": "test prompt"}}]]},
    )
    
    # Build output data structure mimicking real Bedrock responses
    output_data: dict = {}
    if include_output:
        output_data["response"] = {
            "generations": [[{"text": "This is a test response from Claude."}]],
        }
    
    if include_token_usage:
        output_data["tokenUsage"] = {
            "promptTokens": 15,
            "completionTokens": 25,
            "totalTokens": 40,
        }
    
    # Add model identifier to output (Bedrock includes this in response metadata)
    output_data["model"] = model_id
    
    bedrock_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 10,
        executionTime=1500,
        executionStatus="success",
        data=output_data,
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    
    runData = {"Starter": [starter_run], node_name: [bedrock_run]}
    
    # Build workflow node with optional parameters
    workflow_node = WorkflowNode(
        name=node_name,
        type=node_type,
        parameters=parameters or {},
    )
    
    rec = N8nExecutionRecord(
        id=1000,
        workflowId="wf-bedrock-anthropic",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-bedrock-anthropic",
            name="Bedrock Anthropic Test",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                workflow_node,
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))
        ),
    )
    return rec


def test_bedrock_claude_3_5_sonnet_v2_generation_detection():
    """Test generation detection for Claude 3.5 Sonnet v2 on Bedrock."""
    rec = _build_bedrock_record(
        node_name="Claude 3.5 Sonnet v2",
        node_type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    
    span = next(s for s in trace.spans if s.name == "Claude 3.5 Sonnet v2")
    assert span.observation_type == "generation", (
        "Claude 3.5 Sonnet v2 on Bedrock should be classified as generation"
    )
    assert span.usage is not None, "Token usage should be extracted"
    assert span.usage.input == 15
    assert span.usage.output == 25
    assert span.usage.total == 40


def test_bedrock_claude_sonnet_4_5_generation_detection():
    """Test generation detection for Claude Sonnet 4.5 on Bedrock."""
    rec = _build_bedrock_record(
        node_name="Claude Sonnet 4.5",
        node_type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
        model_id="anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    
    span = next(s for s in trace.spans if s.name == "Claude Sonnet 4.5")
    assert span.observation_type == "generation", (
        "Claude Sonnet 4.5 on Bedrock should be classified as generation"
    )
    assert span.usage is not None, "Token usage should be extracted"
    assert span.usage.input == 15
    assert span.usage.output == 25
    assert span.usage.total == 40


def test_bedrock_claude_opus_4_6_generation_detection():
    """Test generation detection for Claude Opus 4.6 on Bedrock."""
    rec = _build_bedrock_record(
        node_name="Claude Opus 4.6",
        node_type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
        model_id="anthropic.claude-opus-4-6-v1",
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    
    span = next(s for s in trace.spans if s.name == "Claude Opus 4.6")
    assert span.observation_type == "generation", (
        "Claude Opus 4.6 on Bedrock should be classified as generation"
    )
    assert span.usage is not None, "Token usage should be extracted"


def test_bedrock_model_extraction_claude_3_5_sonnet_v2():
    """Test model name extraction for Claude 3.5 Sonnet v2."""
    rec = _build_bedrock_record(
        node_name="Test Node",
        node_type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    
    span = next(s for s in trace.spans if s.name == "Test Node")
    assert span.model == "anthropic.claude-3-5-sonnet-20241022-v2:0", (
        "Model ID should be extracted from Bedrock response"
    )


def test_bedrock_model_extraction_claude_sonnet_4_5():
    """Test model name extraction for Claude Sonnet 4.5."""
    rec = _build_bedrock_record(
        node_name="Test Node",
        node_type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
        model_id="anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    
    span = next(s for s in trace.spans if s.name == "Test Node")
    assert span.model == "anthropic.claude-sonnet-4-5-20250929-v1:0", (
        "Model ID should be extracted from Bedrock response"
    )


def test_bedrock_model_extraction_claude_opus_4_6():
    """Test model name extraction for Claude Opus 4.6."""
    rec = _build_bedrock_record(
        node_name="Test Node",
        node_type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
        model_id="anthropic.claude-opus-4-6-v1",
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    
    span = next(s for s in trace.spans if s.name == "Test Node")
    assert span.model == "anthropic.claude-opus-4-6-v1", (
        "Model ID should be extracted from Bedrock response"
    )


def test_bedrock_model_from_parameters():
    """Test model extraction from static parameters when not in runtime data."""
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    now = datetime.now(timezone.utc)

    starter_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=5,
        executionStatus="success",
        data={"main": [[{"json": {"input": "test prompt"}}]]},
    )

    # Runtime data intentionally omits any "model" field to force parameter fallback
    bedrock_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 10,
        executionTime=1500,
        executionStatus="success",
        data={
            "response": {"generations": [[{"text": "Test response"}]]},
            "tokenUsage": {
                "promptTokens": 10,
                "completionTokens": 20,
                "totalTokens": 30,
            },
        },
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )

    workflow = WorkflowData(
        id="1",
        name="Test Workflow",
        nodes=[
            WorkflowNode(
                name="Starter",
                type="n8n-nodes-base.start",
            ),
            WorkflowNode(
                name="Test Node",
                type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
                parameters={
                    "model": model_id,
                    "region": "us-east-1",
                },
            ),
        ],
        connections={
            "Starter": {
                "main": [[{"node": "Test Node", "type": "main", "index": 0}]],
            },
        },
    )

    execution_data = ExecutionData(
        executionData=ExecutionDataDetails(
            resultData=ResultData(
                runData={
                    "Starter": [starter_run],
                    "Test Node": [bedrock_run],
                },
            ),
        ),
    )

    rec = N8nExecutionRecord(
        id=1,
        workflowId="wf1",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=workflow,
        data=execution_data,
    )

    trace = map_execution_to_langfuse(rec, truncate_limit=None)

    span = next(s for s in trace.spans if s.name == "Test Node")
    # Model should be extracted from parameters as fallback
    assert span.model == model_id, "Model should be extracted from parameters"
    assert span.metadata.get("n8n.model.from_parameters") is True
    assert span.metadata.get("n8n.model.parameter_key") == "parameters.model"


def test_bedrock_model_from_modelSource_parameter():
    """Test model extraction with real n8n Bedrock structure (both model and modelSource)."""
    # Real n8n v2.6.4 Bedrock node structure where:
    # - parameters.modelSource = "onDemand" (UI selector)
    # - parameters.model = actual model ID
    now = datetime.now(timezone.utc)
    starter_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=5,
        executionStatus="success",
        data={"main": [[{"json": {"input": "test prompt"}}]]},
    )
    
    # Runtime data without model field (forcing parameter fallback)
    bedrock_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 10,
        executionTime=1500,
        executionStatus="success",
        data={
            "response": {"generations": [[{"text": "Test response"}]]},
            "tokenUsage": {"promptTokens": 10, "completionTokens": 20, "totalTokens": 30},
        },
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    
    runData = {"Starter": [starter_run], "Bedrock LLM": [bedrock_run]}
    
    # Real n8n Bedrock node parameter structure (n8n v2.6.4)
    rec = N8nExecutionRecord(
        id=2001,
        workflowId="wf-bedrock-modelsource",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-bedrock-modelsource",
            name="Bedrock ModelSource Test",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(
                    name="Bedrock LLM",
                    type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
                    parameters={
                        "modelSource": "onDemand",  # UI selector value
                        "model": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",  # Actual model ID
                        "region": "us-east-1",
                    },
                ),
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))
        ),
    )
    
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "Bedrock LLM")
    
    # Verify model extracted from parameters.model (NOT parameters.modelSource="onDemand")
    assert span.model == "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", (
        f"Expected model from parameters.model, got {span.model}"
    )
    assert span.metadata.get("n8n.model.from_parameters") is True, (
        "Should have n8n.model.from_parameters metadata"
    )
    param_key = span.metadata.get("n8n.model.parameter_key")
    assert param_key == "parameters.model", (
        "Expected parameter_key to be parameters.model, got "
        f"{param_key}"
    )


def test_bedrock_generation_without_explicit_token_usage():
    """Test that Bedrock nodes are still classified as generation by type marker."""
    # Build record without tokenUsage - should still be detected via 'bedrock' marker
    now = datetime.now(timezone.utc)
    starter_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=5,
        executionStatus="success",
        data={"main": [[{"json": {"input": "test"}}]]},
    )
    
    bedrock_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 10,
        executionTime=1200,
        executionStatus="success",
        data={"response": {"generations": [[{"text": "response"}]]}},
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    
    runData = {"Starter": [starter_run], "Bedrock Node": [bedrock_run]}
    rec = N8nExecutionRecord(
        id=1001,
        workflowId="wf-bedrock-no-usage",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-bedrock-no-usage",
            name="Bedrock Without Usage",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(
                    name="Bedrock Node",
                    type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
                ),
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))
        ),
    )
    
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "Bedrock Node")
    assert span.observation_type == "generation", (
        "Bedrock nodes should be classified as generation even without tokenUsage"
    )


def test_bedrock_anthropic_with_nested_ai_channel():
    """Test Bedrock Anthropic with nested ai_languageModel channel structure."""
    now = datetime.now(timezone.utc)
    starter_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=5,
        executionStatus="success",
        data={"main": [[{"json": {"input": "test"}}]]},
    )
    
    # Simulate nested ai_languageModel channel output
    bedrock_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 10,
        executionTime=1300,
        executionStatus="success",
        data={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "response": {"generations": [[{"text": "Nested response"}]]},
                            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                            "tokenUsage": {
                                "promptTokens": 20,
                                "completionTokens": 30,
                                "totalTokens": 50,
                            },
                        }
                    }
                ]
            ]
        },
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    
    runData = {"Starter": [starter_run], "Nested Bedrock": [bedrock_run]}
    rec = N8nExecutionRecord(
        id=1002,
        workflowId="wf-bedrock-nested",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-bedrock-nested",
            name="Bedrock Nested",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(
                    name="Nested Bedrock",
                    type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
                ),
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))
        ),
    )
    
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "Nested Bedrock")
    assert span.observation_type == "generation"
    assert span.usage is not None
    assert span.usage.input == 20
    assert span.usage.output == 30
    assert span.usage.total == 50
    assert span.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_bedrock_multiple_anthropic_models_in_workflow():
    """Test workflow with multiple different Bedrock Anthropic models."""
    now = datetime.now(timezone.utc)
    starter_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=5,
        executionStatus="success",
        data={"main": [[{"json": {"input": "start"}}]]},
    )
    
    # Claude 3.5 Sonnet v2
    claude_35_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 10,
        executionTime=1000,
        executionStatus="success",
        data={
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "tokenUsage": {"promptTokens": 10, "completionTokens": 20, "totalTokens": 30},
        },
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    
    # Claude Sonnet 4.5
    claude_45_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 1020,
        executionTime=1100,
        executionStatus="success",
        data={
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "tokenUsage": {"promptTokens": 15, "completionTokens": 35, "totalTokens": 50},
        },
        source=[NodeRunSource(previousNode="Claude 3.5 Sonnet v2", previousNodeRun=0)],
    )
    
    # Claude Opus 4.6
    claude_46_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 2130,
        executionTime=1500,
        executionStatus="success",
        data={
            "model": "anthropic.claude-opus-4-6-v1",
            "tokenUsage": {"promptTokens": 20, "completionTokens": 80, "totalTokens": 100},
        },
        source=[NodeRunSource(previousNode="Claude Sonnet 4.5", previousNodeRun=0)],
    )
    
    runData = {
        "Starter": [starter_run],
        "Claude 3.5 Sonnet v2": [claude_35_run],
        "Claude Sonnet 4.5": [claude_45_run],
        "Claude Opus 4.6": [claude_46_run],
    }
    
    rec = N8nExecutionRecord(
        id=1003,
        workflowId="wf-bedrock-multi",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-bedrock-multi",
            name="Multi Bedrock Models",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(
                    name="Claude 3.5 Sonnet v2",
                    type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
                ),
                WorkflowNode(
                    name="Claude Sonnet 4.5",
                    type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
                ),
                WorkflowNode(
                    name="Claude Opus 4.6",
                    type="@n8n/n8n-nodes-langchain.lmChatAwsBedrock",
                ),
            ],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))
        ),
    )
    
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    
    # Verify all three models are correctly detected and processed
    claude_35_span = next(s for s in trace.spans if s.name == "Claude 3.5 Sonnet v2")
    assert claude_35_span.observation_type == "generation"
    assert claude_35_span.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert claude_35_span.usage.total == 30
    
    claude_45_span = next(s for s in trace.spans if s.name == "Claude Sonnet 4.5")
    assert claude_45_span.observation_type == "generation"
    assert claude_45_span.model == "anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert claude_45_span.usage.total == 50
    
    claude_46_span = next(s for s in trace.spans if s.name == "Claude Opus 4.6")
    assert claude_46_span.observation_type == "generation"
    assert claude_46_span.model == "anthropic.claude-opus-4-6-v1"
    assert claude_46_span.usage.total == 100
