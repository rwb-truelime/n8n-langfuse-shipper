from __future__ import annotations

from datetime import datetime, timezone
from src.mapper import map_execution_to_langfuse
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


def _build_record(node_pairs):
    now = datetime.now(timezone.utc)
    starter_run = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=5,
        executionStatus="success",
        data={"x": 1},
    )
    runData = {"Starter": [starter_run]}
    nodes = [WorkflowNode(name="Starter", type="ToolWorkflow")]
    offset = 5
    for name, type_name in node_pairs:
        runData[name] = [
            NodeRun(
                startTime=int(now.timestamp() * 1000) + offset,
                executionTime=7,
                executionStatus="success",
                data={"model": type_name.lower()},
                source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
            )
        ]
        offset += 5
        nodes.append(WorkflowNode(name=name, type=type_name))
    rec = N8nExecutionRecord(
        id=700,
        workflowId="wf-gen-matrix",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(id="wf-gen-matrix", name="Gen Matrix", nodes=nodes),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    return rec


def test_generation_detection_provider_matrix():
    # Each tuple: (node name, node type) covering provider markers without tokenUsage
    providers = [
        ("AnthropicNode", "AnthropicChat"),
        ("OpenAiNode", "OpenAi"),
        ("GeminiNode", "GoogleGemini"),
        ("MistralNode", "MistralCloud"),
        ("GroqNode", "Groq"),
        ("CohereNode", "CohereChat"),
        ("DeepSeekNode", "DeepSeek"),
        ("OllamaNode", "LmChatOllama"),
        ("OpenRouterNode", "LmChatOpenRouter"),
        ("BedrockNode", "LmChatAwsBedrock"),
        ("VertexNode", "LmChatGoogleVertex"),
        ("HuggingFaceNode", "LmOpenHuggingFaceInference"),
        ("XaiNode", "LmChatXAiGrok"),
    ]
    rec = _build_record(providers)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    names = {p[0] for p in providers}
    for span in trace.spans:
        if span.name in names:
            assert span.observation_type == "generation", f"Expected generation for {span.name}/{span.metadata.get('n8n.node.type')}"


def test_generation_excludes_embeddings():
    rec = _build_record([("EmbeddingsNode", "EmbeddingsOpenAI")])
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "EmbeddingsNode")
    assert span.observation_type != "generation", "Embeddings should not be classified as generation"


def test_generation_detection_nested_token_usage():
    # Simulate n8n nested output channel wrapping tokenUsage
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    nested_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 10,
        executionTime=9,
        executionStatus="success",
        data={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "response": {"generations": [[{"text": "Hello world"}]]},
                            "tokenUsage": {"promptTokens": 12, "completionTokens": 3, "totalTokens": 15},
                        }
                    }
                ]
            ]
        },
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    starter = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=1,
        executionStatus="success",
        data={"main": [[{"json": {"ok": True}}]]},
    )
    runData = {"Starter": [starter], "NestedModel": [nested_run]}
    rec = N8nExecutionRecord(
        id=905,
        workflowId="wf-nested-gen",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-nested-gen",
            name="Nested Gen",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(name="NestedModel", type="GoogleGemini"),
            ],
        ),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "NestedModel")
    assert span.observation_type == "generation", "Nested tokenUsage not detected"
    assert span.usage is not None
    assert span.usage.input == 12 and span.usage.output == 3 and span.usage.total == 15


def test_nested_model_extraction():
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    nested_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 5,
        executionTime=4,
        executionStatus="success",
        data={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "model": "gpt-4o-mini",
                            "tokenUsage": {"promptTokens": 2, "completionTokens": 1, "totalTokens": 3},
                        }
                    }
                ]
            ]
        },
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    starter = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=1,
        executionStatus="success",
        data={"main": [[{"json": {"ok": True}}]]},
    )
    runData = {"Starter": [starter], "NestedModel": [nested_run]}
    rec = N8nExecutionRecord(
        id=906,
        workflowId="wf-nested-model",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-nested-model",
            name="Nested Model",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(name="NestedModel", type="OpenAi"),
            ],
        ),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "NestedModel")
    assert span.model == "gpt-4o-mini"


def test_model_extraction_variant_keys():
    # Ensure alternative key names are detected
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    variant_run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 5,
        executionTime=4,
        executionStatus="success",
        data={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "response": {"generations": [[{"text": "hi"}]]},
                            "model_name": "cohere-command-r-plus",
                        }
                    }
                ]
            ]
        },
    )
    starter = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=1,
        executionStatus="success",
        data={"main": [[{"json": {"ok": True}}]]},
    )
    runData = {"Starter": [starter], "VariantNode": [variant_run]}
    rec = N8nExecutionRecord(
        id=907,
        workflowId="wf-model-variant",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-model-variant",
            name="Model Variant",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(name="VariantNode", type="CohereChat"),
            ],
        ),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "VariantNode")
    assert span.model == "cohere-command-r-plus"


def test_model_extraction_ai_languageModel_options_path():
    # Mirrors screenshot: ai_languageModel -> [[ { json: { messages:[], estimatedTokens, options:{ base_url, model } } } ]]
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    run = NodeRun(
        startTime=int(now.timestamp() * 1000) + 10,
        executionTime=20,
        executionStatus="success",
        data={
            "ai_languageModel": [
                [
                    {
                        "json": {
                            "messages": ["Human: Hello"],
                            "estimatedTokens": 13,
                            "options": {
                                "google_api_key": {"cle": 1},
                                "base_url": "https://generativelanguage.googleapis.com",
                                "model": "gemini-2.5-pro",
                            },
                        }
                    }
                ]
            ]
        },
    )
    starter = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=1,
        executionStatus="success",
        data={"main": [[{"json": {"ok": True}}]]},
    )
    runData = {"Starter": [starter], "Google Gemini Chat Model": [run]}
    rec = N8nExecutionRecord(
        id=908,
        workflowId="wf-model-gemini-options",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-model-gemini-options",
            name="Gemini Options",
            nodes=[
                WorkflowNode(name="Starter", type="ToolWorkflow"),
                WorkflowNode(name="Google Gemini Chat Model", type="GoogleGemini"),
            ],
        ),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == "Google Gemini Chat Model")
    assert span.model == "gemini-2.5-pro"

