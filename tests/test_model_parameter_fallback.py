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


def _base_record(node: WorkflowNode, run: NodeRun) -> N8nExecutionRecord:
    now = datetime.now(timezone.utc)
    starter = NodeRun(
        startTime=int(now.timestamp() * 1000),
        executionTime=1,
        executionStatus="success",
        data={"main": [[{"json": {"ok": True}}]]},
    )
    runData = {"Starter": [starter], node.name: [run]}
    rec = N8nExecutionRecord(
        id=1337,
        workflowId="wf-params-model",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-params-model",
            name="Params Model",
            nodes=[WorkflowNode(name="Starter", type="ToolWorkflow"), node],
        ),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runData))),
    )
    return rec


def test_model_parameter_fallback_basic():
    """Model absent in runtime data; extracted from parameters.model."""
    node = WorkflowNode(
        name="Azure Chat",
        type="@n8n/n8n-nodes-langchain.lmChatAzureOpenAi",
        parameters={"model": "gpt-5-standard"},
    )
    run = NodeRun(
        startTime=int(datetime.now(timezone.utc).timestamp() * 1000) + 5,
        executionTime=10,
        executionStatus="success",
        data={"response": {"choices": []}},  # no model returned
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    rec = _base_record(node, run)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == f"{node.name} #0")
    assert span.model == "gpt-5-standard"
    assert span.metadata.get("n8n.model.from_parameters") is True
    assert span.metadata.get("n8n.model.parameter_key") == "parameters.model"
    # Azure heuristic marks deployment
    assert span.metadata.get("n8n.model.is_deployment") is True


def test_model_parameter_fallback_nested_options():
    node = WorkflowNode(
        name="Vendor Chat",
        type="SomeVendorChat",
        parameters={"options": {"deploymentName": "custom-deploy-1"}},
    )
    run = NodeRun(
        startTime=int(datetime.now(timezone.utc).timestamp() * 1000) + 5,
        executionTime=10,
        executionStatus="success",
        data={"content": {"messages": []}},
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    rec = _base_record(node, run)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == f"{node.name} #0")
    assert span.model == "custom-deploy-1"
    assert span.metadata.get("n8n.model.parameter_key") == "parameters.options.deploymentName"


def test_model_parameter_fallback_does_not_override_runtime():
    node = WorkflowNode(
        name="LLM Node",
        type="OpenAiChat",
        parameters={"model": "deployment-alias"},
    )
    run = NodeRun(
        startTime=int(datetime.now(timezone.utc).timestamp() * 1000) + 5,
        executionTime=10,
        executionStatus="success",
        data={"model": "gpt-4-real", "tokenUsage": {"input": 1}},
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    rec = _base_record(node, run)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == f"{node.name} #0")
    assert span.model == "gpt-4-real"  # runtime value prevails
    assert span.metadata.get("n8n.model.from_parameters") is None


def test_model_parameter_fallback_ignores_expression():
    node = WorkflowNode(
        name="Expr Chat",
        type="ProviderChat",
        parameters={"model": "={{ $json.runtimeModel }}"},
    )
    run = NodeRun(
        startTime=int(datetime.now(timezone.utc).timestamp() * 1000) + 5,
        executionTime=10,
        executionStatus="success",
        data={"tokenUsage": {"input": 3}},  # generation classification triggers fallback attempt
        source=[NodeRunSource(previousNode="Starter", previousNodeRun=0)],
    )
    rec = _base_record(node, run)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    span = next(s for s in trace.spans if s.name == f"{node.name} #0")
    assert span.model is None
    assert span.metadata.get("n8n.model.missing") is True
