from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid5

from n8n_langfuse_shipper.mapper import (
    SPAN_NAMESPACE,
    _detect_gemini_empty_output_anomaly,
    _extract_model_and_metadata,
    _resolve_parent,
)
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


def _simple_record(nodes: list[WorkflowNode], run_data_map):
    now = datetime.now(timezone.utc)
    return N8nExecutionRecord(
        id=9999,
        workflowId="wf-test",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-test", name="HelperTest", nodes=nodes
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_data_map)
            )
        ),
    )


def test_resolve_parent_precedence_agent_hierarchy():
    # Build minimal structures
    node_a = WorkflowNode(name="ToolA", type="CalcNode")
    node_agent = WorkflowNode(name="Agent", type="AgentNode")
    run_a = NodeRun(startTime=1000, executionTime=5, executionStatus="success", data={})
    run_agent = NodeRun(
        startTime=1010,
        executionTime=5,
        executionStatus="success",
        data={},
        source=[NodeRunSource(previousNode="ToolA", previousNodeRun=0)],
    )
    record = _simple_record([node_a, node_agent], {"ToolA": [run_a], "Agent": [run_agent]})
    # Agent hierarchy map: ToolA child of Agent via ai_tool edge (simulate)
    child_agent_map = {"ToolA": ("Agent", "ai_tool")}
    last_span_for_node = {
        "Agent": str(uuid5(SPAN_NAMESPACE, f"{record.id}:Agent:0"))
    }
    parent_id, prev_node, prev_run = _resolve_parent(
        node_name="ToolA",
        run=run_a,
        trace_id=str(record.id),
        child_agent_map=child_agent_map,
        last_span_for_node=last_span_for_node,
        reverse_edges={},
        root_span_id=str(uuid5(SPAN_NAMESPACE, f"{record.id}:root")),
    )
    assert prev_node == "Agent"
    assert prev_run is None
    assert parent_id == last_span_for_node["Agent"]


def test_extract_model_and_metadata_parameter_fallback():
    node = WorkflowNode(name="LLM", type="AzureChat", parameters={"model": "gpt-x"})
    run = NodeRun(
        startTime=1,
        executionTime=1,
        executionStatus="success",
        data={"tokenUsage": {"input": 1}},  # generation triggers fallback attempt
    )
    model_val, meta = _extract_model_and_metadata(
        run=run,
        node_name="LLM",
        node_type=node.type,
        is_generation=True,
        wf_node_obj={"LLM": node},
        raw_input_obj=None,
    )
    assert model_val == "gpt-x"
    assert meta.get("n8n.model.from_parameters") is True
    assert meta.get("n8n.model.parameter_key") == "parameters.model"


def test_gemini_empty_output_anomaly_detection():
    # Minimal Gemini-like empty output shape
    run = NodeRun(
        startTime=1,
        executionTime=1,
        executionStatus="success",
        data={
            "response": {
                # Expected nesting: generations -> [ [ { text:"" } ] ]
                "generations": [[{"text": "", "generationInfo": {}}]],
                "tokenUsage": {
                    "promptTokens": 5,
                    "completionTokens": 0,
                    "totalTokens": 5,
                },
            }
        },
    )
    status_override, meta = _detect_gemini_empty_output_anomaly(
        is_generation=True,
        norm_output_obj=run.data.get("response"),
        run=run,
        node_name="GeminiNode",
    )
    assert status_override == "error"
    assert meta.get("n8n.gen.empty_output_bug") is True
    # synthetic error inserted
    assert run.error and "Gemini empty output" in run.error.get("message", "")


def test_gemini_empty_output_tool_call_suppression():
    # Generation span followed by tool span; should suppress error and add
    # tool_calls metadata instead of anomaly error markers.
    gen_run = NodeRun(
        startTime=1,
        executionTime=1,
        executionStatus="success",
        data={
            "response": {
                "generations": [[{"text": "", "generationInfo": {}}]],
                "tokenUsage": {
                    "promptTokens": 7,
                    "completionTokens": 0,
                    "totalTokens": 7,
                },
            }
        },
    )
    # We pass next_observation_type manually simulating lookahead classification
    status_override, meta = _detect_gemini_empty_output_anomaly(
        is_generation=True,
        norm_output_obj=gen_run.data.get("response"),
        run=gen_run,
        node_name="GeminiNode",
        next_observation_type="tool",
    )
    assert status_override is None  # no error status override
    assert "n8n.gen.empty_output_bug" not in meta
    assert meta.get("n8n.gen.tool_calls_pending") is True
    # Ensure no synthetic error injected on run
    assert gen_run.error is None
