from datetime import datetime, timezone

from n8n_langfuse_shipper.config import Settings
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


def _build_record(runs_by_node: dict[str, list[NodeRun]]) -> N8nExecutionRecord:
    return N8nExecutionRecord(
        id=4242,
        workflowId="wf1",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wf1",
            name="RootIOTest",
            nodes=[WorkflowNode(name=n, type="custom") for n in runs_by_node.keys()],
            connections={},
        ),
        data=ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData=runs_by_node))),
    )


def test_root_span_input_and_output_population(monkeypatch):
    # Two runs of same node; last run should be chosen
    runs = [
        NodeRun(
            startTime=1710000000000,
            executionTime=10,
            executionStatus="success",
            data={"value": "first"},
            source=None,
            inputOverride={"prompt": "hi"},
            error=None,
        ),
        NodeRun(
            startTime=1710000001000,
            executionTime=5,
            executionStatus="success",
            data={"value": "second"},
            source=None,
            inputOverride={"prompt": "bye"},
            error=None,
        ),
    ]
    rec = _build_record({"MyNode": runs})
    settings = Settings(DB_TABLE_PREFIX="n8n_", ROOT_SPAN_INPUT_NODE="mynode", ROOT_SPAN_OUTPUT_NODE="MYNODE")
    monkeypatch.setattr("n8n_langfuse_shipper.config.get_settings", lambda: settings)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    root = next(s for s in trace.spans if s.name == trace.name)
    assert root.input is not None and "bye" in root.input, "expected last run input propagated"
    assert root.output is not None and "second" in root.output, "expected last run output propagated"
    assert root.metadata.get("n8n.root.input_node") == "mynode"
    assert root.metadata.get("n8n.root.output_node") == "MYNODE"
    assert root.metadata.get("n8n.root.input_run_index") == 1
    assert root.metadata.get("n8n.root.output_run_index") == 1


def test_root_span_input_only(monkeypatch):
    run = NodeRun(
        startTime=1710000000000,
        executionTime=5,
        executionStatus="success",
        data={"answer": 42},
        source=None,
        inputOverride={"question": "life"},
        error=None,
    )
    rec = _build_record({"Question": [run]})
    settings = Settings(DB_TABLE_PREFIX="n8n_", ROOT_SPAN_INPUT_NODE="question")
    monkeypatch.setattr("n8n_langfuse_shipper.config.get_settings", lambda: settings)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    root = next(s for s in trace.spans if s.name == trace.name)
    assert root.input is not None and "life" in root.input
    assert root.output is None, "output should remain None"
    # Global env may specify ROOT_SPAN_OUTPUT_NODE causing not-found metadata; tolerate its presence.
    assert root.metadata.get("n8n.root.output_node_not_found") in (None, True)


def test_root_span_output_only(monkeypatch):
    run = NodeRun(
        startTime=1710000000000,
        executionTime=5,
        executionStatus="success",
        data={"answer": 99},
        source=None,
        inputOverride=None,
        error=None,
    )
    rec = _build_record({"Compute": [run]})
    settings = Settings(DB_TABLE_PREFIX="n8n_", ROOT_SPAN_OUTPUT_NODE="compute")
    monkeypatch.setattr("n8n_langfuse_shipper.config.get_settings", lambda: settings)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    root = next(s for s in trace.spans if s.name == trace.name)
    assert root.output is not None and "99" in root.output
    assert root.input is None


def test_root_span_node_absent(monkeypatch):
    run = NodeRun(
        startTime=1710000000000,
        executionTime=5,
        executionStatus="success",
        data={"x": 1},
        source=None,
        inputOverride=None,
        error=None,
    )
    rec = _build_record({"Real": [run]})
    settings = Settings(DB_TABLE_PREFIX="n8n_", ROOT_SPAN_INPUT_NODE="MissingNode")
    monkeypatch.setattr("n8n_langfuse_shipper.config.get_settings", lambda: settings)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    root = next(s for s in trace.spans if s.name == trace.name)
    assert root.input is None
    assert root.metadata.get("n8n.root.input_node_not_found") is True


def test_root_span_same_node_for_both(monkeypatch):
    run = NodeRun(
        startTime=1710000000000,
        executionTime=5,
        executionStatus="success",
        data={"y": "out"},
        source=None,
        inputOverride={"y": "in"},
        error=None,
    )
    rec = _build_record({"Echo": [run]})
    settings = Settings(DB_TABLE_PREFIX="n8n_", ROOT_SPAN_INPUT_NODE="echo", ROOT_SPAN_OUTPUT_NODE="echo")
    monkeypatch.setattr("n8n_langfuse_shipper.config.get_settings", lambda: settings)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    root = next(s for s in trace.spans if s.name == trace.name)
    assert root.input is not None and "in" in root.input
    assert root.output is not None and "out" in root.output


def test_root_span_error_span(monkeypatch):
    run_ok = NodeRun(
        startTime=1710000000000,
        executionTime=5,
        executionStatus="success",
        data={"res": 1},
        source=None,
        inputOverride={"ask": "one"},
        error=None,
    )
    run_err = NodeRun(
        startTime=1710000005000,
        executionTime=3,
        executionStatus="error",
        data={"res": 2},
        source=None,
        inputOverride={"ask": "two"},
        error={"message": "boom"},
    )
    rec = _build_record({"LLM": [run_ok, run_err]})
    settings = Settings(DB_TABLE_PREFIX="n8n_", ROOT_SPAN_INPUT_NODE="llm", ROOT_SPAN_OUTPUT_NODE="llm")
    monkeypatch.setattr("n8n_langfuse_shipper.config.get_settings", lambda: settings)
    trace = map_execution_to_langfuse(rec, truncate_limit=None)
    root = next(s for s in trace.spans if s.name == trace.name)
    assert "two" in str(root.input)
    assert "2" in str(root.output)
    # Ensure metadata points to last run (index 1)
    assert root.metadata.get("n8n.root.input_run_index") == 1
    assert root.metadata.get("n8n.root.output_run_index") == 1
