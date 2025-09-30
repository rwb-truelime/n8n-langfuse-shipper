import json
from datetime import datetime, timezone

from src.mapper import map_execution_with_assets
from src.models.n8n import (
    N8nExecutionRecord,
    WorkflowData,
    WorkflowNode,
    ExecutionData,
    ExecutionDataDetails,
    ResultData,
    NodeRun,
)


def _make_run() -> N8nExecutionRecord:
    # Long base64 string (>100) to trigger asset collection
    b64 = "aGVsbG9iaW5hcnk=" * 20
    run = NodeRun(
        startTime=1710000001000,
        executionTime=12,
        executionStatus="success",
        data={
            # Canonical n8n binary section
            "binary": {
                "file": {
                    "data": b64,
                    "mimeType": "text/plain",
                    "fileName": "test.txt",
                }
            },
            # Wrapper style structure we want to unwrap *without* losing binary
            "main": [[[{"json": {"foo": "bar"}}]]],
        },
        source=None,
        inputOverride=None,
        error=None,
    )
    return N8nExecutionRecord(
        id=777,
        workflowId="wf-bin-merge",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wf-bin-merge",
            name="wf-bin-merge",
            nodes=[WorkflowNode(name="NodeA", type="custom.binaryMerge")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"NodeA": [run]})
            )
        ),
    )


def test_binary_preserved_after_unwrap_merge():
    rec = _make_run()
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    span = next(s for s in mapped.trace.spans if s.name == "NodeA")
    assert span.output is not None
    # Output is now a flattened dict (not JSON string)
    assert isinstance(span.output, dict), "Output must be flattened dict"
    # Assert unwrapped json content present in flattened form
    assert span.output.get("foo") == "bar", "Expected foo key in flattened output"
    # Assert binary placeholder (media pending) preserved as nested dict
    # Media placeholders are kept nested (not flattened) for media API patching
    data_val = span.output.get("binary.file.data")
    assert isinstance(data_val, dict), "Expected nested placeholder dict"
    assert data_val.get("_media_pending") is True, "Expected _media_pending flag"
    assert "sha256" in data_val, "Expected sha256 in placeholder"
    assert "base64_len" in data_val, "Expected base64_len in placeholder"
