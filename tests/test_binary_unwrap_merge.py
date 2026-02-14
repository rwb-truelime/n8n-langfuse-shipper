import json
from datetime import datetime, timezone

from n8n_langfuse_shipper.mapper import map_execution_with_assets
from n8n_langfuse_shipper.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
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
    span = next(s for s in mapped.trace.spans if s.name == "NodeA #0")
    assert span.output is not None
    parsed = json.loads(str(span.output))
    # Assert unwrapped json content present
    assert parsed.get("foo") == "bar" or (
        isinstance(parsed, dict) and any(v.get("foo") == "bar" for v in parsed.values() if isinstance(v, dict))
    )
    # Assert binary placeholder (media pending) preserved
    bin_section = parsed.get("binary")
    assert isinstance(bin_section, dict) and "file" in bin_section
    file_entry = bin_section["file"]
    assert isinstance(file_entry, dict)
    # Placeholder inserted by _collect_binary_assets
    assert isinstance(file_entry.get("data"), dict) and file_entry["data"].get("_media_pending") is True
    assert "base64_len" in file_entry["data"] and isinstance(file_entry["data"]["base64_len"], int)
