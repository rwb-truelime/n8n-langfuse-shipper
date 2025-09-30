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

# Two discoverable assets so that with cap=1 we trigger scan_asset_limit.
BASE64_A = ("aGVsbG8=" * 15)  # ensure length >=64 chars
BASE64_B = ("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 4)


def _record(payload):
    run = NodeRun(
        startTime=1710002000000,
        executionTime=5,
        executionStatus="success",
        data=payload,
        source=None,
        inputOverride=None,
        error=None,
    )
    return N8nExecutionRecord(
        id=9101,
        workflowId="wf-ext-bin-cap",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wf-ext-bin-cap",
            name="wf-ext-bin-cap",
            nodes=[WorkflowNode(name="NodeCap", type="custom.extBinaryCap")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"NodeCap": [run]})
            )
        ),
    )


def _extract_output_span(trace):
    return next(s for s in trace.spans if s.name == "NodeCap")


def test_extended_scan_limit_triggers_error_code(monkeypatch):
    # Force cap to 1 so second asset causes limit hit.
    monkeypatch.setenv("EXTENDED_MEDIA_SCAN_MAX_ASSETS", "1")
    # Clear cached settings so new env var is picked up.
    from src.config import get_settings  # local import
    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    # Provide two separate discoverable assets: one file-like dict and one contextual base64.
    rec = _record({
        "file": {"mimeType": "text/plain", "fileName": "a.txt", "data": BASE64_A},
        "foo": {"mimeType": "application/octet-stream", "fileName": "b.bin", "payload": BASE64_B},
    })
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    span = _extract_output_span(mapped.trace)
    # We expect at least one placeholder inserted on first asset and second ignored due to cap.
    # Output is now a flattened dict (media placeholders stay nested)
    assert isinstance(span.output, dict), "Output must be flattened dict"
    data_val = span.output.get("file.data")
    assert isinstance(data_val, dict), "Expected nested placeholder"
    assert data_val.get("_media_pending") is True
    assert "base64_len" in data_val
    # Metadata should reflect scan_asset_limit error code.
    codes = span.metadata.get("n8n.media.error_codes") if span.metadata else None
    assert codes and "scan_asset_limit" in codes
    assert span.metadata.get("n8n.media.upload_failed") is True
