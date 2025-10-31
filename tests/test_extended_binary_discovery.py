import json
from datetime import datetime, timezone

from src.mapper import map_execution_with_assets
from src.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)

BASE64_IMG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Yl8m38AAAAASUVORK5CYII=" * 2
DATA_URL = "data:image/png;base64," + BASE64_IMG


def _record(payload):
    run = NodeRun(
        startTime=1710001000000,
        executionTime=5,
        executionStatus="success",
        data=payload,
        source=None,
        inputOverride=None,
        error=None,
    )
    return N8nExecutionRecord(
        id=9001,
        workflowId="wf-ext-bin",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wf-ext-bin",
            name="wf-ext-bin",
            nodes=[WorkflowNode(name="NodeX", type="custom.extBinary")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"NodeX": [run]})
            )
        ),
    )


def _extract_output_span(trace):
    return next(s for s in trace.spans if s.name == "NodeX")


def test_data_url_discovered():
    rec = _record({"image": {"data": DATA_URL, "mimeType": "image/png", "fileName": "x.png"}})
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    span = _extract_output_span(mapped.trace)
    parsed = json.loads(str(span.output))
    image_obj = parsed.get("image")
    assert isinstance(image_obj, dict)
    # Placeholder should be inserted for image.data
    assert isinstance(image_obj.get("data"), dict) and image_obj["data"].get("_media_pending") is True
    assert "base64_len" in image_obj["data"] and isinstance(image_obj["data"]["base64_len"], int)


def test_file_like_dict_discovered():
    rec = _record({"file": {"mimeType": "text/plain", "fileName": "a.txt", "data": ("aGVsbG8=" * 12)}})
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    span = _extract_output_span(mapped.trace)
    parsed = json.loads(str(span.output))
    file_obj = parsed.get("file")
    assert isinstance(file_obj, dict)
    assert isinstance(file_obj.get("data"), dict) and file_obj["data"].get("_media_pending") is True
    assert "base64_len" in file_obj["data"] and isinstance(file_obj["data"]["base64_len"], int)


def test_long_base64_with_context_keys():
    rec = _record({
        "foo": {
            "mimeType": "application/octet-stream",
            "fileName": "blob.bin",
            "payload": ("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 5),
        }
    })
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    span = _extract_output_span(mapped.trace)
    parsed = json.loads(str(span.output))
    foo = parsed.get("foo")
    assert isinstance(foo, dict)
    assert isinstance(foo.get("payload"), dict) and foo["payload"].get("_media_pending") is True
    assert "base64_len" in foo["payload"] and isinstance(foo["payload"]["base64_len"], int)
