import json
from datetime import datetime, timezone

from n8n_langfuse_shipper.config import Settings
from n8n_langfuse_shipper.mapper import _map_execution  # type: ignore
from n8n_langfuse_shipper.media_api import MappedTraceWithAssets, patch_and_upload_media
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


def _build_simple_record(base64_payload: str) -> N8nExecutionRecord:
    node = WorkflowNode(name="ImageNode", type="customImage", category="ai")
    run = NodeRun(
        startTime=int(datetime.now(tz=timezone.utc).timestamp() * 1000),
        executionTime=5,
        executionStatus="success",
        data={
            "binary": {
                "image": {
                    "mimeType": "image/jpeg",
                    "fileName": "pic.jpg",
                    "data": base64_payload,
                }
            }
        },
        source=[NodeRunSource(previousNode=None, previousNodeRun=None)],
        inputOverride=None,
        error=None,
    )
    record = N8nExecutionRecord(
        id=123,
        workflowId="wf1",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(id="wf1", name="wf", nodes=[node], connections={}),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"ImageNode": [run]})
            )
        ),
    )
    return record


class DummySettings(Settings):  # override only needed fields
    ENABLE_MEDIA_UPLOAD: bool = True
    LANGFUSE_HOST: str = "http://localhost"  # dummy
    LANGFUSE_PUBLIC_KEY: str = "pk"
    LANGFUSE_SECRET_KEY: str = "sk"
    MEDIA_SURFACE_MODE: str = "inplace"


def test_inplace_surface_replaces_placeholder(monkeypatch):
    # Fake httpx responses for create (dedupe path, no uploadUrl) and patch
    import httpx

    class Resp:
        def __init__(self, json_data, status=200):
            self._json = json_data
            self.status_code = status
            self.text = json.dumps(json_data)
        def json(self):
            return self._json

    def fake_post(url, json=None, auth=None, timeout=None):  # noqa: A002
        return Resp({"mediaId": "m123", "observationId": "obs_123"})

    def fake_put(url, content=None, headers=None, timeout=None):  # noqa: A002
        return Resp({}, 200)

    def fake_patch(url, json=None, auth=None, timeout=None):  # noqa: A002
        return Resp({}, 200)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "put", fake_put)
    monkeypatch.setattr(httpx, "patch", fake_patch)

    b64 = "/9j/" + ("A" * 500)  # minimal JPEG-looking base64
    record = _build_simple_record(b64)
    trace, assets = _map_execution(record, truncate_limit=0, collect_binaries=True)
    mapped = MappedTraceWithAssets(trace=trace, assets=assets)
    settings = DummySettings(DB_TABLE_PREFIX="n8n_")  # required field
    patch_and_upload_media(mapped, settings)
    span = [s for s in mapped.trace.spans if s.name == "ImageNode"][0]
    assert span.output is not None
    parsed = json.loads(span.output)
    # Token must have replaced original placeholder in-place under binary.image.data
    assert parsed.get("binary", {}).get("image", {}).get("data", "").startswith("@@@langfuseMedia:")
    # A shallow promotion adds key 'image'
    assert parsed.get("image", "").startswith("@@@langfuseMedia:")
    assert span.metadata.get("n8n.media.surface_mode") == "inplace"
    assert span.metadata.get("n8n.media.preview_surface") is True
    assert span.metadata.get("n8n.media.promoted_from_binary") is True
