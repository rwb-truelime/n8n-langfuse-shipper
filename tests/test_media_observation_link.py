import json
from datetime import datetime, timezone

from src.config import Settings
from src.mapper import map_execution_with_assets
from src.media_api import patch_and_upload_media
from src.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)


def _sample_exec() -> N8nExecutionRecord:
    b64 = ("aGVsbG93b3JsZA==" * 20)
    run = NodeRun(
        startTime=1710000000000,
        executionTime=5,
        executionStatus="success",
        data={
            "binary": {
                "img": {
                    "data": b64,
                    "mimeType": "image/jpeg",
                    "fileName": "x.jpg",
                }
            }
        },
        source=None,
        inputOverride=None,
        error=None,
    )
    return N8nExecutionRecord(
        id=9999,
        workflowId="wfX",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wfX",
            name="wfX",
            nodes=[WorkflowNode(name="ReadImage", type="custom.image")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData={"ReadImage": [run]}))
        ),
    )


def test_media_observation_id_included(monkeypatch):
    rec = _sample_exec()
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    span = next(s for s in mapped.trace.spans if s.name == "ReadImage")
    created_payloads: list[dict] = []

    class _Resp:
        status_code = 200

        def __init__(self, upload=True):
            self._upload = upload

        def json(self):  # noqa: D401
            return {"id": "mid123", "uploadUrl": "https://u/" if self._upload else None}

    def fake_post(url: str, json: dict, auth, timeout: int):  # type: ignore[override]
        created_payloads.append(json)
        return _Resp(upload=True)

    def fake_put(url: str, content: bytes, headers: dict, timeout: int):  # type: ignore[override]
        class _PR:
            status_code = 200
            text = ""

        return _PR()

    import httpx as _httpx

    monkeypatch.setattr(_httpx, "post", fake_post)
    monkeypatch.setattr(_httpx, "put", fake_put)

    settings = Settings(
        DB_TABLE_PREFIX="n8n_",
        ENABLE_MEDIA_UPLOAD=True,
        LANGFUSE_HOST="https://cloud.langfuse.com",
        LANGFUSE_PUBLIC_KEY="pk",
        LANGFUSE_SECRET_KEY="sk",
    )
    patch_and_upload_media(mapped, settings)
    assert created_payloads, "expected at least one media create payload"
    payload = created_payloads[0]
    # Ensure observationId present and matches span.id
    assert payload.get("observationId") == span.id
    # Ensure token replacement occurred
    assert span.output is not None
    parsed = json.loads(str(span.output))
    token = parsed["binary"]["img"]["data"]
    assert token.startswith("@@@langfuseMedia:") and "id=mid123" in token
    assert span.metadata.get("n8n.media.asset_count") == 1
