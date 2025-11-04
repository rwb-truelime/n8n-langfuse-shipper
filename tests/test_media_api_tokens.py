import json
from datetime import datetime, timezone
from typing import Any

import pytest

from n8n_langfuse_shipper.config import Settings
from n8n_langfuse_shipper.mapper import map_execution_to_langfuse, map_execution_with_assets
from n8n_langfuse_shipper.media_api import patch_and_upload_media
from n8n_langfuse_shipper.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)


def _sample_execution_with_binary() -> N8nExecutionRecord:
    b64 = ("aGVsbG93b3JsZA==" * 20)  # ensure >100 length threshold
    run = NodeRun(
        startTime=1710000000000,
        executionTime=5,
        executionStatus="success",
        data={
            "binary": {
                "file": {
                    "data": b64,
                    "mimeType": "text/plain",
                    "fileName": "hello.txt",
                }
            }
        },
        source=None,
        inputOverride=None,
        error=None,
    )
    return N8nExecutionRecord(
        id=321,
        workflowId="wf1",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wf1",
            name="wf1",
            nodes=[WorkflowNode(name="BinNode", type="custom.binary")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"BinNode": [run]})
            )
        ),
    )


class _FakeMediaClientResponse:
    def __init__(self, media_id: str, upload_url: str | None):
        self.media_id = media_id
        self.upload_url = upload_url


@pytest.fixture(autouse=True)
def _patch_httpx(monkeypatch):
    # Intercept httpx.post (create media) and httpx.put (upload) calls.
    created: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict, auth, timeout: int):  # type: ignore[override]
        class _Resp:
            status_code = 200

            def json(self):  # noqa: D401
                idx = len(created)
                media_id = f"m_{idx}"
                created.append(json)
                return {
                    "id": media_id,
                    "uploadUrl": "https://upload.example.com/" if idx == 0 else None,
                }

        return _Resp()

    def fake_put(url: str, content: bytes, headers: dict, timeout: int):  # type: ignore[override]
        class _Resp:
            status_code = 200
            text = ""

        return _Resp()

    import httpx as _httpx

    monkeypatch.setattr(_httpx, "post", fake_post)
    monkeypatch.setattr(_httpx, "put", fake_put)


def test_media_disabled_equivalence():
    rec = _sample_execution_with_binary()
    trace_plain = map_execution_to_langfuse(rec)
    mapped_assets = map_execution_with_assets(rec, collect_binaries=False)
    assert mapped_assets.assets == []
    assert len(trace_plain.spans) == len(mapped_assets.trace.spans)
    assert trace_plain.spans[0].id == mapped_assets.trace.spans[0].id


def test_media_api_token_patch():
    rec = _sample_execution_with_binary()
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    assert mapped.assets, "Expected asset collection"
    settings = Settings(
        DB_TABLE_PREFIX="n8n_",
        ENABLE_MEDIA_UPLOAD=True,
        LANGFUSE_HOST="https://cloud.langfuse.com",
        LANGFUSE_PUBLIC_KEY="pk",
        LANGFUSE_SECRET_KEY="sk",
    )
    patch_and_upload_media(mapped, settings)
    span = next(s for s in mapped.trace.spans if s.name == "BinNode")
    assert span.output is not None
    assert span.output is not None
    parsed = json.loads(str(span.output))
    token = parsed["binary"]["file"]["data"]
    assert isinstance(token, str) and token.startswith("@@@langfuseMedia:")
    # Ensure asset count metadata present
    assert span.metadata.get("n8n.media.asset_count") == 1


def test_media_api_token_patch_deduplicated(monkeypatch):
    """Simulate create API returning id without uploadUrl (dedupe scenario)."""
    rec = _sample_execution_with_binary()
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    assert mapped.assets

    def fake_post(url: str, json: dict, auth, timeout: int):  # type: ignore[override]
        class _Resp:
            status_code = 200

            def json(self):  # noqa: D401
                return {"id": "dedupe_1", "uploadUrl": None}

        return _Resp()

    import httpx as _httpx

    monkeypatch.setattr(_httpx, "post", fake_post)

    settings = Settings(
        DB_TABLE_PREFIX="n8n_",
        ENABLE_MEDIA_UPLOAD=True,
        LANGFUSE_HOST="https://cloud.langfuse.com",
        LANGFUSE_PUBLIC_KEY="pk",
        LANGFUSE_SECRET_KEY="sk",
    )
    patch_and_upload_media(mapped, settings)
    span = next(s for s in mapped.trace.spans if s.name == "BinNode")
    # Ensure non-None then cast to string for static type checkers (span.output: Any|None)
    assert span.output is not None
    parsed = json.loads(str(span.output))
    token = parsed["binary"]["file"]["data"]
    # Token should reference deduplicated id from fake_post response OR follow m_* pattern
    assert token.startswith("@@@langfuseMedia:")
    assert ("dedupe_1" in token) or ("|id=m_" in token)
    assert span.metadata.get("n8n.media.asset_count") == 1
    # Ensure no upload_failed flag
    assert span.metadata.get("n8n.media.upload_failed") is not True
