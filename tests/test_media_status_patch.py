from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from typing import ClassVar

from src.media_api import (
    BinaryAsset,
    MappedTraceWithAssets,
    patch_and_upload_media,
)
from src.models.langfuse import LangfuseSpan, LangfuseTrace


class _Settings:
    ENABLE_MEDIA_UPLOAD: ClassVar[bool] = True
    LANGFUSE_HOST: ClassVar[str] = "https://example.com"
    LANGFUSE_PUBLIC_KEY: ClassVar[str] = "pk"
    LANGFUSE_SECRET_KEY: ClassVar[str] = "sk"
    OTEL_EXPORTER_OTLP_TIMEOUT: ClassVar[int] = 5
    MEDIA_MAX_BYTES: ClassVar[int] = 1_000_000


class _ClientOK:
    def __init__(self, **_):
        self.patched = False

    def create_media(self, **_):
        return {"mediaId": "mid", "uploadUrl": "https://upload"}

    def upload_binary(self, *_, **__):
        return None

    def patch_media_status(self, **_):
        self.patched = True


class _ClientPatchFail(_ClientOK):
    def patch_media_status(self, **_):  # pragma: no cover - error scenario
        raise RuntimeError("boom")


def _build_span():
    content = b"imgdata"
    b64 = base64.b64encode(content).decode("ascii")
    shahex = hashlib.sha256(content).hexdigest()
    placeholder = {
        "_media_pending": True,
        "sha256": shahex,
        "bytes": len(content),
        "base64_len": len(b64),
    }
    output = {"binary": {"slot": {"data": placeholder}}}
    span = LangfuseSpan(
        id="s1",
        trace_id="t1",
        name="node",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        observation_type="span",
        output=output,  # Now a dict, matching mapper behavior
        metadata={"n8n.node.run_index": 0},
    )
    trace = LangfuseTrace(
        id="1",
        name="t",
        timestamp=datetime.now(timezone.utc),
        spans=[span],
    )
    asset = BinaryAsset(
        execution_id=1,
        node_name="node",
        run_index=0,
        field_path="output.binary.slot.data",
        mime_type="image/jpeg",
        filename="x.jpg",
        size_bytes=len(content),
        sha256=shahex,
        base64_len=len(b64),
        content_b64=b64,
    )
    return span, trace, asset


def test_status_patch_success(monkeypatch):
    client = _ClientOK()
    monkeypatch.setattr("src.media_api._MediaClient", lambda **_: client)
    span, trace, asset = _build_span()
    mapped = MappedTraceWithAssets(trace=trace, assets=[asset])
    patch_and_upload_media(mapped, _Settings)
    assert client.patched is True
    assert span.metadata.get("n8n.media.error_codes") is None


def test_status_patch_failure(monkeypatch):
    client = _ClientPatchFail()
    monkeypatch.setattr("src.media_api._MediaClient", lambda **_: client)
    span, trace, asset = _build_span()
    mapped = MappedTraceWithAssets(trace=trace, assets=[asset])
    patch_and_upload_media(mapped, _Settings)
    # Failure should append status_patch_error but not mark upload_failed
    codes = span.metadata.get("n8n.media.error_codes") or []
    assert "status_patch_error" in codes
    assert span.metadata.get("n8n.media.upload_failed") is not True
