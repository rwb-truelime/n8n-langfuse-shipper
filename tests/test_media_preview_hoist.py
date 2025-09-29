from __future__ import annotations

import json
import base64
import hashlib
from datetime import datetime, timezone

from src.media_api import (
    BinaryAsset,
    MappedTraceWithAssets,
    patch_and_upload_media,
)
from src.models.langfuse import LangfuseTrace, LangfuseSpan


class _Settings:
    ENABLE_MEDIA_UPLOAD = True
    LANGFUSE_HOST = "https://example.com"
    LANGFUSE_PUBLIC_KEY = "pk"
    LANGFUSE_SECRET_KEY = "sk"
    OTEL_EXPORTER_OTLP_TIMEOUT = 5
    MEDIA_MAX_BYTES = 1_000_000


class _DummyClient:
    def __init__(self, **_):  # pragma: no cover - trivial init
        pass

    def create_media(self, **_):  # Dedup path: no upload
        return {"mediaId": "preview-test-id", "uploadUrl": None}

    def upload_binary(self, *_, **__):  # pragma: no cover - not invoked
        raise AssertionError("upload_binary should not be called")


def test_media_inplace_preview(monkeypatch):
    monkeypatch.setattr("src.media_api._MediaClient", _DummyClient)
    content = b"hello-image"
    b64 = base64.b64encode(content).decode("ascii")
    sha256_hex = hashlib.sha256(content).hexdigest()
    placeholder = {
        "_media_pending": True,
        "sha256": sha256_hex,
        "bytes": len(content),
        "base64_len": len(b64),
        "slot": "data",
    }
    nested = {
        "binary": {
            "data": {
                "mimeType": "image/jpeg",
                "fileName": "x.jpg",
                "data": placeholder,
            }
        }
    }
    span = LangfuseSpan(
        id="span1",
        trace_id="t1",
        name="node",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        observation_type="span",
        output=json.dumps(nested),
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
        field_path="output.binary.data.data",
        mime_type="image/jpeg",
        filename="x.jpg",
        size_bytes=len(content),
        sha256=sha256_hex,
        base64_len=len(b64),
        content_b64=b64,
    )
    mapped = MappedTraceWithAssets(trace=trace, assets=[asset])
    patch_and_upload_media(mapped, _Settings)
    assert span.output is not None
    parsed = json.loads(span.output)
    # In-place surfacing removed hoist; ensure token present at original nested path.
    assert "_media_preview" not in parsed, "Hoist key should not exist after simplification"
    assert parsed["binary"]["data"]["data"].startswith(
        "@@@langfuseMedia:"
    ), "Nested token should be replaced in-place"
    # This path includes an 'output.' prefix in field_path so promotion condition
    # (asset.field_path.startswith("binary.")) is not met; no shallow key expected.
    assert "data" not in parsed, "Did not expect shallow promotion for non-canonical field path"
    # Metadata flags (preview_surface still set due to in-place replacement)
    assert span.metadata.get("n8n.media.preview_surface") is True
    assert span.metadata.get("n8n.media.promoted_from_binary") is not True
    assert span.metadata.get("n8n.media.surface_mode") == "inplace"