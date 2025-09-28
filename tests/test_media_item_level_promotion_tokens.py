import json
from datetime import datetime, timezone
import base64

from src.mapper import map_execution_with_assets
from src.media_api import patch_and_upload_media  # type: ignore
from src.models.n8n import (
    N8nExecutionRecord,
    WorkflowData,
    WorkflowNode,
    ExecutionData,
    ExecutionDataDetails,
    ResultData,
    NodeRun,
)
from src.models.langfuse import LangfuseTrace

class DummyMediaClient:
    def __init__(self, *args, **kwargs):
        self.created = []
        self.uploaded = []

    def create_media(self, trace_id, content_type, content_length, sha256_b64, field):
        media_id = f"m_{len(self.created)}"
        self.created.append((trace_id, content_type, content_length, sha256_b64, field))
        # Dedup path: no uploadUrl triggers immediate token substitution without upload
        return {"mediaId": media_id}

    def upload_binary(self, *args, **kwargs):  # pragma: no cover - not used
        self.uploaded.append(args)

class Settings:
    ENABLE_MEDIA_UPLOAD = True
    MEDIA_MAX_BYTES = 25_000_000
    EXTENDED_MEDIA_SCAN_MAX_ASSETS = 250
    LANGFUSE_HOST = "https://dummy"
    LANGFUSE_PUBLIC_KEY = "pk_test"
    LANGFUSE_SECRET_KEY = "sk_test"
    OTEL_EXPORTER_OTLP_TIMEOUT = 10

# Monkeypatch get_settings in media_api to return our settings
from src import media_api as media_mod  # noqa
media_mod.get_settings = lambda: Settings()  # type: ignore

# Monkeypatch internal _MediaClient class used by patch_and_upload_media
media_mod._MediaClient = DummyMediaClient  # type: ignore

B64 = ("aGVsbG9faXRlbV9wcm9tb3RlZF9iaW5hcnk=" * 8)

def _record():
    payload = {
        "main": [
            [
                [
                    {
                        "json": {"ok": True, "idx": 0},
                        "binary": {"slotA": {"data": B64, "mimeType": "image/png", "fileName": "a.png"}},
                    },
                    {
                        "json": {"ok": True, "idx": 1},
                        "binary": {"slotB": {"data": B64, "mimeType": "image/png", "fileName": "b.png"}},
                    },
                ]
            ]
        ]
    }
    run = NodeRun(
        startTime=1711000000000,
        executionTime=10,
        executionStatus="success",
        data=payload,
        source=None,
        inputOverride=None,
        error=None,
    )
    return N8nExecutionRecord(
        id=9200,
        workflowId="wf-media-promote",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wf-media-promote",
            name="wf-media-promote",
            nodes=[WorkflowNode(name="NodePromote", type="custom.itemBinaryMedia")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"NodePromote": [run]})
            )
        ),
    )

def test_media_token_substitution_with_item_level_promotion(monkeypatch):
    rec = _record()
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    trace = mapped.trace
    # Create dummy index for patch phase
    settings = Settings()
    patch_and_upload_media(mapped, settings)
    span = next(s for s in trace.spans if s.name == "NodePromote")
    assert span.metadata.get("n8n.io.promoted_item_binary") is True
    assert span.metadata.get("n8n.media.asset_count") == 2
    assert span.output is not None
    parsed = json.loads(span.output)
    # Walk to ensure tokens present replacing placeholders
    def _contains_token(o):
        if isinstance(o, str) and o.startswith("@@@langfuseMedia:"):
            return True
        if isinstance(o, dict):
            return any(_contains_token(v) for v in o.values())
        if isinstance(o, list):
            return any(_contains_token(v) for v in o)
        return False
    assert _contains_token(parsed), "Expected media tokens in span output after patch phase"
