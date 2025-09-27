import json
from datetime import datetime, timezone
from types import SimpleNamespace, ModuleType
import sys

from src.mapper import map_execution_with_assets
from src.media_uploader import upload_media_assets
from src.models.n8n import (
    N8nExecutionRecord,
    WorkflowData,
    WorkflowNode,
    ExecutionData,
    ExecutionDataDetails,
    ResultData,
    NodeRun,
)
from src.config import Settings


class _FakeBlobClient:
    def __init__(self, store: dict[str, bytes], name: str):
        self._store = store
        self._name = name

    def get_blob_properties(self):  # existence check
        if self._name not in self._store:
            raise Exception("NotFound")
        return {}

    def upload_blob(self, data: bytes, overwrite: bool = False):
        if self._name in self._store and not overwrite:
            raise Exception("Exists")
        self._store[self._name] = data


class _FakeContainerClient:
    def __init__(self, store: dict[str, bytes]):
        self._store = store

    def get_container_properties(self):
        return {}

    def create_container(self):  # pragma: no cover - not used
        return None

    def get_blob_client(self, name: str):
        return _FakeBlobClient(self._store, name)


class _FakeBlobServiceClient:
    def __init__(self, account_url: str, credential: str):  # noqa: D401 - signature mimic
        self._store: dict[str, bytes] = {}

    def get_container_client(self, container: str):
        return _FakeContainerClient(self._store)


def _sample_execution_with_binary():
    # Minimal base64 content (long enough to pass >100 length threshold)
    b64 = ("aGVsbG93b3JsZA==" * 10)  # repeated to extend length
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
    record = N8nExecutionRecord(
        id=999,
        workflowId="wfX",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wfX",
            name="wfX",
            nodes=[WorkflowNode(name="BinNode", type="custom.binary")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"BinNode": [run]})
            )
        ),
    )
    return record


def test_media_upload_success(monkeypatch):
    """End-to-end: collect assets -> mock upload -> replacement patch."""
    record = _sample_execution_with_binary()
    mapped = map_execution_with_assets(record, collect_binaries=True)
    assert mapped.assets, "Expected binary asset to be collected"
    # Fake settings enabling feature
    settings = Settings(
        DB_TABLE_PREFIX="n8n_",
        ENABLE_MEDIA_UPLOAD=True,
        LANGFUSE_USE_AZURE_BLOB=True,
        LANGFUSE_S3_EVENT_UPLOAD_BUCKET="container",
        LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID="devstoreaccount1",
        LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY="dummykey",
    )

    # Inject fake azure modules into sys.modules before lazy import in uploader
    azure_mod = ModuleType("azure")
    storage_mod = ModuleType("azure.storage")
    blob_mod = ModuleType("azure.storage.blob")
    blob_mod.BlobServiceClient = _FakeBlobServiceClient  # type: ignore[attr-defined]
    azure_mod.storage = storage_mod  # type: ignore[attr-defined]
    storage_mod.blob = blob_mod  # type: ignore[attr-defined]
    sys.modules["azure"] = azure_mod
    sys.modules["azure.storage"] = storage_mod
    sys.modules["azure.storage.blob"] = blob_mod

    upload_media_assets(mapped, settings)
    # After upload, span output should contain replacement object
    # Locate span for node
    span = next(s for s in mapped.trace.spans if s.name == "BinNode")
    assert span.output is not None
    parsed = json.loads(span.output)
    assert parsed["binary"]["file"]["data"]["_media"] is True
    assert parsed["binary"]["file"]["data"]["storage"] == "azure"
    assert span.metadata.get("n8n.media.asset_count") == 1
    assert "upload_failed" not in ";".join(span.metadata.keys())
