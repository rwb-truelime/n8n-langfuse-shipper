"""Media upload module (Azure Blob only, initial phase).

This module is intentionally separate from `mapper.py` to preserve the purity
guarantee of the mapping stage (no network I/O). The mapper (when feature flag
enabled) will return a `MappedTraceWithAssets` object containing the normal
`LangfuseTrace` plus a list of extracted `BinaryAsset` descriptors. We then
upload those assets here and patch span outputs to reference blob locations.

Environment variables (mirroring Langfuse Azure docs naming):
  ENABLE_MEDIA_UPLOAD (shipper feature flag; must be true)
  LANGFUSE_USE_AZURE_BLOB=true
  LANGFUSE_S3_EVENT_UPLOAD_BUCKET -> Azure container name
  LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID -> Azure storage account name
  LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY -> Azure storage account key
  LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT -> Optional endpoint (e.g. Azurite)

Reference object inserted into span output JSON in place of original binary:
  {
    "_media": true,
    "storage": "azure",
    "container": <container>,
    "blob": <blob_path>,
    "sha256": <content_hash>,
    "bytes": <size>
  }

Failure handling: fail-open. If an upload fails, the placeholder remains a
legacy redaction form and span metadata gets `n8n.media.upload_failed=true`.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models.langfuse import LangfuseTrace

logger = logging.getLogger(__name__)


@dataclass
class BinaryAsset:
    execution_id: int
    node_name: str
    run_index: int
    field_path: str  # path inside run.data (dot/bracket notation)
    mime_type: Optional[str]
    filename: Optional[str]
    size_bytes: int
    sha256: str
    base64_len: int
    # Raw base64 content (only retained until upload; can be large). We do not
    # persist decoded bytes to keep memory lower; decode on demand.
    content_b64: str


@dataclass
class MappedTraceWithAssets:
    trace: LangfuseTrace
    assets: List[BinaryAsset]


def _build_blob_name(asset: BinaryAsset) -> str:
    # Stable deterministic name: execId/node/runIndex/sha256 (retain original
    # filename extension if present and safe)
    ext = ""
    if asset.filename and "." in asset.filename:
        candidate = asset.filename.rsplit(".", 1)[-1].lower()[:16]
        if candidate.isalnum():
            ext = f".{candidate}"
    return (
        f"{asset.execution_id}/{asset.node_name}/{asset.run_index}/"
        f"{asset.sha256}{ext}"
    )


def _decode_b64_to_bytes(b64_str: str, size_cap: int) -> Optional[bytes]:
    try:
        # Quick pre-check via length (approximate base64 to bytes ratio ~ 4/3)
        est = (len(b64_str) * 3) // 4
        if est > size_cap:
            return None
        return base64.b64decode(b64_str, validate=False)
    except Exception:  # pragma: no cover - defensive
        return None


def upload_media_assets(mapped: MappedTraceWithAssets, settings: Any) -> None:
    """Upload binary assets to Azure Blob (if enabled) and patch spans.

    Mutates span outputs (stringified JSON) by replacing temporary placeholders
    with reference objects. We must re-serialize JSON if we patch.
    """
    if not (settings.ENABLE_MEDIA_UPLOAD and settings.LANGFUSE_USE_AZURE_BLOB):
        return
    if not mapped.assets:
        return
    container = settings.LANGFUSE_S3_EVENT_UPLOAD_BUCKET
    account = settings.LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID
    key = settings.LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY
    endpoint = settings.LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT
    if not (container and account and key):
        logger.warning(
            "Media upload enabled but Azure credentials/container missing; skipping uploads"
        )
        return
    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        logger.error(
            "azure-storage-blob dependency missing or failed to import: %s", e
        )
        return
    conn_url = endpoint or f"https://{account}.blob.core.windows.net"
    try:
        bsc = BlobServiceClient(
            account_url=conn_url,
            credential=key,
        )
        container_client = bsc.get_container_client(container)
        try:
            container_client.get_container_properties()
        except Exception:
            try:
                container_client.create_container()
            except Exception:
                logger.warning("Failed to ensure container existence: %s", container)
    except Exception as e:  # pragma: no cover - connection init
        logger.error("Failed initializing Azure Blob client: %s", e)
        return

    # Index spans by (node_name, run_index) for quick patch
    span_index: Dict[tuple[str, int], Any] = {}
    for s in mapped.trace.spans:
        node_name = s.name
        run_index = s.metadata.get("n8n.node.run_index") if s.metadata else None
        if isinstance(run_index, int):
            span_index[(node_name, run_index)] = s

    for asset in mapped.assets:
        span = span_index.get((asset.node_name, asset.run_index))
        if not span:
            continue
        # Skip if span output already patched (idempotence in repeated calls)
        try:
            import json
            parsed = json.loads(span.output) if span.output else None
        except Exception:
            parsed = None
        if not isinstance(parsed, (dict, list)):
            continue
        # Navigate to placeholder via path tokens (path recorded as a.b[c].d)
        # We store only top-level binary replacements inside run.data so we do
        # best-effort path resolution.
        def _replace(obj: Any) -> int:
            replaced = 0
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if (
                        isinstance(v, dict)
                        and v.get("_media_pending")
                        and v.get("sha256") == asset.sha256
                    ):
                        obj[k] = {
                            "_media": True,
                            "storage": "azure",
                            "container": container,
                            "blob": _build_blob_name(asset),
                            "sha256": asset.sha256,
                            "bytes": asset.size_bytes,
                        }
                        replaced += 1
                    else:
                        replaced += _replace(v)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if (
                        isinstance(v, dict)
                        and v.get("_media_pending")
                        and v.get("sha256") == asset.sha256
                    ):
                        obj[i] = {
                            "_media": True,
                            "storage": "azure",
                            "container": container,
                            "blob": _build_blob_name(asset),
                            "sha256": asset.sha256,
                            "bytes": asset.size_bytes,
                        }
                        replaced += 1
                    else:
                        replaced += _replace(v)
            return replaced
        # Upload first (so we only patch on success)
        decoded = _decode_b64_to_bytes(asset.content_b64, settings.MEDIA_MAX_BYTES)
        if decoded is None:
            # Size too large OR decode error – leave placeholder unpatched
            if span.metadata is not None:
                span.metadata["n8n.media.upload_failed"] = True
            continue
        blob_name = _build_blob_name(asset)
        blob_client = container_client.get_blob_client(blob_name)
        exists = False
        try:
            blob_client.get_blob_properties()
            exists = True
        except Exception:
            exists = False
        if not exists:
            try:
                blob_client.upload_blob(
                    decoded,
                    overwrite=False,
                )
            except Exception as e:  # pragma: no cover - network error branch
                logger.warning("Upload failed for blob %s: %s", blob_name, e)
                if span.metadata is not None:
                    span.metadata["n8n.media.upload_failed"] = True
                continue
        # Patch JSON structure
        replacements = _replace(parsed)
        try:
            import json
            new_output = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            span.output = new_output
            # Media counter (increment only if we patched a placeholder)
            if replacements > 0 and span.metadata is not None:
                current = span.metadata.get("n8n.media.asset_count") or 0
                span.metadata["n8n.media.asset_count"] = int(current) + replacements
        except Exception:
            pass

__all__ = [
    "BinaryAsset",
    "MappedTraceWithAssets",
    "upload_media_assets",
]
