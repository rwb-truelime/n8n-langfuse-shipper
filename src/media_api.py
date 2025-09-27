"""Langfuse Media API integration (breaking change replacing Azure Blob path).

This module supersedes the previous `media_uploader` Azure‑only implementation.
We now follow the official Langfuse multi‑modality flow:

1. Mapper (when ENABLE_MEDIA_UPLOAD true) collects binary assets and inserts
   temporary placeholders of shape:
       { "_media_pending": true, "sha256": "<hash>", "bytes": <int> }
2. Here we call the Langfuse Media API for every asset, obtaining either:
   a. An `uploadUrl` (HTTP PUT pre‑signed) + `id` (media identifier) OR
   b. A deduplicated response with only an `id` (no upload needed).
3. If `uploadUrl` present we PUT the decoded bytes (binary) exactly once.
4. We then patch span output JSON replacing the placeholder with the canonical
   Langfuse media token string:
       @@@langfuseMedia:type=<mime>|id=<id>|source=n8n@@@
   Notes:
     * `type=<mime>` only included when mime type known.
     * `source=n8n` constant to allow downstream grouping/analytics.
5. Span metadata `n8n.media.asset_count` records number of successful patches.
6. Failures (API error, decode error, oversize, failed PUT) leave placeholder
   intact and set `n8n.media.upload_failed=true` (single flag even if multiple
   failures) but do not abort the trace.

Breaking Changes vs prior Azure implementation:
* Removed Azure environment variables & dependency (azure-storage-blob).
* Reference object JSON replaced by token string.
* Deterministic blob naming no longer used; server assigns media id.

Environment Variables (new / simplified):
  ENABLE_MEDIA_UPLOAD=true  -> activates collection + upload phase.
  LANGFUSE_HOST             -> base; we derive media endpoint
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY for auth (same as OTLP).
  MEDIA_MAX_BYTES           -> size guard (same semantic as before).

Any change to token structure or flow must update README and instructions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
import base64
import json

import httpx

from .models.langfuse import LangfuseTrace

logger = logging.getLogger(__name__)


@dataclass
class BinaryAsset:
    execution_id: int
    node_name: str
    run_index: int
    field_path: str
    mime_type: Optional[str]
    filename: Optional[str]
    size_bytes: int
    sha256: str
    base64_len: int
    content_b64: str


@dataclass
class MappedTraceWithAssets:
    trace: LangfuseTrace
    assets: List[BinaryAsset]


class _MediaClient:
    def __init__(self, *, host: str, public_key: str, secret_key: str, timeout: int = 30):
        if host.endswith("/"):
            host = host[:-1]
        self.base = host
        self.auth = (public_key, secret_key)
        self.timeout = timeout

    def create_media(self, *, size: int, mime: Optional[str]) -> dict[str, Any]:
        url = f"{self.base}/api/public/media"
        payload: dict[str, Any] = {"size": size}
        if mime:
            payload["mimeType"] = mime
        try:
            resp = httpx.post(
                url,
                json=payload,
                auth=self.auth,
                timeout=self.timeout,
            )
        except Exception as e:  # pragma: no cover - network error
            raise RuntimeError(f"media create request failed: {e}") from e
        if resp.status_code >= 400:
            raise RuntimeError(
                f"media create failed status={resp.status_code} body={resp.text[:500]}"
            )
        try:
            return resp.json()
        except Exception as e:  # pragma: no cover - unexpected response
            raise RuntimeError(f"invalid media create JSON: {e}") from e

    def upload_binary(self, upload_url: str, data: bytes):
        try:
            # Langfuse presigned endpoints generally expect PUT
            resp = httpx.put(upload_url, content=data, timeout=self.timeout)
        except Exception as e:  # pragma: no cover - network error
            raise RuntimeError(f"media upload failed: {e}") from e
        if resp.status_code >= 400:
            raise RuntimeError(
                f"media upload failed status={resp.status_code} body={resp.text[:300]}"
            )


def _decode_base64(b64: str, size_cap: int) -> Optional[bytes]:
    try:
        est = (len(b64) * 3) // 4
        if est > size_cap:
            return None
        return base64.b64decode(b64, validate=False)
    except Exception:
        return None


def _build_token(media_id: str, mime: Optional[str]) -> str:
    # Token fields are pipe delimited key=value pairs inside sentinel wrappers
    parts = []
    if mime:
        parts.append(f"type={mime}")
    parts.append(f"id={media_id}")
    parts.append("source=n8n")
    inner = "|".join(parts)
    return f"@@@langfuseMedia:{inner}@@@"


def patch_and_upload_media(mapped: MappedTraceWithAssets, settings) -> None:
    if not settings.ENABLE_MEDIA_UPLOAD:
        return
    if not mapped.assets:
        return
    host = settings.LANGFUSE_HOST
    if not host or not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.warning("Media upload enabled but Langfuse host/keys missing; skipping media phase")
        return
    client = _MediaClient(
        host=host,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        timeout=settings.OTEL_EXPORTER_OTLP_TIMEOUT,
    )
    # Index spans by (node, run_index)
    span_index: Dict[tuple[str, int], Any] = {}
    for s in mapped.trace.spans:
        run_index = s.metadata.get("n8n.node.run_index") if s.metadata else None
        if isinstance(run_index, int):
            span_index[(s.name, run_index)] = s

    for asset in mapped.assets:
        span = span_index.get((asset.node_name, asset.run_index))
        if not span:
            continue
        # Parse existing JSON output
        try:
            parsed = json.loads(span.output) if span.output else None
        except Exception:
            parsed = None
        if not isinstance(parsed, (dict, list)):
            continue
        # Decode base64 with size guard
        decoded = _decode_base64(asset.content_b64, settings.MEDIA_MAX_BYTES)
        if decoded is None:
            logger.warning(
                "media skip decode_failure asset=%s reason=%s size_cap=%s",
                asset.sha256,
                "oversize_or_decode_error",
                settings.MEDIA_MAX_BYTES,
            )
            if span.metadata is not None:
                span.metadata["n8n.media.upload_failed"] = True
                # Structured error code for diagnostics
                span.metadata.setdefault("n8n.media.error_codes", []).append("decode_or_oversize")
            continue
        # Create media via API
        try:
            create_resp = client.create_media(size=len(decoded), mime=asset.mime_type)
        except Exception as e:
            logger.warning(
                "media create_failed asset=%s code=%s err=%s",
                asset.sha256,
                "create_api_error",
                e,
            )
            if span.metadata is not None:
                span.metadata["n8n.media.upload_failed"] = True
                span.metadata.setdefault("n8n.media.error_codes", []).append("create_api_error")
            continue
        media_id = str(create_resp.get("id")) if create_resp.get("id") else None
        upload_url = create_resp.get("uploadUrl")
        if not media_id:
            logger.warning(
                "media create_failed asset=%s code=%s reason=%s",
                asset.sha256,
                "missing_id",
                "no id in create response",
            )
            if span.metadata is not None:
                span.metadata["n8n.media.upload_failed"] = True
                span.metadata.setdefault("n8n.media.error_codes", []).append("missing_id")
            continue
        # Upload if required
        if upload_url:
            try:
                client.upload_binary(upload_url, decoded)
            except Exception as e:
                logger.warning(
                    "media upload_failed asset=%s media_id=%s code=%s err=%s",
                    asset.sha256,
                    media_id,
                    "upload_put_error",
                    e,
                )
                if span.metadata is not None:
                    span.metadata["n8n.media.upload_failed"] = True
                    span.metadata.setdefault("n8n.media.error_codes", []).append("upload_put_error")
                continue
        token = _build_token(media_id, asset.mime_type)

        def _patch(obj) -> int:
            replaced = 0
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if (
                        isinstance(v, dict)
                        and v.get("_media_pending")
                        and v.get("sha256") == asset.sha256
                    ):
                        obj[k] = token
                        replaced += 1
                    else:
                        replaced += _patch(v)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if (
                        isinstance(v, dict)
                        and v.get("_media_pending")
                        and v.get("sha256") == asset.sha256
                    ):
                        obj[i] = token
                        replaced += 1
                    else:
                        replaced += _patch(v)
            return replaced

        replacements = _patch(parsed)
        if replacements > 0:
            try:
                span.output = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                if span.metadata is not None:
                    current = span.metadata.get("n8n.media.asset_count") or 0
                    span.metadata["n8n.media.asset_count"] = int(current) + replacements
            except Exception:
                # If serialization fails we leave old output; mark failure
                logger.warning(
                    "media patch_failed asset=%s media_id=%s code=%s",
                    asset.sha256,
                    media_id,
                    "serialization_error",
                )
                if span.metadata is not None:
                    span.metadata["n8n.media.upload_failed"] = True
                    span.metadata.setdefault("n8n.media.error_codes", []).append(
                        "serialization_error"
                    )


__all__ = [
    "BinaryAsset",
    "MappedTraceWithAssets",
    "patch_and_upload_media",
]
