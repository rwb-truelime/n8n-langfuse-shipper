"""Langfuse Media API integration (breaking change replacing Azure Blob path).

This module supersedes the previous `media_uploader` Azure‑only implementation.
We now follow the official Langfuse multi‑modality flow:

1. Mapper (when ENABLE_MEDIA_UPLOAD true) collects binary assets and inserts
   temporary placeholders of authoritative shape:
       {
           "_media_pending": true,
           "sha256": "<hex sha256>",
           "bytes": <decoded_size_bytes>,
           "base64_len": <original_base64_length>,
           "slot": <optional origin slot or field path>
       }
2. Here we call the Langfuse Media API for every asset, obtaining either:
   a. An `uploadUrl` (HTTP PUT pre‑signed) + `id` (media identifier) OR
   b. A deduplicated response with only an `id` (no upload needed).
3. If `uploadUrl` present we PUT the decoded bytes (binary) exactly once.
4. We then patch span output JSON replacing the placeholder with the canonical
     Langfuse media token string:
             @@@langfuseMedia:type=<mime>|id=<id>|source=base64_data_uri@@@
     Notes:
         * `type=<mime>` only included when mime type known.
         * `source=base64_data_uri` standard Langfuse source form.
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

Any change to token structure, placeholder keys, create payload fields, or
required presigned upload headers must update README + instructions + tests.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, cast

import httpx

from .models.langfuse import LangfuseTrace

# Allowed mime types (aligned with Langfuse server error list). If an asset's
# mime is not in this set we omit it (letting server infer or reject based on
# content) instead of sending an unsupported value that would 400.
_ALLOWED_MIME: set[str] = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
    "image/svg+xml",
    "image/tiff",
    "image/bmp",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/ogg",
    "audio/oga",
    "audio/aac",
    "audio/mp4",
    "audio/flac",
    "video/mp4",
    "video/webm",
    "text/plain",
    "text/html",
    "text/css",
    "text/csv",
    "application/pdf",
    "application/json",
}


def _sanitize_mime(mime: Optional[str]) -> Optional[str]:
    if not mime:
        return None
    m = mime.lower()
    if m in _ALLOWED_MIME:
        return m
    # Generic / unsupported types are omitted (fail-open: server may auto-detect)
    if m in {"application/octet-stream", "binary/octet-stream"}:
        return None
    # Conservative: drop unlisted subtype; do not guess a replacement.
    return None

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
    # Original path segments (dot path exploded, list indices removed) used for
    # in-place replacement. When None we fall back to recursive placeholder search.
    original_path_segments: Optional[list[str]] = None


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

    def create_media(
        self,
        *,
        trace_id: str,
        observation_id: str | None,
        content_type: Optional[str],
        content_length: int,
        sha256_b64: str,
        field: str,
    ) -> dict[str, Any]:
        """Call Langfuse media create endpoint.

        Aligns with public docs expecting keys: traceId, contentType, contentLength,
        sha256Hash, field. (Older internal draft used size/mimeType; removed.)
        """
        url = f"{self.base}/api/public/media"
        payload: dict[str, Any] = {
            "traceId": trace_id,
            "contentLength": content_length,
            "sha256Hash": sha256_b64,
            "field": field,
        }
        # observationId optional – when provided links media to observation so
        # UI observation view fetch (observation_media) returns it. Without it
        # media is only trace-scoped and won't appear inside span preview.
        if observation_id:
            payload["observationId"] = observation_id
        if content_type:
            payload["contentType"] = content_type
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
            return cast(dict[str, Any], resp.json())
        except Exception as e:  # pragma: no cover - unexpected response
            raise RuntimeError(f"invalid media create JSON: {e}") from e

    def patch_media_status(
        self,
        *,
        media_id: str,
        upload_status: int,
        upload_error: Optional[str] = None,
    ) -> None:
        """Patch upload status (finalization step).

        Mirrors docs: send uploadedAt, uploadHttpStatus, uploadHttpError. This may
        help the UI mark asset as available for preview immediately.
        """
        url = f"{self.base}/api/public/media/{media_id}"
        from datetime import datetime, timezone

        body = {
            "uploadedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "uploadHttpStatus": upload_status,
            "uploadHttpError": upload_error,
        }
        try:
            resp = httpx.patch(
                url,
                json=body,
                auth=self.auth,
                timeout=self.timeout,
            )
        except Exception as e:  # pragma: no cover - network failure
            raise RuntimeError(f"media status patch failed: {e}") from e
        if resp.status_code >= 400:
            raise RuntimeError(
                f"media status patch failed status={resp.status_code} body={resp.text[:300]}"
            )

    def upload_binary(
        self, upload_url: str, data: bytes, *, content_type: str, sha256_b64: str
    ) -> None:
        """Upload binary to presigned URL.

        The presigned URL (S3 style) includes X-Amz-SignedHeaders list; current
        Langfuse deployment signs at least: content-length, content-type, host,
        x-amz-checksum-sha256. We therefore MUST send both content-type and the
        checksum header or the request is rejected with AccessDenied.
        """
        headers = {
            "Content-Type": content_type,
            # S3 path: raw base64 encoded SHA256 (not hex) required.
            "x-amz-checksum-sha256": sha256_b64,
        }
        # Azure Blob presigned URLs (observed format: https://<acct>.blob.core.windows.net/...)
        # reject PUT without mandatory header x-ms-blob-type. Langfuse multi-provider
        # deployments may issue either S3 or Azure style URLs. We detect Azure by host
        # substring 'blob.core.windows.net' (scheme‑agnostic) and inject the header.
        # Block blobs are the standard type for arbitrary binary uploads.
        try:
            if "blob.core.windows.net" in upload_url:
                headers.setdefault("x-ms-blob-type", "BlockBlob")
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            resp = httpx.put(upload_url, content=data, headers=headers, timeout=self.timeout)
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
    # Use standard Langfuse source type. User preference: represent origin as base64_data_uri
    # (UI treats bytes/base64_data_uri/file equivalently for rendering).
    parts.append("source=base64_data_uri")
    inner = "|".join(parts)
    return f"@@@langfuseMedia:{inner}@@@"


class _SettingsProto(Protocol):  # pragma: no cover - structural typing helper
    ENABLE_MEDIA_UPLOAD: bool
    LANGFUSE_HOST: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    OTEL_EXPORTER_OTLP_TIMEOUT: int
    MEDIA_MAX_BYTES: int


def patch_and_upload_media(mapped: MappedTraceWithAssets, settings: _SettingsProto) -> None:
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

    # Always operate in simplified in-place surfacing mode (legacy modes removed).
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
        # Build create payload & call API
        try:
            sanitized_mime = _sanitize_mime(asset.mime_type)
            # Convert stored hex sha256 -> base64 digest required by API example
            try:
                sha_bytes = bytes.fromhex(asset.sha256)
                sha_b64 = base64.b64encode(sha_bytes).decode("ascii")
            except (ValueError, binascii.Error):  # pragma: no cover - malformed
                sha_b64 = ""
            # Derive field (default to output)
            field = "output"
            lowered_path = asset.field_path.lower()
            if lowered_path.startswith("input"):
                field = "input"
            elif lowered_path.startswith("metadata"):
                field = "metadata"
            # Use real OTLP span id when available; fallback to logical id for
            # backwards compatibility in tests (pre-export usage). The OTLP id
            # is required for Langfuse to associate media with observations.
            obs_id = getattr(span, "otel_span_id", None) or span.id
            # Use OTLP 32-hex trace id when available to ensure alignment with
            # observation rows; fallback to logical id only if export did not
            # populate it (e.g. dry-run).
            create_resp = client.create_media(
                trace_id=getattr(mapped.trace, "otel_trace_id_hex", None) or str(mapped.trace.id),
                observation_id=obs_id,
                content_type=sanitized_mime or "application/octet-stream",
                content_length=len(decoded),
                sha256_b64=sha_b64,
                field=field,
            )
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
            logger.debug(
                "media create_resp media_id=%s observation_id_sent=%s resp_keys=%s",
                create_resp.get("mediaId") or create_resp.get("id"),
                getattr(span, "otel_span_id", None) or span.id,
                list(create_resp.keys()),
            )
        # Server may return either `mediaId` (docs) or legacy `id`
        media_id = (
            str(create_resp.get("mediaId"))
            if create_resp.get("mediaId")
            else (str(create_resp.get("id")) if create_resp.get("id") else None)
        )
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
            upload_error: Optional[str] = None
            try:
                client.upload_binary(
                    upload_url,
                    decoded,
                    content_type=sanitized_mime or "application/octet-stream",
                    sha256_b64=sha_b64,
                )
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
            # Attempt status patch (best-effort). Failure recorded but does not abort.
            try:
                client.patch_media_status(
                    media_id=media_id,
                    upload_status=200,
                    upload_error=upload_error,
                )
            except Exception as e:  # pragma: no cover - patch failure path
                logger.warning(
                    "media status_patch_failed asset=%s media_id=%s code=%s err=%s",
                    asset.sha256,
                    media_id,
                    "status_patch_error",
                    e,
                )
                if span.metadata is not None:
                    span.metadata.setdefault("n8n.media.error_codes", []).append(
                        "status_patch_error"
                    )
        # media_id guaranteed non-None here
        token = _build_token(str(media_id), sanitized_mime or asset.mime_type)

        def _inplace_by_segments(root: Any, segments: list[str]) -> bool:
            try:
                if not segments:
                    return False
                cursor = root
                for seg in segments[:-1]:
                    if not isinstance(cursor, dict):
                        return False
                    cursor = cursor.get(seg)
                if not isinstance(cursor, dict):
                    return False
                leaf = segments[-1]
                val = cursor.get(leaf)
                if (
                    isinstance(val, dict)
                    and val.get("_media_pending")
                    and val.get("sha256") == asset.sha256
                ):
                    cursor[leaf] = token
                    return True
            except Exception:
                return False
            return False

        def _recursive_patch(obj: Any) -> int:
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
                        replaced += _recursive_patch(v)
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
                        replaced += _recursive_patch(v)
            return replaced

        replacements = 0
        if asset.original_path_segments:
            if _inplace_by_segments(parsed, asset.original_path_segments):
                replacements = 1
        if replacements == 0:  # fallback recursive search
            replacements = _recursive_patch(parsed)
        # Promote canonical binary slot token to shallow key (one-time) for preview.
        if (
            replacements > 0
            and isinstance(parsed, dict)
            and asset.field_path.startswith("binary.")
        ):
            slot = asset.field_path.split(".", 1)[1] if "." in asset.field_path else asset.field_path
            if slot and slot not in parsed:
                parsed[slot] = token
                if span.metadata is not None:
                    span.metadata["n8n.media.promoted_from_binary"] = True
        if span.metadata is not None and replacements > 0:
            # Mark that at least one media token is available at (or promoted to)
            # a surface location suitable for preview heuristics.
            span.metadata["n8n.media.preview_surface"] = True
        if replacements > 0:
            try:
                span.output = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                if span.metadata is not None:
                    current = span.metadata.get("n8n.media.asset_count") or 0
                    span.metadata["n8n.media.asset_count"] = int(current) + replacements
                    # Surface mode constant (removed configurability) for observability.
                    span.metadata["n8n.media.surface_mode"] = "inplace"
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

    # Legacy hoist pass removed – in-place surfacing ensures preview depth.


def relink_media_observations(mapped: MappedTraceWithAssets, settings: _SettingsProto) -> None:
    """Post-export idempotent relink ensuring observation-level media association.

    Rationale: The initial media create calls may occur *before* the OTLP exporter
    has flushed spans and the Langfuse backend has persisted observation rows.
    Some deployments appear to only create `trace_media` (trace-scoped) entries
    when the referenced observation id does not yet exist, resulting in tokens
    present in span output JSON but no previews inside the span-level Media UI.

    Calling create_media again *after* export (and a minimal delay) is safe:
    - If observation linkage already exists this is a deduplicated no-op.
    - If only trace-level linkage exists the backend can now add
      `observation_media` row, enabling previews.

    We do *not* re-upload bytes; dedupe path returns an id without uploadUrl.
    """
    if not settings.ENABLE_MEDIA_UPLOAD:
        return
    if not mapped.assets:
        return
    host = settings.LANGFUSE_HOST
    if not host or not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return
    # Small delay gives the batch span processor (if any) a chance to persist
    # observations before linking. Adjust if future async exporter semantics change.
    try:  # pragma: no cover - timing nondeterministic in tests
        import time
        time.sleep(0.25)
    except Exception:  # pragma: no cover - defensive
        pass
    client = _MediaClient(
        host=host,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        timeout=settings.OTEL_EXPORTER_OTLP_TIMEOUT,
    )
    span_index: Dict[tuple[str, int], Any] = {}
    for s in mapped.trace.spans:
        run_index = s.metadata.get("n8n.node.run_index") if s.metadata else None
        if isinstance(run_index, int):
            span_index[(s.name, run_index)] = s
    for asset in mapped.assets:
        span = span_index.get((asset.node_name, asset.run_index))
        if not span:
            continue
        # Reconstruct values (no need to decode content again).
        try:
            try:
                sha_bytes = bytes.fromhex(asset.sha256)
                sha_b64 = base64.b64encode(sha_bytes).decode("ascii")
            except Exception:  # pragma: no cover - malformed hash edge
                sha_b64 = ""
            field = "output"
            lowered_path = asset.field_path.lower()
            if lowered_path.startswith("input"):
                field = "input"
            elif lowered_path.startswith("metadata"):
                field = "metadata"
            client.create_media(
                trace_id=str(mapped.trace.id),
                observation_id=span.id,
                content_type=_sanitize_mime(asset.mime_type) or "application/octet-stream",
                content_length=asset.size_bytes,
                sha256_b64=sha_b64,
                field=field,
            )
            logger.debug(
                "media relink_ok media_sha256=%s span=%s field=%s", asset.sha256, span.id, field
            )
        except Exception as e:  # pragma: no cover - network / backend error path
            # Non-fatal: previews may still appear if original call sufficed.
            logger.debug(
                "media relink_failed media_sha256=%s span=%s err=%s", asset.sha256, span.id, e
            )


__all__ = [
    "BinaryAsset",
    "MappedTraceWithAssets",
    "patch_and_upload_media",
    "relink_media_observations",
]
