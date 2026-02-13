"""Shipper: converts internal Langfuse models to real OpenTelemetry spans and exports.

This module is the "L" (Load) in the ETL pipeline. It takes the LangfuseTrace
objects produced by the n8n_langfuse_shipper.mapper module and converts them into
OpenTelemetry (OTLP) spans. These spans are then exported to a configured Langfuse
OTLP endpoint.

Key responsibilities include:
- One-time initialization of the OTLP exporter with credentials and endpoint config.
- Mapping a LangfuseTrace object to a hierarchy of OTLP spans.
- Translating fields from internal LangfuseSpan models into OTLP attributes
  according to Langfuse conventions (e.g., langfuse.observation.type).
- Generating a human-readable but spec-compliant OTLP trace ID that embeds the
  original n8n execution ID for easy cross-referencing.
- Handling dry-run mode for testing and validation.
- Implementing a simple backpressure mechanism to prevent memory overruns during
  high-throughput exports.
"""
from __future__ import annotations

import base64
import logging
import re
import time
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    TraceState,
    set_span_in_context,
)

from .config import Settings
from .models.langfuse import LangfuseSpan, LangfuseTrace


def _order_children_for_export(
    trace_model: LangfuseTrace, root_id: str
) -> list[LangfuseSpan]:
    """Return child spans ordered so known parents are emitted first.

    Mapping preserves chronological ordering, but some valid parent spans
    (especially agent hierarchy fixups) can start later than their children.
    Exporting strictly in chronological order can therefore cause child spans
    to fall back to root parent context. This helper performs a deterministic
    dependency-first ordering over children while preserving original order when
    dependencies do not require movement.

    Args:
        trace_model: Trace model containing root + child spans.
        root_id: Root span id.

    Returns:
        Reordered list of child spans (excluding root).
    """
    original_children = list(trace_model.spans[1:])
    if not original_children:
        return []

    by_id: Dict[str, LangfuseSpan] = {span.id: span for span in original_children}
    pending = list(original_children)
    ordered: list[LangfuseSpan] = []
    created_ids: set[str] = {root_id}

    while pending:
        progressed = False
        next_pending: list[LangfuseSpan] = []

        for span in pending:
            parent_id = span.parent_id
            parent_known_child = parent_id in by_id if parent_id else False
            parent_ready = (
                parent_id is None
                or parent_id in created_ids
                or not parent_known_child
            )
            if parent_ready:
                ordered.append(span)
                created_ids.add(span.id)
                progressed = True
            else:
                next_pending.append(span)

        if not progressed:
            # Defensive cycle-breaker: append remaining spans in original order.
            ordered.extend(next_pending)
            break

        pending = next_pending

    return ordered

_initialized = False
logger = logging.getLogger(__name__)
__all__ = [
    "export_trace",
    "shutdown_exporter",
]


def _init_otel(settings: Settings) -> None:
    """Initialize the OpenTelemetry tracer provider and OTLP exporter.

    This function sets up the global tracer provider with a batch span processor
    and an OTLPSpanExporter configured for the Langfuse endpoint. It handles
    authentication by encoding the public and secret keys into a Basic Auth
    header.

    The initialization is idempotent and will only run once.

    Args:
        settings: The application settings containing configuration for the
            Langfuse host, keys, and OTLP exporter parameters.
    """
    global _initialized
    if _initialized:
        return
    # Determine correct OTLP trace endpoint.
    # Langfuse base host (e.g. https://cloud.langfuse.com) expects /api/public/otel/v1/traces
    # Some users may already provide a partial or full path; normalize minimally.
    endpoint: str | None
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/")
    else:
        base = settings.LANGFUSE_HOST.rstrip("/") if settings.LANGFUSE_HOST else ""
        if base and not base.endswith("/api/public/otel"):
            # If user already appended /api/public/otel/v1/traces we won't duplicate
            if base.endswith("/api/public/otel/v1/traces"):
                endpoint = base
            else:
                endpoint = base + "/api/public/otel/v1/traces"
        else:
            endpoint = base + "/v1/traces" if base else None
    if not endpoint:
        logger.warning("No Langfuse endpoint configured; skipping OTLP initialization")
        return
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        logger.warning("Missing Langfuse keys; skipping OTLP initialization")
        return
    auth_raw = f"{settings.LANGFUSE_PUBLIC_KEY}:{settings.LANGFUSE_SECRET_KEY}".encode()
    auth_b64 = base64.b64encode(auth_raw).decode()
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={"Authorization": f"Basic {auth_b64}"},
        timeout=settings.OTEL_EXPORTER_OTLP_TIMEOUT,
    )
    resource = Resource.create(
        {
            "service.name": "n8n-langfuse-shipper",
            "service.version": "0.1.0",  # TODO: optionally load from package metadata
            "telemetry.sdk.language": "python",
            "telemetry.auto.version": "manual",  # indicates manual instrumentation
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            exporter,
            max_queue_size=settings.OTEL_MAX_QUEUE_SIZE,
            max_export_batch_size=settings.OTEL_MAX_EXPORT_BATCH_SIZE,
            schedule_delay_millis=settings.OTEL_SCHEDULED_DELAY_MILLIS,
        )
    )
    trace.set_tracer_provider(provider)
    _initialized = True
    logger.info("Initialized OTLP exporter for Langfuse endpoint %s", endpoint)


def _apply_span_attributes(span_ot: Any, span_model: LangfuseSpan) -> None:
    """Map attributes from a LangfuseSpan model to an OTel span.

    This function translates the fields of the internal `LangfuseSpan` data model
    into the corresponding OpenTelemetry attributes, following the conventions
    expected by Langfuse.

    Args:
        span_ot: The OpenTelemetry Span object to which attributes will be applied.
        span_model: The internal `LangfuseSpan` model containing the data.
    """
    span_ot.set_attribute("langfuse.observation.type", span_model.observation_type)
    if span_model.model:
        span_ot.set_attribute("langfuse.observation.model.name", span_model.model)
        span_ot.set_attribute("model", span_model.model)
    # Emit prompt linking attributes for generation spans
    if span_model.prompt_name and span_model.prompt_version is not None:
        span_ot.set_attribute(
            "langfuse.observation.prompt.name", span_model.prompt_name
        )
        span_ot.set_attribute(
            "langfuse.observation.prompt.version", span_model.prompt_version
        )
    if span_model.status:
        span_ot.set_attribute("langfuse.observation.status", span_model.status)
    # Explicit level / status_message provided (non-error spans included)
    # Optional level / status_message (fields present on model)
    if getattr(span_model, "level", None):  # guard for older serialized variants
        span_ot.set_attribute("langfuse.observation.level", span_model.level)
    if getattr(span_model, "status_message", None):
        span_ot.set_attribute(
            "langfuse.observation.status_message", span_model.status_message
        )
    if span_model.usage:
        if span_model.usage.input is not None:
            span_ot.set_attribute("gen_ai.usage.input_tokens", span_model.usage.input)
        if span_model.usage.output is not None:
            span_ot.set_attribute("gen_ai.usage.output_tokens", span_model.usage.output)
        if span_model.usage.total is not None:
            span_ot.set_attribute("gen_ai.usage.total_tokens", span_model.usage.total)
        # Emit consolidated usage_details JSON with only present *_tokens keys.
        usage_details = {}
        if span_model.usage.input is not None:
            usage_details["input_tokens"] = span_model.usage.input
        if span_model.usage.output is not None:
            usage_details["output_tokens"] = span_model.usage.output
        if span_model.usage.total is not None:
            usage_details["total_tokens"] = span_model.usage.total
        if usage_details:
            import json as _json
            span_ot.set_attribute(
                "langfuse.observation.usage_details",
                _json.dumps(usage_details, separators=(",", ":")),
            )
    # Basic metadata mapping
    for k, v in (span_model.metadata or {}).items():
        if v is not None:
            span_ot.set_attribute(f"langfuse.observation.metadata.{k}", str(v))
    if span_model.error:
        # Only override level/status_message if not explicitly set above
        if "langfuse.observation.level" not in getattr(
            span_ot, "attributes", {}
        ):  # defensive for dummy test spans
            span_ot.set_attribute("langfuse.observation.level", "ERROR")
        if not span_model.status_message:
            span_ot.set_attribute(
                "langfuse.observation.status_message", str(span_model.error)
            )


_trace_export_count = 0
_total_spans_created = 0
_last_flushed_spans_created = 0


def _extract_trace_id_from_workflow_data(
    serialized_data: str,
    langfuse_trace_id_field_name: str,
) -> str:
    """Extract custom trace ID from workflowData JSON field if specified.

    The provided trace ID needs to be a valid hex string of length 16.

    Args:
        serialized_data: json serialized data to search within
        langfuse_trace_id_field_name: Field name to extract trace ID from
    """
    match_pattern = re.compile(
        r'"' + re.escape(langfuse_trace_id_field_name) + r'\\?"\s*:\s*\\?"([^\\"]+)\\?"'
    )
    match = match_pattern.search(serialized_data)
    if match is None:
        raise ValueError(
            f"Custom trace ID field '{langfuse_trace_id_field_name}' not found in workflow data."
        )
    trace_id = match.group(1)

    try:
        int(trace_id, 16)
    except ValueError as err:
        raise ValueError(
            f"Extracted trace ID '{trace_id}' is not a valid hexadecimal string."
        ) from err

    logger.debug(
        "Extracted custom trace ID '%s' from field '%s'.",
        trace_id,
        langfuse_trace_id_field_name,
    )
    return trace_id


def _build_human_trace_id(execution_id_str: str) -> tuple[int, str]:
    """Return (int_trace_id, hex_string) embedding the decimal execution id digits.

    OTel requires a 16-byte (32 hex char) trace id. We embed the raw decimal digits at the end
    and left-pad with zeros. This keeps the execution id visually searchable (suffix match)
    while remaining spec compliant.
    """
    digits = ''.join(ch for ch in execution_id_str if ch.isdigit()) or '0'
    if len(digits) > 32:
        # Very unlikely (n8n execution ids won't be this long); keep last 32 digits
        digits = digits[-32:]
    hex_str = digits.rjust(32, '0')  # still valid hex (digits only)
    return int(hex_str, 16), hex_str


def export_trace(
    trace_model: LangfuseTrace,
    settings: Settings,
    dry_run: bool = True,
    langfuse_trace_id_field_name: str | None = None,
) -> None:
    """Export a LangfuseTrace object as a collection of OTLP spans.

    This function orchestrates the conversion of a single `LangfuseTrace` into
    a root OTLP span and its children. It sets up the parent-child relationships,
    applies all attributes, and manages the lifecycle of the spans.

    A simple backpressure mechanism is included: if the number of unflushed spans
    exceeds a soft limit, the function will sleep briefly to allow the exporter
    to catch up.

    Args:
        trace_model: The internal trace model to be exported.
        settings: Application settings, used for OTLP initialization and
            backpressure configuration.
        dry_run: If True, logs the intent to export but does not send any data.
            If False, initializes OTLP (if needed) and exports the trace.
        langfuse_trace_id_field_name: Optional field name of an externally
            provided langfuse trace ID.
    """
    trace_id = trace_model.id
    if langfuse_trace_id_field_name:
        try:
            trace_id = _extract_trace_id_from_workflow_data(
                trace_model.model_dump_json(), langfuse_trace_id_field_name
            )
        except ValueError as e:
            logger.warning(
                "Failed to extract custom trace ID from field '%s'. "
                "Defaulting to execution ID '%s' as trace ID. %s",
                langfuse_trace_id_field_name,
                trace_model.id,
                e
            )

    if dry_run:
        logging.getLogger(__name__).info(
            "Dry-run export: trace_id=%s spans=%d", trace_id, len(trace_model.spans)
        )
        return
    _init_otel(settings)
    tracer = trace.get_tracer(__name__)

    # Keep root span open until after children finish to reduce risk of dropped root under load
    if not trace_model.spans:
        return
    global _trace_export_count
    root_model = trace_model.spans[0]

    try:
        if trace_id:
            int_trace_id = int(trace_id, 16)
            hex_trace_id = trace_id
            logger.debug("Using user-provided trace ID: %s", trace_id)
        else:
            # Build deterministic human-searchable trace id: 000...<executionId>
            int_trace_id, hex_trace_id = _build_human_trace_id(trace_model.id)
        try:
            trace_model.otel_trace_id_hex = hex_trace_id
        except Exception:  # pragma: no cover - defensive
            pass
        # Derive a deterministic parent span id seed from the hex trace id
        import hashlib
        parent_seed = hashlib.sha256(
            (hex_trace_id + ':parent').encode()
        ).hexdigest()[:16]
        parent_span_id = int(parent_seed, 16)
        parent_sc = SpanContext(
            trace_id=int_trace_id,
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=TraceState(),
        )
        parent_ctx = set_span_in_context(NonRecordingSpan(parent_sc))
    except Exception:  # pragma: no cover - fallback safety
        parent_ctx = None
    start_ns_root = int(root_model.start_time.timestamp() * 1e9)
    end_ns_root = int(root_model.end_time.timestamp() * 1e9)
    root_span = tracer.start_span(
        name=root_model.name, start_time=start_ns_root, context=parent_ctx
    )
    try:  # Capture OTLP span id for downstream media linkage
        root_model.otel_span_id = format(root_span.get_span_context().span_id, "016x")
    except Exception:  # pragma: no cover - defensive
        root_model.otel_span_id = None
    _apply_span_attributes(root_span, root_model)
    # trace-level attributes
    for k, v in trace_model.metadata.items():
        root_span.set_attribute(f"langfuse.trace.metadata.{k}", str(v))
    root_span.set_attribute("langfuse.trace.name", trace_model.name)
    # Mark root span explicitly
    # Updated per Langfuse constants: internal root marker
    root_span.set_attribute("langfuse.internal.as_root", True)
    # Trace identity / classification fields (only if provided)
    # Updated to canonical user/session ids per Langfuse constants (user.id / session.id)
    if trace_model.user_id is not None:
        root_span.set_attribute("user.id", trace_model.user_id)
    if trace_model.session_id is not None:
        root_span.set_attribute("session.id", trace_model.session_id)
    if trace_model.tags:
        import json as _json
        root_span.set_attribute(
            "langfuse.trace.tags",
            _json.dumps(trace_model.tags, separators=(",", ":")),
        )
    if trace_model.trace_input is not None:
        import json as _json
        try:
            root_span.set_attribute(
                "langfuse.trace.input",
                _json.dumps(trace_model.trace_input, separators=(",", ":")),
            )
        except Exception:
            root_span.set_attribute("langfuse.trace.input", str(trace_model.trace_input))
    if trace_model.trace_output is not None:
        import json as _json
        try:
            root_span.set_attribute(
                "langfuse.trace.output",
                _json.dumps(trace_model.trace_output, separators=(",", ":")),
            )
        except Exception:
            root_span.set_attribute("langfuse.trace.output", str(trace_model.trace_output))
    wf_id_val = trace_model.metadata.get("workflowId")
    if wf_id_val is not None:
        root_span.set_attribute("langfuse.workflow.id", wf_id_val)
    if root_model.input is not None:
        import json as _json
        try:
            root_span.set_attribute(
                "langfuse.observation.input",
                _json.dumps(root_model.input, separators=(",", ":"), ensure_ascii=False)
            )
        except Exception:
            root_span.set_attribute("langfuse.observation.input", str(root_model.input))
    if root_model.output is not None:
        import json as _json
        try:
            root_span.set_attribute(
                "langfuse.observation.output",
                _json.dumps(root_model.output, separators=(",", ":"), ensure_ascii=False)
            )
        except Exception:
            root_span.set_attribute("langfuse.observation.output", str(root_model.output))
    ctx_lookup: Dict[str, SpanContext] = {root_model.id: root_span.get_span_context()}

    # Children
    ordered_children = _order_children_for_export(trace_model, root_model.id)
    for child in ordered_children:
        # Resolve parent span context (default to root if missing / None)
        if child.parent_id and child.parent_id in ctx_lookup:
            parent_span_context = ctx_lookup[child.parent_id]
        else:
            parent_span_context = root_span.get_span_context()
        parent_ctx = set_span_in_context(NonRecordingSpan(parent_span_context))
        start_ns = int(child.start_time.timestamp() * 1e9)
        end_ns = int(child.end_time.timestamp() * 1e9)
        span_ot = tracer.start_span(
            name=child.name,
            context=parent_ctx,
            start_time=start_ns,
        )
        try:  # Persist OTLP span id (needed for media observationId)
            child.otel_span_id = format(span_ot.get_span_context().span_id, "016x")
        except Exception:  # pragma: no cover - defensive
            child.otel_span_id = None
        _apply_span_attributes(span_ot, child)
        if child.input is not None:
            import json as _json
            try:
                span_ot.set_attribute(
                    "langfuse.observation.input",
                    _json.dumps(child.input, separators=(",", ":"), ensure_ascii=False)
                )
            except Exception:
                span_ot.set_attribute("langfuse.observation.input", str(child.input))
        if child.output is not None:
            import json as _json
            try:
                span_ot.set_attribute(
                    "langfuse.observation.output",
                    _json.dumps(child.output, separators=(",", ":"), ensure_ascii=False)
                )
            except Exception:
                span_ot.set_attribute("langfuse.observation.output", str(child.output))
        span_ot.end(end_time=end_ns)
        ctx_lookup[child.id] = span_ot.get_span_context()

    # End root last
    root_span.end(end_time=end_ns_root)

    global _total_spans_created, _last_flushed_spans_created
    _trace_export_count += 1
    _total_spans_created += len(trace_model.spans)
    flushed_this_cycle = False
    if _trace_export_count % max(1, settings.FLUSH_EVERY_N_TRACES) == 0:
        try:
            provider = trace.get_tracer_provider()
            flush_fn = getattr(provider, "force_flush", None)
            if callable(flush_fn):
                flush_fn()
                _last_flushed_spans_created = _total_spans_created
                flushed_this_cycle = True
        except Exception:  # pragma: no cover
            logger.debug("Error forcing flush", exc_info=True)
    # Backpressure: approximate unflushed backlog
    backlog = _total_spans_created - _last_flushed_spans_created
    if backlog > settings.EXPORT_QUEUE_SOFT_LIMIT:
        sleep_s = max(0, settings.EXPORT_SLEEP_MS) / 1000.0
        logger.debug(
            "Backpressure sleep: backlog=%d soft_limit=%d sleep_ms=%d flushed_now=%s",
            backlog,
            settings.EXPORT_QUEUE_SOFT_LIMIT,
            settings.EXPORT_SLEEP_MS,
            flushed_this_cycle,
        )
        time.sleep(sleep_s)


def shutdown_exporter() -> None:  # pragma: no cover - simple shutdown hook
    """Flush any buffered spans and shut down the OTLP exporter.

    This function should be called at the end of the application's lifecycle to
    ensure that all telemetry data is sent before the process exits.
    """
    provider = trace.get_tracer_provider()
    try:  # provider from global tracer; attributes dynamic in SDK
        shut_fn = getattr(provider, "shutdown", None)
        if callable(shut_fn):
            shut_fn()
    except Exception:  # pragma: no cover
        logger.debug("Error during tracer provider shutdown", exc_info=True)
