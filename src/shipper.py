"""Shipper: converts internal Langfuse models to real OpenTelemetry spans and exports.

Iteration 2: minimal implementation creating a trace with correct timing. More advanced
attribute enrichment and media handling comes later.
"""
from __future__ import annotations

import base64
import time
import logging
from typing import Dict
from datetime import datetime

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, set_span_in_context, SpanContext, TraceFlags, TraceState

from .models.langfuse import LangfuseTrace, LangfuseSpan
from .config import Settings

_initialized = False
logger = logging.getLogger(__name__)


def _init_otel(settings: Settings) -> None:
    global _initialized
    if _initialized:
        return
    # Determine correct OTLP trace endpoint.
    # Langfuse base host (e.g. https://cloud.langfuse.com) expects /api/public/otel/v1/traces
    # Some users may already provide a partial or full path; normalize minimally.
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


def _apply_span_attributes(span_ot, span_model: LangfuseSpan) -> None:  # type: ignore
    span_ot.set_attribute("langfuse.observation.type", span_model.observation_type)
    if span_model.model:
        span_ot.set_attribute("langfuse.observation.model.name", span_model.model)
        span_ot.set_attribute("model", span_model.model)
    if span_model.status:
        span_ot.set_attribute("langfuse.observation.status", span_model.status)
    # Explicit level / status_message provided (non-error spans included)
    if span_model.level:
        span_ot.set_attribute("langfuse.observation.level", span_model.level)
    if span_model.status_message:
        span_ot.set_attribute("langfuse.observation.status_message", span_model.status_message)
    if span_model.usage:
        if span_model.usage.input is not None:
            span_ot.set_attribute("gen_ai.usage.input_tokens", span_model.usage.input)
        if span_model.usage.output is not None:
            span_ot.set_attribute("gen_ai.usage.output_tokens", span_model.usage.output)
        if span_model.usage.total is not None:
            span_ot.set_attribute("gen_ai.usage.total_tokens", span_model.usage.total)
        # Emit consolidated JSON usage_details with only present keys (Langfuse parity) using *_tokens names
        usage_details = {}
        if span_model.usage.input is not None:
            usage_details["input_tokens"] = span_model.usage.input
        if span_model.usage.output is not None:
            usage_details["output_tokens"] = span_model.usage.output
        if span_model.usage.total is not None:
            usage_details["total_tokens"] = span_model.usage.total
        if usage_details:
            import json as _json
            span_ot.set_attribute("langfuse.observation.usage_details", _json.dumps(usage_details, separators=(",", ":")))
    # Basic metadata mapping
    for k, v in (span_model.metadata or {}).items():
        if v is not None:
            span_ot.set_attribute(f"langfuse.observation.metadata.{k}", str(v))
    if span_model.error:
        # Only override level/status_message if not explicitly set above
        if "langfuse.observation.level" not in getattr(span_ot, "attributes", {}):  # defensive for dummy test spans
            span_ot.set_attribute("langfuse.observation.level", "ERROR")
        if not span_model.status_message:
            span_ot.set_attribute("langfuse.observation.status_message", str(span_model.error))


_trace_export_count = 0
_total_spans_created = 0
_last_flushed_spans_created = 0


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


def export_trace(trace_model: LangfuseTrace, settings: Settings, dry_run: bool = True) -> None:
    """Export a LangfuseTrace via OTLP.

    dry_run: if True, skip actual OTLP network send.
    """
    if dry_run:
        logging.getLogger(__name__).info(
            "Dry-run export: trace_id=%s spans=%d", trace_model.id, len(trace_model.spans)
        )
        return
    _init_otel(settings)
    tracer = trace.get_tracer(__name__)

    # Keep root span open until after children finish to reduce risk of dropped root under load
    if not trace_model.spans:
        return
    global _trace_export_count
    root_model = trace_model.spans[0]
    # Build deterministic human-searchable trace id: 000...<executionId>
    try:
        int_trace_id, hex_trace_id = _build_human_trace_id(trace_model.id)
        # Derive a deterministic parent span id seed from the hex trace id
        import hashlib
        parent_seed = hashlib.sha256((hex_trace_id + ':parent').encode()).hexdigest()[:16]
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
        parent_ctx = None  # type: ignore
    start_ns_root = int(root_model.start_time.timestamp() * 1e9)
    end_ns_root = int(root_model.end_time.timestamp() * 1e9)
    root_span = tracer.start_span(name=root_model.name, start_time=start_ns_root, context=parent_ctx)  # type: ignore[arg-type]
    _apply_span_attributes(root_span, root_model)
    # trace-level attributes
    for k, v in trace_model.metadata.items():
        root_span.set_attribute(f"langfuse.trace.metadata.{k}", str(v))
    root_span.set_attribute("langfuse.trace.name", trace_model.name)
    # Mark root span explicitly
    root_span.set_attribute("langfuse.as_root", True)
    # Trace identity / classification fields (only if provided)
    if trace_model.user_id is not None:
        root_span.set_attribute("langfuse.trace.user_id", trace_model.user_id)
    if trace_model.session_id is not None:
        root_span.set_attribute("langfuse.trace.session_id", trace_model.session_id)
    if trace_model.tags:
        import json as _json
        root_span.set_attribute("langfuse.trace.tags", _json.dumps(trace_model.tags, separators=(",", ":")))
    if trace_model.trace_input is not None:
        import json as _json
        try:
            root_span.set_attribute("langfuse.trace.input", _json.dumps(trace_model.trace_input, separators=(",", ":")))
        except Exception:
            root_span.set_attribute("langfuse.trace.input", str(trace_model.trace_input))
    if trace_model.trace_output is not None:
        import json as _json
        try:
            root_span.set_attribute("langfuse.trace.output", _json.dumps(trace_model.trace_output, separators=(",", ":")))
        except Exception:
            root_span.set_attribute("langfuse.trace.output", str(trace_model.trace_output))
    wf_id_val = trace_model.metadata.get("workflowId")
    if wf_id_val is not None:
        root_span.set_attribute("langfuse.workflow.id", wf_id_val)
    if root_model.input is not None:
        root_span.set_attribute("langfuse.observation.input", str(root_model.input))
    if root_model.output is not None:
        root_span.set_attribute("langfuse.observation.output", str(root_model.output))
    ctx_lookup: Dict[str, SpanContext] = {root_model.id: root_span.get_span_context()}

    # Children
    for child in trace_model.spans[1:]:
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
            context=parent_ctx,  # type: ignore[arg-type]
            start_time=start_ns,
        )
        _apply_span_attributes(span_ot, child)
        if child.input is not None:
            span_ot.set_attribute("langfuse.observation.input", str(child.input))
        if child.output is not None:
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
            if hasattr(provider, "force_flush"):
                provider.force_flush()  # type: ignore[attr-defined]
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


def shutdown_exporter():  # pragma: no cover - simple shutdown hook
    """Flush and shutdown tracer provider (call at end of short-lived process)."""
    provider = trace.get_tracer_provider()
    try:
        if hasattr(provider, "shutdown"):
            provider.shutdown()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        logger.debug("Error during tracer provider shutdown", exc_info=True)
