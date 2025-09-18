"""Shipper: converts internal Langfuse models to real OpenTelemetry spans and exports.

Iteration 2: minimal implementation creating a trace with correct timing. More advanced
attribute enrichment and media handling comes later.
"""
from __future__ import annotations

import base64
import logging
from typing import Dict
from datetime import datetime

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, set_span_in_context

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
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
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
    if span_model.token_usage:
        if span_model.token_usage.promptTokens is not None:
            span_ot.set_attribute("gen_ai.usage.prompt_tokens", span_model.token_usage.promptTokens)
        if span_model.token_usage.completionTokens is not None:
            span_ot.set_attribute(
                "gen_ai.usage.completion_tokens", span_model.token_usage.completionTokens
            )
        if span_model.token_usage.totalTokens is not None:
            span_ot.set_attribute("gen_ai.usage.total_tokens", span_model.token_usage.totalTokens)
    # Basic metadata mapping
    for k, v in (span_model.metadata or {}).items():
        if v is not None:
            span_ot.set_attribute(f"langfuse.observation.metadata.{k}", str(v))
    if span_model.error:
        span_ot.set_attribute("langfuse.observation.level", "ERROR")
        span_ot.set_attribute("langfuse.observation.status_message", str(span_model.error))


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

    # mapping of span id -> context for parenting
    ctx_lookup: Dict[str, object] = {}
    # Ensure root first (simple ordering assumption: root is first element)
    for span_model in trace_model.spans:
        parent_ctx = None
        if span_model.parent_id and span_model.parent_id in ctx_lookup:
            from opentelemetry.trace import SpanContext  # local import to avoid circular
            parent_span_context_obj = ctx_lookup[span_model.parent_id]
            if isinstance(parent_span_context_obj, SpanContext):
                parent_ctx = set_span_in_context(NonRecordingSpan(parent_span_context_obj))
        start_ns = int(span_model.start_time.timestamp() * 1e9)
        end_ns = int(span_model.end_time.timestamp() * 1e9)
        span_ot = tracer.start_span(
            name=span_model.name,
            context=parent_ctx,  # type: ignore[arg-type]
            start_time=start_ns,
        )
        _apply_span_attributes(span_ot, span_model)
        # Optionally attach (possibly truncated) input/output as attributes for visibility
        if span_model.input is not None:
            span_ot.set_attribute("langfuse.observation.input", str(span_model.input))
        if span_model.output is not None:
            span_ot.set_attribute("langfuse.observation.output", str(span_model.output))
        # Set trace-level attributes on root
        if span_model.parent_id is None:
            for k, v in trace_model.metadata.items():
                span_ot.set_attribute(f"langfuse.trace.metadata.{k}", str(v))
            span_ot.set_attribute("langfuse.trace.name", trace_model.name)
        span_ot.end(end_time=end_ns)
        ctx_lookup[span_model.id] = span_ot.get_span_context()
