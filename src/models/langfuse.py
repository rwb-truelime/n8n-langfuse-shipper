"""Pydantic models for the internal representation of Langfuse objects.

These models define the logical structure of traces, spans, and usage data
before they are converted into OpenTelemetry (OTLP) objects by the `shipper`.
They serve as the target data structure for the `mapper` module.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class LangfuseUsage(BaseModel):
    """Represents canonical token usage counts for a generation.

    The mapper normalizes various `tokenUsage` formats from n8n into this
    standardized structure. The exporter then maps these fields to the
    corresponding OTel GenAI semantic conventions.
    """
    input: Optional[int] = None
    output: Optional[int] = None
    total: Optional[int] = None


class LangfuseSpan(BaseModel):
    """Represents a single logical observation (span, generation, event, etc.).

    This model is the internal representation of a span before it is converted
    to an OTLP span. It includes core timing and hierarchy information, I/O
    payloads, and Langfuse-specific metadata like observation type, model, and
    token usage.
    """

    id: str
    trace_id: str
    parent_id: Optional[str] = None
    name: str
    start_time: datetime
    end_time: datetime
    observation_type: str = "span"  # "span" | "generation" | future: "event"
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    # Token usage (normalized input/output/total). See LangfuseUsage docstring.
    usage: Optional[LangfuseUsage] = None
    status: Optional[str] = None  # normalized business outcome (e.g. success, error)
    level: Optional[str] = None  # severity level (DEBUG|DEFAULT|WARNING|ERROR)
    status_message: Optional[str] = None  # human-readable status / error message
    # OTLP span id (16 hex chars) captured during export. This differs from the
    # deterministic logical UUIDv5 `id` above. Media uploads must use this
    # value for `observationId` so the Langfuse backend links media to the
    # persisted observation row. Absent until `export_trace` runs.
    otel_span_id: Optional[str] = None

    @model_validator(mode="after")
    def _noop(self) -> "LangfuseSpan":
        """No-op validator placeholder for future cross-field validation."""
        return self


class LangfuseTrace(BaseModel):
    """Represents a complete logical trace, containing one or more spans.

    This is the root object created by the `mapper` for each n8n execution. It
    holds all associated spans and trace-level metadata that will be attached
    to the root span during OTLP export.
    """

    id: str
    name: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    spans: List[LangfuseSpan] = Field(default_factory=list)
    # Removed generations list; generation semantics rely solely on spans with observation_type == "generation".
    # New optional trace-level identity & classification fields
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    release: Optional[str] = None
    public: Optional[bool] = None
    tags: List[str] = Field(default_factory=list)
    version: Optional[str] = None
    environment: Optional[str] = None
    trace_input: Optional[Any] = None
    trace_output: Optional[Any] = None
    # OTLP 32-hex trace id actually sent to exporter (human-embedded form). Set
    # during export; used by media create calls so traceId matches observation
    # rows. Raw logical id (execution id string) remains in `id`.
    otel_trace_id_hex: Optional[str] = None
