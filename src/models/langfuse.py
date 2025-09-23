from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class LangfuseUsage(BaseModel):
    """Canonical token usage counts.

    Mapper normalizes any incoming n8n `tokenUsage` variants (promptTokens/completionTokens/totalTokens
    OR prompt/completion/total OR already-normalized input/output/total) into this shape.
    Exporter maps (OTel GenAI spec current names, legacy names removed):
        input  -> gen_ai.usage.input_tokens
        output -> gen_ai.usage.output_tokens
        total  -> gen_ai.usage.total_tokens (still accepted in spec)
    """
    input: Optional[int] = None
    output: Optional[int] = None
    total: Optional[int] = None


class LangfuseSpan(BaseModel):
    """Logical span / observation prior to OTLP emission.

    Added optional fields (Priority 1):
    - usage (preferred) + deprecated token_usage alias for backwards compatibility
    - level (Langfuse severity mapping)
    - status_message (error message / diagnostic detail)
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

    @model_validator(mode="after")
    def _noop(self):  # type: ignore[override]
        return self


class LangfuseTrace(BaseModel):
    """Logical trace container.

    Added optional fields (Priority 2) for closer parity with Langfuse OTLP property mapping:
    - user_id, session_id, release, public, tags, version, environment
    - trace_input / trace_output (distinct from root span input/output; exporter may map these)
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
