"""Application configuration management using Pydantic Settings.

This module defines the `Settings` class, which loads configuration parameters
from environment variables and a `.env` file. It centralizes all tunable
parameters, from database connection details to logging levels and exporter
behavior.

The `get_settings` function provides a cached, singleton instance of the
configuration, ensuring consistent settings throughout the application.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict as _SettingsConfigDict


class Settings(BaseSettings):
    """Defines all application configuration parameters.

    This class uses `pydantic-settings` to automatically load values from
    environment variables or a `.env` file. It includes a validator to
    dynamically construct the PostgreSQL DSN from component parts if it's not
    provided directly, offering flexible connection configuration.
    """

    model_config = _SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Postgres
    PG_DSN: str = Field(
        default="", description="PostgreSQL DSN, e.g. postgres://user:pass@host:5432/dbname"
    )
    # Optional n8n-style component vars (used if PG_DSN not directly supplied)
    DB_POSTGRESDB_HOST: Optional[str] = None
    DB_POSTGRESDB_PORT: Optional[int] = 5432
    DB_POSTGRESDB_DATABASE: Optional[str] = None
    DB_POSTGRESDB_USER: Optional[str] = None
    DB_POSTGRESDB_PASSWORD: Optional[str] = None
    DB_POSTGRESDB_SCHEMA: Optional[str] = None
    # Mandatory prefix (empty string means none). Must be provided explicitly.
    DB_TABLE_PREFIX: str = Field(
        description="Mandatory table prefix. Set 'n8n_' explicitly or blank for none."
    )

    # Langfuse / OTLP
    LANGFUSE_HOST: str = Field(default="", description="Base URL for Langfuse host")
    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="Langfuse public key")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="Langfuse secret key")

    # Logging & runtime behavior
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    FETCH_BATCH_SIZE: int = Field(default=100, description="Batch size for fetching executions")
    # NOTE: Default truncation disabled per design choice: keep full textual
    # JSON except binary/base64
    TRUNCATE_FIELD_LEN: int = Field(
        default=0,
        description=(
            "Max length for large text fields (input/output) before truncation "
            "(0 = disabled)"
        ),
    )

    # Future: additional tuning parameters
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(
        default=None, description="Override OTLP endpoint (otherwise derived from LANGFUSE_HOST)"
    )
    OTEL_EXPORTER_OTLP_TIMEOUT: int = Field(
        default=30, description="Timeout (seconds) for OTLP HTTP export requests"
    )

    # Checkpointing
    CHECKPOINT_FILE: str = Field(
        default=".backfill_checkpoint",
        description="Path to file storing last processed execution id",
    )

    # ---------------- Execution Processing Behavior -----------------
    DRY_RUN: bool = Field(
        default=True,
        description="If true, do not send spans to Langfuse (mapping only, no export)",
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable verbose debug logging for execution data parsing",
    )
    ATTEMPT_DECOMPRESS: bool = Field(
        default=False,
        description="Attempt decompression of execution data payloads (currently placeholder)",
    )
    DEBUG_DUMP_DIR: Optional[str] = Field(
        default=None,
        description="Directory to dump raw execution data JSON when debug enabled",
    )

    # Filtering: only process executions that have at least one metadata row
    # whose key='executionId' and whose value equals the execution id.
    REQUIRE_EXECUTION_METADATA: bool = Field(
        default=False,
        description=(
            "If true, only executions having a row in <prefix>execution_metadata "
            "where key='executionId' and value matches the execution id are fetched."
        ),
    )

    # Phase 1 export reliability controls
    FLUSH_EVERY_N_TRACES: int = Field(
        default=1,
        description="Force flush the exporter after every N traces (1 = after each trace)",
    )
    OTEL_MAX_QUEUE_SIZE: int = Field(
        default=10000, description="BatchSpanProcessor max queue size for spans"
    )
    OTEL_MAX_EXPORT_BATCH_SIZE: int = Field(
        default=512, description="Maximum number of spans per export batch"
    )
    OTEL_SCHEDULED_DELAY_MILLIS: int = Field(
        default=200, description="BatchSpanProcessor scheduled delay in milliseconds before a flush"
    )
    EXPORT_QUEUE_SOFT_LIMIT: int = Field(
        default=5000,
        description=(
            "Approximate soft limit of spans queued (created - "
            "last_flushed_estimate) before introducing a short sleep"
        ),
    )
    EXPORT_SLEEP_MS: int = Field(
        default=75,
        description=(
            "Sleep duration in milliseconds when soft queue limit exceeded "
            "(backpressure throttle)"
        ),
    )

    # ---------------- Media / Binary Upload (Langfuse Media API) -----------------
    # BREAKING CHANGE: Azure Blob direct upload path removed. We now rely on
    # Langfuse Media API (create + optional presigned upload) and emit token
    # references. Only a single flag is required besides host + auth keys.
    ENABLE_MEDIA_UPLOAD: bool = Field(
        default=False,
        description=(
            "Feature flag: when true, collect binary assets and invoke Langfuse Media API to "
            "produce media tokens. When false, legacy unconditional redaction remains."
        ),
    )
    # Maximum size (bytes) of a single media object to upload. Large blobs are
    # left as redacted placeholders if exceeded.
    MEDIA_MAX_BYTES: int = Field(
        default=25_000_000,
        description="Maximum decoded binary size (bytes) allowed for upload (default 25MB)",
    )
    # Cap for extended (non-canonical) media asset discovery to avoid runaway
    # scanning when many small base64 snippets exist. 0 or negative disables
    # the cap (not recommended for very large executions).
    EXTENDED_MEDIA_SCAN_MAX_ASSETS: int = Field(
        default=250,
        description=(
            "Maximum number of non-canonical discovered binary assets to collect per node run "
            "(to bound CPU & memory). Set <=0 to disable the cap."
        ),
    )

    # ---------------- Root Span I/O Surfacing -----------------
    # User-configurable node names whose LAST run input/output are copied verbatim
    # (already normalized/truncated/binary-sanitized) onto the root span.
    # Empty / unset values disable each direction independently. Matching is
    # case-insensitive with surrounding whitespace trimmed for ergonomic UX.
    ROOT_SPAN_INPUT_NODE: Optional[str] = Field(
        default=None,
        description=(
            "Optional node name whose LAST run input will populate the root span input. "
            "Case-insensitive; surrounding whitespace ignored."
        ),
    )
    ROOT_SPAN_OUTPUT_NODE: Optional[str] = Field(
        default=None,
        description=(
            "Optional node name whose LAST run output will populate the root span output. "
            "Case-insensitive; surrounding whitespace ignored."
        ),
    )

    # ---------------- Filtering (AI-only spans) -----------------
    FILTER_AI_ONLY: bool = Field(
        default=False,
        description=(
            "If true, only export spans for AI-related nodes (nodes from the "
            "@n8n/n8n-nodes-langchain package). Root span always included; "
            "non-AI parents on path to AI nodes preserved. Executions with no "
            "AI nodes export root only with n8n.filter.no_ai_spans=true."
        ),
    )
    # General workflow id filtering (independent of AI-only). Optional list of
    # workflowId values; when non-empty only executions whose workflowId is in
    # this list are fetched. Empty list means no filtering (process all).
    FILTER_WORKFLOW_IDS: Any = Field(
        default_factory=list,
        description=(
            "Optional comma-separated list of workflowId values to restrict "
            "processing to. Example: FILTER_WORKFLOW_IDS=abc123,def456. Empty "
            "list (default) means no workflowId filtering. This filter applies "
            "at DB fetch stage and is orthogonal to FILTER_AI_ONLY."
        ),
    )

    # ---------------- Node Extraction (for AI-only filtering) -----------------
    # Use Any type to prevent Pydantic Settings JSON decoding; validator converts to list[str]
    FILTER_AI_EXTRACTION_NODES: Any = Field(
        default_factory=list,
        description=(
            "Comma-separated list of node names to extract input/output data from "
            "when FILTER_AI_ONLY is enabled. Data stored in root span metadata "
            "under n8n.extracted_nodes.<node_name>. Empty list = no extraction."
        ),
    )
    FILTER_AI_EXTRACTION_INCLUDE_KEYS: Any = Field(
        default_factory=list,
        description=(
            "Optional wildcard patterns for keys to include in extracted data. "
            "Supports glob patterns (e.g., userId, *Id, headers.*). "
            "Empty list = extract all keys (subject to exclude patterns)."
        ),
    )
    FILTER_AI_EXTRACTION_EXCLUDE_KEYS: Any = Field(
        default_factory=list,
        description=(
            "Optional wildcard patterns for keys to exclude from extracted data. "
            "Supports glob patterns (e.g., *password*, *token*, *secret*). "
            "Applied after include patterns. Empty list = no exclusions."
        ),
    )
    FILTER_AI_EXTRACTION_MAX_VALUE_LEN: int = Field(
        default=10000,
        description=(
            "Maximum string length for any single value in extracted node data. "
            "Values exceeding this are truncated with suffix marker. "
            "Prevents metadata bloat. Default 10KB."
        ),
    )

    # Media surfacing: always in-place replacement (legacy/mirror modes removed
    # for simplicity per project policy of no backward compatibility). Tokens
    # are inserted exactly where base64 strings were, with shallow promotion of
    # canonical binary slots. (Former MEDIA_SURFACE_MODE removed.)

    @field_validator(
        "FILTER_AI_EXTRACTION_NODES",
        "FILTER_AI_EXTRACTION_INCLUDE_KEYS",
        "FILTER_AI_EXTRACTION_EXCLUDE_KEYS",
        "FILTER_WORKFLOW_IDS",
        mode="before",
    )
    @classmethod
    def parse_comma_separated(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string into list of stripped strings.

        Supports both direct list input (from code/tests) and comma-separated
        string input (from environment variables). Empty strings result in
        empty list.

        Args:
            v: Input value (string or list)

        Returns:
            List of stripped strings with empty entries removed
        """
        if isinstance(v, list):
            return [s.strip() for s in v if s.strip()]
        if isinstance(v, str):
            if not v.strip():
                return []
            return [s.strip() for s in v.split(",") if s.strip()]
        return []

    @field_validator("ROOT_SPAN_INPUT_NODE", "ROOT_SPAN_OUTPUT_NODE", mode="before")
    @classmethod
    def normalize_root_span_nodes(cls, v: Any) -> Optional[str]:
        """Trim whitespace and normalize blank -> None for root span node names.

        Matching later occurs case-insensitively. We retain original casing for
        metadata transparency when a match occurs (source node name emitted).
        """
        if v is None:
            return None
        if isinstance(v, str):
            trimmed = v.strip()
            if not trimmed:
                return None
            return trimmed
        return None

    @model_validator(mode="after")
    def build_dsn_if_needed(self) -> "Settings":
        """Construct the PostgreSQL DSN from component parts if not provided.

        This validator runs after initial model creation. If `PG_DSN` is not set,
        it attempts to build it from individual `DB_POSTGRESDB_*` environment
        variables. This provides compatibility with n8n's typical environment
        setup.

        Returns:
            The validated `Settings` instance, with `PG_DSN` potentially populated.
        """
        if not self.PG_DSN and self.DB_POSTGRESDB_HOST and self.DB_POSTGRESDB_DATABASE:
            user = self.DB_POSTGRESDB_USER or "postgres"
            pwd = self.DB_POSTGRESDB_PASSWORD or ""
            auth = f"{user}:{pwd}@" if pwd else f"{user}@"
            port = self.DB_POSTGRESDB_PORT or 5432
            self.PG_DSN = f"postgresql://{auth}{self.DB_POSTGRESDB_HOST}:{port}/{self.DB_POSTGRESDB_DATABASE}"
        # DB_TABLE_PREFIX is required (pydantic enforces presence). Allow empty string.
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:  # pragma: no cover - trivial
    """Return a cached, singleton instance of the application settings.

    Provides a clearer error if mandatory env vars are missing.
    """
    try:
        # Pylance complains that required parameter "DB_TABLE_PREFIX" is missing
        # because this field is intentionally required (must be provided via env).
        # Pydantic BaseSettings injects it from the environment at runtime, so this
        # call is correct. We suppress the static type checker warning.
        return Settings()  # type: ignore[call-arg]
    except ValidationError as e:  # pragma: no cover - defensive
        # Provide a targeted helpful message for missing DB_TABLE_PREFIX
        missing_prefix = any(
            err.get("loc") in [("DB_TABLE_PREFIX",)] and err.get("type") == "missing"
            for err in e.errors()
        )
        if missing_prefix:
            raise RuntimeError(
                "DB_TABLE_PREFIX is required. Set DB_TABLE_PREFIX=n8n_ for standard installs "
                "or DB_TABLE_PREFIX= (blank) for no prefix."
            ) from e
        raise
