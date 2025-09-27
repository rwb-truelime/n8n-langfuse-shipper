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
from typing import Optional

from pydantic import Field, ValidationError, model_validator
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

    # ---------------- Media / Binary Upload (Phase 1 Azure Only) -----------------
    # We intentionally mirror Langfuse's environment variable naming for Azure
    # Blob usage (even though they look S3-ish) to avoid user confusion. Langfuse
    # docs (Azure section) repurpose these variable names when
    # LANGFUSE_USE_AZURE_BLOB=true.
    ENABLE_MEDIA_UPLOAD: bool = Field(
        default=False,
        description=(
            "Feature flag: when true, collect binary assets and upload them to Azure Blob before "
            "export. When false, legacy unconditional redaction remains."
        ),
    )
    # Langfuse Azure toggle (mirrors docs). We still require user to set this
    # for clarity; both must be true to activate uploads.
    LANGFUSE_USE_AZURE_BLOB: bool = Field(
        default=False, description="Enable Azure Blob mode (Langfuse naming parity)"
    )
    # Container name (maps to Azure container). Langfuse docs reuse the S3 style
    # variable for Azure as container name.
    LANGFUSE_S3_EVENT_UPLOAD_BUCKET: Optional[str] = None
    # Azure storage account name (docs reuse ACCESS_KEY_ID var).
    LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: Optional[str] = None
    # Azure storage account key (docs reuse SECRET_ACCESS_KEY var).
    LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: Optional[str] = None
    # Optional endpoint override (Azurite/local). (Name consistent with docs.)
    LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: Optional[str] = None
    # Maximum size (bytes) of a single media object to upload. Large blobs are
    # left as redacted placeholders if exceeded.
    MEDIA_MAX_BYTES: int = Field(
        default=25_000_000,
        description="Maximum decoded binary size (bytes) allowed for upload (default 25MB)",
    )


    @model_validator(mode="after")
    def build_dsn_if_needed(self):  # type: ignore[override]
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
