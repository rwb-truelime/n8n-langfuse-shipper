from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict as _SettingsConfigDict
from pydantic import Field, model_validator
from typing import Optional


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All fields are optional to allow flexible composition in different environments; validation
    for critical fields (like credentials) can happen at runtime when components requiring them
    are initialized.
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
    DB_TABLE_PREFIX: Optional[str] = None

    # Langfuse / OTLP
    LANGFUSE_HOST: str = Field(default="", description="Base URL for Langfuse host")
    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="Langfuse public key")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="Langfuse secret key")

    # Logging & runtime behavior
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    FETCH_BATCH_SIZE: int = Field(default=100, description="Batch size for fetching executions")
    # NOTE: Default truncation disabled per design choice: keep full textual JSON except binary/base64
    TRUNCATE_FIELD_LEN: int = Field(
        default=0,
        description="Max length for large text fields (input/output) before truncation (0 = disabled)",
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

    # Filtering: only process executions that have at least one metadata row whose key='executionId' and whose value equals the execution id.
    REQUIRE_EXECUTION_METADATA: bool = Field(
        default=False,
        description=(
            "If true, only executions having a row in <prefix>execution_metadata where key='executionId' and value matches the execution id are fetched."
        ),
    )

    @model_validator(mode="after")
    def build_dsn_if_needed(self):  # type: ignore[override]
        if not self.PG_DSN and self.DB_POSTGRESDB_HOST and self.DB_POSTGRESDB_DATABASE:
            user = self.DB_POSTGRESDB_USER or "postgres"
            pwd = self.DB_POSTGRESDB_PASSWORD or ""
            auth = f"{user}:{pwd}@" if pwd else f"{user}@"
            port = self.DB_POSTGRESDB_PORT or 5432
            self.PG_DSN = f"postgresql://{auth}{self.DB_POSTGRESDB_HOST}:{port}/{self.DB_POSTGRESDB_DATABASE}"
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:  # pragma: no cover - trivial
    return Settings()
