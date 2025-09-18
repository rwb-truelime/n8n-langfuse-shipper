from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All fields are optional to allow flexible composition in different environments; validation
    for critical fields (like credentials) can happen at runtime when components requiring them
    are initialized.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Postgres
    PG_DSN: str = Field(
        default="", description="PostgreSQL DSN, e.g. postgres://user:pass@host:5432/dbname"
    )

    # Langfuse / OTLP
    LANGFUSE_HOST: str = Field(default="", description="Base URL for Langfuse host")
    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="Langfuse public key")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="Langfuse secret key")

    # Logging & runtime behavior
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    FETCH_BATCH_SIZE: int = Field(default=100, description="Batch size for fetching executions")
    TRUNCATE_FIELD_LEN: int = Field(
        default=4000,
        description="Max length for large text fields (input/output) before truncation",
    )

    # Future: additional tuning parameters
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(
        default=None, description="Override OTLP endpoint (otherwise derived from LANGFUSE_HOST)"
    )

    # Checkpointing
    CHECKPOINT_FILE: str = Field(
        default=".backfill_checkpoint",
        description="Path to file storing last processed execution id",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:  # pragma: no cover - trivial
    return Settings()
