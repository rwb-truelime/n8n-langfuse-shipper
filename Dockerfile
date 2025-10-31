FROM python:3.12-slim AS base

# Build-time metadata labels (optional but helpful)
LABEL org.opencontainers.image.title="n8n-langfuse-shipper" \
      org.opencontainers.image.description="High-throughput backfill of n8n executions to Langfuse" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default flush every trace (can override at runtime)
    FLUSH_EVERY_N_TRACES=1 \
    # Make sure timezone aware behavior consistent
    TZ=UTC

# System deps kept minimal (psycopg[binary] ships wheels). Add libpq if needed later.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency manifests first for better layer caching
COPY pyproject.toml README.md ./

# Install project (no dev extras) using pip's PEP 517 build
RUN pip install --upgrade pip \
    && pip install .

# Add SSL certificate to Python's certifi bundle (used by requests/urllib3)
COPY sertigo_intermediate.crt /tmp/sertigo_intermediate.crt
RUN cat /tmp/sertigo_intermediate.crt >> $(python -c "import certifi; print(certifi.where())") \
    && rm /tmp/sertigo_intermediate.crt

# Copy source code (after deps for caching efficiency)
COPY src ./src

# Create volume location for checkpoint (persists last processed execution id)
VOLUME ["/data"]
ENV CHECKPOINT_FILE=/data/.backfill_checkpoint

# Expose no ports (acts as batch job / worker). Healthcheck is simple mapper import.
HEALTHCHECK --interval=1m --timeout=10s --start-period=30s CMD python -c \
    "import importlib,sys; importlib.import_module('src.mapper'); sys.exit(0)" || exit 1

# Default command runs backfill in dry-run mode (safe). Override with --no-dry-run in runtime args.
ENTRYPOINT ["python", "-m", "src", "backfill"]
CMD ["--limit", "50", "--no-dry-run"]
