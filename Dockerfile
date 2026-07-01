FROM python:3.12-alpine AS base

# Build-time metadata labels (optional but helpful)
LABEL org.opencontainers.image.title="n8n-langfuse-shipper" \
      org.opencontainers.image.description="High-throughput shipper of n8n executions to Langfuse" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="0.5.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default flush every trace (can override at runtime)
    FLUSH_EVERY_N_TRACES=1 \
    # Make sure timezone aware behavior consistent
    TZ=UTC

# System deps kept minimal (psycopg[binary] ships wheels). Add libpq if needed later.
RUN apk add --no-cache ca-certificates curl

WORKDIR /app

# Copy project metadata and source. Source must exist for setuptools packages.find.
# (This slightly reduces layer caching efficiency vs copying only pyproject first,
# but avoids build failure when wheel metadata resolution expects ./src.)
COPY pyproject.toml README.md ./
COPY src ./src

# Install project (no dev extras) using pip's PEP 517 build. We install after copying
# source so that setuptools can locate packages under ./src without egg_base errors.
RUN pip install --upgrade pip \
    && pip install .

# Add Sectigo TLS intermediates to Python's certifi bundle (used by HTTPX/urllib3).
COPY sectigo_tls_intermediates.pem /tmp/sectigo_tls_intermediates.pem
RUN cat /tmp/sectigo_tls_intermediates.pem >> $(python -c "import certifi; print(certifi.where())") \
    && rm /tmp/sectigo_tls_intermediates.pem

# (Source already copied earlier for build correctness.)

# Create volume location for checkpoint (persists last processed execution id)
VOLUME ["/data"]
ENV CHECKPOINT_FILE=/data/.shipper_checkpoint

# Expose no ports (acts as batch job / worker). Healthcheck is simple mapper import.
HEALTHCHECK --interval=1m --timeout=10s --start-period=30s CMD python -c \
    "import importlib,sys; importlib.import_module('n8n_langfuse_shipper.mapper'); sys.exit(0)" || exit 1

# Default command runs shipper in dry-run mode (safe). Override with --no-dry-run in runtime args.
ENTRYPOINT ["n8n-shipper", "shipper"]
CMD ["--limit", "50", "--no-dry-run"]
