# n8n-langfuse-shipper
Python project to ship N8N execution data to Langfuse using OTEL (OpenTelemetry) API.

## CLI Usage

After installing dependencies (editable install recommended):

```bash
pip install -e .
```

Invoke the CLI (Iteration 1 placeholder):

```bash
python -m src --help
```

Run a backfill limiting number of executions (synthetic placeholder data in iteration 1):

```bash
python -m src backfill --limit 5
```

Specify a starting execution id:

```bash
python -m src backfill --start-after-id 1200 --limit 10
```

Environment variables (see `src/config.py`) can be exported before running, e.g.:

```bash
export PG_DSN="postgres://user:pass@localhost:5432/n8n"
export LANGFUSE_HOST="https://cloud.langfuse.com"
export LANGFUSE_PUBLIC_KEY="lf_pk_xxx"
export LANGFUSE_SECRET_KEY="lf_sk_xxx"
```

Then rerun the CLI.

> Note: Current iteration emits synthetic records only; real database streaming and OTLP export arrive in later iterations.

