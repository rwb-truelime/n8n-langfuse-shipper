# n8n-langfuse-shipper

<!-- Badges -->
![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Status](https://img.shields.io/badge/status-Iteration%204-informational)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Type Checking](https://img.shields.io/badge/mypy-strict-blue)
![Lint](https://img.shields.io/badge/ruff-enabled-brightgreen)
<!-- (Optionally replace placeholders with real workflow badges once CI is added) -->

High‑throughput backfill service that reads historical n8n execution data from PostgreSQL and ships it to Langfuse via the OpenTelemetry (OTLP/HTTP) endpoint.

Current status: Iteration 4 (hierarchical Agent/Tool/LLM parenting + pointer‑compressed execution decoding + stable OTLP exporting).

---

## Features Implemented (Iteration 4)

- Pydantic v2 models for raw n8n execution JSON (`src/models/n8n.py`).
- Internal Langfuse models (`src/models/langfuse.py`).
- Observation type inference ported from JS mapper (`src/observation_mapper.py`).
- Deterministic trace & span IDs (trace id = raw `<executionId>` string, span IDs via UUIDv5). Trace name equals the workflow name (fallback: `execution`). Execution id exposed as span metadata key `n8n.execution.id` only (not duplicated in trace metadata). This shipper assumes exactly ONE n8n instance per Langfuse project—use separate Langfuse projects (keys) for additional instances to avoid execution id collisions.
- Hierarchical AI Agent/Tool/LanguageModel/Memory parenting using `workflowData.connections` `ai_tool`, `ai_languageModel`, `ai_memory` edge types (metadata: `n8n.agent.parent`, `n8n.agent.link_type`).
- Chronological span emission to guarantee agent span exists before children.
- Sequential + graph fallback parent inference (runtime `source.previousNodeRun` > last seen node span > static graph > root).
- Pointer‑compressed execution data decoding (list/pointer array format) seamlessly reconstructed into standard `runData` (`_decode_compact_pointer_execution`).
- Input propagation: child span input inferred from parent’s last output when `inputOverride` absent.
- Generation detection + token usage extraction (`tokenUsage` → `gen_ai.usage.*` and Langfuse generation listing).
- OTLP exporter with correct parent context handling (no orphan traces) and attribute mapping (`langfuse.observation.*`, `model`, `gen_ai.usage.*`).
- Real PostgreSQL streaming with batching, retry & schema/prefix awareness (`src/db.py`).
- CLI (`backfill`) with `--start-after-id`, `--limit`, `--dry-run`, plus deterministic resume via checkpoint.
- Auto-construction of `PG_DSN` from n8n style `.env` variables if not explicitly set.
- Truncation with per-span flags (`n8n.truncated.input` / `n8n.truncated.output`).
- Comprehensive mapper tests (determinism, hierarchy parenting, generation detection, truncation, graph fallback) & checkpoint tests.
- File-based checkpointing for resumability.
- Explicit schema/prefix override passed from settings to extractor (blank `DB_TABLE_PREFIX` respected, no silent fallback) plus startup diagnostic log of resolved tables.

---

## Architecture

```mermaid
graph TD
	PG[(PostgreSQL n8n DB)] -->|batches| EX[Extractor db.py]
	EX -->|rows| MAP[Mapper mapper.py]
	MAP --> SHIP[Shipper shipper.py]
	SHIP -->|OTLP spans| LF[Langfuse OTLP Endpoint]
	MAP --> OBS[Observation Mapper]
	MAP --> ID[(UUIDv5 Namespace)]
	MAP --> GEN[Generation Detection]
```

Key points:
- Streaming reader keeps memory footprint low (batch size controlled by `FETCH_BATCH_SIZE`).
- Pure transform layer builds internal Pydantic models before any OTLP emission (testable & dry-run friendly).
- Deterministic IDs ensure idempotent re-processing (safe restarts & checkpoint replay).
- Generation detection augments spans with GenAI semantic attributes so Langfuse auto-classifies them.

---

## Installation

Create / activate a Python 3.12 environment, then:

```bash
pip install -e .[dev]
```

Run tests (optional):

```bash
pytest -q
```

---

## Configuration Overview

The shipper is configured by environment variables (loaded via `pydantic-settings`) and optionally overridden by CLI flags. This section lists **all** supported knobs with defaults, effects, and override pathways.

### Precedence Rules
1. CLI flag (if provided)
2. Environment variable
3. Internal default

### Core Environment Variables

| Variable | Default | CLI Override | Description |
|----------|---------|--------------|-------------|
| `PG_DSN` | "" | (none) | Full Postgres DSN. If set, component DB vars are ignored for DSN construction. |
| `DB_POSTGRESDB_HOST` | - | (none) | DB host (used only if `PG_DSN` empty). |
| `DB_POSTGRESDB_PORT` | `5432` | (none) | DB port. |
| `DB_POSTGRESDB_DATABASE` | - | (none) | DB name. |
| `DB_POSTGRESDB_USER` | `postgres` | (none) | DB user (if DSN built). |
| `DB_POSTGRESDB_PASSWORD` | "" | (none) | DB password (optional). |
| `DB_POSTGRESDB_SCHEMA` | `public` | (none) | DB schema for tables. |
| `DB_TABLE_PREFIX` | `n8n_` (unset) / none (blank) | (none) | Table name prefix. Unset ⇒ `n8n_`; set to empty string ⇒ no prefix; any other explicit value used verbatim. |
| `FETCH_BATCH_SIZE` | `100` | (none) | Max executions fetched per DB batch. |
| `CHECKPOINT_FILE` | `.backfill_checkpoint` | `--checkpoint-file` | Path for last processed execution id. |
| `TRUNCATE_FIELD_LEN` | `0` | `--truncate-len` | Max chars for input/output before truncation. `0` ⇒ disabled (binary still stripped). |
| `REQUIRE_EXECUTION_METADATA` | `false` | `--require-execution-metadata / --no-require-execution-metadata` | Only include executions having at least one row in `<prefix>execution_metadata` with matching id. |
| `LOG_LEVEL` | `INFO` | (none) | Python logging level (`DEBUG`, `INFO`, etc.). |

### Langfuse / OTLP Export

| Variable | Default | CLI Override | Description |
|----------|---------|--------------|-------------|
| `LANGFUSE_HOST` | "" | (none) | Base Langfuse host; exporter appends `/api/public/otel/v1/traces` if needed. |
| `LANGFUSE_PUBLIC_KEY` | "" | (none) | Langfuse public key (Basic Auth). |
| `LANGFUSE_SECRET_KEY` | "" | (none) | Langfuse secret key (Basic Auth). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `None` | (none) | Full OTLP trace endpoint override (bypasses host derivation). |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | `30` | (none) | OTLP HTTP request timeout (seconds). |

### Reliability / Backpressure (Phase 1)

| Variable | Default | CLI Override | Description |
|----------|---------|--------------|-------------|
| `FLUSH_EVERY_N_TRACES` | `1` | (none) | Force `force_flush()` after every N traces (1 = after each trace). |
| `OTEL_MAX_QUEUE_SIZE` | `10000` | (none) | Span queue capacity for `BatchSpanProcessor`. Prevents unbounded memory. |
| `OTEL_MAX_EXPORT_BATCH_SIZE` | `512` | (none) | Max spans per OTLP export request. |
| `OTEL_SCHEDULED_DELAY_MILLIS` | `200` | (none) | Max delay before a batch is exported (ms). Lower = lower latency, higher overhead. |
| `EXPORT_QUEUE_SOFT_LIMIT` | `5000` | `--export-queue-soft-limit` | Approx backlog (created - last flushed) threshold triggering a small sleep. |
| `EXPORT_SLEEP_MS` | `75` | `--export-sleep-ms` | Sleep duration (ms) when backlog exceeds soft limit. |

### Derived / Internal Behavior
| Aspect | Determination |
|--------|--------------|
| DSN Construction | If `PG_DSN` empty and host+db present, DSN auto-built. |
| Prefix Logic | Unset prefix ⇒ `n8n_`; explicit empty string ⇒ no prefix. |
| Binary Redaction | Always on (independent of truncation). Large/base64-like payloads replaced with placeholders. |
| Root Span ID Strategy | Deterministic (UUIDv5 seed) with root span ended *after* children for reliability. |
| Flush Strategy | Forced every `FLUSH_EVERY_N_TRACES`. Backpressure sleep if backlog > soft limit. |

### Example Minimal Environment (fish shell)
Use either a single `PG_DSN` or the component variables:
```fish
set -x PG_DSN postgresql://n8n:n8n@localhost:5432/n8n
set -x LANGFUSE_HOST https://cloud.langfuse.com
set -x LANGFUSE_PUBLIC_KEY lf_pk_...
set -x LANGFUSE_SECRET_KEY lf_sk_...
```

Component style (auto-build DSN):
```fish
set -x DB_POSTGRESDB_HOST localhost
set -x DB_POSTGRESDB_DATABASE n8n
set -x DB_POSTGRESDB_USER n8n
set -x DB_POSTGRESDB_PASSWORD n8n
set -x LANGFUSE_HOST https://cloud.langfuse.com
set -x LANGFUSE_PUBLIC_KEY lf_pk_...
set -x LANGFUSE_SECRET_KEY lf_sk_...
```

### Override Reliability at Runtime (CLI)
If you want to temporarily throttle without changing env:
```fish
python -m src backfill --limit 500 --no-dry-run \
	--export-queue-soft-limit 2000 \
	--export-sleep-ms 120
```

### Truncation & Binary Clarifications
| Case | Behavior |
|------|----------|
| `TRUNCATE_FIELD_LEN=0` | No textual truncation. Binary/base64 still redacted. |
| `TRUNCATE_FIELD_LEN>0` | Input/output longer than limit truncated; metadata flags set. |
| Binary field (n8n `binary` object) | `data` replaced with placeholder & omitted length. |
| Long base64-looking string | Replaced with structured placeholder object. |

### Metadata Filtering Semantics
When `REQUIRE_EXECUTION_METADATA=true` (or CLI flag), the query adds an `EXISTS` subquery requiring at least one row in `<prefix>execution_metadata` where `executionId = execution.id`. This is not a key/value equality filter; presence alone suffices.

### Reliability Tuning Guidelines
- Increase `OTEL_MAX_QUEUE_SIZE` if you see frequent backpressure sleeps but have memory headroom.
- Increase `FLUSH_EVERY_N_TRACES` (e.g. 5 or 10) to reduce flush overhead on very large runs; retain periodic flush to avoid root-span loss on abrupt exit.
- Lower `OTEL_SCHEDULED_DELAY_MILLIS` for near real-time ingestion; raise it to batch more aggressively.
- Adjust `EXPORT_QUEUE_SOFT_LIMIT` and `EXPORT_SLEEP_MS` together; a larger limit with a slightly longer sleep can smooth throughput without overloading the exporter.

---

**Legacy Note:** Earlier iterations duplicated the execution id in trace metadata; now it appears only once as root span metadata key `n8n.execution.id`.

---

Example (fish shell):

```fish
set -x DB_POSTGRESDB_HOST localhost
set -x DB_POSTGRESDB_DATABASE n8n
set -x DB_POSTGRESDB_USER n8n
set -x DB_POSTGRESDB_PASSWORD n8n
set -x DB_POSTGRESDB_SCHEMA public
set -x DB_TABLE_PREFIX n8n_
set -x LANGFUSE_HOST https://cloud.langfuse.com
set -x LANGFUSE_PUBLIC_KEY lf_pk_... 
set -x LANGFUSE_SECRET_KEY lf_sk_...
```

If both component vars and `PG_DSN` are set, `PG_DSN` takes precedence.

Prefix semantics: If `DB_TABLE_PREFIX` is unset, it defaults to `n8n_`. If it is present but blank (`DB_TABLE_PREFIX=`), no prefix is applied (tables expected as `execution_entity`, `execution_data`).

---

## CLI Usage

Show help:

```bash
python -m src --help
```

Run a dry‑run backfill (fetch + map only, no network export):

```bash
python -m src backfill --limit 50 --dry-run
```

Export to Langfuse (remove `--dry-run`):

```bash
python -m src backfill --limit 50 --no-dry-run
```

Resume after a specific execution id:

```bash
python -m src backfill --start-after-id 12345 --limit 500 --dry-run
```

---

## Mapping Details

| n8n Concept | Langfuse Structure |
|-------------|-------------------|
| Execution row | One trace (root span represents whole execution; trace name = workflow name, fallback `execution`) |
| Node run | Child span (deterministic ID) |
| Agent/Tool/LLM/Memory relationship | Child span parented to Agent span via `ai_*` connection types |
| LLM / embedding node with token usage | Span + generation (usage + model attributes) |
| Node type/category | Observation type (`agent`, `tool`, `chain`, `retriever`, etc.) via mapper |
| Token usage (`tokenUsage`) | GenAI semantic attributes (`gen_ai.usage.*`) + Langfuse generation entry |

Parenting precedence order:
1. Agent hierarchy (if node has an `ai_tool` / `ai_languageModel` / `ai_memory` edge to an agent, parent = agent span).
2. Runtime sequential (`source.previousNodeRun` → specific run ID).
3. Runtime sequential (`source.previousNode` → last seen span for that node).
4. Static graph fallback (reverse edge inference) if runtime links are absent.
5. Root span fallback.

Inputs are inferred from the resolved parent’s last output when a node lacks `inputOverride` (captured as JSON and truncated if necessary).

Metadata: The trace carries `workflowId` and `status`. The execution id is exposed once via root span metadata key `n8n.execution.id` (not duplicated in trace metadata). Each span includes execution timing/status, hierarchy flags (`n8n.agent.*`), truncation flags, inferred parent markers, and previous node linkage.

### Execution Data Formats

The n8n `data` column can appear in two shapes:
- Standard object with `executionData.resultData.runData`.
- Pointer‑compressed top‑level JSON array: entries reference earlier indices (saves space). The shipper reconstructs this via `_decode_compact_pointer_execution` into canonical `runData` transparently.

### AI Agent Hierarchy Mapping

Agent nodes (e.g. `HAL9000`) become parents of LLM, Tool, and Memory nodes connected by `ai_languageModel`, `ai_tool`, or `ai_memory` edges. This produces a nested trace tree that mirrors n8n’s LangChain-style clusters. Child spans carry:
- `n8n.agent.parent` = agent node name
- `n8n.agent.link_type` = one of `ai_languageModel`, `ai_tool`, `ai_memory`

This prevents fragmentation into multiple traces and yields a faithful hierarchical representation in Langfuse.

Each node span now includes metadata:
- `n8n.node.run_index`, `n8n.node.execution_time_ms`, `n8n.node.execution_status`
- Truncation flags: `n8n.truncated.input`, `n8n.truncated.output` when applicable
- Previous linkage: `n8n.node.previous_node`, `n8n.node.previous_node_run` when present

---

## Examples

### 1. Quick Dry‑Run (no network)
Fetch first 25 executions, map to traces/spans, just log summary:

```fish
python -m src backfill --limit 25 --dry-run
```

### 2. Real Export
Assuming you have set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (and optionally `LANGFUSE_HOST`):

```fish
python -m src backfill --limit 100 --no-dry-run
```

### 3. Resume After a Known Execution ID

```fish
python -m src backfill --start-after-id 420000 --limit 500 --dry-run
```

### 4. Narrow Window (combine with future filters)
Planned future flags may allow filtering by status or date; for now you can externally pipe IDs.

### 5. Programmatic Use (Embedding as a Library)
If you want to call the mapper directly (e.g. inside another orchestration script):

```python
from src.mapper import map_execution_to_langfuse
from src.shipper import LangfuseOTLPShipper

# record = <load one execution row & JSON just like db.stream yields>
trace = map_execution_to_langfuse(record)
shipper = LangfuseOTLPShipper(dry_run=True)  # set False to export
shipper.export_trace(trace)
```

### 6. Environment Convenience (fish)
Persist frequently used variables in a local script:

```fish
function env.n8n-lf
	set -x DB_POSTGRESDB_HOST localhost
	set -x DB_POSTGRESDB_DATABASE n8n
	set -x DB_POSTGRESDB_USER n8n
	set -x DB_POSTGRESDB_PASSWORD n8n
	set -x LANGFUSE_HOST https://cloud.langfuse.com
	set -x LANGFUSE_PUBLIC_KEY lf_pk_xxx
	set -x LANGFUSE_SECRET_KEY lf_sk_xxx
end

env.n8n-lf; python -m src backfill --limit 50 --dry-run
```

### 7. Troubleshooting Tips
- If you see zero rows: verify `DB_TABLE_PREFIX` and `DB_POSTGRESDB_SCHEMA` match your n8n deployment.
- Large payload warnings: reduce `FETCH_BATCH_SIZE` or lower `TRUNCATE_FIELD_LEN`. Set `--truncate-len 0` (or env `TRUNCATE_FIELD_LEN=0`) to disable truncation (binary data still omitted with placeholder).
- Need verbose output: set `LOG_LEVEL=DEBUG` before running the CLI.
- Prefix mismatch / missing tables: On startup you should see a line like:
	`DB init: schema=public prefix='' entity_table=execution_entity data_table=execution_data (explicit_prefix=True)`
	If it still shows `prefix='n8n_'` when you expected none, ensure `.env` has `DB_TABLE_PREFIX=` (not commented) and is loaded. If your DB actually uses prefixed tables, either unset the variable or set `DB_TABLE_PREFIX=n8n_`.
	An `UndefinedTable` or error mentioning `n8n_execution_entity` means the prefix doesn't match your actual table names.

---

## Development Scripts (fish)

Install dev deps:

```fish
pip install -e .[dev]
```

Run lint (ruff) & type check (mypy):

```fish
ruff check .
mypy src
```

Run focused test:

```fish
pytest tests/test_db_stream.py::test_stream_reads_rows_without_modification -q
```

### Pre-commit Hooks

Install git hooks (includes NOTICE guard):

```fish
pre-commit install
```

The custom hook `ensure-notice-present` blocks commits if the `NOTICE` header is changed or removed. Update `scripts/check_notice.py` intentionally if the copyright year rolls over.

---

## Security & Safety

- Read-only queries: only `SELECT` statements used; no mutations or DDL.
- Deterministic IDs allow idempotent re-runs without duplicating spans.
- Truncation prevents oversized OTLP payloads.

---

## Binary / Large Payload Handling

Binary or very large base64 payloads are removed pre-export to avoid excessive OTLP span sizes while retaining structural context.

Detection heuristics:
- Standard n8n `binary` node structure (`binary` -> item -> `{ data: <b64>, mimeType: ... }`).
- Any long (>200 chars) base64-looking string (regex match) or strings starting with common base64 magic like `/9j/` (JPEG).

Replacement:
- In `binary` objects: `data` value replaced with `binary omitted` plus `_omitted_len` metadata.
- Standalone strings: replaced with `{ "_binary": true, "note": "binary omitted", "_omitted_len": <len> }`.

This occurs even when truncation is disabled (`TRUNCATE_FIELD_LEN=0`). Future environment knobs (planned): `BINARY_PLACEHOLDER`, `BINARY_MIN_LEN`.

## Roadmap (Next Iterations)

1. Media upload workflow (store omitted binaries via Langfuse media API + token substitution).
2. Error retries / resilient OTLP + media upload with dead-letter logging.
3. Performance tuning (parallel export, async batching, memory caps for large runs).
4. Additional filtering flags (status, time window, workflow id inclusion/exclusion).
5. Masking / PII scrubbing and configurable redaction rules.
6. Extended observation classification & multimodal span enrichment.
7. Optional tagging of agent root spans (`n8n.agent.root=true`) and richer lineage metadata.

---

## Contributing

1. Fork & branch.
2. Add/adjust tests.
3. Run formatting/lint: `ruff check --fix .`.
4. Ensure tests pass.
5. Open PR with context (data volume considerations welcome).

---

## Attribution

This project requests (but does not legally require beyond the Apache 2.0 NOTICE preservation) a visible credit in any UI, documentation, or public site where its functionality is exposed. A suggested phrase:

> Powered by n8n-langfuse-shipper (Apache 2.0)

If space allows, please link to the repository:
`https://github.com/rwb-truelime/n8n-langfuse-shipper`

If you redistribute source or binaries, you **must** retain the `LICENSE` file and the `NOTICE` file per the Apache License 2.0. You may append your own notices to `NOTICE` when distributing a derivative.

For commercial or closed-source use, no special permission is required beyond compliance with the Apache 2.0 terms.

If you have an alternative attribution format (e.g. consolidated vendor credits page), that is fine—keep the contents of `NOTICE` intact.

### NOTICE Guard Configuration
The pre-commit hook reads `notice_check.toml` for required substrings and header validation. Adjust that file (not the script) when:
- Updating the copyright year
- Adding additional mandatory attribution lines

Example (`notice_check.toml`):
```
[notice]
required_substrings = ["n8n-langfuse-shipper", "Apache License, Version 2.0", "Copyright 2025 Rodger Blom"]
header_must_contain = "n8n-langfuse-shipper"
header_search_chars = 250
```

## License

Apache License 2.0. See `LICENSE` for the full text and `NOTICE` for attribution.


