# n8n-langfuse-shipper

<!-- Badges -->
![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Status](https://img.shields.io/badge/status-active--development-informational)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Type Checking](https://img.shields.io/badge/mypy-strict-blue)
![Lint](https://img.shields.io/badge/ruff-enabled-brightgreen)
<!-- (Optionally replace placeholders with real workflow badges once CI is added) -->

High‑throughput backfill service that reads historical n8n execution data from PostgreSQL and ships it to Langfuse via the OpenTelemetry (OTLP/HTTP) endpoint.

Current focus: optional Langfuse media token upload (feature‑flag gated) alongside hierarchical Agent/Tool/LLM parenting, pointer‑compressed execution decoding, and stable OTLP exporting.
New capability: AI-only span filtering (`--filter-ai-only` flag or `FILTER_AI_ONLY=true` env) to export
only AI-related node spans (plus ancestor context) for focused GenAI analytics.

Note (Removal): The legacy Azure‑Blob specific module `media_uploader.py` has been fully removed.
Media handling now exclusively uses the Langfuse Media API token flow implemented in `src/media_api.py`.
If you previously depended on the Azure container naming scheme emitted by the old uploader, migrate
by relying on the stable token string format documented in the Media Upload section. No direct
blob storage interaction is performed by this service anymore.

---

## Quick Start

Get running in under a minute.

1. Install (Python 3.12+):
	```bash
	pip install -e .[dev]
	```
2. Set the essential environment (fish shell shown – replace values):
	```fish
	set -x PG_DSN postgresql://n8n:n8n@localhost:5432/n8n  # or use component vars
	set -x LANGFUSE_HOST https://cloud.langfuse.com        # optional if using default cloud host
	set -x LANGFUSE_PUBLIC_KEY lf_pk_xxx
	set -x LANGFUSE_SECRET_KEY lf_sk_xxx
	```
	(Alternatively set `DB_POSTGRESDB_HOST`, `DB_POSTGRESDB_DATABASE`, `DB_POSTGRESDB_USER`, `DB_POSTGRESDB_PASSWORD` instead of `PG_DSN`.)
3. Dry run (map only, nothing sent):
	```bash
	python -m src backfill --limit 25 --dry-run
	```
4. Real export (removes `--dry-run`):
	```bash
	python -m src backfill --limit 100 --no-dry-run
	```
5. Resume later (automatic checkpoint file `.backfill_checkpoint`): just re-run the command; or start after a known id:
	```bash
	python -m src backfill --start-after-id 12345 --limit 200 --no-dry-run
	```

Need more? Expand the detailed sections below.

---
<details>
<summary><strong>Key Features</strong></summary>

## Key Features

- Pydantic v2 models for raw n8n execution JSON (`src/models/n8n.py`).
- Internal Langfuse models (`src/models/langfuse.py`).
- Observation type inference ported from JS mapper (`src/observation_mapper.py`).
- Deterministic trace & span IDs (trace id = raw `<executionId>` string, span IDs via UUIDv5). Trace name equals the workflow name (fallback: `execution`). Execution id exposed as span metadata key `n8n.execution.id` only (not duplicated in trace metadata). This shipper assumes exactly ONE n8n instance per Langfuse project—use separate Langfuse projects (keys) for additional instances to avoid execution id collisions.
- Hierarchical AI Agent parenting using any `ai_*` connection type in `workflowData.connections` (metadata: `n8n.agent.parent`, `n8n.agent.link_type`).
- Optional AI-only filtering retaining only AI node spans (nodes from
	`@n8n/n8n-nodes-langchain` or category `AI/LangChain Nodes`) plus non-AI ancestors; root span
	always kept. Metadata: `n8n.filter.ai_only=true`, `n8n.filter.excluded_node_count`, and
	`n8n.filter.no_ai_spans=true` when no AI spans exist.
- Chronological span emission to guarantee agent span exists before children.
- Sequential + graph fallback parent inference (runtime `source.previousNodeRun` > last seen node span > static graph > root).
- Pointer‑compressed execution data decoding (list/pointer array format) seamlessly reconstructed into standard `runData` (`_decode_compact_pointer_execution`).
- Input propagation: child span input inferred from parent’s last output when `inputOverride` absent.
- Generation detection + token usage extraction (`tokenUsage` variants normalized to input/output/total → `gen_ai.usage.*`).
			- Gemini empty-output anomaly detection: flags spans where `text==""`, `completionTokens==0`, `promptTokens>0`, `totalTokens>=promptTokens` (and optional empty `generationInfo {}`) with metadata (`n8n.gen.empty_output_bug`, `n8n.gen.empty_generation_info`, token counters) and forces status=error.
			- Provider markers: `openai`, `anthropic`, `gemini`, `mistral`, `groq`, `lmchat`, `lmopenai`, `cohere`, `deepseek`, `ollama`, `openrouter`, `bedrock`, `vertex`, `huggingface`, `xai`, `limescape` (excludes embeddings/rerankers unless `tokenUsage` present; nested `tokenUsage` & `model` discovered via depth-limited search inside node output channel wrappers like `ai_languageModel`).
            - Concise generation span output extraction: Gemini/Vertex responses extract first non-empty `generations[0][0].text`; Limescape Docs node extracts the `markdown` field (when present) instead of full JSON wrapper.
            - LangChain LMChat System Prompt Stripping: Automatically strips System prompts from `@n8n/n8n-nodes-langchain.lmChat*` generation inputs. Searches recursively (up to 25 levels deep) for `messages` arrays containing the split marker `\n\n## START PROCESSING\n\nHuman: ##`, keeping only the User prompt portion. Handles both string arrays and dict-with-content formats. Applied before normalization to preserve structure.
- OTLP exporter with correct parent context handling (no orphan traces) and attribute mapping (`langfuse.observation.*`, `model`, `gen_ai.usage.*`, consolidated `langfuse.observation.usage_details`, root marker `langfuse.internal.as_root`, optional trace identity fields `user.id|session.id|langfuse.trace.tags|langfuse.trace.input|langfuse.trace.output`).
- Real PostgreSQL streaming with batching, retry & schema/prefix awareness (`src/db.py`).
- CLI (`backfill`) with `--start-after-id`, `--limit`, `--dry-run`, plus deterministic resume via checkpoint.
- Auto-construction of `PG_DSN` from n8n style `.env` variables if not explicitly set.
- Truncation with per-span flags (`n8n.truncated.input` / `n8n.truncated.output`).
- Comprehensive mapper tests (determinism, hierarchy parenting, generation detection, truncation, graph fallback) & checkpoint tests.
- File-based checkpointing for resumability.
- Explicit schema/prefix override passed from settings to extractor (blank `DB_TABLE_PREFIX` respected, no silent fallback) plus startup diagnostic log of resolved tables.

Additional documentation-aligned capabilities:
- Timezone normalization: all timestamps coerced to timezone-aware UTC early; naive datetimes are forbidden (see Timezone Policy section + tests).
- Human-readable OTLP trace id embedding: exporter derives a 32-hex trace id embedding execution id digits (see `test_trace_id_embedding.py`).
- Parent precedence table (agent hierarchy → runtime exact run → runtime last run → static reverse graph → root) formally documented; changes require test + instructions update.
- Backpressure & flush strategy: forced `force_flush()` after `FLUSH_EVERY_N_TRACES` and soft backlog sleep when `EXPORT_QUEUE_SOFT_LIMIT` exceeded.
- Input propagation semantics: child input inferred from resolved parent output unless `inputOverride` present (documented + tested).
- Pointer-compressed (`list` + index pointer) execution decoding detailed and resilient; failure falls back gracefully to path probing.
- Regression checklist enumerated in `.github/copilot-instructions.md` to prevent silent drift (IDs, execution id placement, timezone, binary stripping, generation heuristics, pointer decoding, env vars, purity).
- Binary handling invariants: unconditional stripping independent of truncation (clarified in docs & tests).
- Optional media token upload (feature-flag gated) replacing redaction placeholders with Langfuse media token strings (see Media Upload section). Disabled ⇒ legacy redaction.

Guardrail: Any new environment variable (e.g. future media upload flags) MUST be added to this Feature list, the Configuration Overview tables, `.github/copilot-instructions.md`, and have corresponding tests in the same PR.

---

</details>

## Architecture

```mermaid
graph TD
	PG[(PostgreSQL n8n DB)] -->|batches| EX[Extractor db.py]
	EX -->|rows| MAP[Mapper mapper.py]
	MAP --> MU[Media Upload]
	MU --> SHIP[Shipper shipper.py]
	SHIP -->|OTLP spans| LF[Langfuse OTLP Endpoint]
	MAP --> OBS[Observation Mapper]
	MAP --> ID[(UUIDv5 Namespace)]
	MAP --> GEN[Generation Detection]

	%% Media Upload optional (enabled only when ENABLE_MEDIA_UPLOAD true)
```

Key points:
- Streaming reader keeps memory footprint low (batch size controlled by `FETCH_BATCH_SIZE`).
- Pure transform layer builds internal Pydantic models before any OTLP emission (testable & dry-run friendly).
- Deterministic IDs ensure idempotent re-processing (safe restarts & checkpoint replay).
- Generation detection augments spans with GenAI semantic attributes so Langfuse auto-classifies them.
- All timestamps are normalized to timezone-aware UTC early (naive inputs get `tzinfo=UTC`) ensuring consistent ordering & avoiding deprecated naive datetime usage.
- Media upload (Langfuse Media API phase) occurs after mapping and before exporting; when disabled the pipeline links Mapper directly to Shipper.

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

## Containerization & Deployment (Docker / Azure Container Apps)

You can run the shipper fully containerized. A production‑lean Dockerfile and a
`docker-compose.yml` are included. The container defaults to a safe dry‑run
(`--limit 50 --dry-run`). Override the command for real exporting.

### Build Image Locally
```fish
docker build -t n8n-langfuse-shipper:local .
```

### Dry Run (no export)
```fish
docker run --rm \
	-e PG_DSN=$PG_DSN \
	-e LANGFUSE_PUBLIC_KEY=$LANGFUSE_PUBLIC_KEY \
	-e LANGFUSE_SECRET_KEY=$LANGFUSE_SECRET_KEY \
	-e DB_TABLE_PREFIX=n8n_ \
	-v shipper_checkpoint:/data \
	n8n-langfuse-shipper:local --limit 25 --dry-run
```

### Real Export
```fish
docker run --rm \
	-e PG_DSN=$PG_DSN \
	-e LANGFUSE_PUBLIC_KEY=$LANGFUSE_PUBLIC_KEY \
	-e LANGFUSE_SECRET_KEY=$LANGFUSE_SECRET_KEY \
	-e DB_TABLE_PREFIX=n8n_ \
	-v shipper_checkpoint:/data \
	n8n-langfuse-shipper:local --limit 500 --no-dry-run
```

The named volume `shipper_checkpoint` persists the checkpoint file so restarts
resume where they left off.

### Using docker-compose
Edit environment values (or export them in your shell) then:
```fish
docker compose up --build shipper
```
Compose mounts a persistent volume for `/data/.backfill_checkpoint`. Adjust the
`command` field to change limits, enable dry‑run, etc.

### Minimal Required Environment Variables
Provide either `PG_DSN` OR the component variables (`DB_POSTGRESDB_HOST`, etc.),
plus Langfuse credentials and the table prefix:
```fish
set -x PG_DSN postgresql://n8n:n8n@postgres:5432/n8n
set -x LANGFUSE_PUBLIC_KEY lf_pk_...
set -x LANGFUSE_SECRET_KEY lf_sk_...
set -x DB_TABLE_PREFIX n8n_
```

Optional: `LANGFUSE_HOST` if self‑hosted; `ENABLE_MEDIA_UPLOAD=true` to enable
media token flow.

### Azure Container Registry (ACR) Build & Push
Assuming you already created an Azure Container Registry and are logged in with
`az` CLI:
```fish
set -x ACR_NAME myregistry   # your ACR name (without domain)
set -x IMAGE_TAG (git rev-parse --short HEAD)
docker build -t $ACR_NAME.azurecr.io/n8n-langfuse-shipper:$IMAGE_TAG .
az acr login --name $ACR_NAME
docker push $ACR_NAME.azurecr.io/n8n-langfuse-shipper:$IMAGE_TAG
```

Optionally add `:latest` tag as well:
```fish
docker tag $ACR_NAME.azurecr.io/n8n-langfuse-shipper:$IMAGE_TAG \
	$ACR_NAME.azurecr.io/n8n-langfuse-shipper:latest
docker push $ACR_NAME.azurecr.io/n8n-langfuse-shipper:latest
```

### Deploy to Azure Container Apps
You already have a Container Apps Environment; supply its name and resource
group. Example (single revision performing continuous backfill):
```fish
set -x ACR_NAME myregistry
set -x RG_NAME my-resource-group
set -x ENV_NAME my-container-app-env
set -x APP_NAME n8n-langfuse-shipper
set -x IMAGE_TAG latest  # or specific commit tag

az containerapp create \
	--name $APP_NAME \
	--resource-group $RG_NAME \
	--environment $ENV_NAME \
	--image $ACR_NAME.azurecr.io/n8n-langfuse-shipper:$IMAGE_TAG \
	--registry-server $ACR_NAME.azurecr.io \
	--ingress disabled \
	--cpu 0.5 --memory 1Gi \
	--args "--limit" "500" "--no-dry-run" \
	--env-vars \
		PG_DSN=$PG_DSN \
		LANGFUSE_PUBLIC_KEY=$LANGFUSE_PUBLIC_KEY \
		LANGFUSE_SECRET_KEY=$LANGFUSE_SECRET_KEY \
		LANGFUSE_HOST=$LANGFUSE_HOST \
		DB_TABLE_PREFIX=n8n_ \
		LOG_LEVEL=INFO
```

Add media upload (optional):
```fish
az containerapp update \
	--name $APP_NAME --resource-group $RG_NAME \
	--set-env-vars ENABLE_MEDIA_UPLOAD=true MEDIA_MAX_BYTES=25000000
```

### Updating Image in Azure
Push a new image (new tag) then:
```fish
az containerapp update \
	--name $APP_NAME --resource-group $RG_NAME \
	--image $ACR_NAME.azurecr.io/n8n-langfuse-shipper:$IMAGE_TAG
```

### Operational Notes
| Topic | Guidance |
|-------|----------|
| Scaling | Start single replica; increase only if DB & Langfuse throughput allow. |
| Checkpoint | Persisted at `/data/.backfill_checkpoint`; mount Azure File Share for durability. |
| Dry Runs | Always test config with `--dry-run` before a large migration. |
| Logs | Set `LOG_LEVEL=DEBUG` for detailed mapping & media debug info. |
| Safe Re-run | Deterministic IDs prevent duplication; re-processing yields identical spans. |
| Media Upload | Ensure network egress to Langfuse host and object storage endpoints. |

If you need a one-shot batch run, deploy with an argument like `--limit 5000`.
The container will exit when finished (Container Apps will mark it as stopped).
For continuous ingestion, omit `--limit` so it processes batches indefinitely.

---

---

<details>
<summary><strong>Configuration Overview</strong></summary>

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
| `DB_TABLE_PREFIX` | (required) | (none) | Mandatory table name prefix. Set `DB_TABLE_PREFIX=n8n_` explicitly or blank (`DB_TABLE_PREFIX=`) for none. |
| `FETCH_BATCH_SIZE` | `100` | (none) | Max executions fetched per DB batch. |
| `CHECKPOINT_FILE` | `.backfill_checkpoint` | `--checkpoint-file` | Path for last processed execution id. |
| `TRUNCATE_FIELD_LEN` | `0` | `--truncate-len` | Max chars for input/output before truncation. `0` ⇒ disabled (binary still stripped). |
| `ENABLE_MEDIA_UPLOAD` | `false` | (none) | Master media feature flag. When false only binary redaction; no token API calls. |
| `MEDIA_MAX_BYTES` | `25000000` | (none) | Max decoded size per asset; larger assets remain redaction placeholders. |
| `EXTENDED_MEDIA_SCAN_MAX_ASSETS` | `250` | (none) | Cap on number of non-canonical discovered binary assets per node run (data URL / file-like dict / contextual base64). <=0 disables. |
| `REQUIRE_EXECUTION_METADATA` | `false` | `--require-execution-metadata / --no-require-execution-metadata` | Only include executions having at least one row in `<prefix>execution_metadata` with matching id. |
| `FILTER_AI_ONLY` | `false` | `--filter-ai-only / --no-filter-ai-only` | Export only AI-related spans (plus ancestor chain); root span always retained. |
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
| AI-only Filtering | When enabled: keep root, AI spans, ancestor chain; attach filtering metadata keys. |
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
	--filter-ai-only
```

### Truncation, Binary & Media Clarifications
| Case | Behavior |
|------|----------|
| `TRUNCATE_FIELD_LEN=0` | No textual truncation. Binary/base64 still redacted. |
| `TRUNCATE_FIELD_LEN>0` | Input/output longer than limit truncated; metadata flags set. |
| Binary field (n8n `binary` object) | `data` replaced with placeholder & omitted length. |
| Long base64-looking string | Replaced with structured placeholder object. |
| Media upload enabled + eligible asset | Placeholder with `_media_pending=true` replaced by Langfuse media token string `@@@langfuseMedia:type=<mime>|id=<mediaId>|source=base64_data_uri@@@`. |
| Media upload failure / size over cap | Placeholder stays (legacy redaction), `n8n.media.upload_failed=true` metadata added. |
| AI-only filtering enabled | Non-AI node spans excluded unless ancestor of an AI span; root span kept. |
| AI-only filtering enabled, no AI spans | Only root span exported; root has `n8n.filter.no_ai_spans=true`. |
| Media create deduplicated (no uploadUrl) | Token substituted (counts toward asset_count); no failure flags or error codes. |
| Extended discovery asset (data URL / file-like / contextual base64) | Treated like canonical asset: placeholder then token (or redaction if upload disabled). Capped by EXTENDED_MEDIA_SCAN_MAX_ASSETS. |

### Media Troubleshooting

| Symptom | Likely Cause | Expected Behavior / Action |
|---------|--------------|----------------------------|
| Placeholder dict with `_media_pending` remains instead of token | API create/upload failure or feature disabled | Span metadata `n8n.media.upload_failed=true` when failure; verify host/keys & logs. |
| Missing expected non-AI spans | AI-only filtering enabled | Disable `FILTER_AI_ONLY` or pass `--no-filter-ai-only` to restore full span set. |
| Placeholder remains for large file | Asset exceeded `MEDIA_MAX_BYTES` | Flag `n8n.media.upload_failed=true`; increase limit if acceptable. |
| `n8n.media.asset_count` missing | No successful token substitutions | Normal when all assets failed or feature disabled. |
| Token present but media not visible in UI | (Rare) UI lag or later deletion | Token is opaque; confirm in Langfuse project media list. |
| High memory usage | Extremely large base64 inputs in many spans | Lower `FETCH_BATCH_SIZE`, enable truncation for large text wrappers; binary already stripped early. |
| `n8n.media.error_codes` present | One or more media processing failures | Inspect list values (see Media Error Codes) to triage root cause. |
| Missing expected data URL placeholder | Data URL below 64 chars or scan cap reached | Increase payload size or raise EXTENDED_MEDIA_SCAN_MAX_ASSETS. |

Media Error Codes (each appended once per span per affected asset type):
* `decode_or_oversize` – Base64 decode failed or asset exceeded `MEDIA_MAX_BYTES` threshold.
* `create_api_error` – Non-2xx status from Langfuse media create request.
* `missing_id` – Successful create call but response lacked media id.
* `upload_put_error` – Presigned binary upload (PUT) returned non-2xx.
* `serialization_error` – Failed to serialize span output after token insertion.
* `scan_asset_limit` – Extended discovery exceeded EXTENDED_MEDIA_SCAN_MAX_ASSETS cap (remaining assets ignored).
* `status_patch_error` – Upload succeeded but final status PATCH request failed.

Deduplicated Media (No `uploadUrl`): The create endpoint may return an id without an upload URL when the
exact asset (by content hash) already exists. This is treated as success: a token is emitted, it increments
`n8n.media.asset_count`, and neither `n8n.media.upload_failed` nor `n8n.media.error_codes` is set.

Token Format: `@@@langfuseMedia:type=<mime>|id=<mediaId>|source=base64_data_uri@@@` (type omitted if unknown).

Preview Surfacing & Status Patch:

The shipper performs **in-place media surfacing**: each media token replaces the original base64 payload
directly at its JSON path. Canonical `binary.<slot>` tokens are additionally promoted to a shallow `<slot>`
key when absent, enabling Langfuse UI preview heuristics without extra wrapper keys. Legacy surfacing modes
(`legacy`, `mirror`) and the `_media_preview` hoist key were removed to simplify behavior and guarantee a
single deterministic structure.

Current metadata additions (per span):
* `n8n.media.surface_mode` – constant string `inplace` (observability only).
* `n8n.media.preview_surface` – at least one token replaced in-place or promoted for preview.
* `n8n.media.promoted_from_binary` – a canonical `binary.<slot>` was surfaced also as shallow `<slot>`.

Status patch behavior: after upload (or dedupe) a PATCH sets `uploadedAt`, `uploadHttpStatus`, and optional
`uploadHttpError`; failures append `status_patch_error` to `n8n.media.error_codes` but do not abort export.

### Metadata Filtering Semantics
When `REQUIRE_EXECUTION_METADATA=true` (or CLI flag), the query adds an `EXISTS` subquery requiring at least one row in `<prefix>execution_metadata` where `executionId = execution.id`. This is not a key/value equality filter; presence alone suffices.

### Reliability Tuning Guidelines
- Increase `OTEL_MAX_QUEUE_SIZE` if you see frequent backpressure sleeps but have memory headroom.
- Increase `FLUSH_EVERY_N_TRACES` (e.g. 5 or 10) to reduce flush overhead on very large runs; retain periodic flush to avoid root-span loss on abrupt exit.
- Lower `OTEL_SCHEDULED_DELAY_MILLIS` for near real-time ingestion; raise it to batch more aggressively.
- Adjust `EXPORT_QUEUE_SOFT_LIMIT` and `EXPORT_SLEEP_MS` together; a larger limit with a slightly longer sleep can smooth throughput without overloading the exporter.

---

**Legacy Note:** Early versions duplicated the execution id in trace metadata; now it appears only once as root span metadata key `n8n.execution.id`.

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
set -x FILTER_AI_ONLY true  # optional: export only AI spans
```

If both component vars and `PG_DSN` are set, `PG_DSN` takes precedence.

Prefix semantics: `DB_TABLE_PREFIX` is required. Provide `DB_TABLE_PREFIX=n8n_` for standard prefixed tables or `DB_TABLE_PREFIX=` (blank) for unprefixed tables. No implicit default.

---

</details>

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
| LLM / embedding node with token usage | Single span classified as generation (usage + model attributes) |
| Node type/category | Observation type (`agent`, `tool`, `chain`, `retriever`, etc.) via mapper |
| Token usage (`tokenUsage` legacy promptTokens/completionTokens/totalTokens OR input/output/total; plus flattened `totalInputTokens` / `totalOutputTokens` / `totalTokens`) | Normalized to input/output/total then emitted as `gen_ai.usage.input_tokens|output_tokens|total_tokens` (legacy prompt/completion names removed) |
| Gemini empty output anomaly (Gemini / Vertex chat bug) | Detection rule: `generations[0][0].text==""` AND `promptTokens>0` AND `totalTokens>=promptTokens` AND `completionTokens==0 or missing` (optional `generationInfo=={}`); sets span status=error, metadata keys: `n8n.gen.empty_output_bug=true`, `n8n.gen.empty_generation_info=true` (if applicable), plus token counts. |

Parenting precedence order:
1. Agent hierarchy (if node has an `ai_*` edge to an agent, parent = agent span).
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

Agent nodes (e.g. `HAL9000`) become parents of downstream AI component nodes connected by any `ai_*` edge (e.g. `ai_languageModel`, `ai_tool`, `ai_memory`, `ai_outputParser`, `ai_retriever`). This produces a nested trace tree that mirrors n8n’s LangChain-style clusters. Child spans carry:
- `n8n.agent.parent` = agent node name
- `n8n.agent.link_type` = the concrete `ai_*` connection type

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

---

## Testing

This project treats the mapping & export invariants as a contract. Each invariant has one or more tests. When you change mapper / shipper behavior, update or add tests in the same PR (and update `.github/copilot-instructions.md`).

### Test Suites Overview

| File | Focus | Key Invariants Covered |
|------|-------|------------------------|
| `tests/test_mapper.py` | Core mapping (determinism, truncation flag, runtime & graph parenting, agent hierarchy link types, generation w/ tokenUsage) | Deterministic span IDs; parent precedence; generation detection; truncation flags; agent `n8n.agent.link_type` |
| `tests/test_binary_and_truncation.py` | Binary/base64 stripping + truncation interplay | Binary stripping always on (even when truncation disabled); truncation flags coexist with placeholders |
| `tests/test_input_propagation.py` | Inferred input propagation & size guard | Parent output inference; size guard blocking when truncation active |
| `tests/test_pointer_decoding.py` | Pointer‑compressed array decoding | Resilient decoding of compact execution format |
| `tests/test_trace_and_metadata.py` | Root metadata + deterministic IDs across truncation settings | Root-only `n8n.execution.id`; span ID stability irrespective of truncation |
| `tests/test_generation_heuristic.py` | Provider substring heuristic | Generation detection across provider matrix & exclusion for embeddings |
| `tests/test_lmchat_system_prompt_strip.py` | LangChain LMChat system prompt stripping | Strips System prompts from lmChat generation inputs; handles dict/string messages, deep nesting, graceful fallback |
| `tests/test_error_status_and_observation.py` | Error normalization | Span status becomes `error` when `NodeRun.error` set |
| `tests/test_trace_id_embedding.py` | Human-readable trace id embedding | 32-hex trace id suffix matches execution id digits; non-digit fallback |
| `tests/test_observation_mapping.py` | Observation type classification breadth | Exact sets, regex fallbacks, category fallback semantics |
| `tests/test_negative_inferred_parent.py` | Negative inferred-parent case | `n8n.graph.inferred_parent` absent when runtime source present |
| `tests/test_config_settings.py` | Settings precedence & prefix semantics | DSN construction; prefix unset/empty/custom behavior |
| `tests/test_db_stream.py` (integration) | DB streaming (requires real DB) | Read-only access; start_after ordering |

### Running All Tests

```fish
pytest -q
```

### Fast Feedback (Core Logic Only)
Skip integration DB test (marks it skipped automatically if `PG_DSN` missing):
```fish
pytest -q -k 'not stream_reads_rows_without_modification and not stream_respects_start_after_id'
```

### Run A Single File
```fish
pytest tests/test_input_propagation.py -q
```

### Run A Single Test
```fish
pytest tests/test_trace_id_embedding.py::test_trace_id_embedding_simple -q
```

### Integration Database Test
Provide a real n8n Postgres with data and set `PG_DSN`:
```fish
set -x PG_DSN postgresql://user:pass@localhost:5432/n8n
pytest tests/test_db_stream.py -q
```
If no rows exist, the test will skip gracefully.

### Adding New Tests (Checklist)
When you add or alter behavior:
1. New metadata key? -> Add assertions locating at least one span with that key.
2. New parent resolution rule? -> Add a scenario that exercises the new precedence plus a negative control.
3. Change to binary stripping? -> Add case containing old vs new structure to prevent regressions.
4. New observation type / mapping rule? -> Extend `test_observation_mapping.py` with representative node type.
5. New environment setting? -> Add coverage in `test_config_settings.py` and document in README + instructions.

### Determinism Guidance
Run the same mapper input twice and compare ordered span IDs and generation list lengths. Any difference demands investigation. See `test_trace_and_metadata.py` for the pattern used.

### Timezone & Temporal Data Policy
All datetimes must be timezone-aware (UTC). Rationale:
1. Prevent silent drift / ambiguity when comparing timestamps across systems.
2. Avoid deprecated `datetime.utcnow()` usage (Python deprecation trajectory).
3. Ensure OTLP exporters receive consistent RFC3339/ISO8601 semantics.

Enforcement:
- `map_execution_to_langfuse` normalizes any naive `startedAt` / `stoppedAt` to `timezone.utc`.
- Test `tests/test_timezone_awareness.py::test_no_naive_datetime_patterns` scans both `src/` and `tests/` directories for forbidden patterns.
- Forbidden patterns: `datetime.utcnow(`, bare `datetime.now()` with no timezone argument, or `datetime.now(<args>)` missing `timezone.utc` or `tz=`.
- Allowed patterns: `datetime.now(timezone.utc)`, `datetime.now(tz=timezone.utc)`, or any timezone-aware object sourced from DB drivers.
- Intentional naive usage inside a test fixture can be whitelisted by adding an inline comment marker `# allow-naive-datetime` on the same line.

If you introduce code manipulating timestamps:
- Always preserve existing timezone info; only attach `timezone.utc` if `tzinfo is None`.
- If you ingest external (user) timestamps, parse them with a library that preserves/attaches timezone offsets (future enhancement may add strict parsing utilities).

### Debug Tips
| Symptom | Likely Cause | How to Isolate |
|---------|-------------|----------------|
| Missing child span | Parent resolution precedence mismatch | Add a minimal two-node fixture replicating timing & sources |
| Input not inferred | Parent output not cacheable (not dict) OR size guard | Re-run with `truncate_limit=None` in test and assert difference |
| Trace id not readable | Old exporter logic still cached | Ensure you restarted the process; `_build_human_trace_id` tests should pass |
| Binary leaked in span output | New payload shape not covered by stripping | Create failing test with exact shape then patch `_strip_binary_payload` |

### Coverage Philosophy
We prioritize behavior regression detection over line coverage metrics. Each invariant has at least one direct assertion. Avoid asserting on incidental formatting (e.g., not the entire JSON string), prefer presence/absence of semantic markers and flags.

---

### AI-Only Filtering

Enable focused GenAI span export by filtering out non-AI workflow logic.

Activation (CLI):
```fish
python -m src backfill --limit 100 --no-dry-run --filter-ai-only
```
Environment alternative:
```fish
set -x FILTER_AI_ONLY true
python -m src backfill --limit 100 --no-dry-run
```

AI Node Detection:
1. Category equals `AI/LangChain Nodes` (fast path)
2. Node type present in authoritative set from `@n8n/n8n-nodes-langchain` (see `observation_mapper.py`)

Retention Rules:
* Always keep root span.
* Keep AI spans.
* Keep non-AI ancestors along parent chain to each AI span.
* Remove other non-AI spans.
* If no AI spans exist: export only root (metadata `n8n.filter.no_ai_spans=true`).

Metadata (root span):
* `n8n.filter.ai_only=true`
* `n8n.filter.excluded_node_count=<int>`
* `n8n.filter.no_ai_spans=true` (only when no AI spans remain)

Use Cases:
* Reduce payload volume in large historical backfills.
* Focus dashboards on LLM / embedding performance.
* Exclude control-flow noise (Set/If/Wait nodes) when analyzing model quality.

Disable via `--no-filter-ai-only` or `set -x FILTER_AI_ONLY false`.

Test Coverage: `tests/test_ai_filtering.py` asserts classification, ancestor retention, exclusion counts, and empty-AI execution behavior.

---


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

Binary or very large base64 payloads are removed pre-export to avoid excessive OTLP span sizes while retaining structural context. When media upload is enabled these stripped assets are converted into Langfuse media objects and span outputs patched with media token strings.

Detection heuristics:
- Standard n8n `binary` node structure (`binary` -> item -> `{ data: <b64>, mimeType: ... }`).
- Any long (>200 chars) base64-looking string (regex match) or strings starting with common base64 magic like `/9j/` (JPEG).
- Item-level per-output binaries: when outputs are emitted as nested list structures of objects each containing a sibling `binary` and `json` key (e.g. `main` -> list[list[list[{ json, binary }]]] ) their `binary` slot maps are **promoted** and merged into a synthetic top-level `binary` object before placeholder insertion. First occurrence of a slot name wins (deterministic). Span metadata gains `n8n.io.promoted_item_binary=true` when this promotion occurs. If normalization unwraps list content to a list root, the promoted `binary` dict is preserved by wrapping the final output as `{ "binary": { ... }, "_items": [ ... ] }`.

Replacement (legacy / disabled media):
- In `binary` objects: `data` value replaced with `binary omitted` plus `_omitted_len` metadata.
- Standalone strings: replaced with `{ "_binary": true, "note": "binary omitted", "_omitted_len": <len> }`.

Media upload (enabled):
1. Mapper collects each binary asset (base64 payload, filename, mimeType, size, SHA256) and inserts a placeholder:
	 ```json
	 {
		 "_media_pending": true,
		 "sha256": "<hex>",
		 "bytes": <decoded_size_bytes>,
		 "base64_len": <original_base64_length>,
		 "slot": "<optional slot or field path>"
	 }
	 ```
2. `media_api.py` calls Langfuse `POST /api/public/media` with payload keys `traceId`, `contentType`, `contentLength`, `sha256Hash` (base64 digest), and `field`. Response may include a presigned `uploadUrl` (if new) or omit it (deduplicated existing media).
3. When `uploadUrl` is present the raw bytes are PUT once with required headers `Content-Type` and `x-amz-checksum-sha256` (same base64 digest); otherwise skip upload.
4. Placeholder is replaced by a token string:
```
@@@langfuseMedia:type=<mime>|id=<mediaId>|source=base64_data_uri@@@
```
5. Failures (API / upload / oversize / decode) leave placeholder intact and set `n8n.media.upload_failed=true` (fail-open).
6. `n8n.media.asset_count` increments per successfully patched placeholder.
7. Extended discovery (data URLs, file-like dicts, contextual base64) produces identical placeholders; capped by EXTENDED_MEDIA_SCAN_MAX_ASSETS. Exceeding cap leaves remaining raw values untouched and records `scan_asset_limit` in `n8n.media.error_codes`.

Item-level promotion edge cases:
- If multiple list items contain the same binary slot name only the first is retained (idempotent order defined by traversal).
- Promotion happens only when no top-level `binary` key already exists (non-destructive).
- Wrapper reattachment (the `{ "binary": ..., "_items": [...] }` shape) occurs only when normalization would otherwise drop the promoted `binary` due to list unwrapping.

7. Failures accumulate codes in `n8n.media.error_codes` (list) for diagnostics (see Media Error Codes table).
	Deduplicated (no `uploadUrl`) responses are considered successful and produce no error codes.

Disable by leaving `ENABLE_MEDIA_UPLOAD` false (default); behavior reverts to legacy redaction.

This occurs even when truncation is disabled (`TRUNCATE_FIELD_LEN=0`). Future environment knobs (planned): `BINARY_PLACEHOLDER`, `BINARY_MIN_LEN`.

## Media Upload (Langfuse Token) Quick Start

Enable (fish):
```fish
set -x ENABLE_MEDIA_UPLOAD true
set -x LANGFUSE_HOST https://cloud.langfuse.com
set -x LANGFUSE_PUBLIC_KEY lf_pk_...
set -x LANGFUSE_SECRET_KEY lf_sk_...
set -x FILTER_AI_ONLY false  # explicitly disable AI filtering
python -m src backfill --limit 10 --dry-run
```
If correctly configured you will see patched span outputs containing `@@@langfuseMedia:` tokens and metadata `n8n.media.asset_count`.

Disable:
```fish
set -e ENABLE_MEDIA_UPLOAD
```

Planned evolutions: additional failure retries, large streaming upload optimizations.
Recent addition: item-level binary promotion + preservation wrapper (see Binary / Large Payload Handling section) to ensure media tokens appear for nodes emitting per-item binaries only.

### Media Upload Troubleshooting

#### **Problem: Media tokens present in span output but NO PREVIEWS in Langfuse UI**

**Primary Cause (Current)**: Missing `observationId` linkage in the `POST /api/public/media` create
payload. The Langfuse UI does **not** scrape token strings from JSON to discover assets; it queries the
backend `observation_media` table keyed by span id. If the create call omitted `observationId`, the
media becomes trace-scoped only and will not appear inside that span's Media panel.

**Invariant**: Every asset originating from a span must call create with:
```
{
	"traceId": <execution_id_str>,
	"observationId": <span.id>,
	"contentLength": <bytes>,
	"sha256Hash": <base64_sha256>,
	"field": "output" | "input" | "metadata",
	"contentType": <mime?>
}
```
If `observationId` is missing you will see tokens like
`@@@langfuseMedia:type=image/png|id=...|source=base64_data_uri@@@` in the span output JSON but an empty
media panel.

**Secondary (Legacy) Cause (Now Fixed)**: Earlier versions double-stringified span outputs with Python
`str()` causing invalid JSON for token parsing. This is no longer the active failure mode but remains
covered by `tests/test_media_token_json_serialization.py`.

**Diagnosis Steps**:
1. Enable DEBUG logs and confirm lines: `media create ... observation_id=<span_id>` appear.
2. Inspect one create request (proxy/log) → ensure `observationId` present.
3. Check span metadata: `n8n.media.asset_count` > 0 and absence of `n8n.media.upload_failed`.
4. If assets deduplicated (no `uploadUrl`), still expect previews provided `observationId` was sent.
5. For self-hosted: verify object storage (S3/minio) accessible; network 403/AccessDenied indicates
	 missing required headers (`Content-Type`, `x-amz-checksum-sha256`).

**Remediation**:
- Upgrade to a version including observation linkage (look for `test_media_observation_link.py` in repo).
- If you forked earlier code, patch `media_api.py` so `create_media(... observation_id=span.id ...)` is used.

**Quick Verification (Python REPL)**:
After a run, locate a span with media metadata and assert the output contains tokens and metadata count:
```python
assert '"n8n.media.asset_count"' in json.dumps(span.metadata)
assert '@@@langfuseMedia:' in span.output
```

**Relevant Tests**:
- `test_media_observation_link.py` – ensures `observationId` included.
- `test_media_api_tokens.py` – token substitution & dedupe path.
- `test_media_token_json_serialization.py` – guards against double-stringify regression.


## Roadmap (Upcoming Enhancements)

1. Media upload workflow (store omitted binaries via Langfuse media API + token substitution).
2. Error retries / resilient OTLP + media upload with dead-letter logging.
3. Performance tuning (parallel export, async batching, memory caps for large runs).
4. Additional filtering flags (status, time window, workflow id inclusion/exclusion).
5. Masking / PII scrubbing and configurable redaction rules.
6. Extended observation classification & multimodal span enrichment.
7. Optional tagging of agent root spans (`n8n.agent.root=true`) and richer lineage metadata.

### Documentation Sync Summary
The following sections were added or substantively updated in `.github/copilot-instructions.md` and are reflected here:

Added:
- Timezone Normalization / Policy
- Human Trace ID Embedding
- Parent Precedence Table
- Input Propagation Semantics
- Backpressure & Flush Strategy
- Detailed Pointer Decoding Algorithm
- Regression Checklist

Updated:
- Generation Detection Heuristics
- Observation Mapping Clarifications
- Binary Handling (unconditional stripping statement)
- Media / Multimodal roadmap consolidation
- Development Plan (future iterations reordered / clarified)
 - Added media token upload feature (Langfuse Media API)

Hardened:
- Guardrails around purity (mapper has no network I/O)
- Environment variable completeness & precedence semantics
- Testing obligations (each invariant requires at least one assertion)

Consolidated:
- Media + multimodality roadmap (single authoritative location)

If any future change affects these documented areas, update both this README and `.github/copilot-instructions.md` plus add/adjust tests in the same pull request. Media enhancements MUST add tests and expand the Media section (both files) in the same PR.

---

## Contributing

1. Fork & branch.
2. Add/adjust tests.
3. Run formatting/lint: `ruff check --fix .`.
4. Ensure tests pass.
5. Open PR with context (data volume considerations welcome).

### Development: Timezone Policy Recap
Do not introduce naive datetimes. Use `datetime.now(timezone.utc)` instead of `datetime.utcnow()`. If you receive a naive datetime from an external library, explicitly attach `timezone.utc` before further processing. The CI test `test_timezone_awareness.py` will fail your PR if forbidden patterns are detected. For rare cases where a naive datetime is intentionally required for a test scenario, append `# allow-naive-datetime` to that specific line.


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
