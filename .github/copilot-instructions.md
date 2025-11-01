# AI Coding Agent Instructions for n8n-langfuse-shipper

## Quick Reference (AI Agent Onboarding)

**What**: Python ETL pipeline streaming n8n PostgreSQL execution history → Langfuse traces via OTLP
**Why**: Backfill historical workflow executions for AI observability and debugging
**Stack**: Python 3.12+, Pydantic models, OpenTelemetry SDK, psycopg3, Fish shell

### 5-Minute Orientation
- **Entry point**: `src/__main__.py` - Typer CLI orchestrating Extract→Transform→Load pipeline
- **Core transform**: `src/mapping/orchestrator.py` - maps n8n NodeRuns to Langfuse spans with deterministic IDs
- **Models**: `src/models/n8n.py` + `src/models/langfuse.py` - Pydantic validation throughout
- **Config**: `src/config.py` - Pydantic Settings from env vars (see table line ~140)
- **Tests**: Run `pytest` or `pytest -q` in Fish shell - 30+ test files assert invariants
- **Dev workflow**: `ruff check . && mypy src && pytest` before commits (line-length cap: 100 chars)

### Critical Invariants (NEVER VIOLATE - See line ~115)
1. **Deterministic IDs**: UUIDv5 with `SPAN_NAMESPACE` in `mapping/id_utils.py` - changing breaks idempotency
2. **Execution ID appears once**: Root span metadata `n8n.execution.id` only (never duplicated)
3. **Parent resolution**: 5-tier precedence (Agent Hierarchy → Runtime Exact → Runtime Last → Static Graph → Root) - see line ~390
4. **Binary stripping**: ALWAYS unconditional (even when truncation disabled) - see line ~230
5. **Timezone aware**: All datetimes UTC-aware, NEVER use `datetime.utcnow()` (test enforces this)
6. **Fish shell**: NO Bash heredocs/arrays; use `set -x VAR value` for exports; Python one-liners: `python -c "import sys; print('ok')"` pattern

### Where to Look (Line References)
- **Mapping modules**: `src/mapping/` - 11 pure submodules (orchestrator, binary_sanitizer, generation, parent_resolution, etc.)
- **Parent resolution rules**: Line ~390 (Precedence table) + line ~230 (Agent Hierarchy explanation)
- **Generation detection heuristics**: Line ~290 (tokenUsage presence OR provider markers)
- **Binary & Media handling**: Line ~340 (stripping) + line ~370 (media token flow when `ENABLE_MEDIA_UPLOAD=true`)
- **Reserved metadata keys**: Line ~630 (root span, spans, Gemini anomaly, media flags)
- **Testing contract**: Line ~540 (deterministic IDs, parent precedence, binary stripping, timezone tests)
- **Environment variables**: Line ~140 (complete table with defaults)

### Common Tasks
```fish
# Run specific test file
pytest tests/test_mapper.py -v

# Type check only mapping module
mypy src/mapping/

# Dry-run first 25 executions
python -m src backfill --limit 25 --dry-run

# Check for line length violations
ruff check . --select E501
```

### Module Refactor Status (Line ~50)
Mapper logic extracted into `src/mapping/` subpackage (completed). Facade `src/mapper.py` preserves public API for backward compatibility. Future extractions documented at line ~70.

---

## Purpose
Python-based microservice for high-throughput backfill of historical n8n execution data from PostgreSQL to Langfuse via OpenTelemetry (OTLP) endpoint. Focus: correctness, performance, robustness for large-scale data migration.

## Document Sync & Version Policy
This file is a normative contract. Any behavioral change to mapping, identifiers, timestamps, parent resolution, truncation, binary stripping, generation detection, or environment semantics MUST be reflected here and in matching tests plus the README in the same pull request. Tests are the authority when ambiguity arises; if tests and this document diverge, update this document. Do not introduce silent behavior drift.

---

## Glossary (Authoritative Definitions)

Use these exact meanings in code comments, docs, and tests. Adding a new term? Update here in same PR.

* **Agent Hierarchy:** Parent-child relationship inferred from any non-`main` workflow connection whose type starts with `ai_` making the agent span the parent.
* **Backpressure:** Soft limiting mechanism (queue size vs `EXPORT_QUEUE_SOFT_LIMIT`) triggering exporter flush + sleep.
* **Binary Stripping:** Unconditional replacement of binary/base64-like payloads with stable placeholders prior to (optional) truncation.
* **Checkpoint:** Persistent last processed execution id stored on successful export; guarantees idempotent resume.
* **Deterministic IDs:** Stable UUIDv5 span ids and raw execution id as logical trace id; re-processing identical input yields identical structure.
* **Execution:** Single joined `execution_entity` + `execution_data` record; maps to exactly one Langfuse trace.
* **Execution id:** Raw integer primary key; appears only once as root span metadata `n8n.execution.id`.
* **Generation:** Span classified as LLM call via heuristics (`tokenUsage` presence OR provider substring) optionally with token usage and model metadata.
* **Human-readable Trace ID (OTLP):** 32-hex trace id produced by `_build_human_trace_id` embedding zero-padded execution id digits as suffix.
* **Input Propagation:** Injecting parent span raw output as child logical input when `inputOverride` absent.
* **Multimodality / Media Upload:** Feature-flag gated Langfuse Media API token flow. Mapper collects binary assets inserting temporary placeholders; post-map phase exchanges each for stable Langfuse media token string. Fail-open: on error or oversize, original redaction placeholder stays.
* **NodeRun:** Runtime execution instance of a workflow node (timing, status, source chain, outputs, error).
* **Observation Type:** Semantic classification (agent/tool/chain/etc.) via fallback chain: exact → regex → category → default span.
* **Parent Resolution Precedence:** Ordered strategy: Agent Hierarchy → Runtime exact run → Runtime last span → Static reverse graph → Root.
* **Pointer-Compressed Execution:** Alternative list-based JSON encoding using index string references; reconstructed into canonical `runData`.
* **Root Span:** Span representing entire execution; holds execution id metadata; parent of all top-level node spans.
* **runData:** Canonical dict mapping node name → list[NodeRun] reconstructed or parsed from execution data JSON.
* **Truncation:** Optional length limiting of stringified input/output fields when `TRUNCATE_FIELD_LEN > 0`; does not disable binary stripping.
* **Usage Extraction:** Normalize `tokenUsage` variants into canonical `input`, `output`, `total` counts (synthesizing `total` as input+output when absent).

---

## Architecture

The service operates as a standalone ETL process, designed for containerized, cron-based execution.

```mermaid
graph TD
    subgraph PIPELINE [Python Service]
        A[Extract] --> PD[Pointer Decode]
        PD --> TR[Transform]
        TR --> PR[Parent Resolve]
        PR --> IP[Input Propagate]
        IP --> BS[Binary Strip/Truncate]
        BS --> MU[Media Upload]
        MU --> EX[Export]
        EX --> BP[Backpressure]
        BP --> CK[Checkpoint]
    end
    EX --> LF[Langfuse OTLP]
    CK -.-> A
```

    ### Module Boundaries (Refactor Phase)

    Mapper logic is being decomposed into a `src/mapping/` subpackage. Purity and
    determinism MUST be preserved. New modules (initial tranche):

    * `mapping.time_utils` – epoch millisecond → UTC conversion helpers.
    * `mapping.id_utils` – deterministic UUIDv5 span id helpers (`SPAN_NAMESPACE`)
      – ID format MUST remain unchanged; tests assert stability.
        * `mapping.binary_sanitizer` – binary/base64 detection & redaction utilities.
        * `mapping.orchestrator` – procedural mapping loop (execution → spans), AI filtering
            window logic, binary asset collection, IO prep, model & anomaly helpers; keeps
            `mapper.py` as thin facade. No behavior drift allowed.

    Refactor Rules:
    1. Public API (`map_execution_to_langfuse`, `map_execution_with_assets`) stays in
        `mapper.py` (facade pattern) to avoid import breakage in tests & external
        integrations.
    2. No behavioral drift: any change to detection / stripping / ID generation
        requires test updates and documentation adjustments in this file + README.
    3. Subpackage modules MUST remain pure (no network / DB writes) and side-effect
        free apart from deterministic logging.
    4. Deterministic ID namespace/seeds unchanged; `SPAN_NAMESPACE` exported from
        `mapping.id_utils` and re-imported into facade to keep existing tests green.
    5. Binary stripping invariants unchanged: unconditional redaction prior to (optional)
        truncation.
    6. Future planned extractions: IO normalization (`io_normalizer.py`), generation
        classification (`generation.py`), model extraction (`model_extractor.py`), parent
        resolution (`parent_resolution.py`), mapping context state holder (`mapping_context.py`).
    7. Each extraction phase MUST run full test suite; snapshot (golden) comparison ensures
        structural parity of spans & metadata.

    Violation of these rules (silent drift, dependency cycles, network calls) is a
    contract breach.

**Pipeline Stages:**
1. **Extract:** Stream execution records from PostgreSQL (`execution_entity` + `execution_data`).
2. **Pointer Decode:** Reconstruct `runData` when compact list format detected.
3. **Transform:** Map execution to Langfuse trace; each node run to span.
4. **Parent Resolve:** Apply precedence: Agent Hierarchy → Runtime exact → Runtime last → Static graph → Root.
5. **Input Propagate:** Infer child input from parent output when `inputOverride` absent.
6. **Binary Strip/Truncate:** Unconditional binary redaction; optional truncation when `TRUNCATE_FIELD_LEN > 0`.
7. **Media Upload:** (Optional) Exchange binary placeholders for Langfuse media tokens when `ENABLE_MEDIA_UPLOAD=true`.
8. **Export:** Send OTLP spans to Langfuse.
9. **Backpressure:** Flush + sleep when queue exceeds soft limit.
10. **Checkpoint:** Persist last successful execution id.

**Root-only fallback:** Malformed/missing `runData` still produces trace with execution root span; never drop execution silently.

---

## Core Data Models (Pydantic)

All internal structures defined with Pydantic models for type safety and validation. Canonical definitions live in `src/models/`.

### N8N Models (`src/models/n8n.py`)

Represent JSON data from `n8n_execution_data` and `n8n_execution_entity` tables.

```python
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class NodeRunSource(BaseModel):
    previousNode: Optional[str] = None
    previousNodeRun: Optional[int] = None

class NodeRun(BaseModel):
    startTime: int
    executionTime: int
    executionStatus: str
    data: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[List[NodeRunSource]] = None
    inputOverride: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class ResultData(BaseModel):
    runData: Dict[str, List[NodeRun]] = Field(default_factory=dict)

class ExecutionDataDetails(BaseModel):
    resultData: ResultData

class ExecutionData(BaseModel):
    executionData: ExecutionDataDetails

class WorkflowNode(BaseModel):
    name: str
    type: str
    category: Optional[str] = None

class WorkflowData(BaseModel):
    nodes: List[WorkflowNode]
    connections: Dict[str, Any] = Field(default_factory=dict)

class N8nExecutionRecord(BaseModel):
    id: int
    workflowId: str
    status: str
    startedAt: datetime
    stoppedAt: datetime
    workflowData: WorkflowData
    data: ExecutionData
```

**DO NOT inline-edit model shapes here.** Change code + tests, then update this narrative only if semantics (fields/meaning) shift.

### Langfuse Models (`src/models/langfuse.py`)

Internal logical structures prior to OTLP span creation. Extend with extreme caution: new fields demand tests & README alignment.

```python
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class LangfuseUsage(BaseModel):
    input: Optional[int] = None
    output: Optional[int] = None
    total: Optional[int] = None

class LangfuseSpan(BaseModel):
    id: str
    trace_id: str
    parent_id: Optional[str] = None
    name: str
    start_time: datetime
    end_time: datetime
    observation_type: str = "span"
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    token_usage: Optional[LangfuseUsage] = None
    status: Optional[str] = None

class LangfuseTrace(BaseModel):
    id: str
    name: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    spans: List[LangfuseSpan] = Field(default_factory=list)
```

---

## Core Invariants

**Critical rules enforced across the codebase:**

1. One n8n execution row maps to exactly one Langfuse trace.
2. Node span id format: `UUIDv5(namespace=SPAN_NAMESPACE, name=f"{trace_id}:{node_name}:{run_index}")` (deterministic).
3. Execution id appears only once: root span metadata key `n8n.execution.id` (never duplicated elsewhere).
4. All timestamps are timezone-aware UTC. Naive datetimes normalized to UTC. Never use `datetime.utcnow()`.
5. Spans emitted strictly in chronological order (NodeRun `startTime`) so parents precede children.
6. Binary/base64 stripping ALWAYS applies. Truncation is opt-in (`TRUNCATE_FIELD_LEN=0` disables truncation but not binary stripping).
7. Parsing failures still yield a root span; never drop execution silently.
8. All internal data structures defined with Pydantic models (validation + type safety mandatory).
9. Database access is read-only (SELECTs only); respects dynamic table prefix logic.
10. Determinism: identical input rows yield identical span/trace ids & structures.
11. AI-only filtering (when enabled) must preserve root span, retain all AI node spans, and include
    ancestor chain context; executions with no AI nodes produce root-only trace flagged via
    `n8n.filter.no_ai_spans=true`.

---

## Data Parsing & Resilience

The `data` column in `n8n_execution_data` supports multiple formats:

* **Standard:** JSON object containing `executionData` key.
* **Pointer-Compressed:** Top-level JSON array where objects reference other elements by index. Function `_decode_compact_pointer_execution` reconstructs canonical `runData`. Uses cycle protection and memoization. On failure returns `None` for path probing fallback.
* **Alternative Paths:** `runData` might be at different nested paths. Parser `_build_execution_data` probes multiple candidate paths.
* **Empty/Invalid:** If `runData` cannot be found, processing continues → trace with root span only.

**Pointer-Compressed Decoding Steps:**
1. Detect top-level list form; if not list → return None.
2. Treat string indices as pointers to other list entries.
3. Recursively expand with memoization; cycle detection aborts → return None.
4. Search reconstructed structure for `executionData.resultData.runData` paths.
5. Validate `runData` shape (dict[str, list[NodeRun-like]]). Success → build model; else fallback to path probing.

---

## Mapping Logic

Core transformation in `mapper` module. Uses hybrid approach combining runtime data with static workflow graph.

### Agent/Tool Hierarchy

n8n pattern (especially LangChain nodes): "Agent" node uses other nodes as "Tools", "LLMs", or "Memory". Revealed by `workflowData.connections`:
* `type: "main"` → sequential step.
* `type` beginning with `ai_*` (e.g. `ai_tool`, `ai_languageModel`, `ai_memory`, `ai_outputParser`, `ai_retriever`) from component to agent → **hierarchical relationship**. Agent span is parent of component span.

### Trace Mapping

* `N8nExecutionRecord` → single `LangfuseTrace`.
* `LangfuseTrace.id` deterministic: `str(record.id)` (raw execution id as string). **One n8n instance per Langfuse project REQUIRED** (avoid execution id collisions).
* Root `LangfuseSpan` created to represent entire execution. All top-level node spans are children.

### Span Mapping

* Each `NodeRun` → `LangfuseSpan`.
* `LangfuseSpan.id` deterministic: UUIDv5 hash of `f"{trace_id}:{node_name}:{run_index}"`.

**Parent Resolution (Multi-tier precedence):**
1. **Agent Hierarchy:** Check `workflowData.connections`. If node connected to Agent via non-`main` `ai_*` type → parent is most recent Agent span. Sets metadata: `n8n.agent.parent`, `n8n.agent.link_type`.
2. **Runtime Sequential (Exact Run):** Use `run.source[0].previousNode` + `previousNodeRun` → link to exact run index. Sets: `n8n.node.previous_node_run`.
3. **Runtime Sequential (Last Seen):** `previousNode` without run index → last seen span for that node. Sets: `n8n.node.previous_node`.
4. **Static Graph Fallback:** Reverse-edge map from `workflowData.connections` infers likely parent. Sets: `n8n.graph.inferred_parent=true`.
5. **Root Fallback:** If no parent determined → link to root execution span.

**Input Propagation:**
If `NodeRun` lacks `inputOverride`, logical input inferred from cached raw output of resolved parent: `{ "inferredFrom": <parent>, "data": <parent_raw_output> }`. Propagation always occurs; size guard limiting cached output applies only when truncation enabled (`TRUNCATE_FIELD_LEN > 0`).

### Observation Type Mapping

Use `observation_mapper.py` to classify each node based on type and category. Fallback chain: exact → regex → category → default span.

### Generation Detection

**Heuristics (ordered, current implementation):**
1. Presence of `tokenUsage` object at any depth inside `run.data` (depth-limited recursive search; explicit signal).
2. Fallback: node type (case-insensitive) contains provider marker: `openai`, `anthropic`, `gemini`, `mistral`, `groq`, `lmchat`, `lmopenai`, `cohere`, `deepseek`, `ollama`, `openrouter`, `bedrock`, `vertex`, `huggingface`, `xai`, `limescape`.
   * **Exclusions:** If type also contains `embedding`, `embeddings`, or `reranker` → NOT classified as generation.

**If matched:**
* Populate `LangfuseSpan.model` best-effort from node type/name or breadth-first nested search for variant keys (`model`, `model_name`, `modelId`, `model_id`) inside `run.data` output. When missing, attach debug metadata flag `n8n.model.missing=true`.
* `_extract_usage` normalizes to `input`/`output`/`total`; if `total` absent but input & output present → synthesized (input+output). Precedence: existing input/output/total > promptTokens/completionTokens/totalTokens > prompt/completion/total. Custom flattened Limescape Docs counters (`totalInputTokens`, `totalOutputTokens`, `totalTokens`) also detected.
* OTLP exporter emits `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens` only for provided fields plus `model`, `langfuse.observation.model.name` when `model` populated.

**Generation Output Text Extraction (Concise Behavior):**
1. Gemini/Vertex chat: first non-empty `response.generations[0][0].text`.
2. Limescape Docs custom node: prefer top-level `markdown` field (post normalization) when non-empty.
3. Fallback: serialized normalized JSON output.

**Notes:**
* Provider marker expansions MUST update this section, README, and `tests/test_generation_heuristic.py` in same PR.
* Never infer generation purely from output length or text presence.
* Embedding/reranker nodes intentionally excluded unless explicit `tokenUsage`.

**LangChain LMChat System Prompt Stripping:**

LangChain LMChat nodes (@n8n/n8n-nodes-langchain.lmChat*) combine System and User prompts in one message blob.
Split marker sequence (literal token sequence):

```text
\n\n## START PROCESSING\n\nHuman: ##
```

For clarity: the System segment ends just before the line containing ## START PROCESSING; the User
segment begins at the line starting with Human: ## (hash symbols are part of the literal delimiter
and are not Markdown headings). This explicit code block avoids editor diagnostics that might
interpret the double-hash sequences as tool directives.

Function _strip_system_prompt_from_langchain_lmchat() strips System prompts:
* Called in `_prepare_io_and_output()` **BEFORE** `_normalize_node_io()` (preserves structure).
* Only processes generation spans with "lmchat" in node type (case-insensitive).
* **Recursively searches** for `messages` arrays at any depth (up to 25 levels) since actual n8n data nests messages deeply (e.g., `ai_languageModel[0][0]['json']['messages']`).
* Handles both message formats: list of strings `["System: ... Human: ## ..."]` and list of dicts `[{"content": "System: ... Human: ## ..."}]`.
* Strips everything before the Human: ## marker when found.
* Fail-open: returns original input on any error.
* Logs depth and characters removed when stripping occurs.

Tests: `test_lmchat_system_prompt_strip.py` (6 tests covering dict format, string format, deeply nested, missing marker, non-lmChat, multiple messages).

### Binary & Multimodality

**Binary Stripping (Unconditional):**
* `binary` objects: replace `data` with `"binary omitted"` + attach `_omitted_len`; retain metadata (filename, mimeType, etc.).
* Standalone base64 strings: replace with `{ "_binary": true, "note": "binary omitted", "_omitted_len": <length> }`.
* Helpers: `_likely_binary_b64`, `_contains_binary_marker`, `_strip_binary_payload`.

**Wrapper Unwrapping & Binary Preservation:**
When normalizing node I/O, unwrap channel/list/json wrappers while merging back any top-level `binary` block. Prevents loss of media placeholders/redaction objects when output contains both wrapper and binary. Tests: `test_binary_unwrap_merge.py`.

**Media Upload (Token Flow) — When `ENABLE_MEDIA_UPLOAD=true`:**
1. Inline binary asset collection during mapping produces `MappedTraceWithAssets`.
2. Temporary placeholder inserted where binary/base64 detected:
   ```json
   {
       "_media_pending": true,
       "sha256": "<hex>",
       "bytes": <decoded_size_bytes>,
       "base64_len": <original_base64_length>,
       "slot": "<optional source slot or field path>"
   }
   ```
3. Post-map patch phase (`media_api.py`) iterates collected assets.
4. For each asset ≤ `MEDIA_MAX_BYTES` (decoded): call Langfuse Media API create endpoint.
5. If response includes upload instruction (presigned URL): upload raw bytes.
6. On success, placeholder replaced with stable token: `@@@langfuseMedia:type=<mime>|id=<mediaId>|source=base64_data_uri@@@`.
7. Per-span replacement count aggregated into metadata `n8n.media.asset_count` (only successful tokens).
8. Fail-open: oversize, decode errors, API/upload failures leave placeholder + set `n8n.media.upload_failed=true`. Processing continues; export never aborted.
9. Determinism: same binary input (SHA256) yields idempotent create semantics on Langfuse side.
 10. Multi-provider header injection: If presigned upload URL host contains `blob.core.windows.net` (Azure Blob) we MUST send `x-ms-blob-type: BlockBlob` in addition to existing S3-style headers. Absence previously produced 400 `MissingRequiredHeader` (HeaderName `x-ms-blob-type`). S3 paths ignore this extra header (fail-open). Detection is substring-based and documented here; any change requires test + README update.

**Extended Binary Discovery:**
In addition to canonical `run.data.binary` blocks, mapper scans for:
1. Data URLs: `data:<mime>;base64,<payload>`
2. File-like dicts: `{ mimeType|fileType, fileName|name, data:<base64> }`
3. Long base64 strings (≥64 chars) inside dicts with file-indicative keys (`mimeType`, `fileName`, `fileType`).

Discovered assets receive `_media_pending` placeholders and uploaded/tokenized like canonical assets. Scan bounded by `EXTENDED_MEDIA_SCAN_MAX_ASSETS` per node run. Tests: `test_extended_binary_discovery.py`.

**Item-Level Binary Promotion:**
Some nodes emit outputs as nested list wrappers with per-item `binary` and `json` keys. Mapper now:
1. Scans wrapper list structures (depth ≤3) for dict items with `binary` key when no top-level `binary` exists.
2. Merges all discovered slot maps into synthetic top-level `binary` dict (first occurrence wins).
3. Continues canonical placeholder insertion on promoted block.
4. Adds span metadata `n8n.io.promoted_item_binary=true` if promotion occurred.
5. If normalization would drop promoted `binary`, wraps as: `{ "binary": { ... }, "_items": [ <unwrapped> ] }`.

Edge cases: promotion only when original lacks top-level `binary`; bounded depth 3 / 100 items. Test: `test_item_level_binary_promotion.py`.

**Media Guardrails:**
* Mapper purity (network I/O only in post-map patch phase).
* New/changed env vars require synchronized README + tests + this file.
* Disabled flag path yields identical span outputs as legacy redaction path (asserted by tests).

### Custom Node Classification (Limescape Docs)

Node type `n8n-nodes-limescape-docs.limescapeDocs` force-classified as `generation` observation even when `tokenUsage` absent. Provider marker `limescape` in generation heuristic list. Flattened usage keys (`totalInputTokens`, `totalOutputTokens`, `totalTokens`) recognized and mapped to `gen_ai.usage.*` attributes.

---

## OpenTelemetry Shipper

The `shipper.py` module converts internal `LangfuseTrace` model into OTel spans and exports them.

**Initialization:** OTLP exporter configured once with Langfuse endpoint and Basic Auth credentials.

**Span Creation:** For each `LangfuseSpan`, create OTel span with exact `start_time` and `end_time`.

**Attribute Mapping:**
* `langfuse.observation.type` ← `observation_type`
* `model` & `langfuse.observation.model.name` ← `model`
* `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens` ← normalized `token_usage`
* `langfuse.observation.usage_details` ← JSON string containing only present keys among `input`/`output`/`total`
* `langfuse.observation.status` ← normalized status (when available)
* `langfuse.observation.metadata.*` ← flattened span metadata
* Root span only: `langfuse.internal.as_root=true`
* Root span trace identity (when provided on `LangfuseTrace`): `user.id`, `session.id`, `langfuse.trace.tags` (JSON array), `langfuse.trace.input`, `langfuse.trace.output` (JSON serialized best-effort)

**Trace Attributes:** Set trace-level attributes on root span.

**Notes:**
* Root span metadata holds `n8n.execution.id`; not duplicated on trace metadata.
* Export order mirrors creation order; parents precede children.
* Dry-run mode constructs spans but does not send them.
* Media upload (if enabled) occurs before shipper export; span outputs already patched with tokens so exporter remains oblivious to raw binaries.
* Human-readable OTLP trace id embedding: exporter derives deterministic 32-hex trace id ending with zero-padded execution id digits (function `_build_human_trace_id`). Logical `LangfuseTrace.id` stays raw execution id string.

---

## Application Flow & Control

**Main Loop:** CLI script (`__main__.py`) loads checkpoint, streams execution batches from PostgreSQL, maps each record to `LangfuseTrace`, passes to shipper, updates checkpoint.

**Checkpointing:** `checkpoint.py` module atomically stores last successfully processed `executionId` in file.

**CLI Interface:** Use `Typer`. The `backfill` command supports:
* `--start-after-id`, `--limit`, `--dry-run`, `--debug`, `--debug-dump-dir`
* `--truncate-len` (0 disables truncation)
* `--require-execution-metadata` (only process if row exists in `<prefix>execution_metadata` with matching executionId)

---

## Environment Variables

### Core Database & Langfuse

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_DSN` | "" | Full PostgreSQL DSN. Takes precedence over component variables. |
| `DB_POSTGRESDB_HOST` | - | DB host (used only if `PG_DSN` empty). |
| `DB_POSTGRESDB_PORT` | `5432` | DB port. |
| `DB_POSTGRESDB_DATABASE` | - | DB name. |
| `DB_POSTGRESDB_USER` | `postgres` | DB user. |
| `DB_POSTGRESDB_PASSWORD` | "" | DB password. |
| `DB_POSTGRESDB_SCHEMA` | `public` | DB schema. |
| `DB_TABLE_PREFIX` | (required) | Mandatory table prefix. Set `n8n_` explicitly or blank for none. |
| `LANGFUSE_HOST` | "" | Base Langfuse host; exporter appends OTLP path if needed. |
| `LANGFUSE_PUBLIC_KEY` | "" | Langfuse public key (Basic Auth). |
| `LANGFUSE_SECRET_KEY` | "" | Langfuse secret key (Basic Auth). |

### Processing & Truncation

| Variable | Default | Description |
|----------|---------|-------------|
| `FETCH_BATCH_SIZE` | `100` | Max executions fetched per DB batch. |
| `CHECKPOINT_FILE` | `.backfill_checkpoint` | Path for last processed execution id. |
| `TRUNCATE_FIELD_LEN` | `0` | Max chars for input/output before truncation. `0` ⇒ disabled (binary still stripped). |
| `REQUIRE_EXECUTION_METADATA` | `false` | Only include executions having row in `<prefix>execution_metadata`. |
| `FILTER_AI_ONLY` | `false` | Export only AI node spans plus ancestor chain; root span always retained; adds metadata flags. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

### Media Upload (Token Flow)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MEDIA_UPLOAD` | `false` | Master feature flag. When false: only binary redaction; no token API calls. |
| `MEDIA_MAX_BYTES` | `25_000_000` | Max decoded size per asset; oversize remain redaction placeholders. |
| `EXTENDED_MEDIA_SCAN_MAX_ASSETS` | `250` | Cap on non-canonical discovered assets per node run. ≤0 disables. |

### Reliability / Backpressure

| Variable | Default | Description |
|----------|---------|-------------|
| `FLUSH_EVERY_N_TRACES` | `1` | Force `force_flush()` after every N traces. |
| `OTEL_MAX_QUEUE_SIZE` | `10000` | Span queue capacity for `BatchSpanProcessor`. |
| `OTEL_MAX_EXPORT_BATCH_SIZE` | `512` | Max spans per OTLP export request. |
| `OTEL_SCHEDULED_DELAY_MILLIS` | `200` | Max delay before batch export (ms). |
| `EXPORT_QUEUE_SOFT_LIMIT` | `5000` | Approx backlog threshold triggering sleep. |
| `EXPORT_SLEEP_MS` | `75` | Sleep duration (ms) when backlog exceeds soft limit. |

### Precedence Rules

1. `PG_DSN` (if non-empty) overrides component DB variables.
2. `DB_TABLE_PREFIX` must be explicitly set (blank allowed = no prefix).
3. `TRUNCATE_FIELD_LEN=0` means disabled (still strip binary); positive value triggers truncation & size guard.
4. CLI flags override env values for that invocation.

---

## Reserved Metadata Keys

**Root span:**
* `n8n.execution.id`

**All spans (optional context):**
* `n8n.node.type`, `n8n.node.category`, `n8n.node.run_index`, `n8n.node.execution_time_ms`, `n8n.node.execution_status`
* `n8n.node.previous_node`, `n8n.node.previous_node_run`
* `n8n.graph.inferred_parent`
* `n8n.agent.parent`, `n8n.agent.link_type`
* `n8n.truncated.input`, `n8n.truncated.output`
* `n8n.filter.ai_only`, `n8n.filter.excluded_node_count`, `n8n.filter.no_ai_spans`

**Gemini empty-output anomaly (Gemini/Vertex chat bug) & tool-calls suppression:**
* `n8n.gen.empty_output_bug` (bool) – true only when classified as anomaly
* `n8n.gen.empty_generation_info` (bool, if `generationInfo` object present but empty)
* `n8n.gen.tool_calls_pending` (bool) – empty output immediately followed by a `tool` span; suppression path
* `n8n.gen.prompt_tokens`, `n8n.gen.total_tokens`, `n8n.gen.completion_tokens`

**Condition (all true unless noted):**
1. `generations[0][0].text == ""`
2. `promptTokens > 0`
3. `totalTokens >= promptTokens`
4. `completionTokens == 0` OR field absent
5. (Optional) `generationInfo == {}` ⇒ sets `n8n.gen.empty_generation_info`

**Effect:**
* When NOT followed by a tool span: span `status` forced to `error`; synthetic `error.message = "Gemini empty output anomaly detected"` inserted when original `run.error` missing; structured log line emitted.
* When immediately followed by a tool span ("tool_calls" transition): error suppression; span retains original status; metadata `n8n.gen.tool_calls_pending=true` emitted with token counters; no synthetic error message.

**Span status normalization:** `LangfuseSpan.status` derived from logical success/error outcome; distinct from raw `n8n.node.execution_status` metadata.

**Media-specific keys:**
* `n8n.media.asset_count` – number of binary placeholders replaced.
* `n8n.media.upload_failed` – at least one asset failed.
* `n8n.media.error_codes` – list of failure codes.
* `n8n.media.surface_mode` – constant `inplace`.
* `n8n.media.preview_surface` – placeholder replaced at original path.
* `n8n.media.promoted_from_binary` – canonical slot also surfaced as shallow key.

**Media failure codes:**
* `decode_or_oversize` – decode error OR exceeded `MEDIA_MAX_BYTES`.
* `create_api_error` – non-2xx from Langfuse create endpoint.
* `missing_id` – create response missing media id.
* `upload_put_error` – presigned upload returned non-2xx.
* `serialization_error` – failed to serialize patched output.
* `scan_asset_limit` – exceeded `EXTENDED_MEDIA_SCAN_MAX_ASSETS` cap.
* `status_patch_error` – PATCH finalization failed.

---

## Testing Contract

### Testing Contract Overview

| Area | Guarantee | Representative Tests |
|------|-----------|----------------------|
| Deterministic IDs | Same input → identical trace & span ids | `test_deterministic_ids.py` |
| Parent Resolution | Precedence order enforced | `test_parent_resolution_*` |
| Binary Stripping | All large/base64 & binary objects redacted | `test_binary_stripping.py` |
| Truncation & Propagation | Size guard only active when >0; propagation always | `test_truncation_propagation.py` |
| Generation Detection | Only spans meeting heuristics classified | `test_generation_detection.py` |
| Pointer Decoding | Pointer-compressed arrays reconstructed | `test_pointer_decoding.py` |
| Timezone Enforcement | No naive datetimes; normalization to UTC | `test_timezone_awareness.py` |
| Trace ID Embedding | Embedded OTLP hex trace id matches expected format | `test_trace_id_embedding.py` |
| AI-only Filtering | Root retention, ancestor preservation, metadata flags | `test_ai_filtering.py` |

### Parent Resolution Precedence (Authoritative Table)

| Priority | Strategy | Description | Metadata Signals |
|----------|----------|-------------|------------------|
| 1 | Agent Hierarchy | Non-`main` connection with `ai_*` type makes agent parent | `n8n.agent.parent`, `n8n.agent.link_type` |
| 2 | Runtime Sequential (Exact Run) | previousNode + previousNodeRun points to exact span | `n8n.node.previous_node_run` |
| 3 | Runtime Sequential (Last Seen) | previousNode without run index → last span for node | `n8n.node.previous_node` |
| 4 | Static Reverse Graph Fallback | Reverse edge from workflow connections | `n8n.graph.inferred_parent=true` |
| 5 | Root Fallback | No parent resolved → root span | (none) |

### Media Token Format

Stable replacement string: `@@@langfuseMedia:type=<mime>|id=<mediaId>|source=base64_data_uri@@@`

Temporary pre-upload placeholder (never exported if token succeeds):
```json
{
    "_media_pending": true,
    "sha256": "<hex>",
    "bytes": <decoded_size_bytes>,
    "base64_len": <original_base64_length>,
    "slot": "<optional source slot or field path>"
}
```

### Media Create Payload

POST `/api/public/media` keys:

| Key | Source | Notes |
|-----|--------|-------|
| traceId | OTLP hex trace id | 32-hex with execution id suffix |
| observationId | OTLP span id (16-hex) | **REQUIRED** for observation-level previews |
| contentType | Sanitized mime or default | Omitted if unsupported |
| contentLength | Decoded byte length | Must equal uploaded bytes |
| sha256Hash | Base64 SHA256 digest | Derived from hex |
| field | Derived from placeholder path | First segment → `input`/`output`/`metadata` |

Presigned upload required headers: `Content-Type`, `x-amz-checksum-sha256`.
 Azure Blob variant additionally requires: `x-ms-blob-type: BlockBlob` (auto-added when URL contains `blob.core.windows.net`).

**Observation Linking Invariant:** Every successful media create from a span MUST include `observationId=<otel_span_id>` or UI preview pane will remain empty. Tests: `test_media_observation_link.py`.

### Media Upload Tests

| Test File | Scenario |
|-----------|----------|
| `test_media_api_tokens.py` | Enabled vs disabled parity, token substitution, metadata flags, oversize skip, deduplicated create. |
| `test_media_failure_and_oversize.py` | API failure path, error_codes, upload_failed, oversize skip. |

### Timezone Policy

* All datetimes exported to OTLP are timezone-aware UTC.
* Naive datetimes from DB/JSON localized to UTC (assuming naive == UTC).
* `datetime.utcnow()` forbidden; enforcement test scans `src/` and `tests/`.

### Testing Guidance

Add/update tests when:
1. Changing parent resolution logic.
2. Adjusting binary stripping heuristics.
3. Modifying generation detection or usage extraction.
4. Updating pointer-compressed decoding.
5. Changing truncation/propagation coupling.
6. Always open `FISH` terminal and use `pytest` or `pytest -q` for test runs!

Each new metadata key must appear in at least one assertion. Forbidden duplications (e.g., execution id outside root span) asserted negative.

---

## Shell & Command Execution Policy (Fish)

Development and CI use Fish shell. Follow these strict rules:

1. Always assume Fish syntax (NOT Bash). No heredocs, no `$()` inside single quotes, no Bash arrays.
2. Environment variable assignment: `set -x VAR value` (export) not `VAR=value command`.
3. Path modification: `set -x PATH /custom/bin $PATH`.
4. Python one-liners MUST use this pattern:
   ```fish
   python -c "import sys; print('example')"
   ```
   Rationale: Fish-safe quoting; avoids unintended interpolation; inner single quotes available for Python.
5. If snippet >~120 chars or needs multiline logic: write temporary script file.
6. Never `source .venv/bin/activate`; prefer explicit interpreter paths. Activation differs under Fish (`activate.fish`).
7. Sequential test commands: each on own line in fenced block with `fish` language tag.
8. Avoid trailing backslashes; prefer separate lines.
9. Module checking: only use standardized command pattern (maintain consistency).
10. No Bash `${VAR}` unless inside Python string; in Fish use `$VAR`.

Deviation = correctness issue; correct in same PR.

---

## Conventions & Best Practices

* **Idempotency:** Deterministic UUIDs for spans and traces.
* **Error Handling:** Use `tenacity` for retrying database connections.
* **Logging:** Structured logging for clear diagnostics.
* **Truncation:** Disabled by default; when enabled post-binary stripping, sets `n8n.truncated.*` metadata.
* **Binary Stripping:** Always on; do not remove without tests + docs.
* **Parent Resolution Order:** Fixed precedence (see table). Do not reorder without updating tests.
* **Input Propagation:** Uses cached parent output unless explicit `inputOverride`.
* **Generation Detection:** Conservative; update tests when heuristics change.
* **Metadata Keys:** Additions require justification + test coverage.
* **Hard Line Length Cap (E501 ≤100 chars):** All new & modified lines MUST be ≤100 characters. Wrap proactively. Only allow overage when absolutely unavoidable; append `# noqa: E501  # reason`.

### Line Length Wrapping Guidance

1. Use parentheses for multi-line expressions; never backslashes.
2. Break boolean conditions one clause per line inside parentheses.
3. Long format/log strings: split using implicit concatenation.
4. Function definitions: one parameter per line when near limit.
5. Dict/list literals: trailing commas + one key/value per line when wrapping.
6. Avoid nested f-strings; compute intermediate variables.
7. Refactor instead of `# noqa` whenever possible.
8. Before committing: visually scan or `ruff check` to ensure no >100 lines.

---

## Mermaid Diagram Authoring Guidelines (Strict)

ALL Mermaid diagrams MUST follow simplified style:

1. Use only `graph TD` (top-down) unless compelling reason; document rationale inline.
2. Node labels: ASCII only, short (1–3 words). No HTML tags, no Unicode arrows, no emojis, no parentheses in labels.
3. Multi-line labels forbidden. Extra context → `%%` comment beneath node.
4. Subgraph syntax: `subgraph IDENTIFIER [Readable Title]` — identifier alphanumeric/underscore only.
5. Edges: plain `-->` or dotted `-.->` only. No link text or styles unless absolutely required; justify if used.
6. Comments: only `%%` full-line comments. Remove HTML comment forms.
7. No class/style/link directives unless prior review + justification note.
8. Keep diagrams minimal: prefer one high-level pipeline diagram.
9. Validate in Mermaid Live Editor (default settings) before committing.
10. Richer diagram needed? Add second simplified fallback version; mark richer one optional.

Violation = breaking invariant; submit corrective patch.

---

## Contribution Guardrails (AI & Humans)

* No new dependencies without `pyproject.toml` update + rationale.
* Preserve deterministic UUIDv5 namespace + seed formats (ID immutability). Changing trace id embedding requires test + doc updates.
* Mapper stays pure (no network / DB writes). Media upload must reside in separate phase/module.
* Do NOT move execution id outside root span or duplicate it (breaks single-source invariant).
* Never introduce `datetime.utcnow()` (timezone invariant). All timestamps must be UTC-aware.
* Backwards compatibility: breaking changes allowed (early development) but must be documented in release notes. Never implement backwards compatibility; not required at this stage.
* Update `README.md` + `copilot-instructions.md` for new env vars / CLI flags / behavior changes in same PR.
* NOTICE & LICENSE untouched aside from annual year updates.
* New metadata keys: add tests asserting presence & absence where appropriate.

---

## Failure & Resilience Philosophy

* Root-only fallback: malformed/missing `runData` still produces trace with execution root span.
* DB transient errors: retried (exponential backoff); checkpoint only advances after successful export (prevents duplication).
* Exporter backpressure: soft-limit queue metrics induce flush + sleep preventing memory growth while preserving ordering.
* Purity: mapper must not introduce network I/O or writes (idempotent replay guarantee).
* Fatal stop conditions limited to configuration errors (missing credentials, invalid DSN).
* Planned: dead-letter queue for repeated mapping failures (retain error context without blocking pipeline).
* Logging: structured logging required for parse failures (`error`, `execution_id`, `phase`).

---

## Regression Checklist

Before merging changes:

1. Deterministic IDs unchanged.
2. Execution id only on root span.
3. UTC timezone invariants respected (`test_timezone_awareness` green).
4. Parent precedence unchanged or tests updated.
5. Binary placeholders unchanged (size/marker keys stable) unless tests updated.
6. Generation heuristics stable (token usage mapping preserved).
7. Pointer decoding passes existing scenarios.
8. New env vars documented & tested.
9. No mapper network calls added.
10. Metadata key additions/removals reflected in tests + README.
11. AI-only filtering logic unchanged (root span retained, exclusion counts correct) unless tests & docs updated.

---

## Development Plan (Planned Enhancements)

1. Media resilience (retry, circuit breaker, streaming uploads) & richer diagnostics.
2. Advanced filtering flags (status, workflow, time window, id bounds).
3. Parallel/async export pipeline (maintain deterministic ordering).
4. PII/sensitive field redaction (configurable; test enforced).
5. Dead-letter queue for persistent mapping failures.
6. Provider-specific model normalization improvements.
7. Metrics / Prometheus endpoint (optional instrumentation).
8. Adaptive backpressure metrics (dynamic sleep based on exporter latency).
9. Selective path-based truncation overrides.
10. Circuit breaker for media upload failures.

---

## Key Files & Project Structure

```
n8n-langfuse-shipper/
├── src/
│   ├── __main__.py
│   ├── config.py
│   ├── db.py
│   ├── mapper.py
│   ├── observation_mapper.py
│   ├── shipper.py
│   ├── checkpoint.py
│   ├── media_api.py
│   └── models/
│       ├── __init__.py
│       ├── n8n.py
│       └── langfuse.py
├── tests/
│   ├── test_mapper.py
│   ├── test_generation_heuristic.py
│   ├── test_media_api_tokens.py
│   └── ...
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Documentation Policy (MANDATORY)

**CRITICAL: All documentation MUST go in EXACTLY these files ONLY:**

1. **`.github/copilot-instructions.md`** - Technical implementation details, architecture, invariants, contracts, testing requirements
2. **`README.md`** - User-facing documentation, configuration, examples, troubleshooting
3. **`./infra/README.md`** - Azure resource deployment instructions using Bicep

**FORBIDDEN:**
* ❌ NEVER create separate markdown files like `FEATURE_NAME.md`, `IMPLEMENTATION_SUMMARY.md`, `CHANGES.md` UNLESS THE USER REQUESTS IT EXPLICITLY!
* ❌ NEVER create standalone documentation files in the project root or any subdirectory UNLESS THE USER REQUESTS IT EXPLICITLY!
* ❌ NEVER split documentation across multiple files

**REQUIRED Process for ANY code change:**
1. Implement the feature with tests
2. Update **copilot-instructions.md** with technical details (algorithms, invariants, test requirements)
3. Update **README.md** with user-facing information (configuration, usage, examples)
4. ALL changes MUST happen in the SAME PR

**Rationale:**
* Single source of truth prevents documentation drift
* Easy to find information (only two places to look)
* Forces documentation discipline
* Reduces maintenance burden

**Examples of what goes where:**

`copilot-instructions.md`:
- Implementation algorithms and data structures
- Parent resolution precedence logic
- Binary stripping heuristics
- Test coverage requirements
- Metadata key definitions
- Reserved keyword contracts

`README.md`:
- Environment variable tables
- CLI command examples
- Configuration examples
- Troubleshooting guides
- Feature descriptions for users
- Installation and quick start

**Violation = Breaking Invariant:** Creating additional documentation files is considered a critical error. If you create extra docs, you MUST consolidate them into the two canonical files and delete the extra file in the same PR.

---

**This document is the authoritative contract.** Any significant code change altering behavior here must include an update to this file in the same PR.
