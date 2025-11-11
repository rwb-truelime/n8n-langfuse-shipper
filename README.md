# n8n-langfuse-shipper

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Status](https://img.shields.io/badge/status-active--development-informational)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Type Checking](https://img.shields.io/badge/mypy-strict-blue)
![Lint](https://img.shields.io/badge/ruff-enabled-brightgreen)

This is a high-performance tool for shippering historical n8n workflow executions into [Langfuse](https://langfuse.com/) for observability. It reads execution data directly from your n8n PostgreSQL database, transforms it into Langfuse traces, and sends it to your Langfuse project.

It's designed for developers and teams who use n8n for AI-powered workflows and need to analyze, debug, or monitor their past executions in Langfuse.

---

## Key Features

- **High-Throughput shipper**: Efficiently processes and exports thousands of n8n executions.
- **Rich Trace Data**: Intelligently maps n8n concepts to Langfuse, including:
    - **Agent & Tool Hierarchy**: Correctly identifies parent-child relationships in LangChain nodes.
    - **Generation Spans**: Automatically detects LLM calls, extracting model names and token usage.
    - **Prompt Management Integration**: Links generation spans to Langfuse prompt versions, enabling prompt tracking and versioning.
    - **Error & Status**: Maps n8n node errors to Langfuse span statuses.
- **Idempotent & Resumable**: Uses deterministic IDs and a checkpoint file to prevent duplicate data and allow you to safely resume exports.
- **Media Uploads**: Can upload binary files (images, documents) from your workflows to Langfuse and link them to your traces.
- **AI-Only Filtering**: Option to export only AI-related spans and their direct ancestors, cutting down on noise.
- **Resilient**: Handles various n8n data formats and recovers from transient database errors.

---

## Quick Start

Get up and running in a few steps.

1.  **Install (Python 3.12+)**
    ```bash
    pip install -e .[dev]
    ```

2.  **Configure Environment**
    Set the following environment variables:

    **Fish shell:**
    ```fish
    # Your n8n database connection string
    set -x PG_DSN postgresql://user:pass@host:5432/n8n

    # Your Langfuse project credentials
    set -x LANGFUSE_PUBLIC_KEY lf_pk_...
    set -x LANGFUSE_SECRET_KEY lf_sk_...

    # The table prefix used by your n8n instance (often 'n8n_')
    set -x DB_TABLE_PREFIX n8n_

    # Langfuse host URL
    set -x LANGFUSE_HOST https://cloud.langfuse.com
    ```

    **Bash/Zsh:**
    ```bash
    # Your n8n database connection string
    export PG_DSN=postgresql://user:pass@host:5432/n8n

    # Your Langfuse project credentials
    export LANGFUSE_PUBLIC_KEY=lf_pk_...
    export LANGFUSE_SECRET_KEY=lf_sk_...

    # The table prefix used by your n8n instance (often 'n8n_')
    export DB_TABLE_PREFIX=n8n_

    # Langfuse host URL
    export LANGFUSE_HOST=https://cloud.langfuse.com
    ```

3.  **Run a Dry Run**
    Process the first 25 executions without sending any data to Langfuse. This is great for testing your configuration.

    ```bash
    n8n-shipper shipper --limit 25 --dry-run
    ```

4.  **Run a Real Export**
    To perform a real export, use the `--no-dry-run` flag.

    ```bash
    n8n-shipper shipper --limit 100 --no-dry-run
    ```

The shipper creates a `.shipper_checkpoint` file to remember the last exported execution. The next time you run the command, it will automatically resume from where it left off.

---

## How It Works

The shipper is an ETL (Extract, Transform, Load) pipeline that connects your n8n database to Langfuse.

```mermaid
graph TD
    subgraph "n8n-langfuse-shipper"
        A[1. Extract]
        B[2. Transform]
        C[3. Load]
    end

    PG[(PostgreSQL n8n DB)] -- "Reads execution history" --> A
    A -- "Execution records" --> B
    B -- "Langfuse traces & spans" --> C
    C -- "Sends OTLP data" --> LF[Langfuse API]

    subgraph "Transformation Details"
        B --> B1[Map Nodes to Spans]
        B --> B2[Detect Generations & Usage]
        B --> B3[Resolve Parent-Child Hierarchy]
        B --> B4[Handle Media/Binary Data]
    end
```

1.  **Extract**: It connects to your PostgreSQL database and streams n8n execution records in batches.
2.  **Transform**: For each execution, it creates a Langfuse trace. It analyzes the workflow structure (`workflowData`) and the runtime output (`runData`) to map each executed node to a Langfuse span. This is where it identifies generations, token usage, and agent/tool relationships.
3.  **Load**: It sends the fully formed traces to the Langfuse OTLP endpoint.

---

## Prompt Management Integration

The shipper automatically links LLM generation spans to their corresponding Langfuse prompt versions, enabling you to track and manage prompts in the Langfuse UI.

### How It Works

1. **Automatic Detection**: The shipper identifies nodes in your n8n workflows that fetch prompts from Langfuse:
   - Official Langfuse prompt nodes (e.g., `@n8n/n8n-nodes-langchain.lmPromptSelector`)
   - HTTP Request nodes calling Langfuse prompt API (`/api/public/v2/prompts/<name>`)
   - Any node with prompt-shaped output (containing `name`, `version`, `prompt`/`config`)

2. **Ancestor Chain Resolution**: For each LLM generation span, the shipper walks backward through the workflow execution chain to find the closest ancestor that fetched a prompt, then extracts the prompt name and version.

   **Multiple Prompt Disambiguation**: When multiple prompts are at the same distance from a generation span (e.g., two prompts feeding into an agent), the shipper uses **fingerprint matching** to select the correct prompt:
   - Extracts the first 300 characters of each candidate prompt text during detection
   - Computes a fingerprint (SHA256 hash) for each prompt
   - When resolving a generation, extracts the actual prompt text from the agent's input
   - Strips LangChain prefixes (`System: `) and computes the input fingerprint
   - Matches the input fingerprint against candidate fingerprints to select the correct prompt
   - Falls back to alphabetical ordering if no match is found

3. **Environment-Aware Version Resolution**:
   - **Production**: Uses the exact version number from the workflow execution
   - **Dev/Staging**: Automatically maps to the latest active version in your dev/staging
     Langfuse environment (prevents linkage failures when production versions don't exist)

4. **OTLP Attribute Emission**: The prompt name and resolved version number are attached to the generation span as OpenTelemetry attributes (`langfuse.observation.prompt.name` and `langfuse.observation.prompt.version`), enabling the Langfuse UI to display prompt metadata.

### Configuration

Set the `LANGFUSE_ENV` environment variable to control version resolution behavior:

**Fish shell:**
```fish
# Production environment (default) - no API queries, uses exact versions from executions
set -x LANGFUSE_ENV production

# Development environment - queries Langfuse API to resolve version labels
set -x LANGFUSE_ENV dev

# Staging environment - also queries Langfuse API
set -x LANGFUSE_ENV staging
```

**Bash/Zsh:**
```bash
# Production environment (default) - no API queries, uses exact versions from executions
export LANGFUSE_ENV=production

# Development environment - queries Langfuse API to resolve version labels
export LANGFUSE_ENV=dev

# Staging environment - also queries Langfuse API
export LANGFUSE_ENV=staging
```

**Important**:
- The environment value is **case-sensitive** and must be lowercase.
- In production, the shipper never queries the Langfuse API (for security and performance).
- In dev/staging, API queries timeout after 5 seconds by default (configurable via
  `PROMPT_VERSION_API_TIMEOUT`).
- If credentials are missing from environment, the shipper falls back to reading a `.env`
  file in the current directory (format: `KEY=VALUE`, one per line).

### Troubleshooting

**Prompt version too high / not found in dev:**
- Ensure `LANGFUSE_ENV=dev` (not production).
- Confirm credentials are set (check logs for "Created prompt version resolver").
- Verify `LANGFUSE_HOST` points to your dev environment.

### Debug Metadata

The shipper always attaches debug metadata to generation spans for troubleshooting:
- `n8n.prompt.resolution_method`: How the version was resolved (`env_latest`,
  `production_passthrough`, etc.)
- `n8n.prompt.confidence`: Confidence level (`high`, `medium`, `low`, `none`)
- `n8n.prompt.ancestor_distance`: Number of workflow nodes between the generation and
  prompt fetch
- `n8n.prompt.fetch_node_name`: Name of the node that fetched the prompt

---

## Configuration

The tool is configured via environment variables, which can be overridden by command-line arguments.

### Core Settings

| Environment Variable | CLI Argument | Default | Description |
|---|---|---|---|
| `PG_DSN` | (none) | - | Full PostgreSQL DSN. Overrides individual `DB_*` vars. |
| `DB_POSTGRESDB_HOST` | (none) | - | Database host. |
| `DB_POSTGRESDB_PORT` | (none) | `5432` | Database port. |
| `DB_POSTGRESDB_DATABASE`| (none) | - | Database name. |
| `DB_POSTGRESDB_USER` | (none) | `postgres` | Database user. |
| `DB_POSTGRESDB_PASSWORD`| (none) | "" | Database password. |
| `DB_POSTGRESDB_SCHEMA` | (none) | `public` | The database schema where n8n tables reside. |
| `DB_TABLE_PREFIX` | (none) | **(required)** | Table prefix for n8n tables (e.g., `n8n_`). Set to `""` for no prefix. |
| `LANGFUSE_HOST` | (none) | `https://cloud.langfuse.com` | The base URL for your Langfuse instance. |
| `LANGFUSE_PUBLIC_KEY` | (none) | **(required)** | Your Langfuse public key. Falls back to `.env` file. |
| `LANGFUSE_SECRET_KEY` | (none) | **(required)** | Your Langfuse secret key. Falls back to `.env` file. |

### Processing Controls

| Environment Variable | CLI Argument | Default | Description |
|---|---|---|---|
| `FETCH_BATCH_SIZE` | (none) | `100` | Number of executions to fetch from the database at once. |
| `CHECKPOINT_FILE` | `--checkpoint-file` | `.shipper_checkpoint` | Path to the file that stores the last processed execution ID. |
| `DRY_RUN` | `--dry-run / --no-dry-run` | `true` | If `true`, only mapping is performed (no export to Langfuse). Set to `false` or use `--no-dry-run` to actually export. |
| `DEBUG` | `--debug / --no-debug` | `false` | Enable special debug features for execution data parsing (e.g., pointer decoding). **This is NOT the logging level.** |
| `ATTEMPT_DECOMPRESS` | `--attempt-decompress / --no-attempt-decompress` | `false` | Attempt decompression of execution data payloads. |
| `DEBUG_DUMP_DIR` | `--debug-dump-dir` | (none) | Directory to dump raw execution data JSON files when `DEBUG=true`. |
| `TRUNCATE_FIELD_LEN` | `--truncate-len` | `0` | Maximum length for input/output fields. `0` disables truncation. Binary data is always stripped regardless of this setting. |
| `FILTER_AI_ONLY` | `--filter-ai-only / --no-filter-ai-only` | `false` | If `true`, exports only AI-related spans (LangChain nodes) and their ancestors. Root span always included. |
| `FILTER_WORKFLOW_IDS` | (none) | `""` | Comma-separated workflowId allow-list to restrict processing. Example: `abc123,def456`. Empty = no workflowId filtering. |
| `FILTER_AI_EXTRACTION_NODES` | (none) | `""` | Comma-separated node names or wildcard patterns for extracting node data to root metadata when `FILTER_AI_ONLY=true`. Example: `Tool*,Agent*`. Empty disables extraction. |
| `FILTER_AI_EXTRACTION_INCLUDE_KEYS` | (none) | `""` | Comma-separated wildcard patterns for keys to include in extracted data. Patterns match full flattened paths like `main.0.0.json.fieldname`. Example: `*url,*token*`. Empty includes all keys. |
| `FILTER_AI_EXTRACTION_EXCLUDE_KEYS` | (none) | `""` | Comma-separated wildcard patterns for keys to exclude from extracted data. Applied after include filter. Example: `*secret*,*password*`. Empty excludes nothing. |
| `FILTER_AI_EXTRACTION_MAX_VALUE_LEN` | (none) | `10000` | Maximum string length per extracted value. Prevents excessively large metadata payloads. |
| `REQUIRE_EXECUTION_METADATA` | `--require-execution-metadata / --no-require-execution-metadata` | `false` | If `true`, only process executions that have a matching row in `execution_metadata` table. **Critical for selective processing.** |
| `LOG_LEVEL` | (none) | `INFO` | **Logging verbosity level.** Values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Use `LOG_LEVEL=DEBUG` for detailed logs. |
| `LANGFUSE_ENV` | (none) | `production` | Environment for prompt version resolution. Values: `production` (no API queries), `dev`, or `staging` (API queries enabled). Must be lowercase. |
| `PROMPT_VERSION_API_TIMEOUT` | (none) | `5` | Timeout (seconds) for Langfuse prompt API queries in dev/staging environments. |

### Export Backpressure & Reliability

| Environment Variable | CLI Argument | Default | Description |
|---|---|---|---|
| `FLUSH_EVERY_N_TRACES` | (none) | `1` | Force flush the exporter after every N traces. |
| `OTEL_MAX_QUEUE_SIZE` | (none) | `10000` | Maximum queue size for spans in the batch processor. |
| `OTEL_MAX_EXPORT_BATCH_SIZE` | (none) | `512` | Maximum number of spans per export batch. |
| `OTEL_SCHEDULED_DELAY_MILLIS` | (none) | `200` | Delay in milliseconds before a batch is exported. |
| `EXPORT_QUEUE_SOFT_LIMIT` | `--export-queue-soft-limit` | `5000` | Soft limit for queued spans before introducing backpressure sleep. |
| `EXPORT_SLEEP_MS` | `--export-sleep-ms` | `75` | Sleep duration in milliseconds when soft queue limit is exceeded. |

### Media Uploads

| Environment Variable | CLI Argument | Default | Description |
|---|---|---|---|
| `ENABLE_MEDIA_UPLOAD` | (none) | `false` | Set to `true` to enable uploading binary files to Langfuse. |
| `MEDIA_MAX_BYTES` | (none) | `25_000_000` | Maximum size (in bytes) for a single file upload. Files larger than this will be omitted. |
| `EXTENDED_MEDIA_SCAN_MAX_ASSETS` | (none) | `250` | Maximum number of binary assets to discover per node run from non-standard locations (e.g., data URLs). |

---

## Node Extraction Feature

When `FILTER_AI_ONLY=true`, you can optionally extract input/output data from filtered-out (non-AI) nodes into the root span's metadata for debugging. This preserves visibility into tool calls and other nodes that would otherwise be excluded.

### Configuration

**`FILTER_AI_EXTRACTION_NODES`** - Specify which nodes to extract:
- Comma-separated list of node names: `WebScraperTool,DatabaseQuery`
- Wildcard patterns: `Tool*,*API*` (matches `ToolA`, `ToolB`, `MyAPI`, etc.)
- Empty (default): No extraction

**`FILTER_AI_EXTRACTION_INCLUDE_KEYS`** - Filter which data keys to include:
- Patterns match **full flattened paths** like `main.0.0.json.url`
- Example: `*url,*token*` includes any path containing "url" or "token"
- Empty (default): Include all keys

**`FILTER_AI_EXTRACTION_EXCLUDE_KEYS`** - Filter which data keys to exclude:
- Applied AFTER include filter
- Example: `*secret*,*password*` excludes sensitive fields
- Patterns match full flattened paths like `main.0.0.json.secret_connection_string`
- Empty (default): Exclude nothing

**`FILTER_AI_EXTRACTION_MAX_VALUE_LEN`** - Limit extracted value size:
- Maximum characters per extracted string value
- Default: `10000` (10KB per field)
- Prevents metadata bloat

### Pattern Matching Rules

**Critical:** Patterns match the **full flattened key path**, not just the field name.

n8n stores node output as nested structures like:
```json
{
  "main": [
    [
      {
        "json": {
          "url": "https://example.com",
          "secret_token": "abc123"
        }
      }
    ]
  ]
}
```

This flattens to:
- `main.0.0.json.url`
- `main.0.0.json.secret_token`

To match these, use wildcards:
- ✅ `*url` matches `main.0.0.json.url`
- ✅ `*secret*` matches `main.0.0.json.secret_token`
- ❌ `url` does NOT match (no wildcard for path prefix)
- ❌ `secret_token` does NOT match (no wildcard for path prefix)

### Example Configurations

**Extract all data from tool nodes:**

*Fish shell:*
```fish
set -x FILTER_AI_ONLY true
set -x FILTER_AI_EXTRACTION_NODES "WebScraperTool,DatabaseQueryTool"
n8n-shipper shipper --no-dry-run
```

*Bash/Zsh:*
```bash
export FILTER_AI_ONLY=true
export FILTER_AI_EXTRACTION_NODES="WebScraperTool,DatabaseQueryTool"
n8n-shipper shipper --no-dry-run
```

**Extract only URLs and response bodies, exclude secrets:**

*Fish shell:*
```fish
set -x FILTER_AI_ONLY true
set -x FILTER_AI_EXTRACTION_NODES "*Tool*"
set -x FILTER_AI_EXTRACTION_INCLUDE_KEYS "*url,*response*"
set -x FILTER_AI_EXTRACTION_EXCLUDE_KEYS "*secret*,*password*,*key*"
n8n-shipper shipper --no-dry-run
```

*Bash/Zsh:*
```bash
export FILTER_AI_ONLY=true
export FILTER_AI_EXTRACTION_NODES="*Tool*"
export FILTER_AI_EXTRACTION_INCLUDE_KEYS="*url,*response*"
export FILTER_AI_EXTRACTION_EXCLUDE_KEYS="*secret*,*password*,*key*"
n8n-shipper shipper --no-dry-run
```

**Extract specific fields with size limit:**

*Fish shell:*
```fish
set -x FILTER_AI_ONLY true
set -x FILTER_AI_EXTRACTION_NODES "Agent*"
set -x FILTER_AI_EXTRACTION_INCLUDE_KEYS "*input*,*output*"
set -x FILTER_AI_EXTRACTION_MAX_VALUE_LEN 5000
n8n-shipper shipper --no-dry-run
```

*Bash/Zsh:*
```bash
export FILTER_AI_ONLY=true
export FILTER_AI_EXTRACTION_NODES="Agent*"
export FILTER_AI_EXTRACTION_INCLUDE_KEYS="*input*,*output*"
export FILTER_AI_EXTRACTION_MAX_VALUE_LEN=5000
n8n-shipper shipper --no-dry-run
```

### Metadata Structure

Extracted data appears in root span metadata under `n8n.extracted_nodes`:

```json
{
  "n8n.extracted_nodes": {
    "_meta": {
      "extracted_count": 2,
      "nodes_requested": 2,
      "extraction_config": {
        "include_keys": ["*url", "*token*"],
        "exclude_keys": ["*secret*"]
      }
    },
    "WebScraperTool": {
      "runs": [
        {
          "run_index": 0,
          "execution_status": "success",
          "input": null,
          "output": {
            "main": [
              [
                {
                  "json": {
                    "url": "https://example.com",
                    "token_count": 150
                  }
                }
              ]
            ]
          },
          "_truncated": false
        }
      ]
    }
  }
}
```

### Troubleshooting

**Problem:** Extraction returns empty or missing data

**Solutions:**
1. **Check node names** - Must exactly match node names in workflow, or use wildcards (`Tool*`)
2. **Verify patterns** - Use wildcards for paths: `*fieldname` not `fieldname`
3. **Check filter order** - Include filter applied first, then exclude filter
4. **Inspect flattened paths** - Enable debug logging to see actual flattened key names

**Problem:** Too much data in metadata

**Solutions:**
1. Use `FILTER_AI_EXTRACTION_MAX_VALUE_LEN` to limit value sizes
2. Use `FILTER_AI_EXTRACTION_INCLUDE_KEYS` to whitelist only needed fields
3. Use `FILTER_AI_EXTRACTION_EXCLUDE_KEYS` to blacklist verbose fields like `*html*`

**Problem:** Sensitive data leaked

**Solutions:**
1. Always set `FILTER_AI_EXTRACTION_EXCLUDE_KEYS` with patterns like `*secret*,*password*,*token*,*key*`
2. Test patterns with `--dry-run` first
3. Binary data is always automatically stripped before extraction

---

## Command-Line Usage

The main command is `shipper`.

```bash
# Show all available commands and options
n8n-shipper --help
```

### Common Examples

**Start a new export, processing up to 500 executions.**
```bash
n8n-shipper shipper --limit 500 --no-dry-run
```

**Resume an export, starting after a specific execution ID.**
This overrides the checkpoint file.
```bash
n8n-shipper shipper --start-after-id 42000 --no-dry-run
```

**Export only executions that have metadata rows (selective processing).**
```bash
n8n-shipper shipper --require-execution-metadata --no-dry-run
```

**Export only AI-related spans (LangChain nodes).**
```bash
n8n-shipper shipper --filter-ai-only --no-dry-run
```

**Combine filters: AI-only + metadata requirement.**
```bash
n8n-shipper shipper --filter-ai-only --require-execution-metadata --no-dry-run
```

**Enable verbose logging (shows detailed processing information).**
```bash
LOG_LEVEL=DEBUG n8n-shipper shipper --limit 10 --dry-run
```

**Enable debug mode (dumps raw execution data to files).**
```bash
n8n-shipper shipper --limit 10 --debug --debug-dump-dir=./debug_dumps --dry-run
```

**Dry run (mapping only, no export) - default behavior.**
```bash
n8n-shipper shipper --limit 10
# Same as: n8n-shipper shipper --limit 10 --dry-run
```

---

## Development and Testing

Contributions are welcome. Please follow these steps to set up your development environment.

1.  **Install with Dev Dependencies**
    ```bash
    pip install -e .[dev]
    ```

2.  **Run Linters and Type Checkers**
    We use `ruff` for linting and `mypy` for type checking.
    ```bash
    ruff check .
    mypy src
    ```

3.  **Run Tests**
    The test suite covers the core mapping logic, data parsing, and integration points.
    ```fish
    # Run all tests
    pytest

    # Run tests in a specific file
    pytest tests/test_mapper.py
    ```

---

## License

Apache License 2.0. See `LICENSE` for the full text and `NOTICE` for attribution.
