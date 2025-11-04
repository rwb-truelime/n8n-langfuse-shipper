# n8n-langfuse-shipper

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Status](https://img.shields.io/badge/status-active--development-informational)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Type Checking](https://img.shields.io/badge/mypy-strict-blue)
![Lint](https://img.shields.io/badge/ruff-enabled-brightgreen)

This is a high-performance tool for backfilling historical n8n workflow executions into [Langfuse](https://langfuse.com/) for observability. It reads execution data directly from your n8n PostgreSQL database, transforms it into Langfuse traces, and sends it to your Langfuse project.

It's designed for developers and teams who use n8n for AI-powered workflows and need to analyze, debug, or monitor their past executions in Langfuse.

---

## Key Features

- **High-Throughput Backfill**: Efficiently processes and exports thousands of n8n executions.
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
    Set the following environment variables. The example below uses the [Fish shell](https://fishshell.com/).

    ```fish
    # Your n8n database connection string
    set -x PG_DSN postgresql://user:pass@host:5432/n8n

    # Your Langfuse project credentials
    set -x LANGFUSE_PUBLIC_KEY lf_pk_...
    set -x LANGFUSE_SECRET_KEY lf_sk_...

    # The table prefix used by your n8n instance (often 'n8n_')
    set -x DB_TABLE_PREFIX n8n_

    # Optional: Set if you are self-hosting Langfuse
    set -x LANGFUSE_HOST http://your-langfuse-host:3000
    ```

3.  **Run a Dry Run**
    Process the first 25 executions without sending any data to Langfuse. This is great for testing your configuration.

    ```bash
    n8n-shipper backfill --limit 25 --dry-run
    ```

4.  **Run a Real Export**
    To perform a real export, use the `--no-dry-run` flag.

    ```bash
    n8n-shipper backfill --limit 100 --no-dry-run
    ```

The shipper creates a `.backfill_checkpoint` file to remember the last exported execution. The next time you run the command, it will automatically resume from where it left off.

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
   - **Production**: Uses the exact version number from the workflow execution (no API queries)
   - **Dev/Staging**: Queries the Langfuse API to resolve version labels (e.g., "latest") to actual version numbers

4. **OTLP Attribute Emission**: The prompt name and resolved version number are attached to the generation span as OpenTelemetry attributes (`langfuse.observation.prompt.name` and `langfuse.observation.prompt.version`), enabling the Langfuse UI to display prompt metadata.

### Configuration

Set the `LANGFUSE_ENV` environment variable to control version resolution behavior:

```fish
# Production environment (default) - no API queries, uses exact versions from executions
set -x LANGFUSE_ENV production

# Development environment - queries Langfuse API to resolve version labels
set -x LANGFUSE_ENV dev

# Staging environment - also queries Langfuse API
set -x LANGFUSE_ENV staging
```

**Important**:
- The environment value is **case-sensitive** and must be lowercase.
- In production, the shipper never queries the Langfuse API (for security and performance).
- In dev/staging, API queries timeout after 5 seconds by default (configurable via `PROMPT_VERSION_API_TIMEOUT`).

### Debug Metadata

The shipper always attaches debug metadata to generation spans for troubleshooting:
- `n8n.prompt.resolution_method`: How the version was resolved (`exact_match`, `fallback_latest`, `production_passthrough`, etc.)
- `n8n.prompt.confidence`: Confidence level (`high`, `medium`, `low`, `none`)
- `n8n.prompt.ancestor_distance`: Number of workflow nodes between the generation and prompt fetch
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
| `LANGFUSE_PUBLIC_KEY` | (none) | **(required)** | Your Langfuse public key. |
| `LANGFUSE_SECRET_KEY` | (none) | **(required)** | Your Langfuse secret key. |

### Processing Controls

| Environment Variable | CLI Argument | Default | Description |
|---|---|---|---|
| `FETCH_BATCH_SIZE` | (none) | `100` | Number of executions to fetch from the database at once. |
| `CHECKPOINT_FILE` | `--checkpoint-file` | `.backfill_checkpoint` | Path to the file that stores the last processed execution ID. |
| `TRUNCATE_FIELD_LEN` | `--truncate-len` | `0` | Maximum length for input/output fields. `0` disables truncation. Binary data is always stripped regardless of this setting. |
| `FILTER_AI_ONLY` | `--filter-ai-only` | `false` | If `true`, exports only AI-related spans and their ancestors. |
| `LOG_LEVEL` | (none) | `INFO` | Set the logging level (e.g., `DEBUG`, `INFO`, `WARNING`). |
| `LANGFUSE_ENV` | (none) | `production` | Environment for prompt version resolution. Values: `production` (no API queries), `dev`, or `staging` (API queries enabled). Must be lowercase. |
| `PROMPT_VERSION_API_TIMEOUT` | (none) | `5` | Timeout (seconds) for Langfuse prompt API queries in dev/staging environments. |

### Media Uploads

| Environment Variable | CLI Argument | Default | Description |
|---|---|---|---|
| `ENABLE_MEDIA_UPLOAD` | (none) | `false` | Set to `true` to enable uploading binary files to Langfuse. |
| `MEDIA_MAX_BYTES` | (none) | `25_000_000` | Maximum size (in bytes) for a single file upload. Files larger than this will be omitted. |
| `EXTENDED_MEDIA_SCAN_MAX_ASSETS` | (none) | `250` | Maximum number of binary assets to discover per node run from non-standard locations (e.g., data URLs). |

---

## Command-Line Usage

The main command is `backfill`.

```bash
# Show all available commands and options
n8n-shipper --help
```

### Common Examples

**Start a new export, processing up to 500 executions.**
```bash
n8n-shipper backfill --limit 500 --no-dry-run
```

**Resume an export, starting after a specific execution ID.**
This overrides the checkpoint file.
```bash
n8n-shipper backfill --start-after-id 42000 --no-dry-run
```

**Export only AI-related spans.**
```bash
n8n-shipper backfill --filter-ai-only --no-dry-run
```

**Run with verbose logging for debugging.**
```bash
LOG_LEVEL=DEBUG n8n-shipper backfill --limit 10 --dry-run
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
