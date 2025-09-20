# AI Coding Agent Instructions for n8n-langfuse-shipper

## Purpose
This project is a Python-based microservice to perform a high-throughput backfill of historical n8n execution data from a PostgreSQL database to Langfuse. The service will map n8n's execution model to the Langfuse data model and transmit the data via the OpenTelemetry (OTLP) endpoint. The focus is on correctness, performance, and robustness for large-scale data migration.

## Big Picture Architecture
The service operates as a standalone ETL (Extract, Transform, Load) process, designed for containerized, cron-based execution.

```mermaid
graph TD
    A[PostgreSQL Database] -->|1. Read Executions| B{n8n-langfuse-shipper}
    B -->|2. Map to OTel Traces| C[Langfuse OTLP Endpoint]

    subgraph Python Service (Docker)
        B
    end

```

1.  **Extract:** The service connects to the n8n PostgreSQL database and streams execution records (`n8n_execution_entity` joined with `n8n_execution_data`).
2.  **Transform:** Each execution record is transformed into a single Langfuse trace. The nodes within the execution are mapped to nested OpenTelemetry spans. This includes mapping specific AI node runs to Langfuse `Generation` objects via semantic attributes and handling multimodal (binary) data.
3.  **Load:** The transformed trace and its spans are exported to the Langfuse OTLP endpoint using the OpenTelemetry SDK.

## Core Data Models (Pydantic)
All internal data structures must be defined using Pydantic models for type safety and validation.

### 1. Raw N8N Data Models
These models represent the JSON data retrieved from the `n8n_execution_data` and `n8n_execution_entity` tables.

```python
# src/models/n8n.py
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
    id: str
    name: str
    nodes: List[WorkflowNode] = Field(default_factory=list)
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

### 2. Langfuse Target Models
These models represent the logical structure before creating OTel objects.

```python
# src/models/langfuse.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LangfuseUsage(BaseModel):
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    totalTokens: Optional[int] = None


class LangfuseGeneration(BaseModel):
    span_id: str
    model: Optional[str] = None
    usage: Optional[LangfuseUsage] = None
    input: Optional[Any] = None
    output: Optional[Any] = None


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
    generations: List[LangfuseGeneration] = Field(default_factory=list)
```

## Data Parsing & Resilience
The `data` column in `n8n_execution_data` can have multiple formats. The application must robustly parse them.
- **Standard Format:** A JSON object containing an `executionData` key.
- **Pointer-Compressed Format:** A top-level JSON array where objects reference other array elements by their index. Implement a resolver (`_decode_compact_pointer_execution`) to reconstruct the `runData` from this format.
- **Alternative Paths:** The `runData` object might be located at different nested paths. The parser (`_build_execution_data`) must probe multiple candidate paths to find it.
- **Empty/Invalid Data:** If `runData` cannot be found, the execution should still be processed, resulting in a trace with only a root span.

## Mapping Logic
The core transformation logic resides in the `mapper` module. It must use a hybrid approach, combining runtime data with the static workflow graph.

### The Agent/Tool Hierarchy
A key pattern in n8n, especially with LangChain nodes, is an "Agent" node that uses other nodes as "Tools", "LLMs", or "Memory". The `workflowData.connections` object reveals this.
- A connection with `type: "main"` represents a sequential step.
- A connection with `type: "ai_tool"`, `type: "ai_languageModel"`, or `type: "ai_memory"` from a component node (e.g., `Calculator`) *to* an agent node (e.g., `HAL9000`) signifies a **hierarchical relationship**.
- In this case, the agent's span is the **parent** of the component's span.

### Trace Mapping
- An `N8nExecutionRecord` maps to a single `LangfuseTrace`.
- `LangfuseTrace.id` must be deterministic: `f"n8n-exec-{record.id}"`.
- A root `LangfuseSpan` is created to represent the entire execution. All top-level node spans are children of this root span.

### Span Mapping
- Each `NodeRun` maps to a `LangfuseSpan`.
- `LangfuseSpan.id` must be deterministic: a UUIDv5 hash of `f"{trace_id}:{node_name}:{run_index}"`.
- **Parent-Child Linking:** Implement a multi-tier logic for parent resolution:
    1.  **Hierarchical (Agent/Tool):** First, check `workflowData.connections`. If a node run corresponds to a node that is connected to an Agent via a non-`main` connection type, its parent is the most recent span of that Agent.
    2.  **Sequential (Runtime):** If not part of a hierarchy, use `run.source[0].previousNode` to find the immediate predecessor. Link to the exact run index (`previousNodeRun`) if available, otherwise link to the last seen span for that predecessor node.
    3.  **Sequential (Static Fallback):** If runtime `source` is missing, use a reverse-edge map built from `workflowData.connections` to infer the most likely parent from the static graph.
    4.  **Root Fallback:** If no parent can be determined, link to the root execution span.
- **I/O Propagation:** If a `NodeRun` lacks `inputOverride`, its logical input should be inferred from the `data` field of its resolved parent node.

### Observation Type Mapping
- Use the `observation_mapper.py` module to classify each node based on its type and category. This determines the `LangfuseSpan.observation_type`.

### Generation Mapping
- The mapper identifies LLM node runs via heuristics (`_detect_generation`) and extracts token usage (`_extract_usage`).
- The `LangfuseSpan` is populated with `model` and `token_usage` data. The shipper will use these to set OTel attributes (`gen_ai.usage.*`, `model`), allowing Langfuse to classify the span as a `generation`.

### Multimodality Mapping
- This is a future requirement. The logic will be:
    1.  **Detect:** Identify binary data fields (e.g., base64 strings in I/O).
    2.  **Upload:** Use the Langfuse REST API (`POST /api/public/media`) to upload the binary content.
    3.  **Replace:** In the span's I/O field, replace the binary data with the Langfuse Media Token string (`@@@langfuseMedia:...`).

## OpenTelemetry Shipper
The `shipper.py` module converts the internal `LangfuseTrace` model into OTel spans and exports them.
- **Initialization:** The OTLP exporter is configured once with the Langfuse endpoint and Basic Auth credentials.
- **Span Creation:** For each `LangfuseSpan`, create an OTel span with the exact `start_time` and `end_time`.
- **Attribute Mapping:** The shipper sets OTel attributes based on the `LangfuseSpan` model:
    - `langfuse.observation.type` <- `observation_type`
    - `model` & `langfuse.observation.model.name` <- `model`
    - `gen_ai.usage.*` <- `token_usage` fields
    - `langfuse.observation.metadata.*` <- `metadata` dictionary
    - `langfuse.observation.level` and `status_message` for errors.
- **Trace Attributes:** Set trace-level attributes (`langfuse.trace.name`, `langfuse.trace.metadata.*`) on the root span.

## Application Flow & Control
- **Main Loop:** A CLI script (`__main__.py`) that loads a checkpoint, streams execution batches from PostgreSQL, maps each record to a `LangfuseTrace`, passes it to the shipper, and updates the checkpoint.
- **Checkpointing:** Use the `checkpoint.py` module to atomically store the last successfully processed `executionId` in a file.
- **CLI Interface:** Use `Typer`. The `backfill` command must support:
    - `--start-after-id`, `--limit`, `--dry-run`, `--debug`, `--debug-dump-dir`.

## Key Environment Variables
- `PG_DSN`: Full PostgreSQL connection string (takes precedence).
- `DB_POSTGRESDB_HOST`, `DB_POSTGRESDB_PORT`, `DB_POSTGRESDB_DATABASE`, `DB_POSTGRESDB_USER`, `DB_POSTGRESDB_PASSWORD`: Component-based DB connection variables.
- `DB_POSTGRESDB_SCHEMA`: Database schema (default: `public`).
- `DB_TABLE_PREFIX`: n8n table prefix (default: `n8n_`).
- `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
- `LOG_LEVEL`, `FETCH_BATCH_SIZE`, `TRUNCATE_FIELD_LEN`.

## Development Plan (Next Iterations)
1.  **Agent Hierarchy Mapping:** Implement the hierarchical parent-child linking logic using the `connections` object. This is the highest priority.
2.  **Media Handling:** Implement the full multimodality mapping workflow (detect, upload, replace).
3.  **Error Handling:** Add robust retry mechanisms for media uploads and a dead-letter queue for executions that fail to map.
4.  **Performance Tuning:** Investigate parallel processing of batches and asynchronous span exporting.
5.  **Advanced Filtering:** Add CLI flags to filter executions by status, time window, or workflow ID.

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
│   └── models/
│       ├── __init__.py
│       ├── n8n.py
│       └── langfuse.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Conventions & Best Practices
- **Idempotency:** Use deterministic UUIDs for spans and traces.
- **Error Handling:** Use `tenacity` for retrying database connections.
- **Logging:** Use structured logging for clear diagnostics.
- **Data Truncation:** Aggressively truncate large I/O fields and indicate this with a metadata flag.