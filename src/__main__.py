"""Main CLI entry point for the n8n-langfuse-shipper.

This module provides a command-line interface using Typer to run the backfill
process. It orchestrates the entire ETL pipeline:
1.  Loading configuration and checkpoints.
2.  Streaming execution records from the database (`db.py`).
3.  Parsing and validating the raw data, including handling complex formats like
    pointer-compressed executions.
4.  Mapping the records to Langfuse traces (`mapper.py`).
5.  Exporting the traces via OTLP (`shipper.py`).
6.  Storing the new checkpoint upon successful processing.
"""
from __future__ import annotations

import typer
from typing import Optional, Any, List, Dict
from datetime import datetime
import logging
import asyncio
from pydantic import ValidationError
import os

from .config import get_settings
from .db import ExecutionSource
from .models.n8n import N8nExecutionRecord, WorkflowData, ExecutionData, ExecutionDataDetails, ResultData
from .mapper import map_execution_to_langfuse, map_execution_with_assets
from .shipper import export_trace, shutdown_exporter
from .checkpoint import load_checkpoint, store_checkpoint
from .media_api import patch_and_upload_media

app = typer.Typer(help="n8n to Langfuse backfill shipper CLI")


def _build_execution_data(
    raw_data: Optional[dict[str, Any] | str | list[Any]],
    workflow_data_raw: Optional[dict[str, Any]] = None,
    *,
    debug: bool = False,
    attempt_decompress: bool = False,
    execution_id: Optional[int] = None,
) -> ExecutionData:
    """Robustly parse the `data` column from an n8n execution record.

    The `data` column in `n8n_execution_data` can have several formats. This
    function attempts to find and parse the `runData` object, which contains the
    critical information about each node's execution.

    It employs a resilient, multi-step strategy:
    1.  If the data is a JSON string, it's parsed into a Python object.
    2.  If the data is a list, it's assumed to be the "pointer-compressed"
        format and is decoded by `_decode_compact_pointer_execution`.
    3.  If it's a dictionary, it first attempts a direct Pydantic validation.
    4.  If that fails, it probes a series of common alternative paths where
        `runData` might be nested.
    5.  As a last resort, it checks the raw `workflowData` for `runData`.
    6.  If all attempts fail, it returns an empty `ExecutionData` object,
        ensuring that a root trace is still created for the execution.

    Args:
        raw_data: The raw content of the `data` column.
        workflow_data_raw: The raw `workflowData` object, used as a fallback.
        debug: If True, enables verbose logging of parsing attempts.
        attempt_decompress: Flag to enable future decompression logic.
        execution_id: The ID of the execution, for logging purposes.

    Returns:
        A parsed `ExecutionData` model, which may be empty if `runData` could
        not be found.
    """
    logger = logging.getLogger(__name__)
    empty = ExecutionData(executionData=ExecutionDataDetails(resultData=ResultData(runData={})))

    # Accept JSON string from DB driver if not auto-decoded.
    if isinstance(raw_data, str):
        try:
            import json
            raw_data = json.loads(raw_data)
        except Exception:
            logger.debug("Failed to json.loads execution data string; returning empty runData")
            return empty

    # Optionally attempt decompression if payload looks like base64+gzip and flag enabled.
    if attempt_decompress and isinstance(raw_data, (bytes, str)):
        # Not implemented yet; placeholder for future extension.
        logger.debug("attempt_decompress flag set but decompression logic not implemented; skipping")

    # Special case: some n8n instances store a pointer-compressed array (list) root instead of dict.
    if isinstance(raw_data, list):
        decoded = _decode_compact_pointer_execution(raw_data, debug=debug, execution_id=execution_id)
        if decoded is not None:
            return decoded

    if not raw_data or not isinstance(raw_data, dict):
        # Fallback: attempt to derive runData from workflowData raw if provided (edge cases / custom storage)
        if workflow_data_raw and isinstance(workflow_data_raw, dict):
            maybe_rd = workflow_data_raw.get("runData") or workflow_data_raw.get("resultData", {}).get("runData")
            if isinstance(maybe_rd, dict) and maybe_rd:
                logger.warning("Using workflowData payload as source for runData (data column empty)")
                return ExecutionData(
                    executionData=ExecutionDataDetails(resultData=ResultData(runData=maybe_rd))
                )
        return empty

    # Helper to materialize ExecutionData from a run_data dict
    def _from_run_data(rd: dict[str, Any]) -> ExecutionData:
        return ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=rd)
            )
        )

    # Attempt full pydantic parse first if key present
    if "executionData" in raw_data:
        try:
            parsed = ExecutionData(**raw_data)
            if parsed.executionData.resultData.runData:
                if debug:
                    logger.info(
                        "Execution %s: Parsed runData via standard path with %d node keys",
                        execution_id,
                        len(parsed.executionData.resultData.runData),
                    )
                return parsed
            elif debug:
                logger.info("Execution %s: executionData present but runData empty", execution_id)
        except ValidationError as ve:
            logger.debug("ExecutionData validation failed: %s", ve)

    # Probe multiple candidate paths for runData
    candidates: list[Any] = []
    try:
        candidates.append(raw_data.get("executionData", {}).get("resultData", {}).get("runData"))
    except Exception:
        pass
    try:
        candidates.append(raw_data.get("resultData", {}).get("runData"))
    except Exception:
        pass
    candidates.append(raw_data.get("runData"))
    # Some n8n versions embed an "data" key with nested executionData again.
    try:
        nested_data = raw_data.get("data")
        if isinstance(nested_data, dict):
            candidates.append(nested_data.get("executionData", {}).get("resultData", {}).get("runData"))
            candidates.append(nested_data.get("resultData", {}).get("runData"))
    except Exception:
        pass

    for cand in candidates:
        if isinstance(cand, dict) and cand:
            if debug:
                logger.info(
                    "Execution %s: Recovered runData via alternative path with %d node keys",
                    execution_id,
                    len(cand),
                )
            return _from_run_data(cand)

    # Last chance: workflowData fallback (non-standard)
    if workflow_data_raw and isinstance(workflow_data_raw, dict):
        maybe_rd = workflow_data_raw.get("runData") or workflow_data_raw.get("resultData", {}).get("runData")
        if isinstance(maybe_rd, dict) and maybe_rd:
            logger.warning("Recovered runData from workflowData (non-standard storage)")
            return _from_run_data(maybe_rd)

    if debug:
        logger.warning(
            "Execution %s: runData empty (data.keys=%s workflowData.keys=%s)",
            execution_id,
            list(raw_data.keys())[:20],
            list(workflow_data_raw.keys())[:20] if isinstance(workflow_data_raw, dict) else None,
        )
    return empty


def _decode_compact_pointer_execution(
    pool: list[Any],
    debug: bool = False,
    execution_id: Optional[int] = None,
) -> Optional[ExecutionData]:
    """Decode the "pointer-compressed" execution format into `ExecutionData`.

    Some n8n versions store execution data in a compact format where the top-level
    object is a list (the "pool"). Objects within this list reference each other
    using stringified integer indices (e.g., a key with value "4" points to
    `pool[4]`).

    This function recursively resolves these pointers to reconstruct the full
    `runData` structure. It includes memoization and cycle detection to handle
    complex or malformed data safely.

    Args:
        pool: The top-level list from the `data` column.
        debug: If True, enables verbose logging of the decoding process.
        execution_id: The ID of the execution, for logging purposes.

    Returns:
        A parsed `ExecutionData` model if decoding is successful, otherwise None.
    """
    logger = logging.getLogger(__name__)
    if not pool or not isinstance(pool, list) or not pool:
        return None
    if not isinstance(pool[0], dict):
        return None

    # Generic pointer resolver with memoization & cycle guard
    memo: dict[int, Any] = {}
    resolving: set[int] = set()

    def resolve_index(i: int) -> Any:
        if i in memo:
            return memo[i]
        if i < 0 or i >= len(pool):
            return None
        if i in resolving:  # cycle
            return None
        resolving.add(i)
        obj = pool[i]
        res = _resolve_value(obj)
        resolving.remove(i)
        memo[i] = res
        return res

    def _resolve_value(v: Any) -> Any:
        if isinstance(v, str) and v.isdigit():
            return resolve_index(int(v))
        if isinstance(v, list):
            return [_resolve_value(x) for x in v]
        if isinstance(v, dict):
            return {k: _resolve_value(val) for k, val in v.items()}
        return v

    root = pool[0]
    # Locate resultData pointer (could be under 'resultData' key or nested variant)
    result_ptr = root.get("resultData")
    result_obj = _resolve_value(result_ptr) if result_ptr is not None else None
    if not isinstance(result_obj, dict):
        return None
    run_data_ptr = result_obj.get("runData")
    run_data_obj = _resolve_value(run_data_ptr) if run_data_ptr is not None else None
    if not isinstance(run_data_obj, dict):
        return None

    # Collect NodeRun compatible structures
    from .models.n8n import NodeRun, NodeRunSource, ExecutionData, ExecutionDataDetails, ResultData
    node_run_map: dict[str, list[NodeRun]] = {}

    def collect_runs(val: Any) -> List[dict[str, Any]]:
        out: List[dict[str, Any]] = []
        if isinstance(val, dict):
            if "startTime" in val and "executionStatus" in val:
                out.append(val)
            else:
                # explore values
                for sub in val.values():
                    out.extend(collect_runs(sub))
        elif isinstance(val, list):
            for item in val:
                out.extend(collect_runs(item))
        return out

    total_runs = 0
    for node, ref in run_data_obj.items():
        resolved = _resolve_value(ref)
        run_dicts = collect_runs(resolved)
        node_runs: list[NodeRun] = []
        for rd in run_dicts:
            try:
                start_time = int(rd.get("startTime", 0))
                execution_time = int(rd.get("executionTime", rd.get("execution_time", 0)) or 0)
                status = str(rd.get("executionStatus") or rd.get("status") or "unknown")
                data_resolved = _resolve_value(rd.get("data"))
                input_override_resolved = _resolve_value(rd.get("inputOverride")) if rd.get("inputOverride") else None
                source_resolved = _resolve_value(rd.get("source"))
                sources: list[NodeRunSource] = []
                if isinstance(source_resolved, dict):
                    if any(k.startswith("previousNode") for k in source_resolved.keys()):
                        pn = source_resolved.get("previousNode")
                        if not isinstance(pn, (str, type(None))):
                            pn = None
                        pnr = source_resolved.get("previousNodeRun")
                        if isinstance(pnr, str) and pnr.isdigit():
                            pnr = int(pnr)
                        if not isinstance(pnr, (int, type(None))):
                            pnr = None
                        sources.append(NodeRunSource(previousNode=pn, previousNodeRun=pnr))
                elif isinstance(source_resolved, list):
                    for s in source_resolved:
                        if isinstance(s, dict) and ("previousNode" in s or "previousNodeRun" in s):
                            pn = s.get("previousNode")
                            if not isinstance(pn, (str, type(None))):
                                pn = None
                            pnr = s.get("previousNodeRun")
                            if isinstance(pnr, str) and pnr.isdigit():
                                pnr = int(pnr)
                            if not isinstance(pnr, (int, type(None))):
                                pnr = None
                            sources.append(NodeRunSource(previousNode=pn, previousNodeRun=pnr))
                # Token usage may be referenced separately; attempt inline
                token_usage = None
                if isinstance(data_resolved, dict) and "tokenUsage" not in data_resolved:
                    # Heuristic: rd may have tokenUsage pointer
                    tu = _resolve_value(rd.get("tokenUsage"))
                    if isinstance(tu, dict):
                        data_resolved["tokenUsage"] = tu
                node_run = NodeRun(
                    startTime=start_time,
                    executionTime=execution_time,
                    executionStatus=status,
                    data=data_resolved if isinstance(data_resolved, dict) else {"value": data_resolved},
                    source=sources or None,
                    inputOverride=input_override_resolved if isinstance(input_override_resolved, dict) else None,
                    error=None,
                )
                node_runs.append(node_run)
            except Exception as e:  # pragma: no cover - best effort decoding
                logger.debug("Failed to decode compact run for node %s: %s", node, e)
        if node_runs:
            node_run_map[node] = node_runs
            total_runs += len(node_runs)

    if not node_run_map:
        return None
    if debug:
        logger.info(
            "Execution %s: Decoded compact pointer execution format: nodes=%d runs=%d",
            execution_id,
            len(node_run_map),
            total_runs,
        )
    return ExecutionData(
        executionData=ExecutionDataDetails(
            resultData=ResultData(runData=node_run_map)
        )
    )


@app.callback()
def main() -> None:  # pragma: no cover - simple callback
    """n8n-langfuse-shipper CLI.

    Use a subcommand like 'backfill' to run a process.
    """
    pass


@app.command(help="Run a single backfill cycle (Iteration 2 basic mapping).")
def backfill(
    start_after_id: Optional[int] = typer.Option(
        None, help="Start processing executions with id greater than this value (overrides checkpoint)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Maximum number of executions to process in this run"
    ),
    dry_run: bool = typer.Option(True, help="If true, do not send spans to Langfuse (mapping only)"),
    checkpoint_file: Optional[str] = typer.Option(
        None, help="Path to checkpoint file (defaults to settings.CHECKPOINT_FILE)"
    ),
    debug: bool = typer.Option(False, help="Enable verbose debug for execution data parsing"),
    attempt_decompress: bool = typer.Option(
        False, help="Attempt decompression of execution data payloads (currently placeholder)"
    ),
    debug_dump_dir: Optional[str] = typer.Option(
        None, help="Directory to dump raw execution data JSON when debug enabled"
    ),
    truncate_len: Optional[int] = typer.Option(
        None,
        help="Override truncation length for input/output serialization (0 disables truncation). Overrides TRUNCATE_FIELD_LEN env setting.",
    ),
    require_execution_metadata: bool = typer.Option(
        None,
        help="If set, only process executions that have a metadata row (execution_metadata) with key='executionId' and value matching the execution id.",
    ),
    export_queue_soft_limit: Optional[int] = typer.Option(
        None,
        help="Override EXPORT_QUEUE_SOFT_LIMIT (approx backlog spans before applying sleep)",
    ),
    export_sleep_ms: Optional[int] = typer.Option(
        None,
        help="Override EXPORT_SLEEP_MS (sleep duration in ms when backlog exceeds soft limit)",
    ),
) -> None:
    """Run a backfill cycle to process and export n8n executions.

    This command orchestrates the ETL process:
    - Determines the starting execution ID from the checkpoint or CLI argument.
    - Streams execution records from the database.
    - For each record, maps it to a Langfuse trace and exports it.
    - Updates the checkpoint with the ID of the last processed record.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    if not os.getenv("SUPPRESS_SHIPPER_CREDIT"):
        typer.echo("Powered by n8n-langfuse-shipper (Apache 2.0) - https://github.com/rwb-truelime/n8n-langfuse-shipper")
    typer.echo("Starting backfill with mapping...")
    # Apply optional runtime overrides for export backpressure tuning
    if export_queue_soft_limit is not None:
        # Runtime override of settings attribute (present on Settings model)
        setattr(settings, "EXPORT_QUEUE_SOFT_LIMIT", int(export_queue_soft_limit))
    if export_sleep_ms is not None:
        setattr(settings, "EXPORT_SLEEP_MS", int(export_sleep_ms))
    # Determine metadata filter flag: CLI overrides env/settings
    require_meta_flag = (
        require_execution_metadata
        if require_execution_metadata is not None
        else settings.REQUIRE_EXECUTION_METADATA
    )
    source = ExecutionSource(
        settings.PG_DSN,
        batch_size=settings.FETCH_BATCH_SIZE,
        schema=settings.DB_POSTGRESDB_SCHEMA or None,
        table_prefix=settings.DB_TABLE_PREFIX if settings.DB_TABLE_PREFIX is not None else None,
        require_execution_metadata=require_meta_flag,
    )

    cp_path = checkpoint_file or settings.CHECKPOINT_FILE
    effective_start_after = start_after_id
    if effective_start_after is None:
        loaded = load_checkpoint(cp_path)
        if loaded is not None:
            effective_start_after = loaded
            logging.getLogger(__name__).info(
                "Loaded checkpoint id %s from %s", loaded, cp_path
            )

    async def _run() -> None:
        count: int = 0
        last_id: Optional[int] = effective_start_after
        # Track earliest and latest startedAt among processed executions for user reconciliation.
        earliest_started: Optional[datetime] = None
        latest_started: Optional[datetime] = None

        async for raw in source.stream(start_after_id=effective_start_after, limit=limit):
            record = N8nExecutionRecord(
                id=raw["id"],
                workflowId=raw["workflowId"],
                status=raw["status"],
                startedAt=raw["startedAt"],
                stoppedAt=raw["stoppedAt"],
                workflowData=WorkflowData(**raw["workflowData"]),
                # Attempt to parse full execution data (with runData). Fallback to empty if shape unexpected.
                data=_build_execution_data(
                    raw.get("data"),
                    workflow_data_raw=raw.get("workflowData"),
                    debug=debug,
                    attempt_decompress=attempt_decompress,
                    execution_id=raw["id"],
                ),
            )
            if debug and debug_dump_dir:
                try:
                    import json
                    import os as _os
                    _os.makedirs(debug_dump_dir, exist_ok=True)
                    dump_path = _os.path.join(debug_dump_dir, f"execution_{record.id}_data.json")
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(raw.get("data"), f, ensure_ascii=False, indent=2)
                    logging.getLogger(__name__).info("Dumped raw data JSON to %s", dump_path)
                except Exception as e:
                    logging.getLogger(__name__).warning("Failed dumping raw data JSON: %s", e)
            effective_trunc: Optional[int] = (
                settings.TRUNCATE_FIELD_LEN if truncate_len is None else truncate_len
            )
            if effective_trunc == 0:
                effective_trunc = None  # signal no truncation
            # Media upload feature path (Langfuse Media API). When enabled we
            # always collect binaries then patch with media tokens.
            if settings.ENABLE_MEDIA_UPLOAD:
                mapped = map_execution_with_assets(
                    record,
                    truncate_limit=effective_trunc,
                    collect_binaries=True,
                )
                patch_and_upload_media(mapped, settings)
                trace = mapped.trace
            else:
                trace = map_execution_to_langfuse(record, truncate_limit=effective_trunc)
            span_count = len(trace.spans)
            if span_count <= 1:
                logging.getLogger(__name__).warning(
                    "Execution %s produced %d span(s); likely missing runData. workflowId=%s", record.id, span_count, record.workflowId
                )
            else:
                logging.getLogger(__name__).debug(
                    "Execution %s mapped to %d spans", record.id, span_count
                )
            export_trace(trace, settings, dry_run=dry_run)
            # Track earliest / latest window for user reconciliation with Langfuse UI filters.
            if earliest_started is None or record.startedAt < earliest_started:
                earliest_started = record.startedAt
            if latest_started is None or record.startedAt > latest_started:
                latest_started = record.startedAt
            if debug:
                logging.getLogger(__name__).info(
                    "Exported execution %s -> trace %s spans=%d startedAt=%s",
                    record.id,
                    trace.id,
                    len(trace.spans),
                    record.startedAt.isoformat(),
                )
            count += 1
            last_id = int(record.id)
        if not dry_run and last_id is not None:
            store_checkpoint(cp_path, last_id)
            logging.getLogger(__name__).info(
                "Stored checkpoint id %s to %s", last_id, cp_path
            )
        typer.echo(
            f"Processed {count} execution(s). dry_run={dry_run} start_after={effective_start_after}"
        )
        if count:
            logging.getLogger(__name__).info(
                (
                    "Execution time window processed: earliest_started=%s "
                    "latest_started=%s (UTC). If Langfuse UI date filter excludes part of "
                    "this range, displayed trace count may be lower."
                ),
                earliest_started.isoformat() if earliest_started else None,
                latest_started.isoformat() if latest_started else None,
            )

    asyncio.run(_run())
    # Ensure exporter flush & shutdown for short-lived process reliability
    shutdown_exporter()


if __name__ == "__main__":  # pragma: no cover
    app()
