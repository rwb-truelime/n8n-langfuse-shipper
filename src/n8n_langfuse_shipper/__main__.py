"""Main CLI entry point for the n8n-langfuse-shipper.

This module provides a command-line interface using Typer to run the shipper
process. It orchestrates the entire ETL pipeline:
1.  Loading configuration and checkpoints.
2.  Streaming execution records from the database (n8n_langfuse_shipper.db).
3.  Parsing and validating the raw data, including handling complex formats like
    pointer-compressed executions.
4.  Mapping the records to Langfuse traces (n8n_langfuse_shipper.mapper).
5.  Exporting the traces via OTLP (n8n_langfuse_shipper.shipper).
6.  Storing the new checkpoint upon successful processing.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from pydantic import ValidationError

# Load .env file if present (before any config access)
try:
    from dotenv import load_dotenv, find_dotenv

    env_file = find_dotenv(usecwd=True) or find_dotenv()
    if env_file:
        load_dotenv(env_file)
        logging.debug("Loaded environment from %s", env_file)
except Exception:
    pass

from .checkpoint import load_checkpoint, store_checkpoint
from .config import get_settings
from .db import ExecutionSource
from .mapper import map_execution_to_langfuse, map_execution_with_assets
from .media_api import patch_and_upload_media
from .models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    ResultData,
    WorkflowData,
)
from .shipper import export_trace, shutdown_exporter

app = typer.Typer(help="n8n to Langfuse shipper shipper CLI")


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

    # Pointer-compressed (flatted) format: detect list root and attempt upstream flatted parse.
    # If the DB driver already decoded JSON into a list, re-serialize for parser.
    if isinstance(raw_data, list):
        try:
            import json
            from .vendor.flatted import parse as flatted_parse  # type: ignore
            from .vendor import flatted as _flatted_mod  # for _String unwrap
            def _sanitize_flatted(val: Any) -> Any:
                # Recursively convert leftover _String wrapper instances to raw string values.
                if isinstance(val, getattr(_flatted_mod, "_String")):
                    return val.value
                if isinstance(val, list):
                    return [_sanitize_flatted(x) for x in val]
                if isinstance(val, dict):
                    return {k: _sanitize_flatted(v) for k, v in val.items()}
                return val
            serialized = json.dumps(raw_data)
            parsed_root = flatted_parse(serialized)
            parsed_root = _sanitize_flatted(parsed_root)
            # Expect structure: root.resultData.runData
            run_data = (
                parsed_root.get("resultData", {})
                .get("runData", {})
            ) if isinstance(parsed_root, dict) else {}
            if isinstance(run_data, dict) and run_data:
                if debug:
                    logger.info(
                        "Execution %s: Parsed flatted pointer-compressed format with %d node keys",
                        execution_id,
                        len(run_data),
                    )
                return ExecutionData(
                    executionData=ExecutionDataDetails(
                        resultData=ResultData(runData=run_data)
                    )
                )
        except Exception as e:  # pragma: no cover - fail open
            if debug:
                logger.warning(
                    "Execution %s: flatted parse failed (%s); falling back to other paths",
                    execution_id,
                    e,
                )

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




@app.callback()
def main() -> None:  # pragma: no cover - simple callback
    """n8n-langfuse-shipper CLI.

    Use a subcommand like 'shipper' to run a process.
    """
    pass


@app.command(help="Run a single shipper cycle (Iteration 2 basic mapping).")
def shipper(
    start_after_id: Optional[int] = typer.Option(
        None, help="Start processing executions with id greater than this value (overrides checkpoint)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Maximum number of executions to process in this run"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="If true, do not send spans to Langfuse (mapping only). If not specified, uses DRY_RUN from config/env.",
    ),
    checkpoint_file: Optional[str] = typer.Option(
        None, help="Path to checkpoint file (defaults to settings.CHECKPOINT_FILE)"
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Enable verbose debug for execution data parsing. If not specified, uses DEBUG from config/env.",
    ),
    attempt_decompress: bool = typer.Option(
        False,
        "--attempt-decompress/--no-attempt-decompress",
        help="Attempt decompression of execution data payloads. If not specified, uses ATTEMPT_DECOMPRESS from config/env.",
    ),
    debug_dump_dir: Optional[str] = typer.Option(
        None, help="Directory to dump raw execution data JSON when debug enabled (overrides DEBUG_DUMP_DIR)"
    ),
    truncate_len: Optional[int] = typer.Option(
        None,
        help="Override truncation length for input/output serialization (0 disables truncation). Overrides TRUNCATE_FIELD_LEN env setting.",
    ),
    require_execution_metadata: bool = typer.Option(
        False,
        "--require-execution-metadata/--no-require-execution-metadata",
        help=(
            "If set, only process executions that have a metadata row (execution_metadata) "
            "with key='executionId' and value matching the execution id. "
            "If not specified, uses REQUIRE_EXECUTION_METADATA from config/env."
        ),
    ),
    export_queue_soft_limit: Optional[int] = typer.Option(
        None,
        help="Override EXPORT_QUEUE_SOFT_LIMIT (approx backlog spans before applying sleep)",
    ),
    export_sleep_ms: Optional[int] = typer.Option(
        None,
        help="Override EXPORT_SLEEP_MS (sleep duration in ms when backlog exceeds soft limit)",
    ),
    filter_ai_only: bool = typer.Option(
        False,
        "--filter-ai-only/--no-filter-ai-only",
        help=(
            "Only export spans for AI-related nodes (LangChain package). Root span always "
            "included; non-AI parents of AI nodes preserved. Executions with no AI nodes "
            "export root span only with n8n.filter.no_ai_spans=true. "
            "If not specified, uses FILTER_AI_ONLY from config/env."
        ),
    ),
) -> None:
    """Run a shipper cycle to process and export n8n executions.

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
    typer.echo("Starting shipper with mapping...")

    # Check sys.argv to detect if flags were explicitly provided
    # This allows respecting .env settings when flags are omitted
    import sys

    dry_run_explicit = "--dry-run" in sys.argv or "--no-dry-run" in sys.argv
    debug_explicit = "--debug" in sys.argv or "--no-debug" in sys.argv
    decompress_explicit = "--attempt-decompress" in sys.argv or "--no-attempt-decompress" in sys.argv

    effective_dry_run = dry_run if dry_run_explicit else settings.DRY_RUN
    effective_debug = debug if debug_explicit else settings.DEBUG
    effective_decompress = attempt_decompress if decompress_explicit else settings.ATTEMPT_DECOMPRESS
    effective_dump_dir = debug_dump_dir or settings.DEBUG_DUMP_DIR

    # Apply optional runtime overrides for export backpressure tuning
    if export_queue_soft_limit is not None:
        # Runtime override of settings attribute (present on Settings model)
        settings.EXPORT_QUEUE_SOFT_LIMIT = int(export_queue_soft_limit)
    if export_sleep_ms is not None:
        settings.EXPORT_SLEEP_MS = int(export_sleep_ms)

    # Check sys.argv for filter-ai-only and require-execution-metadata flags
    filter_ai_explicit = "--filter-ai-only" in sys.argv or "--no-filter-ai-only" in sys.argv
    require_meta_explicit = "--require-execution-metadata" in sys.argv or "--no-require-execution-metadata" in sys.argv

    effective_filter_ai_only = filter_ai_only if filter_ai_explicit else settings.FILTER_AI_ONLY
    require_meta_flag = require_execution_metadata if require_meta_explicit else settings.REQUIRE_EXECUTION_METADATA

    source = ExecutionSource(
        settings.PG_DSN,
        batch_size=settings.FETCH_BATCH_SIZE,
        schema=settings.DB_POSTGRESDB_SCHEMA or None,
        table_prefix=settings.DB_TABLE_PREFIX if settings.DB_TABLE_PREFIX is not None else None,
        require_execution_metadata=require_meta_flag,
        filter_workflow_ids=settings.FILTER_WORKFLOW_IDS,
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
                    debug=effective_debug,
                    attempt_decompress=effective_decompress,
                    execution_id=raw["id"],
                ),
            )
            if effective_debug and effective_dump_dir:
                try:
                    import json
                    import os as _os
                    _os.makedirs(effective_dump_dir, exist_ok=True)
                    dump_path = _os.path.join(effective_dump_dir, f"execution_{record.id}_data.json")
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
            # Media upload feature path (Langfuse Media API).
            # Phase order change: we first export spans to obtain OTLP span ids
            # (observation ids) then run media upload so create_media can link
            # assets to observations. Tokens patched locally after export; the
            # OTLP-exported span output may not include tokens (contract
            # update documented in instructions & README).
            mapped = None  # for media upload path later
            if settings.ENABLE_MEDIA_UPLOAD:
                mapped = map_execution_with_assets(
                    record,
                    truncate_limit=effective_trunc,
                    collect_binaries=True,
                    filter_ai_only=effective_filter_ai_only,
                )
                trace = mapped.trace
            else:
                trace = map_execution_to_langfuse(
                    record,
                    truncate_limit=effective_trunc,
                    filter_ai_only=effective_filter_ai_only,
                )
            span_count = len(trace.spans)
            if span_count <= 1:
                logging.getLogger(__name__).warning(
                    "Execution %s produced %d span(s); likely missing runData. workflowId=%s", record.id, span_count, record.workflowId
                )
            else:
                logging.getLogger(__name__).debug(
                    "Execution %s mapped to %d spans", record.id, span_count
                )
            export_trace(trace, settings, dry_run=effective_dry_run)
            if settings.ENABLE_MEDIA_UPLOAD and mapped is not None:
                # Now that OTLP span ids are populated, perform media create + upload.
                try:
                    patch_and_upload_media(mapped, settings)
                except Exception as e:  # pragma: no cover - non-fatal path
                    logging.getLogger(__name__).warning(
                        "media upload phase failed execution=%s err=%s", record.id, e
                    )
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
        if not effective_dry_run and last_id is not None:
            store_checkpoint(cp_path, last_id)
            logging.getLogger(__name__).info(
                "Stored checkpoint id %s to %s", last_id, cp_path
            )
        typer.echo(
            f"Processed {count} execution(s). dry_run={effective_dry_run} start_after={effective_start_after}"
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
