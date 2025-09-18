from __future__ import annotations

import typer
from typing import Optional, Any
import logging
import asyncio
from pydantic import ValidationError

from .config import get_settings
from .db import ExecutionSource
from .models.n8n import N8nExecutionRecord, WorkflowData, ExecutionData, ExecutionDataDetails, ResultData
from .mapper import map_execution_to_langfuse
from .shipper import export_trace
from .checkpoint import load_checkpoint, store_checkpoint

app = typer.Typer(help="n8n to Langfuse backfill shipper CLI")


def _build_execution_data(
    raw_data: Optional[dict | str | list],
    workflow_data_raw: Optional[dict] = None,
    *,
    debug: bool = False,
    attempt_decompress: bool = False,
    execution_id: Optional[int] = None,
) -> ExecutionData:
    """Build ExecutionData model from raw DB JSON.

    Defined before app() invocation to avoid NameError when module executed as script.

    The n8n execution_data.data column usually contains a structure:
    {
      "executionData": {"resultData": {"runData": {...}} , ...}
      ... other keys ...
    }
    We only care (for now) about executionData.resultData.runData.
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
    def _from_run_data(rd: dict) -> ExecutionData:
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
    candidates = []
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


def _decode_compact_pointer_execution(pool: list, debug: bool = False, execution_id: Optional[int] = None) -> Optional[ExecutionData]:
    """Decode alternative compact pointer-array execution format into ExecutionData.

    Observed Format (heuristic):
      - Top-level is a list (pool) of heterogeneous entries (dict/list/str).
      - Objects reference other entries by stringified integer indices (e.g., "4").
      - pool[0] contains keys like resultData / executionData referencing other indices.
      - resultData object contains key runData -> pointer to dict mapping node name -> pointer(s) to run(s).
      - Each run object includes: startTime, executionTime, executionStatus, data, source, inputOverride.
    """
    logger = logging.getLogger(__name__)
    if not pool or not isinstance(pool, list) or not pool:
        return None
    if not isinstance(pool[0], dict):
        return None

    # Generic pointer resolver with memoization & cycle guard
    memo: dict[int, Any] = {}
    resolving: set[int] = set()

    def resolve_index(i: int):
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

    def _resolve_value(v: Any):
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

    def collect_runs(val: Any) -> list[dict]:
        out: list[dict] = []
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
def main():  # pragma: no cover - simple callback
    """n8n-langfuse-shipper CLI. Use subcommands like 'backfill'."""
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
):
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    typer.echo("Starting backfill with mapping...")
    source = ExecutionSource(settings.PG_DSN)

    cp_path = checkpoint_file or settings.CHECKPOINT_FILE
    effective_start_after = start_after_id
    if effective_start_after is None:
        loaded = load_checkpoint(cp_path)
        if loaded is not None:
            effective_start_after = loaded
            logging.getLogger(__name__).info(
                "Loaded checkpoint id %s from %s", loaded, cp_path
            )

    async def _run():
        count = 0
        last_id = effective_start_after
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
                    import json, os
                    os.makedirs(debug_dump_dir, exist_ok=True)
                    dump_path = os.path.join(debug_dump_dir, f"execution_{record.id}_data.json")
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(raw.get("data"), f, ensure_ascii=False, indent=2)
                    logging.getLogger(__name__).info("Dumped raw data JSON to %s", dump_path)
                except Exception as e:
                    logging.getLogger(__name__).warning("Failed dumping raw data JSON: %s", e)
            trace = map_execution_to_langfuse(record, truncate_limit=settings.TRUNCATE_FIELD_LEN)
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
            count += 1
            last_id = record.id
        if not dry_run and last_id is not None:
            store_checkpoint(cp_path, last_id)
            logging.getLogger(__name__).info(
                "Stored checkpoint id %s to %s", last_id, cp_path
            )
        typer.echo(
            f"Processed {count} execution(s). dry_run={dry_run} start_after={effective_start_after}"
        )

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    app()
