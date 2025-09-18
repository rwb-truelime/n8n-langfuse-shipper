from __future__ import annotations

import typer
from typing import Optional
import logging
import asyncio

from .config import get_settings
from .db import ExecutionSource
from .models.n8n import N8nExecutionRecord, WorkflowData, ExecutionData, ExecutionDataDetails, ResultData
from .mapper import map_execution_to_langfuse
from .shipper import export_trace

app = typer.Typer(help="n8n to Langfuse backfill shipper CLI")


@app.callback()
def main():  # pragma: no cover - simple callback
    """n8n-langfuse-shipper CLI. Use subcommands like 'backfill'."""
    pass


@app.command(help="Run a single backfill cycle (Iteration 2 basic mapping).")
def backfill(
    start_after_id: Optional[int] = typer.Option(
        None, help="Start processing executions with id greater than this value"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Maximum number of executions to process in this run"
    ),
    dry_run: bool = typer.Option(
        True, help="If true, do not send spans to Langfuse (mapping only)"
    ),
):
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    typer.echo("Starting backfill with mapping...")
    source = ExecutionSource(settings.PG_DSN)

    async def _run():
        count = 0
        async for raw in source.stream(start_after_id=start_after_id, limit=limit):
            record = N8nExecutionRecord(
                id=raw["id"],
                workflowId=raw["workflowId"],
                status=raw["status"],
                startedAt=raw["startedAt"],
                stoppedAt=raw["stoppedAt"],
                workflowData=WorkflowData(**raw["workflowData"]),
                data=ExecutionData(
                    executionData=ExecutionDataDetails(
                        resultData=ResultData(runData={})
                    )
                ),
            )
            trace = map_execution_to_langfuse(record, truncate_limit=settings.TRUNCATE_FIELD_LEN)
            export_trace(trace, settings, dry_run=dry_run)
            count += 1
        typer.echo(f"Processed {count} execution(s). dry_run={dry_run}")

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    app()
