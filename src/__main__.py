from __future__ import annotations

import typer
from typing import Optional
import logging

from .config import get_settings
from .db import ExecutionSource

app = typer.Typer(help="n8n to Langfuse backfill shipper CLI")


@app.command()
def backfill(
    start_after_id: Optional[int] = typer.Option(
        None, help="Start processing executions with id greater than this value"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Maximum number of executions to process in this run"
    ),
):
    """Run a single backfill cycle (Iteration 1 placeholder)."""
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    typer.echo("Starting backfill (placeholder implementation)...")
    source = ExecutionSource(settings.PG_DSN)
    # In future iterations we'll consume source.stream() and map to Langfuse traces
    typer.echo(
        f"Initialized data source with DSN length {len(settings.PG_DSN)}; start_after_id={start_after_id}, limit={limit}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
