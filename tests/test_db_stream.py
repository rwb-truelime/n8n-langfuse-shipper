import asyncio
import os
import pytest

from src.config import get_settings
from src.db import ExecutionSource

pytestmark = pytest.mark.asyncio


async def _collect(limit: int):
    settings = get_settings()
    assert settings.PG_DSN, "PG_DSN must be set via .env or environment for integration test"
    source = ExecutionSource(settings.PG_DSN, batch_size=5)
    rows = []
    async for row in source.stream(limit=limit):
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


async def test_stream_reads_rows_without_modification():
    rows = await _collect(limit=3)
    # We do not assert exact schema contents, only presence of required keys.
    for r in rows:
        for key in ["id", "workflowId", "status", "startedAt", "workflowData", "data"]:
            assert key in r, f"Missing key {key} in row"


async def test_stream_respects_start_after_id():
    rows1 = await _collect(limit=1)
    if not rows1:
        pytest.skip("No executions in database to test start_after_id")
    first_id = rows1[0]["id"]
    settings = get_settings()
    source = ExecutionSource(settings.PG_DSN, batch_size=5)
    rows_after = []
    async for row in source.stream(start_after_id=first_id, limit=1):
        rows_after.append(row)
    if rows_after:  # Only assert ordering if a subsequent row exists
        assert rows_after[0]["id"] > first_id
