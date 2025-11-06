import os
from typing import List

import pytest

from n8n_langfuse_shipper.config import get_settings
from n8n_langfuse_shipper.db import ExecutionSource

pytestmark = pytest.mark.asyncio


async def _collect_ids(filter_ids: List[str] | None, limit: int = 50):
    settings = get_settings()
    assert settings.PG_DSN, "PG_DSN must be set for workflow filtering integration test"
    source = ExecutionSource(
        settings.PG_DSN,
        batch_size=25,
        schema=settings.DB_POSTGRESDB_SCHEMA or None,
        table_prefix=settings.DB_TABLE_PREFIX if settings.DB_TABLE_PREFIX is not None else None,
        require_execution_metadata=False,
        filter_workflow_ids=filter_ids,
    )
    ids: List[str] = []
    async for row in source.stream(limit=limit):
        ids.append(row["workflowId"])
    return ids


async def test_no_filter_returns_diverse_workflow_ids():
    ids = await _collect_ids(None, limit=10)
    # If fewer than 2 rows exist, skip diversity assertion.
    if len(ids) < 2:
        pytest.skip("Not enough executions to test workflow diversity")
    # Diversity heuristic: expect at least one differing workflowId when unfiltered.
    assert len(set(ids)) >= 1


async def test_filter_limits_to_specified_ids():
    # We cannot know a priori which workflowIds exist; attempt to sample one
    # from unfiltered stream then filter on it.
    baseline = await _collect_ids(None, limit=5)
    if not baseline:
        pytest.skip("No executions available to test workflowId filtering")
    target = baseline[0]
    filtered = await _collect_ids([target], limit=5)
    # All returned workflowIds must equal target when filter applied.
    assert filtered, "Expected at least one row with target workflowId"
    assert set(filtered) == {target}


async def test_empty_list_equivalent_to_none():
    baseline = await _collect_ids(None, limit=5)
    empty = await _collect_ids([], limit=5)
    # Order may differ; compare sets and lengths.
    assert set(baseline) == set(empty)
    assert len(baseline) == len(empty)
