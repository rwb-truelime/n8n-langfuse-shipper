from __future__ import annotations

from typing import AsyncGenerator, Optional
import asyncio
import logging

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - psycopg installed in runtime environment
    psycopg = None  # type: ignore

logger = logging.getLogger(__name__)


class ExecutionSource:
    """Data source for streaming n8n execution records.

    Iteration 1: minimal placeholder. In later iterations this will:
      * Open a connection / pool to PostgreSQL
      * Fetch batches of executions joined with execution data JSON
      * Yield parsed Pydantic model instances
    """

    def __init__(self, dsn: str):
        self._dsn = dsn

    async def stream(
        self, start_after_id: Optional[int] = None, limit: Optional[int] = None
    ) -> AsyncGenerator[dict, None]:
        """Placeholder async generator yielding mock execution rows.

        Later this will execute SQL like:
        SELECT e.id, e.workflowId, e.status, e."startedAt", e."stoppedAt", e."workflowData", d.data
        FROM n8n_execution_entity e
        JOIN n8n_execution_data d ON e.id = d.executionId
        WHERE e.id > %s ORDER BY e.id ASC LIMIT %s
        """
        emitted = 0
        current_id = (start_after_id or 0) + 1
        while limit is None or emitted < limit:
            # Placeholder synthetic record
            yield {
                "id": current_id,
                "workflowId": "wf-1",
                "status": "success",
                "startedAt": "2024-01-01T00:00:00Z",
                "stoppedAt": "2024-01-01T00:00:05Z",
                "workflowData": {"id": "wf-1", "name": "Demo Workflow", "nodes": []},
                "data": {"executionData": {"resultData": {"runData": {}}}},
            }
            emitted += 1
            current_id += 1
            await asyncio.sleep(0)  # allow event loop scheduling

            if limit is not None and emitted >= limit:
                break
