from __future__ import annotations

from typing import AsyncGenerator, Optional, List, Dict, Any
import asyncio
import logging
from contextlib import asynccontextmanager

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import re

try:  # pragma: no cover - import guarded for type checkers
    import psycopg
    from psycopg.rows import dict_row
    from psycopg import AsyncConnection  # type: ignore
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore
    AsyncConnection = Any  # type: ignore

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 100


class ExecutionSource:
    """Data source for streaming n8n execution records from PostgreSQL.

    Fetches joined rows from `n8n_execution_entity` and `n8n_execution_data` incrementally by id.
    """

    def __init__(self, dsn: str, batch_size: int = DEFAULT_BATCH_SIZE):
        self._dsn = dsn
        self._batch_size = batch_size
        self._schema = os.environ.get("DB_POSTGRESDB_SCHEMA") or "public"
        self._table_prefix = os.environ.get("DB_TABLE_PREFIX") or "n8n_"
        # Basic safety: allow only alnum + underscore in prefix & schema
        if not re.fullmatch(r"[A-Za-z0-9_]+", self._schema):
            raise ValueError("Invalid schema name")
        if not re.fullmatch(r"[A-Za-z0-9_]*", self._table_prefix):
            raise ValueError("Invalid table prefix")

    @asynccontextmanager
    async def _connect(self) -> AsyncGenerator[AsyncConnection, None]:  # type: ignore
        if not self._dsn:
            raise RuntimeError("PG_DSN is empty; cannot establish database connection")
        if psycopg is None:  # pragma: no cover
            raise RuntimeError("psycopg not installed in current environment")
        conn = await psycopg.AsyncConnection.connect(self._dsn)  # type: ignore[attr-defined]
        try:
            yield conn
        finally:
            try:
                await conn.close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("Error closing Postgres connection", exc_info=True)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def _fetch_batch(
        self, conn: Any, last_id: int, limit: int
    ) -> List[Dict[str, Any]]:  # conn typed as Any for compatibility
        # Table names with prefix
        entity_table = f'"{self._schema}"."{self._table_prefix}execution_entity"'
        data_table = f'"{self._schema}"."{self._table_prefix}execution_data"'
        sql = (
            f'SELECT e.id, e."workflowId" AS "workflowId", e.status, '
            f'e."startedAt" AS "startedAt", e."stoppedAt" AS "stoppedAt", '
            f'd."workflowData" AS "workflowData", d."data" AS data '
            f'FROM {entity_table} e '
            f'JOIN {data_table} d ON e.id = d."executionId" '
            'WHERE e.id > %s '
            'ORDER BY e.id ASC '
            'LIMIT %s'
        )
        async with conn.cursor(row_factory=dict_row) as cur:  # type: ignore
            await cur.execute(sql, (last_id, limit))
            rows = await cur.fetchall()
            return rows  # type: ignore[return-value]

    async def stream(
        self, start_after_id: Optional[int] = None, limit: Optional[int] = None
    ) -> AsyncGenerator[dict, None]:
        """Stream execution records.

        Args:
            start_after_id: resume after this execution id (exclusive).
            limit: maximum number of executions to yield (None for unlimited).
        Yields:
            Dict with keys: id, workflowId, status, startedAt, stoppedAt, workflowData, data
        """
        if not self._dsn:
            logger.warning("PG_DSN not set; no executions will be streamed")
            return

        last_id = start_after_id or 0
        yielded = 0

        async with self._connect() as conn:
            while True:
                if limit is not None:
                    remaining = limit - yielded
                    if remaining <= 0:
                        break
                    batch_limit = min(self._batch_size, remaining)
                else:
                    batch_limit = self._batch_size

                try:
                    rows = await self._fetch_batch(conn, last_id, batch_limit)
                except Exception as e:
                    logger.error("Failed fetching batch after id %s: %s", last_id, e, exc_info=True)
                    raise

                if not rows:
                    break

                for row in rows:
                    # Row already dict via dict_row factory; yield directly.
                    yield row
                    yielded += 1
                    last_id = row["id"]
                    if limit is not None and yielded >= limit:
                        break

                if limit is not None and yielded >= limit:
                    break

                # Yield control to event loop
                await asyncio.sleep(0)

        logger.info(
            "Stream completed: yielded=%d start_after_id=%s final_last_id=%d", yielded, start_after_id, last_id
        )
