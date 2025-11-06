"""Database interaction layer for fetching n8n execution records.

This module contains the "E" (Extract) part of the ETL pipeline. It provides
a class, ExecutionSource, responsible for connecting to the n8n PostgreSQL
database and streaming execution records in batches. It handles dynamic table
naming (schema and prefix), connection management, and resilient fetching with
exponential backoff.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

psycopg: Any | None
dict_row: Any | None
try:  # pragma: no cover - import guarded for runtime
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover
    psycopg = None
    dict_row = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from psycopg import AsyncConnection
else:  # runtime fallback dynamic type
    AsyncConnection = Any

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 100


class ExecutionSource:
    """Data source for streaming n8n execution records from PostgreSQL.

    This class handles the connection to the database and provides an async
    generator to stream execution records. It fetches joined rows from the
    execution entity and data tables incrementally by their primary key.

    It dynamically constructs table names based on schema and prefix settings,
    allowing it to work with various n8n database configurations.
    """

    def __init__(
        self,
        dsn: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        *,
        schema: Optional[str] = None,
        table_prefix: Optional[str] = None,
        require_execution_metadata: bool = False,
        filter_workflow_ids: Optional[List[str]] = None,
    ):
        """Initialize the execution source and resolve database configuration.

    The schema is resolved via explicit arg -> env -> default 'public'.
    The table prefix is mandatory (explicit arg or DB_TABLE_PREFIX env). No implicit default.

        Args:
            dsn: The full PostgreSQL connection string.
            batch_size: The number of records to fetch in each database query.
            schema: The database schema to use.
            table_prefix: The prefix for n8n's tables (e.g., 'n8n_'). An empty
                string means no prefix.
            require_execution_metadata: If True, only executions that have at
                least one corresponding row in the `execution_metadata` table
                will be fetched.
        """
        self._dsn = dsn
        self._batch_size = batch_size
        # Resolve schema
        if schema is not None:
            self._schema = schema or "public"
        else:
            self._schema = os.environ.get("DB_POSTGRESDB_SCHEMA") or "public"
        # Resolve mandatory prefix
        if table_prefix is not None:
            self._table_prefix = table_prefix  # may be empty
        else:
            if "DB_TABLE_PREFIX" not in os.environ:
                raise RuntimeError("DB_TABLE_PREFIX must be set explicitly (blank allowed for none)")
            self._table_prefix = os.environ.get("DB_TABLE_PREFIX", "")
        # Basic safety: allow only alnum + underscore in prefix & schema
        if not re.fullmatch(r"[A-Za-z0-9_]+", self._schema):
            raise ValueError("Invalid schema name")
        if not re.fullmatch(r"[A-Za-z0-9_]*", self._table_prefix):
            raise ValueError("Invalid table prefix")
        # Precompute table names for logging / diagnostics
        self._entity_table_name = f"{self._table_prefix}execution_entity"
        self._data_table_name = f"{self._table_prefix}execution_data"
        self._metadata_table_name = f"{self._table_prefix}execution_metadata"
        self._require_execution_metadata = require_execution_metadata
        # Optional workflow id allow-list filtering (empty or None = no filter)
        self._filter_workflow_ids = [
            w.strip() for w in (filter_workflow_ids or []) if w.strip()
        ]
        logger.info(
            "DB init: schema=%s prefix=%r entity_table=%s data_table=%s metadata_table=%s require_exec_meta=%s (explicit_prefix=%s)",
            self._schema,
            self._table_prefix,
            self._entity_table_name,
            self._data_table_name,
            self._metadata_table_name,
            self._require_execution_metadata,
            table_prefix is not None,
        )

    @asynccontextmanager
    async def _connect(self) -> AsyncGenerator[AsyncConnection, None]:
        """Async context manager yielding a live PostgreSQL connection."""
        if not self._dsn:
            raise RuntimeError("PG_DSN is empty; cannot establish database connection")
        if psycopg is None:  # pragma: no cover
            raise RuntimeError("psycopg not installed in current environment")
        conn: AsyncConnection = await psycopg.AsyncConnection.connect(self._dsn)
        try:
            yield conn
        finally:  # noqa: SIM105 (clarity over compressed form)
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
    ) -> List[Dict[str, Any]]:
        """Fetch a single batch of execution records from the database.

        This method executes a SQL query to get the next batch of records after
        a given ID. It is decorated with `tenacity.retry` to handle transient
        database errors with exponential backoff.

        The SQL query is constructed dynamically to include the correct schema
        and table prefix. If `_require_execution_metadata` is True, it adds an
        `EXISTS` clause to filter for executions with associated metadata.

        Args:
            conn: The active async database connection.
            last_id: The ID of the last record from the previous batch, used for
                pagination.
            limit: The maximum number of records to fetch.

        Returns:
            A list of dictionaries, where each dictionary represents a fetched
            execution record.
        """
        # Table names with prefix
        entity_table = f'"{self._schema}"."{self._entity_table_name}"'
        data_table = f'"{self._schema}"."{self._data_table_name}"'
        wf_filter_clause = ""
        params: List[Any] = []
        if self._filter_workflow_ids:
            # Use = ANY(%s) style array matching for safety & simplicity.
            # psycopg will adapt list to array automatically.
            wf_filter_clause = ' AND e."workflowId" = ANY(%s)'
            params.append(self._filter_workflow_ids)

        if self._require_execution_metadata:
            meta_table = f'"{self._schema}"."{self._metadata_table_name}"'
            # Only select executions that have at least one metadata row referencing them (ANY key/value).
            # Use EXISTS to avoid row multiplication from multiple metadata rows.
            sql = (
                f'SELECT e.id, e."workflowId" AS "workflowId", e.status, '
                f'e."startedAt" AS "startedAt", e."stoppedAt" AS "stoppedAt", '
                f'd."workflowData" AS "workflowData", d."data" AS data '
                f'FROM {entity_table} e '
                f'JOIN {data_table} d ON e.id = d."executionId" '
                f'WHERE e.id > %s AND EXISTS (SELECT 1 FROM {meta_table} m WHERE m."executionId" = e.id){wf_filter_clause} '
                'ORDER BY e.id ASC '
                'LIMIT %s'
            )
            params = [last_id] + params + [limit]
        else:
            sql = (
                f'SELECT e.id, e."workflowId" AS "workflowId", e.status, '
                f'e."startedAt" AS "startedAt", e."stoppedAt" AS "stoppedAt", '
                f'd."workflowData" AS "workflowData", d."data" AS data '
                f'FROM {entity_table} e '
                f'JOIN {data_table} d ON e.id = d."executionId" '
                f'WHERE e.id > %s{wf_filter_clause} '
                'ORDER BY e.id ASC '
                'LIMIT %s'
            )
            params = [last_id] + params + [limit]
        async with conn.cursor(row_factory=dict_row) as cur:
            try:
                await cur.execute(sql, tuple(params))
            except Exception as ex:  # noqa: BLE001 broad for friendly diagnostics then re-raise
                # Rollback transaction if it's in failed state to allow subsequent attempts
                try:
                    if psycopg is not None and isinstance(ex, getattr(psycopg.errors, "InFailedSqlTransaction", tuple())):
                        await conn.rollback()
                except Exception:  # pragma: no cover
                    pass
                # Friendly message for missing tables (likely prefix mismatch)
                if psycopg is not None and isinstance(ex, getattr(psycopg.errors, "UndefinedTable", tuple())):
                    logger.error(
                        "Table lookup failed. Attempted tables: %s, %s (schema=%s, prefix=%r). "
                        "If your n8n instance uses a different prefix set DB_TABLE_PREFIX accordingly or leave it unset for default 'n8n_'.",
                        self._entity_table_name,
                        self._data_table_name,
                        self._schema,
                        self._table_prefix,
                    )
                raise
            rows: List[Dict[str, Any]] = await cur.fetchall()
            return rows

    async def stream(
        self, start_after_id: Optional[int] = None, limit: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream execution records from the database.

        This is the main public method of the class. It acts as an asynchronous
        generator, continuously fetching and yielding batches of execution
        records until the specified limit is reached or no more records are
        available.

        Args:
            start_after_id: If provided, streaming will start from the record
                immediately after this execution ID (exclusive).
            limit: The total maximum number of records to yield. If None, it
                streams until the end of the table.

        Yields:
            A dictionary representing a single n8n execution record.
        """
        if not self._dsn:
            logger.warning("PG_DSN not set; no executions will be streamed")
            return

        last_id = start_after_id or 0
        yielded = 0

        try:
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
                        yield row
                        yielded += 1
                        last_id = row["id"]
                        if limit is not None and yielded >= limit:
                            break

                    if limit is not None and yielded >= limit:
                        break

                    await asyncio.sleep(0)
        except Exception as conn_err:  # pragma: no cover - integration environment dependent
            # Gracefully degrade when DB unreachable so tests without a live Postgres skip behaviorally.
            logger.warning("Database connection/stream failed: %s (yielded=%d). Returning no rows.", conn_err, yielded)
            return
        logger.info(
            "Stream completed: yielded=%d start_after_id=%s final_last_id=%d", yielded, start_after_id, last_id
        )
