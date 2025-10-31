"""Public mapping facade (kept intentionally small).

All substantive logic lives in ``mapping.orchestrator`` and helper modules
under ``src/mapping/``. This file provides a stable import path for external
callers and tests: ``from mapper import map_execution_to_langfuse``.
"""

from __future__ import annotations

from typing import Optional

from .mapping.orchestrator import (
    _apply_ai_filter,
    _map_execution,
    _extract_model_and_metadata,
    _detect_gemini_empty_output_anomaly,
    _flatten_runs,
    _resolve_parent,  # re-export for tests relying on direct import
)
from .mapping.id_utils import SPAN_NAMESPACE  # deterministic UUIDv5 namespace
from .media_api import MappedTraceWithAssets
from .models.langfuse import LangfuseTrace
from .models.n8n import N8nExecutionRecord

__all__ = [
    "map_execution_to_langfuse",
    "map_execution_with_assets",
    "MappedTraceWithAssets",
    # Helper re-exports (test-only / internal use)
    "_extract_model_and_metadata",
    "_detect_gemini_empty_output_anomaly",
    "_flatten_runs",
    "_resolve_parent",
    "SPAN_NAMESPACE",  # stable namespace export (tests depend on this)
]


def map_execution_to_langfuse(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    *,
    filter_ai_only: bool = False,
) -> LangfuseTrace:
    """Map an execution record to a LangfuseTrace.

    Args:
        record: Parsed n8n execution entity+data record.
        truncate_limit: Max chars for input/output text (0 disables truncation;
            binary stripping always on).
        filter_ai_only: Retain only AI spans plus context window when True.
    """
    trace, _assets = _map_execution(
        record, truncate_limit=truncate_limit, collect_binaries=False
    )
    if filter_ai_only:
        _apply_ai_filter(trace, record)
    return trace


def map_execution_with_assets(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    collect_binaries: bool = False,
    *,
    filter_ai_only: bool = False,
) -> MappedTraceWithAssets:
    """Map an execution and optionally collect binary asset placeholders.

    When `collect_binaries` is True, a list of stripped binary/base64 placeholder
    objects is returned for later media token exchange. Otherwise the list is
    empty (even though mapping ran the same core logic).
    """
    trace, assets = _map_execution(
        record, truncate_limit=truncate_limit, collect_binaries=collect_binaries
    )
    if filter_ai_only:
        _apply_ai_filter(trace, record)
    return MappedTraceWithAssets(trace=trace, assets=assets if collect_binaries else [])
