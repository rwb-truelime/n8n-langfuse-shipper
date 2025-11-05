"""Public facade for n8n execution to Langfuse trace mapping.

This module provides the stable public API for converting n8n execution records
into Langfuse-compatible trace structures. All complex mapping logic is delegated
to the n8n_langfuse_shipper.mapping.orchestrator module and its helpers in the
n8n_langfuse_shipper.mapping package.

The facade pattern preserves backwards compatibility for existing imports while
allowing internal refactoring of the mapping implementation.

Public Functions:
    map_execution_to_langfuse: Convert n8n execution to LangfuseTrace
    map_execution_with_assets: Convert execution and collect binary assets

Internal Re-exports:
    _extract_model_and_metadata: Model extraction helper (test usage)
    _detect_gemini_empty_output_anomaly: Gemini bug detection (test usage)
    _flatten_runs: NodeRun chronological sorting (test usage)
    _resolve_parent: Parent span resolution (test usage)
    SPAN_NAMESPACE: Deterministic UUID namespace (test usage)
"""

from __future__ import annotations

from typing import Optional

from .config import get_settings
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
    """Convert n8n execution record to Langfuse trace structure.

    Orchestrates the complete mapping pipeline: creates deterministic trace and span
    IDs, resolves parent-child relationships, normalizes I/O data, detects generation
    spans, and applies optional AI-only filtering.

    Args:
        record: n8n execution with workflow data and node run results
        truncate_limit: Maximum characters for input/output fields before truncation;
            0 disables truncation but binary stripping always applies regardless
        filter_ai_only: When True, retains only AI-related spans plus immediate context
            window (2 before, 2 after, and chain connectors); root span always included

    Returns:
        Complete LangfuseTrace with deterministic IDs and hierarchical spans

    Note:
        Binary data is always stripped and replaced with placeholders even when
        truncation is disabled. This ensures consistent output size constraints.
    """
    trace, _assets = _map_execution(
        record, truncate_limit=truncate_limit, collect_binaries=False
    )
    if filter_ai_only:
        settings = get_settings()
        _apply_ai_filter(
            trace=trace,
            record=record,
            run_data=record.data.executionData.resultData.runData,
            extraction_nodes=settings.FILTER_AI_EXTRACTION_NODES,
            include_patterns=settings.FILTER_AI_EXTRACTION_INCLUDE_KEYS,
            exclude_patterns=settings.FILTER_AI_EXTRACTION_EXCLUDE_KEYS,
            max_value_len=settings.FILTER_AI_EXTRACTION_MAX_VALUE_LEN,
        )
    return trace


def map_execution_with_assets(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    collect_binaries: bool = False,
    *,
    filter_ai_only: bool = False,
) -> MappedTraceWithAssets:
    """Convert n8n execution to trace and optionally extract binary assets.

    Performs standard trace mapping while optionally collecting discovered binary/base64
    assets for subsequent media token exchange via Langfuse Media API. Binary data is
    replaced with temporary placeholders containing SHA256 hash and size metadata.

    Args:
        record: n8n execution with workflow data and node run results
        truncate_limit: Maximum characters for input/output before truncation (0 disables)
        collect_binaries: When True, extracts binary assets and inserts placeholders;
            when False, returns empty asset list but still applies binary redaction
        filter_ai_only: Retain only AI spans plus context window when True

    Returns:
        MappedTraceWithAssets containing the trace and list of discovered binary assets

    Note:
        Asset collection is independent from binary stripping. Stripping always occurs;
        collection determines whether asset metadata is gathered for later upload.
    """
    trace, assets = _map_execution(
        record, truncate_limit=truncate_limit, collect_binaries=collect_binaries
    )
    if filter_ai_only:
        settings = get_settings()
        _apply_ai_filter(
            trace=trace,
            record=record,
            run_data=record.data.executionData.resultData.runData,
            extraction_nodes=settings.FILTER_AI_EXTRACTION_NODES,
            include_patterns=settings.FILTER_AI_EXTRACTION_INCLUDE_KEYS,
            exclude_patterns=settings.FILTER_AI_EXTRACTION_EXCLUDE_KEYS,
            max_value_len=settings.FILTER_AI_EXTRACTION_MAX_VALUE_LEN,
        )
    return MappedTraceWithAssets(trace=trace, assets=assets if collect_binaries else [])
