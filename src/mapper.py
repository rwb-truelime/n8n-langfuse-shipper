"""Core mapping logic from n8n execution records to Langfuse internal models.

This module is the "T" (Transform) in the ETL pipeline. It takes raw n8n
execution records fetched from PostgreSQL and converts them into a structured
LangfuseTrace object, which is a serializable representation of a Langfuse trace
and its nested spans.

Key responsibilities include:
- Mapping one n8n execution to one Langfuse trace.
- Mapping each n8n node run to a Langfuse span.
- Resolving parent-child relationships between spans using a multi-tiered
  precedence logic (agent hierarchy, runtime pointers, static graph analysis).
- Detecting and classifying "generation" spans (LLM calls).
- Extracting and normalizing token usage and model names for generations.
- Sanitizing and preparing I/O data by stripping binary content and applying
  optional truncation.
- Propagating outputs from parent spans as inputs to child spans when no
  explicit input override is present.
- Generating deterministic, repeatable IDs for traces and spans.
"""
from __future__ import annotations

import logging
from typing import Optional

from .mapping.orchestrator import (
    _apply_ai_filter,
    _map_execution,
)
from .media_api import MappedTraceWithAssets
from .models.langfuse import LangfuseTrace  # needed for facade return types
from .models.n8n import N8nExecutionRecord

logger = logging.getLogger(__name__)


# Helper imports moved entirely into orchestrator & submodules. Mapper remains
# a thin facade re-exporting selected private helpers for test parity.


## System prompt stripping & IO normalization now reside in
## mapping.io_normalizer; legacy local copies removed.




 # Nested key search moved to mapping.generation module.


## Model extraction helpers relocated to mapping.model_extractor.



## Flatten + parent resolution helpers now sourced from orchestrator & submodules.


## IO preparation now provided by orchestrator._prepare_io_and_output.


# --------------------------- Stage 3 Refactor Helpers ---------------------------

## Model metadata extraction now provided by orchestrator._extract_model_and_metadata.


## Gemini anomaly detection now provided by orchestrator._detect_gemini_empty_output_anomaly.




## Mapping core now provided by orchestrator._map_execution.


def map_execution_to_langfuse(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    *,
    filter_ai_only: bool = False,
) -> LangfuseTrace:
    trace, _assets = _map_execution(
        record, truncate_limit=truncate_limit, collect_binaries=False
    )
    if filter_ai_only:
        _apply_ai_filter(trace, record)
    return trace


## Binary asset collection & discovery now in orchestrator.


## Extended binary discovery now provided by orchestrator._discover_additional_binary_assets.


def map_execution_with_assets(
    record: N8nExecutionRecord,
    truncate_limit: Optional[int] = 4000,
    collect_binaries: bool = False,
    *,
    filter_ai_only: bool = False,
) -> MappedTraceWithAssets:
    # collect_binaries kept for backwards compatibility in case external code
    # passed False explicitly.
    trace, assets = _map_execution(
        record, truncate_limit=truncate_limit, collect_binaries=collect_binaries
    )
    if filter_ai_only:
        _apply_ai_filter(trace, record)
    return MappedTraceWithAssets(trace=trace, assets=assets if collect_binaries else [])


## AI filter now provided by orchestrator._apply_ai_filter.
