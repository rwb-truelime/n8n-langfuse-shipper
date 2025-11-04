"""Internal mapping subpackage for decomposed transformation logic.

This package contains the core implementation of n8n execution to Langfuse trace
mapping, decomposed into focused, single-responsibility modules. All functions
within this package are pure (no network I/O or database writes) and deterministic.

The public API remains in the top-level `mapper.py` facade to maintain backwards
compatibility with existing imports. Callers should not import directly from this
package unless accessing internal helpers for testing purposes.

Modules:
    orchestrator: Central mapping pipeline coordinating all transformation steps
    parent_resolution: Span parent-child relationship determination
    generation: AI generation detection and usage extraction heuristics
    model_extractor: Model name extraction from runtime data and static parameters
    io_normalizer: Input/output structure unwrapping and system prompt stripping
    binary_sanitizer: Binary/base64 detection and redaction utilities
    id_utils: Deterministic UUIDv5 span and trace ID generation
    time_utils: Timestamp conversion and timezone normalization
    mapping_context: State container for mapping iteration

Design Invariants:
    - No network calls or database writes permitted
    - Deterministic output for identical inputs (stable IDs, reproducible structure)
    - Binary stripping always applies regardless of truncation settings
    - Parent resolution follows fixed precedence order
    - Timezone-aware UTC timestamps only
"""
from __future__ import annotations

from . import binary_sanitizer as binary_sanitizer  # noqa: F401
from . import id_utils as id_utils  # noqa: F401
from . import time_utils as time_utils  # noqa: F401

__all__ = ["time_utils", "id_utils", "binary_sanitizer"]
