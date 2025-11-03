"""Timestamp conversion and timezone normalization for UTC enforcement.

This module provides utilities for converting epoch timestamps (milliseconds or
seconds) to timezone-aware UTC datetime objects. All exported timestamps MUST be
UTC-aware to satisfy the timezone invariant enforced by tests.

Conversion Heuristic:
    Values < 1_000_000_000_000 treated as seconds (Unix epoch range ~1970-2033)
    Values >= 1_000_000_000_000 treated as milliseconds (n8n default format)

Public Functions:
    epoch_ms_to_dt: Convert epoch timestamp to UTC-aware datetime

Design Invariant:
    All datetimes exported to OTLP MUST be timezone-aware UTC. Naive datetimes
    (no tzinfo) are forbidden. Tests scan codebase for datetime.utcnow() usage  # allow-naive-datetime
    and fail if found (naive datetime anti-pattern).
"""
from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["epoch_ms_to_dt"]


def epoch_ms_to_dt(ms: int) -> datetime:
    """Convert epoch timestamp (milliseconds or seconds) to timezone-aware UTC datetime.

    Uses heuristic to distinguish seconds from milliseconds:
    - Values < 1_000_000_000_000 treated as seconds (Unix epoch ~1970-2033)
    - Values >= 1_000_000_000_000 treated as milliseconds (n8n default format)

    Args:
        ms: Epoch timestamp in milliseconds or seconds

    Returns:
        Timezone-aware datetime in UTC

    Note:
        All exported timestamps MUST be UTC-aware to satisfy timezone invariant.
        Naive datetimes are forbidden.
    """
    if ms < 1_000_000_000_000:
        return datetime.fromtimestamp(ms, tz=timezone.utc)
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
