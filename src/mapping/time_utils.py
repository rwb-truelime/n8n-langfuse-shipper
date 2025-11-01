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
    (no tzinfo) are forbidden. Tests scan codebase for datetime.utcnow() usage
    and fail if found (naive datetime anti-pattern).
"""
from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["epoch_ms_to_dt"]


def epoch_ms_to_dt(ms: int) -> datetime:
    """Convert an epoch timestamp (ms or s) to a timezone-aware UTC datetime.

    Heuristic: values < 10^12 treated as seconds.
    """
    if ms < 1_000_000_000_000:
        return datetime.fromtimestamp(ms, tz=timezone.utc)
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
