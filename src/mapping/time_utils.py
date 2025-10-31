"""Time conversion and normalization helpers (extracted from mapper)."""
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
