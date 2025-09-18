from __future__ import annotations

import os
from typing import Optional


def load_checkpoint(path: str) -> Optional[int]:
    """Load last processed execution id from path. Returns None if missing/invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return None
        return int(raw)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def store_checkpoint(path: str, execution_id: int) -> None:
    """Atomically store execution id to path."""
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(str(execution_id))
    os.replace(tmp_path, path)


__all__ = ["load_checkpoint", "store_checkpoint"]
