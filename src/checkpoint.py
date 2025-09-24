"""Checkpoint management for idempotent execution processing.

This module provides simple, file-based checkpointing to track the last
successfully processed n8n execution ID. This allows the backfill process to
be stopped and resumed without reprocessing data.

The `store_checkpoint` function uses an atomic write pattern (write to a
temporary file then rename) to prevent checkpoint corruption if the process is
interrupted.
"""
from __future__ import annotations

import os
from typing import Optional


def load_checkpoint(path: str) -> Optional[int]:
    """Load the last processed execution ID from a checkpoint file.

    This function reads an integer ID from the specified file path. It safely
    handles cases where the file does not exist, is empty, or contains invalid
    data by returning None.

    Args:
        path: The path to the checkpoint file.

    Returns:
        The integer execution ID if the file is valid, otherwise None.
    """
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
    """Atomically store an execution ID to the checkpoint file.

    This function writes the given execution ID to the specified path. It
    performs an atomic write by first writing to a temporary file and then
    renaming it to the final path. This prevents the checkpoint file from
    becoming corrupted if the process is terminated during the write.

    Args:
        path: The path to the checkpoint file.
        execution_id: The last successfully processed execution ID to store.
    """
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(str(execution_id))
    os.replace(tmp_path, path)


__all__ = ["load_checkpoint", "store_checkpoint"]
