#!/usr/bin/env python3
"""Fail commit if NOTICE file missing or its required header block changed.

Rules:
 1. NOTICE must exist at repo root.
 2. First non-empty lines must contain project name and copyright line.
 3. Must reference Apache 2.0.

This is intentionally lightweight (no strict hashing) to allow appending third-party notices.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTICE_PATH = REPO_ROOT / "NOTICE"
CONFIG_PATH = REPO_ROOT / "notice_check.toml"


from typing import Any, Dict


def load_config() -> Dict[str, Any]:
    """Load configuration from notice_check.toml if present.

    Structure:
      [notice]
      required_substrings = ["..."]
      header_must_contain = "n8n-langfuse-shipper"
      header_search_chars = 200
    """
    default: Dict[str, Any] = {
        "required_substrings": [
            "n8n-langfuse-shipper",
            "Apache License, Version 2.0",
            "Copyright 2025 Rodger Blom",
        ],
        "header_must_contain": "n8n-langfuse-shipper",
        "header_search_chars": 200,
    }
    if not CONFIG_PATH.exists():
        return default
    try:
        import tomllib  # Python 3.11+

        with CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
        notice_cfg = data.get("notice", {}) if isinstance(data, dict) else {}
        # Merge defaults
        merged = default.copy()
        for k in ("required_substrings", "header_must_contain", "header_search_chars"):
            if k in notice_cfg:
                merged[k] = notice_cfg[k]
        # Basic validation
        if not isinstance(merged["required_substrings"], list) or not all(
            isinstance(s, str) for s in merged["required_substrings"]
        ):
            raise ValueError("required_substrings must be a list of strings")
        return merged
    except Exception as e:  # pragma: no cover - config parse robustness
        print(
            f"WARNING: Failed to parse {CONFIG_PATH.name}; falling back to defaults. Error: {e}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return default


def main() -> int:
    cfg = load_config()
    required_substrings = cfg["required_substrings"]
    header_must_contain = cfg["header_must_contain"]
    header_chars = int(cfg.get("header_search_chars", 200) or 200)

    if not NOTICE_PATH.exists():
        print("ERROR: NOTICE file is missing.", file=sys.stderr)
        return 1
    try:
        text = NOTICE_PATH.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:  # pragma: no cover
        print(f"ERROR: Unable to read NOTICE file: {e}", file=sys.stderr)
        return 1
    missing = [s for s in required_substrings if s not in text]
    if missing:
        print("ERROR: NOTICE file is missing required content:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "If this change is intentional (e.g., year rollover), update notice_check.toml (or adjust script)."
        )
        return 1
    # Basic header position sanity: set by config
    header_slice = text[:header_chars]
    if header_must_contain and header_must_contain not in header_slice:
        print(
            "ERROR: Required header token not found near top of NOTICE file (unexpected reformat).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
