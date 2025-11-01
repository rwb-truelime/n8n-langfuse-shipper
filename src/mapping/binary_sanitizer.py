"""Binary data detection and redaction for size-controlled span outputs.

This module provides unconditional binary stripping that applies regardless of
truncation settings. Detects base64-encoded data and binary objects in canonical
run.data.binary blocks and standalone contexts, replacing with stable placeholders
that preserve metadata without large payloads.

Detection Heuristics:
    - Base64 regex: ^[A-Za-z0-9+/]{200,}={0,2}$
    - JPEG marker: starts with /9j/
    - Diversity check: <4 unique chars in first 120 suggests false positive
    - Context validation: Requires file-indicative keys unless in binary block

Redaction Strategy:
    Canonical binary objects:
        - Replace data field with "binary omitted"
        - Attach _omitted_len with original length
        - Retain metadata (mimeType, fileName, etc.)

    Standalone base64 strings:
        - Replace with dict: {"_binary": true, "note": "binary omitted", "_omitted_len": N}

Traversal:
    - Bounded depth 25 to prevent stack overflow
    - Recursive scan through dicts, lists, and nested structures
    - Fail-open on detection errors (continue without stripping)

Public Functions:
    likely_binary_b64: Heuristic base64 detection with context awareness
    contains_binary_marker: Check if structure contains any binary data
    strip_binary_payload: Recursively replace binary with placeholders

Design Invariant:
    Binary stripping ALWAYS applies. Truncation is independent and optional.
    Tests assert this invariant; changing behavior breaks contract.
"""
from __future__ import annotations

import re
from typing import Any, Dict

__all__ = [
    "likely_binary_b64",
    "contains_binary_marker",
    "strip_binary_payload",
]

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{200,}={0,2}$")


def likely_binary_b64(
    s: str,
    *,
    context_key: str | None = None,
    in_binary_block: bool = False,
) -> bool:
    if len(s) < 200:
        return False
    if not _BASE64_RE.match(s) and not s.startswith("/9j/"):
        return False
    diversity = len(set(s[:120]))
    if diversity < 4 and not in_binary_block:
        lowered = (context_key or "").lower()
        if not any(h in lowered for h in ("data", "base64", "file", "binary")):
            return False
    return True


def contains_binary_marker(d: Any, depth: int = 0) -> bool:
    if depth > 25:
        return False
    try:
        if isinstance(d, dict):
            if "binary" in d and isinstance(d["binary"], dict):
                for v in d["binary"].values():
                    if isinstance(v, dict):
                        data_section = v.get("data") if isinstance(v.get("data"), dict) else v
                        if isinstance(data_section, dict):
                            mime = (
                                data_section.get("mimeType")
                                or data_section.get("mime_type")
                            )
                            b64 = data_section.get("data")
                            if (
                                mime
                                and isinstance(b64, str)
                                and (
                                    len(b64) > 200
                                    and (
                                        _BASE64_RE.match(b64)
                                        or b64.startswith("/9j/")
                                    )
                                )
                            ):
                                return True
            for val in d.values():
                if contains_binary_marker(val, depth + 1):
                    return True
        elif isinstance(d, list):
            for item in d[:25]:
                if contains_binary_marker(item, depth + 1):
                    return True
        elif isinstance(d, str):
            if likely_binary_b64(d):
                return True
    except Exception:
        return False
    return False


def strip_binary_payload(obj: Any) -> Any:
    placeholder_text = "binary omitted"
    try:
        if isinstance(obj, dict):
            new_d: Dict[str, Any] = {}
            for k, v in obj.items():
                if k == "binary" and isinstance(v, dict):
                    new_bin: Dict[str, Any] = {}
                    for bk, bv in v.items():
                        if isinstance(bv, dict):
                            bv2 = dict(bv)
                            data_section = bv2.get("data")
                            if (
                                isinstance(data_section, dict)
                                and isinstance(data_section.get("data"), str)
                            ):
                                ds = data_section.copy()
                                raw_b64 = ds.get("data", "")
                                if len(raw_b64) > 64:
                                    ds["data"] = placeholder_text
                                    ds["_omitted_len"] = len(raw_b64)
                                    bv2["data"] = ds
                            elif (
                                isinstance(data_section, str)
                                and (
                                    len(data_section) > 64
                                    and (
                                        _BASE64_RE.match(data_section)
                                        or data_section.startswith("/9j/")
                                    )
                                )
                            ):
                                bv2["_omitted_len"] = len(data_section)
                                bv2["data"] = placeholder_text
                            new_bin[bk] = bv2
                        else:
                            new_bin[bk] = bv
                    new_d[k] = new_bin
                elif isinstance(v, str) and likely_binary_b64(v, context_key=k):
                    new_d[k] = {
                        "_binary": True,
                        "note": placeholder_text,
                        "_omitted_len": len(v),
                    }
                else:
                    new_d[k] = strip_binary_payload(v)
            return new_d
        if isinstance(obj, list):
            out_list = []
            for x in obj:
                if isinstance(x, str) and likely_binary_b64(x):
                    out_list.append(
                        {
                            "_binary": True,
                            "note": placeholder_text,
                            "_omitted_len": len(x),
                        }
                    )
                else:
                    out_list.append(strip_binary_payload(x))
            return out_list
    except Exception:
        return obj
    return obj
