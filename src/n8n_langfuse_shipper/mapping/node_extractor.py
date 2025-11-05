"""Node data extraction for AI-only filtering with metadata surfacing.

This module provides functionality to extract input/output data from specified
nodes during AI-only filtering, attaching the extracted data to root span
metadata for visibility in Langfuse UI. Extracted nodes would otherwise be
completely excluded from traces but may contain critical configuration,
business context, or intermediate results needed for debugging.

Extraction Features:
    - Multi-run support: Extract data from all runs of specified nodes
    - Wildcard filtering: Include/exclude keys via fnmatch glob patterns
    - Binary stripping: Reuse existing sanitization for safe serialization
    - Size limiting: Per-value length caps to prevent metadata bloat
    - Fail-open: Errors in extraction don't block trace export

Key Filtering Logic:
    1. Flatten nested dict to dotted paths (e.g., "headers.x-correlation-id")
    2. If include_patterns: only keep paths matching at least one pattern
    3. If exclude_patterns: remove paths matching any exclude pattern
    4. Apply size limits to string values
    5. Reconstruct nested structure from filtered paths

Metadata Structure:
    Root span metadata under n8n.extracted_nodes:
    {
        "<node_name>": {
            "runs": [
                {
                    "run_index": 0,
                    "execution_status": "success",
                    "input": {...},
                    "output": {...},
                    "_truncated": false
                }
            ]
        },
        "_meta": {
            "extracted_count": 2,
            "nodes_requested": 3,
            "nodes_not_found": ["MissingNode"],
            "extraction_config": {...}
        }
    }

Public Functions:
    extract_nodes_data: Main extraction orchestrator
    matches_pattern: Wildcard pattern matching for key filtering

Design Invariants:
    - Pure functions: no side effects beyond logging
    - Deterministic: same input + config = same output structure
    - Binary stripping always applied before size limiting
    - Empty filtering results include null values for visibility
    - Serialization errors fail-open with logged warnings
"""
from __future__ import annotations

import fnmatch
import json
import logging
from typing import Any

from n8n_langfuse_shipper.mapping.binary_sanitizer import strip_binary_payload
from n8n_langfuse_shipper.models.n8n import NodeRun

logger = logging.getLogger(__name__)

__all__ = [
    "extract_nodes_data",
    "matches_pattern",
]


def matches_pattern(key: str, patterns: list[str]) -> bool:
    """Check if a key matches any wildcard pattern in the list.

    Uses fnmatch for glob-style pattern matching (*, ?, [seq], [!seq]).
    Case-sensitive matching.

    Args:
        key: Dotted key path to test (e.g., "headers.x-correlation-id")
        patterns: List of glob patterns (e.g., ["*Id", "headers.*"])

    Returns:
        True if key matches at least one pattern, False otherwise

    Examples:
        >>> matches_pattern("userId", ["*Id"])
        True
        >>> matches_pattern("headers.x-token", ["headers.*"])
        True
        >>> matches_pattern("password", ["*password*"])
        True
        >>> matches_pattern("userName", ["*Id"])
        False
    """
    if not patterns:
        return False
    return any(fnmatch.fnmatch(key, pattern) for pattern in patterns)


def _flatten_dict(
    data: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten nested dict into dotted-path key-value pairs.

    Args:
        data: Nested dictionary to flatten
        parent_key: Prefix for current level keys
        sep: Separator for path components

    Returns:
        Flattened dict with dotted paths as keys

    Examples:
        >>> _flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
        >>> _flatten_dict({"headers": {"x-id": "123"}})
        {"headers.x-id": "123"}
    """
    items: list[tuple[str, Any]] = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Keep lists as-is at this level (don't flatten list indices)
            items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(flat_dict: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Reconstruct nested dict from flattened dotted-path keys.

    Args:
        flat_dict: Flattened dict with dotted path keys
        sep: Separator used in path components

    Returns:
        Nested dict reconstructed from flat paths

    Examples:
        >>> _unflatten_dict({"a.b": 1, "a.c": 2})
        {"a": {"b": 1, "c": 2}}
    """
    result: dict[str, Any] = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def _apply_size_limit(value: Any, max_len: int) -> tuple[Any, bool]:
    """Apply size limit to string values, truncating with marker if needed.

    Args:
        value: Value to check and potentially truncate
        max_len: Maximum string length allowed

    Returns:
        Tuple of (potentially truncated value, was_truncated boolean)
    """
    if not isinstance(value, str):
        return value, False
    if len(value) <= max_len:
        return value, False
    truncated = value[:max_len] + "... [truncated]"
    return truncated, True


def _filter_and_limit_data(
    data: Any,
    include_patterns: list[str],
    exclude_patterns: list[str],
    max_value_len: int,
) -> tuple[Any, bool]:
    """Filter keys and apply size limits to extracted data.

    Args:
        data: Input/output data to filter and limit
        include_patterns: Patterns for keys to include (empty = all)
        exclude_patterns: Patterns for keys to exclude
        max_value_len: Maximum string length per value

    Returns:
        Tuple of (filtered/limited data, was_any_truncated boolean)
    """
    if data is None:
        return None, False

    # Non-dict passthrough with size limiting
    if not isinstance(data, dict):
        return _apply_size_limit(data, max_value_len)

    # Flatten, filter, limit, unflatten
    flat = _flatten_dict(data)
    any_truncated = False

    # Apply include filter
    if include_patterns:
        flat = {k: v for k, v in flat.items() if matches_pattern(k, include_patterns)}

    # Apply exclude filter
    if exclude_patterns:
        flat = {k: v for k, v in flat.items() if not matches_pattern(k, exclude_patterns)}

    # Apply size limits
    limited_flat = {}
    for k, v in flat.items():
        limited_v, truncated = _apply_size_limit(v, max_value_len)
        limited_flat[k] = limited_v
        if truncated:
            any_truncated = True

    # Reconstruct nested structure
    if not limited_flat:
        return None, any_truncated

    return _unflatten_dict(limited_flat), any_truncated


def _extract_single_node_runs(
    node_name: str,
    runs: list[NodeRun],
    include_patterns: list[str],
    exclude_patterns: list[str],
    max_value_len: int,
) -> dict[str, Any]:
    """Extract data from all runs of a single node.

    Args:
        node_name: Name of node being extracted
        runs: List of NodeRun instances for this node
        include_patterns: Key patterns to include
        exclude_patterns: Key patterns to exclude
        max_value_len: Max length per value

    Returns:
        Dict with "runs" list containing extracted data for each run
    """
    extracted_runs = []

    for run_idx, node_run in enumerate(runs):
        # Apply binary stripping first (unconditional)
        input_data = strip_binary_payload(
            node_run.inputOverride if node_run.inputOverride else None
        )
        output_data = strip_binary_payload(node_run.data)

        # Filter and limit
        filtered_input, input_truncated = _filter_and_limit_data(
            input_data, include_patterns, exclude_patterns, max_value_len
        )
        filtered_output, output_truncated = _filter_and_limit_data(
            output_data, include_patterns, exclude_patterns, max_value_len
        )

        run_entry = {
            "run_index": run_idx,
            "execution_status": node_run.executionStatus,
            "input": filtered_input,
            "output": filtered_output,
            "_truncated": input_truncated or output_truncated,
        }

        extracted_runs.append(run_entry)

    return {"runs": extracted_runs}


def extract_nodes_data(
    run_data: dict[str, list[NodeRun]],
    extraction_nodes: list[str],
    include_patterns: list[str],
    exclude_patterns: list[str],
    max_value_len: int,
) -> dict[str, Any]:
    """Extract input/output data from specified nodes for root span metadata.

    Main orchestrator for node extraction feature. Called during AI-only
    filtering before excluded nodes are dropped from trace structure.

    Args:
        run_data: Complete runData dict (node_name -> list[NodeRun])
        extraction_nodes: List of node names to extract data from
        include_patterns: Wildcard patterns for keys to include (empty = all)
        exclude_patterns: Wildcard patterns for keys to exclude
        max_value_len: Maximum string length per value (default 10KB)

    Returns:
        Dict suitable for root span metadata under n8n.extracted_nodes key.
        Contains extracted node data plus _meta section with extraction info.

    Example:
        >>> run_data = {"WebhookTrigger": [node_run1], "DataTransform": [node_run2]}
        >>> result = extract_nodes_data(
        ...     run_data,
        ...     extraction_nodes=["WebhookTrigger", "MissingNode"],
        ...     include_patterns=["userId"],
        ...     exclude_patterns=["*password*"],
        ...     max_value_len=10000,
        ... )
        >>> result.keys()
        dict_keys(['WebhookTrigger', '_meta'])
        >>> result["_meta"]["nodes_not_found"]
        ["MissingNode"]
    """
    if not extraction_nodes:
        logger.debug("No extraction nodes configured, skipping extraction")
        return {}

    logger.info(
        f"Extracting data from {len(extraction_nodes)} nodes: {extraction_nodes}"
    )

    extracted_data: dict[str, Any] = {}
    nodes_not_found = []

    for node_name in extraction_nodes:
        if node_name not in run_data:
            logger.warning(
                f"Extraction node '{node_name}' not found in execution runData"
            )
            nodes_not_found.append(node_name)
            continue

        runs = run_data[node_name]
        if not runs:
            logger.debug(f"Node '{node_name}' has no runs, skipping")
            continue

        try:
            node_extracted = _extract_single_node_runs(
                node_name,
                runs,
                include_patterns,
                exclude_patterns,
                max_value_len,
            )
            extracted_data[node_name] = node_extracted
            logger.debug(
                f"Extracted {len(node_extracted['runs'])} runs from '{node_name}'"
            )
        except Exception as e:
            logger.error(
                f"Failed to extract data from node '{node_name}': {e}",
                exc_info=True,
            )
            # Fail-open: continue with other nodes
            continue

    # Build metadata section
    meta: dict[str, Any] = {
        "extracted_count": len(extracted_data),
        "nodes_requested": len(extraction_nodes),
    }

    if nodes_not_found:
        meta["nodes_not_found"] = nodes_not_found

    # Include extraction config for transparency
    config_info: dict[str, Any] = {}
    if include_patterns:
        config_info["include_keys"] = include_patterns
    if exclude_patterns:
        config_info["exclude_keys"] = exclude_patterns
    if config_info:
        meta["extraction_config"] = config_info

    extracted_data["_meta"] = meta

    logger.info(
        f"Extraction complete: {meta['extracted_count']}/{meta['nodes_requested']} "
        f"nodes extracted"
    )

    return extracted_data
