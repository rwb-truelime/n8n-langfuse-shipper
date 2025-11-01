"""Deterministic span and trace ID generation using UUIDv5.

This module provides the stable namespace and helper functions for generating
deterministic span IDs from execution and node metadata. UUIDv5 ensures identical
inputs produce identical IDs across runs, enabling idempotent replay and consistent
trace structures.

Constants:
    SPAN_NAMESPACE: UUIDv5 namespace derived from DNS namespace + seed string
        "n8n-langfuse-shipper-span". This value MUST NOT change as it would
        invalidate all historical span ID mappings.

ID Format:
    Span ID: UUIDv5(SPAN_NAMESPACE, f"{trace_id}:{node_name}:{run_index}")
    Root Span ID: UUIDv5(SPAN_NAMESPACE, f"{trace_id}:root")
    Trace ID: Raw execution.id as string (human-readable in Langfuse UI)

Public Functions:
    span_uuid: Generate span ID from trace ID, node name, and optional run index

Design Invariant:
    SPAN_NAMESPACE and ID seed format are immutable contracts. Changing either
    breaks determinism guarantee and requires major version bump + migration path.
"""
from __future__ import annotations

from uuid import NAMESPACE_DNS, uuid5

SPAN_NAMESPACE = uuid5(NAMESPACE_DNS, "n8n-langfuse-shipper-span")

__all__ = ["SPAN_NAMESPACE", "span_uuid"]


def span_uuid(trace_id: str, node_name: str, run_index: int | None = None) -> str:
    """Generate deterministic span UUIDv5 from trace ID, node name, and run index.

    Creates stable span identifiers using fixed namespace and composite seed string.
    Identical inputs always produce identical UUIDs, enabling idempotent replay.

    Seed format:
    - With run_index: f"{trace_id}:{node_name}:{run_index}"
    - Without run_index: f"{trace_id}:{node_name}"

    Args:
        trace_id: Trace identifier (typically str(execution.id))
        node_name: Workflow node name
        run_index: Optional run instance index within node (None for single-run nodes)

    Returns:
        UUIDv5 string representation
    """
    if run_index is None:
        name = f"{trace_id}:{node_name}"
    else:
        name = f"{trace_id}:{node_name}:{run_index}"
    return str(uuid5(SPAN_NAMESPACE, name))
