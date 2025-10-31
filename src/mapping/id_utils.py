"""Deterministic ID utilities (UUIDv5 span namespace extraction)."""
from __future__ import annotations
from uuid import NAMESPACE_DNS, uuid5, UUID

SPAN_NAMESPACE = uuid5(NAMESPACE_DNS, "n8n-langfuse-shipper-span")

__all__ = ["SPAN_NAMESPACE", "span_uuid"]


def span_uuid(trace_id: str, node_name: str, run_index: int | None = None) -> str:
    """Generate deterministic span UUIDv5 consistent with legacy logic."""
    if run_index is None:
        name = f"{trace_id}:{node_name}"
    else:
        name = f"{trace_id}:{node_name}:{run_index}"
    return str(uuid5(SPAN_NAMESPACE, name))
