from __future__ import annotations

from src.shipper import _build_human_trace_id


def test_trace_id_embedding_simple():
    int_id, hex_id = _build_human_trace_id("12345")
    assert hex_id.endswith("12345")
    assert len(hex_id) == 32
    assert int(hex_id, 16) == int_id


def test_trace_id_embedding_strips_non_digits():
    int_id, hex_id = _build_human_trace_id("exec-789")
    assert hex_id.endswith("789")
    assert len(hex_id) == 32


def test_trace_id_embedding_long_digits_truncated():
    long_digits = "9" * 40
    int_id, hex_id = _build_human_trace_id(long_digits)
    assert len(hex_id) == 32
    # Should be last 32 of original
    assert hex_id.endswith("9" * 32)
    assert int(hex_id, 16) == int_id


def test_trace_id_embedding_when_no_digits():
    int_id, hex_id = _build_human_trace_id("no-digits-here")
    # Should fallback to '0' then pad
    assert hex_id.endswith("0")
    assert len(hex_id) == 32
    assert int(hex_id, 16) == int_id
