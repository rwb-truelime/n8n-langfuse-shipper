"""Test prompt resolution via ancestor chain traversal.

Tests ancestor walk logic, distance calculation, multiple candidate handling,
and fingerprint-based disambiguation.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List

from n8n_langfuse_shipper.mapping.prompt_resolution import (
    PromptResolutionResult,
    resolve_prompt_for_generation,
    _compute_text_fingerprint,
    _extract_ancestor_chain,
)
from n8n_langfuse_shipper.mapping.prompt_detection import PromptMetadata


def test_resolve_immediate_parent_prompt():
    """Verify resolution of prompt in immediate parent (distance=1)."""
    # Generation node → Prompt fetch node
    run_data = {
        "Fetch Prompt": [{
            "startTime": 1000,
            "executionTime": 50,
            "executionStatus": "success",
            "data": {},
            "source": [],
        }],
        "LLM Call": [{
            "startTime": 2000,
            "executionTime": 500,
            "executionStatus": "success",
            "data": {},
            "source": [{"previousNode": "Fetch Prompt", "previousNodeRun": 0}],
        }],
    }

    prompt_registry = {
        ("Fetch Prompt", 0): PromptMetadata(
            name="Test Prompt",
            version=10,
            type="chat",
        ),
    }

    result = resolve_prompt_for_generation(
        node_name="LLM Call",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=None,
    )

    assert result.prompt_name == "Test Prompt"
    assert result.prompt_version == 10
    assert result.resolution_method == "ancestor"
    assert result.confidence == "high"
    assert result.ancestor_distance == 1
    assert result.candidate_count == 1
    assert result.ambiguous is False


def test_resolve_distant_ancestor():
    """Verify resolution across multiple hops (5+ distance)."""
    # Agent → Preprocessor → Formatter → Router → Extractor → Prompt
    run_data = {
        "Prompt": [{
            "startTime": 1000,
            "source": [],
        }],
        "Extractor": [{
            "startTime": 1100,
            "source": [{"previousNode": "Prompt"}],
        }],
        "Router": [{
            "startTime": 1200,
            "source": [{"previousNode": "Extractor"}],
        }],
        "Formatter": [{
            "startTime": 1300,
            "source": [{"previousNode": "Router"}],
        }],
        "Preprocessor": [{
            "startTime": 1400,
            "source": [{"previousNode": "Formatter"}],
        }],
        "Agent": [{
            "startTime": 1500,
            "source": [{"previousNode": "Preprocessor"}],
        }],
    }

    prompt_registry = {
        ("Prompt", 0): PromptMetadata(
            name="Distant Prompt",
            version=42,
        ),
    }

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=None,
    )

    assert result.prompt_name == "Distant Prompt"
    assert result.prompt_version == 42
    assert result.ancestor_distance == 5
    assert result.confidence == "high"


def test_resolve_no_prompt_found():
    """Verify fallback when no prompt ancestor exists."""
    run_data = {
        "Start": [{"source": []}],
        "Agent": [{"source": [{"previousNode": "Start"}]}],
    }

    prompt_registry = {}  # No prompts

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=None,
    )

    assert result.prompt_name is None
    assert result.prompt_version is None
    assert result.resolution_method == "none"
    assert result.confidence == "none"
    assert result.candidate_count == 0


def test_resolve_closest_among_multiple():
    """Verify closest prompt selected when multiple ancestors."""
    # Agent → Processing → [Prompt A (close), Prompt B (far)]
    run_data = {
        "Prompt B": [{
            "startTime": 1000,
            "source": [],
        }],
        "Intermediate": [{
            "startTime": 1100,
            "source": [{"previousNode": "Prompt B"}],
        }],
        "Prompt A": [{
            "startTime": 1200,
            "source": [{"previousNode": "Intermediate"}],
        }],
        "Processing": [{
            "startTime": 1300,
            "source": [{"previousNode": "Prompt A"}],
        }],
        "Agent": [{
            "startTime": 1400,
            "source": [{"previousNode": "Processing"}],
        }],
    }

    prompt_registry = {
        ("Prompt A", 0): PromptMetadata(name="Close Prompt", version=20),
        ("Prompt B", 0): PromptMetadata(name="Far Prompt", version=10),
    }

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=None,
    )

    # Should pick closest (Prompt A, distance=2)
    assert result.prompt_name == "Close Prompt"
    assert result.prompt_version == 20
    assert result.ancestor_distance == 2
    assert result.candidate_count == 2
    assert result.ambiguous is True  # Multiple candidates detected
    assert result.confidence == "medium"


def test_compute_fingerprint_short_text():
    """Verify fingerprint of short text (< 50 chars returns empty)."""
    text = "Short text."
    fp = _compute_text_fingerprint(text)

    # Too short, should return empty string
    assert fp == ""

    # Sufficient length (>= 50 chars)
    long_text = "This is a longer prompt text with more than fifty characters."
    fp_long = _compute_text_fingerprint(long_text)
    assert len(fp_long) == 16  # First 16 chars of SHA256


def test_compute_fingerprint_long_text():
    """Verify fingerprint uses first 300 chars only."""
    # First 300 chars are the same
    prefix = "A" * 300

    text1 = prefix + ("B" * 10000)
    text2 = prefix + ("C" * 10000)

    fp1 = _compute_text_fingerprint(text1)
    fp2 = _compute_text_fingerprint(text2)

    # Should match (same first 300 chars)
    assert fp1 == fp2
    assert len(fp1) == 16

    # Different prefix should produce different fingerprint
    text3 = ("Z" * 300) + ("B" * 10000)
    fp3 = _compute_text_fingerprint(text3)
    assert fp1 != fp3


def test_extract_ancestor_chain_simple():
    """Verify basic ancestor chain extraction."""
    run_data = {
        "A": [{"source": []}],
        "B": [{"source": [{"previousNode": "A"}]}],
        "C": [{"source": [{"previousNode": "B"}]}],
    }

    chain = _extract_ancestor_chain("C", 0, run_data)

    # Chain should be: C → B → A
    assert len(chain) == 2  # Excludes starting node
    assert chain[0] == ("B", 0, 1)  # (node, run_idx, distance)
    assert chain[1] == ("A", 0, 2)


def test_extract_ancestor_chain_with_run_indices():
    """Verify chain extraction with specific run indices."""
    run_data = {
        "A": [
            {"source": []},  # run 0
            {"source": []},  # run 1
        ],
        "B": [
            {"source": [{"previousNode": "A", "previousNodeRun": 1}]},
        ],
    }

    chain = _extract_ancestor_chain("B", 0, run_data)

    assert len(chain) == 1
    assert chain[0] == ("A", 1, 1)  # Should link to run 1


def test_extract_ancestor_chain_cycle_protection():
    """Verify cycle detection prevents infinite loops."""
    run_data = {
        "A": [{"source": [{"previousNode": "B"}]}],  # Cycle!
        "B": [{"source": [{"previousNode": "A"}]}],
    }

    # Should not hang; returns partial chain
    chain = _extract_ancestor_chain("A", 0, run_data, max_depth=10)

    # Should stop at cycle detection
    assert len(chain) <= 10


def test_resolve_with_multiple_runs():
    """Verify resolution handles nodes with multiple runs."""
    run_data = {
        "Prompt": [
            {"source": []},  # run 0
            {"source": []},  # run 1 (different version)
        ],
        "Agent": [
            {"source": [{"previousNode": "Prompt", "previousNodeRun": 1}]},
        ],
    }

    prompt_registry = {
        ("Prompt", 0): PromptMetadata(name="V1", version=5),
        ("Prompt", 1): PromptMetadata(name="V2", version=6),
    }

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=None,
    )

    # Should resolve to run 1 (V2)
    assert result.prompt_name == "V2"
    assert result.prompt_version == 6


def test_resolve_ambiguous_metadata():
    """Verify ambiguous flag and candidate count when multiple prompts."""
    run_data = {
        "Prompt1": [{"source": []}],
        "Prompt2": [{"source": []}],
        "Merge": [{
            "source": [
                {"previousNode": "Prompt1"},
                {"previousNode": "Prompt2"},
            ]
        }],
        "Agent": [{"source": [{"previousNode": "Merge"}]}],
    }

    prompt_registry = {
        ("Prompt1", 0): PromptMetadata(name="P1", version=1),
        ("Prompt2", 0): PromptMetadata(name="P2", version=2),
    }

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=None,
    )

    assert result.ambiguous is True
    assert result.candidate_count == 2
    assert result.confidence == "low"  # Equidistant → low confidence fallback
    # Should pick one of them (first by sort order when equidistant)
    assert result.prompt_name in ["P1", "P2"]


def test_resolve_missing_source_field():
    """Verify graceful handling when source field missing."""
    run_data = {
        "Prompt": [{}],  # No source field
        "Agent": [{}],  # No source field
    }

    prompt_registry = {
        ("Prompt", 0): PromptMetadata(name="Test", version=1),
    }

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=None,
    )

    # Should fail gracefully
    assert result.resolution_method == "none"
    assert result.prompt_name is None


def test_resolve_empty_run_data():
    """Verify handling of empty/missing run_data."""
    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data={},
        prompt_registry={},
        agent_input=None,
    )

    assert result.resolution_method == "none"
    assert result.candidate_count == 0
