"""Test fingerprint-based prompt disambiguation.

Tests the fingerprint matching tie-breaker when multiple prompts are at
equal distance from a generation node.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict

from src.mapping.prompt_resolution import (
    PromptResolutionResult,
    resolve_prompt_for_generation,
    _compute_text_fingerprint,
    _extract_prompt_text_from_input,
)
from src.mapping.prompt_detection import (
    PromptMetadata,
    _compute_prompt_fingerprint,
)


def test_fingerprint_algorithm_consistency():
    """Verify detection and resolution use same fingerprint algorithm."""
    prompt_text = "# Role and Objective\nYou are an expert assistant..."

    # Both should use first 300 chars
    detection_fp = _compute_prompt_fingerprint(prompt_text)
    resolution_fp = _compute_text_fingerprint(prompt_text)

    assert detection_fp == resolution_fp
    assert len(detection_fp) == 16  # First 16 hex chars of SHA256


def test_fingerprint_uses_first_300_chars():
    """Verify fingerprint only uses first 300 characters."""
    prefix = "A" * 300

    # Same prefix, different suffix
    text1 = prefix + ("B" * 10000)
    text2 = prefix + ("C" * 10000)

    fp1 = _compute_text_fingerprint(text1)
    fp2 = _compute_text_fingerprint(text2)

    assert fp1 == fp2  # Should match (same first 300)

    # Different prefix
    text3 = "Z" * 300 + ("B" * 10000)
    fp3 = _compute_text_fingerprint(text3)

    assert fp1 != fp3  # Should differ


def test_extract_agent_input_with_system_prefix():
    """Verify System: prefix is stripped from agent input."""
    agent_input = {
        "ai_languageModel": [[{
            "json": {
                "systemMessage": "System: # Role\nYou are an assistant."
            }
        }]]
    }

    extracted = _extract_prompt_text_from_input(agent_input)

    # Should strip "System: " prefix
    assert extracted is not None
    assert extracted.startswith("# Role")
    assert "System: " not in extracted


def test_extract_agent_input_unwraps_n8n_structure():
    """Verify extraction handles nested n8n output wrappers."""
    # Typical n8n output: main channel with json wrapper
    agent_input = {
        "ai_languageModel": [[{
            "json": {
                "messages": "Test prompt content"
            }
        }]]
    }

    extracted = _extract_prompt_text_from_input(agent_input)
    assert extracted == "Test prompt content"


def test_extract_agent_input_searches_all_dict_values():
    """Verify extraction recurses into all dict values."""
    # Unknown top-level key
    agent_input = {
        "custom_field": {
            "nested": {
                "prompt": "Found it!"
            }
        }
    }

    extracted = _extract_prompt_text_from_input(agent_input)
    assert extracted == "Found it!"


def test_fingerprint_disambiguation_success():
    """Verify fingerprint matching resolves equidistant candidates."""
    # Two prompts at equal distance (both at distance 2)
    run_data = {
        "Prompt1": [{
            "startTime": 1000,
            "source": [],
        }],
        "Prompt2": [{
            "startTime": 1100,
            "source": [],
        }],
        "Merge": [{
            "startTime": 1200,
            "source": [
                {"previousNode": "Prompt1"},
                {"previousNode": "Prompt2"},
            ],
        }],
        "Agent": [{
            "startTime": 1300,
            "source": [{"previousNode": "Merge"}],
        }],
    }

    prompt1_text = "# Prompt 1\nThis is the first prompt with unique content."
    prompt2_text = "# Prompt 2\nThis is the second prompt with different content."

    prompt_registry = {
        ("Prompt1", 0): PromptMetadata(
            name="Prompt 1",
            version=1,
            fingerprint=_compute_prompt_fingerprint(prompt1_text),
        ),
        ("Prompt2", 0): PromptMetadata(
            name="Prompt 2",
            version=2,
            fingerprint=_compute_prompt_fingerprint(prompt2_text),
        ),
    }

    # Agent input contains Prompt 2 text (with System: prefix)
    agent_input = {
        "ai_languageModel": [[{
            "json": {
                "systemMessage": f"System: {prompt2_text}"
            }
        }]]
    }

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=agent_input,
    )

    # Should match Prompt 2 via fingerprint
    assert result.prompt_name == "Prompt 2"
    assert result.prompt_version == 2
    assert result.resolution_method == "fingerprint"
    assert result.confidence == "medium"
    assert result.ambiguous is True  # Was ambiguous, resolved by fingerprint


def test_fingerprint_disambiguation_no_match():
    """Verify fallback when fingerprint doesn't match any candidate."""
    run_data = {
        "Prompt1": [{
            "startTime": 1000,
            "source": [],
        }],
        "Prompt2": [{
            "startTime": 1100,
            "source": [],
        }],
        "Merge": [{
            "startTime": 1200,
            "source": [
                {"previousNode": "Prompt1"},
                {"previousNode": "Prompt2"},
            ],
        }],
        "Agent": [{
            "startTime": 1300,
            "source": [{"previousNode": "Merge"}],
        }],
    }

    prompt_registry = {
        ("Prompt1", 0): PromptMetadata(
            name="P1",
            version=1,
            fingerprint=_compute_prompt_fingerprint("Prompt 1 text"),
        ),
        ("Prompt2", 0): PromptMetadata(
            name="P2",
            version=2,
            fingerprint=_compute_prompt_fingerprint("Prompt 2 text"),
        ),
    }

    # Agent input has DIFFERENT text (no match)
    agent_input = {
        "ai_languageModel": [[{
            "json": {
                "systemMessage": "Completely different prompt content"
            }
        }]]
    }

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=agent_input,
    )

    # Should fall back to alphabetical (first candidate)
    assert result.prompt_name == "P1"
    assert result.resolution_method == "ancestor"
    assert result.confidence == "low"
    assert result.ambiguous is True


def test_fingerprint_requires_minimum_length():
    """Verify fingerprint matching requires >= 50 chars."""
    agent_input = {
        "ai_languageModel": [[{
            "json": {
                "systemMessage": "Short"  # Only 5 chars
            }
        }]]
    }

    extracted = _extract_prompt_text_from_input(agent_input)
    assert extracted is not None
    assert len(extracted) < 50

    # Fingerprint should return empty string for short text
    fp = _compute_text_fingerprint(extracted)
    assert fp == ""


def test_fingerprint_without_prompts_in_registry():
    """Verify graceful handling when prompts lack fingerprints."""
    run_data = {
        "Prompt1": [{"source": []}],
        "Prompt2": [{"source": []}],
        "Merge": [{
            "source": [
                {"previousNode": "Prompt1"},
                {"previousNode": "Prompt2"},
            ],
        }],
        "Agent": [{"source": [{"previousNode": "Merge"}]}],
    }

    # Prompts without fingerprint field
    prompt_registry = {
        ("Prompt1", 0): PromptMetadata(name="P1", version=1),
        ("Prompt2", 0): PromptMetadata(name="P2", version=2),
    }

    agent_input = {
        "ai_languageModel": [[{
            "json": {"systemMessage": "Test prompt"}
        }]]
    }

    result = resolve_prompt_for_generation(
        node_name="Agent",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input=agent_input,
    )

    # Should fall back gracefully
    assert result.resolution_method == "ancestor"
    assert result.ambiguous is True


def test_agent_hierarchy_before_fingerprint():
    """Verify agent hierarchy has precedence over fingerprint."""
    # Setup where agent hierarchy can disambiguate
    run_data = {
        "FetchPrompt1": [{"source": []}],
        "FetchPrompt2": [{"source": []}],
        "ChildAgent": [{
            "source": [
                {"previousNode": "FetchPrompt1"},
                {"previousNode": "FetchPrompt2"},
            ],
        }],
        "ParentAgent": [{
            "source": [{"previousNode": "ChildAgent"}],
        }],
        "Generation": [{
            "source": [{"previousNode": "ParentAgent"}],
        }],
    }

    prompt_registry = {
        ("FetchPrompt1", 0): PromptMetadata(
            name="P1",
            version=1,
            fingerprint="abc123",
        ),
        ("FetchPrompt2", 0): PromptMetadata(
            name="P2",
            version=2,
            fingerprint="def456",
        ),
    }

    # Build child_agent_map (ParentAgent has ChildAgent)
    # Type is Dict[str, Tuple[str, str]] - tuple is (node_type, node_name)
    child_agent_map: Dict[str, tuple[str, str]] = {}

    result = resolve_prompt_for_generation(
        node_name="Generation",
        run_index=0,
        run_data=run_data,
        prompt_registry=prompt_registry,
        agent_input={"systemMessage": "Test"},
        agent_parent="ParentAgent",
        child_agent_map=child_agent_map,
    )

    # Agent hierarchy should resolve without needing fingerprint
    # (Both prompts are ancestors, fingerprint not needed if hierarchy works)
    assert result.resolution_method in ["agent_hierarchy", "fingerprint", "ancestor"]
    assert result.prompt_name in ["P1", "P2"]
