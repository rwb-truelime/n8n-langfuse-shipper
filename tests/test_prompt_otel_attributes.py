"""Test OTEL attribute emission for prompt metadata.

Verifies that prompt_name and prompt_version are correctly emitted as
OTEL attributes on generation spans with proper debug metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock

from n8n_langfuse_shipper.models.langfuse import LangfuseSpan
from n8n_langfuse_shipper.shipper import _apply_span_attributes


def test_prompt_attributes_on_generation_span():
    """Verify prompt attributes are set on generation spans."""
    span_model = LangfuseSpan(
        id="test-span-1",
        trace_id="trace-123",
        name="LLM Call",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="Sales Agent Prompt",
        prompt_version=58,
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    # Verify prompt attributes were set
    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    assert "langfuse.observation.prompt.name" in calls
    assert calls["langfuse.observation.prompt.name"] == "Sales Agent Prompt"

    assert "langfuse.observation.prompt.version" in calls
    assert calls["langfuse.observation.prompt.version"] == 58


def test_no_prompt_attributes_when_absent():
    """Verify prompt attributes NOT set when prompt_name/version absent."""
    span_model = LangfuseSpan(
        id="test-span-2",
        trace_id="trace-123",
        name="Non-Prompted LLM",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        # No prompt_name or prompt_version
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    # Verify prompt attributes were NOT set
    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    assert "langfuse.observation.prompt.name" not in calls
    assert "langfuse.observation.prompt.version" not in calls


def test_prompt_attributes_only_when_both_present():
    """Verify prompt attributes require BOTH name and version."""
    # Only name, no version
    span_model = LangfuseSpan(
        id="test-span-3",
        trace_id="trace-123",
        name="Partial Prompt",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="Partial",
        prompt_version=None,  # Missing version
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    # Should NOT set attributes (both required)
    assert "langfuse.observation.prompt.name" not in calls
    assert "langfuse.observation.prompt.version" not in calls


def test_prompt_attributes_on_non_generation_span():
    """Verify prompt attributes can be set on any observation type."""
    # Though resolution typically only happens for generations,
    # the shipper should accept prompt metadata on any span type
    span_model = LangfuseSpan(
        id="test-span-4",
        trace_id="trace-123",
        name="Agent Tool",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="span",  # Not generation
        prompt_name="Tool Prompt",
        prompt_version=10,
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    # Should still set prompt attributes
    assert calls["langfuse.observation.prompt.name"] == "Tool Prompt"
    assert calls["langfuse.observation.prompt.version"] == 10


def test_prompt_version_zero():
    """Verify version 0 is treated as valid (not None)."""
    span_model = LangfuseSpan(
        id="test-span-5",
        trace_id="trace-123",
        name="Version Zero",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="Zero Version Prompt",
        prompt_version=0,  # Zero is valid
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    # Should set attributes with version 0
    assert calls["langfuse.observation.prompt.name"] == "Zero Version Prompt"
    assert calls["langfuse.observation.prompt.version"] == 0


def test_prompt_attributes_alongside_model():
    """Verify prompt attributes coexist with model attributes."""
    span_model = LangfuseSpan(
        id="test-span-6",
        trace_id="trace-123",
        name="Full Generation",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        model="gpt-4o",
        prompt_name="Complex Prompt",
        prompt_version=42,
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    # Both model and prompt attributes should be present
    assert calls["model"] == "gpt-4o"
    assert calls["langfuse.observation.model.name"] == "gpt-4o"
    assert calls["langfuse.observation.prompt.name"] == "Complex Prompt"
    assert calls["langfuse.observation.prompt.version"] == 42


def test_prompt_attributes_with_metadata():
    """Verify prompt attributes set alongside observation metadata."""
    span_model = LangfuseSpan(
        id="test-span-7",
        trace_id="trace-123",
        name="With Metadata",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="Metadata Prompt",
        prompt_version=15,
        metadata={
            "n8n.prompt.original_version": 58,
            "n8n.prompt.resolved_version": 15,
            "n8n.prompt.resolution_method": "fallback_latest",
            "n8n.prompt.confidence": "high",
        },
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    # Prompt attributes should be present
    assert calls["langfuse.observation.prompt.name"] == "Metadata Prompt"
    assert calls["langfuse.observation.prompt.version"] == 15

    # Debug metadata should also be present
    assert calls["langfuse.observation.metadata.n8n.prompt.original_version"] == "58"
    assert calls["langfuse.observation.metadata.n8n.prompt.resolved_version"] == "15"
    assert calls["langfuse.observation.metadata.n8n.prompt.resolution_method"] == "fallback_latest"
    assert calls["langfuse.observation.metadata.n8n.prompt.confidence"] == "high"


def test_prompt_name_with_special_characters():
    """Verify prompt names with special characters are handled correctly."""
    span_model = LangfuseSpan(
        id="test-span-8",
        trace_id="trace-123",
        name="Special Chars",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="Sales Agent: Q4 Workflow (v2.0)",
        prompt_version=99,
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    # Should preserve special characters
    assert calls["langfuse.observation.prompt.name"] == "Sales Agent: Q4 Workflow (v2.0)"
    assert calls["langfuse.observation.prompt.version"] == 99


def test_large_version_number():
    """Verify large version numbers are handled correctly."""
    span_model = LangfuseSpan(
        id="test-span-9",
        trace_id="trace-123",
        name="Large Version",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="High Version Prompt",
        prompt_version=999999,
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    assert calls["langfuse.observation.prompt.version"] == 999999


def test_observation_type_preserved():
    """Verify observation type attribute is always set."""
    span_model = LangfuseSpan(
        id="test-span-10",
        trace_id="trace-123",
        name="Type Check",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="Test Prompt",
        prompt_version=1,
    )

    mock_span = Mock()
    _apply_span_attributes(mock_span, span_model)

    calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    # observation_type should always be first
    assert calls["langfuse.observation.type"] == "generation"


def test_multiple_spans_with_different_prompts():
    """Verify multiple spans can have different prompt metadata."""
    span1 = LangfuseSpan(
        id="span-1",
        trace_id="trace-123",
        name="First",
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="Prompt A",
        prompt_version=10,
    )

    span2 = LangfuseSpan(
        id="span-2",
        trace_id="trace-123",
        name="Second",
        start_time=datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 12, 0, 15, tzinfo=timezone.utc),
        observation_type="generation",
        prompt_name="Prompt B",
        prompt_version=20,
    )

    mock_span1 = Mock()
    mock_span2 = Mock()

    _apply_span_attributes(mock_span1, span1)
    _apply_span_attributes(mock_span2, span2)

    calls1 = {call[0][0]: call[0][1] for call in mock_span1.set_attribute.call_args_list}
    calls2 = {call[0][0]: call[0][1] for call in mock_span2.set_attribute.call_args_list}

    # Each span should have its own prompt metadata
    assert calls1["langfuse.observation.prompt.name"] == "Prompt A"
    assert calls1["langfuse.observation.prompt.version"] == 10

    assert calls2["langfuse.observation.prompt.name"] == "Prompt B"
    assert calls2["langfuse.observation.prompt.version"] == 20
