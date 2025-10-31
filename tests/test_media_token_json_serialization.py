"""Test that media tokens are properly JSON-serialized in OTLP output.

This test ensures that the shipper correctly serializes span input/output
containing media tokens as JSON strings (not Python str() representation),
so the Langfuse UI can detect and render the media tokens.

Regression guard for: STILL NO PREVIEWS IN LANGFUSE issue.
Root cause: str(child.output) double-stringified media tokens, preventing UI parsing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.config import Settings
from src.models.langfuse import LangfuseSpan, LangfuseTrace
from src.shipper import export_trace


def test_media_token_json_serialized_in_output():
    """Verify media token in span output is JSON-serialized (not str-mangled)."""
    # Build a trace with a span containing a media token in output
    media_token = (
        "@@@langfuseMedia:type=image/jpeg|"
        "id=MwoGlsMS6lW8ijWeRyZKfD|source=base64_data_uri@@@"
    )

    now = datetime.now(timezone.utc)
    root_span = LangfuseSpan(
        id="root-span-id",
        trace_id="test-trace-123",
        name="Test Execution",
        start_time=now,
        end_time=now,
        observation_type="span",
        metadata={},
    )

    child_span = LangfuseSpan(
        id="child-span-id",
        trace_id="test-trace-123",
        parent_id="root-span-id",
        name="Image Processing Node",
        start_time=now,
        end_time=now,
        observation_type="span",
        # Media token in output (as it would be after media_api.py patching)
        output={
            "image": media_token,
            "metadata": {"filename": "test.jpg", "size": 12345},
        },
        metadata={},
    )

    trace = LangfuseTrace(
        id="test-trace-123",
        name="Test Execution",
        timestamp=now,
        metadata={},
        spans=[root_span, child_span],
    )

    # Mock the OTel span to capture what attributes are set
    captured_attributes = {}

    def mock_set_attribute(key: str, value: str):
        captured_attributes[key] = value

    mock_otel_span = MagicMock()
    mock_otel_span.set_attribute = mock_set_attribute
    mock_otel_span.get_span_context.return_value = MagicMock()

    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_otel_span

    settings = Settings(
        LANGFUSE_HOST="https://test.langfuse.com",
        LANGFUSE_PUBLIC_KEY="test-key",
        LANGFUSE_SECRET_KEY="test-secret",
        DB_TABLE_PREFIX="test_",
        TRUNCATE_FIELD_LEN=0,
    )

    with patch("src.shipper.trace.get_tracer", return_value=mock_tracer):
        export_trace(trace, settings, dry_run=False)

    # Verify output was JSON-serialized
    assert "langfuse.observation.output" in captured_attributes
    output_attr = captured_attributes["langfuse.observation.output"]

    # Should be valid JSON string
    parsed = json.loads(output_attr)

    # Should contain the media token as a string (not double-escaped)
    assert "image" in parsed
    assert parsed["image"] == media_token

    # Token should NOT be wrapped in quotes beyond JSON string encoding
    # (i.e., no Python repr like "{'image': '@@@langfuseMedia:...'}")
    assert parsed["image"].startswith("@@@langfuseMedia:")
    assert parsed["image"].endswith("@@@")

    # Verify metadata also preserved
    assert parsed["metadata"]["filename"] == "test.jpg"
    assert parsed["metadata"]["size"] == 12345


def test_media_token_json_serialized_in_input():
    """Verify media token in span input is also JSON-serialized correctly."""
    media_token = (
        "@@@langfuseMedia:type=audio/mpeg|"
        "id=AbCdEf1234567890XyZ|source=base64_data_uri@@@"
    )

    now = datetime.now(timezone.utc)
    root_span = LangfuseSpan(
        id="root-span-id",
        trace_id="test-trace-456",
        name="Test Execution",
        start_time=now,
        end_time=now,
        observation_type="span",
        metadata={},
    )

    child_span = LangfuseSpan(
        id="child-span-id",
        trace_id="test-trace-456",
        parent_id="root-span-id",
        name="Audio Transcription Node",
        start_time=now,
        end_time=now,
        observation_type="span",
        input={
            "audio_file": media_token,
            "format": "mpeg",
        },
        metadata={},
    )

    trace = LangfuseTrace(
        id="test-trace-456",
        name="Test Execution",
        timestamp=now,
        metadata={},
        spans=[root_span, child_span],
    )

    captured_attributes = {}

    def mock_set_attribute(key: str, value: str):
        captured_attributes[key] = value

    mock_otel_span = MagicMock()
    mock_otel_span.set_attribute = mock_set_attribute
    mock_otel_span.get_span_context.return_value = MagicMock()

    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_otel_span

    settings = Settings(
        LANGFUSE_HOST="https://test.langfuse.com",
        LANGFUSE_PUBLIC_KEY="test-key",
        LANGFUSE_SECRET_KEY="test-secret",
        DB_TABLE_PREFIX="test_",
        TRUNCATE_FIELD_LEN=0,
    )

    with patch("src.shipper.trace.get_tracer", return_value=mock_tracer):
        export_trace(trace, settings, dry_run=False)

    # Verify input was JSON-serialized
    assert "langfuse.observation.input" in captured_attributes
    input_attr = captured_attributes["langfuse.observation.input"]

    # Should be valid JSON string
    parsed = json.loads(input_attr)

    # Should contain the media token
    assert "audio_file" in parsed
    assert parsed["audio_file"] == media_token
    assert parsed["audio_file"].startswith("@@@langfuseMedia:")
    assert parsed["format"] == "mpeg"


def test_media_token_not_double_stringified():
    """Ensure media tokens aren't wrapped in extra quotes via str()."""
    # This test specifically guards against the bug where:
    # str({'image': '@@@langfuseMedia:...'})
    # produces: "{'image': '@@@langfuseMedia:...'}"
    # instead of proper JSON: '{"image":"@@@langfuseMedia:..."}'

    media_token = "@@@langfuseMedia:type=image/png|id=TestID123|source=base64_data_uri@@@"

    now = datetime.now(timezone.utc)
    root_span = LangfuseSpan(
        id="root-span-id",
        trace_id="test-trace-789",
        name="Test Execution",
        start_time=now,
        end_time=now,
        observation_type="span",
        metadata={},
    )

    child_span = LangfuseSpan(
        id="child-span-id",
        trace_id="test-trace-789",
        parent_id="root-span-id",
        name="Image Node",
        start_time=now,
        end_time=now,
        observation_type="span",
        output={"result": media_token},
        metadata={},
    )

    trace = LangfuseTrace(
        id="test-trace-789",
        name="Test Execution",
        timestamp=now,
        metadata={},
        spans=[root_span, child_span],
    )

    captured_attributes = {}

    def mock_set_attribute(key: str, value: str):
        captured_attributes[key] = value

    mock_otel_span = MagicMock()
    mock_otel_span.set_attribute = mock_set_attribute
    mock_otel_span.get_span_context.return_value = MagicMock()

    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_otel_span

    settings = Settings(
        LANGFUSE_HOST="https://test.langfuse.com",
        LANGFUSE_PUBLIC_KEY="test-key",
        LANGFUSE_SECRET_KEY="test-secret",
        DB_TABLE_PREFIX="test_",
        TRUNCATE_FIELD_LEN=0,
    )

    with patch("src.shipper.trace.get_tracer", return_value=mock_tracer):
        export_trace(trace, settings, dry_run=False)

    output_attr = captured_attributes["langfuse.observation.output"]

    # Should NOT contain Python dict repr syntax
    assert "{'result':" not in output_attr  # Python str() artifact
    assert '"result"' in output_attr  # JSON key syntax

    # Should be valid JSON
    parsed = json.loads(output_attr)
    assert parsed["result"] == media_token

    # Token should be directly searchable (not nested in escaped string)
    assert media_token in output_attr
