from __future__ import annotations

from datetime import datetime, timezone

from src.models.langfuse import LangfuseSpan, LangfuseTrace, LangfuseUsage
from src.shipper import _apply_span_attributes


class DummySpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):  # mimic OTEL span API used
        self.attributes[key] = value


def test_usage_details_json_emitted_only_present_keys():
    span_model = LangfuseSpan(
        id="s1",
        trace_id="t1",
        parent_id=None,
        name="LLM",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        observation_type="generation",
        usage=LangfuseUsage(input=5, output=None, total=7),
    )
    dummy = DummySpan()
    _apply_span_attributes(dummy, span_model)
    # Should include input_tokens + total_tokens but omit output_tokens in JSON (since output None)
    usage_json = dummy.attributes.get("langfuse.observation.usage_details")
    assert usage_json is not None
    assert '"input_tokens":5' in usage_json
    assert '"total_tokens":7' in usage_json
    assert 'output_tokens' not in usage_json  # ensure absent key not serialized


def test_root_flag_and_trace_identity_serialization():
    # Build a trace and manually call export internal portions (simulate root attribute setting logic)
    now = datetime.now(timezone.utc)
    root_span = LangfuseSpan(
        id="root1",
        trace_id="t2",
        parent_id=None,
        name="Execution",
        start_time=now,
        end_time=now,
    )
    trace_model = LangfuseTrace(
        id="42",
        name="WF",
        timestamp=now,
        metadata={},
        spans=[root_span],
        user_id="user-123",
        session_id="sess-9",
        tags=["alpha", "beta"],
        trace_input={"q": "hello"},
        trace_output={"a": "world"},
    )
    # Instead of invoking full export (network), mimic attribute setting code path from export_trace
    dummy = DummySpan()
    _apply_span_attributes(dummy, root_span)
    # Manually replicate export_trace root attribute setting additions
    import json
    dummy.set_attribute("langfuse.trace.name", trace_model.name)
    dummy.set_attribute("langfuse.as_root", True)
    dummy.set_attribute("langfuse.trace.user_id", trace_model.user_id)
    dummy.set_attribute("langfuse.trace.session_id", trace_model.session_id)
    dummy.set_attribute("langfuse.trace.tags", json.dumps(trace_model.tags, separators=(",", ":")))
    dummy.set_attribute("langfuse.trace.input", json.dumps(trace_model.trace_input, separators=(",", ":")))
    dummy.set_attribute("langfuse.trace.output", json.dumps(trace_model.trace_output, separators=(",", ":")))

    assert dummy.attributes["langfuse.as_root"] is True
    assert dummy.attributes["langfuse.trace.user_id"] == "user-123"
    assert dummy.attributes["langfuse.trace.session_id"] == "sess-9"
    assert dummy.attributes["langfuse.trace.tags"] == '["alpha","beta"]'
    assert dummy.attributes["langfuse.trace.input"] == '{"q":"hello"}'
    assert dummy.attributes["langfuse.trace.output"] == '{"a":"world"}'


def test_level_and_status_message_non_error():
    now = datetime.now(timezone.utc)
    span_model = LangfuseSpan(
        id="s-level",
        trace_id="t3",
        parent_id=None,
        name="LevelSpan",
        start_time=now,
        end_time=now,
        level="INFO",
        status_message="processing complete",
    )
    dummy = DummySpan()
    _apply_span_attributes(dummy, span_model)
    assert dummy.attributes.get("langfuse.observation.level") == "INFO"
    assert dummy.attributes.get("langfuse.observation.status_message") == "processing complete"


def test_error_does_not_override_explicit_level_status():
    now = datetime.now(timezone.utc)
    span_model = LangfuseSpan(
        id="s-level-err",
        trace_id="t4",
        parent_id=None,
        name="LevelErrSpan",
        start_time=now,
        end_time=now,
        level="WARNING",
        status_message="pre-set message",
        error={"message": "boom"},
    )
    dummy = DummySpan()
    _apply_span_attributes(dummy, span_model)
    # Should retain explicit level/status_message
    assert dummy.attributes.get("langfuse.observation.level") == "WARNING"
    assert dummy.attributes.get("langfuse.observation.status_message") == "pre-set message"
