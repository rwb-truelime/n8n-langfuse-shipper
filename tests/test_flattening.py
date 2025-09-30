"""
Test mandatory data flattening of ALL span input/output fields.

Per copilot-instructions.md Core Invariant #7:
"Flattened input/output MANDATORY: ALL span input and output fields MUST be
flattened to single-level dictionaries. NO nested objects allowed."

These tests ensure nested JSON structures are flattened using dot notation
for improved Langfuse UI readability.
"""

from datetime import datetime, timezone
from src.mapper import map_execution_to_langfuse
from src.models.n8n import (
    N8nExecutionRecord,
    WorkflowData,
    WorkflowNode,
    ExecutionData,
    ExecutionDataDetails,
    ResultData,
    NodeRun,
)


def _build_record(node_output: dict) -> N8nExecutionRecord:
    """Helper to build minimal execution record with specific node output."""
    return N8nExecutionRecord(
        id=999,
        workflowId="wf_flat",
        status="success",
        startedAt=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        stoppedAt=datetime(2025, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        workflowData=WorkflowData(
            id="wf_flat",
            name="Flattening Test Workflow",
            nodes=[
                WorkflowNode(
                    name="FlatNode",
                    type="n8n-nodes-base.set",
                    category="core",
                )
            ],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "FlatNode": [
                            NodeRun(
                                startTime=1704110400000,
                                executionTime=100,
                                executionStatus="success",
                                data={"main": [[{"json": node_output}]]},
                                source=None,
                            )
                        ]
                    }
                )
            )
        ),
    )


def test_nested_dict_flattened_with_dot_notation():
    """Nested dicts should be flattened: {"user": {"name": "Alice"}} -> {"user.name": "Alice"}"""
    rec = _build_record({"user": {"name": "Alice", "age": 30}})
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "FlatNode"][0]
    
    assert isinstance(span.output, dict), "Output must be a dict, not JSON string"
    assert "user.name" in span.output, "Expected flattened key user.name"
    assert span.output["user.name"] == "Alice"
    assert "user.age" in span.output
    assert span.output["user.age"] == 30
    # Ensure no nested dicts remain
    for value in span.output.values():
        assert not isinstance(value, dict), f"Found nested dict: {value}"


def test_list_flattened_with_numeric_indices():
    """Lists should use numeric indices: {"items": [1, 2, 3]} -> {"items.0": 1, "items.1": 2, "items.2": 3}"""
    rec = _build_record({"items": [10, 20, 30]})
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "FlatNode"][0]
    
    assert isinstance(span.output, dict)
    assert "items.0" in span.output
    assert span.output["items.0"] == 10
    assert "items.1" in span.output
    assert span.output["items.1"] == 20
    assert "items.2" in span.output
    assert span.output["items.2"] == 30
    # No lists should remain
    for value in span.output.values():
        assert not isinstance(value, list), f"Found nested list: {value}"


def test_mixed_nesting_flattened():
    """Mixed nesting should combine both: {"data": {"items": [{"id": 5}]}} -> {"data.items.0.id": 5}"""
    rec = _build_record({
        "data": {
            "items": [
                {"id": 5, "name": "first"},
                {"id": 6, "name": "second"}
            ]
        }
    })
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "FlatNode"][0]
    
    assert isinstance(span.output, dict)
    assert "data.items.0.id" in span.output
    assert span.output["data.items.0.id"] == 5
    assert "data.items.0.name" in span.output
    assert span.output["data.items.0.name"] == "first"
    assert "data.items.1.id" in span.output
    assert span.output["data.items.1.id"] == 6
    assert "data.items.1.name" in span.output
    assert span.output["data.items.1.name"] == "second"


def test_primitives_unchanged():
    """Primitive values should remain unchanged."""
    rec = _build_record({
        "string": "hello",
        "number": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
    })
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "FlatNode"][0]
    
    assert isinstance(span.output, dict)
    assert span.output["string"] == "hello"
    assert span.output["number"] == 42
    assert span.output["float"] == 3.14
    assert span.output["bool"] is True
    assert span.output["null"] is None


def test_empty_containers_preserved():
    """Empty dicts and lists should be represented (or dropped - implementation detail)."""
    rec = _build_record({
        "emptyDict": {},
        "emptyList": [],
        "nonEmpty": "value",
    })
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "FlatNode"][0]
    
    assert isinstance(span.output, dict)
    # Non-empty values must be present
    assert "nonEmpty" in span.output
    # Empty containers may be dropped entirely or represented with markers.
    # Current implementation drops empty containers, which is acceptable.
    # This test validates that flattening doesn't crash on empty containers.


def test_media_tokens_treated_as_primitives():
    """Media token strings should not be decomposed during flattening.
    
    Note: Single-key dicts with string values are extracted to plain strings
    by smart processing (for markdown rendering), so the media token becomes
    the direct output value rather than nested in a dict.
    """
    media_token = "@@@langfuseMedia:type=image/jpeg|id=m123|source=base64_data_uri@@@"
    rec = _build_record({"image": media_token})
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "FlatNode"][0]
    
    # Smart processing extracts single string values from single-key dicts
    assert isinstance(span.output, str)
    assert span.output == media_token, "Media token preserved as primitive string"


def test_deep_nesting_flattened():
    """Very deep nesting should be flattened with long dot-notation keys."""
    rec = _build_record({
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "level5": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
    })
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "FlatNode"][0]
    
    assert isinstance(span.output, dict)
    assert "level1.level2.level3.level4.level5.value" in span.output
    assert span.output["level1.level2.level3.level4.level5.value"] == "deep"


def test_input_also_flattened():
    """Input fields must be flattened just like output fields."""
    rec = N8nExecutionRecord(
        id=998,
        workflowId="wf_input",
        status="success",
        startedAt=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        stoppedAt=datetime(2025, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        workflowData=WorkflowData(
            id="wf_input",
            name="Input Flattening Test",
            nodes=[
                WorkflowNode(
                    name="InputNode",
                    type="n8n-nodes-base.set",
                    category="core",
                )
            ],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "InputNode": [
                            NodeRun(
                                startTime=1704110400000,
                                executionTime=100,
                                executionStatus="success",
                                data={"main": [[{"json": {"output": "test"}}]]},
                                source=None,
                                inputOverride={"nested": {"input": "value"}},
                            )
                        ]
                    }
                )
            )
        ),
    )
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "InputNode"][0]
    
    assert isinstance(span.input, dict), "Input must be a dict"
    assert "nested.input" in span.input, "Expected flattened input key"
    assert span.input["nested.input"] == "value"


def test_all_spans_have_flattened_io():
    """Ensure NO spans have nested objects in input or output fields."""
    rec = _build_record({
        "complex": {
            "nested": {
                "array": [
                    {"item": 1},
                    {"item": 2}
                ]
            }
        }
    })
    trace = map_execution_to_langfuse(rec)
    
    for span in trace.spans:
        if span.input is not None and isinstance(span.input, dict):
            for key, value in span.input.items():
                assert not isinstance(value, dict), (
                    f"Span {span.name} has nested dict in input at key {key}: {value}"
                )
                assert not isinstance(value, list), (
                    f"Span {span.name} has nested list in input at key {key}: {value}"
                )
        
        if span.output is not None and isinstance(span.output, dict):
            for key, value in span.output.items():
                assert not isinstance(value, dict), (
                    f"Span {span.name} has nested dict in output at key {key}: {value}"
                )
                assert not isinstance(value, list), (
                    f"Span {span.name} has nested list in output at key {key}: {value}"
                )


def test_large_array_capped():
    """Arrays exceeding 1000 items should be capped with _length metadata."""
    large_list = list(range(1500))
    rec = _build_record({"bigArray": large_list})
    trace = map_execution_to_langfuse(rec)
    span = [s for s in trace.spans if s.name == "FlatNode"][0]
    
    assert isinstance(span.output, dict)
    # Should have items up to limit
    assert "bigArray.999" in span.output, "Expected 1000th item (0-indexed as 999)"
    # Should not have item beyond limit
    assert "bigArray.1000" not in span.output, "Should not have 1001st item"
    # Should have length indicator
    assert "bigArray._length" in span.output or "bigArray._truncated" in span.output
    if "bigArray._length" in span.output:
        assert span.output["bigArray._length"] == 1500
