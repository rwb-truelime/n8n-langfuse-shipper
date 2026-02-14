from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from n8n_langfuse_shipper.mapper import map_execution_to_langfuse
from n8n_langfuse_shipper.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)
from n8n_langfuse_shipper.vendor.flatted import parse as flatted_parse


def _build_execution_record(run_data_raw: dict[str, list[dict[str, Any]]]):
    run_data: dict[str, list[NodeRun]] = {}
    for node, runs in run_data_raw.items():
        converted: list[NodeRun] = []
        for r in runs:
            converted.append(
                NodeRun(
                    startTime=int(r.get("startTime", 0)),
                    executionTime=int(r.get("executionTime", 0)),
                    executionStatus=str(r.get("executionStatus", "unknown")),
                    data=r.get("data", {}),
                    source=None,
                    inputOverride=None,
                    error=None,
                )
            )
        run_data[node] = converted
    record = N8nExecutionRecord(
        id=999,
        workflowId="wf-flatted",
        status="success",
        startedAt=datetime.now(timezone.utc),
        stoppedAt=datetime.now(timezone.utc),
        workflowData=WorkflowData(
            id="wf-flatted",
            name="Flatted WF",
            nodes=[
                WorkflowNode(name=n, type="ToolWorkflow") for n in run_data.keys()
            ],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData=run_data)
            )
        ),
    )
    return record


def test_flatted_parsing_basic():
    # Equivalent to original compact pool but already serialized by flatted.
    # Build a minimal flatted JSON string using the upstream algorithm semantics.
    # For test simplicity we manually craft the expected array: root, runData wrapper, node runs.
    pool = [
        {"resultData": "1"},
        {"runData": "2"},
        {"NodeA": "3"},
        [
            {
                "startTime": 1700000000,
                "executionTime": 10,
                "executionStatus": "success",
                "data": {"main": [[{"json": {"value": 1}}]]},
            }
        ],
    ]
    raw = json.dumps(pool)
    parsed = flatted_parse(raw)
    run_data_raw = parsed.get("resultData", {}).get("runData", {})
    assert "NodeA" in run_data_raw
    record = _build_execution_record(run_data_raw)
    trace = map_execution_to_langfuse(record, truncate_limit=None)
    span_names = [s.name for s in trace.spans]
    assert "NodeA #0" in span_names


def test_flatted_numeric_literal_preserved():
    # Ensure numeric-looking strings are preserved and not dereferenced.
    pool = [
        {"resultData": "1"},
        {"runData": "2"},
        {"Schedule Trigger": "3"},
        [
            {
                "startTime": 1700000100,
                "executionTime": 5,
                "executionStatus": "success",
                "data": {
                    "main": [[{"json": {
                        "Day": "26", "Year": "2025", "Hour": "16"
                    }}]]
                },
            }
        ],
    ]
    raw = json.dumps(pool)
    parsed = flatted_parse(raw)
    run_data_raw = parsed.get("resultData", {}).get("runData", {})
    trigger_runs = run_data_raw["Schedule Trigger"]
    first = trigger_runs[0]["data"]["main"][0][0]["json"]
    assert first["Day"] == "26"
    assert first["Year"] == "2025"
    assert first["Hour"] == "16"


def test_flatted_malformed_fallback():
    # Provide malformed flatted payload (truncated) and ensure parse raises, caught externally.
    bad_raw = "[ {\"resultData\": \"1\" }"  # missing closing elements
    try:
        flatted_parse(bad_raw)
    except Exception:  # Expected failure
        return
    raise AssertionError("Malformed flatted payload should raise an exception")
