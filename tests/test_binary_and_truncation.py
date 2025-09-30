from __future__ import annotations

import json
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

BASE64_LONG = "A" * 300  # length >=200 triggers base64 heuristic (all 'A's still valid base64 padding-wise)


def _record_with_output(output_dict: dict, exec_id: int = 901) -> N8nExecutionRecord:
    now = datetime.now(timezone.utc)
    return N8nExecutionRecord(
        id=exec_id,
        workflowId="wf-bin",
        status="success",
        startedAt=now,
        stoppedAt=now,
        workflowData=WorkflowData(
            id="wf-bin",
            name="Binary Test WF",
            nodes=[WorkflowNode(name="BinaryNode", type="ToolWorkflow")],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "BinaryNode": [
                            NodeRun(
                                startTime=int(now.timestamp() * 1000),
                                executionTime=10,
                                executionStatus="success",
                                data=output_dict,
                            )
                        ]
                    }
                )
            )
        ),
    )


def test_binary_stripping_always_on_when_truncation_disabled():
    rec = _record_with_output(
        {
            "binary": {
                "file": {
                    "data": BASE64_LONG,
                    "mimeType": "image/jpeg",
                    "fileName": "x.jpg",
                }
            },
            "rawBase64": BASE64_LONG,
        }
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=None)  # None -> no truncation
    span = trace.spans[1]  # index 0 is root
    # Output is now a flattened dict (not JSON string)
    assert isinstance(span.output, dict), "Output must be flattened dict"
    # Check for binary redaction in flattened keys
    output_str = str(span.output)
    assert "binary omitted" in output_str, "Expected binary placeholder text"
    assert BASE64_LONG not in output_str, "Original base64 should be stripped"
    # Ensure no truncation flags set since truncation disabled
    assert "n8n.truncated.output" not in span.metadata


def test_truncation_flag_and_placeholder_coexist():
    rec = _record_with_output({"rawBase64": BASE64_LONG, "text": "x" * 500})
    trace = map_execution_to_langfuse(rec, truncate_limit=100)
    span = trace.spans[1]
    assert BASE64_LONG not in (span.output or "")
    assert span.metadata.get("n8n.truncated.output") is True

