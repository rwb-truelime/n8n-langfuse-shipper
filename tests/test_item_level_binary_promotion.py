from datetime import datetime, timezone

from src.mapper import map_execution_with_assets
from src.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)

# Short base64 repeated to exceed >100 char canonical threshold
B64 = "aGVsbG9fYnVpbGRfYmluYXJ5X2RhdGE=" * 8


def _record():
    # Shape: output list -> nested list(s) -> dict(s) each with 'binary' sibling to 'json'
    payload = {
        "main": [
            [
                [
                    {
                        "json": {"ok": True, "value": 1},
                        "binary": {
                            "img": {
                                "data": B64,
                                "mimeType": "image/png",
                                "fileName": "img.png",
                            }
                        },
                    },
                    {
                        "json": {"ok": True, "value": 2},
                        "binary": {
                            "img2": {
                                "data": B64,
                                "mimeType": "image/png",
                                "fileName": "img2.png",
                            }
                        },
                    },
                ]
            ]
        ]
    }
    run = NodeRun(
        startTime=1710002000000,
        executionTime=7,
        executionStatus="success",
        data=payload,
        source=None,
        inputOverride=None,
        error=None,
    )
    return N8nExecutionRecord(
        id=9100,
        workflowId="wf-item-promote",
        status="success",
        startedAt=datetime.now(tz=timezone.utc),
        stoppedAt=datetime.now(tz=timezone.utc),
        workflowData=WorkflowData(
            id="wf-item-promote",
            name="wf-item-promote",
            nodes=[WorkflowNode(name="NodeY", type="custom.itemBinary")],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(resultData=ResultData(runData={"NodeY": [run]}))
        ),
    )


def test_item_level_binary_promoted():
    rec = _record()
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    span = next(s for s in mapped.trace.spans if s.name == "NodeY")
    assert span.output is not None
    # Output is now a flattened dict
    assert isinstance(span.output, dict), "Output must be flattened dict"
    # Check for promoted binary keys in flattened form: binary.img, binary.img2
    assert any(
        "binary.img." in k for k in span.output.keys()
    ), "Expected binary.img in flattened output"
    assert any(
        "binary.img2." in k for k in span.output.keys()
    ), "Expected binary.img2 in flattened output"
    # Placeholders inserted - media placeholders stay nested
    img_data = span.output.get("binary.img.data")
    img2_data = span.output.get("binary.img2.data")
    assert isinstance(img_data, dict), "Expected nested placeholder for img"
    assert img_data.get("_media_pending") is True
    assert isinstance(img2_data, dict), "Expected nested placeholder for img2"
    assert img2_data.get("_media_pending") is True
    # Metadata flag present
    assert span.metadata.get("n8n.io.promoted_item_binary") is True
    # Ensure base64_len recorded
    assert "base64_len" in img_data
    assert "base64_len" in img2_data
