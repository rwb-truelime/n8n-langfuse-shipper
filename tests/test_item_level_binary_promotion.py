import json
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
B64 = ("aGVsbG9fYnVpbGRfYmluYXJ5X2RhdGE=" * 8)


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
            executionData=ExecutionDataDetails(
                resultData=ResultData(runData={"NodeY": [run]})
            )
        ),
    )


def test_item_level_binary_promoted():
    rec = _record()
    mapped = map_execution_with_assets(rec, collect_binaries=True)
    span = next(s for s in mapped.trace.spans if s.name == "NodeY")
    assert span.output is not None
    parsed = json.loads(str(span.output))
    # Normalization may unwrap to list root (e.g. merged json objects); when list, search
    # for dict element containing promoted binary section.
    if isinstance(parsed, list):
        cand = None
        for elem in parsed:
            if isinstance(elem, dict) and isinstance(elem.get("binary"), dict):
                cand = elem
                break
        assert cand is not None, "Promoted binary section not found in list root"
        bin_section = cand.get("binary")
    else:
        bin_section = parsed.get("binary")
    assert isinstance(bin_section, dict) and "img" in bin_section and "img2" in bin_section
    # Placeholders inserted
    img_entry = bin_section["img"].get("data") if isinstance(bin_section.get("img"), dict) else None
    img2_entry = bin_section["img2"].get("data") if isinstance(bin_section.get("img2"), dict) else None
    assert isinstance(img_entry, dict) and img_entry.get("_media_pending") is True
    assert isinstance(img2_entry, dict) and img2_entry.get("_media_pending") is True
    # Metadata flag present
    assert span.metadata.get("n8n.io.promoted_item_binary") is True
    # Ensure base64_len recorded
    assert "base64_len" in img_entry and isinstance(img_entry["base64_len"], int)
    assert "base64_len" in img2_entry and isinstance(img2_entry["base64_len"], int)
