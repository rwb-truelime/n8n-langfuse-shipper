from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.mapper import map_execution_to_langfuse
from src.models.n8n import (
    ExecutionData,
    ExecutionDataDetails,
    N8nExecutionRecord,
    NodeRun,
    ResultData,
    WorkflowData,
    WorkflowNode,
)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
TESTS_DIR = Path(__file__).resolve().parent  # current tests directory


def test_no_naive_datetime_patterns():
    """Ensure the codebase does not use deprecated/naive UTC constructors.

    We forbid `datetime.utcnow(` entirely and bare `datetime.now()` without a timezone argument.
    Allow: datetime.now(timezone.utc) or datetime.now(tz=timezone.utc).
    """
    forbidden = []
    search_dirs = [SRC_DIR, TESTS_DIR]
    this_file = Path(__file__).resolve()
    for base in search_dirs:
        for py_file in base.rglob("*.py"):
            if py_file.resolve() == this_file:
                # Skip self to avoid flagging example strings & detection logic
                continue
            text = py_file.read_text(encoding="utf-8")
            lines = text.splitlines()
            for i, line in enumerate(lines, start=1):
                stripped = line.strip()
                # Skip comment-only or docstring lines to reduce false positives (documentation references)
                if stripped.startswith("#") or stripped.startswith("\""):
                    continue
                if "datetime.utcnow(" in stripped:
                    if "allow-naive-datetime" not in stripped:
                        forbidden.append((py_file, i, "datetime.utcnow(", stripped))
                if "datetime.now(" in stripped:
                    if "allow-naive-datetime" in stripped:
                        continue
                    for m in re.finditer(r"datetime\.now\(([^)]*)\)", stripped):
                        inner = m.group(1).strip()
                        if not inner:
                            forbidden.append((py_file, i, "datetime.now() naive", stripped))
                        elif "timezone.utc" not in inner and "tz=" not in inner:
                            forbidden.append((py_file, i, f"datetime.now({inner}) lacks explicit timezone", stripped))
    assert not forbidden, (
        "Found naive datetime usages (add '# allow-naive-datetime' comment to intentionally allow):\n" +
        "\n".join(f"{p}:{ln}: {reason} -> {snippet}" for p, ln, reason, snippet in forbidden)
    )


def test_mapper_normalizes_naive_datetimes():
    """Provide a record with naive datetimes and verify span timestamps are timezone-aware."""
    naive_start = datetime.now()  # intentionally naive # allow-naive-datetime
    naive_stop = naive_start
    rec = N8nExecutionRecord(
        id=321,
        workflowId="wf-naive",
        status="success",
        startedAt=naive_start,
        stoppedAt=naive_stop,
        workflowData=WorkflowData(
            id="wf-naive",
            name="Naive WF",
            nodes=[WorkflowNode(name="Only", type="ToolWorkflow")],
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "Only": [
                            NodeRun(
                                startTime=int(naive_start.timestamp() * 1000),
                                executionTime=5,
                                executionStatus="success",
                                data={"ok": True},
                            )
                        ]
                    }
                )
            )
        ),
    )
    trace = map_execution_to_langfuse(rec, truncate_limit=0)
    root = trace.spans[0]
    assert root.start_time.tzinfo is not None, "Root span start_time should be timezone-aware"
    assert root.end_time.tzinfo is not None, "Root span end_time should be timezone-aware"
