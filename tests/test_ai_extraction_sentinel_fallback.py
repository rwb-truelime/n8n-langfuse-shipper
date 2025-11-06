from datetime import datetime, timezone

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
from n8n_langfuse_shipper.config import get_settings


def _build_record():
    return N8nExecutionRecord(
        id=424242,
        workflowId="wf-sentinel",
        status="success",
        startedAt=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        stoppedAt=datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        workflowData=WorkflowData(
            id="wf-sentinel",
            name="Sentinel Workflow",
            nodes=[
                WorkflowNode(
                    name="LoopNode",
                    type="n8n-nodes-base.function",
                ),
                WorkflowNode(
                    name="AIChatNode",
                    type="@n8n/langchain.lmchat",
                    category="AI/LangChain Nodes",
                ),
            ],
            connections={},
        ),
        data=ExecutionData(
            executionData=ExecutionDataDetails(
                resultData=ResultData(
                    runData={
                        "LoopNode": [
                            NodeRun(
                                startTime=1704067200000,
                                executionTime=10,
                                executionStatus="success",
                                data={"main": [[{"json": {"iteration": 1, "other": "x"}}]]},
                            )
                        ],
                        "AIChatNode": [
                            NodeRun(
                                startTime=1704067201000,
                                executionTime=10,
                                executionStatus="success",
                                data={
                                    "main": [[{"json": {"response": "ok"}}]],
                                    "ai_languageModel": [
                                        [{"json": {"tokenUsage": {"total": 5}}}]
                                    ],
                                },
                            )
                        ],
                    }
                )
            )
        ),
    )


def test_sentinel_patterns_fallback(monkeypatch):
    """Sentinel include patterns only should fallback to full capture (not None)."""
    monkeypatch.setenv("FILTER_AI_EXTRACTION_NODES", "LoopNode")
    # Simulate lingering sentinel patterns (not explicitly set in this test env)
    # We do NOT set FILTER_AI_EXTRACTION_INCLUDE_KEYS so settings may carry prior state.
    get_settings.cache_clear()
    rec = _build_record()
    trace = map_execution_to_langfuse(rec, filter_ai_only=True)
    root_span = next(s for s in trace.spans if s.parent_id is None)
    extracted = root_span.metadata.get("n8n.extracted_nodes", {})
    loop = extracted.get("LoopNode")
    assert loop, "LoopNode should be extracted"
    run0 = loop["runs"][0]
    assert run0["output"], "Output should not be None after fallback"
    # Ensure expected nested structure preserved
    assert run0["output"]["main"][0][0]["json"]["iteration"] == 1
