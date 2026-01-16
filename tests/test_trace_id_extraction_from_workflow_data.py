import pytest

from n8n_langfuse_shipper.shipper import _extract_trace_id_from_workflow_data


def _generate_test_data(expected_trace_id: str | None, trace_id_field_name: str) -> str:
    data = '''{
      "id": "11",
      "name": "My workflow",
      "timestamp": "2025-12-16T08:28:37.755000Z",
      "metadata": {
        "workflowId": "wd7c2KbKnqhlZ1yK",
        "status": "success"
      },
      "spans": [
        {
          "id": "be2bdb89-dce0-5ef9-b9ef-eb8b9f8400cf",
          "trace_id": "11",
          "parent_id": null,
          "name": "My workflow",
          "start_time": "2025-12-16T08:28:37.755000Z",
          "end_time": "2025-12-16T08:28:39.285000Z",
          "observation_type": "span",
          "input": null,
          "output": null,
          "metadata": {
            "n8n.execution.id": 11
          },
          "error": null,
          "model": null,
          "usage": null,
          "status": null,
          "level": null,
          "status_message": null,
          "prompt_name": null,
          "prompt_version": null,
          "otel_span_id": null
        },
        {
          "id": "1d161f89-f06d-5b88-aebc-cfbd45834c33",
          "trace_id": "11",
          "parent_id": "be2bdb89-dce0-5ef9-b9ef-eb8b9f8400cf",
          "name": "When clicking ‘Execute workflow’",
          "start_time": "2025-12-16T08:28:37.765000Z",
          "end_time": "2025-12-16T08:28:37.766000Z",
          "observation_type": "span",
          "input": null,
          "output": "{\"main\":[[{\"json\":{},\"pairedItem\":{\"item\":0}}]]}",
          "metadata": {
            "n8n.node.type": "n8n-nodes-base.manualTrigger",
            "n8n.node.category": null,
            "n8n.node.run_index": 0,
            "n8n.node.execution_time_ms": 1,
            "n8n.node.execution_status": "success"
          },
          "error": null,
          "model": null,
          "usage": null,
          "status": "success",
          "level": null,
          "status_message": null,
          "prompt_name": null,
          "prompt_version": null,
          "otel_span_id": null
        },
        {
          "id": "69368786-f387-5242-acb2-dd0d9a2f17b7",
          "trace_id": "11",
          "parent_id": "1d161f89-f06d-5b88-aebc-cfbd45834c33",
          "name": "Edit Fields",
          "start_time": "2025-12-16T08:28:37.767000Z",
          "end_time": "2025-12-16T08:28:37.774000Z",
          "observation_type": "span",
          "input": "{\"inferredFrom\":\"When clicking ‘Execute workflow’\",\"data\":{\"main\":[[{\"json\":{},\"pairedItem\":{\"item\":0}}]]}}",
          "output": "{\"TRACE_ID_NAME":\"TRACE_ID_VALUE\"}",
          "metadata": {
            "n8n.node.type": "n8n-nodes-base.set",
            "n8n.node.category": null,
            "n8n.node.run_index": 0,
            "n8n.node.execution_time_ms": 7,
            "n8n.node.execution_status": "success",
            "n8n.io.unwrapped_json_root": true,
            "n8n.node.previous_node": "When clicking ‘Execute workflow’",
            "n8n.graph.inferred_parent": true
          },
          "error": null,
          "model": null,
          "usage": null,
          "status": "success",
          "level": null,
          "status_message": null,
          "prompt_name": null,
          "prompt_version": null,
          "otel_span_id": null
        },
        {
          "id": "7f3203c5-314c-59ee-9448-c507eb13a577",
          "trace_id": "11",
          "parent_id": "69368786-f387-5242-acb2-dd0d9a2f17b7",
          "name": "Get a prompt",
          "start_time": "2025-12-16T08:28:37.774000Z",
          "end_time": "2025-12-16T08:28:37.880000Z",
          "observation_type": "span",
          "input": "{\"TRACE_ID_NAME":\"TRACE_ID_VALUE\"}",
          "output": "{\"id\":\"607b1105-8055-47c9-901e-a0110371e774\",\"createdAt\":\"2025-12-15T22:30:09.072Z\",\"updatedAt\":\"2025-12-15T22:30:09.072Z\",\"projectId\":\"smart-article\",\"createdBy\":\"cmhlvuz4v0002jz0747krsqqm\",\"prompt\":\"test\",\"name\":\"test\",\"version\":1,\"type\":\"text\",\"isActive\":null,\"config\":{},\"tags\":[],\"labels\":[\"production\",\"latest\"],\"commitMessage\":null,\"resolutionGraph\":null}",
          "metadata": {
            "n8n.node.type": "@langfuse/n8n-nodes-langfuse.langfuse",
            "n8n.node.category": null,
            "n8n.node.run_index": 0,
            "n8n.node.execution_time_ms": 106,
            "n8n.node.execution_status": "success",
            "n8n.io.unwrapped_json_root": true,
            "n8n.node.previous_node": "Edit Fields",
            "n8n.graph.inferred_parent": true
          },
          "error": null,
          "model": null,
          "usage": null,
          "status": "success",
          "level": null,
          "status_message": null,
          "prompt_name": null,
          "prompt_version": null,
          "otel_span_id": null
        },
        {
          "id": "ee73664a-3ea3-547c-b179-7d682a146258",
          "trace_id": "11",
          "parent_id": "7f3203c5-314c-59ee-9448-c507eb13a577",
          "name": "Message a model",
          "start_time": "2025-12-16T08:28:37.881000Z",
          "end_time": "2025-12-16T08:28:39.284000Z",
          "observation_type": "generation",
          "input": "{\"id\":\"607b1105-8055-47c9-901e-a0110371e774\",\"createdAt\":\"2025-12-15T22:30:09.072Z\",\"updatedAt\":\"2025-12-15T22:30:09.072Z\",\"projectId\":\"smart-article\",\"createdBy\":\"cmhlvuz4v0002jz0747krsqqm\",\"prompt\":\"test\",\"name\":\"test\",\"version\":1,\"type\":\"text\",\"isActive\":null,\"config\":{},\"tags\":[],\"labels\":[\"production\",\"latest\"],\"commitMessage\":null,\"resolutionGraph\":null}",
          "output": "{\"output\":[{\"type\":\"message\",\"id\":\"NhhBafKjH5vD78EP2_fhuQ4\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Understood. How can I help you today?\\n\\nAre you testing:\\n*   My capabilities?\\n*   A particular feature?\\n*   A specific question or task?\\n\\nLet me know what you'd like to test!\",\"annotations\":[]}]}]}",
          "metadata": {
            "n8n.node.type": "@n8n/n8n-nodes-langchain.openAi",
            "n8n.node.category": null,
            "n8n.node.run_index": 0,
            "n8n.node.execution_time_ms": 1403,
            "n8n.node.execution_status": "success",
            "n8n.io.unwrapped_json_root": true,
            "n8n.node.previous_node": "Get a prompt",
            "n8n.graph.inferred_parent": true,
            "n8n.model.missing": true,
            "n8n.model.search_keys": [
              "main"
            ],
            "n8n.prompt.version.original": 1,
            "n8n.prompt.resolution_method": "ancestor",
            "n8n.prompt.confidence": "high",
            "n8n.prompt.ancestor_distance": 1,
            "n8n.prompt.candidate_count": 1
          },
          "error": null,
          "model": null,
          "usage": null,
          "status": "success",
          "level": null,
          "status_message": null,
          "prompt_name": "test",
          "prompt_version": 1,
          "otel_span_id": null
        }
      ],
      "user_id": null,
      "session_id": null,
      "release": null,
      "public": null,
      "tags": [],
      "version": null,
      "environment": null,
      "trace_input": null,
      "trace_output": null,
      "otel_trace_id_hex": null
    }
    '''.replace("TRACE_ID_NAME", trace_id_field_name).replace("TRACE_ID_VALUE", expected_trace_id)
    return data


async def test_extract_trace_id_from_workflow_data():
    trace_id_field_name: str = "custom_trace_id"
    expected_trace_id: str = "ea9c9b6216164408a23fc032842a11b8"

    serialized_data = _generate_test_data(expected_trace_id, trace_id_field_name)

    trace_id = _extract_trace_id_from_workflow_data(serialized_data, trace_id_field_name,)

    assert trace_id == expected_trace_id


async def test_extract_trace_id_from_workflow_data_no_trace_id():
    trace_id_field_name: str = "custom_trace_id"

    serialized_data = _generate_test_data("", trace_id_field_name)

    with pytest.raises(ValueError):
        _extract_trace_id_from_workflow_data(serialized_data, trace_id_field_name,)


async def test_extract_trace_id_fieldname_wrong():

    serialized_data = _generate_test_data("1234", "test_field_name")

    with pytest.raises(ValueError):
        _extract_trace_id_from_workflow_data(serialized_data, "test_field",)


async def test_extract_trace_id_wrong_format():
    trace_id_field_name: str = "custom_trace_id"
    expected_trace_id: str = "wrong-format-trace-id-1234"

    serialized_data = _generate_test_data(expected_trace_id, trace_id_field_name)

    with pytest.raises(ValueError) as e:
        _extract_trace_id_from_workflow_data(serialized_data, trace_id_field_name,)
    assert f"Extracted trace ID '{expected_trace_id}' is not a valid hexadecimal string." in str(e.value)
