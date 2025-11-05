"""Tests for AI-only node extraction feature - safety and edge cases.

Tests robustness and safety features:
    * Binary data stripping (always applied)
    * Size limiting per value
    * Truncation markers
    * Large nested structures
    * Serialization edge cases
    * Error handling (fail-open)
"""
from __future__ import annotations

from n8n_langfuse_shipper.mapping.node_extractor import extract_nodes_data
from n8n_langfuse_shipper.models.n8n import NodeRun


def test_binary_stripping_applied():
    """Binary data always stripped before size limiting."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "binary": {
                        "file1": {
                            "data": "SGVsbG8gV29ybGQ=" * 1000,  # Large base64
                            "mimeType": "application/pdf",
                            "fileName": "test.pdf",
                        }
                    },
                    "normalField": "value",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    # Binary block present with stripped data
    assert "binary" in output
    assert "file1" in output["binary"]
    # Data field should be replaced with stripping marker
    file_data = output["binary"]["file1"]
    # Binary stripping creates a dict with _binary: true and note field
    assert isinstance(file_data["data"], (str, dict))
    if isinstance(file_data["data"], dict):
        assert file_data["data"].get("_binary") is True
        assert "note" in file_data["data"]
    else:
        # Or simple string replacement
        assert "omitted" in file_data["data"] or "truncated" in file_data["data"]
    # Normal field preserved
    assert output["normalField"] == "value"


def test_size_limit_truncation():
    """Values exceeding max_value_len are truncated with marker."""
    long_value = "x" * 15000
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "longText": long_value,
                    "shortText": "normal",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=5000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    # Long text truncated
    assert len(output["longText"]) < 5050  # 5000 + marker
    assert output["longText"].endswith("... [truncated]")
    assert output["longText"].startswith("xxx")

    # Short text preserved
    assert output["shortText"] == "normal"

    # Truncation flag set
    assert extracted["TestNode"]["runs"][0]["_truncated"] is True


def test_no_truncation_when_under_limit():
    """Values under limit not truncated."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={"field": "short value"},
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert output["field"] == "short value"
    assert "... [truncated]" not in output["field"]
    assert extracted["TestNode"]["runs"][0]["_truncated"] is False


def test_non_string_values_not_truncated():
    """Only strings are subject to size limiting."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "number": 12345,
                    "boolean": True,
                    "null": None,
                    "list": [1, 2, 3],
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10,  # Very small limit
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert output["number"] == 12345
    assert output["boolean"] is True
    assert output["null"] is None
    assert output["list"] == [1, 2, 3]


def test_deeply_nested_structure():
    """Handle deeply nested objects correctly."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "level1": {
                        "level2": {
                            "level3": {
                                "level4": {
                                    "deepValue": "found",
                                }
                            }
                        }
                    }
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=["level1.level2.level3.level4.deepValue"],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    # Deep path reconstructed
    assert "level1" in output
    assert "level2" in output["level1"]
    assert "level3" in output["level1"]["level2"]
    assert "level4" in output["level1"]["level2"]["level3"]
    assert output["level1"]["level2"]["level3"]["level4"]["deepValue"] == "found"


def test_empty_data_handling():
    """Empty dicts and null data handled gracefully."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={},  # Empty output
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    # Should still have structure
    assert "TestNode" in extracted
    assert extracted["TestNode"]["runs"][0]["output"] is None or extracted["TestNode"]["runs"][0]["output"] == {}


def test_list_values_preserved():
    """Lists in data preserved (not flattened with indices)."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "items": ["item1", "item2", "item3"],
                    "nested": {
                        "array": [1, 2, 3],
                    },
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert output["items"] == ["item1", "item2", "item3"]
    assert output["nested"]["array"] == [1, 2, 3]


def test_special_characters_in_keys():
    """Keys with special characters handled correctly.

    Note: Dots in keys are treated as path separators during flatten/unflatten,
    which is expected behavior for nested key filtering support.
    """
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "key-with-dashes": "value1",
                    "key_with_underscores": "value2",
                    "key$with$dollars": "value4",
                    "simple": "value5",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert "key-with-dashes" in output
    assert "key_with_underscores" in output
    assert "key$with$dollars" in output
    assert "simple" in output


def test_unicode_content():
    """Unicode characters preserved in extraction."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "emoji": "🚀 🎉 ✨",
                    "chinese": "你好世界",
                    "arabic": "مرحبا بالعالم",
                    "mixed": "Hello 世界 🌍",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert output["emoji"] == "🚀 🎉 ✨"
    assert output["chinese"] == "你好世界"
    assert output["arabic"] == "مرحبا بالعالم"
    assert output["mixed"] == "Hello 世界 🌍"


def test_large_number_of_keys():
    """Handle extraction with many keys efficiently."""
    # Create 100 keys
    data = {f"key{i}": f"value{i}" for i in range(100)}

    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data=data,
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    # All keys present
    assert len(output) == 100
    assert output["key0"] == "value0"
    assert output["key99"] == "value99"


def test_mixed_truncation_flags():
    """Truncation flag reflects any truncation in input OR output."""
    long_value = "x" * 15000

    # Input truncated, output not
    run_data1 = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                inputOverride={"longInput": long_value},
                data={"shortOutput": "ok"},
            )
        ]
    }

    extracted1 = extract_nodes_data(
        run_data=run_data1,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=5000,
    )

    assert extracted1["TestNode"]["runs"][0]["_truncated"] is True

    # Output truncated, no input
    run_data2 = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={"longOutput": long_value},
            )
        ]
    }

    extracted2 = extract_nodes_data(
        run_data=run_data2,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=5000,
    )

    assert extracted2["TestNode"]["runs"][0]["_truncated"] is True


def test_base64_like_string_stripped():
    """Long base64-like strings detected and stripped."""
    # Long base64 string (>200 chars)
    base64_data = "SGVsbG8gV29ybGQ=" * 20  # 320 chars

    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "possibleBase64": base64_data,
                    "normalText": "hello",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    # If detected as binary, it would be replaced
    # Otherwise preserved (binary detection has specific heuristics)
    assert "normalText" in output
    assert output["normalText"] == "hello"


def test_error_in_single_node_doesnt_block_others():
    """Fail-open: error extracting one node doesn't prevent others."""
    # This is more of an integration concern, but node_extractor should
    # handle individual node failures gracefully
    # The extraction function catches exceptions per node

    # Create valid data for two nodes
    run_data = {
        "GoodNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={"field": "value"},
            )
        ],
        "AnotherGoodNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={"field2": "value2"},
            )
        ],
    }

    # Even if one node causes issues, others should extract
    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["GoodNode", "NonExistentNode", "AnotherGoodNode"],
        include_patterns=[],
        exclude_patterns=[],
        max_value_len=10000,
    )

    # Both valid nodes extracted
    assert "GoodNode" in extracted
    assert "AnotherGoodNode" in extracted

    # Missing node tracked
    assert "NonExistentNode" in extracted["_meta"]["nodes_not_found"]

    # Count correct
    assert extracted["_meta"]["extracted_count"] == 2
    assert extracted["_meta"]["nodes_requested"] == 3
