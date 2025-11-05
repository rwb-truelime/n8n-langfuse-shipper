"""Tests for AI-only node extraction feature - key filtering.

Tests wildcard pattern matching for include/exclude filtering:
    * fnmatch glob patterns (*, ?, [seq])
    * Nested key paths (dot notation)
    * Include patterns (whitelist mode)
    * Exclude patterns (blacklist mode)
    * Combined include + exclude
    * Pattern precedence (exclude after include)
"""
from __future__ import annotations

from n8n_langfuse_shipper.mapping.node_extractor import (
    extract_nodes_data,
    matches_pattern,
)
from n8n_langfuse_shipper.models.n8n import NodeRun


def test_matches_pattern_exact():
    """Test exact string matching."""
    assert matches_pattern("userId", ["userId"]) is True
    assert matches_pattern("userId", ["userName"]) is False


def test_matches_pattern_wildcard_suffix():
    """Test * wildcard at end."""
    assert matches_pattern("userId", ["user*"]) is True
    assert matches_pattern("userName", ["user*"]) is True
    assert matches_pattern("accountId", ["user*"]) is False


def test_matches_pattern_wildcard_prefix():
    """Test * wildcard at start."""
    assert matches_pattern("userId", ["*Id"]) is True
    assert matches_pattern("accountId", ["*Id"]) is True
    assert matches_pattern("userName", ["*Id"]) is False


def test_matches_pattern_wildcard_middle():
    """Test * wildcard in middle (case-sensitive)."""
    assert matches_pattern("password", ["*password*"]) is True
    assert matches_pattern("passwordHash", ["*password*"]) is True
    assert matches_pattern("userPassword", ["*Password*"]) is True  # Case matters!
    assert matches_pattern("token", ["*password*"]) is False


def test_matches_pattern_multiple_patterns():
    """Test matching against multiple patterns (OR logic, case-sensitive)."""
    patterns = ["userId", "*Token*", "api*"]  # Case-sensitive patterns
    assert matches_pattern("userId", patterns) is True
    assert matches_pattern("apiKey", patterns) is True
    assert matches_pattern("authToken", ["*Token*"]) is True  # Correct case
    assert matches_pattern("userName", patterns) is False


def test_matches_pattern_nested_paths():
    """Test dotted path matching."""
    assert matches_pattern("headers.x-correlation-id", ["headers.*"]) is True
    assert matches_pattern("body.userId", ["body.*"]) is True
    assert matches_pattern("metadata.timestamp", ["metadata.*"]) is True
    assert matches_pattern("userId", ["headers.*"]) is False


def test_matches_pattern_question_mark():
    """Test ? single character wildcard."""
    assert matches_pattern("user1", ["user?"]) is True
    assert matches_pattern("user2", ["user?"]) is True
    assert matches_pattern("userAB", ["user?"]) is False
    assert matches_pattern("usr1", ["user?"]) is False


def test_matches_pattern_empty_patterns():
    """Empty pattern list matches nothing."""
    assert matches_pattern("anything", []) is False


def test_include_keys_only():
    """Extract only specified keys via include patterns."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "userId": "123",
                    "userName": "Alice",
                    "password": "secret",
                    "token": "abc-def",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=["userId", "userName"],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert "userId" in output
    assert "userName" in output
    assert "password" not in output
    assert "token" not in output


def test_exclude_keys_only():
    """Exclude sensitive keys while keeping others (case-sensitive matching)."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "userId": "123",
                    "userName": "Alice",
                    "password": "secret",
                    "apitoken": "xyz",  # lowercase to match pattern
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],  # No include = include all
        exclude_patterns=["*password*", "*token*"],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert "userId" in output
    assert "userName" in output
    assert "password" not in output
    assert "apitoken" not in output


def test_include_then_exclude():
    """Exclude patterns applied after include patterns (case-sensitive)."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "userId": "123",
                    "userpassword": "secret",  # lowercase to match pattern
                    "userName": "Alice",
                    "accountId": "456",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=["user*"],  # Include all user* keys
        exclude_patterns=["*password*"],  # But exclude passwords
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert "userId" in output
    assert "userName" in output
    assert "userpassword" not in output  # Excluded despite matching include
    assert "accountId" not in output  # Didn't match include


def test_nested_key_filtering():
    """Filter nested object keys using dot notation."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "headers": {
                        "x-correlation-id": "abc-123",
                        "authorization": "Bearer token",
                        "content-type": "application/json",
                    },
                    "body": {
                        "userId": "456",
                        "action": "create",
                    },
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=["headers.x-correlation-id", "body.userId"],
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    # Nested structure reconstructed
    assert "headers" in output
    assert "x-correlation-id" in output["headers"]
    assert output["headers"]["x-correlation-id"] == "abc-123"

    assert "body" in output
    assert "userId" in output["body"]
    assert output["body"]["userId"] == "456"

    # Excluded keys not present
    assert "authorization" not in output.get("headers", {})
    assert "action" not in output.get("body", {})


def test_wildcard_nested_filtering():
    """Use wildcards to include/exclude nested keys."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "headers": {
                        "x-id": "123",
                        "x-token": "secret",
                        "content-type": "text/html",
                    },
                    "metadata": {
                        "timestamp": "2024-01-01",
                        "requestId": "req-456",
                    },
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=["headers.*", "metadata.*"],
        exclude_patterns=["*token*"],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert "headers" in output
    assert "x-id" in output["headers"]
    assert "x-token" not in output["headers"]  # Excluded

    assert "metadata" in output
    assert "timestamp" in output["metadata"]
    assert "requestId" in output["metadata"]


def test_all_keys_filtered_out():
    """When all keys filtered, result is None (for visibility)."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "password": "secret",
                    "token": "xyz",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=["*password*", "*token*"],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    # All keys filtered out -> None (not empty dict)
    assert output is None


def test_case_sensitive_matching():
    """Pattern matching is case-sensitive."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={
                    "UserId": "123",
                    "userId": "456",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=["userId"],  # Lowercase only
        exclude_patterns=[],
        max_value_len=10000,
    )

    output = extracted["TestNode"]["runs"][0]["output"]
    assert "userId" in output
    assert "UserId" not in output


def test_input_filtering_separate():
    """Input and output filtered independently."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                inputOverride={
                    "requestId": "req-123",
                    "password": "secret-in",
                },
                data={
                    "responseId": "res-456",
                    "token": "secret-out",
                },
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=[],
        exclude_patterns=["*password*", "*token*"],
        max_value_len=10000,
    )

    run_entry = extracted["TestNode"]["runs"][0]

    # Input filtered
    assert "requestId" in run_entry["input"]
    assert "password" not in run_entry["input"]

    # Output filtered
    assert "responseId" in run_entry["output"]
    assert "token" not in run_entry["output"]


def test_extraction_config_in_metadata():
    """Extraction config included in _meta when patterns specified."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={"key": "value"},
            )
        ]
    }

    extracted = extract_nodes_data(
        run_data=run_data,
        extraction_nodes=["TestNode"],
        include_patterns=["userId", "*Id"],
        exclude_patterns=["*password*", "*secret*"],
        max_value_len=10000,
    )

    meta = extracted["_meta"]
    assert "extraction_config" in meta
    config = meta["extraction_config"]

    assert config["include_keys"] == ["userId", "*Id"]
    assert config["exclude_keys"] == ["*password*", "*secret*"]


def test_no_config_when_no_patterns():
    """_meta omits extraction_config when no patterns specified."""
    run_data = {
        "TestNode": [
            NodeRun(
                startTime=1704067200000,
                executionTime=100,
                executionStatus="success",
                data={"key": "value"},
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

    meta = extracted["_meta"]
    assert "extraction_config" not in meta
