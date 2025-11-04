"""Test prompt version resolver with Langfuse API integration.

Tests API querying, caching, environment-aware version resolution,
and fallback behavior.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
import httpx

from n8n_langfuse_shipper.mapping.prompt_version_resolver import (
    PromptVersionResolver,
    create_version_resolver_from_env,
)


@pytest.fixture
def mock_resolver():
    """Create resolver with mock credentials for testing."""
    return PromptVersionResolver(
        langfuse_host="https://test.langfuse.com",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        environment="dev",
        timeout=5,
    )


def test_production_passthrough(mock_resolver):
    """Verify production environment uses original version (no API calls)."""
    resolver = PromptVersionResolver(
        langfuse_host="https://test.langfuse.com",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        environment="production",
        timeout=5,
    )

    # Should not make any API calls
    resolved_version, source = resolver.resolve_version(
        prompt_name="Test Prompt",
        original_version=58,
    )

    assert resolved_version == 58
    assert source == "passthrough"
    # Cache should be empty (no API calls made)
    assert len(resolver._version_cache) == 0


def test_exact_version_match(mock_resolver):
    """Verify exact version match when version exists in target env."""
    # Mock API response (v2 API returns single prompt object)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Test Prompt",
        "version": 58,
        "labels": ["production", "latest"],
    }

    with patch("httpx.get", return_value=mock_response):
        resolved_version, source = mock_resolver.resolve_version(
            prompt_name="Test Prompt",
            original_version=58,
        )

    assert resolved_version == 58
    assert source == "exact_match"


def test_fallback_to_latest(mock_resolver):
    """Verify fallback to latest when original version missing."""
    # Mock API response (v2 API returns latest version)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Test Prompt",
        "version": 15,  # Latest available in dev
        "labels": ["production", "latest"],
    }

    with patch("httpx.get", return_value=mock_response):
        resolved_version, source = mock_resolver.resolve_version(
            prompt_name="Test Prompt",
            original_version=58,  # Not available
        )

    assert resolved_version == 15  # Latest available
    assert source == "fallback_latest"


def test_caching_behavior(mock_resolver):
    """Verify API results are cached per export run."""
    # Mock API response (v2 API)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Cached Prompt",
        "version": 1,
        "labels": ["latest"],
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        # First call - should hit API
        v1, _ = mock_resolver.resolve_version("Cached Prompt", 2)
        assert v1 == 1
        assert mock_get.call_count == 1

        # Second call - should use cache
        v2, _ = mock_resolver.resolve_version("Cached Prompt", 2)
        assert v2 == 1
        assert mock_get.call_count == 1  # No additional API call


def test_prompt_not_found_404(mock_resolver):
    """Verify handling of 404 when prompt doesn't exist."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=Mock(), response=mock_response
    )

    with patch("httpx.get", return_value=mock_response):
        resolved_version, source = mock_resolver.resolve_version(
            prompt_name="Nonexistent",
            original_version=10,
        )

    # Should fallback to original version
    assert resolved_version == 10
    assert source == "not_found"


def test_api_timeout_error(mock_resolver):
    """Verify graceful handling of API timeouts."""
    with patch("httpx.get", side_effect=httpx.TimeoutException("Timeout")):
        resolved_version, source = mock_resolver.resolve_version(
            prompt_name="Timeout Prompt",
            original_version=5,
        )

    # Should fallback to original version
    assert resolved_version == 5
    assert source == "not_found"
    # Error should be cached to avoid retries
    assert "Timeout Prompt" in mock_resolver._version_cache


def test_api_http_error(mock_resolver):
    """Verify handling of non-404 HTTP errors."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=Mock(), response=mock_response
    )

    with patch("httpx.get", return_value=mock_response):
        resolved_version, source = mock_resolver.resolve_version(
            prompt_name="Error Prompt",
            original_version=20,
        )

    assert resolved_version == 20
    assert source == "not_found"


def test_empty_versions_list(mock_resolver):
    """Verify handling when API returns response without version field."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Empty Prompt"
        # No version field - prompt exists but no versions available
    }

    with patch("httpx.get", return_value=mock_response):
        resolved_version, source = mock_resolver.resolve_version(
            prompt_name="Empty Prompt",
            original_version=1,
        )

    assert resolved_version == 1
    assert source == "not_found"


def test_single_version_format(mock_resolver):
    """Verify handling of single version (non-array) response format."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Single Version Prompt",
        "version": 42,  # Single version, not array
    }

    with patch("httpx.get", return_value=mock_response):
        resolved_version, source = mock_resolver.resolve_version(
            prompt_name="Single Version Prompt",
            original_version=42,
        )

    assert resolved_version == 42
    assert source == "exact_match"


def test_clear_cache(mock_resolver):
    """Verify cache clearing functionality."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Test",
        "version": 1,
        "labels": ["latest"],
    }

    with patch("httpx.get", return_value=mock_response):
        mock_resolver.resolve_version("Test", 1)
        assert len(mock_resolver._version_cache) == 1

        mock_resolver.clear_cache()
        assert len(mock_resolver._version_cache) == 0


def test_create_from_env_with_all_vars():
    """Verify factory function with complete environment variables."""
    env_vars = {
        "LANGFUSE_HOST": "https://env.langfuse.com",
        "LANGFUSE_PUBLIC_KEY": "pk-env",
        "LANGFUSE_SECRET_KEY": "sk-env",
        "LANGFUSE_ENV": "staging",
        "PROMPT_VERSION_API_TIMEOUT": "10",
    }

    with patch.dict("os.environ", env_vars, clear=True):
        resolver = create_version_resolver_from_env()

    assert resolver is not None
    assert resolver.langfuse_host == "https://env.langfuse.com"
    assert resolver.auth == ("pk-env", "sk-env")
    assert resolver.environment == "staging"
    assert resolver.timeout == 10


def test_create_from_env_missing_credentials():
    """Verify factory returns None when credentials missing."""
    env_vars = {
        "LANGFUSE_HOST": "https://test.langfuse.com",
        # Missing PUBLIC_KEY and SECRET_KEY
    }

    with patch.dict("os.environ", env_vars, clear=True):
        resolver = create_version_resolver_from_env()

    assert resolver is None


def test_create_from_env_default_values():
    """Verify factory uses defaults for optional variables."""
    env_vars = {
        "LANGFUSE_HOST": "https://test.langfuse.com",
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
        # No LANGFUSE_ENV or TIMEOUT specified
    }

    with patch.dict("os.environ", env_vars, clear=True):
        resolver = create_version_resolver_from_env()

    assert resolver is not None
    assert resolver.environment == "production"  # Default
    assert resolver.timeout == 5  # Default


def test_version_sorting(mock_resolver):
    """Verify v2 API returns single latest version (no array sorting)."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Sorted Test",
        "version": 100,  # v2 API returns highest/latest version only
        "labels": ["latest", "production"]
    }

    with patch("httpx.get", return_value=mock_response):
        # Request version that doesn't exist
        resolved_version, source = mock_resolver.resolve_version(
            prompt_name="Sorted Test",
            original_version=999,
        )

    # v2 API always returns latest version (100)
    assert resolved_version == 100
    assert source == "fallback_latest"


def test_auth_credentials_passed(mock_resolver):
    """Verify auth credentials are correctly passed to API."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Auth Test",
        "version": 1,
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        mock_resolver.resolve_version("Auth Test", 1)

        # Verify auth tuple was passed
        call_args = mock_get.call_args
        assert call_args.kwargs["auth"] == ("pk-test", "sk-test")


def test_url_construction(mock_resolver):
    """Verify correct v2 API URL construction."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "My Prompt",
        "version": 1,
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        mock_resolver.resolve_version("My Prompt", 1)

        # Verify v2 API URL format
        call_args = mock_get.call_args
        expected_url = (
            "https://test.langfuse.com/api/public/v2/prompts/My Prompt"
        )
        assert call_args.args[0] == expected_url


def test_timeout_parameter(mock_resolver):
    """Verify timeout parameter is passed to httpx."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Timeout Test",
        "version": 1,
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        mock_resolver.resolve_version("Timeout Test", 1)

        # Verify timeout was passed
        call_args = mock_get.call_args
        assert call_args.kwargs["timeout"] == 5
