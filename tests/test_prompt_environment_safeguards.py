"""Test environment safeguards for prompt version resolution.

Verifies that production environment never makes API calls and that
dev/staging environments properly query the API with correct metadata.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from src.mapping.prompt_version_resolver import PromptVersionResolver


def test_production_never_queries_api():
    """Verify production environment strictly passes through without API calls."""
    resolver = PromptVersionResolver(
        langfuse_host="https://prod.langfuse.com",
        langfuse_public_key="pk-prod",
        langfuse_secret_key="sk-prod",
        environment="production",
        timeout=5,
    )

    with patch("httpx.get") as mock_get:
        # Multiple resolve calls in production
        resolver.resolve_version("Prompt A", 10)
        resolver.resolve_version("Prompt B", 20)
        resolver.resolve_version("Prompt C", 30)

        # Should NEVER call API
        assert mock_get.call_count == 0

        # Cache should remain empty
        assert len(resolver._version_cache) == 0


def test_dev_environment_queries_api():
    """Verify dev environment makes API calls for version resolution."""
    resolver = PromptVersionResolver(
        langfuse_host="https://dev.langfuse.com",
        langfuse_public_key="pk-dev",
        langfuse_secret_key="sk-dev",
        environment="dev",
        timeout=5,
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Dev Prompt",
        "version": 10,  # v2 returns single latest version
        "labels": ["latest", "production"]
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        resolver.resolve_version("Dev Prompt", 10)

        # Should call API in dev
        assert mock_get.call_count == 1

        # Verify dev host used
        call_args = mock_get.call_args
        assert "dev.langfuse.com" in call_args.args[0]


def test_staging_environment_queries_api():
    """Verify staging environment makes API calls for version resolution."""
    resolver = PromptVersionResolver(
        langfuse_host="https://staging.langfuse.com",
        langfuse_public_key="pk-staging",
        langfuse_secret_key="sk-staging",
        environment="staging",
        timeout=5,
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Staging Prompt",
        "version": 2,  # v2 returns single latest version
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        resolver.resolve_version("Staging Prompt", 2)

        # Should call API in staging
        assert mock_get.call_count == 1


def test_resolution_source_metadata():
    """Verify resolution_source metadata reflects environment behavior."""
    # Production: passthrough
    prod_resolver = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="production",
    )

    version, source = prod_resolver.resolve_version("Test", 42)
    assert source == "passthrough"
    assert version == 42

    # Dev: exact_match
    dev_resolver = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="dev",
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Test",
        "version": 42,
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response):
        version, source = dev_resolver.resolve_version("Test", 42)
        assert source == "exact_match"
        assert version == 42


def test_fallback_latest_in_non_production():
    """Verify fallback_latest only occurs in non-production environments."""
    resolver = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="dev",
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Fallback Test",
        "version": 12,  # v2 always returns latest labeled version
        "labels": ["latest", "production"]
    }

    with patch("httpx.get", return_value=mock_response):
        version, source = resolver.resolve_version("Fallback Test", 100)

        assert version == 12  # Latest available
        assert source == "fallback_latest"


def test_environment_case_insensitive():
    """Verify environment string is case-sensitive (lowercase required)."""
    # PRODUCTION (uppercase) should NOT pass through - treated as non-prod
    resolver = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="PRODUCTION",  # Uppercase - not recognized as production
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Test",
        "version": 10,
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        version, source = resolver.resolve_version("Test", 10)

        # Should query API (uppercase not recognized)
        assert mock_get.call_count == 1
        assert source == "exact_match"  # Found version 10 in response


def test_unknown_environment_treated_as_non_production():
    """Verify unknown environment values behave like dev (query API)."""
    resolver = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="custom-env",  # Unknown value
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Custom Env Test",
        "version": 5,
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        resolver.resolve_version("Custom Env Test", 5)

        # Should query API (not production passthrough)
        assert mock_get.call_count == 1


def test_production_with_version_not_in_api():
    """Verify production uses original version even if it doesn't exist in API."""
    resolver = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="production",
    )

    # Production should never check API, so version always accepted
    version, source = resolver.resolve_version("Any Prompt", 999999)

    assert version == 999999
    assert source == "passthrough"


def test_multiple_environments_separate_resolvers():
    """Verify multiple resolvers with different environments work independently."""
    prod = PromptVersionResolver(
        langfuse_host="https://prod.com",
        langfuse_public_key="pk-prod",
        langfuse_secret_key="sk-prod",
        environment="production",
    )

    dev = PromptVersionResolver(
        langfuse_host="https://dev.com",
        langfuse_public_key="pk-dev",
        langfuse_secret_key="sk-dev",
        environment="dev",
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Test",
        "version": 5,
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response) as mock_get:
        # Production: no API call
        prod_version, prod_source = prod.resolve_version("Test", 10)
        assert mock_get.call_count == 0
        assert prod_source == "passthrough"

        # Dev: API call
        dev_version, dev_source = dev.resolve_version("Test", 10)
        assert mock_get.call_count == 1
        # Version 10 not in mock response (only 5), so fallback to latest
        assert dev_source == "fallback_latest"
        assert dev_version == 5  # Latest available


def test_environment_isolation():
    """Verify environment setting doesn't leak between instances."""
    resolver1 = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="production",
    )

    resolver2 = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="dev",
    )

    # Resolver1 should still be production
    assert resolver1.environment == "production"
    assert resolver2.environment == "dev"

    with patch("httpx.get") as mock_get:
        resolver1.resolve_version("Test", 1)
        assert mock_get.call_count == 0  # Still production


def test_cache_isolation_between_environments():
    """Verify cache doesn't leak between different environment resolvers."""
    prod = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="production",
    )

    dev = PromptVersionResolver(
        langfuse_host="https://test.com",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        environment="dev",
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "Cached",
        "version": 5,
        "labels": ["latest"]
    }

    with patch("httpx.get", return_value=mock_response):
        # Dev populates cache
        dev.resolve_version("Cached", 5)
        assert len(dev._version_cache) == 1

        # Prod cache should be empty
        assert len(prod._version_cache) == 0

        # Prod operation doesn't affect dev cache
        prod.resolve_version("Cached", 5)
        assert len(prod._version_cache) == 0
        assert len(dev._version_cache) == 1
