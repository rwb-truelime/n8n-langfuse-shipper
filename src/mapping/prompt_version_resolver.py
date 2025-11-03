"""Langfuse API prompt version resolution.

This module handles cross-environment prompt version resolution by querying
the Langfuse API for available prompt versions. When the production version
doesn't exist in the target environment (e.g., dev), it falls back to the
latest available version.

Key features:
- Environment-aware: Only resolves in non-production environments
- Caching: Caches API results per export run to avoid redundant calls
- Safe fallback: Uses latest available version when exact match not found
- Full transparency: Emits debug metadata showing original vs resolved versions
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class PromptVersionResolver:
    """Resolves prompt versions across Langfuse environments.

    Queries Langfuse API to find available versions and maps production
    versions to dev/staging equivalents when needed.
    """

    def __init__(
        self,
        langfuse_host: str,
        langfuse_public_key: str,
        langfuse_secret_key: str,
        environment: str = "production",
        timeout: int = 5,
    ):
        """Initialize version resolver.

        Args:
            langfuse_host: Langfuse API base URL
            langfuse_public_key: Public API key
            langfuse_secret_key: Secret API key
            environment: Target environment (production|dev|staging)
            timeout: API request timeout in seconds
        """
        self.langfuse_host = langfuse_host.rstrip("/")
        self.auth = (langfuse_public_key, langfuse_secret_key)
        self.environment = environment
        self.timeout = timeout

        # Cache: prompt_name → list of available versions
        self._version_cache: Dict[str, List[int]] = {}

    def resolve_version(
        self, prompt_name: str, original_version: int
    ) -> tuple[int, str]:
        """Resolve prompt version for target environment.

        Args:
            prompt_name: Name of the prompt
            original_version: Version from source environment (production)

        Returns:
            Tuple of (resolved_version, resolution_source)
            resolution_source: "passthrough" | "exact_match" | "fallback_latest"
        """
        # Production: always use original version (no resolution)
        if self.environment == "production":
            logger.debug(
                f"Environment is production; using original version "
                f"{original_version} for prompt '{prompt_name}'"
            )
            return original_version, "passthrough"

        # Non-production: check if version exists in target environment
        available_versions = self._get_available_versions(prompt_name)

        if not available_versions:
            logger.warning(
                f"No versions found for prompt '{prompt_name}' in "
                f"environment '{self.environment}'. Using original "
                f"version {original_version} (may fail in Langfuse UI)"
            )
            return original_version, "not_found"

        # Check for exact match
        if original_version in available_versions:
            logger.debug(
                f"Exact version match: prompt '{prompt_name}' version "
                f"{original_version} exists in '{self.environment}'"
            )
            return original_version, "exact_match"

        # Fallback to latest available version
        latest_version = max(available_versions)
        logger.info(
            f"Version {original_version} of prompt '{prompt_name}' not "
            f"found in '{self.environment}'. Falling back to latest "
            f"available version: {latest_version}"
        )
        return latest_version, "fallback_latest"

    def _get_available_versions(self, prompt_name: str) -> List[int]:
        """Query Langfuse API for available prompt versions.

        Caches results per export run to avoid redundant API calls.

        Args:
            prompt_name: Name of the prompt

        Returns:
            List of available version numbers (sorted)
        """
        # Check cache
        if prompt_name in self._version_cache:
            return self._version_cache[prompt_name]

        # Query API
        try:
            url = f"{self.langfuse_host}/api/public/prompts/{prompt_name}"
            response = httpx.get(
                url, auth=self.auth, timeout=self.timeout
            )

            if response.status_code == 404:
                logger.warning(
                    f"Prompt '{prompt_name}' not found in Langfuse "
                    f"environment '{self.environment}'"
                )
                self._version_cache[prompt_name] = []
                return []

            response.raise_for_status()
            prompt_data = response.json()

            # Extract versions
            # API response structure may vary; handle both direct version list
            # and nested structure
            versions = []
            if isinstance(prompt_data, dict):
                # Check for versions array
                if "versions" in prompt_data:
                    versions = [
                        v["version"]
                        for v in prompt_data["versions"]
                        if "version" in v
                    ]
                # Or single version in prompt object
                elif "version" in prompt_data:
                    versions = [prompt_data["version"]]

            versions.sort()
            self._version_cache[prompt_name] = versions

            logger.debug(
                f"Fetched {len(versions)} versions for prompt "
                f"'{prompt_name}': {versions}"
            )
            return versions

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching versions for prompt '{prompt_name}': "
                f"{e.response.status_code} - {e.response.text}"
            )
            self._version_cache[prompt_name] = []
            return []

        except httpx.RequestError as e:
            logger.error(
                f"Request error fetching versions for prompt "
                f"'{prompt_name}': {e}"
            )
            self._version_cache[prompt_name] = []
            return []

        except Exception as e:
            logger.error(
                f"Unexpected error fetching versions for prompt "
                f"'{prompt_name}': {e}"
            )
            self._version_cache[prompt_name] = []
            return []

    def clear_cache(self):
        """Clear version cache. Useful between export runs."""
        self._version_cache.clear()


def create_version_resolver_from_env() -> Optional[PromptVersionResolver]:
    """Create PromptVersionResolver from environment variables.

    Required environment variables:
    - LANGFUSE_HOST
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY

    Optional:
    - LANGFUSE_ENV (default: "production")
    - PROMPT_VERSION_API_TIMEOUT (default: 5 seconds)

    Returns:
        PromptVersionResolver instance or None if credentials missing
    """
    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not all([host, public_key, secret_key]):
        logger.warning(
            "Langfuse credentials not fully configured. Prompt version "
            "resolution disabled. Set LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, "
            "and LANGFUSE_SECRET_KEY environment variables."
        )
        return None

    environment = os.getenv("LANGFUSE_ENV", "production")
    timeout = int(os.getenv("PROMPT_VERSION_API_TIMEOUT", "5"))

    logger.info(
        f"Created prompt version resolver for environment: {environment}"
    )
    return PromptVersionResolver(
        langfuse_host=host,  # type: ignore[arg-type]
        langfuse_public_key=public_key,  # type: ignore[arg-type]
        langfuse_secret_key=secret_key,  # type: ignore[arg-type]
        environment=environment,
        timeout=timeout,
    )
