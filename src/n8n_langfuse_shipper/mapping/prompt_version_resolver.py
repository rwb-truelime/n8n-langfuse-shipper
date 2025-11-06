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

        In production we passthrough the original version (deterministic
        replay guarantee). In non-production (dev/staging) we **always**
        map to the latest available version for the prompt name to ensure
        a valid linkage even when the production version does not exist.

        Args:
            prompt_name: Name of the prompt
            original_version: Version embedded in execution data

        Returns:
            (resolved_version, resolution_source)
            resolution_source values:
            - "passthrough" (production, unchanged)
            - "env_latest" (non-production mapped to latest available)
            - "not_found" (no versions available in target env)
        """
        if self.environment == "production":
            logger.debug(
                f"Environment is production; using original version "
                f"{original_version} for prompt '{prompt_name}'"
            )
            return original_version, "passthrough"

        available_versions = self._get_available_versions(prompt_name)
        if not available_versions:
            logger.warning(
                f"No versions found for prompt '{prompt_name}' in "
                f"environment '{self.environment}'. Using original "
                f"version {original_version} (may fail in Langfuse UI)"
            )
            return original_version, "not_found"

        latest_version = max(available_versions)
        if latest_version != original_version:
            logger.info(
                f"Mapping prompt '{prompt_name}' version {original_version} "
                f"→ latest {latest_version} for env '{self.environment}'"
            )
        else:
            logger.debug(
                f"Prompt '{prompt_name}' original version {original_version} "
                f"already equals latest for env '{self.environment}'"
            )
        return latest_version, "env_latest"

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

        # Query API (use v2 endpoint for OSS compatibility)
        try:
            url = f"{self.langfuse_host}/api/public/v2/prompts/{prompt_name}"
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

            # V2 API returns single prompt version (typically 'latest')
            # Extract version number
            versions = []
            if isinstance(prompt_data, dict) and "version" in prompt_data:
                version = prompt_data["version"]
                versions = [version]
                logger.debug(
                    f"Fetched prompt '{prompt_name}' v{version} "
                    f"(labels: {prompt_data.get('labels', [])})"
                )

            versions.sort()
            self._version_cache[prompt_name] = versions

            if not versions:
                logger.warning(
                    f"No version found in API response for prompt "
                    f"'{prompt_name}'"
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

    def clear_cache(self) -> None:
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

    # Fallback: attempt lightweight .env parsing if any credential missing.
    # This allows running CLI with a populated project .env file without
    # explicit fish exports. We purposely avoid adding a dependency.
    if not all([host, public_key, secret_key]):
        env_path = os.path.join(os.getcwd(), ".env")
        try:
            if os.path.isfile(env_path):
                with open(env_path, "r", encoding="utf-8") as f:  # noqa: PTH123
                    for line in f:
                        # Ignore comments / blank lines
                        if not line.strip() or line.strip().startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\n\r")
                        # Do not override existing env variables
                        if k and v and k not in os.environ:
                            os.environ[k] = v
                # Re-fetch after parsing
                host = os.getenv("LANGFUSE_HOST")
                public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to parse .env for Langfuse credentials: {e}")

    if not all([host, public_key, secret_key]):
        logger.warning(
            "Langfuse credentials not fully configured. Prompt version "
            "resolution disabled. Set LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, "
            "and LANGFUSE_SECRET_KEY environment variables (or ensure .env is parsed)."
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
