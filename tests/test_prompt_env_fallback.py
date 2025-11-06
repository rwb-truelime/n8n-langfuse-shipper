from __future__ import annotations

import os
from unittest.mock import Mock, patch

from n8n_langfuse_shipper.mapping.prompt_version_resolver import create_version_resolver_from_env


def test_env_fallback_loads_credentials(tmp_path, monkeypatch):
    # Create temporary .env file with credentials
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
LANGFUSE_HOST=https://fallback.langfuse.local
LANGFUSE_PUBLIC_KEY=pk-fallback
LANGFUSE_SECRET_KEY=sk-fallback
LANGFUSE_ENV=dev
        """.strip()
    )
    # Switch cwd to temp path so fallback loader sees .env
    monkeypatch.chdir(tmp_path)
    # Ensure variables are NOT pre-set in environment
    for k in ["LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_ENV"]:
        os.environ.pop(k, None)
    resolver = create_version_resolver_from_env()
    assert resolver is not None, "Resolver should be created from .env fallback"
    assert resolver.langfuse_host == "https://fallback.langfuse.local"
    assert resolver.auth == ("pk-fallback", "sk-fallback")
    assert resolver.environment == "dev"

    # Mock API to assert that version resolution now works (env_latest override)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "X", "version": 3}
    with patch("httpx.get", return_value=mock_response):
        version, source = resolver.resolve_version("X", 99)
    assert version == 3 and source == "env_latest"
