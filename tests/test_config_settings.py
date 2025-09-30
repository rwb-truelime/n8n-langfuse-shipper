from __future__ import annotations

import os
from importlib import reload

from src import config as config_module


def _reload_with_env(env: dict):
    # Clear any cached settings
    if "Settings" in config_module.__dict__:
        pass
    for k in list(os.environ.keys()):
        if k.startswith("PG_DSN") or k.startswith("DB_POSTGRESDB_"):
            # allow override by test values
            pass
    # Ensure table prefix does not leak between scenarios unless explicitly set
    os.environ.pop("DB_TABLE_PREFIX", None)
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    # Bust lru_cache by reloading module
    reload(config_module)
    return config_module.get_settings()


def test_required_prefix(monkeypatch):
    # PG_DSN provided with explicit prefix
    s = _reload_with_env(
        {
            "PG_DSN": "postgresql://u:p@h:5432/db1",
            "DB_TABLE_PREFIX": "n8n_",
        }
    )
    assert s.PG_DSN.endswith("/db1")
    assert s.DB_TABLE_PREFIX == "n8n_"

    # Build DSN from components with empty prefix
    s = _reload_with_env(
        {
            "PG_DSN": "",
            "DB_POSTGRESDB_HOST": "localhost",
            "DB_POSTGRESDB_DATABASE": "n8n",
            "DB_POSTGRESDB_USER": "n8n",
            "DB_POSTGRESDB_PASSWORD": "pw",
            "DB_TABLE_PREFIX": "",
        }
    )
    assert "pw@localhost" in s.PG_DSN
    assert s.DB_TABLE_PREFIX == ""

    # Omit prefix scenario: .env provides blank DB_TABLE_PREFIX so result is blank
    s = _reload_with_env(
        {
            "PG_DSN": "postgresql://u@h:5432/db2",
        }
    )
    assert s.DB_TABLE_PREFIX == ""

    # Custom prefix
    s = _reload_with_env(
        {
            "PG_DSN": "postgresql://u@h:5432/db4",
            "DB_TABLE_PREFIX": "custom_",
        }
    )
    assert s.DB_TABLE_PREFIX == "custom_"
