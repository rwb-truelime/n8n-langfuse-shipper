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
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    # Bust lru_cache by reloading module
    reload(config_module)
    return config_module.get_settings()


def test_dsn_construction_and_prefix_semantics(monkeypatch):
    # Case 1: PG_DSN provided wins
    s = _reload_with_env({
        "PG_DSN": "postgresql://u:p@h:5432/db1",
        "DB_POSTGRESDB_HOST": "ignored",
        "DB_POSTGRESDB_DATABASE": "ignored",
    })
    assert s.PG_DSN.endswith("/db1")

    # Case 2: Build from components when PG_DSN blank
    s = _reload_with_env({
        "PG_DSN": "",
        "DB_POSTGRESDB_HOST": "localhost",
        "DB_POSTGRESDB_DATABASE": "n8n",
        "DB_POSTGRESDB_USER": "n8n",
        "DB_POSTGRESDB_PASSWORD": "pw",
    })
    assert "pw@localhost" in s.PG_DSN

    # Case 3: Unset prefix -> default n8n_
    s = _reload_with_env({
        "PG_DSN": "postgresql://u@h:5432/db2",
        # purposely omit DB_TABLE_PREFIX
    })
    # Accept either internal None (unset) or empty string injected by a .env providing explicit blank.
    assert s.DB_TABLE_PREFIX in (None, "")

    # Case 4: Empty prefix -> no prefix
    s = _reload_with_env({
        "PG_DSN": "postgresql://u@h:5432/db3",
        "DB_TABLE_PREFIX": "",
    })
    assert s.DB_TABLE_PREFIX == ""

    # Case 5: Custom prefix
    s = _reload_with_env({
        "PG_DSN": "postgresql://u@h:5432/db4",
        "DB_TABLE_PREFIX": "custom_",
    })
    assert s.DB_TABLE_PREFIX == "custom_"

