import sys
from pathlib import Path

import pytest

from n8n_langfuse_shipper.config import get_settings

# Ensure project root (containing `src`) is on sys.path for tests when not installed editable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_filter_ai_mode_env(monkeypatch):
    monkeypatch.delenv("FILTER_AI_MODE", raising=False)
    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
