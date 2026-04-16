import pytest
from ext.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("REDIS_URL", "redis://r:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://q:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    s = Settings()
    assert s.database_url == "postgresql+asyncpg://u:p@h/d"
    assert s.redis_url == "redis://r:6379/0"
    assert s.qdrant_url == "http://q:6333"
    assert s.session_secret == "x" * 32


def test_session_secret_min_length(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("REDIS_URL", "redis://r:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://q:6333")
    monkeypatch.setenv("SESSION_SECRET", "short")
    with pytest.raises(Exception):
        Settings()
