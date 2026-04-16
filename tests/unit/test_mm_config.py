import pytest
from model_manager.config import ModelManagerSettings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("VISION_URL", "http://v:8000")
    monkeypatch.setenv("WHISPER_URL", "http://w:9000")
    s = ModelManagerSettings()
    assert s.vision_url == "http://v:8000"
    assert s.whisper_url == "http://w:9000"
    assert s.idle_secs == 300
    assert s.poll_secs == 15
    assert s.host == "0.0.0.0"
    assert s.port == 8080


def test_settings_override_idle(monkeypatch):
    monkeypatch.setenv("VISION_URL", "http://v:8000")
    monkeypatch.setenv("WHISPER_URL", "http://w:9000")
    monkeypatch.setenv("MODEL_UNLOAD_IDLE_SECS", "60")
    monkeypatch.setenv("MODEL_MANAGER_POLL_SECS", "5")
    s = ModelManagerSettings()
    assert s.idle_secs == 60
    assert s.poll_secs == 5


def test_settings_require_backend_urls(monkeypatch):
    monkeypatch.delenv("VISION_URL", raising=False)
    monkeypatch.delenv("WHISPER_URL", raising=False)
    with pytest.raises(Exception):
        ModelManagerSettings()
