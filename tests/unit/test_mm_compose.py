from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_model_manager_service_in_compose():
    c = (ROOT / "compose/docker-compose.yml").read_text()
    assert "model-manager:" in c
    assert "VISION_URL: http://vllm-vision:8000" in c
    assert "WHISPER_URL: http://whisper:9000" in c
    assert "../model_manager" in c


def test_env_example_has_model_manager_keys():
    c = (ROOT / "compose/.env.example").read_text()
    assert "MODEL_UNLOAD_IDLE_SECS=" in c
