from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_has_healthcheck():
    content = (ROOT / "model_manager/Dockerfile").read_text()
    assert "HEALTHCHECK" in content
    assert "uvicorn" in content


def test_requirements_lists_fastapi_httpx():
    content = (ROOT / "model_manager/requirements.txt").read_text()
    for pkg in ["fastapi", "uvicorn", "httpx", "pydantic-settings"]:
        assert pkg in content, f"missing dep: {pkg}"
