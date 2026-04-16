"""Integration test for model-manager with stubbed vllm-vision + whisper backends."""
import pytest
import pytest_asyncio
import httpx
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


def _stub_backend(state: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"state": state["state"]}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "stub"}]}

    @app.post("/wake_up")
    async def wake():
        state["state"] = "awake"
        return {"state": "awake"}

    @app.post("/sleep")
    async def sleep():
        state["state"] = "asleep"
        return {"state": "asleep"}

    return app


@pytest_asyncio.fixture
async def vision_state():
    return {"state": "asleep"}


@pytest_asyncio.fixture
async def whisper_state():
    return {"state": "asleep"}


@pytest_asyncio.fixture
async def mm_client(vision_state, whisper_state, monkeypatch):
    vision_app  = _stub_backend(vision_state)
    whisper_app = _stub_backend(whisper_state)

    from model_manager.client import ModelClient
    from model_manager.tracker import IdleTracker
    from model_manager.app import build_app

    vision_client = ModelClient(
        base_url="http://stub-vision", health_path="/v1/models",
        transport=httpx.ASGITransport(app=vision_app),
    )
    whisper_client = ModelClient(
        base_url="http://stub-whisper", health_path="/health",
        transport=httpx.ASGITransport(app=whisper_app),
    )

    monkeypatch.setenv("VISION_URL", "http://stub-vision")
    monkeypatch.setenv("WHISPER_URL", "http://stub-whisper")
    app = build_app(
        clients={"vision": vision_client, "whisper": whisper_client},
        tracker=IdleTracker(now_fn=lambda: 1000.0),
        start_sleeper=False,
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    await vision_client.aclose()
    await whisper_client.aclose()


@pytest.mark.asyncio
async def test_status_lists_all_models(mm_client):
    r = await mm_client.get("/models/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"vision", "whisper"}


@pytest.mark.asyncio
async def test_wake_updates_state(mm_client, vision_state):
    r = await mm_client.post("/models/vision/wake")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "awake"
    assert vision_state["state"] == "awake"
    r = await mm_client.get("/models/status")
    assert r.json()["vision"]["state"] == "awake"


@pytest.mark.asyncio
async def test_sleep_updates_state(mm_client, whisper_state):
    await mm_client.post("/models/whisper/wake")
    r = await mm_client.post("/models/whisper/sleep")
    assert r.status_code == 200
    assert r.json()["state"] == "asleep"
    assert whisper_state["state"] == "asleep"


@pytest.mark.asyncio
async def test_touch_does_not_wake(mm_client, vision_state):
    r = await mm_client.post("/models/vision/touch")
    assert r.status_code == 200
    assert vision_state["state"] == "asleep"
    r = await mm_client.get("/models/status")
    assert r.json()["vision"]["state"] == "asleep"


@pytest.mark.asyncio
async def test_unknown_model_404(mm_client):
    r = await mm_client.post("/models/chat/wake")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_healthz(mm_client):
    r = await mm_client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
