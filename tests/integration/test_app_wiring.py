import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_healthz(engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)

    from ext.app import build_app
    app = build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_app_mounts_kb_admin_and_retrieval(engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from ext.app import build_app
    app = build_app()
    paths = {r.path for r in app.routes}
    assert "/api/kb" in paths
    assert "/api/kb/available" in paths
    assert "/api/chats/{chat_id}/kb_config" in paths
    assert "/healthz" in paths
