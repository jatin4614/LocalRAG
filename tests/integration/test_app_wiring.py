import pytest
from httpx import AsyncClient, ASGITransport

pytestmark = pytest.mark.integration


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


@pytest.mark.asyncio
async def test_app_mounts_upload_and_rag(engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    monkeypatch.setenv("TEI_URL", "http://localhost:80")
    from ext.app import build_app
    app = build_app()
    paths = {r.path for r in app.routes}
    assert "/api/kb/{kb_id}/subtag/{subtag_id}/upload" in paths
    assert "/api/chats/{chat_id}/private_docs/upload" in paths
    assert "/api/rag/retrieve" in paths


@pytest.mark.asyncio
async def test_admin_ui_requires_admin(engine, monkeypatch):
    """M1 — ``GET /api/kb/admin-ui`` must be auth-gated to admins.

    Previously the route served the static HTML to any caller (including
    unauthenticated ones). The HTML was XSS-prone (C1) so any caller
    that could pull it could potentially exfiltrate JWTs from a co-resident
    admin's localStorage. Defense-in-depth: gate the page itself.

    Stub auth mode is used here — no JWT, just X-User-Role header — so
    the test exercises the dependency wiring, not the JWT verifier.
    """
    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    monkeypatch.setenv("AUTH_MODE", "stub")

    from ext.app import build_app
    app = build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        # No auth at all → 401.
        r = await c.get("/api/kb/admin-ui")
        assert r.status_code in (401, 403), r.text

        # Non-admin role → 403.
        r = await c.get(
            "/api/kb/admin-ui",
            headers={"X-User-Id": "u1", "X-User-Role": "user"},
        )
        assert r.status_code == 403, r.text

        # Admin → 200 + HTML body.
        r = await c.get(
            "/api/kb/admin-ui",
            headers={"X-User-Id": "admin1", "X-User-Role": "admin"},
        )
        assert r.status_code == 200, r.text
        assert "<html" in r.text.lower()
