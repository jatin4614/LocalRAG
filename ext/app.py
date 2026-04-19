"""FastAPI application entry point for the KB management + retrieval + RAG API."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import clear_settings_cache, get_settings
from .db.session import make_engine, make_sessionmaker
from .routers import kb_admin, kb_retrieval, rag, upload
from .services.embedder import TEIEmbedder
from .services.vector_store import VectorStore

_logger = logging.getLogger(__name__)


def _mount_metrics(app: FastAPI) -> None:
    """Mount ``/metrics`` via prometheus_client's ASGI app.

    Fail-open: if ``prometheus_client`` is not installed (it is listed
    in ``[project].dependencies`` but may be absent in slim test envs)
    we log a warning and skip mounting — the rest of the app still
    boots normally.
    """
    try:
        from prometheus_client import make_asgi_app

        app.mount("/metrics", make_asgi_app())
    except Exception as e:  # pragma: no cover - defensive
        _logger.warning("prometheus_client unavailable — /metrics disabled: %s", e)


def build_app() -> FastAPI:
    clear_settings_cache()
    settings = get_settings()
    engine = make_engine(settings.async_database_url)
    SessionLocal = make_sessionmaker(engine)

    vs = VectorStore(url=settings.qdrant_url, vector_size=settings.vector_size)
    emb = TEIEmbedder(base_url=settings.tei_url)

    kb_admin.configure(sessionmaker=SessionLocal, vector_store=vs)
    kb_retrieval.set_sessionmaker(SessionLocal)
    upload.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    rag.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)

    app = FastAPI(title="orgchat-kb", version="0.4.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/api/kb/admin-ui", response_class=HTMLResponse)
    async def kb_admin_ui():
        html = Path(__file__).parent / "static" / "kb-admin.html"
        return HTMLResponse(html.read_text())

    # Retrieval router MUST be registered before admin router to avoid /available shadowing.
    app.include_router(kb_retrieval.router)
    app.include_router(kb_admin.router)
    app.include_router(upload.router)
    app.include_router(rag.router)

    # P2.5 — Prometheus metrics. Fail-open if prometheus_client missing.
    _mount_metrics(app)
    return app


def build_ext_routers():
    """Return the list of APIRouters to mount on an external FastAPI app (upstream).

    Caller is responsible for setting env so our settings + services bootstrap cleanly:
      DATABASE_URL, QDRANT_URL, TEI_URL, RAG_VECTOR_SIZE, WEBUI_SECRET_KEY, AUTH_MODE=jwt.
    """
    from .config import clear_settings_cache, get_settings
    from .db.session import make_engine, make_sessionmaker
    from .routers import kb_admin, kb_retrieval, rag, upload
    from .services import auth as auth_svc
    from .services.embedder import TEIEmbedder
    from .services.vector_store import VectorStore

    clear_settings_cache()
    settings = get_settings()
    engine = make_engine(settings.async_database_url)
    SessionLocal = make_sessionmaker(engine)

    vs = VectorStore(url=settings.qdrant_url, vector_size=settings.vector_size)
    emb = TEIEmbedder(base_url=settings.tei_url)

    auth_svc.configure_jwt(sessionmaker=SessionLocal)
    kb_admin.configure(sessionmaker=SessionLocal, vector_store=vs)
    kb_retrieval.set_sessionmaker(SessionLocal)
    upload.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    rag.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)

    # Configure RAG bridge for middleware injection
    from .services import chat_rag_bridge
    chat_rag_bridge.configure(vector_store=vs, embedder=emb, sessionmaker=SessionLocal)

    # Admin UI page (standalone HTML — no Svelte needed)
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse as HR
    ui_router = APIRouter()

    @ui_router.get("/api/kb/admin-ui", response_class=HR)
    async def _kb_admin_ui():
        html = Path(__file__).parent / "static" / "kb-admin.html"
        return HR(html.read_text())

    # Retrieval first — avoids /available shadowing by admin's /{kb_id}.
    return [ui_router, kb_retrieval.router, kb_admin.router, upload.router, rag.router]
