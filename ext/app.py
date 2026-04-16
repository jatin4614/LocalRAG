"""FastAPI application entry point for the KB management + retrieval + RAG API."""
from __future__ import annotations

from fastapi import FastAPI

from .config import clear_settings_cache, get_settings
from .db.session import make_engine, make_sessionmaker
from .routers import kb_admin, kb_retrieval, rag, upload
from .services.embedder import TEIEmbedder
from .services.vector_store import VectorStore


def build_app() -> FastAPI:
    clear_settings_cache()
    settings = get_settings()
    engine = make_engine(settings.database_url)
    SessionLocal = make_sessionmaker(engine)

    vs = VectorStore(url=settings.qdrant_url, vector_size=settings.vector_size)
    emb = TEIEmbedder(base_url=settings.tei_url)

    kb_admin.set_sessionmaker(SessionLocal)
    kb_retrieval.set_sessionmaker(SessionLocal)
    upload.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)
    rag.configure(sessionmaker=SessionLocal, vector_store=vs, embedder=emb)

    app = FastAPI(title="orgchat-kb", version="0.4.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    # Retrieval router MUST be registered before admin router to avoid /available shadowing.
    app.include_router(kb_retrieval.router)
    app.include_router(kb_admin.router)
    app.include_router(upload.router)
    app.include_router(rag.router)
    return app


app = None
