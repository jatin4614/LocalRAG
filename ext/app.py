"""FastAPI application entry point for the KB management + retrieval API."""
from __future__ import annotations

from fastapi import FastAPI

from .config import clear_settings_cache, get_settings
from .db.session import make_engine, make_sessionmaker
from .routers import kb_admin, kb_retrieval


def build_app() -> FastAPI:
    clear_settings_cache()
    settings = get_settings()
    engine = make_engine(settings.database_url)
    SessionLocal = make_sessionmaker(engine)

    kb_admin.set_sessionmaker(SessionLocal)
    kb_retrieval.set_sessionmaker(SessionLocal)

    app = FastAPI(title="orgchat-kb", version="0.2.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    app.include_router(kb_admin.router)
    app.include_router(kb_retrieval.router)
    return app


app = None  # uvicorn entrypoint would call build_app(); tests call build_app() directly
