"""FastAPI application entry point for the KB management + retrieval + RAG API."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import clear_settings_cache, get_settings
from .db.session import make_engine, make_sessionmaker
from .routers import ingest_stream, kb_admin, kb_retrieval, rag, rag_stream, upload
from .services.budget import preflight_tokenizer
from .services.chat_model_preflight import preflight_chat_model
from .services.embedder import TEIEmbedder
from .services.logging_setup import configure_json_logging
from .services.obs import init_observability
from .services.vector_store import VectorStore

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def _rbac_subscriber_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan that runs the RBAC pub/sub subscriber.

    Phase 1.5 follow-up: ``subscribe_invalidations`` must run in the
    background of every replica so RBAC revocations published by another
    replica's ``kb_admin`` mutation propagate within milliseconds (instead
    of waiting up to ``RAG_RBAC_CACHE_TTL_SECS`` for the TTL safety net).

    Single-replica today, but flipping a second worker on without this
    wired would leak revoked KB grants for up to 30 seconds — the kind
    of silent regression the TTL is supposed to bound, not the primary
    invalidation channel.

    Fail-open: any exception starting / running the subscriber is logged
    and the app still serves traffic. The TTL safety net keeps the cache
    correct even if pub/sub never delivers.
    """
    task: asyncio.Task | None = None
    try:
        from .services import chat_rag_bridge
        from .services.rbac_cache import subscribe_invalidations

        try:
            redis_handle = chat_rag_bridge._redis_client()
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "rbac subscriber: failed to obtain redis handle (%s) — "
                "skipping. TTL safety net (%ss) still applies.",
                exc,
                __import__("os").environ.get("RAG_RBAC_CACHE_TTL_SECS", "30"),
            )
        else:
            task = asyncio.create_task(
                subscribe_invalidations(redis_handle),
                name="rbac-cache-subscriber",
            )
            _logger.info("rbac subscriber: started (channel=rbac:invalidate)")
    except Exception as exc:  # noqa: BLE001
        _logger.warning("rbac subscriber: startup failed (%s) — continuing", exc)

    try:
        yield
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


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
    configure_json_logging()
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

    app = FastAPI(title="orgchat-kb", version="0.4.0", lifespan=_rbac_subscriber_lifespan)

    # Observability bootstrap — fail-open no-op when OBS_ENABLED != true.
    init_observability(app)

    # Phase 1.1 — tokenizer preflight. Crashes if RAG_BUDGET_TOKENIZER is
    # set explicitly to a non-cl100k HF alias and the tokenizer can't
    # load. Silent fallback drifts budget accounting by ~10-15%, which
    # evicts relevant chunks. Must run before router registration so a
    # misconfigured deploy fails loudly at startup, not on first chat.
    preflight_tokenizer()

    # Bug-fix campaign §6.7 — chat-model preflight. Validates that
    # CHAT_MODEL is in the chat endpoint's /v1/models response, bumping
    # ``chat_model_mismatch_total`` + WARNING on miss. Does NOT crash:
    # operators may use transparent aliases on the endpoint, and
    # vllm-chat can take ~60s to come up so a hard fail would deadlock
    # the open-webui boot. Surfaces real misconfigs (e.g. .env rolled
    # back without redeploying vllm-chat) on the metrics dashboard.
    try:
        preflight_chat_model()
    except Exception as exc:  # noqa: BLE001 — pure best-effort
        _logger.warning(
            "chat-model preflight raised unexpectedly (%s: %s) — "
            "continuing startup. This is a soft check; chat will still "
            "work if the endpoint is up.",
            type(exc).__name__, exc,
        )

    # Phase 1.2 — reranker preload. Loading on first request blocks that
    # request for ~3-5s on GPU cold start. Preloading at app init shifts
    # the cost to startup time and surfaces load failures before user
    # traffic hits. Non-fatal: reranker fails open to heuristic.
    import os as _os
    if _os.environ.get("RAG_RERANK", "0") == "1":
        try:
            from ext.services.cross_encoder_reranker import get_model
            get_model()
        except Exception as exc:
            _logger.error(
                "reranker preload failed (%s: %s) — feature will fail open. "
                "Check GPU 1 VRAM and RAG_RERANK_MODEL cache.",
                type(exc).__name__, exc,
            )

    # B3 — shadow log file handler. Operators can't analyze a stream
    # that only goes to stderr, so when shadow mode is on we pin a
    # rotating JSONL file at ``RAG_QU_SHADOW_LOG_PATH`` (default
    # ``/var/log/orgchat/qu_shadow.jsonl``). Best-effort: install
    # failures log a warning but don't crash startup.
    try:
        from ext.services.query_intent import maybe_install_shadow_log_file_handler
        maybe_install_shadow_log_file_handler()
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "shadow log file handler install failed (%s: %s) — "
            "shadow JSONL will only go to stderr.",
            type(exc).__name__, exc,
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    from fastapi import Depends as _Depends
    from .services.auth import require_admin as _require_admin

    @app.get("/api/kb/admin-ui", response_class=HTMLResponse)
    async def kb_admin_ui(_user=_Depends(_require_admin)):
        html = Path(__file__).parent / "static" / "kb-admin.html"
        return HTMLResponse(html.read_text())

    # Retrieval router MUST be registered before admin router to avoid /available shadowing.
    app.include_router(kb_retrieval.router)
    app.include_router(kb_admin.router)
    app.include_router(upload.router)
    app.include_router(rag.router)
    # P3.0 — SSE progress stream for the RAG pipeline.
    app.include_router(rag_stream.router)
    # SSE progress stream for KB document ingest (per-KB).
    app.include_router(ingest_stream.router)

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
    from .routers import ingest_stream, kb_admin, kb_retrieval, rag, rag_stream, upload
    from .services import auth as auth_svc
    from .services.embedder import TEIEmbedder
    from .services.vector_store import VectorStore

    configure_json_logging()
    # Mounted into upstream — we don't own the top-level FastAPI app, so
    # init without an app (OTel SDK still exports; per-request enrichment
    # happens via upstream's middleware stack when it calls us).
    init_observability(None)

    # Phase 1.1 — tokenizer preflight. Same contract as build_app: if an
    # explicit non-cl100k HF alias can't load, raise here so the upstream
    # process exits at import time instead of silently drifting budgets.
    preflight_tokenizer()

    # Bug-fix campaign §6.7 — chat-model preflight (mirrors build_app).
    # Soft-fail with a metrics counter + WARNING on mismatch so operators
    # can spot a stale CHAT_MODEL env without breaking upstream startup.
    try:
        preflight_chat_model()
    except Exception as exc:  # noqa: BLE001 — pure best-effort
        _logger.warning(
            "chat-model preflight raised unexpectedly (%s: %s) — continuing.",
            type(exc).__name__, exc,
        )

    # Phase 1.2 — reranker preload (mirrors build_app). Surface model load
    # failures at upstream startup, not on first user query. Non-fatal:
    # reranker fails open to heuristic if the load doesn't recover.
    import os as _os
    if _os.environ.get("RAG_RERANK", "0") == "1":
        try:
            from ext.services.cross_encoder_reranker import get_model
            get_model()
        except Exception as exc:
            _logger.error(
                "reranker preload failed (%s: %s) — feature will fail open. "
                "Check GPU 1 VRAM and RAG_RERANK_MODEL cache.",
                type(exc).__name__, exc,
            )

    # B3 — shadow log file handler (mirrors build_app). Without this the
    # shadow JSONL only goes to stderr/docker logs, so the operator
    # analyzer (`scripts/analyze_shadow_log.py`) has nothing persistent
    # to grep. Best-effort: install failures log a warning, never crash.
    try:
        from ext.services.query_intent import maybe_install_shadow_log_file_handler
        maybe_install_shadow_log_file_handler()
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "shadow log file handler install failed (%s: %s) — "
            "shadow JSONL will only go to stderr.",
            type(exc).__name__, exc,
        )

    # Prom `/metrics` collides with upstream Svelte SPA catch-all — expose
    # on a dedicated in-container port (9464) for Prometheus scrape.
    try:
        import os as _os
        from prometheus_client import start_http_server as _start
        _port = int(_os.environ.get("PROM_METRICS_PORT", "9464"))
        _start(_port)
        _logger.info("prometheus_client metrics server listening on :%d", _port)
    except Exception as _e:  # pragma: no cover - defensive
        _logger.warning("prometheus metrics port disabled: %s", _e)

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

    # Phase 1.5 follow-up — start the RBAC pub/sub subscriber on the
    # upstream-mounted path. We don't own upstream's lifespan, so we
    # attach a startup event handler to one of our routers; FastAPI
    # propagates router-level event handlers to the parent app at
    # ``include_router`` time. Single replica today, but the day a second
    # worker comes up, RBAC revocations would otherwise leak for up to
    # ``RAG_RBAC_CACHE_TTL_SECS`` (default 30s). Fail-open: any failure
    # logs and the TTL safety net still applies.
    from .services.rbac_cache import subscribe_invalidations as _sub_inv

    async def _start_rbac_subscriber() -> None:
        try:
            redis_handle = chat_rag_bridge._redis_client()
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "rbac subscriber (upstream): redis handle failed (%s) — "
                "TTL safety net still applies.",
                exc,
            )
            return
        # Detach as a background task so the startup hook returns
        # immediately. The event loop owns the task lifetime.
        import asyncio as _asyncio
        _asyncio.create_task(_sub_inv(redis_handle), name="rbac-cache-subscriber")
        _logger.info(
            "rbac subscriber (upstream): started (channel=rbac:invalidate)"
        )

    rag.router.add_event_handler("startup", _start_rbac_subscriber)

    # Vision preprocessor — converts attached images into text context
    # before RAG runs. Runs against vllm-vision on the internal network.
    from .services import vision as vision_svc
    vision_svc.configure(vector_store=vs, embedder=emb, sessionmaker=SessionLocal)

    # Admin UI page (standalone HTML — no Svelte needed)
    from fastapi import APIRouter, Depends as _Depends
    from fastapi.responses import HTMLResponse as HR
    from .services.auth import require_admin as _require_admin
    ui_router = APIRouter()

    @ui_router.get("/api/kb/admin-ui", response_class=HR)
    async def _kb_admin_ui(_user=_Depends(_require_admin)):
        html = Path(__file__).parent / "static" / "kb-admin.html"
        return HR(html.read_text())

    # Retrieval first — avoids /available shadowing by admin's /{kb_id}.
    return [
        ui_router,
        kb_retrieval.router,
        kb_admin.router,
        upload.router,
        rag.router,
        rag_stream.router,  # P3.0 — SSE progress stream
        ingest_stream.router,  # SSE progress for KB ingest
    ]
