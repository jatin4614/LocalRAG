"""Bridge between upstream's chat middleware and our KB RAG pipeline.

Called from the patched process_chat_payload() in middleware.py.
Retrieves KB context and returns it in upstream's source dict format.
"""
from __future__ import annotations

import contextvars
import logging
import os as _os
import time
from typing import Any, Awaitable, Callable, List, Optional

from . import flags
from .kb_config import config_to_env_overrides, merge_configs

logger = logging.getLogger("orgchat.rag_bridge")

# P0.5 — history-aware query rewrite (flag-gated OFF by default).
# When RAG_DISABLE_REWRITE=1 (default), behavior is byte-identical to pre-P0.5.
# Read at call time via flags.get so per-request overrides still compose (not
# that any KB-config currently sets it, but consistency with the rest of the
# hot path prevents future foot-guns).

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")

# Module-level registry — set by ext.app.build_ext_routers() at startup
_vector_store = None
_embedder = None
_sessionmaker = None


def configure(*, vector_store, embedder, sessionmaker) -> None:
    global _vector_store, _embedder, _sessionmaker
    _vector_store = vector_store
    _embedder = embedder
    _sessionmaker = sessionmaker


async def get_kb_config_for_chat(chat_id: str, user_id: str) -> Optional[list]:
    """Read the chat's selected KB config from upstream's chat.meta JSON column."""
    if _sessionmaker is None:
        return None
    from sqlalchemy import text
    import json
    try:
        async with _sessionmaker() as s:
            row = (await s.execute(
                text('SELECT meta FROM chat WHERE id = :cid AND user_id = :uid'),
                {"cid": chat_id, "uid": user_id},
            )).first()
            if row and row[0]:
                meta = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                return meta.get("kb_config")
    except Exception as e:
        logger.debug("kb_config lookup failed: %s", e)
    return None


async def _load_kb_rag_configs(
    kb_ids: list[int | str], session_factory: Optional[Callable] = None,
) -> list[dict]:
    """Fetch each selected KB's ``rag_config`` JSONB column.

    Returns a list of dicts — one per KB that actually resolved. Missing
    KBs (deleted, or RBAC already filtered them upstream) are skipped.
    On DB error we return an empty list so the caller falls through to
    process-level defaults (fail-open — retrieval still runs).
    """
    if not kb_ids:
        return []
    factory = session_factory or _sessionmaker
    if factory is None:
        return []
    try:
        from sqlalchemy import select
        from ..db.models import KnowledgeBase
        # Normalize IDs to int; skip non-numeric entries.
        ints: list[int] = []
        for k in kb_ids:
            try:
                ints.append(int(k))
            except (TypeError, ValueError):
                continue
        if not ints:
            return []
        async with factory() as s:
            rows = (await s.execute(
                select(KnowledgeBase.id, KnowledgeBase.rag_config).where(
                    KnowledgeBase.id.in_(ints),
                    KnowledgeBase.deleted_at.is_(None),
                )
            )).all()
        out: list[dict] = []
        for _kb_id, cfg in rows:
            if cfg:
                # rag_config is JSONB in Postgres → already a dict on read.
                # SQLAlchemy JSON type returns dicts/lists directly too.
                if isinstance(cfg, dict):
                    out.append(cfg)
        return out
    except Exception as e:
        logger.debug("kb_rag_config load failed: %s", e)
        return []


async def _emit(cb: Optional[Callable[[dict], Awaitable[None]]], event: dict) -> None:
    """Call the progress callback, swallowing any errors so a broken SSE
    client never breaks retrieval. No-op when cb is None."""
    if cb is None:
        return
    try:
        await cb(event)
    except Exception:
        # Fail open — the caller disconnected or the consumer is broken.
        pass


async def retrieve_kb_sources(
    kb_config: list,
    query: str,
    user_id: str,
    chat_id: Optional[str] = None,  # NEW: enables private chat doc retrieval
    history: Optional[List[dict]] = None,  # P0.5: prior turns for query rewrite
    progress_cb: Optional[Callable[[dict], Awaitable[None]]] = None,  # P3.0: SSE progress
) -> List[dict]:
    """Retrieve from our KB pipeline and format as upstream-compatible source dicts.

    Args:
        kb_config: [{"kb_id": 1, "subtag_ids": [...]}, ...]
        query: the user's message text
        user_id: for RBAC verification
        chat_id: current chat ID — if set, private docs in collection chat_{id} are included
        history: prior conversation turns (role/content dicts). When set AND
            RAG_DISABLE_REWRITE=0, the query is rewritten into a standalone form
            before retrieval. Default None preserves pre-P0.5 behavior.
        progress_cb: P3.0 — optional async callback invoked with stage events
            (e.g. ``{"stage": "retrieve", "status": "done", "ms": 9, "hits": 30}``)
            so the SSE endpoint can stream "work in progress" to the UI.
            Default None means zero overhead on every existing code path.

    Returns:
        List of source dicts in upstream's format:
        [{"source": {"id": ..., "name": ...}, "document": [...], "metadata": [...]}, ...]
    """
    if _vector_store is None or _embedder is None or _sessionmaker is None:
        logger.warning("RAG bridge not configured — skipping KB retrieval")
        return []

    if not query:
        return []

    # P0.5: optionally rewrite the query using conversation history.
    # Default OFF (RAG_DISABLE_REWRITE=1) → code path below is byte-identical
    # to the previous release. Flip the flag to engage.
    if flags.get("RAG_DISABLE_REWRITE", "1") != "1" and history:
        from .query_rewriter import rewrite_query as _rewrite_query
        query = await _rewrite_query(
            latest_turn=query,
            history=history,
            chat_url=_os.environ.get("OPENAI_API_BASE_URL", "http://vllm-chat:8000/v1"),
            chat_model=_os.environ.get("REWRITE_MODEL", _os.environ.get("CHAT_MODEL", "orgchat-chat")),
            api_key=_os.environ.get("OPENAI_API_KEY"),
        )

    # If neither KBs nor chat has private docs, early exit
    if not kb_config and not chat_id:
        return []

    import uuid as _uuid
    request_id_var.set(str(_uuid.uuid4())[:8])
    user_id_var.set(user_id)
    logger.info("rag: request started req=%s user=%s kbs=%d chat=%s",
                request_id_var.get(), user_id_var.get(),
                len(kb_config or []), bool(chat_id))

    # RBAC check — only if kb_config is non-empty
    selected_kbs = []
    if kb_config:
        from .rbac import get_allowed_kb_ids
        async with _sessionmaker() as s:
            allowed = set(await get_allowed_kb_ids(s, user_id=user_id))
        selected_kbs = [cfg for cfg in kb_config if cfg.get("kb_id") in allowed]
        if not selected_kbs and not chat_id:
            return []

    # P3.0: per-KB rag_config resolution.
    # Each selected KB carries a ``rag_config`` JSONB column stamped by an
    # admin. Merge them (UNION/MAX policy — strictest wins) into an
    # override dict of RAG_* env-var values, then wrap the whole pipeline
    # in ``flags.with_overrides`` so every stage reads the effective
    # setting without mutating the shared process env.
    kb_rag_configs = await _load_kb_rag_configs(
        [cfg.get("kb_id") for cfg in selected_kbs],
    )
    merged_cfg = merge_configs(kb_rag_configs)
    overrides = config_to_env_overrides(merged_cfg)

    with flags.with_overrides(overrides):
        return await _run_pipeline(
            query=query,
            selected_kbs=selected_kbs,
            user_id=user_id,
            chat_id=chat_id,
            progress_cb=progress_cb,
        )


async def _run_pipeline(
    *,
    query: str,
    selected_kbs: list,
    user_id: str,
    chat_id: Optional[str],
    progress_cb: Optional[Callable[[dict], Awaitable[None]]],
) -> List[dict]:
    """Inner pipeline — runs under a ``with_overrides`` scope.

    Separated from ``retrieve_kb_sources`` so the overlay is active for
    every lazy import and every flag read inside the hot path.
    """
    # Retrieve using our pipeline
    from .retriever import retrieve
    from .reranker import rerank, rerank_with_flag
    from .budget import budget_chunks
    # P2.5 — Prometheus metrics. Metric calls are wrapped in try/except
    # inside the helpers themselves; importing never raises because the
    # module falls back to no-op metrics if prometheus_client is absent.
    from .metrics import flag_state, retrieval_hits_total, time_stage

    # Snapshot current flag state for /metrics (best-effort — any failure
    # must not break retrieval). Read via flags.get so per-request
    # overrides are reflected in the gauge (they're point-in-time anyway).
    try:
        _hybrid_flag = flags.get("RAG_HYBRID", "1") != "0"
        flag_state.labels(flag="hybrid").set(1 if _hybrid_flag else 0)
        flag_state.labels(flag="rerank").set(
            1 if flags.get("RAG_RERANK", "0") == "1" else 0
        )
        flag_state.labels(flag="mmr").set(
            1 if flags.get("RAG_MMR", "0") == "1" else 0
        )
        flag_state.labels(flag="context_expand").set(
            1 if flags.get("RAG_CONTEXT_EXPAND", "0") == "1" else 0
        )
        flag_state.labels(flag="spotlight").set(
            1 if flags.get("RAG_SPOTLIGHT", "0") == "1" else 0
        )
    except Exception:
        _hybrid_flag = True  # best-effort for hit-counter label below

    _pipeline_start = time.perf_counter()

    try:
        with time_stage("total"):
            # --- embed is already implicit inside retrieve() via the embedder;
            # emit a coarse "embed" stage event before the Qdrant fan-out so
            # the UI can show a spinner. The actual embedding happens in the
            # first few ms of retrieve() — we can't split it out without
            # changing retriever's signature, so we bookend retrieve with
            # both embed+retrieve events for now.
            await _emit(progress_cb, {"stage": "embed", "status": "running"})
            _t0 = time.perf_counter()

            with time_stage("retrieve"):
                await _emit(progress_cb, {"stage": "retrieve", "status": "running"})
                _tR = time.perf_counter()
                raw_hits = await retrieve(
                    query=query,
                    selected_kbs=selected_kbs,
                    chat_id=chat_id,  # NOW PASSED THROUGH
                    vector_store=_vector_store,
                    embedder=_embedder,
                    per_kb_limit=10,
                    total_limit=30,
                    # P2.2: enforce per-user isolation on chat-scoped hits.
                    # KB hits are unaffected (retriever only applies the owner
                    # filter to chat namespaces).
                    owner_user_id=user_id,
                )
                _retrieve_ms = int((time.perf_counter() - _tR) * 1000)
                # The embed+retrieve bookend: once Qdrant returns we can
                # emit the "embed done" and "retrieve done" events
                # together since the bridge can't time-slice retrieve's
                # internal embed call without deeper refactoring.
                await _emit(progress_cb, {
                    "stage": "embed", "status": "done",
                    "ms": max(0, _retrieve_ms - _retrieve_ms // 2),
                })
                await _emit(progress_cb, {
                    "stage": "retrieve", "status": "done",
                    "ms": _retrieve_ms, "hits": len(raw_hits),
                })

            # Per-hit KB counter. Fails open — any bad payload is silently
            # tagged as kb="unknown" so the counter is never "missing".
            try:
                _path = "hybrid" if _hybrid_flag else "dense"
                for _h in raw_hits:
                    _payload = getattr(_h, "payload", None) or {}
                    _kb = _payload.get("kb_id")
                    if _kb is None:
                        _kb = "chat" if _payload.get("chat_id") is not None else "unknown"
                    retrieval_hits_total.labels(kb=str(_kb), path=_path).inc()
            except Exception:
                pass

            # P1.2 — dispatch through rerank_with_flag. Default (RAG_RERANK unset/0)
            # calls ``rerank`` which is byte-identical to the previous behaviour.
            #
            # P2 — MMR candidate-pool widening. When MMR is on, ask the reranker
            # for more candidates than the final budget so MMR actually has room
            # to diversify. When MMR is off, the old top-10 behaviour is preserved
            # byte-identically. An operator may override via ``RAG_RERANK_TOP_K``.
            _final_k = 10
            _rerank_top_k_env = flags.get("RAG_RERANK_TOP_K")
            _mmr_on = flags.get("RAG_MMR", "0") == "1"
            if _rerank_top_k_env is not None:
                _rerank_k = max(int(_rerank_top_k_env), _final_k)
            elif _mmr_on:
                _rerank_k = max(_final_k * 2, 20)
            else:
                _rerank_k = _final_k

            _rerank_on = flags.get("RAG_RERANK", "0") == "1"
            if _rerank_on:
                await _emit(progress_cb, {"stage": "rerank", "status": "running"})
            else:
                await _emit(progress_cb, {
                    "stage": "rerank", "status": "skipped",
                    "reason": "flag_off",
                })
            _tK = time.perf_counter()
            with time_stage("rerank"):
                reranked = rerank_with_flag(query, raw_hits, top_k=_rerank_k, fallback_fn=rerank)
            if _rerank_on:
                await _emit(progress_cb, {
                    "stage": "rerank", "status": "done",
                    "ms": int((time.perf_counter() - _tK) * 1000),
                    "top_k": len(reranked),
                })

            # P1.3 — MMR diversification (flag-gated). Reads the flag at call time
            # so tests can monkeypatch the env without module reload. When the flag
            # is off (default) the ``mmr`` module is never imported. Fails open:
            # any MMR error is swallowed and we keep the reranker output.
            if _mmr_on and reranked:
                await _emit(progress_cb, {"stage": "mmr", "status": "running"})
                try:
                    from .mmr import mmr_rerank_from_hits
                    _mmr_lambda = float(flags.get("RAG_MMR_LAMBDA", "0.7"))
                    _tM = time.perf_counter()
                    with time_stage("mmr"):
                        reranked = await mmr_rerank_from_hits(
                            query, reranked, _embedder,
                            top_k=_final_k, lambda_=_mmr_lambda,
                        )
                    await _emit(progress_cb, {
                        "stage": "mmr", "status": "done",
                        "ms": int((time.perf_counter() - _tM) * 1000),
                        "top_k": len(reranked),
                    })
                except Exception:
                    # Fail open: on any MMR error, stick with reranker output.
                    await _emit(progress_cb, {
                        "stage": "mmr", "status": "error",
                    })
            elif len(reranked) > _final_k:
                # No MMR — trim rerank's extra candidates (e.g. RAG_RERANK_TOP_K
                # set by operator) back to _final_k so downstream budget sees the
                # same count as the pre-P2 pipeline.
                reranked = reranked[:_final_k]
                await _emit(progress_cb, {
                    "stage": "mmr", "status": "skipped", "reason": "flag_off",
                })
            else:
                await _emit(progress_cb, {
                    "stage": "mmr", "status": "skipped", "reason": "flag_off",
                })

            # P1.4: context expansion (flag-gated). Reads the flag at call time
            # so tests can monkeypatch without module reload. When off (default)
            # the ``context_expand`` module is never imported. Fails open: any
            # exception keeps the reranker/MMR output untouched.
            _expand_on = flags.get("RAG_CONTEXT_EXPAND", "0") == "1"
            if _expand_on and reranked:
                await _emit(progress_cb, {"stage": "expand", "status": "running"})
                try:
                    from .context_expand import expand_context
                    _window = int(flags.get("RAG_CONTEXT_EXPAND_WINDOW", "1"))
                    _tE = time.perf_counter()
                    _before = len(reranked)
                    with time_stage("expand"):
                        reranked = await expand_context(
                            reranked, vs=_vector_store, window=_window,
                        )
                    await _emit(progress_cb, {
                        "stage": "expand", "status": "done",
                        "ms": int((time.perf_counter() - _tE) * 1000),
                        "siblings_fetched": max(0, len(reranked) - _before),
                    })
                except Exception:
                    # Fail open: on any error, keep reranker/MMR output unchanged.
                    await _emit(progress_cb, {
                        "stage": "expand", "status": "error",
                    })
            else:
                await _emit(progress_cb, {
                    "stage": "expand", "status": "skipped", "reason": "flag_off",
                })

            await _emit(progress_cb, {"stage": "budget", "status": "running"})
            _tB = time.perf_counter()
            with time_stage("budget"):
                # main-WIP bumped from 4000→5000 before upgrade; preserved here.
                budgeted = budget_chunks(reranked, max_tokens=5000)
            await _emit(progress_cb, {
                "stage": "budget", "status": "done",
                "ms": int((time.perf_counter() - _tB) * 1000),
                "chunks": len(budgeted),
            })
    except Exception as e:
        logger.exception("KB retrieval failed: %s", e)
        await _emit(progress_cb, {"stage": "error", "message": str(e)})
        return []

    if not budgeted:
        await _emit(progress_cb, {
            "stage": "done", "total_ms": int((time.perf_counter() - _pipeline_start) * 1000),
            "sources": 0,
        })
        return []

    # Collect doc_ids whose payload lacks a filename so we can back-fill from the DB.
    missing_doc_ids: set[int] = set()
    for hit in budgeted:
        if not hit.payload.get("filename") and hit.payload.get("doc_id"):
            try:
                missing_doc_ids.add(int(hit.payload["doc_id"]))
            except (TypeError, ValueError):
                pass

    doc_filename_map: dict[int, str] = {}
    if missing_doc_ids:
        from sqlalchemy import select
        from ..db.models import KBDocument
        async with _sessionmaker() as s:
            rows = (await s.execute(
                select(KBDocument.id, KBDocument.filename).where(KBDocument.id.in_(missing_doc_ids))
            )).all()
            doc_filename_map = {r[0]: r[1] for r in rows}

    # Group by source document for cleaner citation display
    sources_by_doc: dict[str, dict] = {}
    # P0.6 — spotlighting flag. Read once up-front for the loop.
    _spotlight_on = flags.get("RAG_SPOTLIGHT", "0") == "1"
    for hit in budgeted:
        # For private chat docs, payload may have chat_id instead of doc_id
        doc_id = str(hit.payload.get("doc_id", ""))
        chat_scope = hit.payload.get("chat_id")
        filename = str(hit.payload.get("filename", "") or "")
        if not filename and doc_id:
            try:
                filename = doc_filename_map.get(int(doc_id), "") or ""
            except (TypeError, ValueError):
                filename = ""
        if not filename:
            if chat_scope is not None:
                filename = f"private-doc (chat {chat_scope})"
            elif doc_id:
                filename = f"document-{doc_id}"
            else:
                filename = f"kb-{hit.payload.get('kb_id', '?')}"
        text_content = str(hit.payload.get("text", ""))

        # P0.6 — spotlighting against indirect prompt injection.
        # Read flag at call time so tests can toggle via monkeypatch.setenv
        # without module reload. Default (flag=0) → byte-identical to pre-P0.6.
        if _spotlight_on and text_content:
            from .spotlight import wrap_context as _spotlight_wrap
            text_content = _spotlight_wrap(text_content)

        # Key: prefer doc_id for KB docs, fall back to (chat_id, chunk_index) for private
        if doc_id:
            key = f"kb_{hit.payload.get('kb_id')}_{doc_id}"
        else:
            key = f"chat_{chat_scope}_{hit.payload.get('chunk_index', id(hit))}"

        if key not in sources_by_doc:
            sources_by_doc[key] = {
                "source": {
                    "id": key,
                    "name": filename,
                    "url": filename,
                },
                "document": [],
                # `metadata[].name` is checked first by the frontend badge renderer —
                # setting it guarantees the filename appears even if the metadata
                # `source` field is used for citation-id grouping upstream.
                "metadata": [],
            }
        sources_by_doc[key]["document"].append(text_content)
        sources_by_doc[key]["metadata"].append({
            "source": filename,
            "name": filename,
            "kb_id": hit.payload.get("kb_id"),
            "subtag_id": hit.payload.get("subtag_id"),
            "doc_id": doc_id or None,
            "chat_id": chat_scope,
        })

    sources_out = list(sources_by_doc.values())

    # --- KB catalog preamble ---------------------------------------------
    # Metadata queries ("which reports do I have?", "list documents",
    # "what dates are covered?") can't be answered from content alone —
    # retrieval returns chunks whose *text* mentions dates, not the set
    # of files themselves. We always prepend a compact catalog of the
    # selected KBs so the LLM knows the real scope of available docs.
    # Sourced from Postgres kb_documents (not Qdrant scroll) because it's
    # O(1) indexed lookup and reflects soft-deletes.
    try:
        kb_ids = sorted({int(c["kb_id"]) for c in selected_kbs if c.get("kb_id") is not None})
        if kb_ids and _sessionmaker is not None:
            from sqlalchemy import text as _sql_text
            async with _sessionmaker() as _s:
                res = await _s.execute(
                    _sql_text(
                        "SELECT kb_id, filename FROM kb_documents "
                        "WHERE kb_id = ANY(:ids) AND deleted_at IS NULL "
                        "ORDER BY kb_id, uploaded_at DESC, filename"
                    ),
                    {"ids": kb_ids},
                )
                rows = res.all()
            if rows:
                from collections import defaultdict as _dd
                _by_kb: dict[int, list[str]] = _dd(list)
                for kb_id_row, fn in rows:
                    _by_kb[kb_id_row].append(fn)
                _catalog_lines = []
                for kb_id_row, fns in _by_kb.items():
                    _catalog_lines.append(f"KB {kb_id_row}: {len(fns)} document(s) available")
                    # Cap at 60 filenames per KB to stay cheap on tokens
                    for fn in fns[:60]:
                        _catalog_lines.append(f"  - {fn}")
                    if len(fns) > 60:
                        _catalog_lines.append(f"  ... and {len(fns) - 60} more")
                _catalog_text = (
                    "KNOWLEDGE-BASE CATALOG (authoritative list of documents "
                    "available to you; use this for any 'which files / what "
                    "reports / list dates' question):\n" + "\n".join(_catalog_lines)
                )
                sources_out.insert(0, {
                    "source": {"id": "kb-catalog", "name": "kb-catalog", "url": "kb-catalog"},
                    "document": [_catalog_text],
                    "metadata": [{
                        "source": "kb-catalog",
                        "name": "kb-catalog",
                        "kb_id": None,
                        "doc_id": None,
                        "chat_id": None,
                        "subtag_id": None,
                    }],
                })
    except Exception as _e:
        # Fail-open: catalog is nice-to-have, not a correctness requirement.
        logger.debug("kb catalog preamble skipped: %s", _e)

    # Emit a "hits" event before "done" so the UI can render citation
    # previews as soon as the backend has them, without waiting for the
    # LLM response stream.
    try:
        _hit_summary: list[dict[str, Any]] = []
        for _src in sources_out[:20]:
            meta0 = (_src.get("metadata") or [{}])[0]
            _hit_summary.append({
                "doc_id": meta0.get("doc_id"),
                "filename": _src.get("source", {}).get("name"),
                "kb_id": meta0.get("kb_id"),
            })
        await _emit(progress_cb, {"stage": "hits", "hits": _hit_summary})
    except Exception:
        pass

    await _emit(progress_cb, {
        "stage": "done",
        "total_ms": int((time.perf_counter() - _pipeline_start) * 1000),
        "sources": len(sources_out),
    })
    return sources_out
