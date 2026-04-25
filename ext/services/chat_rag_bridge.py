"""Bridge between upstream's chat middleware and our KB RAG pipeline.

Called from the patched process_chat_payload() in middleware.py.
Retrieves KB context and returns it in upstream's source dict format.
"""
from __future__ import annotations

import contextvars
import json as _json
import logging
import os as _os
import time
from typing import Any, Awaitable, Callable, List, Mapping, Optional

from . import flags
from .kb_config import config_to_env_overrides, merge_configs
from .obs import span

logger = logging.getLogger("orgchat.rag_bridge")

# -----------------------------------------------------------------------
# Phase 4 — intent classification for structured query logging.
#
# PURE string matching, no embedding / LLM call / DB lookup. Runs on every
# query in ``_run_pipeline`` entry so it must be sub-millisecond. Output
# is a single label that's written into the ``rag_query`` log event; it
# does NOT yet influence retrieval routing (deferred to Phase 2 in the
# RAG plan — log-only for now so we can observe the real distribution
# before committing to a router).
#
# Label definitions:
#   * metadata — enumeration / coverage questions ("list", "what files",
#     "how many", etc.). These are answered by the KB-catalog preamble,
#     not by content retrieval.
#   * global   — cross-document aggregation ("compare", "across all",
#     "trends", "summarize"). Best served by a summary index when one
#     exists; today they hit the generic pipeline.
#   * specific — everything else. Single-doc / content-anchored queries
#     that the current top-k pipeline handles well.
# -----------------------------------------------------------------------
_METADATA_PATTERNS = (
    "list",
    "what files",
    "which files",
    "how many",
    "how much",
    "what reports",
    "which reports",
    "catalog",
    "inventory",
    "what documents",
    "which documents",
)
_GLOBAL_PATTERNS = (
    "compare",
    "across all",
    "trends",
    "trend",
    "summarize",
    "overview",
    "recurring",
    "overall",
    "aggregate",
)


def classify_intent(query: str) -> str:
    """Return one of ``metadata`` | ``global`` | ``specific``.

    Lowercase substring match against small, curated keyword lists. Order
    matters: metadata wins over global (e.g. "list all docs comparing
    March" → metadata, because answering with a file list is strictly
    more correct than answering with an aggregated comparison).

    Pure function — no side effects, no I/O. Must be safe to call on an
    empty or ``None``-looking string.
    """
    if not query:
        return "specific"
    q = query.lower()
    for pat in _METADATA_PATTERNS:
        if pat in q:
            return "metadata"
    for pat in _GLOBAL_PATTERNS:
        if pat in q:
            return "global"
    return "specific"


# -----------------------------------------------------------------------
# Phase 2.2 — intent-conditional MMR / context-expand defaults.
#
# Different intent classes need different pipeline shapes:
#   * specific / specific_date — one chunk is the answer; want adjacent
#     chunks for context (RAG_CONTEXT_EXPAND=1), MMR would dilute the
#     single best hit (RAG_MMR=0).
#   * global — broad aggregation across docs; diversity matters
#     (RAG_MMR=1), context expansion just inflates already-broad
#     summaries (RAG_CONTEXT_EXPAND=0).
#   * metadata — answered by the catalog preamble alone; both off.
#
# Reconciled with the 4-class ``query_intent.classify_with_reason``
# (``specific_date`` is mapped like ``specific``). The simple 3-class
# ``classify_intent`` above never emits ``specific_date`` — the entry
# is defensive so swapping classifiers later (Plan B Task 4) is a
# zero-touch change.
# -----------------------------------------------------------------------
_INTENT_FLAG_POLICY: dict[str, dict[str, str]] = {
    "specific":      {"RAG_MMR": "0", "RAG_CONTEXT_EXPAND": "1"},
    "specific_date": {"RAG_MMR": "0", "RAG_CONTEXT_EXPAND": "1"},
    "global":        {"RAG_MMR": "1", "RAG_CONTEXT_EXPAND": "0"},
    "metadata":      {"RAG_MMR": "0", "RAG_CONTEXT_EXPAND": "0"},
}


def resolve_intent_flags(
    *,
    intent: str,
    per_kb_overrides: Mapping[str, str],
) -> dict[str, str]:
    """Return the merged ``{RAG_*: "0"|"1"}`` overlay for ``intent``.

    Policy:
      * Look up ``intent`` in ``_INTENT_FLAG_POLICY``. Unknown intents
        (e.g. typo, future label not yet plumbed) fall back to the
        ``specific`` defaults — the safest catch-all because (a) it's
        the largest bucket on real corpora and (b) "fetch a chunk + its
        neighbours" is the most generally-useful pipeline shape.
      * Per-KB explicit overrides ALWAYS win. An admin who stamped
        ``rag_config.mmr = false`` did so deliberately; we don't
        second-guess. Implementation: per-KB dict is layered on top of
        the intent defaults via ``{**intent, **per_kb}``.

    Pure function — no side effects, no I/O. Inputs are not mutated.
    Returns a fresh dict so the caller can safely merge / mutate.
    """
    intent_defaults = _INTENT_FLAG_POLICY.get(intent, _INTENT_FLAG_POLICY["specific"])
    # Fresh dict — never mutate the policy table or the caller's overrides.
    merged: dict[str, str] = dict(intent_defaults)
    if per_kb_overrides:
        for k, v in per_kb_overrides.items():
            merged[str(k)] = str(v)
    return merged


def _log_rag_query(
    *,
    req_id: str,
    intent: str,
    kbs: list,
    hits: int,
    total_ms: int,
) -> None:
    """Emit one ``event=rag_query`` JSON log line.

    Fail-open: any exception inside ``json.dumps`` / the logger call is
    swallowed because this is a telemetry side-effect — the retrieval
    pipeline must not fail because the log record could not be built.
    """
    try:
        payload = {
            "event": "rag_query",
            "req_id": req_id,
            "intent": intent,
            "kbs": list(kbs or []),
            "hits": int(hits),
            "total_ms": int(total_ms),
        }
        logger.info(_json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass

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

# Phase 1.5 — process-wide redis handle for the RBAC cache. Lazy-init via
# _redis_client() so unit tests that never touch RBAC don't pay the cost
# of opening a connection to a redis that isn't running.
_rbac_redis = None


def _redis_client():
    """Return the shared async redis handle used by the RBAC cache.

    Creates the handle on first call from ``RAG_RBAC_CACHE_REDIS_URL``
    (default ``redis://localhost:6379/3`` so it's isolated from the
    application redis on DB 0). Subsequent calls return the same
    handle.
    """
    global _rbac_redis
    if _rbac_redis is None:
        import redis.asyncio as _redis
        url = _os.environ.get(
            "RAG_RBAC_CACHE_REDIS_URL", "redis://localhost:6379/3"
        )
        _rbac_redis = _redis.from_url(url)
    return _rbac_redis


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


async def _lookup_doc_ids_by_date(
    kb_ids: list[int], date_tuple: tuple[int, str, int],
) -> list[int]:
    """Return ``doc_id``s in ``kb_ids`` whose filename matches the date tuple.

    Used by the Tier-2 ``specific_date`` router. Tolerates the common
    filename-format variants observed in this corpus:

      * zero-padded day vs bare day: ``"05 Jan 2026.docx"`` vs ``"5 Jan 2026.docx"``
      * 4-digit vs 2-digit year: ``"05 Jan 2026.docx"`` vs ``"05 Apr 26.docx"``
      * stray whitespace: ``"01  Feb 2026.docx"`` (double space)
      * case variation: ``"17 JAn 2026.docx"`` (matched case-insensitively)

    Returns an empty list when:
      * ``kb_ids`` is empty
      * ``date_tuple`` is None
      * the session factory is unconfigured
      * no filename matches (caller falls back to generic ``specific``)
      * DB error (caller falls back too; error is logged at DEBUG)
    """
    if not kb_ids or not date_tuple or _sessionmaker is None:
        return []
    day, month, year = date_tuple
    # day_pattern: match 1-digit days with or without a leading zero, 2-digit
    # days verbatim. ``\\s+`` instead of literal space tolerates the corpus's
    # occasional double-space filenames. ``\\b`` at the end prevents
    # "5 Jan 2026" from matching "5 Jan 20260" (defensive — the corpus has
    # no such filenames, but the guard costs nothing).
    day_pattern = f"0?{day}" if day < 10 else str(day)
    year_short = f"{year % 100:02d}"
    # PostgreSQL's Advanced Regex (ARE) uses ``\y`` for a word boundary,
    # NOT Perl's ``\b`` (which PostgreSQL interprets as a literal backspace
    # character). This is the distinction that took down the first
    # iteration — the regex matched in Python tests but returned zero rows
    # from PG. Always ``\y`` here.
    regex = rf"^{day_pattern}\s+{month}\s+({year}|{year_short})\y"
    try:
        from sqlalchemy import text as _sql_text
        async with _sessionmaker() as s:
            rows = (await s.execute(
                _sql_text(
                    "SELECT id FROM kb_documents "
                    "WHERE kb_id = ANY(:kbs) AND deleted_at IS NULL "
                    "AND filename ~* :rx"
                ),
                {"kbs": list(kb_ids), "rx": regex},
            )).all()
        return [int(r[0]) for r in rows]
    except Exception as e:
        logger.debug("_lookup_doc_ids_by_date failed: %s", e)
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

    # Active sessions gauge — inc on entry, dec on exit. Fail-open: any
    # metrics hiccup must not break retrieval. Wrapped below in try/finally.
    _session_inc = False
    try:
        from .metrics import active_sessions
        active_sessions.inc()
        _session_inc = True
    except Exception:
        pass
    try:
        return await _retrieve_kb_sources_inner(
            kb_config=kb_config,
            query=query,
            user_id=user_id,
            chat_id=chat_id,
            history=history,
            progress_cb=progress_cb,
        )
    finally:
        if _session_inc:
            try:
                active_sessions.dec()  # type: ignore[attr-defined]
            except Exception:
                pass


async def _retrieve_kb_sources_inner(
    *,
    kb_config: list,
    query: str,
    user_id: str,
    chat_id: Optional[str],
    history: Optional[List[dict]],
    progress_cb: Optional[Callable[[dict], Awaitable[None]]],
) -> List[dict]:
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

    # RBAC check — only if kb_config is non-empty.
    #
    # Phase 1.5 — cache-first lookup. The cache is keyed on
    # ``rbac:user:{user_id}`` (per-user namespace -> zero collision risk)
    # with TTL = ``RAG_RBAC_CACHE_TTL_SECS`` (default 30s). Cache miss
    # falls through to the DB. Any redis failure (connection refused,
    # timeout, corrupt value) returns ``None`` from the cache layer so
    # we still hit the DB -- the DB query is the source of truth and
    # the cache is purely an accelerator. A sacred CLAUDE.md §2 invariant
    # is that the DB miss MUST always run on cache absence so isolation
    # is never weakened by cache outage.
    selected_kbs = []
    if kb_config:
        from .rbac import get_allowed_kb_ids
        from .rbac_cache import get_shared_cache
        allowed: set[int] | None = None
        try:
            cache = get_shared_cache(redis=_redis_client())
            allowed = await cache.get(user_id=user_id)
        except Exception as _cache_exc:  # noqa: BLE001
            # Fail-open on cache errors: log and fall through to DB.
            logger.debug("rbac cache get failed: %s", _cache_exc)
            allowed = None
        if allowed is None:
            async with _sessionmaker() as s:
                allowed = set(await get_allowed_kb_ids(s, user_id=user_id))
            try:
                cache = get_shared_cache(redis=_redis_client())
                await cache.set(user_id=user_id, allowed_kb_ids=allowed)
            except Exception as _cache_exc:  # noqa: BLE001
                # Cache write failure is non-fatal: next request just
                # re-fetches from the DB.
                logger.debug("rbac cache set failed: %s", _cache_exc)
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

    # Phase 2.2 — intent-conditional defaults for MMR / context_expand.
    # Layer the intent's preferred shape UNDER the per-KB overrides so an
    # admin's explicit ``rag_config`` value always wins. The classifier is
    # the simple regex-based one (matching what's plumbed today); when the
    # 4-class classifier from ``query_intent`` is wired (Plan B Task 4) the
    # ``specific_date`` policy entry will start firing automatically with
    # no code change here.
    _intent_label_for_flags = classify_intent(query)
    _intent_flag_overrides = resolve_intent_flags(
        intent=_intent_label_for_flags, per_kb_overrides=overrides,
    )
    # ``resolve_intent_flags`` already applied the per-KB overrides on top
    # of the intent defaults, so its return value IS the effective overlay.
    merged_overrides = _intent_flag_overrides

    with flags.with_overrides(merged_overrides):
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
    # Phase 4 — intent classification for the structured log event at the
    # tail of this function. Done here (entry of _run_pipeline) so failure
    # modes (empty hits, LLM-side errors) still get an entry. Pure string
    # match — sub-millisecond, no external call.
    _intent_label = classify_intent(query)
    _kb_ids_for_log = [cfg.get("kb_id") for cfg in selected_kbs or []]

    # Phase 1.3 — Qdrant preflight. Fails fast (returning empty sources) if
    # Qdrant is fully down, before we fan out N parallel KB searches that
    # would all time out individually. ``health_check()`` is cached 5s so N
    # concurrent chat turns share one probe. ``getattr`` guard so unit tests
    # that wire an ``object()`` stub as ``_vector_store`` (no health_check
    # attribute) are unaffected — preflight is purely defensive against
    # network outages, not a contract on the VectorStore interface.
    _hc = getattr(_vector_store, "health_check", None)
    if callable(_hc):
        try:
            if not await _hc():
                logger.warning(
                    "rag: qdrant preflight failed; returning empty sources"
                )
                return []
        except Exception as _hc_exc:
            # Probe itself raised — log but DO NOT short-circuit. Falling
            # through lets downstream code surface the real Qdrant error
            # with full context (collection name, filter shape) rather
            # than the opaque preflight failure.
            logger.warning("rag: qdrant preflight raised %s; continuing", _hc_exc)

    # Tier 2 — intent routing (gated by RAG_INTENT_ROUTING, default OFF).
    # When OFF, _intent/_intent_reason are fixed values and every branch
    # below collapses to the pre-Tier-2 ``specific`` path, making the
    # pipeline byte-identical. When ON:
    #   * metadata  → skip chunk retrieve; catalog preamble answers.
    #   * global    → retrieve only level="doc" summary points; skip
    #                 rerank/MMR/expand (summaries are self-contained).
    #   * specific  → unchanged pipeline.
    _intent: str = "specific"
    _intent_reason: str = "default"
    if flags.get("RAG_INTENT_ROUTING", "0") == "1":
        from .query_intent import classify_with_reason as _ci_classify
        _intent, _intent_reason = _ci_classify(query)
        logger.info("rag: intent=%s reason=%s", _intent, _intent_reason)

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

            _kb_ids_csv = ",".join(
                str(c.get("kb_id")) for c in (selected_kbs or [])
                if c.get("kb_id") is not None
            )
            with time_stage("retrieve"), span(
                "retrieve.parallel",
                n_kbs=len(selected_kbs or []),
                kb_ids=_kb_ids_csv,
                chat_id=str(chat_id) if chat_id else "",
            ):
                _tR = time.perf_counter()
                # Tier 2 routing — metadata queries are answered entirely
                # by the catalog preamble appended later; skipping
                # retrieval saves a Qdrant round-trip and avoids feeding
                # irrelevant chunk hits to the LLM for "which files?"
                # type questions.
                if _intent == "metadata":
                    raw_hits = []
                    await _emit(progress_cb, {
                        "stage": "retrieve", "status": "skipped_by_intent",
                        "intent": "metadata", "reason": _intent_reason,
                    })
                else:
                    await _emit(progress_cb, {"stage": "retrieve", "status": "running"})

                    # Tier-2 ``specific_date``: narrow retrieval to documents
                    # whose filename matches the date the user mentioned.
                    # Ranking signals alone can't tell "5 Jan" from "5 Feb" /
                    # "4 Jan" on a daily-reporting corpus — they share the
                    # same numeric tokens in BM25 and the same template in
                    # dense space. A doc_id filter guarantees the right doc.
                    _date_doc_ids: Optional[list[int]] = None
                    if _intent == "specific_date":
                        from .query_intent import extract_date_tuple as _ci_date
                        _date_tuple = _ci_date(query)
                        if _date_tuple:
                            _kb_ids_for_date = [
                                int(c["kb_id"]) for c in selected_kbs
                                if c.get("kb_id") is not None
                            ]
                            _date_doc_ids = await _lookup_doc_ids_by_date(
                                _kb_ids_for_date, _date_tuple,
                            )
                            if not _date_doc_ids:
                                # Date parsed but no filename matches in scope
                                # → fall through to regular specific retrieval.
                                logger.info(
                                    "rag: specific_date → specific (no filename match for %s)",
                                    _date_tuple,
                                )
                                _intent = "specific"
                                _intent_reason = (
                                    f"specific:date_no_filename_match={_date_tuple}"
                                )
                            else:
                                logger.info(
                                    "rag: specific_date matched %d doc(s) for %s: %s",
                                    len(_date_doc_ids), _date_tuple, _date_doc_ids,
                                )

                    # For global intent we search the doc-summary index
                    # (level="doc") so every document contributes exactly
                    # one point — the ideal shape for "list every X"
                    # style questions. For specific intent (ONLY when
                    # routing is enabled) we filter doc-summaries OUT so
                    # factoid queries aren't contaminated by the short
                    # summary points that tend to out-score chunks on
                    # broad vocabulary matches. Pre-routing default
                    # (routing flag OFF) stays ``None`` → byte-identical.
                    _routing_on = flags.get("RAG_INTENT_ROUTING", "0") == "1"
                    if _intent == "global":
                        _level_filter = "doc"
                    elif _routing_on and _intent in ("specific", "specific_date"):
                        _level_filter = "chunk"
                    else:
                        _level_filter = None
                    # Summaries are self-contained and already widely
                    # covered; give global queries a wider top-k so all
                    # documents in the selection can surface. For
                    # ``specific_date`` the doc_id filter already narrows
                    # the corpus to typically <= 2 docs (<= 22 chunks per
                    # doc on this corpus) so a wider top-k lets the
                    # reranker see ALL of that doc's chunks rather than
                    # the first 10.
                    if _intent == "global":
                        _per_kb, _total = 50, 100
                    elif _intent == "specific_date":
                        _per_kb, _total = 30, 60
                    else:
                        _per_kb, _total = 10, 30
                    # For global intent we silence HyDE AND the hybrid
                    # sparse arm. HyDE's hypothetical answers look like
                    # chunk content; BM25 inherently ranks term-rich
                    # chunks above the terser summaries. Both bias the
                    # retrieval pool away from the doc-summary cluster,
                    # leaving the ``level="doc"`` post-filter with
                    # nothing to keep. Pure dense search over the
                    # subtag-scoped summary points is what we want —
                    # summaries are homogeneous in length and style, so
                    # dense similarity separates them cleanly.
                    _retrieve_overrides = (
                        {"RAG_HYDE": "0", "RAG_HYDE_N": "0", "RAG_HYBRID": "0"}
                        if _intent == "global"
                        else {}
                    )
                    with span("embed.query", path="query"), flags.with_overrides(_retrieve_overrides):
                        # ``retrieve`` internally calls embedder.embed() — the
                        # embed.call span will nest under this embed.query span
                        # giving operators a clear query-embed timing.
                        raw_hits = await retrieve(
                            query=query,
                            selected_kbs=selected_kbs,
                            chat_id=chat_id,  # NOW PASSED THROUGH
                            vector_store=_vector_store,
                            embedder=_embedder,
                            per_kb_limit=_per_kb,
                            total_limit=_total,
                            # P2.2: enforce per-user isolation on chat-scoped hits.
                            # KB hits are unaffected (retriever only applies the owner
                            # filter to chat namespaces).
                            owner_user_id=user_id,
                            level_filter=_level_filter,
                            doc_ids=_date_doc_ids,
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
            _mmr_on = flags.get("RAG_MMR", "0") == "1" and not (_intent == "global")
            if _rerank_top_k_env is not None:
                _rerank_k = max(int(_rerank_top_k_env), _final_k)
            elif _mmr_on:
                _rerank_k = max(_final_k * 2, 20)
            else:
                _rerank_k = _final_k

            # Tier 2: for global-intent queries we short-circuit rerank/
            # MMR/expand because doc-level summaries are already self-
            # contained and already one-per-document (no duplication to
            # diversify, no siblings to expand, no cross-encoder gain
            # over summary prose). This keeps latency proportional to
            # the broadened top-k.
            _short_circuit_quality = (_intent == "global")
            _rerank_on = flags.get("RAG_RERANK", "0") == "1" and not _short_circuit_quality
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
            _expand_on = flags.get("RAG_CONTEXT_EXPAND", "0") == "1" and not (_intent == "global")
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

    # Tier 2: for metadata AND global intents, the catalog preamble below
    # is the load-bearing signal — metadata skipped retrieve entirely, and
    # global may post-filter to zero doc-summary hits on thin summary
    # indexes. Skip the empty-short-circuit for both so the preamble still
    # flows to the LLM.
    if not budgeted and _intent not in ("metadata", "global"):
        _total_ms_empty = int((time.perf_counter() - _pipeline_start) * 1000)
        await _emit(progress_cb, {
            "stage": "done", "total_ms": _total_ms_empty,
            "sources": 0,
        })
        _log_rag_query(
            req_id=request_id_var.get(),
            intent=_intent_label,
            kbs=_kb_ids_for_log,
            hits=0,
            total_ms=_total_ms_empty,
        )
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
        # Split selection into two buckets so we respect subtag scoping.
        # A bare kb_id entry (no subtag_ids / empty list) means "whole KB";
        # a non-empty subtag_ids list scopes to just those subtags. Mixing
        # both per-request is allowed, so we build two separate queries and
        # UNION their results in Python.
        whole_kb_ids: list[int] = sorted({
            int(c["kb_id"]) for c in selected_kbs
            if c.get("kb_id") is not None and not c.get("subtag_ids")
        })
        # Deterministic ordering for test repeatability.
        subtag_entries: list[tuple[int, list[int]]] = sorted(
            [
                (int(c["kb_id"]),
                 sorted({int(s) for s in c.get("subtag_ids") or [] if s is not None}))
                for c in selected_kbs
                if c.get("kb_id") is not None and c.get("subtag_ids")
            ],
            key=lambda t: t[0],
        )
        # Drop entries whose subtag list collapsed to empty after filtering
        # None-values (defensive; shouldn't normally happen).
        subtag_entries = [(k, s) for (k, s) in subtag_entries if s]

        if (whole_kb_ids or subtag_entries) and _sessionmaker is not None:
            from sqlalchemy import text as _sql_text
            from collections import defaultdict as _dd

            # (kb_id, subtag_id_or_None, subtag_name_or_None) -> [filename,...]
            _buckets: dict[tuple[int, int | None, str | None], list[str]] = _dd(list)

            async with _sessionmaker() as _s:
                if whole_kb_ids:
                    res = await _s.execute(
                        _sql_text(
                            "SELECT kb_id, subtag_id, NULL::text AS subtag_name, filename "
                            "FROM kb_documents "
                            "WHERE kb_id = ANY(:ids) AND deleted_at IS NULL "
                            "ORDER BY uploaded_at DESC, filename"
                        ),
                        {"ids": whole_kb_ids},
                    )
                    for kb_id_row, _sid, _sname, fn in res.all():
                        # Whole-KB bucket: key on (kb_id, None, None) so all
                        # docs across subtags collapse under one header.
                        _buckets[(kb_id_row, None, None)].append(fn)

                if subtag_entries:
                    # Build one OR-chain expanding each (kb_id, subtag_ids) pair
                    # with its own parameter names to avoid collisions.
                    _where_parts: list[str] = []
                    _params: dict[str, Any] = {}
                    for _i, (_k, _sids) in enumerate(subtag_entries):
                        _where_parts.append(
                            f"(d.kb_id = :k{_i} AND d.subtag_id = ANY(:s{_i}))"
                        )
                        _params[f"k{_i}"] = _k
                        _params[f"s{_i}"] = _sids
                    _sql = (
                        "SELECT d.kb_id, d.subtag_id, t.name AS subtag_name, d.filename "
                        "FROM kb_documents d "
                        "JOIN kb_subtags t ON t.id = d.subtag_id "
                        "WHERE d.deleted_at IS NULL AND (" + " OR ".join(_where_parts) + ") "
                        "ORDER BY d.uploaded_at DESC, d.filename"
                    )
                    res = await _s.execute(_sql_text(_sql), _params)
                    for kb_id_row, sid, sname, fn in res.all():
                        # Group rows by (kb_id, frozenset-of-subtags-chosen)
                        # so the header can name all subtags picked for that
                        # kb in one line. We stash them under a synthetic key
                        # that carries the kb_id only; subtag names are
                        # aggregated below.
                        _buckets[(kb_id_row, sid, sname)].append(fn)

            if _buckets:
                # Cap high enough that a full year of daily reports fits
                # comfortably (365 + headroom). At ~20 chars/filename the
                # whole catalog is <1% of a 32k context, so trading tokens
                # for authoritative coverage is the right call. Operators
                # can lower via RAG_KB_CATALOG_MAX if they ever pack KBs
                # with 10k+ docs. Applied per bucket.
                try:
                    _cap = int(_os.environ.get("RAG_KB_CATALOG_MAX", "500"))
                except (TypeError, ValueError):
                    _cap = 500

                _catalog_lines: list[str] = []

                # Whole-KB buckets first, sorted by kb_id for determinism.
                _whole_keys = sorted(
                    [k for k in _buckets.keys() if k[1] is None],
                    key=lambda t: t[0],
                )
                for _key in _whole_keys:
                    _kb_id = _key[0]
                    _fns = _buckets[_key]
                    _catalog_lines.append(
                        f"KB {_kb_id}: {len(_fns)} document(s) available"
                    )
                    for fn in _fns[:_cap]:
                        _catalog_lines.append(f"  - {fn}")
                    if len(_fns) > _cap:
                        _catalog_lines.append(f"  ... and {len(_fns) - _cap} more")

                # Subtag-scoped buckets: merge rows sharing the same kb_id
                # under one header that lists all picked subtag names.
                _subtag_rows: dict[int, dict[int, tuple[str | None, list[str]]]] = _dd(dict)
                for (_kb_id, _sid, _sname), _fns in _buckets.items():
                    if _sid is None:
                        continue
                    # Merge by (kb_id, subtag_id); if the same subtag appears
                    # in multiple buckets (shouldn't happen given SQL) we
                    # concatenate.
                    if _sid in _subtag_rows[_kb_id]:
                        _prev_name, _prev_fns = _subtag_rows[_kb_id][_sid]
                        _subtag_rows[_kb_id][_sid] = (_prev_name or _sname, _prev_fns + _fns)
                    else:
                        _subtag_rows[_kb_id][_sid] = (_sname, list(_fns))

                for _kb_id in sorted(_subtag_rows.keys()):
                    _per_subtag = _subtag_rows[_kb_id]
                    # Sort subtags by id for deterministic header ordering.
                    _sids_sorted = sorted(_per_subtag.keys())
                    _names = [
                        (_per_subtag[_sid][0] or f"subtag {_sid}")
                        for _sid in _sids_sorted
                    ]
                    # Flatten all filenames in this kb's subtag selection,
                    # preserving the SQL order (uploaded_at DESC, filename).
                    _fns_all: list[str] = []
                    for _sid in _sids_sorted:
                        _fns_all.extend(_per_subtag[_sid][1])
                    _header = (
                        f"KB {_kb_id} \u2192 {', '.join(_names)}: "
                        f"{len(_fns_all)} document(s) available"
                    )
                    _catalog_lines.append(_header)
                    for fn in _fns_all[:_cap]:
                        _catalog_lines.append(f"  - {fn}")
                    if len(_fns_all) > _cap:
                        _catalog_lines.append(f"  ... and {len(_fns_all) - _cap} more")

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

    # --- Current-datetime preamble ---------------------------------------
    # Chat models have a training cutoff and cannot answer "what's today's
    # date" unless we inject the current wall-clock time. Stamp a small
    # pseudo-source so the LLM always has an authoritative anchor.
    # Toggle with ``RAG_INJECT_DATETIME=0`` (default on — cost is ~40 tokens
    # per request; answers "what day is it" correctly every time).
    if _os.environ.get("RAG_INJECT_DATETIME", "1") != "0":
        try:
            import datetime as _dt
            try:
                import zoneinfo as _zi
                _tz_name = _os.environ.get("RAG_TZ", "UTC")
                try:
                    _tz = _zi.ZoneInfo(_tz_name)
                except Exception:
                    _tz, _tz_name = _zi.ZoneInfo("UTC"), "UTC"
                _now = _dt.datetime.now(_tz)
            except Exception:
                # zoneinfo unavailable (shouldn't happen on py3.9+) →
                # fall back to naive local time.
                _now = _dt.datetime.now()
                _tz_name = "local"
            _dt_text = (
                "CURRENT DATE AND TIME (authoritative — answer "
                "'what is today' / 'what date is it' / 'what time is it' "
                "questions using this preamble, not your training data):\n"
                f"Date: {_now.strftime('%A, %B %d, %Y')}\n"
                f"Time: {_now.strftime('%I:%M %p')} {_tz_name}\n"
                f"ISO: {_now.strftime('%Y-%m-%dT%H:%M:%S')}"
            )
            sources_out.insert(0, {
                "source": {"id": "current-datetime", "name": "current-datetime",
                           "url": "current-datetime"},
                "document": [_dt_text],
                "metadata": [{
                    "source": "current-datetime", "name": "current-datetime",
                    "kb_id": None, "doc_id": None, "chat_id": None,
                    "subtag_id": None,
                }],
            })
        except Exception as _e:
            logger.debug("datetime preamble skipped: %s", _e)

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
        await _emit(progress_cb, {
            "stage": "hits", "hits": _hit_summary,
            "intent": _intent, "intent_reason": _intent_reason,
        })
    except Exception:
        pass

    _total_ms_done = int((time.perf_counter() - _pipeline_start) * 1000)
    await _emit(progress_cb, {
        "stage": "done",
        "total_ms": _total_ms_done,
        "sources": len(sources_out),
    })
    # Phase 4 — structured per-query log. Single JSON line so Loki /
    # friends can pivot on ``intent`` without having to parse a
    # human-formatted message. Best-effort: any logging error is
    # swallowed so it can never affect the return value.
    _log_rag_query(
        req_id=request_id_var.get(),
        intent=_intent_label,
        kbs=_kb_ids_for_log,
        hits=len(sources_out),
        total_ms=_total_ms_done,
    )
    return sources_out
