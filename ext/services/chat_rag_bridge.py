"""Bridge between upstream's chat middleware and our KB RAG pipeline.

Called from the patched process_chat_payload() in middleware.py.
Retrieves KB context and returns it in upstream's source dict format.
"""
from __future__ import annotations

import asyncio
import contextvars
import json as _json
import logging
import os as _os
import threading
import time
from typing import Any, Awaitable, Callable, List, Mapping, Optional

from . import flags
from .kb_config import config_to_env_overrides, merge_configs
from .obs import span

logger = logging.getLogger("orgchat.rag_bridge")

# B6 (audit fix) — central helper so every silent-fall-through site looks
# the same: emit a WARNING (operators care), increment a stage-labelled
# counter (Prometheus alerts), and never re-raise (preserves the existing
# pipeline behavior — see CLAUDE.md "fail-open everywhere" §3 invariant).
# Wrapped in its own try/except because metrics import or label cardinality
# must never be the reason retrieval breaks.
def _record_silent_failure(stage: str, err: BaseException) -> None:
    """Log + count one swallowed exception. Never raises."""
    try:
        logger.warning("rag: %s failed (%s): %r", stage, type(err).__name__, err)
    except Exception:
        pass
    try:
        from .metrics import RAG_SILENT_FAILURE

        RAG_SILENT_FAILURE.labels(stage=stage).inc()
    except Exception:
        # Counter import / label / inc must not propagate — that would
        # defeat the point of the helper.
        pass

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
    """DEPRECATED — synchronous regex fallback. Use :func:`_classify_with_qu`
    for new call sites.

    Returns one of ``metadata`` | ``global`` | ``specific``. Lowercase
    substring match against small, curated keyword lists. Order matters:
    metadata wins over global. Pure function — no side effects, no I/O.

    Plan B Phase 4.6 added the async :func:`_classify_with_qu` which
    consults the Qwen3-4B QU LLM on ambiguous queries. Existing sync
    callers (logging hooks, debug endpoints) keep using this helper for
    backward compatibility.
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


# Plan B Phase 5.6 — derive an "evolution" hint when the LLM-classified
# intent is "global" AND the original query carries a comparison verb.
# The 4-class hybrid classifier intentionally never emits "evolution"
# (keeps the 4-class invariant for caches + metrics); instead the
# retriever takes the derived hint and routes level injection (L2 + L3
# instead of L3 + L4).
_EVOLUTION_VERBS: tuple[str, ...] = (
    "compare", "evolve", "evolution", "evolved", "change", "changed",
    "trend", "trending", "differ", "different", "contrast", "shift",
    "shifted", "trajectory", "growth", "vs", "versus",
)


def derive_temporal_intent_hint(*, intent: str, query: str) -> str:
    """Map a 4-class intent to the temporal level-injection rules.

    Returns one of: ``"global"`` | ``"evolution"`` | ``"specific_date"`` |
    ``"specific"`` | ``"metadata"``.

    Plan B Phase 5.6. Pure function — safe to call on the hot path.
    """
    if intent == "global" and query:
        ql = query.lower()
        if any(v in ql for v in _EVOLUTION_VERBS):
            return "evolution"
    return intent


# -----------------------------------------------------------------------
# Plan B Phase 4.6 — async hybrid classifier wrapper with Redis cache.
# -----------------------------------------------------------------------
def _extract_last_turn_id(history: Optional[List[dict]]) -> str:
    """Return the most recent assistant turn's stable ID, or ``""`` if none.

    The chat middleware passes history with optional ``id`` keys per turn.
    When no ID is present we fall back to a short content hash so context
    shifts still invalidate the QU cache (different assistant content →
    different cache key → fresh LLM call).
    """
    if not history:
        return ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            tid = msg.get("id")
            if tid:
                return str(tid)
            content = msg.get("content") or ""
            return _hash_short(content)
    return ""


def _hash_short(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# Lazily-initialized singleton — first call opens the redis connection,
# subsequent calls return the same handle. Reset to None in tests via
# ``monkeypatch.setattr(bridge, "_qu_cache_singleton", None)``.
#
# §7.5: ``_QU_CACHE_LOCK`` guards the read+create against two concurrent
# first-callers each building (and leaking) their own client. Mirrors the
# double-checked-locking pattern in
# ``ext/services/cross_encoder_reranker.py:80-82``.
_qu_cache_singleton = None
_QU_CACHE_LOCK = threading.Lock()


def _get_qu_cache():
    """Return the process-wide :class:`QUCache` singleton, or ``None`` if
    the cache is disabled (``RAG_QU_CACHE_ENABLED=0``) or redis cannot be
    reached. Soft-fails so the bridge never raises on cache infra issues.

    Thread-safe singleton init under ``_QU_CACHE_LOCK`` (§7.5).
    """
    global _qu_cache_singleton
    if _qu_cache_singleton is not None:
        return _qu_cache_singleton
    if _os.environ.get("RAG_QU_CACHE_ENABLED", "1") != "1":
        return None
    with _QU_CACHE_LOCK:
        # Re-check inside the lock — another caller may have raced past the
        # first None-check and finished the init while we were blocked.
        if _qu_cache_singleton is not None:
            return _qu_cache_singleton
        try:
            import redis.asyncio as _redis

            from .qu_cache import QUCache

            url = _os.environ.get("REDIS_URL", "redis://redis:6379")
            # Strip any /<db> suffix so we can override with the dedicated DB
            if "/" in url.rsplit("@", 1)[-1].split("//", 1)[-1]:
                base = url.rsplit("/", 1)[0]
            else:
                base = url
            db = int(_os.environ.get("RAG_QU_REDIS_DB", "4"))
            client = _redis.from_url(f"{base}/{db}", decode_responses=True)
            _qu_cache_singleton = QUCache(redis_client=client)
            return _qu_cache_singleton
        except Exception as e:
            # B6: keep the existing operator-facing warning AND emit the silent-
            # failure counter so QU-cache outages can be alerted on.
            logger.warning("QU cache init failed: %s — running without cache", e)
            _record_silent_failure("qu_cache_init", e)
            return None


async def _classify_with_qu(
    query: str,
    history: Optional[List[dict]],
    last_turn_id: str = "",
):
    """Hybrid regex+LLM classifier with cache.

    Returns a :class:`ext.services.query_intent.HybridClassification`. The
    bridge can rely on this never raising — failures (cache errors, LLM
    timeouts) all soft-fall back to the regex result.

    Cache is consulted when ``RAG_QU_ENABLED=1`` and ``RAG_QU_CACHE_ENABLED=1``.
    Hits are stamped with ``cached=True`` for metric labelling. The cache
    is only WRITTEN for ``source="llm"`` results — regex hits stay out of
    the cache because they're already a regex.search() away.
    """
    from .query_intent import (
        EscalationReason,
        HybridClassification,
        classify_with_qu as _qi_classify,
    )

    cache = _get_qu_cache()
    if cache is not None:
        cached = await cache.get(query, last_turn_id)
        if cached is not None:
            return HybridClassification(
                intent=cached.intent,
                resolved_query=cached.resolved_query,
                temporal_constraint=cached.temporal_constraint,
                entities=cached.entities,
                confidence=cached.confidence,
                source="llm",
                escalation_reason=EscalationReason.NONE,
                cached=True,
            )

    result = await _qi_classify(query=query, history=history)

    if cache is not None and result.source == "llm":
        from .query_understanding import QueryUnderstanding

        qu = QueryUnderstanding(
            intent=result.intent,
            resolved_query=result.resolved_query,
            temporal_constraint=result.temporal_constraint,
            entities=result.entities,
            confidence=result.confidence,
            source="llm",
            cached=False,
        )
        await cache.set(query, last_turn_id, qu)

    return result


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

    Precedence (default mode = ``intent``):
      1. ``_INTENT_FLAG_POLICY[intent]``      — base
      2. per-KB ``rag_config`` overrides      — wins over base
      3. ``flags.with_overrides`` overlay     — applied by caller around
         the returned dict; the returned keys SHADOW process env at lookup.

    With ``RAG_INTENT_OVERLAY_MODE=env``, the precedence inverts for the
    flag-keys we'd otherwise stamp: if the operator set ``RAG_MMR=1`` in
    their env, we DROP ``RAG_MMR`` from the overlay so the env value
    shows through ``flags.get`` unshadowed. Per-KB ``rag_config`` still
    wins over both — it's an explicit per-collection statement.

    Why the toggle exists (B3 design call, 2026-04-25):
      * ``intent`` (default) — operator picks per-intent policy ONCE; the
        intent classifier picks the right shape per query without
        re-deploying. Production-safe defaults baked in. Per-KB
        ``rag_config`` is the right escape hatch for collection-level
        customisation.
      * ``env``               — operator escape hatch. If retrieval is
        misbehaving and we need to force MMR or expand on globally
        without restarting the intent classifier or touching
        ``rag_config``, env vars become a debug knob. Costs: blast
        radius (env stays set forever), defeats the per-intent shaping.

    Per Plan A B3 memory note: A/B both modes against real production
    queries before locking the default. Until then, ``intent`` is the
    safer default and matches the current behaviour Phase 2.2 shipped.

    Unknown intents (typo, future label) fall back to the ``specific``
    defaults — the largest bucket on real corpora, generally-useful
    pipeline shape.

    Pure function — no side effects (env reads are deterministic given
    a process snapshot). Inputs are not mutated. Returns a fresh dict.
    """
    intent_defaults = _INTENT_FLAG_POLICY.get(intent, _INTENT_FLAG_POLICY["specific"])
    merged: dict[str, str] = dict(intent_defaults)

    # B3 mode toggle: when env should win, drop overlay keys that env has set.
    overlay_mode = _os.environ.get("RAG_INTENT_OVERLAY_MODE", "intent").lower()
    if overlay_mode == "env":
        for key in list(merged.keys()):
            if _os.environ.get(key) is not None:
                del merged[key]

    # Per-KB explicit overrides ALWAYS win, in either mode.
    if per_kb_overrides:
        for k, v in per_kb_overrides.items():
            merged[str(k)] = str(v)
    return merged


def _debug_query_extras(query: Optional[str], hits_detail: Optional[list]) -> dict:
    """Wave 2 (review §8.9): build the optional ``query_text`` + ``chunks_summary``
    extras passed via ``logger.info(..., extra={...})`` when ``RAG_LOG_QUERY_TEXT=1``.
    PII-sensitive — see ``_log_rag_query`` docstring + RAG_LOG_QUERY_TEXT comment.
    """
    out: dict = {}
    if query:
        out["query_text"] = str(query)[:1024]
    if hits_detail:
        out["chunks_summary"] = [
            {
                "chunk_id": getattr(h, "id", None),
                "score": float(getattr(h, "score", 0.0) or 0.0),
                "filename": str((getattr(h, "payload", {}) or {}).get("filename", "")),
            }
            for h in list(hits_detail)[:3]
        ]
    return out


def _log_rag_query(
    *,
    req_id: str,
    intent: str,
    kbs: list,
    hits: int,
    total_ms: int,
    query: Optional[str] = None,
    hits_detail: Optional[list] = None,
) -> None:
    """Emit one ``event=rag_query`` JSON log line.

    Fail-open: any exception inside ``json.dumps`` / the logger call is
    swallowed because this is a telemetry side-effect — the retrieval
    pipeline must not fail because the log record could not be built.

    Wave 2 (review §8.9): when ``RAG_LOG_QUERY_TEXT=1`` (default 0,
    PII-sensitive), the logger receives ``query_text`` (truncated to 1 KB)
    and a ``chunks_summary`` (top-3 ``{chunk_id, score, filename}``) via
    ``extra={...}`` so the JSON formatter lifts them into the structured
    payload. The base JSON message is unchanged so existing dashboards
    keep working.
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
        extra = _debug_query_extras(query, hits_detail) if _os.environ.get(
            "RAG_LOG_QUERY_TEXT", "0"
        ) == "1" else {}
        logger.info(_json.dumps(payload, separators=(",", ":")), extra=extra)
    except Exception as _err:
        # B6: telemetry side-effect — must not fail the pipeline. Surface
        # the swallow via warning + counter so a broken json.dumps shape
        # (new field, weird type) is investigable.
        _record_silent_failure("log_rag_query", _err)

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
#
# §7.5: ``_RBAC_REDIS_LOCK`` guards the read+create against two concurrent
# first-callers each building (and leaking) their own client.
_rbac_redis = None
_RBAC_REDIS_LOCK = threading.Lock()


def _redis_client():
    """Return the shared async redis handle used by the RBAC cache.

    Creates the handle on first call from ``RAG_RBAC_CACHE_REDIS_URL``
    (default ``redis://localhost:6379/3`` so it's isolated from the
    application redis on DB 0). Subsequent calls return the same
    handle.

    Thread-safe singleton init under ``_RBAC_REDIS_LOCK`` (§7.5).
    """
    global _rbac_redis
    if _rbac_redis is not None:
        return _rbac_redis
    with _RBAC_REDIS_LOCK:
        # Re-check inside the lock — another caller may have raced past the
        # first None-check and finished the init while we were blocked.
        if _rbac_redis is not None:
            return _rbac_redis
        import redis.asyncio as _redis
        url = _os.environ.get(
            "RAG_RBAC_CACHE_REDIS_URL", "redis://localhost:6379/3"
        )
        _rbac_redis = _redis.from_url(url)
        return _rbac_redis


_CONFIGURED: bool = False


def configure(*, vector_store, embedder, sessionmaker) -> None:
    """Set the module-level singletons used by the retrieval pipeline.

    Wave 2 (review §7.4): idempotent. The first call wins; subsequent
    calls (test-setup runs after app-startup, repeat boot probes, etc.)
    are no-ops. Without the guard, two concurrent first-callers could
    silently overwrite each other's references to vector_store/embedder
    and leak Redis/httpx pools.
    """
    global _vector_store, _embedder, _sessionmaker, _CONFIGURED
    if _CONFIGURED:
        return
    _vector_store = vector_store
    _embedder = embedder
    _sessionmaker = sessionmaker
    _CONFIGURED = True


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
        # B6: bumped from debug → warning + counter. A failing chat-meta
        # lookup means the user gets ZERO KB scoping — silent in prod
        # before this fix.
        _record_silent_failure("kb_config_lookup", e)
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
        # B6: per-KB rag_config drives MMR / hybrid / context-expand
        # toggles. A silent failure here reverts every KB to process
        # defaults — important to alert on, hence warning + counter.
        _record_silent_failure("kb_rag_config_load", e)
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
        # B6: a swallowed DB error here demotes specific_date intent to
        # generic specific (per the caller's fall-through). Operators
        # need to know — bumped to warning + counter.
        _record_silent_failure("date_doc_lookup", e)
        return []


# Wave 2 round 4 (review §5.15) — total-pipeline timeout. Default 30s is
# generous; production pathologies (downstream service stuck, deadlocked
# semaphore) are the only thing that should trip this. Read at request
# time so an operator can dial it without a process restart.
def _total_budget_seconds() -> float:
    raw = _os.environ.get("RAG_TOTAL_BUDGET_SEC", "30")
    try:
        v = float(raw)
        return v if v > 0 else 30.0
    except (TypeError, ValueError):
        return 30.0


# ---------------------------------------------------------------------------
# Wave 2 round 6 (review §6.11) — Calibrated abstention.
#
# The "zero hedging" rule in the analyst system prompt encourages
# confabulation when retrieval is weak. ``compute_abstention_prefix``
# returns a one-line caveat to PREPEND to the system prompt for THIS
# request only, when:
#   * RAG_ENFORCE_ABSTENTION=1 (default 0 = OFF, returns ""), AND
#   * average rerank-top-k score < RAG_ABSTENTION_THRESHOLD (default 0.1).
#
# Don't mutate the global system prompt — callers receive the prefix and
# prepend it themselves so each request is independent. When the helper
# returns a non-empty string, ``rag_abstention_caveat_added_total{intent}``
# is incremented.
#
# Fail-open: any internal exception returns ""; the request still flies.
# Lives at module top-level (NOT inside _run_pipeline) so Wave 6E does not
# touch the protected pipeline body.
# ---------------------------------------------------------------------------
ABSTENTION_CAVEAT = (
    "If the retrieved context is insufficient to answer accurately, "
    "respond 'I don't have enough information to answer that based on "
    "the documents I have access to.'"
)


def _abstention_threshold() -> float:
    """Return RAG_ABSTENTION_THRESHOLD as a float (default 0.1).

    Garbage values fail to parse → fall back to 0.1 (don't crash the
    request because an operator typed ``0,1`` instead of ``0.1``).
    """
    raw = flags.get("RAG_ABSTENTION_THRESHOLD", "0.1")
    try:
        v = float(raw)
        return v if v >= 0 else 0.1
    except (TypeError, ValueError):
        return 0.1


def _hit_score(hit: Any) -> float:
    """Pull a numeric ``score`` from a hit (object attr OR dict key).

    Returns 0.0 if no score is found (so missing-score hits drag the avg
    down — defensive, the abstention path is the safer side to err on).
    """
    # Object with .score attribute
    s = getattr(hit, "score", None)
    if s is None and isinstance(hit, dict):
        # Dict shape — sources_out items / hit_summary entries.
        s = hit.get("score")
    if s is None:
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def compute_abstention_prefix(
    hits: Any,
    *,
    intent: str = "specific",
) -> str:
    """Return a one-line caveat to prepend to the system prompt, or ``""``.

    Args:
      hits: An iterable of hits — anything with a ``score`` attribute or
        ``"score"`` key. Empty iterable is treated as 0-score (driving the
        avg below any positive threshold → caveat added).
      intent: Pipeline intent label used for the
        ``rag_abstention_caveat_added_total`` counter.

    Returns:
      The caveat string when the flag is on AND avg score < threshold;
      empty string otherwise. Always a single line so callers can safely
      prepend it as a one-line system-prompt prefix.
    """
    # Cheapest exit when the flag is off.
    if flags.get("RAG_ENFORCE_ABSTENTION", "0") != "1":
        return ""

    try:
        scores = [_hit_score(h) for h in (hits or [])]
        # Empty hits → avg is 0 (definitely below any positive threshold,
        # which is the most common 'retrieval failed' signal — caveat).
        avg = (sum(scores) / len(scores)) if scores else 0.0
        threshold = _abstention_threshold()
        if avg >= threshold:
            return ""
        # Below threshold → add caveat + bump counter.
        try:
            from .metrics import rag_abstention_caveat_added_total

            rag_abstention_caveat_added_total.labels(intent=str(intent)).inc()
        except Exception:
            # Metric back-end missing or label issue → never break the
            # request. Caveat still returned.
            pass
        return ABSTENTION_CAVEAT
    except Exception as exc:
        # Fail-open: log + return empty so the request continues unchanged.
        try:
            logger.warning(
                "rag: compute_abstention_prefix failed (%s): %r",
                type(exc).__name__, exc,
            )
        except Exception:
            pass
        return ""


def _build_datetime_preamble_source() -> Optional[dict]:
    """Return the canonical ``current-datetime`` source dict (or ``None``).

    Mirrors the inline preamble that ``_run_pipeline`` builds — extracted so
    the §5.15 timeout fallback can return at least a degraded preamble
    without touching the database.
    """
    if _os.environ.get("RAG_INJECT_DATETIME", "1") == "0":
        return None
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
        return {
            "source": {"id": "current-datetime", "name": "current-datetime",
                       "url": "current-datetime"},
            "document": [_dt_text],
            "metadata": [{
                "source": "current-datetime", "name": "current-datetime",
                "kb_id": None, "doc_id": None, "chat_id": None,
                "subtag_id": None,
            }],
        }
    except Exception:
        return None


# Wave 2 round 4 (review §5.2) — cached system-prompt token count for the
# RAG_BUDGET_INCLUDES_PROMPT estimator. Loaded lazily on first call so the
# tokenizer doesn't get touched at import time. ``None`` means "tried and
# failed" — fall back to the static estimate.
_SYSTEM_PROMPT_TOKEN_CACHE: Optional[int] = None
_SYSTEM_PROMPT_HASH_CACHE: Optional[str] = None


def system_prompt_version_hash() -> str:
    """Wave 2 (review §6.6): sha256[:12] of the analyst system prompt.

    Used to label LLM telemetry (prompt_hash) and exposed as a Prometheus
    gauge so a quality regression can be bisected to a specific prompt
    revision (apply_analyst_config.py overwrites silently — without the
    hash there's no audit trail). Computed once on first call; mutates
    only when the file changes (operator must re-import to refresh).
    """
    global _SYSTEM_PROMPT_HASH_CACHE
    if _SYSTEM_PROMPT_HASH_CACHE is not None:
        return _SYSTEM_PROMPT_HASH_CACHE
    try:
        import hashlib
        from pathlib import Path
        _path = Path(__file__).resolve().parents[2] / "scripts" / "system_prompt_analyst.txt"
        if _path.is_file():
            digest = hashlib.sha256(_path.read_bytes()).hexdigest()[:12]
            _SYSTEM_PROMPT_HASH_CACHE = digest
            try:
                from .metrics import RAG_SYSTEM_PROMPT_VERSION
                RAG_SYSTEM_PROMPT_VERSION.labels(hash=digest).set(1)
            except Exception:
                pass
            return digest
    except Exception:
        pass
    _SYSTEM_PROMPT_HASH_CACHE = "unknown"
    return _SYSTEM_PROMPT_HASH_CACHE


def _system_prompt_tokens() -> int:
    """Return the token count of the analyst system prompt (cached).

    Reads ``scripts/system_prompt_analyst.txt`` (the canonical source that
    ``apply_analyst_config.py`` writes into the chat config) and tokenizes
    it with the budget tokenizer. Falls back to a calibrated 1500-token
    estimate if the file is missing or the tokenizer load failed — this
    matches the production prompt's typical size when measured with the
    gemma-4 tokenizer.
    """
    global _SYSTEM_PROMPT_TOKEN_CACHE
    if _SYSTEM_PROMPT_TOKEN_CACHE is not None:
        return _SYSTEM_PROMPT_TOKEN_CACHE
    try:
        from pathlib import Path
        from .budget import _count_tokens
        # ext/services/chat_rag_bridge.py → repo root → scripts/
        _path = Path(__file__).resolve().parents[2] / "scripts" / "system_prompt_analyst.txt"
        if _path.is_file():
            text = _path.read_text(encoding="utf-8")
            _SYSTEM_PROMPT_TOKEN_CACHE = _count_tokens(text)
            return _SYSTEM_PROMPT_TOKEN_CACHE
    except Exception:
        # Tokenizer or filesystem failure — fall through to the static estimate.
        pass
    _SYSTEM_PROMPT_TOKEN_CACHE = 1500  # measured baseline for the analyst prompt
    return _SYSTEM_PROMPT_TOKEN_CACHE


def _estimate_reserved_tokens(*, n_hits: int, intent: str) -> int:
    """Return total non-chunk tokens the budget should pre-deduct.

    Composition:
      * system prompt — measured (cached) via ``_system_prompt_tokens``
      * KB catalog preamble — 800 tokens worst-case (50-doc cap, ~10-15
        tokens per filename + ~120-token header). Catalog is added for
        every intent so we always include it.
      * Datetime preamble — 40 tokens (RAG_INJECT_DATETIME default-on)
      * Spotlight wrap — 30 tokens per hit (open + close + sanitize markers).

    intent is currently informational only — every code path adds catalog
    and datetime. Kept as a parameter so a future intent (e.g. an ablation
    that drops the catalog) can shrink the reserve without a signature break.
    """
    _ = intent  # reserved for future intent-conditional adjustments
    return _system_prompt_tokens() + 800 + 40 + (30 * max(0, int(n_hits)))


def _apply_entity_quota(
    *,
    reranked: list,
    entities: list,
    per_entity_floor: int,
    final_k: int,
) -> list:
    """Build the post-rerank pool with a per-entity floor (multi-entity fix).

    Backstop for the 75 Inf Bde / 5 PoK Bde eviction case (smoke test
    2026-05-03). The plain ``reranked[:final_k]`` trim is global-score
    top-k — entity-blind. When multi-entity decompose was active and one
    entity's chunks happened to score below the dominant entity's tail,
    the trim silently evicted the entire entity bucket. The LLM then
    answered "no information for X" even though direct Qdrant scrolls
    showed matching content.

    Algorithm:
      1. **Quota pass.** For each entity in ``entities``, walk
         ``reranked`` in cross-encoder order and pick the top
         ``per_entity_floor`` hits whose ``payload["text"]`` contains
         the entity name (case-insensitive substring). If an entity has
         FEWER than ``per_entity_floor`` matching chunks in
         ``reranked``, take what's available — don't pad. The
         "no info for X" answer is then accurate, not a quota artefact.
      2. **Top-up pass.** Fill remaining slots up to ``final_k`` with
         the next-highest-score hits not already picked.
      3. **Dedup by id.** A chunk that mentions two entities counts once.
      4. **Final order.** Sort the result by score desc so the LLM sees
         the same "earlier == more relevant" contract as the legacy trim.
      5. **Cap at ``final_k``.**

    Pure function — no I/O, no logging side-effects. Bridge call site
    handles telemetry. ``per_entity_floor=0`` OR ``entities=[]``
    short-circuits to ``reranked[:final_k]`` (quota disabled).
    """
    if not reranked:
        return []
    if not entities or per_entity_floor <= 0:
        return list(reranked[:final_k])

    # Pre-lowercase entity names once.
    entity_needles = [(e, str(e).lower()) for e in entities]

    selected_ids: set = set()
    selected: list = []

    # Step 1 — quota pass. For each entity, take its top per_entity_floor
    # matches in cross-encoder order. Same chunk matching multiple entities
    # only enters the result once (first entity that picks it wins, but
    # subsequent entities can still walk past it without double counting).
    for _entity, needle in entity_needles:
        taken = 0
        for hit in reranked:
            if taken >= per_entity_floor:
                break
            if hit.id in selected_ids:
                # Already in pool from another entity bucket — counts toward
                # this entity's quota too (same chunk mentions both names).
                payload = getattr(hit, "payload", None) or {}
                text = payload.get("text") if isinstance(payload, dict) else None
                if text and needle in str(text).lower():
                    taken += 1
                continue
            payload = getattr(hit, "payload", None) or {}
            text = payload.get("text") if isinstance(payload, dict) else None
            if not text:
                continue
            if needle in str(text).lower():
                selected_ids.add(hit.id)
                selected.append(hit)
                taken += 1

    # Step 2 — top-up pass. Fill remaining slots by next-best score
    # from the original reranked list (cross-encoder order). Skip
    # already-selected hits.
    if len(selected) < final_k:
        for hit in reranked:
            if len(selected) >= final_k:
                break
            if hit.id in selected_ids:
                continue
            selected_ids.add(hit.id)
            selected.append(hit)

    # Step 3 — final sort by cross-encoder score desc, capped.
    selected.sort(key=lambda h: float(getattr(h, "score", 0.0) or 0.0), reverse=True)

    # Phase 3 / 2026-05-03 — observability bump.
    try:
        from .metrics import rag_multi_entity_coverage_total
        counts = {e: 0 for e in entities}
        for hit in selected:
            text_low = ((hit.payload or {}).get("text") or "").lower()
            for e in entities:
                if e.lower() in text_low:
                    counts[e] += 1
                    break  # one entity attribution per chunk
        n_zero = sum(1 for c in counts.values() if c == 0)
        n_partial = sum(1 for c in counts.values() if 0 < c < per_entity_floor)
        outcome = "empty" if n_zero else ("partial" if n_partial else "full")
        rag_multi_entity_coverage_total.labels(
            outcome=outcome, entity_count=str(len(entities)),
        ).inc()
    except Exception:
        # Telemetry is fail-open; never break retrieval on counter error.
        pass

    return selected[:final_k]


async def _emit(cb: Optional[Callable[[dict], Awaitable[None]]], event: dict) -> None:
    """Call the progress callback, swallowing any errors so a broken SSE
    client never breaks retrieval. No-op when cb is None."""
    if cb is None:
        return
    try:
        await cb(event)
    except Exception as _err:
        # Fail open — the caller disconnected or the consumer is broken.
        # B6: still surface via warning + counter; a steady stream of
        # progress_emit failures means SSE is broken upstream.
        _record_silent_failure("progress_emit", _err)


async def _multi_entity_retrieve(
    *,
    entities: list,
    base_query: str,
    selected_kbs,
    chat_id,
    vector_store,
    embedder,
    per_kb_limit: int,
    total_limit: int,
    owner_user_id=None,
    level_filter=None,
    doc_ids=None,
    temporal_constraint=None,
    with_vectors: bool = False,
):
    """Multi-entity decomposed retrieval (Phase 6.X — Methods 3 + 4).

    Fans out one parallel ``retrieve(...)`` call per named entity and
    merges results with a per-entity quota floor. Each sub-call uses a
    focus-shifted sub-query (``"<base> (focus on <entity>)"``); when
    ``RAG_ENTITY_TEXT_FILTER=1`` (Method 4) it also passes
    ``text_filter=entity`` so Qdrant restricts that bucket to chunks
    literally naming the entity.

    Per-entity floor comes from ``RAG_MULTI_ENTITY_MIN_PER_ENTITY``
    (default 10, bounded [1, 50]). The floor is best-effort — when an
    entity bucket has fewer hits than the floor, we take what exists
    and let the top-up pass fill remaining slots from other buckets.

    Method 4 fail-open: when the text filter pulls 0 hits for an
    entity, we retry that entity's sub-query without the filter so a
    too-strict lexical match doesn't silently leave the entity
    unrepresented in the final candidate set. The retry is recorded
    via ``rag_entity_text_filter_total{outcome="filter_empty_retry"}``.

    All sub-calls go through the standard ``retrieve(...)`` so RBAC,
    temporal filters, level filter, owner filter, and doc-id filter
    apply unchanged. Only ``query`` and (optionally) ``text_filter``
    differ across the N parallel calls.
    """
    import asyncio as _asyncio

    from .multi_query import build_sub_queries, merge_with_quota
    from .retriever import retrieve as _retrieve

    pairs = build_sub_queries(base_query, entities)
    text_filter_on = flags.get("RAG_ENTITY_TEXT_FILTER", "0") == "1"
    try:
        floor = int(flags.get("RAG_MULTI_ENTITY_MIN_PER_ENTITY") or "10")
    except (TypeError, ValueError):
        floor = 10
    floor = max(1, min(floor, 50))

    n_kbs = max(1, len(selected_kbs or []))
    # Each per-entity sub-call asks for up to per_kb_limit chunks per KB.
    # The total cap is per_kb_limit * n_kbs so an entity bucket isn't
    # truncated below its KB share before the merge sees it.
    sub_total = per_kb_limit * n_kbs

    async def _retrieve_one(entity, sub_query, *, with_filter=True):
        try:
            return await _retrieve(
                query=sub_query,
                selected_kbs=selected_kbs,
                chat_id=chat_id,
                vector_store=vector_store,
                embedder=embedder,
                per_kb_limit=per_kb_limit,
                total_limit=sub_total,
                owner_user_id=owner_user_id,
                level_filter=level_filter,
                doc_ids=doc_ids,
                temporal_constraint=temporal_constraint,
                text_filter=(entity if (with_filter and text_filter_on) else None),
                with_vectors=with_vectors,
            )
        except Exception as exc:
            _record_silent_failure(
                f"multi_entity.retrieve[{str(entity)[:20]}]", exc,
            )
            return []

    # Fan out N parallel retrievals — N == len(entities) ≤ 8.
    raw_buckets = await _asyncio.gather(
        *(_retrieve_one(e, sq) for (e, sq) in pairs),
    )
    per_entity_hits = {
        e: hits for ((e, _), hits) in zip(pairs, raw_buckets)
    }

    # Method 4 fail-open. When a per-entity bucket comes back empty
    # AND the text filter was on, retry that entity without the filter
    # before declaring it has no data. This guards against a lexical
    # filter that's too strict for an entity whose canonical surface
    # form differs from corpus phrasing.
    if text_filter_on:
        empty = [e for e, hits in per_entity_hits.items() if not hits]
        if empty:
            sub_by_e = dict(pairs)
            retry = await _asyncio.gather(
                *(_retrieve_one(e, sub_by_e[e], with_filter=False) for e in empty),
            )
            for e, hits in zip(empty, retry):
                per_entity_hits[e] = hits
                try:
                    from .metrics import rag_entity_text_filter_total
                    rag_entity_text_filter_total.labels(
                        outcome="filter_empty_retry"
                    ).inc()
                except Exception:
                    pass

    merged = merge_with_quota(
        per_entity_hits=per_entity_hits,
        k_min_per_entity=floor,
        k_total=total_limit,
    )

    try:
        from .metrics import (
            rag_entity_text_filter_total,
            rag_multi_query_decompose_total,
        )

        rag_multi_query_decompose_total.labels(outcome="decomposed").inc()
        rag_entity_text_filter_total.labels(
            outcome="applied" if text_filter_on else "bypassed"
        ).inc()
    except Exception:
        pass

    logger.info(
        "rag: multi-entity decompose entities=%d filter=%s floor=%d total=%d -> %d hits",
        len(entities), text_filter_on, floor, total_limit, len(merged),
    )
    return merged


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
    except Exception as _err:
        # B6: gauge inc failed — usually a prometheus_client missing
        # entirely. Logging once per session is acceptable, dashboards
        # rely on the counter to tell us this is happening.
        _record_silent_failure("session_gauge_inc", _err)
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
            except Exception as _err:
                _record_silent_failure("session_gauge_dec", _err)


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
        # Phase 1.5 — cache-first lookup via the shared helper. Both the
        # SSE / chat path here AND ``/api/rag/retrieve`` (rag.py) call
        # this so the cache contract is identical across surfaces. The
        # helper itself is the source of truth for the cache namespace
        # / TTL / fail-open semantics — see ``rbac.resolved_allowed_kb_ids``.
        #
        # B4 — wrap the RBAC lookup in an OTel span so Jaeger can show
        # the per-user cache vs DB latency split (the helper itself emits
        # no spans). Tag user_id (string), kb-config size, and the
        # resolved allowed-KB count for correlation.
        from .rbac import resolved_allowed_kb_ids
        with span(
            "rag.rbac_check",
            user_id=str(user_id),
            requested_kb_count=len(kb_config or []),
        ) as _rbac_sp:
            try:
                async with _sessionmaker() as s:
                    allowed = await resolved_allowed_kb_ids(
                        s, user_id=user_id, redis=_redis_client(),
                    )
                try:
                    _rbac_sp.set_attribute("allowed_kb_count", len(allowed))
                except Exception:
                    pass
            except Exception as _rbac_exc:
                # Re-raise so the existing outer error path (or caller)
                # surfaces the issue. The span context manager records
                # the exception + sets ERROR status automatically.
                if _rbac_sp is not None:
                    try:
                        _rbac_sp.record_exception(_rbac_exc)
                    except Exception:
                        pass
                raise
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

    # Wave 2 round 4 (review §5.15) — total-pipeline timeout. Wrap the inner
    # pipeline in asyncio.wait_for so a hung downstream service (TEI, vllm-qu,
    # chat model) can't deadlock the chat session indefinitely. On timeout
    # we increment ``rag_pipeline_timeout_total{intent}`` and return an
    # early-degraded source list — at minimum the datetime preamble so the
    # LLM still has an authoritative anchor to answer "what's today" with.
    # Catalog is intentionally omitted because it requires a DB read which
    # may itself be the source of the timeout.
    _budget_sec = _total_budget_seconds()
    _intent_label_for_timeout = _intent_label_for_flags
    with flags.with_overrides(merged_overrides):
        try:
            return await asyncio.wait_for(
                _run_pipeline(
                    query=query,
                    selected_kbs=selected_kbs,
                    user_id=user_id,
                    chat_id=chat_id,
                    history=history,
                    progress_cb=progress_cb,
                ),
                timeout=_budget_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "rag.pipeline_timeout: budget=%.1fs intent=%s",
                _budget_sec, _intent_label_for_timeout,
            )
            try:
                from .metrics import rag_pipeline_timeout_total
                rag_pipeline_timeout_total.labels(
                    intent=str(_intent_label_for_timeout)
                ).inc()
            except Exception:  # pragma: no cover - metrics fail-open
                pass
            try:
                await _emit(progress_cb, {
                    "stage": "error",
                    "message": "pipeline_timeout",
                    "budget_sec": _budget_sec,
                })
            except Exception:
                pass
            degraded: list[dict] = []
            _dt_source = _build_datetime_preamble_source()
            if _dt_source is not None:
                degraded.append(_dt_source)
            return degraded


async def _run_pipeline(
    *,
    query: str,
    selected_kbs: list,
    user_id: str,
    chat_id: Optional[str],
    progress_cb: Optional[Callable[[dict], Awaitable[None]]],
    history: Optional[List[dict]] = None,
) -> List[dict]:
    """Inner pipeline — runs under a ``with_overrides`` scope.

    Separated from ``retrieve_kb_sources`` so the overlay is active for
    every lazy import and every flag read inside the hot path.
    """
    # Plan B Phase 4.6 — hybrid regex+LLM intent classification. The async
    # ``_classify_with_qu`` returns a HybridClassification carrying both the
    # intent label and a resolved (standalone) form of the query. When
    # ``RAG_QU_ENABLED=0`` (default until shadow A/B sign-off) the result
    # is regex-only and ``resolved_query`` equals ``query`` — the pipeline
    # is byte-identical to the pre-Plan-B path.
    #
    # We use ``resolved_query`` for retrieval / rerank because pronouns and
    # relative time have been resolved against history (better dense recall),
    # but keep the original ``query`` for response framing — the user's
    # exact phrasing belongs in the answer's preamble.
    #
    # B4 — wrap the hybrid classifier so Jaeger can correlate intent
    # source (regex vs LLM vs cached), confidence, and any escalation
    # reason with the rest of the pipeline. Failures inside QU still
    # soft-fall to regex (handled inside ``_classify_with_qu``); the
    # span just attributes the chosen result.
    _last_turn_id = _extract_last_turn_id(history)
    with span(
        "rag.intent_classify",
        query_len=len(query or ""),
        history_turns=len(history or []),
    ) as _intent_sp:
        _hybrid = await _classify_with_qu(
            query=query, history=history, last_turn_id=_last_turn_id,
        )
        try:
            _intent_sp.set_attribute("intent", str(_hybrid.intent))
            _intent_sp.set_attribute("source", str(_hybrid.source))
            _intent_sp.set_attribute("confidence", float(_hybrid.confidence or 0.0))
            _intent_sp.set_attribute("cached", bool(getattr(_hybrid, "cached", False)))
            _esc = getattr(_hybrid, "escalation_reason", None)
            if _esc is not None:
                _intent_sp.set_attribute(
                    "escalation_reason",
                    getattr(_esc, "value", None) or str(_esc),
                )
        except Exception:
            # Span attribute set failures are never fatal — continue.
            pass
    _intent_label = _hybrid.intent
    _retrieval_query = (
        _hybrid.resolved_query
        if (_hybrid.source == "llm" and _hybrid.resolved_query
            and _hybrid.resolved_query != query)
        else query
    )
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
            _record_silent_failure("qdrant_preflight", _hc_exc)

    # Tier 2 — intent routing (gated by RAG_INTENT_ROUTING, default OFF).
    # When OFF, _intent/_intent_reason are fixed values and every branch
    # below collapses to the pre-Tier-2 ``specific`` path, making the
    # pipeline byte-identical. When ON:
    #   * metadata  → skip chunk retrieve; catalog preamble answers.
    #   * global    → retrieve only level="doc" summary points; skip
    #                 rerank/MMR/expand (summaries are self-contained).
    #   * specific  → unchanged pipeline.
    #
    # Phase 1.2 fix — when the QU LLM produced a high-confidence intent
    # (``_hybrid.source == "llm"`` and ``confidence >= RAG_QU_INTENT_MIN_CONF``),
    # use it instead of regex. The LLM sees pronouns, multi-clause queries,
    # and "in May, July, December, and February 2023"-style phrasings that
    # the regex's first-match-wins table can't model. Falls back to regex
    # on QU failure (``_hybrid.source != "llm"`` covers timeout/error/cache).
    _intent: str = "specific"
    _intent_reason: str = "default"
    _qu_min_conf = float(flags.get("RAG_QU_INTENT_MIN_CONF") or "0.80")
    if flags.get("RAG_INTENT_ROUTING", "0") == "1":
        from .query_intent import classify_with_reason as _ci_classify
        _regex_intent, _regex_reason = _ci_classify(query)
        if (
            _hybrid is not None
            and _hybrid.source == "llm"
            and _hybrid.intent in ("metadata", "global", "specific", "specific_date")
            and float(_hybrid.confidence or 0.0) >= _qu_min_conf
        ):
            _intent = _hybrid.intent
            _intent_reason = (
                f"llm:conf={_hybrid.confidence:.2f}"
                f"|regex_was={_regex_intent}:{_regex_reason}"
            )
        else:
            _intent, _intent_reason = _regex_intent, _regex_reason
        logger.info(
            "rag: intent=%s reason=%s source=%s",
            _intent, _intent_reason, _hybrid.source if _hybrid else "regex",
        )

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
    except Exception as _err:
        _hybrid_flag = True  # best-effort for hit-counter label below
        _record_silent_failure("flag_state_emit", _err)

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
            # B4 — outer ``rag.retrieve`` span covers routing decisions,
            # date-doc lookup, level-filter selection, AND the actual
            # Qdrant fan-out (the latter remains a nested ``retrieve.parallel``
            # for compatibility with existing dashboards). Sharding mode is
            # tagged via the level filter (doc/chunk/all) and per-KB top-k
            # is recorded so operators can correlate with hit counts.
            _rag_retrieve_span_cm = span(
                "rag.retrieve",
                kb_ids=_kb_ids_csv,
                n_kbs=len(selected_kbs or []),
                top_k=int(flags.get("RAG_RERANK_TOP_K") or 0) or 10,
                intent=str(_intent),
            )
            with time_stage("retrieve"), _rag_retrieve_span_cm as _rag_retrieve_sp, span(
                "retrieve.parallel",
                n_kbs=len(selected_kbs or []),
                kb_ids=_kb_ids_csv,
                chat_id=str(chat_id) if chat_id else "",
            ):
                _tR = time.perf_counter()
                # 2026-05-03 fix: initialize multi-entity quota variables upfront so the
                # metadata-intent path (which short-circuits before the decompose block)
                # doesn't crash when the post-rerank quota check references them.
                # See docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §1.4.
                _do_decompose: bool = False
                _entities: list[str] = []
                _entity_floor: int = 0
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
                    # Per-KB rag_config override (option B from the
                    # 32 Inf Bde eviction case 2026-05-01). When any
                    # selected KB stamps top_k in its rag_config, the
                    # value enters the request overlay as RAG_TOP_K via
                    # config_to_env_overrides; max-merge is already
                    # applied across selected KBs by merge_configs.
                    # Take the larger of the intent default and the
                    # override so an admin who set top_k=24 on a heavy
                    # KB doesn't get *narrowed* by a stricter intent.
                    _kb_top_k = 0
                    try:
                        _kb_top_k = int(flags.get("RAG_TOP_K") or 0)
                    except (TypeError, ValueError):
                        _kb_top_k = 0
                    if _kb_top_k > 0:
                        _per_kb = max(_per_kb, _kb_top_k)
                        _total = max(_total, _kb_top_k * max(1, len(selected_kbs or [])))
                    # Phase 2.3 — multi-temporal scaling. When the QU LLM
                    # extracted N>1 distinct months/quarters/years (so the
                    # query's candidate pool is N monthly shards, not one),
                    # bump the per_kb/total budget so each shard contributes
                    # ~10-12 chunks. Without this, top-30 across 4 shards
                    # gives ~7 hits/shard — and the LLM "no activities for
                    # February 2023" failure mode resurfaces.
                    if _hybrid is not None and _hybrid.source == "llm":
                        _tc = _hybrid.temporal_constraint
                        if _tc:
                            from .retriever import _shard_keys_for_constraint
                            _n_shards = len(_shard_keys_for_constraint(_tc))
                            if _n_shards > 1:
                                _per_kb = max(_per_kb, 12 * _n_shards)
                                _total = max(_total, 12 * _n_shards * max(
                                    1, len(selected_kbs or []),
                                ))
                                logger.info(
                                    "rag: multi-temporal boost per_kb=%d total=%d shards=%d",
                                    _per_kb, _total, _n_shards,
                                )
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
                    # Phase 2.2 — pass the QU LLM's temporal_constraint to
                    # the retriever. It expands ``months``/``years`` into
                    # the matching shard_keys via _shard_keys_for_constraint
                    # and adds a Qdrant ``shard_key`` MatchAny filter on
                    # custom-sharded collections (kb_*_v* with monthly
                    # buckets). Multi-month queries ("May, Jul, Dec, Feb
                    # 2023") thus restrict the candidate pool to those 4
                    # shards, dramatically improving recall versus a flat
                    # top-30 over 8000+ chunks. Falls back to None when
                    # QU returned regex (no temporal_constraint).
                    _temporal_constraint = (
                        _hybrid.temporal_constraint
                        if (_hybrid is not None and _hybrid.source == "llm")
                        else None
                    )
                    # Phase 6.X — multi-entity decomposition gate (Method 3).
                    # When RAG_MULTI_ENTITY_DECOMPOSE=1 AND the query
                    # contains ≥2 named entities, fan out one parallel
                    # retrieve per entity and merge with a per-entity
                    # quota. Default-off path is byte-identical to
                    # pre-Phase-6: same single retrieve call, same
                    # arguments, same return.
                    _decompose_on = flags.get("RAG_MULTI_ENTITY_DECOMPOSE", "0") == "1"
                    _entities: list = []
                    _do_decompose = False
                    if _decompose_on:
                        try:
                            from .entity_extractor import extract_entities
                            from .multi_query import should_decompose

                            _entities = extract_entities(query, qu_result=_hybrid)
                            _do_decompose = should_decompose(
                                entities=_entities,
                                flag_on=True,
                                intent=_intent,
                            )
                        except Exception as _exc:
                            _record_silent_failure(
                                "multi_entity.gate", _exc,
                            )
                            _do_decompose = False
                    if not _do_decompose:
                        # Telemetry for the "flag on, but query was
                        # single-entity" case so operators can spot
                        # corpora where decomposition never fires.
                        try:
                            from .metrics import rag_multi_query_decompose_total
                            rag_multi_query_decompose_total.labels(
                                outcome="skipped_flag_off"
                                if not _decompose_on
                                else "skipped_no_entities"
                            ).inc()
                        except Exception:
                            pass
                    # Wave 2 round 4 (review §5.9) — request dense vectors back
                    # from Qdrant when MMR will run on this request, so the
                    # MMR helper can skip its TEI re-embed of every passage.
                    # MMR runs when ``RAG_MMR=1`` AND intent != "global"
                    # (global skips MMR; doc summaries are self-contained).
                    # Reading the flag through ``flags.get`` honors per-KB
                    # rag_config overrides set up in ``_retrieve_overrides``.
                    with flags.with_overrides(_retrieve_overrides):
                        _with_vectors_for_mmr = (
                            flags.get("RAG_MMR", "0") == "1"
                            and _intent != "global"
                        )
                    with span("embed.query", path="query"), flags.with_overrides(_retrieve_overrides):
                        # ``retrieve`` internally calls embedder.embed() — the
                        # embed.call span will nest under this embed.query span
                        # giving operators a clear query-embed timing.
                        if _do_decompose:
                            raw_hits = await _multi_entity_retrieve(
                                entities=_entities,
                                base_query=_retrieval_query,
                                selected_kbs=selected_kbs,
                                chat_id=chat_id,
                                vector_store=_vector_store,
                                embedder=_embedder,
                                per_kb_limit=_per_kb,
                                total_limit=_total,
                                owner_user_id=user_id,
                                level_filter=_level_filter,
                                doc_ids=_date_doc_ids,
                                temporal_constraint=_temporal_constraint,
                                with_vectors=_with_vectors_for_mmr,
                            )
                        else:
                            raw_hits = await retrieve(
                                query=_retrieval_query,
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
                                temporal_constraint=_temporal_constraint,
                                with_vectors=_with_vectors_for_mmr,
                            )
                _retrieve_ms = int((time.perf_counter() - _tR) * 1000)
                # B4 — tag the rag.retrieve span with the post-retrieval
                # numbers so Jaeger shows hits + latency + sharding mode
                # without needing to descend into retrieve.parallel.
                # ``_level_filter`` only exists in the non-metadata branch;
                # default to "skipped" otherwise so the attribute is always
                # set (Jaeger filters on missing-attr are awkward).
                try:
                    _rag_retrieve_sp.set_attribute("hits", len(raw_hits))
                    _rag_retrieve_sp.set_attribute("latency_ms", _retrieve_ms)
                    _shard_mode = locals().get("_level_filter") or (
                        "skipped" if _intent == "metadata" else "all"
                    )
                    _rag_retrieve_sp.set_attribute("sharding_mode", _shard_mode)
                except Exception:
                    pass
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
            # tagged as kb_primary="unknown" so the counter is never "missing".
            # Review §8.6: labels are (kb_count, kb_primary, path) — kb_count
            # is the request-level selection size (bounded by platform KB
            # count); kb_primary is the hit's own kb_id. Replaces the legacy
            # ``kb`` label that risked becoming a comma-joined cardinality
            # footgun once downstream code joined selection lists.
            try:
                _path = "hybrid" if _hybrid_flag else "dense"
                _kb_count_label = str(len(_kb_ids_for_log))
                for _h in raw_hits:
                    _payload = getattr(_h, "payload", None) or {}
                    _kb = _payload.get("kb_id")
                    if _kb is None:
                        _kb = "chat" if _payload.get("chat_id") is not None else "unknown"
                    retrieval_hits_total.labels(
                        kb_count=_kb_count_label,
                        kb_primary=str(_kb),
                        path=_path,
                    ).inc()
            except Exception as _err:
                _record_silent_failure("hit_counter_emit", _err)

            # 2026-04-29 — Global drill-down (Option A): doc summaries
            # are ~80-token narrative paraphrases ("frequent visits by
            # senior officers including X, Y…") that mention visits
            # exist but lack date-level detail (e.g. "27 Jan: Maj Gen
            # Wajid Aziz, 5 POK Bde, 5-day farewell"). For corpus-wide
            # queries (intent=global) we pull one level=doc summary per
            # doc above; here we fan out to per-doc chunk retrieval and
            # merge so the LLM sees BOTH the per-doc summary AND the
            # granular event chunks. Per-doc cap (RAG_GLOBAL_DRILLDOWN_K)
            # prevents one high-similarity doc from crowding out others.
            # Disabled (default) for non-global intents — no behavior
            # change there. Fail-open: drill-down failure leaves the
            # summary-only stream intact.
            if (
                _intent == "global"
                and raw_hits
                and flags.get("RAG_GLOBAL_DRILLDOWN", "1") == "1"
            ):
                _drilldown_k = int(flags.get("RAG_GLOBAL_DRILLDOWN_K", "2") or "0")
                _doc_ids_for_drilldown = sorted({
                    int(_h.payload["doc_id"])
                    for _h in raw_hits
                    if _h.payload.get("doc_id") is not None
                    and _h.payload.get("level") == "doc"
                })
                if _doc_ids_for_drilldown and _drilldown_k > 0:
                    _drilldown_total = _drilldown_k * len(_doc_ids_for_drilldown)
                    await _emit(progress_cb, {
                        "stage": "drilldown", "status": "running",
                        "n_docs": len(_doc_ids_for_drilldown),
                        "per_doc_k": _drilldown_k,
                    })
                    _tD = time.perf_counter()
                    try:
                        with span(
                            "retrieve.global_drilldown",
                            n_docs=len(_doc_ids_for_drilldown),
                            per_doc_k=_drilldown_k,
                        ), flags.with_overrides({
                            "RAG_HYDE": "0", "RAG_HYDE_N": "0",
                        }):
                            _drilldown_hits = await retrieve(
                                query=_retrieval_query,
                                selected_kbs=selected_kbs,
                                chat_id=chat_id,
                                vector_store=_vector_store,
                                embedder=_embedder,
                                per_kb_limit=_drilldown_total,
                                total_limit=_drilldown_total,
                                owner_user_id=user_id,
                                level_filter="chunk",
                                doc_ids=_doc_ids_for_drilldown,
                                temporal_constraint=_temporal_constraint,
                            )
                        _per_doc_count: dict[int, int] = {}
                        _kept_drilldown: list = []
                        for _h in _drilldown_hits:
                            _did = _h.payload.get("doc_id")
                            if _did is None:
                                continue
                            _did = int(_did)
                            if _per_doc_count.get(_did, 0) >= _drilldown_k:
                                continue
                            _per_doc_count[_did] = _per_doc_count.get(_did, 0) + 1
                            _kept_drilldown.append(_h)
                        raw_hits = list(raw_hits) + _kept_drilldown
                        logger.info(
                            "rag.global_drilldown: docs=%d k=%d retrieved=%d kept=%d total_hits=%d ms=%d",
                            len(_doc_ids_for_drilldown),
                            _drilldown_k,
                            len(_drilldown_hits),
                            len(_kept_drilldown),
                            len(raw_hits),
                            int((time.perf_counter() - _tD) * 1000),
                        )
                        await _emit(progress_cb, {
                            "stage": "drilldown", "status": "done",
                            "ms": int((time.perf_counter() - _tD) * 1000),
                            "added_chunks": len(_kept_drilldown),
                            "total_hits": len(raw_hits),
                        })
                    except Exception as _err:
                        _record_silent_failure("global_drilldown", _err)
                        await _emit(progress_cb, {
                            "stage": "drilldown", "status": "error",
                        })

            # P1.2 — dispatch through rerank_with_flag. Default (RAG_RERANK unset/0)
            # calls ``rerank`` which is byte-identical to the previous behaviour.
            #
            # P2 — MMR candidate-pool widening. When MMR is on, ask the reranker
            # for more candidates than the final budget so MMR actually has room
            # to diversify. When MMR is off, the old top-10 behaviour is preserved
            # byte-identically. An operator may override via ``RAG_RERANK_TOP_K``.
            _final_k = 12
            # For global "summarize all" intent we pull doc-level summaries (one
            # point per document, level=doc); 10 was clipping multi-doc KBs to
            # only 10 of N summaries even though the retrieve already returned
            # all of them. Each summary is ~600 chars so 50 fits comfortably
            # under the context budget. Scales down naturally for tiny KBs
            # (min(50, raw_hits)) and up for medium ones; caps at 50 to keep
            # the prompt tractable. Operator can still override via
            # RAG_GLOBAL_FINAL_K.
            if _intent == "global":
                _global_cap = int(flags.get("RAG_GLOBAL_FINAL_K") or 50)
                _final_k = max(_final_k, min(_global_cap, len(raw_hits) or _final_k))
            # Phase 2.3 — keep at least N×8 final hits for multi-temporal
            # queries so each requested month survives reranker truncation.
            # For 4-month query (months: [2,5,7,12]) this is 32 hits — well
            # under the 5000-token context budget but enough to give the LLM
            # 6-8 evidence chunks per month.
            if _hybrid is not None and _hybrid.source == "llm":
                _tc_for_k = _hybrid.temporal_constraint
                if _tc_for_k:
                    from .retriever import _shard_keys_for_constraint
                    _n_shards_k = len(_shard_keys_for_constraint(_tc_for_k))
                    if _n_shards_k > 1:
                        _final_k = max(_final_k, 8 * _n_shards_k)
            # 2026-05-03 — multi-entity rerank quota bump. When the
            # decompose path was active (>=2 entities), the static
            # _final_k=12 is too tight: the post-rerank trim would cut
            # by global score and silently evict low-frequency entities
            # (smoke test: 75 Inf Bde / 5 PoK Bde returned empty even
            # though Qdrant scrolls confirmed matching chunks). Bump
            # _final_k to n_entities * floor so the per-entity quota
            # (applied in the trim site below) has slots to work with.
            # RAG_MULTI_ENTITY_RERANK_FLOOR=0 disables the bump and the
            # quota; pre-fix behaviour is restored.
            try:
                _entity_floor = int(flags.get("RAG_MULTI_ENTITY_RERANK_FLOOR") or "3")
            except (TypeError, ValueError):
                _entity_floor = 3
            if _do_decompose and _entities and _entity_floor > 0:
                _final_k = max(_final_k, len(_entities) * _entity_floor)
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
                reranked = rerank_with_flag(_retrieval_query, raw_hits, top_k=_rerank_k, fallback_fn=rerank)
            if _rerank_on:
                await _emit(progress_cb, {
                    "stage": "rerank", "status": "done",
                    "ms": int((time.perf_counter() - _tK) * 1000),
                    "top_k": len(reranked),
                })

            # Wave 2 round 4 (review §5.1) — RAG_RERANK_MIN_SCORE post-filter.
            # Default unset = OFF, byte-identical pass-through. When set (e.g.
            # 0.05 after eval-gate), drop hits whose score is strictly below
            # the threshold. Garbage values fail open (no filter applied) so a
            # typo doesn't silently nuke retrieval. Counter labelled by intent
            # so an operator can see which intent class the floor is shaping.
            _min_score_raw = flags.get("RAG_RERANK_MIN_SCORE")
            if _min_score_raw not in (None, "",):
                try:
                    _min_score = float(_min_score_raw)
                except (TypeError, ValueError):
                    _min_score = None
                if _min_score is not None and reranked:
                    _kept_above = [
                        h for h in reranked
                        if float(getattr(h, "score", 0.0) or 0.0) >= _min_score
                    ]
                    _dropped = len(reranked) - len(_kept_above)
                    if _dropped:
                        try:
                            from .metrics import rag_rerank_threshold_dropped_total
                            rag_rerank_threshold_dropped_total.labels(
                                intent=str(_intent)
                            ).inc(_dropped)
                        except Exception:  # pragma: no cover - metrics fail-open
                            pass
                        await _emit(progress_cb, {
                            "stage": "rerank.threshold",
                            "dropped": _dropped,
                            "kept": len(_kept_above),
                        })
                    reranked = _kept_above

            # P1.3 — MMR diversification (flag-gated). Reads the flag at call time
            # so tests can monkeypatch the env without module reload. When the flag
            # is off (default) the ``mmr`` module is never imported. Fails open:
            # any MMR error is swallowed and we keep the reranker output.
            if _mmr_on and reranked:
                await _emit(progress_cb, {"stage": "mmr", "status": "running"})
                try:
                    from .mmr import mmr_rerank_from_hits
                    _mmr_lambda = float(flags.get("RAG_MMR_LAMBDA", "0.7"))
                    # Phase 6.X — cap the MMR re-embed batch so multi-entity
                    # decomposition (which can produce 200+ candidates) doesn't
                    # OOM the GPU shared with TEI / QU / reranker. The
                    # cross-encoder upstream has already ranked candidates;
                    # MMR re-orders the top band only. Default 50; set 0 (or
                    # unset) to restore pre-Phase-6 behaviour (no cap).
                    try:
                        _mmr_cap = int(flags.get("RAG_MMR_MAX_INPUT_SIZE") or "50")
                    except (TypeError, ValueError):
                        _mmr_cap = 50
                    _mmr_cap = None if _mmr_cap <= 0 else _mmr_cap
                    _tM = time.perf_counter()
                    with time_stage("mmr"):
                        reranked = await mmr_rerank_from_hits(
                            _retrieval_query, reranked, _embedder,
                            top_k=_final_k, lambda_=_mmr_lambda,
                            max_input_size=_mmr_cap,
                        )
                    await _emit(progress_cb, {
                        "stage": "mmr", "status": "done",
                        "ms": int((time.perf_counter() - _tM) * 1000),
                        "top_k": len(reranked),
                    })
                except Exception as _err:
                    # Fail open: on any MMR error, stick with reranker output.
                    # B6: still surface via warning + counter so a broken
                    # MMR (bad lambda, embedder issue) is alertable.
                    _record_silent_failure("mmr_rerank", _err)
                    # Phase 6.X (Option B) — when MMR fails AND
                    # ``RAG_MMR_FAIL_TRIM=1``, trim the cross-encoder
                    # output to ``_final_k`` so downstream context_expand
                    # doesn't multiply 50 candidates × 3 siblings each
                    # into a 350K-token blob that the budget evicts down
                    # to 3 hits. The cross-encoder ordering is preserved;
                    # we just take its top _final_k. Default 0 = pre-fix
                    # behaviour (all rerank survivors flow through).
                    if (
                        flags.get("RAG_MMR_FAIL_TRIM", "0") == "1"
                        and len(reranked) > _final_k
                    ):
                        if _do_decompose and _entities and _entity_floor > 0:
                            reranked = _apply_entity_quota(
                                reranked=reranked,
                                entities=list(_entities),
                                per_entity_floor=_entity_floor,
                                final_k=_final_k,
                            )
                            logger.info(
                                "rag: multi-entity rerank quota active "
                                "(mmr fail-trim) — entities=%d floor=%d final_k=%d",
                                len(_entities), _entity_floor, _final_k,
                            )
                            try:
                                from .metrics import (
                                    rag_multi_entity_rerank_quota_total,
                                )
                                rag_multi_entity_rerank_quota_total.labels(
                                    outcome="applied"
                                ).inc()
                            except Exception:
                                pass
                        else:
                            reranked = reranked[:_final_k]
                            logger.info(
                                "rag: mmr fail-trim active, kept top %d of cross-encoder",
                                _final_k,
                            )
                    await _emit(progress_cb, {
                        "stage": "mmr", "status": "error",
                    })
            elif len(reranked) > _final_k:
                # No MMR — trim rerank's extra candidates (e.g. RAG_RERANK_TOP_K
                # set by operator) back to _final_k so downstream budget sees the
                # same count as the pre-P2 pipeline.
                #
                # 2026-05-03 — multi-entity quota fix. When decompose was
                # active, plain top-by-score trim is entity-blind and can
                # silently evict low-frequency entities (75 Inf Bde / 5 PoK
                # Bde smoke test). Use the per-entity quota helper so each
                # entity gets at least RAG_MULTI_ENTITY_RERANK_FLOOR (default
                # 3) chunks in the post-rerank pool. Entity-blind trim is
                # preserved for the non-decompose path.
                if _do_decompose and _entities and _entity_floor > 0:
                    reranked = _apply_entity_quota(
                        reranked=reranked,
                        entities=list(_entities),
                        per_entity_floor=_entity_floor,
                        final_k=_final_k,
                    )
                    logger.info(
                        "rag: multi-entity rerank quota active — "
                        "entities=%d floor=%d final_k=%d",
                        len(_entities), _entity_floor, _final_k,
                    )
                    try:
                        from .metrics import rag_multi_entity_rerank_quota_total
                        rag_multi_entity_rerank_quota_total.labels(
                            outcome="applied"
                        ).inc()
                    except Exception:
                        pass
                else:
                    reranked = reranked[:_final_k]
                    try:
                        from .metrics import rag_multi_entity_rerank_quota_total
                        rag_multi_entity_rerank_quota_total.labels(
                            outcome="skipped"
                        ).inc()
                    except Exception:
                        pass
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
                    # Phase 6.X (Option C) — cap how many top hits get
                    # sibling-expanded. context_expand multiplies each
                    # hit by ~3 (parent + 2 siblings); when the rerank
                    # set is large (e.g. MMR fail keeps 50), the inflated
                    # token count blows the budget and only 3 hits
                    # survive. Cap >0: only the first N hits are sent
                    # to expand_context; the tail is appended unchanged.
                    # Default 0 = unlimited (pre-fix behaviour).
                    try:
                        _expand_cap = int(flags.get("RAG_CONTEXT_EXPAND_MAX_HITS") or "0")
                    except (TypeError, ValueError):
                        _expand_cap = 0
                    with time_stage("expand"):
                        if _expand_cap > 0 and len(reranked) > _expand_cap:
                            _head = reranked[:_expand_cap]
                            _tail = reranked[_expand_cap:]
                            _expanded_head = await expand_context(
                                _head, vs=_vector_store, window=_window,
                            )
                            reranked = list(_expanded_head) + list(_tail)
                            logger.info(
                                "rag: context_expand cap active, expanded top %d of %d",
                                _expand_cap, _before,
                            )
                        else:
                            reranked = await expand_context(
                                reranked, vs=_vector_store, window=_window,
                            )
                    await _emit(progress_cb, {
                        "stage": "expand", "status": "done",
                        "ms": int((time.perf_counter() - _tE) * 1000),
                        "siblings_fetched": max(0, len(reranked) - _before),
                    })
                except Exception as _err:
                    # Fail open: on any error, keep reranker/MMR output unchanged.
                    # B6: log + count so a broken context_expand (Qdrant
                    # scroll error, sibling-fetch failure) is visible.
                    _record_silent_failure("context_expand", _err)
                    await _emit(progress_cb, {
                        "stage": "expand", "status": "error",
                    })
            else:
                await _emit(progress_cb, {
                    "stage": "expand", "status": "skipped", "reason": "flag_off",
                })

            await _emit(progress_cb, {"stage": "budget", "status": "running"})
            _tB = time.perf_counter()
            # B4 — pipeline-level rag.budget span (the inner budget.truncate
            # span lives inside budget_chunks). Tags chunks_in / chunks_kept
            # so a sudden drop in kept-ratio is visible without parsing
            # the inner span tree.
            # 2026-04-29 — Global drill-down (Option A): 12 doc summaries
            # + ~24 drill-down chunks easily exceeds the 10K specific-intent
            # budget. Use a wider cap for global intent only; specific /
            # metadata / specific_date stay at 10K. Caller may override via
            # RAG_BUDGET_TOKENS / RAG_GLOBAL_BUDGET_TOKENS env vars.
            _budget_max = (
                int(flags.get("RAG_GLOBAL_BUDGET_TOKENS", "22000") or "22000")
                if _intent == "global"
                else int(flags.get("RAG_BUDGET_TOKENS", "10000") or "10000")
            )
            # Wave 2 round 4 (review §5.2) — pre-deduct non-chunk prompt parts
            # so the chunk budget no longer overflows the LLM context. Default
            # OFF (env unset / 0) keeps the legacy behaviour. When ON, we
            # estimate reserved_tokens as
            #   system_prompt + catalog_preamble + datetime_preamble + (~30 * len(hits))
            # The catalog/datetime preambles aren't generated yet (they come
            # after this stage), so we use static estimates calibrated against
            # production payload sizes:
            #   * KB Catalog: ~120 tokens header + ~10 tokens per filename
            #     listed; 50-doc cap → ~620 tokens worst-case. Default 800.
            #   * Datetime preamble: ~40 tokens.
            #   * System prompt: tokenize ext.static.system_prompt_analyst.txt
            #     (or its production location) once and cache.
            #   * Spotlight wrap overhead: ~30 tokens per chunk
            #     (open + close + sanitize markers).
            _reserved = 0
            if flags.get("RAG_BUDGET_INCLUDES_PROMPT", "0") == "1":
                try:
                    _reserved = _estimate_reserved_tokens(
                        n_hits=len(reranked),
                        intent=_intent,
                    )
                except Exception as _re:
                    _record_silent_failure("budget.reserve_estimate", _re)
                    _reserved = 0
            with span(
                "rag.budget",
                max_tokens=_budget_max,
                chunks_in=len(reranked),
                reserved_tokens=_reserved,
            ) as _budget_sp, time_stage("budget"):
                # Only pass ``reserved_tokens`` when the flag is on so test
                # stubs that don't accept the kwarg (and the legacy
                # signature) keep working byte-identically.
                if _reserved > 0:
                    budgeted = budget_chunks(
                        reranked,
                        max_tokens=_budget_max,
                        reserved_tokens=_reserved,
                    )
                else:
                    budgeted = budget_chunks(reranked, max_tokens=_budget_max)
                try:
                    _budget_sp.set_attribute("chunks_kept", len(budgeted))
                    _kept_total_tokens = sum(
                        len(str(h.payload.get("text", ""))) // 4
                        for h in budgeted
                    )
                    _budget_sp.set_attribute("total_tokens_est", _kept_total_tokens)
                except Exception:
                    pass
            await _emit(progress_cb, {
                "stage": "budget", "status": "done",
                "ms": int((time.perf_counter() - _tB) * 1000),
                "chunks": len(budgeted),
            })
    except Exception as e:
        logger.exception("KB retrieval failed: %s", e)
        # B6: stage="rag_pipeline" — the outer "any failure inside the
        # retrieve+rerank+budget try-block" catch. Counter ramp here is
        # the loudest signal of a regression.
        _record_silent_failure("rag_pipeline", e)
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
            query=query,
            hits_detail=reranked,
        )
        return []

    # B4 — open the ``rag.context_inject`` span manually (not a ``with``
    # statement) so the existing early-return paths below stay intact.
    # The span is closed in the ``finally`` at function end. ``chunks_in``
    # is the post-budget hit count; ``chunks_in_prompt`` + a rough
    # ``prompt_tokens`` estimate are tagged when the span closes.
    _ctx_inject_span_cm = span(
        "rag.context_inject",
        chunks_in=len(budgeted),
        intent=str(_intent),
    )
    _ctx_inject_sp = _ctx_inject_span_cm.__enter__()
    _ctx_inject_open = True

    def _close_ctx_inject_span(prompt_tokens: int = 0, chunks_in_prompt: int = 0) -> None:
        nonlocal _ctx_inject_open
        if not _ctx_inject_open:
            return
        try:
            _ctx_inject_sp.set_attribute("chunks_in_prompt", int(chunks_in_prompt))
            _ctx_inject_sp.set_attribute("prompt_tokens", int(prompt_tokens))
        except Exception:
            pass
        try:
            _ctx_inject_span_cm.__exit__(None, None, None)
        except Exception:
            pass
        _ctx_inject_open = False

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
    # Wave 2 (review §8.3): record the *real* hit count before preambles are
    # injected — catalog + datetime preambles are always added, so they would
    # mask a genuinely empty retrieval if measured at the return site.
    _real_hits_count = len(sources_out)

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
                    # B9 — only count successfully-ingested documents. A doc
                    # in ``pending`` / ``failed`` status has zero retrievable
                    # chunks; including it in "N documents available" misleads
                    # the LLM (and the user) into believing those files are
                    # already searchable. Live SELECT on every catalog
                    # request — no caching layer between this query and the
                    # database, so the count tracks Postgres truth.
                    res = await _s.execute(
                        _sql_text(
                            "SELECT kb_id, subtag_id, NULL::text AS subtag_name, filename "
                            "FROM kb_documents "
                            "WHERE kb_id = ANY(:ids) "
                            "  AND deleted_at IS NULL "
                            "  AND ingest_status = 'done' "
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
                    # B9 — same ``ingest_status='done'`` filter as the
                    # whole-KB branch above; pending/failed docs would
                    # otherwise pad the subtag-scoped catalog count.
                    _sql = (
                        "SELECT d.kb_id, d.subtag_id, t.name AS subtag_name, d.filename "
                        "FROM kb_documents d "
                        "JOIN kb_subtags t ON t.id = d.subtag_id "
                        "WHERE d.deleted_at IS NULL "
                        "  AND d.ingest_status = 'done' "
                        "  AND (" + " OR ".join(_where_parts) + ") "
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
        # B6: bumped from debug to warning + counter so a broken catalog
        # (e.g. SQL schema drift, missing kb_subtags table) is alertable.
        _record_silent_failure("catalog_render", _e)

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
            # B6: bumped from debug → warning + counter so a broken
            # datetime preamble (zoneinfo missing, formatting error) is
            # alertable in production.
            _record_silent_failure("datetime_preamble", _e)

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
    except Exception as _err:
        # B6: any failure building the hit-summary SSE event is logged so
        # we can spot regressions in the metadata shape.
        _record_silent_failure("hits_emit", _err)

    _total_ms_done = int((time.perf_counter() - _pipeline_start) * 1000)
    await _emit(progress_cb, {
        "stage": "done",
        "total_ms": _total_ms_done,
        "sources": len(sources_out),
    })
    # B4 — close ``rag.context_inject`` with prompt-shape tags. Estimate
    # tokens at 4 chars/token (matches the Plan A budget heuristic) so
    # we never have to call the tokenizer again here.
    _ctx_inject_total_chars = 0
    for _src in sources_out:
        for _doc in _src.get("document") or []:
            try:
                _ctx_inject_total_chars += len(_doc)
            except Exception:
                pass
    _close_ctx_inject_span(
        prompt_tokens=_ctx_inject_total_chars // 4,
        chunks_in_prompt=len(sources_out),
    )
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
        query=query,
        hits_detail=budgeted,
    )
    # Wave 2 (review §8.3): increment empty-retrieval counter when no real
    # hits came back (preambles excluded — see _real_hits_count above).
    if _real_hits_count == 0:
        try:
            from .metrics import rag_retrieval_empty_total
            rag_retrieval_empty_total.labels(
                intent=_intent_label, kb_count=str(len(_kb_ids_for_log)),
            ).inc()
        except Exception as _err:  # pragma: no cover - never break retrieval
            _record_silent_failure("metric_emit", _err)
    return sources_out
