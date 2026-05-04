"""Context-variable overlay on ``os.environ`` for per-request flag overrides.

The RAG hot path (retriever, reranker, MMR, context-expand, spotlight,
semcache) reads feature flags via ``RAG_*`` env vars. Those reads are
process-level and cannot be safely mutated per request — multiple chats
run concurrently in the same process and ``os.environ`` is shared state.

This module introduces ``flags.get(key, default)`` as a drop-in replacement
for ``os.environ.get(key, default)`` on the hot path. Callers that want to
temporarily override a flag for the duration of a single request wrap the
work in ``with_overrides({...})`` — the overlay is stored in a
``contextvars.ContextVar`` so it is safely scoped to the current task/
coroutine and does NOT leak to concurrent tasks.

Why contextvars (not threadlocals):
  * ``contextvars`` is the only primitive that propagates correctly across
    ``await`` boundaries in asyncio — a threadlocal set in one coroutine
    leaks into every other coroutine that shares the event-loop thread.
  * FastAPI's starlette request scope is contextvar-backed for the same
    reason, and we want compatible semantics (each request sees its own
    overrides, even when multiple requests are processed concurrently).
  * Python 3.7+ guarantees ``asyncio.gather`` and ``asyncio.create_task``
    copy the current ``Context`` into the child task at spawn time, so
    nested overrides work correctly in the fan-out pattern the retriever
    uses.

The overlay is ONLY read — it never mutates ``os.environ``. When a caller
exits the ``with_overrides`` block, the overlay is restored to whatever
was in effect before (nested overrides compose naturally via a stack-like
chain of contextvar tokens).

Out of scope (intentional — these stay on ``os.environ``):
  * Process-level infra flags: ``RAG_REDIS_URL``, ``RAG_RERANK_CACHE_TTL``,
    ``RAG_SYNC_INGEST``, ``RAG_CONTEXTUALIZE_KBS``. Admins configure these
    once at deploy time; per-request overrides make no sense.
  * Any non-RAG env var (``AUTH_MODE``, ``DATABASE_URL``, etc.).

Usage::

    from ext.services import flags

    # Hot path — drop-in replacement for os.environ.get:
    if flags.get("RAG_RERANK", "0") == "1":
        ...

    # Bridge — wrap a single request in per-KB overrides:
    with flags.with_overrides({"RAG_RERANK": "1", "RAG_MMR": "1"}):
        await retrieve_kb_sources(...)
"""
from __future__ import annotations

import contextlib
import contextvars
import os
from typing import Any, Iterator, Mapping, Optional

# The contextvar holds the current overlay dict (or None when no overlay
# is active). A fresh dict object is stored per ``with_overrides`` entry
# so nested calls can compose without aliasing.
#
# 2026-05-04 — Phase 3 / item 5: relaxed value type from ``str`` to ``Any``
# so per-KB ``subtopic_keywords`` (dict) and ``synonyms`` (list) can ride
# the same overlay as the string-valued RAG_* env-var overrides. Existing
# call sites pass strings and continue to work; the new helpers
# ``get_dict`` / ``get_list`` validate the type before returning so a
# stringified value never masquerades as the structured type.
_OVERLAY: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "rag_flag_overlay", default=None,
)


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return the overlay value for ``key`` if one is active, else ``os.environ.get``.

    Drop-in replacement for ``os.environ.get(key, default)`` — all callers
    in the RAG hot path should use this instead so that per-request KB
    config overrides (from ``chat_rag_bridge``) take effect.

    2026-05-04 — only string-typed overlay values are returned by this
    helper. Dict/list overlay values (used by Phase 3 ``subtopic_keywords``
    / ``synonyms``) fall through to ``os.environ.get`` so a code path that
    expects a string flag never receives a structured value.
    """
    overlay = _OVERLAY.get()
    if overlay is not None and key in overlay:
        val = overlay[key]
        if isinstance(val, str):
            return val
    return os.environ.get(key, default)


def get_dict(key: str, default: dict | None = None) -> dict | None:
    """Return overlay value for ``key`` if it's a dict; else ``default``.

    2026-05-04 — Phase 3 / item 5 of multi-entity-elaborate-answers spec.
    Used by chat_rag_bridge to read the per-KB ``subtopic_keywords`` table
    out of the rag_config overlay set up in ``_retrieve_overrides``.

    Unlike ``get`` this never falls back to ``os.environ`` (env vars are
    always strings) — the overlay is the only source of dict-typed values.
    """
    overlay = _OVERLAY.get()
    if overlay is None:
        return default
    val = overlay.get(key)
    if isinstance(val, dict):
        return val
    return default


def get_list(key: str, default: list | None = None) -> list | None:
    """Return overlay value for ``key`` if it's a list; else ``default``.

    2026-05-04 — Phase 3 / item 5 of multi-entity-elaborate-answers spec.
    Used by chat_rag_bridge to read the per-KB ``synonyms`` table out of
    the rag_config overlay.

    Unlike ``get`` this never falls back to ``os.environ`` (env vars are
    always strings) — the overlay is the only source of list-typed values.
    """
    overlay = _OVERLAY.get()
    if overlay is None:
        return default
    val = overlay.get(key)
    if isinstance(val, list):
        return val
    return default


@contextlib.contextmanager
def with_overrides(overrides: Mapping[str, Any]) -> Iterator[None]:
    """Temporarily overlay ``overrides`` on top of ``os.environ`` reads.

    Scoped to the current ``contextvars.Context`` — concurrent asyncio
    tasks running outside the ``with`` block will not see these values.
    Nested calls compose: the inner overlay extends/overrides the outer.

    Passing an empty mapping is a no-op (the current overlay, if any, is
    preserved unchanged).

    2026-05-04 — Phase 3 / item 5. Dict and list values are preserved
    as-is (no ``str(...)`` coercion) so ``get_dict`` / ``get_list`` can
    read structured per-KB config (``subtopic_keywords``, ``synonyms``)
    from the same overlay that carries the string RAG_* values. Every
    other value type continues to be stringified — preserving the
    Mapping[str, str] read-side contract that ``flags.get`` enforces.
    """
    if not overrides:
        yield
        return

    current = _OVERLAY.get()
    # Build the merged overlay as a fresh dict so the outer scope's dict
    # is not mutated. Inner keys win over outer.
    merged: dict[str, Any] = dict(current) if current is not None else {}
    for k, v in overrides.items():
        if isinstance(v, (dict, list)):
            merged[str(k)] = v
        else:
            merged[str(k)] = str(v)

    token = _OVERLAY.set(merged)
    try:
        yield
    finally:
        _OVERLAY.reset(token)


def _peek_overlay_for_tests() -> Optional[dict[str, Any]]:
    """Return a copy of the current overlay (for assertions in unit tests)."""
    o = _OVERLAY.get()
    return dict(o) if o is not None else None


__all__ = ["get", "get_dict", "get_list", "with_overrides"]
