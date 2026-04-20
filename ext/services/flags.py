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
from typing import Iterator, Mapping, Optional

# The contextvar holds the current overlay dict (or None when no overlay
# is active). A fresh dict object is stored per ``with_overrides`` entry
# so nested calls can compose without aliasing.
_OVERLAY: contextvars.ContextVar[Optional[dict[str, str]]] = contextvars.ContextVar(
    "rag_flag_overlay", default=None,
)


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return the overlay value for ``key`` if one is active, else ``os.environ.get``.

    Drop-in replacement for ``os.environ.get(key, default)`` — all callers
    in the RAG hot path should use this instead so that per-request KB
    config overrides (from ``chat_rag_bridge``) take effect.
    """
    overlay = _OVERLAY.get()
    if overlay is not None and key in overlay:
        return overlay[key]
    return os.environ.get(key, default)


@contextlib.contextmanager
def with_overrides(overrides: Mapping[str, str]) -> Iterator[None]:
    """Temporarily overlay ``overrides`` on top of ``os.environ`` reads.

    Scoped to the current ``contextvars.Context`` — concurrent asyncio
    tasks running outside the ``with`` block will not see these values.
    Nested calls compose: the inner overlay extends/overrides the outer.

    Passing an empty mapping is a no-op (the current overlay, if any, is
    preserved unchanged).
    """
    if not overrides:
        yield
        return

    current = _OVERLAY.get()
    # Build the merged overlay as a fresh dict so the outer scope's dict
    # is not mutated. Inner keys win over outer.
    merged: dict[str, str] = dict(current) if current is not None else {}
    for k, v in overrides.items():
        merged[str(k)] = str(v)

    token = _OVERLAY.set(merged)
    try:
        yield
    finally:
        _OVERLAY.reset(token)


def _peek_overlay_for_tests() -> Optional[dict[str, str]]:
    """Return a copy of the current overlay (for assertions in unit tests)."""
    o = _OVERLAY.get()
    return dict(o) if o is not None else None


__all__ = ["get", "with_overrides"]
