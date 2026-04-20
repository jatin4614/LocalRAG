"""Rerank hits from parallel KB searches.

Strategy:
- Fast path: if raw top-1 score / top-2 score > FAST_PATH_RATIO, return input unchanged.
- Otherwise: per-KB max-normalize scores, then re-sort global list descending.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, List, Sequence

from . import flags
from .vector_store import Hit


FAST_PATH_RATIO = 2.0


def rerank(hits: List[Hit], *, top_k: int = 10) -> List[Hit]:
    if not hits:
        return []

    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    if len(ordered) >= 2 and ordered[1].score > 0 and (ordered[0].score / ordered[1].score) > FAST_PATH_RATIO:
        return ordered[:top_k]

    def _kb_key(h: Hit) -> Any:
        v = h.payload.get("kb_id", -1)
        try:
            return int(v)
        except (TypeError, ValueError):
            return v  # non-numeric (e.g., "eval") — group by raw value

    max_by_kb: dict[Any, float] = defaultdict(float)
    for h in hits:
        kb = _kb_key(h)
        if h.score > max_by_kb[kb]:
            max_by_kb[kb] = h.score

    def normalized(h: Hit) -> float:
        kb = _kb_key(h)
        m = max_by_kb[kb]
        return h.score / m if m > 0 else h.score

    # Sort by: normalized score desc, then raw score desc, then id (stable tiebreaker for determinism in tests)
    ordered2 = sorted(hits, key=lambda h: (-normalized(h), -h.score, str(h.id)))
    return ordered2[:top_k]


def _read_rerank_flag() -> bool:
    """Read RAG_RERANK at call time so tests can monkeypatch env without reload.

    Uses ``flags.get`` so per-KB config overrides (P3.0) take effect for
    the current request without mutating the shared process env.
    """
    return flags.get("RAG_RERANK", "0") == "1"


def rerank_with_flag(
    query: str,
    hits: Sequence[Any],
    *,
    top_k: int = 10,
    fallback_fn: Callable[..., list[Any]] | None = None,
) -> list[Any]:
    """Dispatch to the cross-encoder reranker if ``RAG_RERANK=1``, else ``fallback_fn``.

    Default behaviour (``RAG_RERANK`` unset or ``0``) is byte-identical to the
    legacy ``rerank(hits, top_k=...)`` path — the cross-encoder module is not
    imported at all.

    Fail-open: if the cross-encoder import or inference raises for any reason
    (missing ``sentence-transformers``, model download failure, etc.) we log
    nothing and silently fall back to the legacy reranker. This keeps
    retrieval working even when the optional dependency is unavailable.
    """
    fn = fallback_fn or rerank
    if not _read_rerank_flag():
        return fn(hits, top_k=top_k)
    try:
        from ext.services.cross_encoder_reranker import rerank_cross_encoder
        return rerank_cross_encoder(query, hits, top_k=top_k)
    except Exception:
        # Fail open: model load or inference failure → legacy path still serves.
        return fn(hits, top_k=top_k)
