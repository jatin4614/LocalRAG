"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Flag-gated by RAG_RERANK. Lazy-loads the model on first call.
CPU-friendly: batches the (query, passage) pairs.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any, Sequence

_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import CrossEncoder
    model_name = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    max_len = int(os.environ.get("RAG_RERANK_MAX_LEN", "512"))
    return CrossEncoder(model_name, max_length=max_len)


class CrossEncoderUnavailable(RuntimeError):
    """Raised when the sentence-transformers / cross-encoder model cannot be loaded."""


def score_pairs(query: str, passages: Sequence[str], *, batch_size: int = 8) -> list[float]:
    """Score each (query, passage) pair. Returns raw relevance scores (higher = better).

    Thread-safe singleton model load under a module lock.
    """
    if not passages:
        return []
    try:
        with _LOCK:
            model = _load_model()
    except ImportError as e:
        raise CrossEncoderUnavailable(
            "sentence-transformers not installed — pip install '.[rerank]' "
            "or pip install sentence-transformers"
        ) from e
    pairs = [(query, p or "") for p in passages]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(s) for s in scores]


def rerank_cross_encoder(
    query: str,
    hits: Sequence[Any],
    *,
    top_k: int = 10,
    batch_size: int = 8,
) -> list[Any]:
    """Rerank hits by cross-encoder relevance to ``query``. Preserves hit objects.

    ``hits`` must expose a ``.payload`` dict with a ``text`` key, OR a ``.text`` attr.
    Returns the top-k hits sorted by cross-encoder score descending.
    """
    if not hits:
        return []
    passages: list[str] = []
    for h in hits:
        payload = getattr(h, "payload", None) or {}
        text = payload.get("text") or getattr(h, "text", "") or ""
        passages.append(str(text))
    scores = score_pairs(query, passages, batch_size=batch_size)
    scored = sorted(zip(scores, hits), key=lambda t: t[0], reverse=True)
    return [h for _, h in scored[:top_k]]
