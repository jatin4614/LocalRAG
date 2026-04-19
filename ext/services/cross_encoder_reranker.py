"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Flag-gated by RAG_RERANK. Lazy-loads the model on first call.

GPU auto-select: when ``RAG_RERANK_DEVICE=auto`` (the default) and
``torch.cuda.is_available()`` is True, the model is pinned to ``cuda:0``.
Override with ``RAG_RERANK_DEVICE=cpu`` or ``RAG_RERANK_DEVICE=cuda:N``.

Score cache: (query, passage) scores are cached in Redis via
``ext.services.rerank_cache`` so repeated queries skip model inference.
Cache is fail-open — any Redis error downgrades to a cache miss and the
reranker still serves correct results from model inference.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any, Sequence

_LOCK = threading.Lock()


def _resolve_device() -> str:
    """Return the torch device string to use for the cross-encoder.

    env ``RAG_RERANK_DEVICE``:
        ``auto`` (default) — pick ``cuda:0`` if available, else ``cpu``
        ``cpu``            — force CPU
        ``cuda``/``cuda:N``— force a specific CUDA device
    """
    pref = os.environ.get("RAG_RERANK_DEVICE", "auto").lower()
    if pref == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"
    return pref


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import CrossEncoder

    model_name = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    max_len = int(os.environ.get("RAG_RERANK_MAX_LEN", "512"))
    device = _resolve_device()
    # sentence-transformers 5.x accepts ``device`` as a keyword; it is
    # forwarded to BaseModel.__init__ and the module is ``.to(device)``-ed.
    return CrossEncoder(model_name, max_length=max_len, device=device)


class CrossEncoderUnavailable(RuntimeError):
    """Raised when the sentence-transformers / cross-encoder model cannot be loaded."""


def _default_batch_size(model) -> int:
    """Return a sensible default batch size based on where the model lives.

    GPU → 32, CPU → 8. ``RAG_RERANK_BATCH_SIZE`` overrides.
    """
    env = os.environ.get("RAG_RERANK_BATCH_SIZE")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    # Prefer ``.device`` (canonical in ST 5.x); fall back to the deprecated
    # ``_target_device`` attribute for older versions.
    device = getattr(model, "device", None) or getattr(model, "_target_device", None)
    on_gpu = "cuda" in str(device).lower() if device is not None else False
    return 32 if on_gpu else 8


def score_pairs(query: str, passages: Sequence[str], *, batch_size: int | None = None) -> list[float]:
    """Score each (query, passage) pair. Returns raw relevance scores (higher = better).

    Thread-safe singleton model load under a module lock.

    When enabled, the Redis-backed score cache (``rerank_cache``) is consulted
    first for each pair; misses are scored by the model in a single batched
    call and then persisted to Redis with TTL. Cache failures are invisible
    (fail-open).
    """
    if not passages:
        return []

    # ------------------------------------------------------------------
    # Phase 1 — probe the cache for pre-computed scores.
    # ------------------------------------------------------------------
    from ext.services.rerank_cache import get_many, is_enabled, put_many

    cache_on = is_enabled()
    model_name = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    cached_scores: dict[int, float] = {}
    missing_indices: list[int] = []
    missing_passages: list[str] = []
    if cache_on:
        keys = [(query, p or "") for p in passages]
        found = get_many(model_name, keys)  # list[Optional[float]]
        for i, v in enumerate(found):
            if v is not None:
                cached_scores[i] = v
            else:
                missing_indices.append(i)
                missing_passages.append(passages[i] or "")
    else:
        missing_indices = list(range(len(passages)))
        missing_passages = [p or "" for p in passages]

    # ------------------------------------------------------------------
    # Phase 2 — score whatever the cache didn't already have.
    # ------------------------------------------------------------------
    infer_scores: list[float] = []
    if missing_passages:
        try:
            with _LOCK:
                model = _load_model()
        except ImportError as e:
            raise CrossEncoderUnavailable(
                "sentence-transformers not installed — pip install '.[rerank]' "
                "or pip install sentence-transformers"
            ) from e

        bs = batch_size if batch_size is not None else _default_batch_size(model)
        pairs = [(query, p) for p in missing_passages]
        raw = model.predict(pairs, batch_size=bs, show_progress_bar=False)
        infer_scores = [float(s) for s in raw]

        if cache_on:
            put_many(
                model_name,
                [(query, p, s) for p, s in zip(missing_passages, infer_scores)],
            )

    # ------------------------------------------------------------------
    # Phase 3 — stitch cached + freshly-inferred scores back into order.
    # ------------------------------------------------------------------
    out: list[float] = [0.0] * len(passages)
    for i, v in cached_scores.items():
        out[i] = v
    for idx, s in zip(missing_indices, infer_scores):
        out[idx] = s
    return out


def rerank_cross_encoder(
    query: str,
    hits: Sequence[Any],
    *,
    top_k: int = 10,
    batch_size: int | None = None,
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
