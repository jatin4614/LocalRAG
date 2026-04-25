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

import logging
import os
import threading
import time
from typing import Any, Sequence

from .obs import span

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# Phase 1.2 — retryable singleton state. The original implementation used
# ``@lru_cache(maxsize=1)`` which had a critical bug: any exception raised
# during model load was re-raised on every subsequent call (lru_cache caches
# the exception object, so the singleton was poisoned forever once a load
# failed — even after the underlying issue resolved). Now we track the loaded
# instance under a lock and only cache successes; failures retry from scratch
# on the next call.
_MODEL_LOCK = threading.Lock()
_MODEL_INSTANCE = None  # type: ignore[var-annotated]


def _reset_model_for_test() -> None:
    """Clear the cached model. Test-only helper."""
    global _MODEL_INSTANCE
    with _MODEL_LOCK:
        _MODEL_INSTANCE = None


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


def get_model():
    """Return the cross-encoder model, loading it if necessary.

    Thread-safe singleton. On transient failure (network blip during HF
    download, OOM on first CUDA init, sentence-transformers ImportError on
    a lazy dep), retries with exponential backoff. On permanent failure
    (max retries exhausted) raises; the NEXT call retries from scratch —
    failures are NOT cached. Regression guard against the original
    @lru_cache behavior which poisoned the singleton forever.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE
    with _MODEL_LOCK:
        if _MODEL_INSTANCE is not None:
            return _MODEL_INSTANCE
        from sentence_transformers import CrossEncoder
        model_name = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        max_len = int(os.environ.get("RAG_RERANK_MAX_LEN", "512"))
        device = _resolve_device()
        retries = int(os.environ.get("RAG_RERANK_LOAD_RETRIES", "3"))
        base_sec = float(os.environ.get("RAG_RERANK_LOAD_RETRY_BASE_SEC", "1.0"))
        last_exc: BaseException | None = None
        for attempt in range(retries):
            try:
                logger.info(
                    "reranker load attempt %d/%d: %s on %s",
                    attempt + 1, retries, model_name, device,
                )
                _MODEL_INSTANCE = CrossEncoder(model_name, max_length=max_len, device=device)
                logger.info("reranker loaded")
                return _MODEL_INSTANCE
            except Exception as exc:
                last_exc = exc
                wait = base_sec * (2 ** attempt)
                logger.warning(
                    "reranker load attempt %d/%d failed (%s: %s); sleeping %.1fs",
                    attempt + 1, retries, type(exc).__name__, exc, wait,
                )
                if attempt < retries - 1:
                    time.sleep(wait)
        assert last_exc is not None
        raise last_exc


def _load_model():
    """Backward-compat shim — existing call sites use ``_load_model()``.

    Kept as a thin wrapper over :func:`get_model` so older code paths and
    monkeypatch-based tests don't break. New code should call ``get_model``
    directly.
    """
    return get_model()


# Backward-compat: the old ``@lru_cache`` ``_load_model`` exposed
# ``cache_clear()`` for tests to reset between cases. Forward that to
# the new singleton-reset helper so existing tests continue to work.
_load_model.cache_clear = _reset_model_for_test  # type: ignore[attr-defined]


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
        _hits = 0
        _misses = 0
        for i, v in enumerate(found):
            if v is not None:
                cached_scores[i] = v
                _hits += 1
            else:
                missing_indices.append(i)
                missing_passages.append(passages[i] or "")
                _misses += 1
        # Batched counter increment — metrics call is best-effort.
        try:
            from ext.services.metrics import rerank_cache_total

            if _hits:
                rerank_cache_total.labels(outcome="hit").inc(_hits)
            if _misses:
                rerank_cache_total.labels(outcome="miss").inc(_misses)
        except Exception:
            pass
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
    model_name = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    with span(
        "rerank.score",
        n_candidates=len(hits),
        model=model_name,
        top_k=top_k,
    ):
        passages: list[str] = []
        for h in hits:
            payload = getattr(h, "payload", None) or {}
            text = payload.get("text") or getattr(h, "text", "") or ""
            passages.append(str(text))
        scores = score_pairs(query, passages, batch_size=batch_size)
        scored = sorted(zip(scores, hits), key=lambda t: t[0], reverse=True)
        return [h for _, h in scored[:top_k]]
