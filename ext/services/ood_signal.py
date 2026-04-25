"""Out-of-domain (OOD) signal computation.

Phase 4 — observability only. This module exposes a single public
function, :func:`compute_ood_score`, which returns cosine similarity
between a query vector and a cached per-KB centroid. A low score
(< 0.3 by convention) indicates the query is semantically far from
anything in the KB — either the KB truly lacks relevant docs, or the
query is malformed / adversarial.

It is NOT wired into the retrieval hot path in this phase (the
execution plan defers routing to Phase 2). Callers who want the signal
invoke this module explicitly and decide what to do with the score.
The intended first use is a WARN-level log from
``chat_rag_bridge._run_pipeline`` once we've observed the real score
distribution on production traffic.

Design notes
------------
* Centroid = mean(unit-normalized chunk vectors for a random sample).
  bge-m3 embeddings come out of TEI already L2-normalized, so this is
  equivalent to the mean direction — the natural prototype for the KB.
* Sample size defaults to 200 random chunks per KB. Large enough that
  the mean converges on bge-m3's dimensionality; small enough that each
  refresh is sub-second even over HTTP Qdrant.
* Cache TTL defaults to 3600 seconds (1 hour). Centroids are stable on
  the timescale of KB drift — no point in recomputing more often.
* Module-level dict keyed by kb_id. Process-local — we accept that each
  worker recomputes once per TTL window. For a 2-worker deployment this
  is ~2× the Qdrant load, which is trivial compared to retrieval.
* Fail-open: any Qdrant / math error returns 1.0 (in-domain by default),
  never raises. The rationale: if we can't compute the signal, we must
  not drop the query — default to letting retrieval run and the
  downstream budget / rerank filter out the noise.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from typing import Any, Optional, Sequence

logger = logging.getLogger("orgchat.ood_signal")


# Cache state. Keyed by kb_id (int). Value = (timestamp, centroid-list).
# Module-level intentionally — the cache is process-local and lives as
# long as the worker does. Hot-reload during tests clears it via the
# ``clear_cache`` hook.
_CENTROID_CACHE: dict[int, tuple[float, list[float]]] = {}

# Serialize recomputes per kb so concurrent callers don't all hit Qdrant
# at once during a cache miss.
_REFRESH_LOCKS: dict[int, asyncio.Lock] = {}


def _ttl_seconds() -> int:
    """Cache TTL from env (``RAG_OOD_CENTROID_TTL``), default 3600s."""
    try:
        return max(60, int(os.environ.get("RAG_OOD_CENTROID_TTL", "3600")))
    except (TypeError, ValueError):
        return 3600


def _sample_size() -> int:
    """Centroid sample size from env (``RAG_OOD_SAMPLE_SIZE``), default 200."""
    try:
        return max(1, int(os.environ.get("RAG_OOD_SAMPLE_SIZE", "200")))
    except (TypeError, ValueError):
        return 200


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 1.0 when either vector is zero-length (degenerate — treat
    as in-domain by default rather than raise). Must not depend on
    numpy — this module is a leaf node and we don't want to force a
    numpy import on every retrieval.
    """
    if not a or not b:
        return 1.0
    if len(a) != len(b):
        # Shape mismatch — likely centroid computed with a different
        # embedder. Don't raise; treat as in-domain.
        return 1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0:
        return 1.0
    return dot / denom


def _normalize(vec: Sequence[float]) -> list[float]:
    """Return a unit-length copy of ``vec``. Identity-ish when already normalized."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return list(vec)
    return [x / norm for x in vec]


def clear_cache() -> None:
    """Test hook — wipe the module-level centroid cache."""
    _CENTROID_CACHE.clear()
    _REFRESH_LOCKS.clear()


async def _fetch_centroid_from_qdrant(
    kb_id: int,
    *,
    qdrant_client: Any,
    sample_size: int,
) -> Optional[list[float]]:
    """Scroll a random sample of points from ``kb_{kb_id}`` and average them.

    ``qdrant_client`` is any object exposing ``scroll(collection_name,
    limit, offset, with_payload, with_vectors)`` compatible with
    AsyncQdrantClient. Returns None if the collection is empty or
    unreachable.
    """
    collection = f"kb_{kb_id}"
    vectors: list[list[float]] = []
    # Scroll a small-ish number of pages, stopping when we have enough
    # points. The sample is "first N random-ish points" — Qdrant's scroll
    # order is arbitrary-but-deterministic within a collection; that's
    # good enough for centroid estimation.
    PAGE = 128
    offset = None
    pulled = 0
    HARD_CAP = max(sample_size * 2, PAGE * 4)  # don't chew all day on huge KBs
    try:
        while pulled < sample_size and pulled < HARD_CAP:
            page, offset = await qdrant_client.scroll(
                collection_name=collection,
                limit=PAGE,
                offset=offset,
                with_payload=False,
                with_vectors=True,
            )
            if not page:
                break
            for pt in page:
                vec = _extract_dense_vector(pt)
                if vec is not None:
                    vectors.append(vec)
                pulled += 1
                if len(vectors) >= sample_size:
                    break
            if offset is None:
                break
    except Exception as e:
        logger.info("ood_signal: qdrant scroll failed for %s: %s", collection, e)
        return None

    if not vectors:
        return None

    # If we pulled more than sample_size, subsample uniformly at random
    # so centroid quality is the same regardless of scroll ordering.
    if len(vectors) > sample_size:
        vectors = random.sample(vectors, sample_size)

    # bge-m3 comes out normalized but re-normalize defensively —
    # scripts/normalize_doc_ids or future embedder swaps may have
    # left un-normalized rows.
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue
        nv = _normalize(v)
        for i in range(dim):
            acc[i] += nv[i]
    # Mean then normalize — equivalent to (sum then normalize) for
    # cosine purposes; keep mean for numerical stability.
    n = float(len(vectors))
    mean = [x / n for x in acc]
    return _normalize(mean)


def _extract_dense_vector(point: Any) -> Optional[list[float]]:
    """Pull the dense vector off a scrolled point.

    Collections with sparse support store vectors as a dict keyed by
    name (``{"dense": [...], "bm25": SparseVector(...)}``). Legacy
    collections store them as a plain list. Returns None if we can't
    find a dense list.
    """
    v = getattr(point, "vector", None)
    if v is None:
        return None
    if isinstance(v, dict):
        d = v.get("dense")
        if isinstance(d, list):
            return [float(x) for x in d]
        # Take first list value defensively.
        for val in v.values():
            if isinstance(val, list):
                return [float(x) for x in val]
        return None
    if isinstance(v, list):
        return [float(x) for x in v]
    return None


async def _get_centroid(
    kb_id: int,
    *,
    qdrant_client: Any,
) -> Optional[list[float]]:
    """Return a cached centroid for ``kb_id`` or compute + cache it.

    Empty / unreachable KB → returns None. Caller treats None as
    "centroid unavailable; default in-domain".
    """
    now = time.time()
    ttl = _ttl_seconds()
    cached = _CENTROID_CACHE.get(kb_id)
    if cached is not None and (now - cached[0]) < ttl:
        return cached[1]

    # One lock per kb so concurrent cache misses for different KBs don't
    # serialize.
    lock = _REFRESH_LOCKS.setdefault(kb_id, asyncio.Lock())
    async with lock:
        # Double-check under the lock — another waiter may have filled
        # the cache while we were queued.
        cached = _CENTROID_CACHE.get(kb_id)
        if cached is not None and (time.time() - cached[0]) < ttl:
            return cached[1]
        centroid = await _fetch_centroid_from_qdrant(
            kb_id,
            qdrant_client=qdrant_client,
            sample_size=_sample_size(),
        )
        if centroid is not None:
            _CENTROID_CACHE[kb_id] = (time.time(), centroid)
        return centroid


async def compute_ood_score(
    query_vec: Sequence[float],
    kb_id: int,
    *,
    qdrant_client: Any = None,
) -> float:
    """Return cosine(query_vec, kb_centroid) in [-1, 1] — higher is more in-domain.

    ``qdrant_client`` is an AsyncQdrantClient-compatible object. If None,
    we lazily import and build one against ``QDRANT_URL`` — but callers
    inside the chat hot path should pass their existing client (via
    ``VectorStore._client``) to avoid opening a new HTTP pool per call.

    Convention:
      * score ≥ 0.5 — clearly in-domain.
      * 0.3 ≤ score < 0.5 — borderline; callers may log but not block.
      * score < 0.3 — likely OOD; callers should log a WARN. This module
        does NOT log for the caller — it just returns the score — so
        the retrieval pipeline can batch the WARN with the rest of its
        request_id context.

    Fail-open: any Qdrant / math error returns 1.0 (default in-domain).
    """
    if not query_vec:
        return 1.0

    client = qdrant_client
    must_close = False
    if client is None:
        try:
            from qdrant_client import AsyncQdrantClient
            qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
            client = AsyncQdrantClient(url=qdrant_url, timeout=10.0)
            must_close = True
        except Exception as e:
            logger.info("ood_signal: cannot build qdrant client: %s", e)
            return 1.0

    try:
        centroid = await _get_centroid(int(kb_id), qdrant_client=client)
    except Exception as e:
        logger.info("ood_signal: centroid lookup failed for kb %s: %s", kb_id, e)
        return 1.0
    finally:
        if must_close:
            try:
                await client.close()
            except Exception:
                pass

    if centroid is None:
        # Empty KB (or unreachable). Treat as in-domain by convention so
        # a fresh KB never blocks retrieval.
        return 1.0

    try:
        return _cosine(list(query_vec), centroid)
    except Exception as e:
        logger.info("ood_signal: cosine failed for kb %s: %s", kb_id, e)
        return 1.0


__all__ = ["compute_ood_score", "clear_cache"]
