"""Maximal Marginal Relevance (MMR) diversification.

After reranking, MMR post-processes the top-k hits to maximise both
relevance (cross-encoder or vector similarity against the query) AND
diversity (pairwise vector distance between selected passages). Gated
by ``RAG_MMR``; default OFF so behaviour is byte-identical when unused.

Embeddings come from re-embedding hit texts against the same TEI client
used for dense retrieval; ~50-200 ms extra latency for top-30 input. The
dense pipeline's embedder already exposes ``embed(list[str]) -> list[list[float]]``
(batch by default) so a single round-trip gets us every passage vector.

Textbook algorithm::

    S = []
    while |S| < k:
        argmax_{d in C - S} [ lambda * sim(q, d) - (1 - lambda) * max_{s in S} sim(d, s) ]
        add to S
    return S

* lambda = 1.0 -> pure relevance (no-op vs. input ordering restricted to
  the top_k prefix)
* lambda = 0.0 -> pure diversity (pick most-relevant first, then always
  the most-dissimilar-from-selected)
* lambda = 0.7 is the default (70% relevance, 30% diversity)

Performance (review §5.10)
--------------------------
TEI's BAAI/bge-m3 returns L2-normalized vectors, so a plain dot product on
unit vectors equals cosine similarity. The hot loop in ``mmr_rerank`` now
pre-normalizes once at entry and uses ``np.dot`` per pair instead of
recomputing ``np.linalg.norm`` on every call. The legacy ``_cosine`` helper
is preserved for callers that import it directly; it is no longer used by
the MMR loop.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

from .obs import span

_log = logging.getLogger(__name__)

# Tolerance for the "vectors should already be unit-norm" sanity check.
# TEI's bge-m3 returns vectors with ||v|| within ~1e-6 of 1.0 in practice;
# we widen to 1e-3 to cover float32 round-trip and the rare op-tag drift.
_UNIT_NORM_TOL = 1e-3


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length float sequences.

    Kept for backward compatibility with callers that may import it directly.
    The MMR hot loop now uses ``np.dot`` on pre-normalized vectors instead;
    see ``mmr_rerank`` for details.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` divided by its L2 norm along the last axis.

    Zero-norm rows are left as zeros (no division-by-zero); their cosine
    similarity to anything will then be 0, matching the legacy ``_cosine``
    behaviour for zero-vector inputs.
    """
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    # Avoid divide-by-zero: clamp norms below tolerance to 1.0 so the row
    # stays at all-zeros (since the dividend is zero).
    safe = np.where(norms > 0.0, norms, 1.0)
    return arr / safe


def mmr_rerank(
    query_vec: Sequence[float],
    passage_vecs: Sequence[Sequence[float]],
    hits: Sequence[Any],
    *,
    top_k: int = 10,
    lambda_: float = 0.7,
) -> list[Any]:
    """Return the top_k hits reordered by MMR over (query_vec, passage_vecs).

    ``hits[i]`` must correspond to ``passage_vecs[i]``. Stable ordering: if
    two candidates tie on MMR score, the earlier input-rank wins (preserves
    the reranker's ordering on ties).

    Vectors are L2-normalized once at entry; the inner loop then uses
    ``np.dot`` (cosine = dot for unit vectors). Behaviour is unchanged
    within float tolerance vs the legacy ``_cosine`` impl.
    """
    if not hits:
        return []
    n = len(hits)
    if top_k >= n:
        # MMR is only meaningful when we are selecting a strict subset. When
        # top_k >= n we have nothing to diversify; return in input order so
        # callers observe a byte-identical pass-through.
        return list(hits)

    # One-shot L2 normalize. After this every np.dot equals cosine similarity.
    q = _l2_normalize(np.asarray(query_vec, dtype=np.float64))
    p = _l2_normalize(np.asarray(passage_vecs, dtype=np.float64))

    relevance = (p @ q).tolist()  # shape (n,)

    # Step 1: seed selection with the most relevant. Tie-break by lower index.
    remaining = set(range(n))
    first = max(remaining, key=lambda i: (relevance[i], -i))
    selected: list[int] = [first]
    remaining.remove(first)

    # Step 2: iteratively pick the best MMR candidate among the remainder.
    while len(selected) < top_k and remaining:
        best_i = -1
        best_score = -float("inf")
        # Pre-stack the selected vectors so one matmul gives all sims.
        sel_mat = p[selected]
        for i in sorted(remaining):
            # cosine(p[i], p[s]) for each s in selected, then take the max.
            sims = sel_mat @ p[i]
            max_sim_to_selected = float(sims.max())
            score = lambda_ * relevance[i] - (1.0 - lambda_) * max_sim_to_selected
            # Tie-break: strict > keeps the earliest-indexed winner because we
            # iterate ``sorted(remaining)``. No explicit tie-break branch needed.
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    return [hits[i] for i in selected]


async def mmr_rerank_from_hits(
    query: str,
    hits: Sequence[Any],
    embedder: Any,
    *,
    top_k: int = 10,
    lambda_: float = 0.7,
    max_input_size: int | None = None,
) -> list[Any]:
    """High-level helper: embed query + passages via embedder, run MMR.

    ``embedder`` must expose ``async embed(texts: list[str]) -> list[list[float]]``
    (the repository-wide Embedder protocol is already batch, so a single
    TEI call handles all passages + query together). A rare fallback path
    covers older clients that only expose ``embed_batch`` separately.

    ``max_input_size`` (Phase 6.X): caps how many hits are sent to the
    embedder for diversity scoring. The cross-encoder upstream has
    already ranked the candidates; MMR's job is to re-order the top
    band, not the entire candidate set. Passing all 200 hits under
    multi-entity decomposition can OOM the GPU shared with TEI / QU /
    reranker. Default ``None`` = no cap (byte-identical to pre-Phase-6
    behaviour). When set, hits beyond the cap are appended after the
    MMR-selected slice so we still return ``top_k`` items deterministically
    even when the cap is below ``top_k``.
    """
    if not hits:
        return []
    if top_k >= len(hits):
        return list(hits)

    hits_list = list(hits)
    if (
        max_input_size is not None
        and max_input_size > 0
        and len(hits_list) > max_input_size
    ):
        # Run MMR on the top-N slice, append the remainder unchanged so
        # the caller still has a full list of length up to len(hits).
        # The cross-encoder ordering of the trimmed slice is preserved
        # in the appended tail, which keeps the rerank-only quality on
        # the rest of the candidate set.
        head = hits_list[:max_input_size]
        tail = hits_list[max_input_size:]
        with span(
            "mmr.dedupe",
            input_size=len(head),
            input_truncated=len(hits_list),
            top_k=top_k,
            lambda_=lambda_,
        ):
            mmr_head = await _mmr_rerank_from_hits_impl(
                query, head, embedder, top_k=top_k, lambda_=lambda_,
            )
        # If MMR returned fewer than top_k (shouldn't happen but defensive),
        # backfill from the tail to keep the contract.
        if len(mmr_head) < top_k:
            mmr_head = mmr_head + tail[: top_k - len(mmr_head)]
        return mmr_head[:top_k]

    with span("mmr.dedupe", input_size=len(hits_list), top_k=top_k, lambda_=lambda_):
        return await _mmr_rerank_from_hits_impl(
            query, hits_list, embedder, top_k=top_k, lambda_=lambda_,
        )


async def _mmr_rerank_from_hits_impl(
    query: str,
    hits: Sequence[Any],
    embedder: Any,
    *,
    top_k: int,
    lambda_: float,
) -> list[Any]:
    # Wave 2 round 4 (review §5.9) — fast path: when EVERY hit carries a
    # ``.vector`` (because the upstream retriever asked Qdrant for
    # ``with_vectors=True``), skip the TEI re-embed of the passages and
    # only embed the query. Mixed input (some hits with vectors, some
    # without) safely falls back to the legacy batch path — interleaving
    # cached and re-embedded vectors is fragile (different model snapshots
    # can drift) and the savings are smaller anyway.
    prefetched_vecs: list[list[float]] | None = None
    if hits:
        candidate = []
        for h in hits:
            v = getattr(h, "vector", None)
            if v is None:
                candidate = None
                break
            candidate.append(list(v))
        if candidate is not None:
            prefetched_vecs = candidate

    if prefetched_vecs is not None:
        # Embed the query alone — single network round-trip, no passage re-embed.
        if hasattr(embedder, "embed"):
            qvec_batch = await embedder.embed([query])
        elif hasattr(embedder, "embed_batch"):
            qvec_batch = await embedder.embed_batch([query])
        else:  # pragma: no cover - defensive
            raise TypeError(
                "embedder must expose async embed(list[str]) or embed_batch(list[str])"
            )
        query_vec = qvec_batch[0]
        passage_vecs = prefetched_vecs
    else:
        texts = [
            (getattr(h, "payload", {}) or {}).get("text")
            or getattr(h, "text", "")
            or ""
            for h in hits
        ]

        # Single batched call: passages first, then the query. Matches the
        # TEIEmbedder.embed(list[str]) signature used everywhere else.
        if hasattr(embedder, "embed"):
            vectors = await embedder.embed(texts + [query])
        elif hasattr(embedder, "embed_batch"):
            vectors = await embedder.embed_batch(texts + [query])
        else:  # pragma: no cover - defensive
            raise TypeError(
                "embedder must expose async embed(list[str]) or embed_batch(list[str])"
            )

        query_vec = vectors[-1]
        passage_vecs = vectors[:-1]

    # §5.10 sanity check — TEI's bge-m3 returns L2-normalized vectors. If the
    # norm drifts (op-tag mismatch, future model swap, broken stub), drop a
    # warning so we notice. We then renormalize defensively inside mmr_rerank
    # so the algorithm stays correct either way.
    try:
        q_norm = float(np.linalg.norm(np.asarray(query_vec, dtype=np.float64)))
        if abs(q_norm - 1.0) > _UNIT_NORM_TOL:
            _log.warning(
                "MMR: query embedding not L2-normalized (||q||=%.4f); "
                "renormalizing in fast-path",
                q_norm,
            )
        if passage_vecs:
            p0_norm = float(
                np.linalg.norm(np.asarray(passage_vecs[0], dtype=np.float64))
            )
            if abs(p0_norm - 1.0) > _UNIT_NORM_TOL:
                _log.warning(
                    "MMR: first passage embedding not L2-normalized "
                    "(||p[0]||=%.4f); renormalizing in fast-path",
                    p0_norm,
                )
    except Exception:  # pragma: no cover - sanity check must never raise
        pass

    return mmr_rerank(query_vec, passage_vecs, hits, top_k=top_k, lambda_=lambda_)
