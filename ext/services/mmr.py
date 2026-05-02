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
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .obs import span


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length float sequences.

    Dense vectors from TEI's BAAI/bge-m3 are already L2-normalised so a dot
    product is equivalent to cosine; we still guard against zero vectors
    (e.g., synthetic test fixtures) to avoid division-by-zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


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
    """
    if not hits:
        return []
    n = len(hits)
    if top_k >= n:
        # MMR is only meaningful when we are selecting a strict subset. When
        # top_k >= n we have nothing to diversify; return in input order so
        # callers observe a byte-identical pass-through.
        return list(hits)

    relevance = [_cosine(query_vec, passage_vecs[i]) for i in range(n)]

    # Step 1: seed selection with the most relevant. Tie-break by lower index.
    remaining = set(range(n))
    first = max(remaining, key=lambda i: (relevance[i], -i))
    selected: list[int] = [first]
    remaining.remove(first)

    # Step 2: iteratively pick the best MMR candidate among the remainder.
    while len(selected) < top_k and remaining:
        best_i = -1
        best_score = -float("inf")
        for i in sorted(remaining):
            max_sim_to_selected = max(
                _cosine(passage_vecs[i], passage_vecs[s]) for s in selected
            )
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

    return mmr_rerank(query_vec, passage_vecs, hits, top_k=top_k, lambda_=lambda_)
