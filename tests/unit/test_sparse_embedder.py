"""Unit tests for ``ext.services.sparse_embedder``.

Skipped as a whole if ``fastembed`` isn't installed — this is a dev-opt-in
dependency (see ``pyproject.toml`` ``[hybrid]`` extra). These tests are
intentionally thin: they verify the thin wrapper over fastembed behaves
correctly (shape, determinism, embed-vs-query-embed IDF difference), not
fastembed itself.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastembed")

from ext.services import sparse_embedder as se  # noqa: E402


def test_embed_sparse_returns_indices_values_pairs() -> None:
    pairs = se.embed_sparse(["the quick brown fox"])
    assert len(pairs) == 1
    indices, values = pairs[0]
    assert isinstance(indices, list)
    assert isinstance(values, list)
    assert len(indices) == len(values)
    assert len(indices) > 0, "BM25 of a non-trivial sentence must produce at least one token"
    assert all(isinstance(i, int) for i in indices)
    assert all(isinstance(v, float) for v in values)


def test_embed_sparse_is_deterministic() -> None:
    """Same input → same output on repeat calls (stable model state)."""
    first = se.embed_sparse(["pricing negotiation strategy"])
    second = se.embed_sparse(["pricing negotiation strategy"])
    assert first == second


def test_embed_sparse_query_differs_from_embed_due_to_idf_marker() -> None:
    """``embed`` returns TF-weighted values; ``query_embed`` returns all-ones
    (the IDF weighting happens server-side in Qdrant when the sparse vector
    is configured with ``Modifier.IDF``).

    A regression that made ``query_embed`` mirror ``embed`` would break
    server-side IDF fusion in hybrid mode, so we lock the distinction.
    """
    pairs = se.embed_sparse(["pricing"])
    doc_idx, doc_vals = pairs[0]
    q_idx, q_vals = se.embed_sparse_query("pricing")

    assert len(doc_idx) > 0
    assert len(q_idx) > 0
    # Index sets are equal (same token) — values differ.
    assert set(doc_idx) == set(q_idx)
    # ``query_embed`` produces all-ones: the IDF multiplier is applied by Qdrant.
    assert all(v == 1.0 for v in q_vals), (
        f"query values should all be 1.0 (IDF applied server-side), got {q_vals}"
    )


def test_embed_sparse_handles_empty_list_cleanly() -> None:
    assert se.embed_sparse([]) == []


def test_embed_sparse_query_returns_empty_for_empty_string() -> None:
    idx, vals = se.embed_sparse_query("")
    assert idx == []
    assert vals == []


def test_embed_sparse_batch_of_multiple() -> None:
    pairs = se.embed_sparse([
        "the quick brown fox",
        "pricing negotiation strategy",
        "the quick brown fox",
    ])
    assert len(pairs) == 3
    # First and third share text → identical output.
    assert pairs[0] == pairs[2]
    # First vs second differ (different tokens).
    assert pairs[0] != pairs[1]
