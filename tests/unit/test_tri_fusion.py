"""Phase 3.5 — tri-fusion RRF helper.

Pure-function unit tests for ``rrf_fuse_heads``: fuses N retrieval heads
(each a ranked list of (doc_id, rank)) into a single sorted list using
the canonical RRF formula ``Σ 1/(k + rank + 1)`` across heads. Used by
the per-KB search path to combine dense + sparse + ColBERT result lists
when ``RAG_COLBERT=1`` and the collection has the named ``colbert``
vector slot. Wiring tests live in the retriever / vector_store hybrid
test modules; here we only validate the math.
"""
from ext.services.retriever import rrf_fuse_heads


def test_rrf_fuses_three_heads():
    dense = [("a", 0), ("b", 1), ("c", 2)]
    sparse = [("b", 0), ("a", 1), ("d", 2)]
    colbert = [("a", 0), ("c", 1), ("e", 2)]

    fused = rrf_fuse_heads([dense, sparse, colbert], k=60, top_k=5)
    ids = [x[0] for x in fused]
    assert ids[0] == "a"
    assert ids[1] in ("b", "c")
    assert set(ids) == {"a", "b", "c", "d", "e"}


def test_rrf_with_two_heads_degrades_gracefully():
    dense = [("a", 0), ("b", 1)]
    sparse = [("b", 0), ("a", 1)]
    fused = rrf_fuse_heads([dense, sparse], k=60, top_k=5)
    ids = [x[0] for x in fused]
    assert set(ids) == {"a", "b"}


def test_rrf_empty_heads():
    assert rrf_fuse_heads([], k=60, top_k=5) == []
    assert rrf_fuse_heads([[]], k=60, top_k=5) == []
