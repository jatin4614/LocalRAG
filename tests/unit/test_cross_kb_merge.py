"""Unit tests for cross-KB merge behaviour in ``ext.services.retriever``.

When the cross-encoder reranker is ON, raw Qdrant scores from different
collections all get re-scored against the same query later anyway — so a
simple global sort by score is fine.

When the reranker is OFF the raw scores aren't comparable across collections
(different score distributions per KB), and a "chatty" KB whose corpus
happens to score systematically higher will dominate. RRF (rank-based fusion)
neutralises the absolute-score asymmetry and gives every KB a fair shot.
"""
from ext.services.retriever import merge_kb_results


def _hit(kb_id, doc_id, chunk_index, score):
    return {
        "kb_id": kb_id, "doc_id": doc_id, "chunk_index": chunk_index,
        "score": score,
    }


def test_merge_when_rerank_is_on_sorts_by_score_global():
    per_kb = {
        1: [_hit(1, 100, 0, 0.95), _hit(1, 101, 0, 0.60)],
        2: [_hit(2, 200, 0, 0.88)],
    }
    out = merge_kb_results(per_kb, rerank_enabled=True, top_k=3)
    assert [h["doc_id"] for h in out] == [100, 200, 101]


def test_merge_when_rerank_is_off_uses_rrf():
    """Without rerank, raw scores aren't comparable. RRF by rank in each KB
    should NOT let a chatty KB dominate."""
    per_kb = {
        1: [_hit(1, 100, 0, 0.9), _hit(1, 101, 0, 0.7), _hit(1, 102, 0, 0.5)],
        2: [_hit(2, 200, 0, 0.3), _hit(2, 201, 0, 0.2), _hit(2, 202, 0, 0.1)],
    }
    out = merge_kb_results(per_kb, rerank_enabled=False, top_k=4)
    top2 = {h["doc_id"] for h in out[:2]}
    assert top2 == {100, 200}, f"RRF must balance across KBs, got {top2}"


def test_merge_empty_kbs_are_tolerated():
    per_kb = {1: [], 2: [_hit(2, 200, 0, 0.5)]}
    out = merge_kb_results(per_kb, rerank_enabled=False, top_k=5)
    assert [h["doc_id"] for h in out] == [200]


def test_merge_preserves_kb_id_payload():
    per_kb = {1: [_hit(1, 100, 0, 0.9)]}
    out = merge_kb_results(per_kb, rerank_enabled=True, top_k=1)
    assert out[0]["kb_id"] == 1
