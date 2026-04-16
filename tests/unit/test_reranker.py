from ext.services.reranker import rerank
from ext.services.vector_store import Hit


def _h(id, score, kb_id):
    return Hit(id=id, score=score, payload={"kb_id": kb_id})


def test_fast_path_returns_top_k_unchanged_when_top1_dominates():
    hits = [_h(1, 0.9, 1), _h(2, 0.3, 1), _h(3, 0.1, 2)]
    out = rerank(hits, top_k=2)
    assert [h.id for h in out] == [1, 2]


def test_normalize_when_close_scores():
    hits = [
        _h(1, 0.8, 1), _h(2, 0.7, 1),
        _h(3, 0.4, 2), _h(4, 0.3, 2),
    ]
    out = rerank(hits, top_k=4)
    ids = [h.id for h in out]
    assert ids.index(1) < ids.index(2)
    assert ids.index(3) < ids.index(4)


def test_empty_returns_empty():
    assert rerank([], top_k=10) == []
