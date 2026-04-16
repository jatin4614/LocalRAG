from ext.services.budget import budget_chunks
from ext.services.vector_store import Hit


def _h(text, score=0.5):
    return Hit(id=1, score=score, payload={"text": text})


def test_all_fit():
    hits = [_h("hi"), _h("ok")]
    out = budget_chunks(hits, max_tokens=100)
    assert len(out) == 2


def test_truncates_from_lowest_rank_last():
    long_text = " ".join(["word"] * 50)
    hits = [_h(long_text, score=0.9), _h(long_text, score=0.7), _h(long_text, score=0.5)]
    out = budget_chunks(hits, max_tokens=60)
    assert len(out) == 1
    assert out[0].score == 0.9


def test_empty_input():
    assert budget_chunks([], max_tokens=100) == []
