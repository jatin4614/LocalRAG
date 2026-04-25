import pytest
from tests.eval.scorer import ndcg_at_k


def test_ndcg_perfect_ranking_is_one():
    retrieved = [1, 2, 3, 99, 98]
    gold = {1, 2, 3}
    assert ndcg_at_k(retrieved, gold, k=5) == pytest.approx(1.0, abs=1e-6)


def test_ndcg_all_gold_missing_is_zero():
    retrieved = [99, 98, 97]
    gold = {1, 2, 3}
    assert ndcg_at_k(retrieved, gold, k=3) == 0.0


def test_ndcg_half_correct_is_below_one():
    retrieved = [1, 99, 2, 98, 3]
    gold = {1, 2, 3}
    score = ndcg_at_k(retrieved, gold, k=5)
    assert 0.5 < score < 1.0


def test_ndcg_empty_retrieved_is_zero():
    assert ndcg_at_k([], {1}, k=5) == 0.0


def test_ndcg_empty_gold_returns_one_by_convention():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 1.0
