import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "eval" / "golden_starter.jsonl"


def _load() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def test_golden_starter_exists_and_nonempty():
    assert GOLDEN.exists(), f"missing {GOLDEN}"
    rows = _load()
    assert len(rows) == 60, f"expected 60 rows, got {len(rows)}"


def test_required_fields_present():
    required = {"query", "intent_label", "year_bucket", "difficulty",
                "expected_doc_ids", "expected_chunk_indices",
                "language", "adversarial_category"}
    for i, row in enumerate(_load()):
        missing = required - set(row.keys())
        assert not missing, f"row {i} missing fields: {missing}"


def test_intent_distribution():
    rows = _load()
    c = Counter(r["intent_label"] for r in rows)
    assert c["specific"] == 30
    assert c["global"] == 15
    assert c["metadata"] == 7
    assert c["multihop"] == 5
    assert c["adversarial"] == 3


def test_year_distribution():
    rows = _load()
    c = Counter(r["year_bucket"] for r in rows if r["intent_label"] != "adversarial")
    assert c["2023"] == 11
    assert c["2024"] == 23
    assert c["2025"] == 15
    assert c["2026"] == 8


def test_non_english_tag_count():
    rows = _load()
    n = sum(1 for r in rows if r["language"] != "en")
    assert n == 5, f"expected 5 non-English tagged queries, got {n}"


def test_adversarial_categories_cover_all_three():
    rows = [r for r in _load() if r["intent_label"] == "adversarial"]
    cats = {r["adversarial_category"] for r in rows}
    assert cats == {"prompt_injection", "cross_user_probe", "empty_retrieval"}


def test_difficulty_values_valid():
    valid = {"easy", "medium", "hard"}
    for r in _load():
        assert r["difficulty"] in valid, f"bad difficulty: {r['difficulty']}"
