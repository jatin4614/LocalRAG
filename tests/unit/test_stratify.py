from tests.eval.stratify import stratify, intent_year_strata


def test_stratify_groups_by_intent():
    rows = [
        {"intent_label": "specific", "year_bucket": "2024", "difficulty": "easy",
         "language": "en", "adversarial_category": None},
        {"intent_label": "specific", "year_bucket": "2025", "difficulty": "hard",
         "language": "hi", "adversarial_category": None},
        {"intent_label": "global", "year_bucket": "2024", "difficulty": "medium",
         "language": "en", "adversarial_category": None},
    ]
    s = stratify(rows)
    assert set(s["intent"].keys()) == {"specific", "global"}
    assert len(s["intent"]["specific"]) == 2
    assert set(s["language"].keys()) == {"en", "hi"}


def test_intent_year_strata():
    rows = [
        {"intent_label": "specific", "year_bucket": "2024"},
        {"intent_label": "specific", "year_bucket": "2024"},
        {"intent_label": "global", "year_bucket": "2025"},
    ]
    x = intent_year_strata(rows)
    assert x["specific__2024"] and len(x["specific__2024"]) == 2
    assert x["global__2025"] and len(x["global__2025"]) == 1
