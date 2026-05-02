"""Unit tests for ext.services.multi_query (Phase 6.X — Method 3).

Three pure functions:

* ``should_decompose(entities, flag_on, intent) -> bool`` — gate.
* ``build_sub_queries(query, entities) -> list[(entity, sub_query)]`` —
  builds focus-shifted sub-queries.
* ``merge_with_quota(per_entity_hits, k_min_per_entity, k_total) ->
  list[hit]`` — merges N per-entity hit lists with a per-entity floor,
  then fills the remainder by score, deduping by point id.
"""
from __future__ import annotations

from dataclasses import dataclass

from ext.services import multi_query


@dataclass
class _FakeHit:
    """Minimal Hit shape — ``id`` + ``score`` are what merge_with_quota uses."""
    id: int | str
    score: float
    payload: dict | None = None


class TestShouldDecompose:
    def test_flag_off_returns_false(self) -> None:
        assert multi_query.should_decompose(
            entities=["A", "B", "C"], flag_on=False, intent="specific",
        ) is False

    def test_zero_entities_returns_false(self) -> None:
        assert multi_query.should_decompose(
            entities=[], flag_on=True, intent="specific",
        ) is False

    def test_one_entity_returns_false(self) -> None:
        # Single-entity queries go through the existing path.
        assert multi_query.should_decompose(
            entities=["A"], flag_on=True, intent="specific",
        ) is False

    def test_metadata_intent_returns_false(self) -> None:
        # Catalog questions never decompose.
        assert multi_query.should_decompose(
            entities=["A", "B"], flag_on=True, intent="metadata",
        ) is False

    def test_two_entities_flag_on_returns_true(self) -> None:
        assert multi_query.should_decompose(
            entities=["A", "B"], flag_on=True, intent="specific",
        ) is True

    def test_intent_none_treated_as_specific(self) -> None:
        # Defensive: if intent classifier didn't run, decompose anyway.
        assert multi_query.should_decompose(
            entities=["A", "B"], flag_on=True, intent=None,
        ) is True

    def test_global_intent_decomposes(self) -> None:
        assert multi_query.should_decompose(
            entities=["A", "B"], flag_on=True, intent="global",
        ) is True


class TestBuildSubQueries:
    def test_one_pair_per_entity(self) -> None:
        out = multi_query.build_sub_queries(
            "Apr 2026 updates", ["A", "B", "C"],
        )
        assert len(out) == 3
        for (entity, _) in out:
            assert entity in ("A", "B", "C")

    def test_sub_query_contains_entity_and_original(self) -> None:
        out = multi_query.build_sub_queries(
            "Apr 2026 updates", ["32 Inf Bde"],
        )
        entity, sub = out[0]
        assert entity == "32 Inf Bde"
        assert "32 Inf Bde" in sub
        assert "Apr 2026" in sub

    def test_empty_entities_returns_empty(self) -> None:
        assert multi_query.build_sub_queries("anything", []) == []

    def test_preserves_entity_order(self) -> None:
        ents = ["E1", "E2", "E3", "E4"]
        out = multi_query.build_sub_queries("query", ents)
        assert [e for (e, _) in out] == ents


class TestMergeWithQuota:
    def test_quota_floor_respected(self) -> None:
        # Each entity has 5 hits; k_min=2 → final list has ≥2 per entity.
        per_entity = {
            "A": [_FakeHit(id=f"A{i}", score=0.9 - i * 0.1) for i in range(5)],
            "B": [_FakeHit(id=f"B{i}", score=0.8 - i * 0.1) for i in range(5)],
            "C": [_FakeHit(id=f"C{i}", score=0.7 - i * 0.1) for i in range(5)],
        }
        out = multi_query.merge_with_quota(
            per_entity_hits=per_entity,
            k_min_per_entity=2,
            k_total=10,
        )
        # Each entity must contribute ≥2 hits (the floor).
        assert sum(1 for h in out if str(h.id).startswith("A")) >= 2
        assert sum(1 for h in out if str(h.id).startswith("B")) >= 2
        assert sum(1 for h in out if str(h.id).startswith("C")) >= 2

    def test_dedupe_by_id(self) -> None:
        # Same point appears in two entity buckets — keep the higher score.
        shared_a = _FakeHit(id="X", score=0.5, payload={"e": "A"})
        shared_b = _FakeHit(id="X", score=0.9, payload={"e": "B"})
        per_entity = {
            "A": [shared_a],
            "B": [shared_b],
        }
        out = multi_query.merge_with_quota(
            per_entity_hits=per_entity,
            k_min_per_entity=1,
            k_total=10,
        )
        assert len(out) == 1
        assert out[0].id == "X"
        # Higher-scoring copy wins.
        assert out[0].score == 0.9

    def test_total_cap_respected(self) -> None:
        # 5 entities × 5 hits = 25 candidates; cap at 12.
        per_entity = {
            f"E{e}": [_FakeHit(id=f"E{e}-{i}", score=1.0 - i * 0.01) for i in range(5)]
            for e in range(5)
        }
        out = multi_query.merge_with_quota(
            per_entity_hits=per_entity,
            k_min_per_entity=1,
            k_total=12,
        )
        assert len(out) == 12

    def test_quota_overrides_pure_score(self) -> None:
        # Entity A has dominant scores; without quota, B/C/D would be evicted.
        # With k_min=2 we MUST see ≥2 from each entity even though A wins on raw rank.
        per_entity = {
            "A": [_FakeHit(id=f"A{i}", score=0.99 - i * 0.001) for i in range(15)],
            "B": [_FakeHit(id=f"B{i}", score=0.50 - i * 0.01) for i in range(3)],
            "C": [_FakeHit(id=f"C{i}", score=0.30 - i * 0.01) for i in range(3)],
            "D": [_FakeHit(id=f"D{i}", score=0.20 - i * 0.01) for i in range(3)],
        }
        out = multi_query.merge_with_quota(
            per_entity_hits=per_entity,
            k_min_per_entity=2,
            k_total=12,
        )
        a = sum(1 for h in out if str(h.id).startswith("A"))
        b = sum(1 for h in out if str(h.id).startswith("B"))
        c = sum(1 for h in out if str(h.id).startswith("C"))
        d = sum(1 for h in out if str(h.id).startswith("D"))
        assert b >= 2
        assert c >= 2
        assert d >= 2
        assert a >= 2  # A still gets its quota floor

    def test_empty_bucket_skipped(self) -> None:
        per_entity = {
            "A": [_FakeHit(id="A1", score=0.9)],
            "B": [],  # empty — entity has no data, not a regression
            "C": [_FakeHit(id="C1", score=0.7)],
        }
        out = multi_query.merge_with_quota(
            per_entity_hits=per_entity,
            k_min_per_entity=2,
            k_total=10,
        )
        # Empty bucket contributes 0; A and C still in the output.
        ids = {h.id for h in out}
        assert "A1" in ids
        assert "C1" in ids

    def test_single_entity_passthrough(self) -> None:
        per_entity = {
            "A": [_FakeHit(id=f"A{i}", score=0.9 - i * 0.1) for i in range(3)],
        }
        out = multi_query.merge_with_quota(
            per_entity_hits=per_entity,
            k_min_per_entity=10,  # > available
            k_total=10,
        )
        # Quota can't be enforced when one entity, but all available hits returned.
        assert len(out) == 3

    def test_below_quota_when_available_lt_floor(self) -> None:
        # Entity has fewer hits than the quota floor — that's OK, take what
        # exists; don't pad with duplicates from elsewhere.
        per_entity = {
            "A": [_FakeHit(id="A1", score=0.9)],  # only 1 hit
            "B": [_FakeHit(id=f"B{i}", score=0.5 - i * 0.01) for i in range(10)],
        }
        out = multi_query.merge_with_quota(
            per_entity_hits=per_entity,
            k_min_per_entity=3,
            k_total=10,
        )
        # A contributes its 1 hit; B fills the rest.
        a = [h for h in out if str(h.id).startswith("A")]
        assert len(a) == 1

    def test_final_sort_by_score(self) -> None:
        per_entity = {
            "A": [_FakeHit(id="A1", score=0.5)],
            "B": [_FakeHit(id="B1", score=0.9)],
            "C": [_FakeHit(id="C1", score=0.7)],
        }
        out = multi_query.merge_with_quota(
            per_entity_hits=per_entity,
            k_min_per_entity=1,
            k_total=10,
        )
        # Quota satisfied (1 each); final list sorted by score desc.
        assert [h.id for h in out] == ["B1", "C1", "A1"]
