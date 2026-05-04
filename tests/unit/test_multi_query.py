"""Unit tests for ext.services.multi_query (Phase 6.X — Method 3).

Three pure functions:

* ``should_decompose(entities, subtopics, flag_on, intent) -> (mode, bool)`` — gate.
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
        mode, on = multi_query.should_decompose(
            entities=["A", "B", "C"], flag_on=False, intent="specific",
        )
        assert on is False

    def test_zero_entities_returns_false(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=[], flag_on=True, intent="specific",
        )
        assert on is False

    def test_one_entity_returns_false(self) -> None:
        # Single-entity queries go through the existing path.
        mode, on = multi_query.should_decompose(
            entities=["A"], flag_on=True, intent="specific",
        )
        assert on is False

    def test_metadata_intent_returns_false(self) -> None:
        # Catalog questions never decompose.
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], flag_on=True, intent="metadata",
        )
        assert on is False

    def test_two_entities_flag_on_returns_true(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("entity", True)

    def test_intent_none_treated_as_specific(self) -> None:
        # Defensive: if intent classifier didn't run, decompose anyway.
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], flag_on=True, intent=None,
        )
        assert on is True

    def test_global_intent_decomposes(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], flag_on=True, intent="global",
        )
        assert on is True


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


class TestShouldDecomposeTwoAxis:
    """Phase 3 — two-axis return: ('none' | 'entity' | 'subtopic' | 'both', bool)."""

    def test_no_entities_no_subtopics_returns_none(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=[], subtopics=[], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("none", False)

    def test_two_entities_no_subtopics_returns_entity(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], subtopics=[], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("entity", True)

    def test_no_entities_two_subtopics_returns_subtopic(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=[], subtopics=["visits", "ops"], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("subtopic", True)

    def test_two_entities_two_subtopics_returns_both(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], subtopics=["x", "y"], flag_on=True, intent="specific",
        )
        assert (mode, on) == ("both", True)

    def test_metadata_intent_returns_none(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], subtopics=["x", "y"],
            flag_on=True, intent="metadata",
        )
        assert (mode, on) == ("none", False)

    def test_flag_off_returns_none(self) -> None:
        mode, on = multi_query.should_decompose(
            entities=["A", "B"], subtopics=["x", "y"],
            flag_on=False, intent="specific",
        )
        assert (mode, on) == ("none", False)


class TestBuildSubQueriesTwoAxis:
    def test_n_x_m_pairs(self) -> None:
        out = multi_query.build_sub_queries_two_axis(
            "Give all updates",
            entities=["A", "B"],
            subtopics=["visits", "ops"],
        )
        assert len(out) == 4
        assert out[0] == ("A", "visits", "Give all updates (focus on A — visits)")
        assert out[1] == ("A", "ops", "Give all updates (focus on A — ops)")
        assert out[2] == ("B", "visits", "Give all updates (focus on B — visits)")
        assert out[3] == ("B", "ops", "Give all updates (focus on B — ops)")

    def test_empty_entities_returns_empty(self) -> None:
        assert multi_query.build_sub_queries_two_axis(
            "x", entities=[], subtopics=["a", "b"],
        ) == []

    def test_empty_subtopics_returns_empty(self) -> None:
        # Two-axis function rejects subtopics=[]; caller should fall back
        # to the single-axis build_sub_queries
        assert multi_query.build_sub_queries_two_axis(
            "x", entities=["A", "B"], subtopics=[],
        ) == []

    def test_blank_query_uses_placeholder(self) -> None:
        out = multi_query.build_sub_queries_two_axis(
            "", entities=["A"], subtopics=["x"],
        )
        assert out == [("A", "x", "(no query) (focus on A — x)")]


class TestMergeWithTwoAxisQuota:
    def _hits(self, scores: list[tuple[str, str, int, float]]):
        # (entity, subtopic, hit_id, score) -> {(e,s): [hit, ...]}
        out: dict = {}
        for e, s, hid, sc in scores:
            out.setdefault((e, s), []).append(_FakeHit(id=hid, score=sc))
        for k in out:
            out[k].sort(key=lambda h: h.score, reverse=True)
        return out

    def test_cell_floor_satisfied(self) -> None:
        per_cell = self._hits([
            ("A", "x", 1, 1.0), ("A", "x", 2, 0.9),
            ("A", "y", 3, 0.8), ("A", "y", 4, 0.7),
            ("B", "x", 5, 0.6), ("B", "x", 6, 0.5),
            ("B", "y", 7, 0.4), ("B", "y", 8, 0.3),
        ])
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=1,
            k_min_per_entity=2,
            k_min_per_subtopic=2,
            k_total=8,
        )
        ids = [h.id for h in out]
        # Each cell got at least 1 hit; final sorted by score desc
        assert ids == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_dedupe_by_id(self) -> None:
        # Same hit appearing in two cells should appear once in output
        h = _FakeHit(id=99, score=1.0)
        per_cell = {
            ("A", "x"): [h],
            ("A", "y"): [h],
            ("B", "x"): [_FakeHit(id=2, score=0.5)],
            ("B", "y"): [_FakeHit(id=3, score=0.4)],
        }
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=1, k_min_per_entity=1,
            k_min_per_subtopic=1, k_total=4,
        )
        ids = [h.id for h in out]
        assert ids.count(99) == 1

    def test_total_cap_respected(self) -> None:
        per_cell = self._hits([
            ("A", "x", i, 1.0 - i * 0.01) for i in range(20)
        ])
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=2,
            k_min_per_entity=2,
            k_min_per_subtopic=2,
            k_total=5,
        )
        assert len(out) == 5

    def test_entity_floor_recovery_when_cell_empty(self) -> None:
        # 5 PoK x visits has 0 hits but 5 PoK x ops has plenty.
        # Entity-floor recovery pulls extra ops chunks so the per-entity
        # floor is met.
        per_cell = self._hits([
            ("75 Inf", "visits", 1, 0.9), ("75 Inf", "visits", 2, 0.8),
            ("75 Inf", "ops",    3, 0.7), ("75 Inf", "ops",    4, 0.6),
            # 5 PoK has 0 visits but 4 ops:
            ("5 PoK", "ops",     5, 0.5), ("5 PoK", "ops",     6, 0.4),
            ("5 PoK", "ops",     7, 0.3), ("5 PoK", "ops",     8, 0.2),
        ])
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=1,
            k_min_per_entity=3,    # 5 PoK must end up with ≥3 chunks
            k_min_per_subtopic=2,
            k_total=10,
        )
        ids = [h.id for h in out]
        # 5 PoK should have ≥3 of its 4 ops chunks (5,6,7,8)
        pok_ids = [i for i in ids if i in (5, 6, 7, 8)]
        assert len(pok_ids) >= 3

    def test_leftover_cell_does_not_inflate_floors(self) -> None:
        """The __leftover__ synthetic cell from _apply_two_axis_quota
        should NOT consume entity/subtopic-floor quota slots — leftover
        hits should only enter via the top-up pass."""
        per_cell = {
            ("A", "x"): [_FakeHit(id=1, score=0.9), _FakeHit(id=2, score=0.8)],
            # Leftover hits — would have inflated floor recovery before fix
            ("__leftover__", "__leftover__"): [
                _FakeHit(id=10, score=0.5), _FakeHit(id=11, score=0.4),
                _FakeHit(id=12, score=0.3),
            ],
        }
        out = multi_query.merge_with_two_axis_quota(
            per_cell_hits=per_cell,
            k_min_per_cell=1,
            k_min_per_entity=2,
            k_min_per_subtopic=2,
            k_total=4,
        )
        ids = [h.id for h in out]
        # All 4 final slots: A's 2 hits (cell-quota + entity-floor satisfied
        # without pulling __leftover__) + 2 leftover via top-up.
        # The fix ensures __leftover__ is NOT treated as a real entity that
        # demands its own k_min_per_entity quota.
        assert 1 in ids and 2 in ids
        assert len(ids) == 4
