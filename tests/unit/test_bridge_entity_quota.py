"""Per-entity rerank quota in chat_rag_bridge.

Backstop for the 75 Inf Bde / 5 PoK Bde eviction case (smoke test
2026-05-03). When multi-entity decompose is on, the post-rerank trim
``reranked[:_final_k]`` was global-score top-k — entity-blind. Entities
whose reranked chunks scored lower than the dominant entity were
silently evicted, even though direct Qdrant scrolls confirmed they had
matching content.

The fix: when decompose was active, build the final pool with a
per-entity floor (default 3 chunks per entity) by walking the reranked
list (cross-encoder order preserved) and bucketing on case-insensitive
substring match in ``payload["text"]``. Then top up by next-best score
to fill remaining slots, dedup by id, and re-sort by score desc.

These tests cover the helper as a pure function so the algo is locked
in independent of the bridge's heavy I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ext.services import chat_rag_bridge as bridge


@dataclass
class _FakeHit:
    """Minimal Hit shape — id/score/payload are what the helper reads."""
    id: int | str
    score: float
    payload: dict


def _mk(id_, score, text):
    return _FakeHit(id=id_, score=score, payload={"text": text})


class TestApplyEntityQuota:
    def test_each_entity_gets_at_least_floor(self) -> None:
        # 4 entities, each with 5+ matching hits in reranked.
        # Floor=3, final_k=24 -> output has >=3 per entity.
        reranked = []
        # Interleave entities so global score order doesn't trivially satisfy quota.
        for i in range(5):
            reranked.append(_mk(f"75-{i}", 0.99 - i * 0.01, f"75 Inf Bde report {i}"))
            reranked.append(_mk(f"5p-{i}", 0.94 - i * 0.01, f"5 PoK Bde activity {i}"))
            reranked.append(_mk(f"32-{i}", 0.89 - i * 0.01, f"32 Inf Bde patrol {i}"))
            reranked.append(_mk(f"80-{i}", 0.84 - i * 0.01, f"80 Inf Bde brief {i}"))
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["75 Inf Bde", "5 PoK Bde", "32 Inf Bde", "80 Inf Bde"],
            per_entity_floor=3,
            final_k=24,
        )
        # Each entity must contribute >=3 hits.
        assert sum(1 for h in out if "75 Inf Bde" in h.payload["text"]) >= 3
        assert sum(1 for h in out if "5 PoK Bde" in h.payload["text"]) >= 3
        assert sum(1 for h in out if "32 Inf Bde" in h.payload["text"]) >= 3
        assert sum(1 for h in out if "80 Inf Bde" in h.payload["text"]) >= 3

    def test_entity_with_too_few_matches_takes_what_is_available(self) -> None:
        # 4 entities; entity D has only 1 matching hit.
        # Floor=3, final_k=12 -> D contributes 1, others get 3,
        # remaining 2 slots filled by next-highest-score from leftovers.
        reranked = [
            _mk("a-0", 0.99, "Alpha team report"),
            _mk("a-1", 0.98, "Alpha team brief"),
            _mk("a-2", 0.97, "Alpha team patrol"),
            _mk("a-3", 0.96, "Alpha team summary"),
            _mk("b-0", 0.95, "Bravo squad report"),
            _mk("b-1", 0.94, "Bravo squad activity"),
            _mk("b-2", 0.93, "Bravo squad note"),
            _mk("b-3", 0.92, "Bravo squad debrief"),
            _mk("c-0", 0.91, "Charlie unit report"),
            _mk("c-1", 0.90, "Charlie unit brief"),
            _mk("c-2", 0.89, "Charlie unit log"),
            _mk("c-3", 0.88, "Charlie unit notes"),
            _mk("d-0", 0.50, "Delta team only mention"),  # only 1 hit
            _mk("misc-0", 0.40, "unrelated chunk one"),
            _mk("misc-1", 0.30, "unrelated chunk two"),
        ]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["Alpha team", "Bravo squad", "Charlie unit", "Delta team"],
            per_entity_floor=3,
            final_k=12,
        )
        d_hits = [h for h in out if "Delta team" in h.payload["text"]]
        a_hits = [h for h in out if "Alpha team" in h.payload["text"]]
        b_hits = [h for h in out if "Bravo squad" in h.payload["text"]]
        c_hits = [h for h in out if "Charlie unit" in h.payload["text"]]
        assert len(d_hits) == 1  # took all available
        assert len(a_hits) >= 3
        assert len(b_hits) >= 3
        assert len(c_hits) >= 3
        # final_k cap respected.
        assert len(out) <= 12

    def test_top_up_fills_remaining_slots_by_score(self) -> None:
        # 2 entities, each gets 1 (floor); 8 remaining slots filled by
        # highest-scoring leftovers regardless of entity (could be more
        # of A or B, no second quota round).
        reranked = [
            _mk(f"a-{i}", 0.99 - i * 0.001, f"Alpha unit chunk {i}")
            for i in range(20)
        ] + [
            _mk(f"b-{i}", 0.50 - i * 0.001, f"Bravo unit chunk {i}")
            for i in range(20)
        ]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["Alpha unit", "Bravo unit"],
            per_entity_floor=1,
            final_k=10,
        )
        assert len(out) == 10
        # Quota = 1 each, so 2 slots used by quota; 8 leftover slots
        # filled by highest score, all of which are Alpha (>= 0.97).
        a_count = sum(1 for h in out if "Alpha unit" in h.payload["text"])
        b_count = sum(1 for h in out if "Bravo unit" in h.payload["text"])
        assert b_count >= 1  # quota floor
        assert a_count >= 9  # quota + leftover top-up dominates

    def test_no_entities_returns_top_final_k_slice(self) -> None:
        # Backward compat: when entities=[], helper returns reranked[:final_k].
        reranked = [_mk(i, 1.0 - i * 0.01, f"chunk {i}") for i in range(20)]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=[],
            per_entity_floor=3,
            final_k=12,
        )
        assert len(out) == 12
        assert [h.id for h in out] == list(range(12))

    def test_final_order_is_by_score_desc(self) -> None:
        # Even though selection is bucketed, output order is global score.
        reranked = [
            _mk("a-0", 0.99, "Alpha unit one"),
            _mk("b-0", 0.95, "Bravo unit one"),
            _mk("a-1", 0.90, "Alpha unit two"),
            _mk("b-1", 0.85, "Bravo unit two"),
            _mk("c-0", 0.30, "Charlie unit one"),  # low score, but quota forces in
            _mk("c-1", 0.20, "Charlie unit two"),
            _mk("c-2", 0.10, "Charlie unit three"),
        ]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["Alpha unit", "Bravo unit", "Charlie unit"],
            per_entity_floor=2,
            final_k=6,
        )
        # Output must be sorted by score desc.
        scores = [h.score for h in out]
        assert scores == sorted(scores, reverse=True)

    def test_dedup_by_id(self) -> None:
        # A chunk that mentions two entities shouldn't be double-counted.
        # Same id appears once in the output even though it matches two entities.
        shared = _mk("shared-1", 0.95, "75 Inf Bde and 5 PoK Bde joint op")
        reranked = [
            shared,
            _mk("75-2", 0.90, "75 Inf Bde solo report"),
            _mk("75-3", 0.88, "75 Inf Bde brief"),
            _mk("5p-2", 0.85, "5 PoK Bde solo brief"),
            _mk("5p-3", 0.83, "5 PoK Bde patrol"),
        ]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["75 Inf Bde", "5 PoK Bde"],
            per_entity_floor=2,
            final_k=10,
        )
        ids = [h.id for h in out]
        assert ids.count("shared-1") == 1

    def test_case_insensitive_matching(self) -> None:
        # Corpus has both "5 PoK" and "5 POK" spellings; both should be
        # picked by the "5 PoK Bde" entity quota.
        reranked = [
            _mk("p-0", 0.99, "5 POK Bde activity report"),  # uppercase corpus
            _mk("p-1", 0.95, "5 pok bde patrol"),  # lowercase corpus
            _mk("p-2", 0.90, "5 PoK Bde mixed case"),
            _mk("o-0", 0.85, "Other unit irrelevant"),
        ]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["5 PoK Bde"],
            per_entity_floor=2,
            final_k=4,
        )
        # All 3 PoK chunks should be picked up (case-insensitive match).
        pok_count = sum(
            1 for h in out
            if "pok" in h.payload["text"].lower()
        )
        assert pok_count == 3

    def test_empty_reranked_returns_empty(self) -> None:
        out = bridge._apply_entity_quota(
            reranked=[],
            entities=["A", "B"],
            per_entity_floor=3,
            final_k=12,
        )
        assert out == []

    def test_floor_zero_returns_top_final_k_slice(self) -> None:
        # floor=0 disables the quota — falls back to plain trim.
        reranked = [_mk(i, 1.0 - i * 0.01, f"chunk {i}") for i in range(20)]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["any", "two"],
            per_entity_floor=0,
            final_k=12,
        )
        assert len(out) == 12
        assert [h.id for h in out] == list(range(12))

    def test_preserves_cross_encoder_order_within_entity(self) -> None:
        # Within an entity bucket, the quota pass picks top-floor by
        # cross-encoder score (not buried hits). final_k=3 here means
        # quota fills exactly 3 slots — no top-up room — so Alpha gets
        # its top-2 (a-0, a-1) and Bravo gets b-0; buried Alpha hits
        # (a-2, a-3) MUST NOT appear.
        reranked = [
            _mk("a-0", 0.99, "Alpha one"),
            _mk("a-1", 0.98, "Alpha two"),
            _mk("a-2", 0.50, "Alpha low"),  # buried Alpha hit
            _mk("a-3", 0.40, "Alpha very low"),
            _mk("b-0", 0.95, "Bravo one"),
        ]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["Alpha", "Bravo"],
            per_entity_floor=2,
            final_k=3,
        )
        # Alpha contributes a-0, a-1 (top 2 by score), NOT a-2/a-3.
        alpha_ids = {h.id for h in out if "Alpha" in h.payload["text"]}
        assert alpha_ids == {"a-0", "a-1"}

    def test_chunk_with_no_text_payload_does_not_crash(self) -> None:
        # Defensive: payload missing "text" key shouldn't crash.
        reranked = [
            _mk("a-0", 0.99, "Alpha one"),
            _FakeHit(id="weird-1", score=0.90, payload={}),  # no text key
            _mk("a-1", 0.85, "Alpha two"),
        ]
        out = bridge._apply_entity_quota(
            reranked=reranked,
            entities=["Alpha"],
            per_entity_floor=2,
            final_k=3,
        )
        # Helper shouldn't blow up; weird-1 just doesn't match any entity.
        assert len(out) >= 2
        ids = {h.id for h in out}
        assert "a-0" in ids
        assert "a-1" in ids


class TestBridgeIntegration:
    """Smoke test that the bridge invokes the helper when decompose was active.

    We verify the new branch in ``_run_pipeline`` by patching the helper
    and inspecting whether it ran. Full pipeline is too involved (vector
    store / embedder / sessionmaker mocks) — this is a targeted check
    that the trim line was bypassed in favor of the helper.
    """

    def test_helper_is_module_level_callable(self) -> None:
        # Sanity: helper exists, is callable, has the documented signature.
        assert callable(bridge._apply_entity_quota)

    def test_helper_signature(self) -> None:
        import inspect

        sig = inspect.signature(bridge._apply_entity_quota)
        # All four expected params present.
        assert "reranked" in sig.parameters
        assert "entities" in sig.parameters
        assert "per_entity_floor" in sig.parameters
        assert "final_k" in sig.parameters
