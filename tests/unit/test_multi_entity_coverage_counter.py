"""Phase 3 — rag_multi_entity_coverage_total counter."""
from prometheus_client import REGISTRY

from ext.services.metrics import rag_multi_entity_coverage_total
from ext.services.chat_rag_bridge import _apply_entity_quota
from ext.services.vector_store import Hit


def _hit(id_, score, text):
    return Hit(id=id_, score=score, payload={"text": text})


def _read(outcome: str, entity_count: str) -> float:
    samples = list(REGISTRY.collect())
    for fam in samples:
        if fam.name == "rag_multi_entity_coverage":
            for s in fam.samples:
                if (s.labels.get("outcome") == outcome
                        and s.labels.get("entity_count") == entity_count
                        and s.name == "rag_multi_entity_coverage_total"):
                    return s.value
    return 0.0


def test_full_coverage_bumps_full_outcome():
    before = _read("full", "2")
    hits = [
        _hit(1, 0.9, "75 Inf Bde visited..."),
        _hit(2, 0.8, "75 Inf Bde inspection..."),
        _hit(3, 0.7, "75 Inf Bde 03 Apr..."),
        _hit(4, 0.6, "5 PoK Bde practice..."),
        _hit(5, 0.5, "5 PoK Bde rotation..."),
        _hit(6, 0.4, "5 PoK Bde meeting..."),
    ]
    _apply_entity_quota(
        reranked=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=3, final_k=12,
    )
    after = _read("full", "2")
    assert after == before + 1


def test_partial_coverage_bumps_partial_outcome():
    before = _read("partial", "2")
    hits = [
        _hit(1, 0.9, "75 Inf Bde visited..."),
        _hit(2, 0.8, "75 Inf Bde inspection..."),
        _hit(3, 0.7, "75 Inf Bde 03 Apr..."),
        _hit(4, 0.6, "5 PoK Bde practice..."),
        # only 1 chunk for 5 PoK Bde — partial
    ]
    _apply_entity_quota(
        reranked=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=3, final_k=12,
    )
    after = _read("partial", "2")
    assert after == before + 1


def test_empty_coverage_bumps_empty_outcome():
    before = _read("empty", "2")
    hits = [
        _hit(1, 0.9, "75 Inf Bde visited..."),
        _hit(2, 0.8, "75 Inf Bde inspection..."),
        # zero chunks for 5 PoK Bde — empty
    ]
    _apply_entity_quota(
        reranked=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=3, final_k=12,
    )
    after = _read("empty", "2")
    assert after == before + 1


def test_counter_measures_returned_slice_not_pre_cap_pool():
    """When final_k < len(selected), counter must reflect the LLM-facing slice.

    6 hits: 3 about 75 Inf Bde (scores 0.95/0.94/0.93), 3 about 5 PoK Bde
    (scores 0.50/0.45/0.40).  Per-entity floor=3 so both entities are quota'd
    into `selected` (all 6 hits).  final_k=3 caps the returned slice to the
    top-3 by score — all of which are 75 Inf Bde.  5 PoK Bde gets zero chunks
    in the LLM-facing slice, so the counter outcome must be 'empty', not 'full'.
    """
    hits = [
        _hit(1, 0.95, "75 Inf Bde 1"),
        _hit(2, 0.94, "75 Inf Bde 2"),
        _hit(3, 0.93, "75 Inf Bde 3"),
        _hit(4, 0.50, "5 PoK Bde 1"),
        _hit(5, 0.45, "5 PoK Bde 2"),
        _hit(6, 0.40, "5 PoK Bde 3"),
    ]
    before_full = _read("full", "2")
    before_empty = _read("empty", "2")
    out = _apply_entity_quota(
        reranked=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=3, final_k=3,  # cap at 3 — drops all 5 PoK hits
    )
    after_full = _read("full", "2")
    after_empty = _read("empty", "2")
    # Sanity: returned slice has 3 chunks, all 75 Inf Bde.
    assert len(out) == 3
    assert all("75 Inf Bde" in h.payload["text"] for h in out)
    # Counter must measure the slice, not the pre-cap pool.
    assert after_empty == before_empty + 1, "expected empty bucket bumped"
    assert after_full == before_full, "full bucket should NOT be bumped"


def test_chunk_mentioning_two_entities_attributed_to_first_match():
    """First-match-wins: a chunk that contains BOTH entity strings counts
    toward only the FIRST entity in the `entities` list. This documents
    the convention encoded by the `break` after attribution.
    """
    before_empty = _read("empty", "2")
    hits = [
        _hit(1, 0.9, "75 Inf Bde and 5 PoK Bde joint patrol"),
        _hit(2, 0.8, "75 Inf Bde and 5 PoK Bde drill"),
        _hit(3, 0.7, "75 Inf Bde and 5 PoK Bde meeting"),
    ]
    _apply_entity_quota(
        reranked=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=2, final_k=3,
    )
    # All 3 chunks attribute to first entity ("75 Inf Bde"); second
    # entity ("5 PoK Bde") gets 0 → outcome=empty.
    # If first-match weren't enforced, both would attribute to both →
    # outcome=full, which would be wrong.
    after_empty = _read("empty", "2")
    assert after_empty == before_empty + 1


def test_counter_fires_on_mmr_success_path():
    """Multi-entity + MMR-on must still bump the counter (regression for the
    gap discovered 2026-05-04 where the counter was wired only into
    _apply_entity_quota, which MMR-on KBs bypass).
    """
    from ext.services.chat_rag_bridge import _bump_coverage_counter

    before_full = _read("full", "2")
    hits = [
        _hit(1, 0.9, "75 Inf Bde visited"),
        _hit(2, 0.8, "75 Inf Bde inspection"),
        _hit(3, 0.7, "75 Inf Bde 03 Apr"),
        _hit(4, 0.6, "5 PoK Bde practice"),
        _hit(5, 0.5, "5 PoK Bde rotation"),
        _hit(6, 0.4, "5 PoK Bde meeting"),
    ]
    _bump_coverage_counter(
        slice_=hits, entities=["75 Inf Bde", "5 PoK Bde"],
        per_entity_floor=3,
    )
    after_full = _read("full", "2")
    assert after_full == before_full + 1, "counter did not bump"


def test_helper_safe_to_call_with_empty_entities():
    """No-op when entities=[]; helper must not crash."""
    from ext.services.chat_rag_bridge import _bump_coverage_counter
    # Should be a no-op silently
    _bump_coverage_counter(slice_=[], entities=[], per_entity_floor=3)
