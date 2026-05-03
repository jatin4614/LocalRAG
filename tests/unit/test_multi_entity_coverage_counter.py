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
