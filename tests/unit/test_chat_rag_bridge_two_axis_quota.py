"""Phase 3 / item 5 — bridge-side two-axis quota helper.

Mirrors `_apply_entity_quota` but attributes hits to (entity, subtopic)
cells using the per-KB subtopic_keywords table.
"""
from __future__ import annotations

from dataclasses import dataclass

from ext.services import chat_rag_bridge as bridge


@dataclass
class _Hit:
    id: int
    score: float
    payload: dict


def _make(text: str, hid: int, score: float):
    return _Hit(id=hid, score=score, payload={"text": text})


def test_attributes_to_correct_cell() -> None:
    reranked = [
        _make("75 Inf Bde visited 77 Mtn Fd Arty Regt", 1, 0.9),  # 75/visits
        _make("75 Inf Bde construction at Lipa", 2, 0.8),         # 75/construction
        _make("5 PoK Bde mov of CO Lt Col Rana", 3, 0.7),         # 5/visits (kw)
        _make("5 PoK Bde firing exercise", 4, 0.6),               # 5/operations
    ]
    out = bridge._apply_two_axis_quota(
        reranked=reranked,
        entities=["75 Inf Bde", "5 PoK Bde"],
        subtopics=["visits", "construction", "operations"],
        subtopic_keywords={
            "visits": ["visited", "visit", "vis", "mov of co"],
            "construction": ["construction", "constr", "bunker"],
            "operations": ["firing", "exercise", "operation"],
        },
        per_cell_floor=1,
        per_entity_floor=2,
        per_subtopic_floor=1,
        final_k=4,
    )
    ids = [h.id for h in out]
    assert ids == [1, 2, 3, 4]


def test_synonyms_for_entity_attribution() -> None:
    """Entity attribution honours the per-KB synonyms table — 5 PoK / 5 POK
    both attribute to the same entity cell."""
    reranked = [
        _make("5 POK Bde construction work", 1, 0.5),
    ]
    out = bridge._apply_two_axis_quota(
        reranked=reranked,
        entities=["5 PoK Bde"],
        subtopics=["construction"],
        subtopic_keywords={"construction": ["construction"]},
        synonyms=[["5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde"]],
        per_cell_floor=1, per_entity_floor=1, per_subtopic_floor=1, final_k=1,
    )
    assert [h.id for h in out] == [1]


def test_empty_inputs_return_top_k_unchanged() -> None:
    """No entities or no subtopics — bypass quota, return reranked[:final_k]."""
    reranked = [_make("x", 1, 0.9), _make("y", 2, 0.8)]
    out = bridge._apply_two_axis_quota(
        reranked=reranked,
        entities=[],
        subtopics=[],
        subtopic_keywords={},
        per_cell_floor=1, per_entity_floor=1, per_subtopic_floor=1, final_k=1,
    )
    assert [h.id for h in out] == [1]
