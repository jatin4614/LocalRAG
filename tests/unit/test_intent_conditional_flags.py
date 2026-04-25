"""Phase 2.2 — intent-conditional MMR / context_expand defaults.

The intent classifier returns one of four labels (``specific``,
``specific_date``, ``global``, ``metadata``). Each label implies a
different ideal pipeline shape:

* ``specific`` / ``specific_date`` — single-fact lookup. Want adjacent
  chunks for context (``RAG_CONTEXT_EXPAND=1``); diversification (``MMR``)
  would dilute the single best hit, so OFF.
* ``global`` — broad aggregation across many docs. Diversity matters
  (``MMR=1``); context expansion just inflates each already-broad
  summary, so OFF.
* ``metadata`` — answered by the catalog preamble alone; neither MMR
  nor expand contributes, so both OFF.

Per-KB ``rag_config`` always wins: an admin who explicitly stamped a
flag on a KB had a reason, and we don't second-guess them.
"""
from __future__ import annotations

from ext.services.chat_rag_bridge import resolve_intent_flags


def test_specific_intent_enables_expand_not_mmr():
    f = resolve_intent_flags(intent="specific", per_kb_overrides={})
    assert f["RAG_MMR"] == "0"
    assert f["RAG_CONTEXT_EXPAND"] == "1"


def test_specific_date_intent_treated_like_specific():
    f = resolve_intent_flags(intent="specific_date", per_kb_overrides={})
    assert f["RAG_MMR"] == "0"
    assert f["RAG_CONTEXT_EXPAND"] == "1"


def test_global_intent_enables_mmr_not_expand():
    f = resolve_intent_flags(intent="global", per_kb_overrides={})
    assert f["RAG_MMR"] == "1"
    assert f["RAG_CONTEXT_EXPAND"] == "0"


def test_metadata_intent_disables_both():
    f = resolve_intent_flags(intent="metadata", per_kb_overrides={})
    assert f["RAG_MMR"] == "0"
    assert f["RAG_CONTEXT_EXPAND"] == "0"


def test_per_kb_override_wins():
    """If a KB has RAG_MMR=0 in its rag_config, it must win over intent default."""
    f = resolve_intent_flags(
        intent="global",
        per_kb_overrides={"RAG_MMR": "0"},
    )
    assert f["RAG_MMR"] == "0"


def test_unknown_intent_defaults_to_specific():
    f = resolve_intent_flags(intent="unknown_thing", per_kb_overrides={})
    assert f["RAG_CONTEXT_EXPAND"] == "1"
    assert f["RAG_MMR"] == "0"
