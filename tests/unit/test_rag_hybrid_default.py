"""Unit tests for the ``RAG_HYBRID`` default-on semantics.

As of 2026-04-19 the default flipped from off to on. These tests pin down the
permissive parsing behaviour: only the literal string ``"0"`` disables hybrid;
every other value — unset, empty, ``"1"``, ``"true"``, ``"yes"``, etc. — is
treated as **on**. Both retriever and ingest share the same semantics.
"""
from __future__ import annotations

import pytest

from ext.services.ingest import _hybrid_enabled as _ingest_hybrid_enabled
from ext.services.retriever import _hybrid_enabled as _retriever_hybrid_enabled


# ---------- retriever._hybrid_enabled ---------------------------------------


def test_retriever_default_when_unset_is_on(monkeypatch) -> None:
    """No RAG_HYBRID in env → default on (new behaviour as of 2026-04-19)."""
    monkeypatch.delenv("RAG_HYBRID", raising=False)
    assert _retriever_hybrid_enabled() is True


def test_retriever_explicit_zero_is_off(monkeypatch) -> None:
    """The literal string "0" is the only disable value."""
    monkeypatch.setenv("RAG_HYBRID", "0")
    assert _retriever_hybrid_enabled() is False


def test_retriever_explicit_one_is_on(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "1")
    assert _retriever_hybrid_enabled() is True


def test_retriever_permissive_true_string_is_on(monkeypatch) -> None:
    """Permissive semantics: non-"0" values are all "on" — matches user intent."""
    monkeypatch.setenv("RAG_HYBRID", "true")
    assert _retriever_hybrid_enabled() is True


def test_retriever_empty_string_is_on(monkeypatch) -> None:
    """Empty string != "0" → defaults to on (flag present but blank)."""
    monkeypatch.setenv("RAG_HYBRID", "")
    assert _retriever_hybrid_enabled() is True


# ---------- ingest._hybrid_enabled (mirror semantics) -----------------------


def test_ingest_default_when_unset_is_on(monkeypatch) -> None:
    monkeypatch.delenv("RAG_HYBRID", raising=False)
    assert _ingest_hybrid_enabled() is True


def test_ingest_explicit_zero_is_off(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "0")
    assert _ingest_hybrid_enabled() is False


def test_ingest_explicit_one_is_on(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "1")
    assert _ingest_hybrid_enabled() is True


def test_ingest_permissive_true_string_is_on(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "true")
    assert _ingest_hybrid_enabled() is True


def test_ingest_empty_string_is_on(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "")
    assert _ingest_hybrid_enabled() is True


# ---------- parity: retriever and ingest must agree -------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),           # unset → on
        ("", True),             # empty → on
        ("0", False),           # the one disable value
        ("1", True),
        ("true", True),
        ("yes", True),
        ("False", True),        # NOTE: permissive — only literal "0" is off
    ],
)
def test_retriever_and_ingest_agree(monkeypatch, value, expected) -> None:
    """Both callers must read the flag identically so ingest/retrieve stay in sync."""
    if value is None:
        monkeypatch.delenv("RAG_HYBRID", raising=False)
    else:
        monkeypatch.setenv("RAG_HYBRID", value)
    assert _retriever_hybrid_enabled() is expected
    assert _ingest_hybrid_enabled() is expected
