"""Unit-test conftest.

Shared fixtures for ``tests/unit/``.

Notable fixtures:
* ``_disable_datetime_preamble`` — autouse, session-wide. ``chat_rag_bridge``
  prepends a ``CURRENT DATE AND TIME`` preamble at index 0 of the returned
  sources list when ``RAG_INJECT_DATETIME`` is unset (the production default
  is on). The preamble shifts every existing index by +1, which broke a batch
  of pre-existing unit tests that asserted on ``out[0]`` / ``len(out)``. The
  preamble is a product feature (so the LLM always knows the wall-clock date),
  but unit tests that mock retrieval don't care about it. Disable globally for
  unit tests so assertions match the underlying retrieval shape, not the
  decorated shape.

  Tests that specifically want to exercise the datetime preamble can opt back
  in with ``monkeypatch.setenv("RAG_INJECT_DATETIME", "1")`` inside the test
  body — autouse session fixtures run before per-test monkeypatch setup, so
  the per-test override wins.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_datetime_preamble(monkeypatch):
    """Suppress chat_rag_bridge's datetime preamble for unit-test stability.

    See module docstring for rationale.
    """
    monkeypatch.setenv("RAG_INJECT_DATETIME", "0")
    yield
