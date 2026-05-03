"""Unit tests for the TEI circuit breaker integration in TEIEmbedder.

Bug-fix campaign §3.5 — wraps the TEI embed call with the existing
per-key circuit breaker (``ext.services.circuit_breaker.breaker_for``)
under key ``"tei"``. When TEI fails ``RAG_CB_FAIL_THRESHOLD`` times in
``RAG_CB_WINDOW_SEC``, the breaker opens for the cooldown window and
``embed()`` raises ``CircuitOpenError`` instead of hammering a known-
broken endpoint. Callers fail-open per CLAUDE.md §1.2.

Behind ``RAG_CB_TEI_ENABLED=0`` (default OFF for the first deploy).
"""
from __future__ import annotations

import os

import httpx
import pytest

from ext.services.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    _BREAKERS,
)
from ext.services.embedder import TEIEmbedder


def _reset_breakers():
    """Drop module-level breaker registry between tests."""
    _BREAKERS.clear()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    _reset_breakers()
    # Default: real CB enabled, TEI breaker enabled.
    monkeypatch.setenv("RAG_CIRCUIT_BREAKER_ENABLED", "1")
    monkeypatch.setenv("RAG_CB_TEI_ENABLED", "1")
    monkeypatch.setenv("RAG_CB_FAIL_THRESHOLD", "3")
    monkeypatch.setenv("RAG_CB_WINDOW_SEC", "60")
    monkeypatch.setenv("RAG_CB_COOLDOWN_SEC", "30")
    # Disable retry decorator so each call counts as ONE failure
    # (otherwise tenacity multiplies the failure count).
    monkeypatch.setenv("RAG_TENACITY_RETRY", "0")
    yield
    _reset_breakers()


def _make_failing_transport(status_code: int = 503) -> httpx.MockTransport:
    """A transport that always returns the given status — for failure-mode tests."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "tei dead"})
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold_failures(monkeypatch):
    """N failed TEI calls → breaker opens → next call raises CircuitOpenError.

    Drives N failed embeds against a mock transport, then asserts that
    the next embed raises the breaker exception WITHOUT hitting TEI.
    """
    transport = _make_failing_transport()
    emb = TEIEmbedder(base_url="http://tei", transport=transport)

    fail_threshold = 3

    # First N failures should propagate the underlying httpx error.
    for _ in range(fail_threshold):
        with pytest.raises(httpx.HTTPStatusError):
            await emb.embed(["hello"])

    # Breaker should now be open. The next call must raise CircuitOpenError
    # without touching the (still-broken) transport.
    with pytest.raises(CircuitOpenError):
        await emb.embed(["world"])


@pytest.mark.asyncio
async def test_breaker_disabled_when_flag_off(monkeypatch):
    """RAG_CB_TEI_ENABLED=0 means failures do NOT open the TEI breaker —
    every call goes straight to the (failing) transport.
    """
    monkeypatch.setenv("RAG_CB_TEI_ENABLED", "0")
    transport = _make_failing_transport()
    emb = TEIEmbedder(base_url="http://tei", transport=transport)

    # Even after many failures, no CircuitOpenError. Each call surfaces
    # the underlying TEI error.
    for _ in range(5):
        with pytest.raises(httpx.HTTPStatusError):
            await emb.embed(["hello"])


@pytest.mark.asyncio
async def test_success_resets_failures(monkeypatch):
    """A success after one failure should reset the failure counter so
    the breaker doesn't trip on intermittent blips.

    2026-05-03 update: the embedder gained a retry-with-halving
    redundancy layer (``_embed_with_redundancy``) that absorbs single
    transient blips BEFORE the breaker sees them. So the test no longer
    drives one failed user-call + one successful user-call — instead it
    asserts that intermittent failures don't surface to the user OR
    open the breaker. The breaker's failure-reset semantics remain
    unchanged; this test pins the higher-order property "intermittent
    blips don't trip the breaker."
    """
    # Disable the test-fixture override — let the new redundancy layer
    # actually retry instead of single-shotting.
    monkeypatch.setenv("RAG_TENACITY_RETRY", "1")

    # Build a transport that fails once, then succeeds. The redundancy
    # layer will retry the 503 at the same batch size and pick up the 200.
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(503, json={})
        return httpx.Response(200, json=[[0.1, 0.2, 0.3]])

    transport = httpx.MockTransport(handler)
    emb = TEIEmbedder(base_url="http://tei", transport=transport)

    # The redundancy layer retries the 503 → embed succeeds on the same
    # user call. Net effect: zero user-visible failures from one TEI blip.
    out = await emb.embed(["x"])
    assert out == [[0.1, 0.2, 0.3]]
    assert state["calls"] == 2  # raw 503 + raw 200

    # Subsequent calls still work; breaker stayed closed (one redundancy-
    # absorbed blip is exactly one breaker success per 2026-05-03 contract).
    out2 = await emb.embed(["y"])
    assert out2 == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_breaker_uses_tei_key(monkeypatch):
    """Lock the breaker key as ``tei`` so dashboards / metrics / logs
    can find it. Operators grep for this name when the breaker trips.
    """
    transport = _make_failing_transport()
    emb = TEIEmbedder(base_url="http://tei", transport=transport)

    # Drive one failure to ensure a breaker is registered.
    with pytest.raises(httpx.HTTPStatusError):
        await emb.embed(["x"])

    assert "tei" in _BREAKERS
    assert isinstance(_BREAKERS["tei"], CircuitBreaker)
