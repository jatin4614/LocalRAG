"""Unit tests for ``ext.services.embed_preflight.preflight_embedder``.

Bug-fix campaign §3.6 — TEI ``GET /info`` model_id vs ``EMBED_MODEL``.
Mirror of ``test_chat_model_preflight``-style coverage.
"""
from __future__ import annotations

import httpx
import pytest

from ext.services import embed_preflight as ep
from ext.services.metrics import embed_model_mismatch_total


def _counter_value():
    """Prometheus Counter doesn't expose .get; sum sample values."""
    try:
        # In test env metrics module may stub Counter — guard for that.
        return embed_model_mismatch_total._value.get()  # type: ignore[attr-defined]
    except AttributeError:
        # Stubbed Counter: no internal value, so fall back to inc-count
        # via .labels-style interface — for simplicity, return 0.
        return 0


def _mock_transport(status: int = 200, body: dict | None = None):
    body = body if body is not None else {"model_id": "BAAI/bge-m3"}
    def handler(req):
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler)


def test_preflight_match_does_not_bump_counter(monkeypatch):
    """When TEI's model_id matches EMBED_MODEL, no counter increment, no
    warning."""
    before = _counter_value()
    ep.preflight_embedder(
        tei_url="http://tei",
        embed_model="BAAI/bge-m3",
        transport=_mock_transport(body={"model_id": "BAAI/bge-m3"}),
    )
    after = _counter_value()
    assert after == before  # no bump


def test_preflight_mismatch_bumps_counter(monkeypatch):
    """When TEI's model_id doesn't match, the counter increments."""
    before = _counter_value()
    ep.preflight_embedder(
        tei_url="http://tei",
        embed_model="BAAI/bge-m3",
        transport=_mock_transport(body={"model_id": "Snowflake/arctic-embed-l"}),
    )
    after = _counter_value()
    assert after == before + 1


def test_preflight_skips_when_env_unset(monkeypatch):
    """No EMBED_MODEL → soft-noop, no exception, no counter bump."""
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    before = _counter_value()
    # No transport needed — preflight should not even hit the network.
    ep.preflight_embedder(tei_url="http://tei")
    assert _counter_value() == before


def test_preflight_endpoint_unreachable_does_not_crash(monkeypatch):
    """5xx / connect error / malformed json → log + return cleanly,
    but the mismatch counter is NOT bumped (the check was inconclusive,
    not failed)."""
    before = _counter_value()
    ep.preflight_embedder(
        tei_url="http://tei",
        embed_model="BAAI/bge-m3",
        transport=_mock_transport(status=503, body={}),
    )
    assert _counter_value() == before


def test_preflight_no_model_id_in_response(monkeypatch):
    """TEI returns 200 but model_id field is absent → log + skip, no
    counter bump (cannot determine mismatch)."""
    before = _counter_value()
    ep.preflight_embedder(
        tei_url="http://tei",
        embed_model="BAAI/bge-m3",
        transport=_mock_transport(body={"version": "1.0", "max_input_length": 8192}),
    )
    assert _counter_value() == before
