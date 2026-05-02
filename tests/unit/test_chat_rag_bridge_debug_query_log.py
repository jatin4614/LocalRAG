"""Wave 2 (review §8.9): RAG_LOG_QUERY_TEXT — optional debug log fields.

When ``RAG_LOG_QUERY_TEXT`` is unset or "0" (the default), ``_log_rag_query``
must NOT include the user's query text or the per-chunk summary in the
emitted log payload. PII-sensitive: opt-in only.

When ``RAG_LOG_QUERY_TEXT=1``, the helper SHOULD additionally include:
  - ``query_text`` (truncated to 1 KB) — the resolved user query
  - ``chunks_summary`` — top-3 hits as ``[{chunk_id, score, filename}, ...]``

The fields must arrive via ``logger.info(..., extra={...})`` so the JSON
formatter (``_FallbackJsonFormatter`` or ``python-json-logger``) lifts
them into the structured payload.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from ext.services import chat_rag_bridge as bridge


@dataclass
class _Hit:
    id: int
    score: float
    payload: dict


def _capture_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "orgchat.rag_bridge"]


def test_debug_log_off_by_default_omits_query_text(monkeypatch, caplog):
    """With RAG_LOG_QUERY_TEXT unset, the rag_query log line must not
    contain the user's query string or chunks_summary."""
    monkeypatch.delenv("RAG_LOG_QUERY_TEXT", raising=False)
    caplog.set_level(logging.INFO, logger="orgchat.rag_bridge")
    bridge._log_rag_query(
        req_id="r1", intent="specific", kbs=[1, 2], hits=3, total_ms=42,
        query="my secret PII query about employee John Doe",
        hits_detail=[_Hit(id=11, score=0.91, payload={"filename": "report.pdf"})],
    )
    records = _capture_records(caplog)
    assert records, "expected at least one orgchat.rag_bridge record"
    msg = records[-1].getMessage()
    # The base payload is JSON-encoded into the message.
    payload = json.loads(msg)
    assert "query_text" not in payload, (
        "query_text must NOT be in payload when RAG_LOG_QUERY_TEXT is unset (PII)"
    )
    assert "chunks_summary" not in payload, (
        "chunks_summary must NOT be in payload when RAG_LOG_QUERY_TEXT is unset"
    )
    # Existing fields must remain.
    assert payload["event"] == "rag_query"
    assert payload["intent"] == "specific"
    assert payload["hits"] == 3
    # Also assert no extra= keys leaked onto the LogRecord.
    rec = records[-1]
    assert not hasattr(rec, "query_text")
    assert not hasattr(rec, "chunks_summary")


def test_debug_log_on_includes_query_text_and_chunks(monkeypatch, caplog):
    """With RAG_LOG_QUERY_TEXT=1, query_text + chunks_summary must be
    attached as extras (so the JSON formatter picks them up)."""
    monkeypatch.setenv("RAG_LOG_QUERY_TEXT", "1")
    caplog.set_level(logging.INFO, logger="orgchat.rag_bridge")
    hits = [
        _Hit(id=11, score=0.91, payload={"filename": "report.pdf", "doc_id": 5}),
        _Hit(id=22, score=0.88, payload={"filename": "memo.docx"}),
        _Hit(id=33, score=0.84, payload={"filename": "minutes.md"}),
        _Hit(id=44, score=0.71, payload={"filename": "noise.txt"}),  # 4th, dropped
    ]
    bridge._log_rag_query(
        req_id="r2", intent="global", kbs=[1], hits=4, total_ms=120,
        query="quarterly revenue summary",
        hits_detail=hits,
    )
    records = _capture_records(caplog)
    assert records
    rec = records[-1]
    # extra={} fields should land as record attributes.
    assert getattr(rec, "query_text", None) == "quarterly revenue summary"
    cs = getattr(rec, "chunks_summary", None)
    assert isinstance(cs, list) and len(cs) == 3, "top-3 only"
    assert cs[0]["chunk_id"] == 11
    assert cs[0]["score"] == pytest.approx(0.91)
    assert cs[0]["filename"] == "report.pdf"
    assert cs[2]["chunk_id"] == 33


def test_debug_log_query_text_truncated_to_1kb(monkeypatch, caplog):
    """query_text longer than 1 KB must be truncated."""
    monkeypatch.setenv("RAG_LOG_QUERY_TEXT", "1")
    caplog.set_level(logging.INFO, logger="orgchat.rag_bridge")
    huge = "x" * 5000
    bridge._log_rag_query(
        req_id="r3", intent="specific", kbs=[1], hits=0, total_ms=10,
        query=huge,
        hits_detail=[],
    )
    rec = _capture_records(caplog)[-1]
    qt = getattr(rec, "query_text", "")
    assert len(qt) <= 1024, f"expected ≤1024 bytes, got {len(qt)}"


def test_debug_log_off_when_flag_is_explicit_zero(monkeypatch, caplog):
    """``RAG_LOG_QUERY_TEXT=0`` is equivalent to unset."""
    monkeypatch.setenv("RAG_LOG_QUERY_TEXT", "0")
    caplog.set_level(logging.INFO, logger="orgchat.rag_bridge")
    bridge._log_rag_query(
        req_id="r4", intent="specific", kbs=[1], hits=1, total_ms=5,
        query="ignored",
        hits_detail=[_Hit(id=1, score=0.5, payload={"filename": "a.txt"})],
    )
    rec = _capture_records(caplog)[-1]
    assert not hasattr(rec, "query_text")
    assert not hasattr(rec, "chunks_summary")


def test_debug_log_call_without_optional_args_still_works(monkeypatch, caplog):
    """Backward compat: callers that don't pass query / hits_detail must
    still be able to invoke ``_log_rag_query`` without TypeError."""
    monkeypatch.setenv("RAG_LOG_QUERY_TEXT", "1")
    caplog.set_level(logging.INFO, logger="orgchat.rag_bridge")
    bridge._log_rag_query(
        req_id="r5", intent="metadata", kbs=[1], hits=0, total_ms=2,
    )
    rec = _capture_records(caplog)[-1]
    # No query → no query_text attr (we don't fabricate "" out of None).
    assert not hasattr(rec, "query_text") or rec.query_text == ""
