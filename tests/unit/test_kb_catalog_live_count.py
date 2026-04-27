"""B9 — KB catalog preamble must filter by ``ingest_status='done'``.

The catalog preamble in :func:`ext.services.chat_rag_bridge.retrieve_kb_sources`
generates the "KB N: M document(s) available" line that the chat model
relies on for metadata-intent queries (`how many documents do I have`).

Earlier audit: a chat answered "110 documents" while the operator's
manual psql count was higher. Investigation showed the SQL filtered only
on ``deleted_at IS NULL`` — so any pending / failed ingest would either
inflate the catalog (counted but unsearchable) or, after a partial
reingest, mask documents whose status flipped to ``failed``.

This test pins the contract: both the whole-KB and subtag-scoped
catalog queries MUST include ``ingest_status = 'done'``. Verifying
the source string keeps the test fast (no DB) and resilient to other
refactors of the retrieval pipeline.

Caching note: we also assert there is no caching layer between this
SQL and the database — every catalog request runs the SELECT in a
fresh ``async with _sessionmaker()`` block.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRIDGE_PATH = ROOT / "ext" / "services" / "chat_rag_bridge.py"


def _bridge_source() -> str:
    return BRIDGE_PATH.read_text(encoding="utf-8")


def test_whole_kb_catalog_filters_by_ingest_status_done() -> None:
    """The ``whole_kb_ids`` SELECT must filter ``ingest_status = 'done'``.

    Without this filter, pending / failed docs are counted as available,
    over-reporting the catalog size and misleading the LLM.
    """
    source = _bridge_source()
    # The whole-KB query is keyed by the literal ``WHERE kb_id = ANY(:ids)``
    # — find that block and confirm ``ingest_status = 'done'`` follows.
    needle = "WHERE kb_id = ANY(:ids)"
    assert needle in source, (
        f"catalog preamble whole-KB SELECT shape changed; expected "
        f"{needle!r} in {BRIDGE_PATH}"
    )
    # Locate the surrounding query and verify the ingest_status guard
    # appears within the same SQL string. Use the explicit substring
    # instead of regex for resilience against whitespace tweaks.
    idx = source.index(needle)
    # Take the next ~400 chars (covers the ORDER BY + closing paren) and
    # verify the filter is present.
    window = source[idx : idx + 400]
    assert "ingest_status = 'done'" in window, (
        "catalog preamble whole-KB SELECT must include "
        "``AND ingest_status = 'done'`` so pending/failed docs do NOT "
        "appear in the 'N document(s) available' count.\n"
        f"Got: {window!r}"
    )


def test_subtag_catalog_filters_by_ingest_status_done() -> None:
    """The subtag-scoped SELECT must also filter on ``ingest_status='done'``.

    The subtag query is built with adjacent string literals across multiple
    Python lines (so SQL whitespace is bunched). Anchor on the SELECT
    column list rather than the FROM clause so the test isn't sensitive
    to whether the JOIN sits on the same string-literal line.
    """
    source = _bridge_source()
    # Subtag query column list is the unique fingerprint of this SELECT.
    needle = "SELECT d.kb_id, d.subtag_id, t.name AS subtag_name, d.filename"
    assert needle in source, (
        f"catalog preamble subtag SELECT shape changed; expected "
        f"{needle!r} in {BRIDGE_PATH}"
    )
    idx = source.index(needle)
    window = source[idx : idx + 800]
    assert "d.ingest_status = 'done'" in window, (
        "catalog preamble subtag-scoped SELECT must include "
        "``AND d.ingest_status = 'done'`` so pending/failed docs do NOT "
        "appear in the per-subtag catalog count.\n"
        f"Got: {window!r}"
    )


def test_catalog_uses_live_session_no_cache() -> None:
    """Catalog SQL runs inside ``async with _sessionmaker() as _s`` — fresh
    DB session every chat invocation, no in-process cache between SELECT
    and Postgres.

    This is the contract that makes the count "live". If a future PR adds
    a TTL cache around the catalog (e.g. to save round-trips on metadata-
    heavy chats), this test forces the author to also weigh whether
    operators can tolerate a stale "N documents available" line.
    """
    source = _bridge_source()
    # The catalog block is preceded by the ``KB catalog preamble`` comment
    # banner and opens its own session. Grab a wide enough slice that we
    # see both the comment and the ``async with _sessionmaker``.
    banner = "--- KB catalog preamble ---"
    assert banner in source, (
        "catalog preamble comment banner missing; section may have been "
        "renamed or removed"
    )
    idx = source.index(banner)
    window = source[idx : idx + 4000]
    assert "async with _sessionmaker()" in window, (
        "catalog preamble no longer opens its own session — possible "
        "cache layer added; verify count freshness contract"
    )
