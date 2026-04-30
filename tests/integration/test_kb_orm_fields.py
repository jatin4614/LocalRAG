"""ORM-level tests for KB schema fields added by migrations.

Covers:
  * KBDocument.doc_summary    (migration 008)
  * KBSubtag.deleted_at        (migration 014)
  * KBAccess.deleted_at        (migration 015)

These tests round-trip values through the SQLAlchemy ORM to prove that
the model fields are wired up correctly (not just that the columns
exist in Postgres).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text

from ext.db.models.kb import KBAccess, KBDocument, KBSubtag, KnowledgeBase

pytestmark = pytest.mark.integration


async def _seed_user_and_kb(session, *, user_id: str = "1", kb_name: str = "K"):
    """Helper: insert a user + KB so we have FK targets for downstream rows."""
    await session.execute(
        text(
            "INSERT INTO users (id, email, password_hash, role) "
            "VALUES (:uid, :email, 'h', 'admin')"
        ),
        {"uid": int(user_id), "email": f"u{user_id}@x"},
    )
    kb = KnowledgeBase(name=kb_name, admin_id=user_id)
    session.add(kb)
    await session.flush()
    return kb


@pytest.mark.asyncio
async def test_kb_document_doc_summary_round_trips(session):
    kb = await _seed_user_and_kb(session)
    sub = KBSubtag(kb_id=kb.id, name="default")
    session.add(sub)
    await session.flush()

    doc = KBDocument(
        kb_id=kb.id,
        subtag_id=sub.id,
        filename="example.pdf",
        uploaded_by="1",
        ingest_status="done",
    )
    session.add(doc)
    await session.commit()

    # The fresh row has no summary yet.
    fresh = (
        await session.execute(select(KBDocument).where(KBDocument.id == doc.id))
    ).scalar_one()
    assert fresh.doc_summary is None

    # Populate via ORM, commit, reload.
    fresh.doc_summary = "A 2026 launch overview spanning ML platform + hiring."
    await session.commit()

    reloaded = (
        await session.execute(select(KBDocument).where(KBDocument.id == doc.id))
    ).scalar_one()
    assert reloaded.doc_summary == "A 2026 launch overview spanning ML platform + hiring."


@pytest.mark.asyncio
async def test_kb_subtag_deleted_at_setter(session):
    """Setter writes through; partial-index query path skips the tombstoned row."""
    kb = await _seed_user_and_kb(session)
    a = KBSubtag(kb_id=kb.id, name="alive")
    d = KBSubtag(kb_id=kb.id, name="dead")
    session.add_all([a, d])
    await session.commit()

    assert a.deleted_at is None
    assert d.deleted_at is None

    d.deleted_at = datetime.now(timezone.utc)
    await session.commit()

    # Only the live row passes the deleted_at filter.
    live = (
        await session.execute(
            select(KBSubtag).where(
                KBSubtag.kb_id == kb.id, KBSubtag.deleted_at.is_(None)
            )
        )
    ).scalars().all()
    assert {s.name for s in live} == {"alive"}

    # Both still exist if no filter.
    everything = (
        await session.execute(
            select(KBSubtag).where(KBSubtag.kb_id == kb.id)
        )
    ).scalars().all()
    assert {s.name for s in everything} == {"alive", "dead"}


@pytest.mark.asyncio
async def test_kb_access_deleted_at_setter(session):
    """Same pattern for kb_access — setter writes through."""
    kb = await _seed_user_and_kb(session)
    grant = KBAccess(kb_id=kb.id, user_id="42", access_type="read")
    session.add(grant)
    await session.commit()

    assert grant.deleted_at is None
    grant.deleted_at = datetime.now(timezone.utc)
    await session.commit()

    reloaded = (
        await session.execute(select(KBAccess).where(KBAccess.id == grant.id))
    ).scalar_one()
    assert reloaded.deleted_at is not None
