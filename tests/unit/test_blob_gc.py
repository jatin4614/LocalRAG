"""Unit tests for ext.services.blob_gc.run_gc.

Uses aiosqlite + an in-memory BlobStore (real FS, temp dir) + a stub
VectorStore so no Qdrant is required. The DB schema is created from the
SQLAlchemy metadata — migration 005 is mirrored by the updated KBDocument
model.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("aiosqlite")

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ext.db.base import Base
from ext.db.models.kb import KBDocument, KBSubtag, KnowledgeBase
# Pull compat models so users/groups tables exist for FK constraints.
from ext.db.models.compat import Group as _Group, User as _User  # noqa: F401
from ext.services.blob_gc import retention_days, run_gc
from ext.services.blob_store import BlobStore


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubVectorStore:
    """Records calls to delete_by_doc; returns configurable outcome."""

    def __init__(self, *, raise_on: set[int] | None = None, zero_on: set[int] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self._raise_on = raise_on or set()
        self._zero_on = zero_on or set()

    async def delete_by_doc(self, collection: str, doc_id: int) -> int:
        self.calls.append((collection, doc_id))
        if doc_id in self._raise_on:
            raise RuntimeError(f"simulated Qdrant failure on doc={doc_id}")
        if doc_id in self._zero_on:
            return 0
        return 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("INSERT INTO users(id,email,password_hash,role) VALUES (1,'a@x','stub','admin')"))
    yield eng
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        yield s


@pytest.fixture
def blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(str(tmp_path / "blobs"))


async def _seed_doc(
    session: AsyncSession,
    *,
    doc_id: int,
    kb_id: int,
    subtag_id: int,
    deleted_at: datetime | None,
    blob_sha: str | None,
) -> KBDocument:
    """Insert a KBDocument (and its KB / subtag if needed). Returns the doc."""
    if (await session.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))).scalar_one_or_none() is None:
        session.add(KnowledgeBase(id=kb_id, name=f"kb{kb_id}", admin_id="1"))
        await session.flush()
    if (await session.execute(select(KBSubtag).where(KBSubtag.id == subtag_id))).scalar_one_or_none() is None:
        session.add(KBSubtag(id=subtag_id, kb_id=kb_id, name="sub"))
        await session.flush()
    doc = KBDocument(
        id=doc_id,
        kb_id=kb_id,
        subtag_id=subtag_id,
        filename=f"doc{doc_id}.pdf",
        uploaded_by="1",
        deleted_at=deleted_at,
        blob_sha=blob_sha,
    )
    session.add(doc)
    await session.commit()
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retention_days_env(monkeypatch) -> None:
    monkeypatch.setenv("RAG_BLOB_RETENTION_DAYS", "7")
    assert retention_days() == 7
    monkeypatch.setenv("RAG_BLOB_RETENTION_DAYS", "not-a-number")
    assert retention_days() == 30
    monkeypatch.setenv("RAG_BLOB_RETENTION_DAYS", "-3")
    assert retention_days() == 30
    monkeypatch.delenv("RAG_BLOB_RETENTION_DAYS", raising=False)
    assert retention_days() == 30


@pytest.mark.asyncio
async def test_dry_run_makes_no_changes(session: AsyncSession, blob_store: BlobStore) -> None:
    sha = blob_store.write(b"hello world")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    await _seed_doc(session, doc_id=1, kb_id=10, subtag_id=100, deleted_at=old, blob_sha=sha)

    vs = StubVectorStore()
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["rows_processed"] == 1
    assert summary["blobs_deleted"] == 1  # counted, but not actually deleted
    # No mutations actually happened.
    assert blob_store.exists(sha), "blob must still exist in dry-run"
    assert vs.calls == [], "dry-run must not call vector_store"
    remaining = (await session.execute(select(KBDocument))).scalars().all()
    assert len(remaining) == 1, "row must still exist in dry-run"


@pytest.mark.asyncio
async def test_apply_deletes_blob_and_row_and_calls_qdrant(session: AsyncSession, blob_store: BlobStore) -> None:
    sha = blob_store.write(b"deleteme")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    await _seed_doc(session, doc_id=7, kb_id=11, subtag_id=110, deleted_at=old, blob_sha=sha)

    vs = StubVectorStore()
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=False,
    )
    assert summary["dry_run"] is False
    assert summary["rows_processed"] == 1
    assert summary["blobs_deleted"] == 1
    assert summary["qdrant_points_deleted"] == 1
    assert summary["rows_deleted"] == 1
    assert not blob_store.exists(sha), "blob must be deleted"
    assert vs.calls == [("kb_11", 7)]
    remaining = (await session.execute(select(KBDocument))).scalars().all()
    assert remaining == []


@pytest.mark.asyncio
async def test_rows_within_retention_are_skipped(session: AsyncSession, blob_store: BlobStore) -> None:
    sha_old = blob_store.write(b"very old")
    sha_fresh = blob_store.write(b"just deleted")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    fresh = datetime.now(timezone.utc) - timedelta(days=1)
    await _seed_doc(session, doc_id=1, kb_id=20, subtag_id=200, deleted_at=old, blob_sha=sha_old)
    await _seed_doc(session, doc_id=2, kb_id=20, subtag_id=200, deleted_at=fresh, blob_sha=sha_fresh)

    vs = StubVectorStore()
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=False,
    )
    assert summary["rows_processed"] == 1
    assert summary["rows_deleted"] == 1
    assert not blob_store.exists(sha_old), "old blob deleted"
    assert blob_store.exists(sha_fresh), "recent-delete blob must be preserved"
    remaining_ids = sorted(
        r.id for r in (await session.execute(select(KBDocument))).scalars().all()
    )
    assert remaining_ids == [2]


@pytest.mark.asyncio
async def test_active_rows_are_never_touched(session: AsyncSession, blob_store: BlobStore) -> None:
    """deleted_at IS NULL → ineligible, regardless of blob_sha."""
    sha = blob_store.write(b"live doc")
    await _seed_doc(session, doc_id=42, kb_id=30, subtag_id=300, deleted_at=None, blob_sha=sha)

    vs = StubVectorStore()
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=False,
    )
    assert summary["rows_processed"] == 0
    assert blob_store.exists(sha)
    assert vs.calls == []
    remaining = (await session.execute(select(KBDocument))).scalars().all()
    assert len(remaining) == 1


@pytest.mark.asyncio
async def test_missing_blob_file_is_counted_and_does_not_crash(
    session: AsyncSession, blob_store: BlobStore
) -> None:
    """Row says blob_sha=X but X is not on disk — counts as blobs_missing."""
    bogus_sha = "deadbeef" * 8  # 64 hex chars, file absent
    assert not blob_store.exists(bogus_sha)
    old = datetime.now(timezone.utc) - timedelta(days=60)
    await _seed_doc(session, doc_id=99, kb_id=40, subtag_id=400, deleted_at=old, blob_sha=bogus_sha)

    vs = StubVectorStore()
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=False,
    )
    assert summary["rows_processed"] == 1
    assert summary["blobs_deleted"] == 0
    assert summary["blobs_missing"] == 1
    assert summary["rows_deleted"] == 1
    # Row still got cleaned up.
    remaining = (await session.execute(select(KBDocument))).scalars().all()
    assert remaining == []


@pytest.mark.asyncio
async def test_null_blob_sha_counted_as_blobs_none(session: AsyncSession, blob_store: BlobStore) -> None:
    """Legacy / sync-ingest rows have no blob_sha; they still get GC'd, just
    nothing to free on disk."""
    old = datetime.now(timezone.utc) - timedelta(days=60)
    await _seed_doc(session, doc_id=5, kb_id=50, subtag_id=500, deleted_at=old, blob_sha=None)

    vs = StubVectorStore()
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=False,
    )
    assert summary["rows_processed"] == 1
    assert summary["blobs_deleted"] == 0
    assert summary["blobs_missing"] == 0
    assert summary["blobs_none"] == 1
    assert summary["rows_deleted"] == 1


@pytest.mark.asyncio
async def test_qdrant_error_increments_counter_and_continues(
    session: AsyncSession, blob_store: BlobStore
) -> None:
    sha1 = blob_store.write(b"doc1")
    sha2 = blob_store.write(b"doc2")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    await _seed_doc(session, doc_id=1, kb_id=60, subtag_id=600, deleted_at=old, blob_sha=sha1)
    await _seed_doc(session, doc_id=2, kb_id=60, subtag_id=600, deleted_at=old, blob_sha=sha2)

    vs = StubVectorStore(raise_on={1})
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=False,
    )
    assert summary["rows_processed"] == 2
    assert summary["blobs_deleted"] == 2
    assert summary["qdrant_errors"] == 1
    assert summary["qdrant_points_deleted"] == 1
    # Blobs freed regardless of Qdrant's outcome.
    assert not blob_store.exists(sha1)
    assert not blob_store.exists(sha2)
    # Both rows removed from DB.
    remaining = (await session.execute(select(KBDocument))).scalars().all()
    assert remaining == []


@pytest.mark.asyncio
async def test_qdrant_zero_return_counts_as_error(session: AsyncSession, blob_store: BlobStore) -> None:
    """delete_by_doc returning 0 signals best-effort failure inside VectorStore."""
    sha = blob_store.write(b"doc")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    await _seed_doc(session, doc_id=1, kb_id=70, subtag_id=700, deleted_at=old, blob_sha=sha)

    vs = StubVectorStore(zero_on={1})
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=False,
    )
    assert summary["qdrant_errors"] == 1
    assert summary["qdrant_points_deleted"] == 0
    # Row still cleaned up.
    assert (await session.execute(select(KBDocument))).scalars().all() == []


@pytest.mark.asyncio
async def test_limit_caps_batch_size(session: AsyncSession, blob_store: BlobStore) -> None:
    old = datetime.now(timezone.utc) - timedelta(days=60)
    shas = []
    for i in range(5):
        sha = blob_store.write(f"doc-{i}".encode())
        shas.append(sha)
        await _seed_doc(session, doc_id=i + 1, kb_id=80, subtag_id=800, deleted_at=old, blob_sha=sha)

    vs = StubVectorStore()
    summary = await run_gc(
        session=session,
        blob_store=blob_store,
        vector_store=vs,
        retention_days=30,
        dry_run=False,
        limit=2,
    )
    assert summary["rows_processed"] == 2
    assert summary["rows_deleted"] == 2
    remaining = (await session.execute(select(KBDocument))).scalars().all()
    assert len(remaining) == 3
