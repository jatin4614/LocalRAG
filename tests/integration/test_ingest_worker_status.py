"""Worker-side _update_doc_status: verifies failed-doc rows persist
their error_message so admins can see *why* an async ingest failed
(instead of an empty cell next to the 'failed' status).

Regression: prior to this fix the worker's failure path called
``_update_doc_status(doc_id, "failed")`` without an error message,
leaving kb_documents.error_message NULL even though the SSE stream
emitted the exception. Admins had to grep celery logs to learn the
cause.
"""
from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ext.workers.ingest_worker import _update_doc_status

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def _seeded_doc(engine):
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text(
            "INSERT INTO users (id, email, password_hash, role) "
            "VALUES (1, 'admin@x', 'h', 'admin') ON CONFLICT DO NOTHING"
        ))
        await s.execute(text(
            "INSERT INTO knowledge_bases (name, admin_id) "
            "VALUES ('K', '1') RETURNING id"
        ))
        kb_id = (await s.execute(text(
            "SELECT id FROM knowledge_bases WHERE name='K'"
        ))).scalar_one()
        sub_id = (await s.execute(text(
            "INSERT INTO kb_subtags (kb_id, name) VALUES (:k, 'main') RETURNING id"
        ), {"k": kb_id})).scalar_one()
        doc_id = (await s.execute(text(
            "INSERT INTO kb_documents (kb_id, subtag_id, filename, uploaded_by, ingest_status) "
            "VALUES (:k, :s, 'x.docx', '1', 'queued') RETURNING id"
        ), {"k": kb_id, "s": sub_id})).scalar_one()
        await s.commit()
    return doc_id


@pytest.mark.asyncio
async def test_update_doc_status_persists_error_message(engine, _seeded_doc, monkeypatch):
    url = engine.url.render_as_string(hide_password=False).replace(
        "postgresql+asyncpg://", "postgresql://"
    )
    monkeypatch.setenv("DATABASE_URL", url)

    await _update_doc_status(
        _seeded_doc, "failed",
        error_message="ResponseHandlingException: All connection attempts failed",
    )

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        row = (await s.execute(text(
            "SELECT ingest_status, error_message, chunk_count "
            "FROM kb_documents WHERE id = :i"
        ), {"i": _seeded_doc})).one()
    assert row.ingest_status == "failed"
    assert row.error_message == "ResponseHandlingException: All connection attempts failed"
    assert row.chunk_count == 0


@pytest.mark.asyncio
async def test_update_doc_status_done_clears_nothing(engine, _seeded_doc, monkeypatch):
    """Success path keeps existing error_message untouched (don't clobber
    diagnostics from a prior attempt that the operator may want to read)."""
    url = engine.url.render_as_string(hide_password=False).replace(
        "postgresql+asyncpg://", "postgresql://"
    )
    monkeypatch.setenv("DATABASE_URL", url)

    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        await s.execute(text(
            "UPDATE kb_documents SET error_message = 'old' WHERE id = :i"
        ), {"i": _seeded_doc})
        await s.commit()

    await _update_doc_status(_seeded_doc, "done", chunk_count=42)

    async with SessionLocal() as s:
        row = (await s.execute(text(
            "SELECT ingest_status, error_message, chunk_count "
            "FROM kb_documents WHERE id = :i"
        ), {"i": _seeded_doc})).one()
    assert row.ingest_status == "done"
    assert row.chunk_count == 42
    assert row.error_message == "old"
