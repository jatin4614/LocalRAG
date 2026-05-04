"""Integration tests for scripts/edit_kb_subtopics.py.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.6
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def kb98_clean():
    import asyncpg, asyncio

    async def setup() -> None:
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("DELETE FROM knowledge_bases WHERE id = 98")
            await conn.execute(
                "INSERT INTO knowledge_bases (id, name, admin_id, rag_config, synonyms) "
                "VALUES (98, 'subtopic-test', '1', '{}'::jsonb, '[]'::jsonb)"
            )
        finally:
            await conn.close()

    asyncio.run(setup())
    yield 98
    async def teardown() -> None:
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("DELETE FROM knowledge_bases WHERE id = 98")
        finally:
            await conn.close()
    asyncio.run(teardown())


@pytest.mark.integration
def test_load_replaces_subtopic_keywords(kb98_clean) -> None:
    payload = {"visits": ["visit", "vis"], "operations": ["operation", "ex"]}
    p = subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--load", "-"],
        input=json.dumps(payload),
        capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0, p.stderr

    import asyncpg, asyncio
    async def check():
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            row = await conn.fetchrow(
                "SELECT rag_config FROM knowledge_bases WHERE id = 98"
            )
        finally:
            await conn.close()
        rc = row["rag_config"] or {}
        if isinstance(rc, str):  # asyncpg returns JSONB as str by default
            rc = json.loads(rc)
        return dict(rc)
    rc = asyncio.run(check())
    assert rc.get("subtopic_keywords") == payload


@pytest.mark.integration
def test_list_prints_current_table(kb98_clean) -> None:
    subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--load", "-"],
        input='{"visits": ["v"]}',
        check=True, capture_output=True, text=True,
    )
    p = subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--list"],
        capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0
    assert "visits" in p.stdout
    assert '"v"' in p.stdout


@pytest.mark.integration
def test_add_merges_new_keys(kb98_clean) -> None:
    subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--load", "-"],
        input='{"visits": ["v"]}',
        check=True, capture_output=True, text=True,
    )
    p = subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "98", "--add", '{"operations": ["o", "op"]}'],
        capture_output=True, text=True, timeout=30,
    )
    assert p.returncode == 0, p.stderr

    import asyncpg, asyncio
    async def check():
        url = os.environ["DATABASE_URL"].replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            row = await conn.fetchrow(
                "SELECT rag_config FROM knowledge_bases WHERE id = 98"
            )
        finally:
            await conn.close()
        rc = row["rag_config"] or {}
        if isinstance(rc, str):
            rc = json.loads(rc)
        return dict(rc)
    rc = asyncio.run(check())
    assert rc["subtopic_keywords"]["visits"] == ["v"]
    assert rc["subtopic_keywords"]["operations"] == ["o", "op"]


@pytest.mark.integration
def test_missing_kb_returns_nonzero(kb98_clean) -> None:
    p = subprocess.run(
        [".venv/bin/python", "scripts/edit_kb_subtopics.py",
         "--kb", "97777", "--list"],
        capture_output=True, text=True,
    )
    assert p.returncode != 0
    assert "not found" in p.stderr.lower()
