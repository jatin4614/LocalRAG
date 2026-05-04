"""Integration test for scripts/wipe_and_reingest.py — backup phase.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §7.2
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def env():
    e = os.environ.copy()

    # Parse compose/.env once to extract all needed vars.
    compose_vars: dict[str, str] = {}
    try:
        env_text = open("compose/.env").read()
        for line in env_text.splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                compose_vars[k.strip()] = v.strip()
    except FileNotFoundError:
        pass

    if "DATABASE_URL" not in e:
        # Resolve the live Postgres container IP (not host-exposed on 5432)
        # and extract the password from compose/.env — same pattern as
        # test_edit_kb_synonyms_cli.py, extended for the unexposed-port case.
        pw = compose_vars.get("POSTGRES_PASSWORD")
        if pw:
            # Postgres is not host-mapped; resolve its container IP.
            ip_result = subprocess.run(
                [
                    "docker", "inspect", "orgchat-postgres",
                    "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                ],
                capture_output=True, text=True,
            )
            container_ip = ip_result.stdout.strip()
            if container_ip:
                e["DATABASE_URL"] = (
                    f"postgresql://orgchat:{pw}@{container_ip}:5432/orgchat"
                )
            else:
                # Fallback: try localhost (works if port is bound to host)
                e["DATABASE_URL"] = (
                    f"postgresql://orgchat:{pw}@localhost:5432/orgchat"
                )

    # Inject Qdrant credentials so the wipe phase can reach the cluster.
    if "QDRANT_API_KEY" not in e and "QDRANT_API_KEY" in compose_vars:
        e["QDRANT_API_KEY"] = compose_vars["QDRANT_API_KEY"]
    if "QDRANT_URL" not in e:
        # Qdrant is not host-mapped on 6333; resolve container IP.
        qdrant_ip_result = subprocess.run(
            [
                "docker", "inspect", "orgchat-qdrant",
                "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            ],
            capture_output=True, text=True,
        )
        qdrant_ip = qdrant_ip_result.stdout.strip()
        if qdrant_ip:
            e["QDRANT_URL"] = f"http://{qdrant_ip}:6333"
        else:
            e["QDRANT_URL"] = "http://localhost:6333"

    return e


@pytest.mark.integration
def test_backup_writes_postgres_json(tmp_path: Path, env) -> None:
    """Backup half: writes kb_documents + rag_config snapshot to backup-dir/postgres.json."""
    backup_dir = tmp_path / "kb2_backup_test"
    result = subprocess.run(
        [
            ".venv/bin/python", "scripts/wipe_and_reingest.py",
            "--kb", "2",
            "--backup-only",
            "--backup-dir", str(backup_dir),
        ],
        capture_output=True, text=True, timeout=60,
        env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # File exists, has the expected shape
    pg_path = backup_dir / "postgres.json"
    assert pg_path.exists()
    data = json.loads(pg_path.read_text())
    assert "kb_id" in data
    assert data["kb_id"] == 2
    assert "rag_config" in data
    assert isinstance(data["rag_config"], dict)
    assert "kb_documents" in data
    assert isinstance(data["kb_documents"], list)


@pytest.mark.integration
def test_wipe_deletes_qdrant_collections_and_postgres_rows(
    tmp_path: Path,
    env,
) -> None:
    """Wipe phase: deletes kb_{id}_v* collections + DELETE kb_documents rows.

    NOTE: this test creates a temporary KB id=99 with one fake document
    so we don't trash the operator's real KBs. The script accepts any kb_id;
    --confirm-wipe gates the destructive ops.
    """
    import asyncpg
    import asyncio

    async def setup_fake_kb() -> None:
        url = env.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
        url = url.replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            # Tear down any leftover state from a prior run
            await conn.execute("DELETE FROM kb_documents WHERE kb_id = 99")
            await conn.execute("DELETE FROM kb_subtags WHERE kb_id = 99")
            await conn.execute("DELETE FROM knowledge_bases WHERE id = 99")
            # Insert KB 99 — admin_id is VARCHAR(255) with no FK to users
            await conn.execute(
                "INSERT INTO knowledge_bases (id, name, admin_id, rag_config, synonyms) "
                "VALUES (99, 'wipe-test', '1', '{}', '[]')"
            )
            # Insert a subtag (kb_documents.subtag_id is NOT NULL REFERENCES kb_subtags)
            await conn.execute(
                "INSERT INTO kb_subtags (kb_id, name) VALUES (99, 'default')"
            )
            subtag_id = await conn.fetchval(
                "SELECT id FROM kb_subtags WHERE kb_id = 99 LIMIT 1"
            )
            await conn.execute(
                "INSERT INTO kb_documents "
                "  (kb_id, subtag_id, filename, ingest_status, chunk_count, uploaded_by) "
                "VALUES (99, $1, 'test.txt', 'done', 5, 'test')",
                subtag_id,
            )
        finally:
            await conn.close()

    asyncio.run(setup_fake_kb())

    async def setup_qdrant_collection(qdrant_url: str, qdrant_key: str) -> None:
        import httpx
        headers = {"api-key": qdrant_key} if qdrant_key else {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Idempotent — delete any leftover from prior failed runs
            await client.delete(
                f"{qdrant_url}/collections/kb_99_v1", headers=headers
            )
            # Create with minimal vector config (2-dim cosine — enough to satisfy
            # Qdrant; we never insert points)
            r = await client.put(
                f"{qdrant_url}/collections/kb_99_v1",
                headers={**headers, "Content-Type": "application/json"},
                json={"vectors": {"size": 2, "distance": "Cosine"}},
            )
            r.raise_for_status()

    qdrant_url = env.get("QDRANT_URL", "http://localhost:6333")
    qdrant_key = env.get("QDRANT_API_KEY", "")
    asyncio.run(setup_qdrant_collection(qdrant_url, qdrant_key))

    backup_dir = tmp_path / "kb99_backup"
    result = subprocess.run(
        [
            ".venv/bin/python", "scripts/wipe_and_reingest.py",
            "--kb", "99",
            "--confirm-wipe",
            "--backup-dir", str(backup_dir),
        ],
        capture_output=True, text=True, timeout=120,
        env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # Backup written
    assert (backup_dir / "postgres.json").exists()

    # Postgres rows gone
    async def check_pg() -> int:
        url = env.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
        url = url.replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            n = await conn.fetchval(
                "SELECT COUNT(*) FROM kb_documents WHERE kb_id = 99 AND deleted_at IS NULL"
            )
        finally:
            await conn.close()
        return n
    assert asyncio.run(check_pg()) == 0

    # Qdrant collection gone
    async def check_qdrant(qdrant_url: str, qdrant_key: str) -> bool:
        import httpx
        headers = {"api-key": qdrant_key} if qdrant_key else {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{qdrant_url}/collections", headers=headers)
            r.raise_for_status()
            names = [c["name"] for c in r.json()["result"]["collections"]]
            return "kb_99_v1" in names

    assert asyncio.run(check_qdrant(qdrant_url, qdrant_key)) is False, \
        "kb_99_v1 should have been deleted by the wipe"

    # rag_config preserved (the script re-stamps it from the backup)
    async def check_kb() -> dict:
        url = env.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
        url = url.replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            row = await conn.fetchrow(
                "SELECT rag_config FROM knowledge_bases WHERE id = 99"
            )
        finally:
            await conn.close()
        raw = row["rag_config"] if row else {}
        if isinstance(raw, str):
            import json as _json
            raw = _json.loads(raw) if raw else {}
        return dict(raw or {})
    assert asyncio.run(check_kb()) == {}  # original empty rag_config preserved

    # cleanup
    async def teardown() -> None:
        url = env.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
        url = url.replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        try:
            await conn.execute("DELETE FROM knowledge_bases WHERE id = 99")
        finally:
            await conn.close()
    asyncio.run(teardown())

    async def teardown_qdrant(qdrant_url: str, qdrant_key: str) -> None:
        import httpx
        headers = {"api-key": qdrant_key} if qdrant_key else {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Idempotent — 404 is fine; the wipe should have deleted it already
            await client.delete(
                f"{qdrant_url}/collections/kb_99_v1", headers=headers
            )
    asyncio.run(teardown_qdrant(qdrant_url, qdrant_key))
