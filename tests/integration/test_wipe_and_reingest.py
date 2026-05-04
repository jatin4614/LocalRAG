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
    if "DATABASE_URL" not in e:
        # Resolve the live Postgres container IP (not host-exposed on 5432)
        # and extract the password from compose/.env — same pattern as
        # test_edit_kb_synonyms_cli.py, extended for the unexposed-port case.
        pw = None
        try:
            env_text = open("compose/.env").read()
            for line in env_text.splitlines():
                if line.startswith("POSTGRES_PASSWORD="):
                    pw = line.split("=", 1)[1].strip()
                    break
        except FileNotFoundError:
            pass

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
