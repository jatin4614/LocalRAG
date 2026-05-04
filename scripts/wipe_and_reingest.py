"""Wipe a KB and prepare for fresh re-ingest.

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §7

Usage:
    .venv/bin/python scripts/wipe_and_reingest.py --kb 2 --backup-only --backup-dir DIR
    .venv/bin/python scripts/wipe_and_reingest.py --kb 2 --confirm-wipe --backup-dir DIR

Phase 1 of this spec ships the --backup-only path. Phase 1 / Task 5 adds
the --confirm-wipe path that deletes Qdrant collections + Postgres rows
after the backup is taken.

Env: DATABASE_URL (required). QDRANT_URL + QDRANT_API_KEY (required for
the wipe path; optional for backup-only since backup doesn't touch
Qdrant).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg


async def _conn() -> asyncpg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)
    url = url.replace("+asyncpg", "")
    return await asyncpg.connect(url)


async def _backup_postgres(kb_id: int, backup_dir: Path) -> Path:
    """Snapshot rag_config + every kb_documents row for the KB to JSON."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    out_path = backup_dir / "postgres.json"

    conn = await _conn()
    try:
        kb_row = await conn.fetchrow(
            "SELECT id, name, rag_config, synonyms FROM knowledge_bases WHERE id = $1",
            kb_id,
        )
        if kb_row is None:
            print(f"error: kb_id={kb_id} not found", file=sys.stderr)
            sys.exit(3)

        doc_rows = await conn.fetch(
            "SELECT id, kb_id, subtag_id, filename, mime_type, bytes, "
            "       ingest_status, chunk_count, pipeline_version, blob_sha, "
            "       doc_summary, uploaded_at, uploaded_by "
            "FROM kb_documents WHERE kb_id = $1 AND deleted_at IS NULL "
            "ORDER BY id",
            kb_id,
        )
    finally:
        await conn.close()

    rag_config_raw = kb_row["rag_config"]
    if isinstance(rag_config_raw, str):
        rag_config_raw = json.loads(rag_config_raw) if rag_config_raw else {}
    synonyms_raw = kb_row["synonyms"]
    if isinstance(synonyms_raw, str):
        synonyms_raw = json.loads(synonyms_raw) if synonyms_raw else []

    payload = {
        "kb_id": kb_row["id"],
        "name": kb_row["name"],
        "rag_config": dict(rag_config_raw or {}),
        "synonyms": synonyms_raw or [],
        "kb_documents": [
            {
                "id": r["id"],
                "kb_id": r["kb_id"],
                "subtag_id": r["subtag_id"],
                "filename": r["filename"],
                "mime_type": r["mime_type"],
                "bytes": r["bytes"],
                "ingest_status": r["ingest_status"],
                "chunk_count": r["chunk_count"],
                "pipeline_version": r["pipeline_version"],
                "blob_sha": r["blob_sha"],
                "doc_summary": r["doc_summary"],
                "uploaded_at": r["uploaded_at"].isoformat() if r["uploaded_at"] else None,
                "uploaded_by": r["uploaded_by"],
            }
            for r in doc_rows
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", type=int, required=True)
    p.add_argument("--backup-dir", type=Path, required=True)
    p.add_argument("--backup-only", action="store_true",
                   help="Run backup phase only; skip wipe")
    p.add_argument("--confirm-wipe", action="store_true",
                   help="Actually wipe (Phase 5 — added in Task 5)")
    args = p.parse_args()

    print(f"== KB {args.kb} backup -> {args.backup_dir} ==")
    backup_path = await _backup_postgres(args.kb, args.backup_dir)
    print(f"  postgres backup: {backup_path}")

    if args.backup_only:
        print("== --backup-only set; not wiping ==")
        return

    if not args.confirm_wipe:
        print("error: pass --confirm-wipe to actually wipe", file=sys.stderr)
        sys.exit(4)

    print("error: wipe phase added in Task 5", file=sys.stderr)
    sys.exit(5)


if __name__ == "__main__":
    asyncio.run(main())
