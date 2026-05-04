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


async def _qdrant_delete_kb_collections(kb_id: int) -> list[str]:
    """Delete every collection whose name matches kb_{id} or kb_{id}_v*."""
    import httpx
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    headers = {"api-key": api_key} if api_key else {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{url}/collections", headers=headers)
        r.raise_for_status()
        all_names = [c["name"] for c in r.json()["result"]["collections"]]

        prefix = f"kb_{kb_id}"
        targets = [
            n for n in all_names
            if n == prefix or n.startswith(f"{prefix}_") or n == f"{prefix}_rebuild"
        ]
        deleted = []
        for name in targets:
            resp = await client.delete(
                f"{url}/collections/{name}", headers=headers,
            )
            if resp.status_code in (200, 404):
                deleted.append(name)
            else:
                # Loud failure — half-deleted Qdrant state is worse than no-op.
                # The operator can re-run the script after fixing the auth /
                # network issue; both phases are idempotent.
                raise RuntimeError(
                    f"Qdrant DELETE /collections/{name} failed: "
                    f"{resp.status_code} {resp.text[:200]}"
                )
        return deleted


async def _postgres_wipe(kb_id: int, restore_rag_config: dict) -> None:
    """DELETE kb_documents rows; preserve knowledge_bases.rag_config."""
    conn = await _conn()
    try:
        async with conn.transaction():
            # Hard-delete docs (FK cascades chunk-level rows in any audit table
            # if those exist; they don't today but the wipe is meant to be
            # destructive).
            await conn.execute("DELETE FROM kb_documents WHERE kb_id = $1", kb_id)
            # Re-stamp rag_config from the backup so chunker/floor settings
            # survive the wipe.
            await conn.execute(
                "UPDATE knowledge_bases SET rag_config = $1::jsonb WHERE id = $2",
                json.dumps(restore_rag_config), kb_id,
            )
    finally:
        await conn.close()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", type=int, required=True)
    p.add_argument("--backup-dir", type=Path, required=True)
    p.add_argument("--backup-only", action="store_true")
    p.add_argument("--confirm-wipe", action="store_true")
    args = p.parse_args()

    print(f"== KB {args.kb} backup -> {args.backup_dir} ==")
    backup_path = await _backup_postgres(args.kb, args.backup_dir)
    backup_data = json.loads(backup_path.read_text())
    print(f"  postgres backup: {backup_path}")
    print(f"  kb_documents in backup: {len(backup_data['kb_documents'])}")

    if args.backup_only:
        print("== --backup-only set; not wiping ==")
        return

    if not args.confirm_wipe:
        print("error: pass --confirm-wipe to actually wipe", file=sys.stderr)
        sys.exit(4)

    # Wipe Qdrant first — if Qdrant fails, Postgres is still consistent.
    print(f"== wiping Qdrant collections for kb_{args.kb} ==")
    deleted = await _qdrant_delete_kb_collections(args.kb)
    for n in deleted:
        print(f"  deleted: {n}")
    if not deleted:
        print(f"  (no collections matched kb_{args.kb}*)")

    # Then Postgres rows.
    print(f"== wiping kb_documents rows for kb_id={args.kb} ==")
    await _postgres_wipe(args.kb, backup_data["rag_config"])
    print(f"  rag_config restored from backup ({len(backup_data['rag_config'])} keys)")

    print()
    print("== DONE ==")
    print(f"Re-upload your docs via /api/kb/{args.kb}/subtag/<sid>/upload — "
          "ingest will pick up the per-KB rag_config restored from the backup.")


if __name__ == "__main__":
    asyncio.run(main())
