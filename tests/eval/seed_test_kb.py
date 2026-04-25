"""Seed kb_eval from version-controlled seed_corpus/.

Idempotent: if kb_eval collection exists with expected point count, no-op.
Used by `make eval-baseline` and in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx


YEAR_BUCKETS = ("2023", "2024", "2025", "2026")


def _hash_doc_id(path: Path) -> int:
    """Deterministic 32-bit positive doc_id from relative filename."""
    h = hashlib.sha256(path.name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def collect_corpus_docs(corpus_dir: Path) -> list[dict]:
    """Walk seed_corpus/{2023..2026}/*.md, produce doc records with year tag + doc_id."""
    docs: list[dict] = []
    for year in YEAR_BUCKETS:
        year_dir = corpus_dir / year
        if not year_dir.is_dir():
            continue
        for md in sorted(year_dir.glob("*.md")):
            docs.append({
                "doc_id": _hash_doc_id(md),
                "filename": md.name,
                "year_bucket": year,
                "content": md.read_text(encoding="utf-8"),
            })
    return docs


async def seed(corpus_dir: Path, api_base_url: str, kb_id: int, admin_token: str) -> int:
    """POST docs to /api/kb/{kb_id}/subtag/{sid}/upload. Returns count seeded.

    Assumes kb_id is already created (via kb_admin API) and has a single subtag
    named 'eval'. Run this once after `make up`.
    """
    docs = collect_corpus_docs(corpus_dir)
    headers = {"Authorization": f"Bearer {admin_token}"}
    seeded = 0
    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        # Look up subtag id by name 'eval' (simplified — real impl pages)
        r = await client.get(f"{api_base_url}/api/kb/{kb_id}/subtags")
        r.raise_for_status()
        subtags = r.json()
        eval_subtag = next((s for s in subtags if s["name"] == "eval"), None)
        if eval_subtag is None:
            raise RuntimeError(f"KB {kb_id} missing subtag 'eval'; create it first via admin API")
        sid = eval_subtag["id"]
        for d in docs:
            files = {"file": (d["filename"], d["content"], "text/markdown")}
            data = {"doc_id_hint": str(d["doc_id"])}
            r = await client.post(
                f"{api_base_url}/api/kb/{kb_id}/subtag/{sid}/upload",
                files=files, data=data,
            )
            if r.status_code == 409:
                # already seeded — idempotent
                continue
            r.raise_for_status()
            seeded += 1
    return seeded


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", type=Path,
                   default=Path(__file__).resolve().parent / "seed_corpus")
    p.add_argument("--api-base-url", default="http://localhost:6100")
    p.add_argument("--kb-id", type=int, required=True)
    p.add_argument("--admin-token", default=os.environ.get("RAG_ADMIN_TOKEN", ""))
    args = p.parse_args()

    if not args.admin_token:
        print("ERROR: --admin-token or RAG_ADMIN_TOKEN required")
        return 2

    import asyncio
    n = asyncio.run(seed(args.corpus_dir, args.api_base_url, args.kb_id, args.admin_token))
    print(f"seeded {n} docs into kb_id={args.kb_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
