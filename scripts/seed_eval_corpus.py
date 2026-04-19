#!/usr/bin/env python3
"""Seed the ``kb_eval`` Qdrant collection with a fixed, reproducible corpus
so that tests/eval can run against a dense, stable set of docs.

Live KB collections (``kb_1``, ``kb_3``, ``kb_4``, ``kb_5``) are too sparse
(1 / 6 / 146 / 0 chunks respectively) for a meaningful eval baseline. This
script ingests a curated set of worktree documentation + any files dropped
into ``tests/eval/seed_corpus/`` into a dedicated ``kb_eval`` collection so
``tests/eval/generate_golden.py`` can produce a 100+ row golden set.

Idempotent: if ``kb_eval`` exists and already holds chunks from the current
pipeline_version, exits with ``already seeded`` and does not touch points.
Pass ``--force`` to wipe-and-re-ingest.

Usage:
    python scripts/seed_eval_corpus.py \\
        --qdrant-url http://localhost:6333 \\
        --tei-url http://localhost:8080

Reproducibility:
    Corpus is {worktree docs on the allowlist below} ∪ {any .md/.txt under
    tests/eval/seed_corpus/}. doc_id is a stable hash of the relative path,
    chunking is the same deterministic sentence-aware chunker used by the
    live ingest path (``ext.services.chunker.chunk_text``).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm


# ``scripts/seed_eval_corpus.py`` → repo root is one level up.
ROOT = Path(__file__).resolve().parents[1]
# Make ``ext.services.*`` importable regardless of how the script is invoked.
sys.path.insert(0, str(ROOT))

from ext.services.chunker import chunk_text  # noqa: E402
from ext.services.pipeline_version import current_version  # noqa: E402


# Stable namespace for deterministic point IDs (matches ext/services/ingest.py).
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


# Fixed allowlist of in-worktree docs we always try to include. Missing files
# are silently skipped so the script works on lightly-populated checkouts.
# Paths are relative to repo root. Order only matters for deterministic
# logging — doc_id is derived from path, not index.
WORKTREE_ALLOWLIST = [
    "CLAUDE.md",
    "Ragupdate.md",
    "README.md",
    "recommendation.md",
    "docs/runbook.md",
    "docs/rag-upgrade-execution-plan.md",
    "docs/rag-upgrade-disaster-recovery.md",
    "docs/superpowers/plans/2026-04-18-rag-pipeline-upgrade.md",
    "docs/superpowers/plans/2026-04-16-org-chat-assistant-master-plan.md",
    "docs/superpowers/plans/2026-04-16-phase-1-foundation.md",
    "docs/superpowers/plans/2026-04-16-phase-2-kb-management-rbac.md",
    "docs/superpowers/plans/2026-04-16-phase-3-model-manager.md",
    "docs/superpowers/plans/2026-04-16-phase-4-rag-pipeline.md",
    "docs/superpowers/plans/2026-04-16-phase-5a-auth-bridge.md",
    "docs/superpowers/plans/2026-04-16-phase-6-testing-battery.md",
    "docs/superpowers/plans/2026-04-16-phase-7-runbook-and-k8s.md",
    "docs/superpowers/specs/2026-04-12-learning-plan.md",
    "docs/superpowers/specs/2026-04-12-org-chat-assistant-design.md",
    "docs/superpowers/specs/2026-04-16-kb-rag-pipeline-workflow.md",
    "docs/superpowers/specs/2026-04-17-kb-admin-ui-and-rag-flow-design.md",
]


def _discover_files(seed_corpus_dir: Path) -> list[Path]:
    """Return the ordered list of source files to ingest (absolute paths).

    Combines:
    * Allowlist entries that exist on disk, in allowlist order.
    * All .md/.txt under tests/eval/seed_corpus/, sorted alphabetically for
      deterministic doc_id assignment across runs.
    """
    picked: list[Path] = []
    seen: set[Path] = set()
    for rel in WORKTREE_ALLOWLIST:
        p = (ROOT / rel).resolve()
        if p.exists() and p.is_file() and p not in seen:
            picked.append(p)
            seen.add(p)

    if seed_corpus_dir.is_dir():
        extras = sorted(
            list(seed_corpus_dir.rglob("*.md")) + list(seed_corpus_dir.rglob("*.txt")),
            key=lambda q: str(q),
        )
        for p in extras:
            if p.is_file() and p not in seen:
                picked.append(p)
                seen.add(p)
    return picked


def _stable_doc_id(rel_path: str) -> int:
    """Hash a path string down to a stable int in [0, 1_000_000).

    Issue 2 established doc_id as an int-in-payload convention. We want
    reproducibility across machines, so md5 → first 4 bytes → mod 1M gives
    a small, collision-unlikely int (birthday paradox: ~0.1% chance of any
    collision within a 50-file corpus).
    """
    h = hashlib.md5(rel_path.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 1_000_000


def _strip_markdown_fences(text: str) -> str:
    """Light cleanup: drop triple-backtick fenced code blocks entirely.

    Code blocks are noise for back-generated queries (the chat model tends
    to ask "what does this code do?" which is not a realistic retrieval
    query). Keep prose intact. This is a best-effort regex; it won't
    handle nested or unclosed fences but markdown spec says unclosed
    fences close at EOF anyway.
    """
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        out.append(line)
    return "\n".join(out)


def _read_source(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".md", ".markdown"):
        return _strip_markdown_fences(raw)
    return raw


async def _tei_embed(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """POST /embed to TEI. Mirrors TEIEmbedder.embed() behaviour."""
    r = await client.post("/embed", json={"inputs": texts})
    r.raise_for_status()
    return r.json()


async def _ensure_collection(
    qdrant: AsyncQdrantClient, name: str, *, vector_size: int
) -> None:
    """Create the collection if missing. Payload indexes match VectorStore."""
    cols = (await qdrant.get_collections()).collections
    if not any(c.name == name for c in cols):
        await qdrant.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=vector_size, distance=qm.Distance.COSINE
            ),
        )
    # Payload indexes — idempotent.
    for field in ("kb_id", "subtag_id", "doc_id", "chat_id", "deleted"):
        try:
            await qdrant.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


async def _collection_point_count(
    qdrant: AsyncQdrantClient, name: str
) -> Optional[int]:
    try:
        resp = await qdrant.count(collection_name=name, exact=True)
        return resp.count
    except Exception:
        return None


def _batched(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def seed(args) -> int:
    seed_corpus_dir = (ROOT / "tests" / "eval" / "seed_corpus").resolve()
    files = _discover_files(seed_corpus_dir)
    if not files:
        print("no source files found — nothing to seed", file=sys.stderr)
        return 3

    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    tei = httpx.AsyncClient(base_url=args.tei_url, timeout=args.timeout)

    try:
        # Idempotency check: bail early if collection looks seeded already.
        if not args.force:
            n = await _collection_point_count(qdrant, args.collection_name)
            if n is not None and n > 0:
                print(
                    f"{args.collection_name} already has {n} points — "
                    f"pass --force to wipe-and-re-ingest",
                    file=sys.stderr,
                )
                return 0

        if args.force:
            # Wipe collection first so old chunks don't linger.
            try:
                await qdrant.delete_collection(args.collection_name)
            except Exception:
                pass

        await _ensure_collection(
            qdrant, args.collection_name, vector_size=args.vector_size
        )

        now = time.time_ns()
        pv = current_version()
        total_chunks = 0
        per_file: list[tuple[str, int]] = []

        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            doc_id = _stable_doc_id(rel)
            text = _read_source(path)
            chunks = chunk_text(
                text,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
            )
            if not chunks:
                per_file.append((rel, 0))
                print(f"  (skip) {rel}: no chunks", file=sys.stderr)
                continue

            # Embed in batches of --batch-size to stay polite to TEI.
            texts = [c.text for c in chunks]
            vectors: list[list[float]] = []
            for batch in _batched(texts, args.batch_size):
                vectors.extend(await _tei_embed(tei, batch))

            points = []
            for chunk, vec in zip(chunks, vectors):
                point_id = str(
                    uuid.uuid5(_POINT_NS, f"doc:{doc_id}:chunk:{chunk.index}")
                )
                payload = {
                    "kb_id": "eval",
                    "subtag_id": None,
                    "doc_id": str(doc_id),
                    "filename": rel,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "uploaded_at": now,
                    "deleted": False,
                    "page": None,
                    "heading_path": [],
                    "sheet": None,
                    "model_version": pv,
                }
                points.append(
                    qm.PointStruct(id=point_id, vector=vec, payload=payload)
                )

            # Upsert each file as one batch — simple and cheap for our sizes.
            await qdrant.upsert(
                collection_name=args.collection_name, points=points, wait=True
            )
            total_chunks += len(points)
            per_file.append((rel, len(points)))
            print(f"  + {rel}: {len(points)} chunks", file=sys.stderr)

        print(
            f"\nseeded {args.collection_name}: {len(files)} files, "
            f"{total_chunks} chunks",
            file=sys.stderr,
        )
        if total_chunks < 50:
            print(
                f"WARN: only {total_chunks} chunks — eval set will be small. "
                f"Add more .md/.txt under tests/eval/seed_corpus/ to densify.",
                file=sys.stderr,
            )
        return 0
    finally:
        await qdrant.close()
        await tei.aclose()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument(
        "--tei-url",
        default="http://localhost:8080",
        help="TEI base URL (exposes /embed directly)",
    )
    p.add_argument(
        "--collection-name",
        default="kb_eval",
        help="target Qdrant collection (default: kb_eval)",
    )
    p.add_argument(
        "--vector-size",
        type=int,
        default=1024,
        help="embedding dim; must match live TEI model (bge-m3 = 1024)",
    )
    p.add_argument("--chunk-tokens", type=int, default=800)
    p.add_argument("--overlap-tokens", type=int, default=100)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="texts per /embed call (default: 16)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="drop the collection and re-ingest from scratch",
    )
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    ns = _parse_args()
    raise SystemExit(asyncio.run(seed(ns)))
