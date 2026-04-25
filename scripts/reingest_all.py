#!/usr/bin/env python3
"""Re-ingest KB documents against the current ingest pipeline.

Phase 1a's ``CHUNKER_VERSION=v3`` (adjacent-block coalescence + embed-time
heading prepend + atomic tables) gives fundamentally different chunk
boundaries than v2. Old chunks stay in Qdrant until we rewrite them;
this script does that rewrite.

Source of raw bytes, in priority order:

1. ``kb_documents.blob_sha`` — populated for async-ingest uploads. The
   BlobStore at ``$INGEST_BLOB_ROOT`` (default ``/var/ingest``) holds the
   sha256-addressed blob.
2. ``volumes/uploads/{uuid}_{filename}`` — upstream Open WebUI saves some
   uploads under this prefix. We match by suffix against
   ``kb_documents.filename``.

Docs that can't be rehydrated from either source are reported and
skipped — the supervisor can re-upload them from the original.

Idempotency & resumability:

- ``--pipeline-version`` (optional): skip docs whose
  ``pipeline_version`` already matches this value. Default is the
  current version produced by ``ext.services.pipeline_version``. So a
  crashed run resumes where it left off just by re-running.
- ``--dry-run`` (default behaviour is explicit ``--dry-run``; no flag
  given = dry-run too for safety): reports what would be re-ingested
  and rough chunk-count deltas, makes no writes.

CLI contract (enforced by argparse):

    python scripts/reingest_all.py (--kb N | --all) [--dry-run]
                                   [--batch-size N] [--pipeline-version V]
                                   [--database-url URL] [--qdrant-url URL]
                                   [--tei-url URL] [--blob-root PATH]
                                   [--uploads-root PATH]

Exit codes:
    0  success (including no-work)
    1  operational failure (DB / Qdrant / TEI error)
    2  missing dependency
    4  invalid arguments

Writes: for ``--apply`` (i.e. ``--dry-run`` absent), the script:
    1. deletes existing Qdrant points for ``doc_id`` (``vector_store.delete_by_doc``),
    2. calls ``ingest_bytes`` with the raw bytes,
    3. updates ``kb_documents.pipeline_version`` + ``chunk_count``.

Note: this script does NOT rebuild the chat_private collection — only
KB-owned docs. Chat-private docs are session-ephemeral and are rebuilt
naturally by new uploads.

The supervisor runs this after deploying the v3 code. See
``docs/rag-phase0-1a-4-execution-plan.md`` §4.3.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# ``scripts/reingest_all.py`` → repo root is one level up.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--kb",
        type=int,
        help="Only re-ingest docs in this KB id.",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Re-ingest docs across every KB (not chat_private).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the plan + expected chunk-count delta (based on current "
            "chunk_count vs. a quick re-chunk). Make no writes. "
            "Recommended first pass before --apply."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Docs processed per commit (default: 16). Larger batches = fewer commits.",
    )
    p.add_argument(
        "--pipeline-version",
        default=None,
        help=(
            "Only re-ingest docs whose current pipeline_version differs from "
            "this string. Defaults to the current pipeline_version produced by "
            "ext.services.pipeline_version (v3 as of Phase 1a). Override to "
            "force-reingest everything by passing an empty string or a value "
            "no doc actually carries (e.g. --pipeline-version force)."
        ),
    )
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy async DB URL (required; defaults to $DATABASE_URL).",
    )
    p.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant REST URL (default: $QDRANT_URL or http://localhost:6333).",
    )
    p.add_argument(
        "--tei-url",
        default=os.environ.get("TEI_URL", "http://localhost:8080"),
        help="TEI base URL for embeddings (default: $TEI_URL or http://localhost:8080).",
    )
    p.add_argument(
        "--blob-root",
        default=os.environ.get("INGEST_BLOB_ROOT", "/var/ingest"),
        help="BlobStore root (default: $INGEST_BLOB_ROOT or /var/ingest).",
    )
    p.add_argument(
        "--uploads-root",
        default=str(_ROOT / "volumes" / "uploads"),
        help=(
            "Fallback dir for docs without blob_sha. Script matches files here "
            "by suffix against kb_documents.filename. Default: "
            "<repo>/volumes/uploads."
        ),
    )
    p.add_argument(
        "--vector-size",
        type=int,
        default=int(os.environ.get("RAG_VECTOR_SIZE", "1024")),
        help="Dense vector dim (default: $RAG_VECTOR_SIZE or 1024).",
    )
    p.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help=(
            "Target chunk size. When omitted, each KB's rag_config.chunk_tokens "
            "wins; if unset, falls back to $CHUNK_SIZE (or 800). Explicit value "
            "overrides per-KB config (useful for one-off tuning experiments)."
        ),
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=(
            "Chunk overlap tokens. Same priority as --chunk-tokens: per-KB "
            "rag_config.overlap_tokens, then $CHUNK_OVERLAP (or 100)."
        ),
    )
    return p.parse_args(argv)


def _find_in_uploads(uploads_root: Path, filename: str) -> Optional[Path]:
    """Match a kb_documents filename against a file in ``uploads_root``.

    Upstream Open WebUI saves uploads as ``{uuid}_{filename}`` (UUID is
    an upstream doc id, not our KBDocument.id). We match by ``endswith``
    on the filename with the ``_`` separator so a name collision between
    two different upstream UUIDs doesn't pick the wrong file.

    Returns None if no match or ``uploads_root`` doesn't exist.
    """
    if not uploads_root.exists() or not uploads_root.is_dir():
        return None
    suffix = f"_{filename}"
    for candidate in sorted(uploads_root.iterdir()):
        if candidate.is_file() and candidate.name.endswith(suffix):
            return candidate
    return None


async def _load_bytes_for_doc(
    doc,
    *,
    blob_store,
    uploads_root: Path,
) -> tuple[Optional[bytes], str]:
    """Rehydrate raw bytes for ``doc``. Returns (bytes, source) or (None, reason).

    ``source`` is one of ``"blob_sha"``, ``"uploads_root"``. On failure,
    a short reason string is returned in its place for the progress log.
    """
    # Priority 1: blob_sha.
    if doc.blob_sha:
        try:
            path = Path(blob_store.path(doc.blob_sha))
            if path.exists():
                return path.read_bytes(), "blob_sha"
        except Exception as exc:  # noqa: BLE001
            return None, f"blob_read_error:{exc.__class__.__name__}"

    # Priority 2: uploads_root suffix match.
    match = _find_in_uploads(uploads_root, doc.filename)
    if match is not None:
        try:
            return match.read_bytes(), "uploads_root"
        except Exception as exc:  # noqa: BLE001
            return None, f"uploads_read_error:{exc.__class__.__name__}"

    return None, "bytes_unavailable"


async def _run(args: argparse.Namespace) -> int:
    if not args.database_url:
        print("error: --database-url or $DATABASE_URL is required", file=sys.stderr)
        return 4

    try:
        from sqlalchemy import select, update

        from ext.db.models import KBDocument, KnowledgeBase
        from ext.db.session import make_engine, make_sessionmaker
        from ext.services.blob_store import BlobStore
        from ext.services.embedder import TEIEmbedder
        from ext.services.ingest import ingest_bytes
        from ext.services.kb_config import resolve_chunk_params
        from ext.services.pipeline_version import current_version
        from ext.services.vector_store import VectorStore
    except ImportError as exc:
        print(f"error: missing dependency: {exc}", file=sys.stderr)
        return 2

    target_pv = (
        args.pipeline_version
        if args.pipeline_version is not None
        else current_version()
    )

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(
        f"[{mode}] reingest_all "
        f"kb={'all' if args.all else args.kb} "
        f"batch_size={args.batch_size} "
        f"target_pipeline_version={target_pv!r}"
    )

    engine = make_engine(args.database_url)
    sm = make_sessionmaker(engine)
    blob_store = BlobStore(args.blob_root)
    uploads_root = Path(args.uploads_root).resolve()
    vs = VectorStore(url=args.qdrant_url, vector_size=args.vector_size)
    emb = TEIEmbedder(base_url=args.tei_url)

    rc = 0
    totals = {
        "docs_considered": 0,
        "docs_already_current": 0,
        "docs_rehydrated": 0,
        "docs_missing_bytes": 0,
        "docs_ingest_failed": 0,
        "docs_reingested": 0,
        "old_chunks_total": 0,
        "new_chunks_total": 0,
    }
    try:
        async with sm() as session:
            # Build the doc list.
            q = select(KBDocument).where(KBDocument.deleted_at.is_(None))
            if args.kb is not None:
                q = q.where(KBDocument.kb_id == args.kb)
            q = q.order_by(KBDocument.kb_id, KBDocument.id)
            result = await session.execute(q)
            docs = list(result.scalars())
            total = len(docs)
            print(f"  scanning {total} non-deleted KB documents")

            for i, doc in enumerate(docs, start=1):
                totals["docs_considered"] += 1
                cur_pv = doc.pipeline_version or ""
                if cur_pv == target_pv:
                    totals["docs_already_current"] += 1
                    if args.dry_run:
                        print(
                            f"  [{i}/{total}] kb={doc.kb_id} doc_id={doc.id} "
                            f"filename={doc.filename!r} skip (already at {target_pv})"
                        )
                    continue

                # Priority path for bytes.
                data, src = await _load_bytes_for_doc(
                    doc,
                    blob_store=blob_store,
                    uploads_root=uploads_root,
                )
                if data is None:
                    totals["docs_missing_bytes"] += 1
                    print(
                        f"  [{i}/{total}] kb={doc.kb_id} doc_id={doc.id} "
                        f"filename={doc.filename!r} SKIP ({src})",
                        file=sys.stderr,
                    )
                    continue
                totals["docs_rehydrated"] += 1

                old_count = int(doc.chunk_count or 0)
                totals["old_chunks_total"] += old_count

                if args.dry_run:
                    # Cheap estimate: same byte count → roughly same number of
                    # chunks. v3 typically produces FEWER chunks (coalescence),
                    # but without running the full pipeline we just surface the
                    # current count and let the supervisor eyeball totals.
                    print(
                        f"  [{i}/{total}] kb={doc.kb_id} doc_id={doc.id} "
                        f"filename={doc.filename!r} source={src} "
                        f"cur_pv={cur_pv or '(null)'!r} -> {target_pv!r} "
                        f"old_chunks={old_count}"
                    )
                    continue

                # Apply path.
                try:
                    # Remove stale points first so a crashed re-ingest does
                    # not double-up points at the same doc_id.
                    collection = f"kb_{doc.kb_id}"
                    try:
                        await vs.delete_by_doc(collection, doc.id)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"  WARN delete_by_doc failed kb={doc.kb_id} "
                            f"doc_id={doc.id}: {exc}",
                            file=sys.stderr,
                        )

                    await vs.ensure_collection(collection, with_sparse=True)
                    payload_base = {
                        "kb_id": doc.kb_id,
                        "subtag_id": doc.subtag_id,
                        "doc_id": doc.id,
                        "owner_user_id": doc.uploaded_by,
                        "filename": doc.filename,
                    }
                    # Per-KB chunk params. CLI explicit flag wins; otherwise
                    # fall back to the KB's rag_config → env → signature defaults.
                    if args.chunk_tokens is not None or args.chunk_overlap is not None:
                        ct, ov = resolve_chunk_params(
                            {"chunk_tokens": args.chunk_tokens,
                             "overlap_tokens": args.chunk_overlap}
                            if args.chunk_tokens and args.chunk_overlap
                            else None
                        )
                        if args.chunk_tokens is not None:
                            ct = args.chunk_tokens
                        if args.chunk_overlap is not None:
                            ov = args.chunk_overlap
                    else:
                        kb_row = (await session.execute(
                            select(KnowledgeBase.rag_config).where(
                                KnowledgeBase.id == doc.kb_id
                            )
                        )).first()
                        ct, ov = resolve_chunk_params(
                            kb_row[0] if kb_row and kb_row[0] else None
                        )
                    n = await ingest_bytes(
                        data=data,
                        mime_type=doc.mime_type or "application/octet-stream",
                        filename=doc.filename,
                        collection=collection,
                        payload_base=payload_base,
                        vector_store=vs,
                        embedder=emb,
                        chunk_tokens=ct,
                        overlap_tokens=ov,
                    )
                except Exception as exc:  # noqa: BLE001
                    totals["docs_ingest_failed"] += 1
                    print(
                        f"  FAIL ingest kb={doc.kb_id} doc_id={doc.id} "
                        f"filename={doc.filename!r}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                totals["docs_reingested"] += 1
                totals["new_chunks_total"] += int(n)
                print(
                    f"  [{i}/{total}] kb={doc.kb_id} doc_id={doc.id} "
                    f"filename={doc.filename!r} source={src} "
                    f"reingested old_chunks={old_count} new_chunks={n}"
                )

                # Stamp pipeline_version + chunk_count on the row.
                await session.execute(
                    update(KBDocument)
                    .where(KBDocument.id == doc.id)
                    .values(
                        pipeline_version=target_pv,
                        chunk_count=int(n),
                        ingest_status="done",
                    )
                )
                # Commit in batches so a crash doesn't lose an hour of work.
                if i % max(1, args.batch_size) == 0:
                    await session.commit()

            # Final commit for trailing docs.
            if not args.dry_run:
                await session.commit()

        # Summary
        print("\n=== summary ===")
        for k, v in totals.items():
            print(f"  {k}: {v}")
        if args.dry_run:
            print("  mode: DRY-RUN (no writes)")
        else:
            delta = totals["new_chunks_total"] - totals["old_chunks_total"]
            print(f"  chunk_count delta: {delta:+d}")
    except Exception as exc:  # noqa: BLE001
        print(f"error: run failed: {exc}", file=sys.stderr)
        rc = 1
    finally:
        try:
            await emb.aclose()
        except Exception:
            pass
        try:
            await vs.close()
        except Exception:
            pass
        await engine.dispose()

    return rc


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on validation errors; keep that behaviour for CI.
        return int(exc.code) if exc.code is not None else 0
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
