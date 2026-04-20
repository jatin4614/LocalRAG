#!/usr/bin/env python3
"""Retrofit hybrid (dense+BM25) support onto a legacy dense-only Qdrant collection.

This script creates a **v2 shadow** collection with the hybrid named-vector
shape (``dense`` + ``bm25``) and copies every point from the legacy source
into it, reusing the existing dense embedding (no re-embed needed) and
computing a fresh BM25 sparse vector from ``payload.text`` via fastembed.

The source collection is **NEVER deleted** — the operator swaps manually
once they've verified the v2 collection is healthy. This makes the reindex
fully reversible until the swap is executed.

Why not in-place? Qdrant's ``UpdateCollection`` cannot change the vector
config of an existing collection (named vs unnamed, dense vs sparse). The
only safe upgrade path is: create fresh → re-populate → swap.

Swap procedure (run manually after reviewing the v2 collection):

    # 1) Stop writes to the source collection (pause ingest worker).
    # 2) Verify counts match:
    curl http://localhost:6333/collections/kb_1 | jq .result.points_count
    curl http://localhost:6333/collections/kb_1_v2 | jq .result.points_count
    # 3) Rename-by-alias (atomic, zero-downtime):
    curl -X POST http://localhost:6333/collections/aliases \\
        -H 'content-type: application/json' \\
        -d '{"actions":[{"delete_alias":{"alias_name":"kb_1"}}]}'
    curl -X POST http://localhost:6333/collections/aliases \\
        -H 'content-type: application/json' \\
        -d '{"actions":[{"create_alias":{"collection_name":"kb_1_v2","alias_name":"kb_1"}}]}'
    # 4) Or a hard rename: delete old + rename new.
    # 5) Resume ingest worker. RAG_HYBRID=1 will now exercise the bm25 arm.

Usage:
    # Dry-run (default) — prints plan, makes no writes.
    python scripts/reindex_hybrid.py \\
        --qdrant-url http://localhost:6333 \\
        --tei-url http://172.19.0.6:80 \\
        --source kb_1 \\
        --target kb_1_v2

    # Apply.
    python scripts/reindex_hybrid.py \\
        --qdrant-url http://localhost:6333 \\
        --tei-url http://172.19.0.6:80 \\
        --source kb_1 \\
        --target kb_1_v2 \\
        --apply

Exit codes:
    0  success (or dry-run finished cleanly)
    1  any error — source missing, target exists without --force, Qdrant error
    2  fastembed not installed / import error
    4  invalid arguments
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Make ``ext.services.*`` importable regardless of how the script is invoked.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Named-vector keys for hybrid collections — must match VectorStore.
_DENSE_NAME = "dense"
_SPARSE_NAME = "bm25"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant REST URL (default: http://localhost:6333).",
    )
    p.add_argument(
        "--tei-url",
        default="http://localhost:8080",
        help=(
            "TEI base URL. Not currently used by reindex (dense vectors are "
            "copied as-is from the source collection) but kept for parity "
            "with sibling scripts and for future re-embed paths."
        ),
    )
    p.add_argument(
        "--source",
        required=True,
        help="Source (legacy dense-only) collection name, e.g. kb_1.",
    )
    p.add_argument(
        "--target",
        required=True,
        help="Target (hybrid) collection name, e.g. kb_1_v2. Must not exist "
        "unless --force is also passed.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Points scrolled / sparse-embedded / upserted per round (default: 64).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing target collection. Refused by default.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for Qdrant calls (default: 60s).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the plan, make no writes (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Actually create target collection and upsert points. Opt-in.",
    )
    return p.parse_args(argv)


def _infer_vector_size_and_distance(info) -> tuple[int, str]:
    """Return (size, distance-name) of the source's dense vector config.

    Handles both legacy unnamed (``VectorParams``) and named-dict shapes.
    For named shapes, prefers the ``_DENSE_NAME`` entry; falls back to the
    first named vector if that's missing (defensive — shouldn't happen for
    legacy kb_N collections).
    """
    vectors = info.config.params.vectors
    if vectors is None:
        raise RuntimeError("source collection has no vectors config")
    # Legacy unnamed VectorParams (direct, not a dict).
    size = getattr(vectors, "size", None)
    dist = getattr(vectors, "distance", None)
    if size is not None and dist is not None:
        return int(size), str(dist)
    # Named-vectors dict (dict[str, VectorParams]).
    if isinstance(vectors, dict):
        vp = vectors.get(_DENSE_NAME)
        if vp is None:
            # Take the first named entry defensively.
            first_key = next(iter(vectors))
            vp = vectors[first_key]
        return int(vp.size), str(vp.distance)
    raise RuntimeError(f"unrecognised vectors config type: {type(vectors).__name__}")


def _dense_vector_for(point) -> list[float] | None:
    """Extract the dense vector from a scrolled point.

    Legacy source collections store the vector as a plain list (unnamed).
    If the source was already hybrid (``dict[str, Any]``), pull the
    ``dense`` entry. Returns None if no vector is attached (point skipped).
    """
    v = point.vector
    if v is None:
        return None
    if isinstance(v, dict):
        dense = v.get(_DENSE_NAME)
        if dense is not None:
            return list(dense)
        # Fall back to first dict value if dense key missing (defensive).
        for val in v.values():
            if isinstance(val, list):
                return list(val)
        return None
    # Unnamed — just a list.
    return list(v)


async def _reindex(args: argparse.Namespace) -> int:
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http import models as qm
    except ImportError as e:
        print(f"error: qdrant-client not installed: {e}", file=sys.stderr)
        return 2
    try:
        from ext.services.sparse_embedder import embed_sparse
    except ImportError as e:
        print(
            f"error: cannot import sparse embedder ({e}). "
            f"Install via `pip install 'fastembed>=0.4'` or `pip install '.[hybrid]'`.",
            file=sys.stderr,
        )
        return 2

    apply = bool(args.apply)  # default stays dry-run
    banner = "APPLY" if apply else "DRY-RUN"
    print(f"[{banner}] reindex {args.source} -> {args.target} @ {args.qdrant_url}")

    qdrant = AsyncQdrantClient(url=args.qdrant_url, timeout=args.timeout)
    try:
        # 1) Source must exist.
        try:
            source_info = await qdrant.get_collection(collection_name=args.source)
        except Exception as e:
            print(f"error: source collection {args.source!r} not found: {e}", file=sys.stderr)
            return 1

        # 2) Target must NOT exist (unless --force).
        cols = {c.name for c in (await qdrant.get_collections()).collections}
        if args.target in cols and not args.force:
            print(
                f"error: target collection {args.target!r} already exists. "
                f"Refusing to clobber — pass --force to delete and recreate.",
                file=sys.stderr,
            )
            return 1

        # 3) Infer vector size + distance from source.
        try:
            vec_size, dist_name = _infer_vector_size_and_distance(source_info)
        except Exception as e:
            print(f"error: cannot infer vector size from source: {e}", file=sys.stderr)
            return 1
        print(f"  source vector size={vec_size}, distance={dist_name}")

        try:
            distance = qm.Distance[dist_name.upper()]
        except KeyError:
            # fall back to raw string (qdrant accepts both enum and string in the params).
            distance = dist_name  # type: ignore[assignment]

        # 4) Plan summary — count source points so dry-run is informative.
        count_resp = await qdrant.count(collection_name=args.source, exact=True)
        source_count = int(count_resp.count)
        print(f"  source has {source_count} points (exact)")

        if not apply:
            print(
                f"  [dry-run] would create {args.target} with "
                f"{{dense: VectorParams(size={vec_size}, distance={dist_name})}} "
                f"+ {{bm25: SparseVectorParams(modifier=IDF)}}"
            )
            print(
                f"  [dry-run] would scroll {args.source} in batches of {args.batch_size}, "
                f"compute bm25 sparse via fastembed, preserve dense + id + payload, "
                f"and upsert into {args.target}"
            )
            print("  [dry-run] pass --apply to execute")
            return 0

        # 5) Apply path: create target.
        if args.target in cols and args.force:
            print(f"  --force set — deleting existing {args.target}")
            try:
                await qdrant.delete_collection(args.target)
            except Exception as e:
                print(f"error: failed to delete existing target: {e}", file=sys.stderr)
                return 1

        try:
            await qdrant.create_collection(
                collection_name=args.target,
                vectors_config={
                    _DENSE_NAME: qm.VectorParams(size=vec_size, distance=distance),
                },
                sparse_vectors_config={
                    _SPARSE_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF),
                },
            )
        except Exception as e:
            print(f"error: failed to create {args.target}: {e}", file=sys.stderr)
            return 1

        # Payload indexes — idempotent on create; match VectorStore's set so
        # later RAG queries aren't slower than the source.
        for field in ("kb_id", "subtag_id", "doc_id", "chat_id", "deleted"):
            try:
                await qdrant.create_payload_index(
                    collection_name=args.target,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

        # 6) Scroll source, compute sparse, upsert into target.
        total = 0
        with_text = 0
        no_text = 0
        offset = None
        while True:
            points, offset = await qdrant.scroll(
                collection_name=args.source,
                limit=args.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break

            # Split batch into "has text" (sparse computed) vs "no text" (dense only).
            upserts: list[qm.PointStruct] = []
            texts_to_embed: list[str] = []
            pending_text_points: list[tuple[object, list[float]]] = []

            for pt in points:
                dense = _dense_vector_for(pt)
                if dense is None:
                    # No vector at all — skip; nothing we can do without re-embed.
                    continue
                payload = dict(pt.payload or {})
                text = payload.get("text")
                if text:
                    texts_to_embed.append(text)
                    pending_text_points.append((pt, dense))
                else:
                    # No text → dense-only point in target (still valid; bm25
                    # arm just contributes nothing for this row).
                    upserts.append(
                        qm.PointStruct(
                            id=pt.id,
                            vector={_DENSE_NAME: dense},
                            payload=payload,
                        )
                    )
                    no_text += 1

            if texts_to_embed:
                sparse_pairs = embed_sparse(texts_to_embed)
                for (pt, dense), (idx, vals) in zip(pending_text_points, sparse_pairs):
                    upserts.append(
                        qm.PointStruct(
                            id=pt.id,
                            vector={
                                _DENSE_NAME: dense,
                                _SPARSE_NAME: qm.SparseVector(
                                    indices=list(idx), values=list(vals)
                                ),
                            },
                            payload=dict(pt.payload or {}),
                        )
                    )
                    with_text += 1

            if upserts:
                await qdrant.upsert(
                    collection_name=args.target, points=upserts, wait=True
                )
                total += len(upserts)

            if offset is None:
                break

        # 7) Verify counts match.
        target_count = int(
            (await qdrant.count(collection_name=args.target, exact=True)).count
        )
        ok = target_count == source_count
        verdict = "OK" if ok else "MISMATCH"
        print(
            f"\n{args.source} -> {args.target}: {total} points reindexed, "
            f"{with_text} with text (sparse computed), "
            f"{no_text} without text (dense only)."
        )
        print(f"  source={source_count} target={target_count} [{verdict}]")
        print(
            f"  target shape: hybrid (named dense + bm25 sparse, IDF modifier). "
            f"Source left in place — swap manually."
        )
        return 0 if ok else 1
    finally:
        await qdrant.close()


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        # argparse exits 2 on validation errors — propagate unchanged.
        return int(e.code) if e.code is not None else 0
    return asyncio.run(_reindex(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
