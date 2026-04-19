#!/usr/bin/env python3
"""Upgrade path stub: attach sparse BM25 vectors to an existing collection.

This script is the partial/interim tool for P1.1. It assumes the target
collection was already created with the hybrid shape (named ``dense`` +
``bm25`` sparse named vector). For every point in the collection, it
computes a sparse BM25 vector from ``payload.text`` via fastembed and
re-upserts the point with the sparse vector attached.

For the common case — a legacy dense-only collection that needs to be
converted to hybrid — this script will emit a clear error and point at the
(yet to be written) ``scripts/reindex_hybrid.py`` P2 reindex job. Qdrant does
not currently allow adding a sparse named vector to an existing collection
in-place (see https://qdrant.tech/documentation/concepts/collections/ —
``UpdateCollection`` cannot change vector shape).

Usage:
    python scripts/add_sparse_to_collection.py \\
        --qdrant-url http://localhost:6333 --collection kb_eval [--apply] [--batch-size 64]

Exit codes:
    0  done (or dry-run finished)
    1  collection lacks sparse named vector — reindex required (P2)
    2  collection not found or other Qdrant error
    4  fastembed not installed
"""
from __future__ import annotations

import argparse
import os
import sys

# When invoked directly from the repo root, make the package importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", required=True, help="Collection name to upgrade")
    p.add_argument("--apply", action="store_true", help="Perform upsert (default: dry run)")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("error: qdrant-client not installed", file=sys.stderr)
        return 4
    try:
        from ext.services.sparse_embedder import embed_sparse
    except ImportError as e:
        print(f"error: cannot import sparse embedder ({e}); install '.[hybrid]'", file=sys.stderr)
        return 4

    client = QdrantClient(url=args.qdrant_url)

    try:
        info = client.get_collection(collection_name=args.collection)
    except Exception as e:
        print(f"error: collection {args.collection!r} not accessible: {e}", file=sys.stderr)
        return 2

    sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
    if not sparse_cfg or "bm25" not in sparse_cfg:
        print(
            f"error: collection {args.collection!r} has no 'bm25' sparse named vector.\n"
            f"       Qdrant does not allow adding sparse vectors to an existing\n"
            f"       collection in-place. Run the full reindex via\n"
            f"       scripts/reindex_hybrid.py (scheduled for P2) which creates a\n"
            f"       fresh hybrid collection and re-ingests every document.",
            file=sys.stderr,
        )
        return 1

    # Scroll through every point and re-upsert with sparse.
    from qdrant_client.http import models as qm

    total = 0
    skipped = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            limit=args.batch_size,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        if not points:
            break
        batch_texts: list[str] = []
        batch_keep: list = []
        for pt in points:
            text = (pt.payload or {}).get("text")
            if not text:
                skipped += 1
                continue
            batch_texts.append(text)
            batch_keep.append(pt)
        if not batch_texts:
            if offset is None:
                break
            continue
        sparse = embed_sparse(batch_texts)
        upserts = []
        for pt, (idx, vals) in zip(batch_keep, sparse):
            # Preserve the existing dense vector (pt.vector is a dict because
            # the collection has named vectors).
            if isinstance(pt.vector, dict):
                vec_map = dict(pt.vector)
            else:  # defensive — shouldn't happen on a hybrid collection
                vec_map = {"dense": pt.vector}
            vec_map["bm25"] = qm.SparseVector(indices=idx, values=vals)
            upserts.append(qm.PointStruct(id=pt.id, vector=vec_map, payload=pt.payload))
        if args.apply:
            client.upsert(collection_name=args.collection, points=upserts, wait=True)
        total += len(upserts)
        if offset is None:
            break

    verb = "upserted" if args.apply else "would upsert (dry-run)"
    print(f"{verb} {total} points; skipped {skipped} (missing payload.text)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
