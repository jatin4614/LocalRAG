#!/usr/bin/env python3
"""Re-embed chunks in-place against a target pipeline version.

Phase 4 deliverable — operational tool for future embedder swaps
(e.g. Phase 1b when Harrier lands). NOT used by the Phase 1a cycle.

What it does:
  1. Scrolls every point in the target KB collection (or all KB
     collections with ``--all``).
  2. For each chunk, reads ``payload.text`` and re-runs it through the
     configured TEI embedder.
  3. Upserts the point back with the new dense vector, preserving
     sparse vectors and payload. Stamps ``payload.pipeline_version`` to
     the CLI-provided ``--pipeline-version``.

What it does NOT do:
  * Re-extract the source document. The original bytes are NOT read.
  * Re-chunk. Chunk boundaries / text are untouched.
  * Re-compute sparse (BM25) vectors — they already reflect the chunk
    text and are stable across embedder swaps.
  * Migrate vector dimensionality. The target collection must have the
    correct size already (handle that by creating a fresh v2 collection
    per ``scripts/reindex_hybrid.py`` and pointing this at it).

Usage:
  # Dry-run against kb_1 — counts points, prints plan, no writes.
  python scripts/reembed_all.py --kb 1 --pipeline-version \\
      'chunker=v3|extractor=v2|embedder=harrier-0.6b|ctx=none'

  # Apply.
  python scripts/reembed_all.py --kb 1 --apply --pipeline-version \\
      'chunker=v3|extractor=v2|embedder=harrier-0.6b|ctx=none'

  # Apply across every KB collection.
  python scripts/reembed_all.py --all --apply --pipeline-version \\
      'chunker=v3|extractor=v2|embedder=harrier-0.6b|ctx=none'

Exit codes:
    0  success (or dry-run finished)
    1  any error — source missing, embedder failure, Qdrant error
    4  invalid arguments
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Make ``ext.*`` importable when invoked from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_DENSE_NAME = "dense"
_SPARSE_NAME = "bm25"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-embed existing Qdrant chunks against a new embedder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant REST URL (default: $QDRANT_URL or http://localhost:6333).",
    )
    p.add_argument(
        "--tei-url",
        default=os.environ.get("TEI_URL", "http://localhost:8080"),
        help="TEI base URL (default: $TEI_URL or http://localhost:8080).",
    )

    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--kb",
        type=int,
        help="Single KB id — rebuilds collection kb_<id>.",
    )
    which.add_argument(
        "--all",
        action="store_true",
        help="Every collection whose name starts with kb_.",
    )

    p.add_argument(
        "--pipeline-version",
        required=True,
        help=(
            "Target pipeline version string stamped on re-embedded points "
            "(e.g. 'chunker=v3|extractor=v2|embedder=harrier-0.6b|ctx=none')."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Points scrolled / re-embedded / upserted per round (default: 64).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for Qdrant/TEI calls (default: 60s).",
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
        help="Actually re-embed and upsert. Opt-in.",
    )

    return p.parse_args(argv)


def _extract_vector_kind(point: Any) -> tuple[Optional[list[float]], Optional[Any]]:
    """Return (dense, sparse_raw) from a scrolled point.

    Hybrid collections store vectors as ``{dense: list, bm25: SparseVector}``.
    Legacy collections store just a list. Sparse is returned as the raw
    SparseVector-ish object (qdrant-client passes them through);
    up-serting accepts the same type so we don't round-trip.
    """
    v = getattr(point, "vector", None)
    if v is None:
        return (None, None)
    if isinstance(v, dict):
        dense = v.get(_DENSE_NAME)
        sparse = v.get(_SPARSE_NAME)
        if isinstance(dense, list):
            return ([float(x) for x in dense], sparse)
        # Fall back to first list value.
        for val in v.values():
            if isinstance(val, list):
                return ([float(x) for x in val], sparse)
        return (None, sparse)
    if isinstance(v, list):
        return ([float(x) for x in v], None)
    return (None, None)


async def _enumerate_collections(qdrant: Any, *, kb: Optional[int], all_kbs: bool) -> list[str]:
    if kb is not None:
        return [f"kb_{kb}"]
    if all_kbs:
        cols = (await qdrant.get_collections()).collections
        return sorted(c.name for c in cols if c.name.startswith("kb_"))
    return []


def _sparse_to_tuple(sparse: Any) -> Optional[tuple[list[int], list[float]]]:
    """Coerce a Qdrant SparseVector-ish object back to ``(indices, values)``.

    VectorStore.upsert expects the tuple form (indices, values); points
    scrolled from Qdrant come back as ``models.SparseVector`` objects
    with ``.indices`` and ``.values`` attributes (or as a dict in some
    qdrant-client versions). Returns None for absent / unrecognized.
    """
    if sparse is None:
        return None
    indices = getattr(sparse, "indices", None)
    values = getattr(sparse, "values", None)
    if indices is None or values is None:
        # Older / mocked shapes may be plain dicts.
        if isinstance(sparse, dict):
            indices = sparse.get("indices")
            values = sparse.get("values")
    if indices is None or values is None:
        return None
    return ([int(i) for i in indices], [float(v) for v in values])


async def _reembed_collection(
    collection: str,
    *,
    qdrant: Any,
    embedder: Any,
    vector_store: Any,
    pipeline_version: str,
    batch_size: int,
    apply: bool,
) -> tuple[int, int]:
    """Return (total_points, re_embedded). Re-embedded may be 0 in dry-run.

    Bug-fix campaign §3.3: writes route through ``VectorStore.upsert``
    (instead of a raw qdrant client upsert from the original script) so
    custom-sharded targets like ``kb_1_v4`` (live since 2026-04-26)
    auto-derive ``shard_key_selector`` from each point's
    ``payload['shard_key']``. Without that, a raw upsert returns
    ``Shard key not specified`` 400 and the operator script blows up
    after the first scrolled page.
    """
    total = 0
    re_embedded = 0
    offset = None
    print(f"  [{'APPLY' if apply else 'DRY-RUN'}] scrolling {collection} …")
    while True:
        page, offset = await qdrant.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not page:
            break

        texts: list[str] = []
        slot_for_idx: list[int] = []
        new_points: list[Any] = []

        for i, pt in enumerate(page):
            dense, sparse = _extract_vector_kind(pt)
            payload = dict(pt.payload or {})
            text = payload.get("text")
            if not text or dense is None:
                # Nothing to re-embed with or no current dense to replace.
                total += 1
                continue
            texts.append(str(text))
            slot_for_idx.append(i)
            new_points.append({
                "id": pt.id,
                "payload": payload,
                "sparse": sparse,
                "_dense_old": dense,
            })
            total += 1

        if not texts:
            if offset is None:
                break
            continue

        if apply:
            new_vecs = await embedder.embed(texts)
            if len(new_vecs) != len(new_points):
                raise RuntimeError(
                    f"embedder returned {len(new_vecs)} vectors for {len(new_points)} texts"
                )

            # VectorStore.upsert takes a list of dicts shaped:
            #   {id, vector (dense list), payload, sparse_vector?, colbert_vector?}
            # It handles named-vector encoding and shard_key derivation
            # internally, so we just hand it the data and let it pick the
            # right path. Custom-sharded collections require each point's
            # payload to carry ``shard_key`` — already preserved here
            # because we copy the original payload through unchanged
            # (only stamping ``pipeline_version`` on top).
            upserts: list[dict] = []
            for np_dict, new_vec in zip(new_points, new_vecs):
                updated_payload = dict(np_dict["payload"])
                updated_payload["pipeline_version"] = pipeline_version
                pt_dict: dict[str, Any] = {
                    "id": np_dict["id"],
                    "vector": list(new_vec),
                    "payload": updated_payload,
                }
                sparse_tup = _sparse_to_tuple(np_dict["sparse"])
                if sparse_tup is not None:
                    pt_dict["sparse_vector"] = sparse_tup
                upserts.append(pt_dict)

            await vector_store.upsert(collection, upserts)
            re_embedded += len(upserts)
        else:
            re_embedded += len(texts)

        if offset is None:
            break

    return (total, re_embedded)


async def _main(args: argparse.Namespace) -> int:
    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError as e:
        print(f"error: qdrant-client not installed: {e}", file=sys.stderr)
        return 1

    try:
        from ext.services.embedder import TEIEmbedder
        # Bug-fix campaign §3.3 — route writes through VectorStore so
        # custom-sharded collections (kb_1_v4 onward) auto-derive
        # shard_key_selector from each point's payload. The raw qdrant
        # client upsert path bypassed that and 400'd on first apply.
        from ext.services.vector_store import VectorStore
    except ImportError as e:
        print(f"error: cannot import dependencies: {e}", file=sys.stderr)
        return 1

    apply = bool(args.apply)
    banner = "APPLY" if apply else "DRY-RUN"
    print(f"[{banner}] reembed target pipeline_version={args.pipeline_version!r}")

    qdrant = AsyncQdrantClient(url=args.qdrant_url, timeout=args.timeout)
    embedder = TEIEmbedder(base_url=args.tei_url, timeout=args.timeout)
    # vector_size on the wrapper isn't consulted on upsert (the dense
    # vector dimensionality follows the embedder output), so 0 is fine
    # — but pick the real bge-m3 size to keep diagnostic logs sane.
    vector_store = VectorStore(url=args.qdrant_url, vector_size=1024)
    try:
        collections = await _enumerate_collections(
            qdrant, kb=args.kb, all_kbs=args.all,
        )
        if not collections:
            print("no collections matched (pass --kb or --all)", file=sys.stderr)
            return 1
        print(f"  target collections: {collections}")

        total_points = 0
        total_reembedded = 0
        for col in collections:
            try:
                t, r = await _reembed_collection(
                    col,
                    qdrant=qdrant,
                    embedder=embedder,
                    vector_store=vector_store,
                    pipeline_version=args.pipeline_version,
                    batch_size=args.batch_size,
                    apply=apply,
                )
                total_points += t
                total_reembedded += r
                print(
                    f"  {col}: {t} points scanned, "
                    f"{r} {'re-embedded' if apply else 'would be re-embedded'}"
                )
            except Exception as e:
                print(f"error: collection {col!r} failed: {e}", file=sys.stderr)
                return 1

        print(
            f"\n{banner}: total points={total_points}, "
            f"re-embedded={total_reembedded}"
        )
        return 0
    finally:
        try:
            await embedder.aclose()
        except Exception:
            pass
        try:
            await vector_store.close()
        except Exception:
            pass
        try:
            await qdrant.close()
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return asyncio.run(_main(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
