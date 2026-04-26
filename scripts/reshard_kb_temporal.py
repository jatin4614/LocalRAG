#!/usr/bin/env python3
"""Reshard a Qdrant collection into per-month custom shards.

Plan B Phase 5.4. Source: existing Plan-A canonical collection (e.g.
``kb_1_v3``). Target: new collection with ``sharding_method=custom`` and
``shard_key="YYYY-MM"`` per month.

Process:
  1. Scroll source, derive shard_key per point from filename + body.
  2. Group points by shard_key.
  3. Create target with ``ensure_collection_temporal(name, all_shard_keys)``.
  4. Upsert each shard's points via the per-shard ``shard_key_selector``.
  5. Verify per-shard counts.

Vectors carried over: dense (always), sparse (if present), colbert (if
present). Payload preserved as-is, plus ``shard_key`` + ``shard_key_origin``
appended.

Usage:
    python scripts/reshard_kb_temporal.py \\
        --source kb_1_v3 \\
        --target kb_1_v4 \\
        --qdrant-url http://localhost:6333

If you also want to apply the alias swap atomically: pass
``--swap-alias kb_1``. The alias swap is OPTIONAL and not done by default
— operator should run eval first, then swap manually.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import pathlib
import sys

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    CreateAlias, CreateAliasOperation,
    DeleteAlias, DeleteAliasOperation,
    PointStruct,
)

# Ensure we can import ext.services from a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ext.services.temporal_shard import extract_shard_key  # noqa: E402
from ext.services.vector_store import VectorStore  # noqa: E402


log = logging.getLogger("reshard")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


async def collect_source_points(client, collection: str, batch_size: int):
    """Yield (point, derived_shard_key, origin) tuples by scrolling source."""
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        for p in points:
            payload = dict(p.payload or {})
            sk, origin = extract_shard_key(
                filename=payload.get("filename", ""),
                body=payload.get("text", ""),
            )
            yield p, sk, origin
        if offset is None:
            break


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--swap-alias",
        default=None,
        help="if set, swap this alias to the new collection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate but do not write target",
    )
    args = parser.parse_args()

    qc = AsyncQdrantClient(url=args.qdrant_url, timeout=300.0)
    vs = VectorStore.__new__(VectorStore)
    vs._url = args.qdrant_url
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    vs._colbert_cache = {}
    vs._client = qc

    log.info("source=%s target=%s", args.source, args.target)

    if not await qc.collection_exists(collection_name=args.source):
        log.error("source collection does not exist: %s", args.source)
        return 2

    # Discover the source vectors config so we know which heads to carry over
    src_info = await qc.get_collection(collection_name=args.source)
    has_sparse = bool(getattr(src_info.config.params, "sparse_vectors", None))
    vectors_attr = src_info.config.params.vectors
    vector_names: list[str] = []
    if isinstance(vectors_attr, dict):
        vector_names = list(vectors_attr.keys())
    has_colbert = any(n.lower().startswith("colbert") for n in vector_names)
    log.info("source has_sparse=%s has_colbert=%s vectors=%s",
             has_sparse, has_colbert, vector_names)

    # Pass 1: enumerate shard_keys + bucket points
    log.info("Pass 1: scrolling source to enumerate shard_keys...")
    buckets: dict[str, list[PointStruct]] = collections.defaultdict(list)
    origin_counter: collections.Counter = collections.Counter()
    total = 0
    async for p, sk, origin in collect_source_points(
        qc, args.source, args.batch_size,
    ):
        payload = dict(p.payload or {})
        payload["shard_key"] = sk
        payload["shard_key_origin"] = origin.value
        buckets[sk].append(
            PointStruct(id=p.id, vector=p.vector, payload=payload)
        )
        origin_counter[origin.value] += 1
        total += 1
    log.info("collected %d points across %d shard_keys", total, len(buckets))
    log.info("shard_key origin distribution: %s", dict(origin_counter))

    if args.dry_run:
        log.info("DRY RUN — exiting before target creation")
        for sk in sorted(buckets):
            log.info("  shard_key %s: %d points", sk, len(buckets[sk]))
        return 0

    # Pass 2: create target collection
    shard_keys = sorted(buckets.keys())
    log.info("creating target %s with %d shard keys", args.target, len(shard_keys))
    await vs.ensure_collection_temporal(
        args.target,
        shard_keys=shard_keys,
        with_sparse=has_sparse,
        with_colbert=has_colbert,
    )

    # Pass 3: upsert per shard, chunked to avoid 400s on large requests.
    # ColBERT multi-vectors blow up the request body — keep batches small.
    UPSERT_CHUNK = 64
    for sk in shard_keys:
        points = buckets[sk]
        log.info("upserting shard_key=%s (%d points, chunk=%d)",
                 sk, len(points), UPSERT_CHUNK)
        for i in range(0, len(points), UPSERT_CHUNK):
            chunk = points[i:i + UPSERT_CHUNK]
            try:
                await qc.upsert(
                    collection_name=args.target,
                    points=chunk,
                    shard_key_selector=sk,
                )
            except Exception as e:
                log.error("upsert chunk %d-%d failed: %s",
                          i, i + len(chunk), repr(e)[:300])
                raise

    # Verify counts
    src_count = (await qc.count(collection_name=args.source)).count
    tgt_count = (await qc.count(collection_name=args.target)).count
    log.info("verify: source=%d target=%d", src_count, tgt_count)
    if src_count != tgt_count:
        log.error("count mismatch — investigate before proceeding")
        return 3

    # Optional alias swap (operator-gated; default is no-op)
    if args.swap_alias:
        log.warning("swapping alias %s -> %s", args.swap_alias, args.target)
        await qc.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=args.swap_alias)
                ),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=args.target,
                        alias_name=args.swap_alias,
                    )
                ),
            ],
        )
        log.info("alias swap complete")

    log.info("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
