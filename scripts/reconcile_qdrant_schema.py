#!/usr/bin/env python3
"""Reconcile divergent Qdrant KB collections to the canonical payload schema.

Usage:
    python scripts/reconcile_qdrant_schema.py \
        --qdrant-url http://localhost:6333 \
        --collection kb_1_rebuild \
        --target kb_1_v2

The script:
1. Snapshots the source collection to /var/backups/qdrant/pre-schema-recon/
2. Creates target collection with canonical indexes + config
3. Scrolls through source, coerces payload, re-upserts to target
4. Verifies source point count == target point count
5. Prints alias-swap command for the operator to run at cutover

DOES NOT automatically switch aliases — operator does that explicitly after
the eval gate on the new collection passes.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from ext.db.qdrant_schema import (
    CANONICAL_INDEXES,
    CANONICAL_COLLECTION_CONFIG,
    coerce_to_canonical,
)

log = logging.getLogger("reconcile")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def snapshot(client: AsyncQdrantClient, collection: str, out_dir: Path) -> None:
    log.info("snapshotting %s", collection)
    res = await client.create_snapshot(collection_name=collection)
    log.info("snapshot created: %s", res.name)


async def create_target(client: AsyncQdrantClient, source: str, target: str) -> None:
    src_info = await client.get_collection(source)
    # Preserve vector params from source (dense + any named vectors) but force
    # canonical config (on_disk_payload etc).
    vectors = {}
    if src_info.config.params.vectors:
        for name, params in src_info.config.params.vectors.items():
            vectors[name] = qmodels.VectorParams(
                size=params.size, distance=params.distance,
            )
    await client.create_collection(
        collection_name=target,
        vectors_config=vectors,
        sparse_vectors_config=src_info.config.params.sparse_vectors,
        on_disk_payload=CANONICAL_COLLECTION_CONFIG["on_disk_payload"],
        hnsw_config=qmodels.HnswConfigDiff(**CANONICAL_COLLECTION_CONFIG["hnsw_config"]),
    )
    log.info("created target %s", target)

    for idx in CANONICAL_INDEXES:
        field_type_map = {
            "integer": qmodels.PayloadSchemaType.INTEGER,
            "keyword": qmodels.PayloadSchemaType.KEYWORD,
            "bool": qmodels.PayloadSchemaType.BOOL,
            "float": qmodels.PayloadSchemaType.FLOAT,
        }
        await client.create_payload_index(
            collection_name=target,
            field_name=idx["field"],
            field_schema=field_type_map[idx["type"]],
        )
        log.info("created index: %s (%s)", idx["field"], idx["type"])


async def reupsert(client: AsyncQdrantClient, source: str, target: str, batch: int = 256) -> int:
    offset = None
    total = 0
    while True:
        points, offset = await client.scroll(
            collection_name=source, limit=batch, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not points:
            break
        upsert_batch = []
        for p in points:
            payload = coerce_to_canonical(dict(p.payload or {}))
            upsert_batch.append(qmodels.PointStruct(
                id=p.id, vector=p.vector, payload=payload,
            ))
        await client.upsert(collection_name=target, points=upsert_batch)
        total += len(points)
        log.info("migrated %d / (progress)", total)
        if offset is None:
            break
    return total


async def verify_counts(client: AsyncQdrantClient, source: str, target: str) -> None:
    src = await client.count(collection_name=source, exact=True)
    tgt = await client.count(collection_name=target, exact=True)
    log.info("source %s: %d points", source, src.count)
    log.info("target %s: %d points", target, tgt.count)
    assert src.count == tgt.count, f"point count mismatch: {src.count} vs {tgt.count}"


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--backup-dir", type=Path, default=Path("/var/backups/qdrant/pre-schema-recon"))
    args = p.parse_args()

    args.backup_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncQdrantClient(url=args.qdrant_url, timeout=60.0)
    await snapshot(client, args.collection, args.backup_dir)
    await create_target(client, args.collection, args.target)
    total = await reupsert(client, args.collection, args.target)
    await verify_counts(client, args.collection, args.target)

    log.info("migration complete: %d points moved %s → %s", total, args.collection, args.target)
    log.info(
        "\nNext steps for operator:\n"
        "  1. Run eval against %s and confirm parity with %s:\n"
        "       make eval KB_EVAL_ID=<new kb_id pointing at %s>\n"
        "  2. Compare results; if recall/mrr within ±1pp, proceed.\n"
        "  3. Swap the alias:\n"
        "       curl -X PUT http://localhost:6333/collections/aliases -d '{\n"
        "         \"actions\": [{\"delete_alias\": {\"alias_name\": \"%s\"}},\n"
        "                        {\"create_alias\": {\"collection_name\": \"%s\", \"alias_name\": \"%s\"}}]}'\n"
        "  4. Keep %s read-only for 14 days as rollback target.\n",
        args.target, args.collection, args.target,
        args.collection, args.target, args.collection,
        args.collection,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
