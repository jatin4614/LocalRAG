#!/usr/bin/env python3
"""Daily hot/warm/cold tier movement for temporally-sharded collections.

Plan B Phase 5.8. For each collection that uses ``shard_key="YYYY-MM"``,
compute the desired tier per shard and apply if changed since last run.

Last-known tier per shard is cached in Redis DB 5 (key: ``tier:<col>:<sk>``).
This avoids re-hitting Qdrant's ``update_collection`` on every run when no
boundary has crossed.

Schedule: invoked by Celery Beat (see ``ext/workers/scheduled_eval.py``)
at 03:00 local. May also be run manually:

    python scripts/tier_storage_cron.py --collection kb_1_v4

Add ``--dry-run`` to print transitions without applying them.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ext.services.vector_store import VectorStore, classify_tier  # noqa: E402


log = logging.getLogger("tier_cron")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


async def discover_shard_keys(qclient: Any, collection: str) -> list[str]:
    """Scroll the collection to learn which shard_keys are populated."""
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = await qclient.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            sk = (p.payload or {}).get("shard_key")
            if sk:
                seen.add(sk)
        if offset is None:
            break
    return sorted(seen)


async def process_collection(
    vs: Any,
    redis_client: Any,
    collection: str,
    shard_keys: list[str],
    *,
    dry_run: bool = False,
) -> dict[str, tuple[str | None, str]]:
    """Apply tier config per shard. Returns map of {shard_key: (previous, desired)}.

    Skips shards where the cached tier already matches the desired tier
    (Redis DB 5 cache key ``tier:<collection>:<shard_key>``).

    When ``dry_run=True``, the desired transitions are computed but
    ``apply_tier_config`` and the Redis ``set`` are skipped.
    """
    transitions: dict[str, tuple[str | None, str]] = {}
    hot_months = int(os.environ.get("RAG_TIER_HOT_MONTHS", "3"))
    warm_months = int(os.environ.get("RAG_TIER_WARM_MONTHS", "12"))

    for sk in shard_keys:
        desired = classify_tier(
            sk, hot_months=hot_months, warm_months=warm_months,
        )
        cache_key = f"tier:{collection}:{sk}"
        previous = await redis_client.get(cache_key)
        if previous == desired:
            continue
        log.info(
            "transition %s/%s: %s -> %s%s",
            collection, sk, previous or "?", desired,
            " [dry-run]" if dry_run else "",
        )
        if not dry_run:
            await vs.apply_tier_config(
                collection=collection, shard_key=sk, tier=desired,
            )
            await redis_client.set(cache_key, desired)
        transitions[sk] = (previous, desired)

    return transitions


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily hot/warm/cold tier movement for temporally-sharded collections.",
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_URL", "redis://redis:6379"),
    )
    parser.add_argument("--redis-db", type=int, default=5)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and log transitions without writing to Qdrant or Redis.",
    )
    args = parser.parse_args()

    vs = VectorStore(url=args.qdrant_url, vector_size=1024)

    # Lazy import: redis is optional at module-import time so the test
    # suite can stub the script without pulling redis into the importer.
    import redis.asyncio as aioredis  # noqa: WPS433
    rc = aioredis.from_url(
        args.redis_url, db=args.redis_db, decode_responses=True,
    )

    shard_keys = await discover_shard_keys(vs._client, args.collection)
    log.info(
        "collection=%s shard_keys=%d dry_run=%s",
        args.collection, len(shard_keys), args.dry_run,
    )

    transitions = await process_collection(
        vs=vs,
        redis_client=rc,
        collection=args.collection,
        shard_keys=shard_keys,
        dry_run=args.dry_run,
    )
    verb = "would apply" if args.dry_run else "applied"
    log.info("%s %d tier transitions", verb, len(transitions))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
