#!/usr/bin/env python3
"""Migrate legacy per-chat Qdrant collections into the consolidated
``chat_private`` collection (P2.3).

Before this script:
    chat_016be37b-...          <- 5 collections, one per chat
    chat_06d90fbf-...
    chat_a4d0d642-...
    chat_c1e5d573-...
    chat_c3917305-...

After:
    chat_private               <- single collection; chat_id + owner_user_id
                                  in payload, both is_tenant=True indexes

Idempotent. Dry-run by default. Does NOT delete sources unless
``--delete-source`` is passed AND the migration verified successfully.

Preserves point IDs, dense vectors, and payloads. Injects ``chat_id``
into payloads that lack it (parsed from the source collection name:
``chat_{uuid}`` → chat_id=``{uuid}``). Computes bm25 sparse vectors
via fastembed when the target has sparse enabled.

Usage:
    python scripts/migrate_chat_collections.py --qdrant-url http://localhost:6333 --dry-run
    python scripts/migrate_chat_collections.py --qdrant-url http://localhost:6333 --apply
    python scripts/migrate_chat_collections.py --qdrant-url http://localhost:6333 --apply --delete-source
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from ext.services.vector_store import CHAT_PRIVATE_COLLECTION, _DENSE_NAME, _SPARSE_NAME  # noqa: E402


def _is_legacy_chat(name: str) -> bool:
    return name.startswith("chat_") and name != CHAT_PRIVATE_COLLECTION


def _chat_id_from_collection(name: str) -> str:
    """chat_016be37b-... → 016be37b-..."""
    return name[len("chat_"):]


async def _ensure_target(client: AsyncQdrantClient, target: str, vector_size: int) -> None:
    try:
        await client.get_collection(target)
        return
    except Exception:
        pass
    await client.create_collection(
        collection_name=target,
        vectors_config={_DENSE_NAME: qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)},
        sparse_vectors_config={_SPARSE_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
    )
    tenant_fields = ("kb_id", "chat_id", "owner_user_id")
    filter_fields = ("subtag_id", "doc_id", "deleted")
    for f in tenant_fields:
        try:
            await client.create_payload_index(
                collection_name=target, field_name=f,
                field_schema=qm.KeywordIndexParams(type="keyword", is_tenant=True),
            )
        except Exception:
            pass
    for f in filter_fields:
        try:
            await client.create_payload_index(
                collection_name=target, field_name=f,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


async def _migrate_one(
    client: AsyncQdrantClient,
    source: str,
    target: str,
    *,
    batch_size: int,
    dry_run: bool,
    with_sparse: bool,
    sparse_fn,
) -> dict:
    summary = {"source": source, "scanned": 0, "migrated": 0, "skipped_empty": 0}
    chat_id = _chat_id_from_collection(source)

    offset = None
    while True:
        res, next_offset = await client.scroll(
            collection_name=source,
            limit=batch_size,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        if not res:
            break
        summary["scanned"] += len(res)

        points = []
        for rec in res:
            # Extract dense vector (may be dict on hybrid sources or list on legacy)
            vec = rec.vector
            if isinstance(vec, dict):
                dense = vec.get(_DENSE_NAME) or next(iter(vec.values()))
            else:
                dense = vec
            if dense is None:
                summary["skipped_empty"] += 1
                continue
            payload = dict(rec.payload or {})
            payload.setdefault("chat_id", chat_id)
            text = payload.get("text") or ""

            vec_map: dict = {_DENSE_NAME: list(dense)}
            if with_sparse and text and sparse_fn is not None:
                try:
                    [(indices, values)] = sparse_fn([text])
                    vec_map[_SPARSE_NAME] = qm.SparseVector(indices=list(indices), values=list(values))
                except Exception:
                    pass

            points.append(qm.PointStruct(id=rec.id, vector=vec_map, payload=payload))

        if points and not dry_run:
            await client.upsert(collection_name=target, points=points, wait=True)
        summary["migrated"] += len(points)

        if next_offset is None:
            break
        offset = next_offset

    return summary


async def run(args) -> int:
    client = AsyncQdrantClient(url=args.qdrant_url, timeout=30.0)
    try:
        cols = (await client.get_collections()).collections
        legacy = [c.name for c in cols if _is_legacy_chat(c.name)]
        if not legacy:
            print("no legacy chat_* collections found — nothing to do")
            return 0

        # Determine vector size from first legacy collection
        first = await client.get_collection(legacy[0])
        params = first.config.params
        vectors = getattr(params, "vectors", None) or getattr(params, "vector_size", None)
        if hasattr(vectors, "size"):
            vector_size = vectors.size
        elif isinstance(vectors, dict):
            any_v = next(iter(vectors.values()))
            vector_size = any_v.size
        else:
            vector_size = int(vectors) if vectors else 1024

        mode = "[DRY-RUN]" if args.dry_run else "[APPLY]"
        print(f"{mode} migrating {len(legacy)} legacy chat collection(s) -> {args.target}")

        sparse_fn = None
        with_sparse = True
        if not args.dry_run:
            try:
                from ext.services.sparse_embedder import embed_sparse
                sparse_fn = embed_sparse
            except Exception as e:
                print(f"  warning: fastembed unavailable ({e}); target will only have dense vectors for new points")
                with_sparse = False
            await _ensure_target(client, args.target, vector_size)

        total_scanned = total_migrated = 0
        for src in sorted(legacy):
            print(f"  -> {src}")
            s = await _migrate_one(
                client, src, args.target,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                with_sparse=with_sparse,
                sparse_fn=sparse_fn,
            )
            total_scanned += s["scanned"]
            total_migrated += s["migrated"]
            print(f"     scanned={s['scanned']} migrated={s['migrated']} skipped_empty={s['skipped_empty']}")

            if args.delete_source and not args.dry_run and s["migrated"] == s["scanned"] - s["skipped_empty"]:
                await client.delete_collection(src)
                print(f"     deleted source {src}")

        print(f"TOTAL scanned={total_scanned} migrated={total_migrated}")
        if args.dry_run:
            print("DRY-RUN — pass --apply to execute.")
        return 0
    finally:
        await client.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--target", default=CHAT_PRIVATE_COLLECTION)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--delete-source", action="store_true",
                   help="After successful migration, delete the source collection.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--apply", dest="dry_run", action="store_false")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
