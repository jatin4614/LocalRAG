"""One-shot rebuild of kb_1 into kb_1_rebuild.

Dumps every point (payload + dense vector) from legacy dense-only ``kb_1``,
creates ``kb_1_rebuild`` with the canonical hybrid shape (via
``VectorStore.ensure_collection(with_sparse=True)``), computes BM25 sparse
vectors locally via fastembed (Qdrant/bm25), and upserts the full set
preserving original UUID point IDs + all payload fields verbatim.

Does NOT drop kb_1. The alias swap is a separate manual step after verification.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from ext.services.vector_store import VectorStore
from ext.services.sparse_embedder import embed_sparse


SRC = "kb_1"
DST = "kb_1_rebuild"
DUMP_PATH = Path("/tmp/kb_1_dump.jsonl")
BATCH = 128


async def dump_source(client: AsyncQdrantClient) -> list[dict]:
    print(f"[dump] scrolling {SRC} ...")
    t0 = time.perf_counter()
    all_points: list[dict] = []
    next_offset = None
    while True:
        resp, next_offset = await client.scroll(
            collection_name=SRC,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in resp:
            all_points.append({
                "id": str(p.id),
                "payload": p.payload or {},
                "vector": list(p.vector) if isinstance(p.vector, list) else p.vector,
            })
        if next_offset is None:
            break
    dt = time.perf_counter() - t0
    print(f"[dump] {len(all_points)} points in {dt:.1f}s")
    # Persist to disk as rollback artifact.
    with DUMP_PATH.open("w") as f:
        for p in all_points:
            f.write(json.dumps({"id": p["id"], "payload": p["payload"]}) + "\n")
    print(f"[dump] payloads saved to {DUMP_PATH}")
    return all_points


async def create_target(vs: VectorStore) -> None:
    print(f"[create] creating {DST} with_sparse=True ...")
    await vs.ensure_collection(DST, with_sparse=True)
    print(f"[create] done")


def compute_sparse(texts: list[str]) -> list[tuple[list[int], list[float]] | None]:
    """Compute BM25 sparse vectors for a list of texts. None for empty strings."""
    # fastembed handles batching internally; but huge single calls are slow.
    # Do chunks of 256 to keep memory bounded.
    out: list[tuple[list[int], list[float]] | None] = [None] * len(texts)
    idxs = [i for i, t in enumerate(texts) if t]
    nonempty = [texts[i] for i in idxs]
    print(f"[sparse] embedding {len(nonempty)} texts (of {len(texts)}) ...")
    t0 = time.perf_counter()
    BS = 256
    pos = 0
    for i in range(0, len(nonempty), BS):
        batch = nonempty[i:i+BS]
        results = embed_sparse(batch)
        for r in results:
            out[idxs[pos]] = r
            pos += 1
        print(f"[sparse]   {pos}/{len(nonempty)}")
    print(f"[sparse] done in {time.perf_counter()-t0:.1f}s")
    return out


async def upsert_target(vs: VectorStore, points: list[dict], sparses: list) -> None:
    print(f"[upsert] {len(points)} points into {DST} ...")
    t0 = time.perf_counter()
    # Warm sparse cache so VectorStore.upsert picks the hybrid path.
    await vs._refresh_sparse_cache(DST)
    assert vs._collection_has_sparse(DST), "target collection missing sparse vectors!"

    pt_buf: list[dict] = []
    total = 0
    for p, sv in zip(points, sparses):
        item = {
            "id": p["id"],
            "vector": p["vector"],
            "payload": p["payload"],
        }
        if sv is not None:
            item["sparse_vector"] = sv
        pt_buf.append(item)
        if len(pt_buf) >= BATCH:
            await vs.upsert(DST, pt_buf)
            total += len(pt_buf)
            print(f"[upsert]   {total}/{len(points)}")
            pt_buf = []
    if pt_buf:
        await vs.upsert(DST, pt_buf)
        total += len(pt_buf)
        print(f"[upsert]   {total}/{len(points)}")
    print(f"[upsert] done in {time.perf_counter()-t0:.1f}s")


async def main() -> None:
    url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    raw = AsyncQdrantClient(url=url, timeout=60.0)
    vs = VectorStore(url=url, vector_size=1024)

    t_all = time.perf_counter()
    points = await dump_source(raw)

    # Filter soft-deleted if any
    live = [p for p in points if not p["payload"].get("deleted")]
    print(f"[dump] {len(live)} live points (excluded {len(points)-len(live)} soft-deleted)")

    # Ensure/create target.
    await create_target(vs)

    # Compute sparse from payload text.
    texts = [p["payload"].get("text", "") or "" for p in live]
    sparses = compute_sparse(texts)

    await upsert_target(vs, live, sparses)

    # Final stats.
    info = await raw.get_collection(DST)
    print("\n=== FINAL STATS ===")
    print(f"points_count          = {info.points_count}")
    print(f"indexed_vectors_count = {info.indexed_vectors_count}")
    print(f"vectors_count         = {info.vectors_count}")
    for field, schema in (info.payload_schema or {}).items():
        print(f"payload_schema[{field}] = points={schema.points}")

    print(f"\nTOTAL elapsed: {time.perf_counter()-t_all:.1f}s")

    await vs.close()
    await raw.close()


if __name__ == "__main__":
    asyncio.run(main())
