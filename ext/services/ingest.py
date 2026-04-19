"""Extract → chunk → embed → upsert pipeline.

Each extracted block carries structural metadata (``page`` / ``heading_path`` /
``sheet``). We chunk the block's text independently and inherit the block's
metadata onto every resulting chunk so Qdrant payloads can surface hints like
"from page 7" or "under heading 'Rollout'" at retrieval time.
"""
from __future__ import annotations

import time
import uuid
from typing import Mapping

from .chunker import chunk_text
from .embedder import Embedder
from .extractor import extract
from .pipeline_version import current_version
from .vector_store import VectorStore

# Stable namespace for deterministic point IDs (UUID5 based on doc_id + chunk_index).
# Using the well-known URL namespace UUID so the value is fixed across deploys.
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


async def ingest_bytes(
    *,
    data: bytes,
    mime_type: str,
    filename: str,
    collection: str,
    payload_base: Mapping[str, int | str],
    vector_store: VectorStore,
    embedder: Embedder,
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> int:
    """Full ingest: returns number of chunks upserted."""
    blocks = extract(data, mime_type, filename)
    if not blocks:
        return 0

    # Chunk per block; carry the source block forward so we can stamp its
    # structural metadata onto each resulting chunk.
    paired: list[tuple[object, object]] = []  # (Chunk, ExtractedBlock)
    for b in blocks:
        for c in chunk_text(
            b.text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
        ):
            paired.append((c, b))
    if not paired:
        return 0

    texts = [c.text for c, _ in paired]
    vectors = await embedder.embed(texts)

    now = time.time_ns()
    pv = current_version()

    # Defensive coercion — main historically passed str(doc.id); we now store
    # doc_id as int consistently. If the caller supplied a numeric string
    # (legacy callers, worker retries), coerce it; non-numeric values are
    # left untouched so we don't mask genuine misuse.
    if "doc_id" in payload_base and payload_base["doc_id"] is not None:
        try:
            payload_base = {**payload_base, "doc_id": int(payload_base["doc_id"])}
        except (ValueError, TypeError):
            pass  # non-numeric doc_id — leave as-is (shouldn't happen in practice)

    doc_id = payload_base.get("doc_id")
    chat_id = payload_base.get("chat_id")

    points = []
    for gidx, ((chunk, block), vec) in enumerate(zip(paired, vectors)):
        payload = dict(payload_base)
        payload["chunk_index"] = gidx
        payload["text"] = chunk.text
        payload["uploaded_at"] = now
        payload["deleted"] = False
        # Structural metadata from the source block (may be None / []).
        payload["page"] = block.page
        payload["heading_path"] = list(block.heading_path)
        payload["sheet"] = block.sheet
        payload["model_version"] = pv

        # Deterministic point ID: same doc + global chunk index always maps
        # to the same Qdrant point. This lets delete_by_doc reconstruct point
        # IDs or use payload filtering to remove vectors when a document is
        # soft-deleted.
        if doc_id is not None:
            id_seed = f"doc:{doc_id}:chunk:{gidx}"
        else:
            id_seed = f"chat:{chat_id}:chunk:{gidx}"
        point_id = str(uuid.uuid5(_POINT_NS, id_seed))

        points.append({
            "id": point_id,
            "vector": vec,
            "payload": payload,
        })
    await vector_store.upsert(collection, points)
    return len(points)
