"""Extract → chunk → embed → upsert pipeline."""
from __future__ import annotations

import time
import uuid
from typing import Mapping

from .chunker import chunk_text
from .embedder import Embedder
from .extractor import extract_text
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
    text = extract_text(data, mime_type, filename)
    chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    vectors = await embedder.embed(texts)

    now = int(time.time())
    points = []
    doc_id = payload_base.get("doc_id")
    chat_id = payload_base.get("chat_id")
    for chunk, vec in zip(chunks, vectors):
        payload = dict(payload_base)
        # Store doc_id as string for consistent filtering/matching downstream.
        if doc_id is not None:
            payload["doc_id"] = str(doc_id)
        payload["chunk_index"] = chunk.index
        payload["text"] = chunk.text
        payload["uploaded_at"] = now
        payload["deleted"] = False

        # Deterministic point ID: same doc + chunk always maps to the same Qdrant point.
        # This lets delete_by_doc reconstruct point IDs or use payload filtering to
        # remove vectors when a document is soft-deleted.
        if doc_id is not None:
            id_seed = f"doc:{doc_id}:chunk:{chunk.index}"
        else:
            id_seed = f"chat:{chat_id}:chunk:{chunk.index}"
        point_id = str(uuid.uuid5(_POINT_NS, id_seed))

        points.append({
            "id": point_id,
            "vector": vec,
            "payload": payload,
        })
    await vector_store.upsert(collection, points)
    return len(points)
