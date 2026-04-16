"""Extract → chunk → embed → upsert pipeline."""
from __future__ import annotations

import time
import uuid
from typing import Mapping

from .chunker import chunk_text
from .embedder import Embedder
from .extractor import extract_text
from .vector_store import VectorStore


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
    for chunk, vec in zip(chunks, vectors):
        payload = dict(payload_base)
        payload["chunk_index"] = chunk.index
        payload["text"] = chunk.text
        payload["uploaded_at"] = now
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": payload,
        })
    await vector_store.upsert(collection, points)
    return len(points)
