"""Extract → chunk → embed → upsert pipeline.

Each extracted block carries structural metadata (``page`` / ``heading_path`` /
``sheet``). We chunk the block's text independently and inherit the block's
metadata onto every resulting chunk so Qdrant payloads can surface hints like
"from page 7" or "under heading 'Rollout'" at retrieval time.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Mapping

from .chunker import chunk_text
from .embedder import Embedder
from .extractor import extract
from .pipeline_version import current_version
from .vector_store import VectorStore


def _hybrid_enabled() -> bool:
    """Read RAG_HYBRID at call time so tests can toggle it without reimport.

    Default on as of 2026-04-19 — eval showed +12pp chunk_recall at +3ms.
    Set RAG_HYBRID=0 to force dense-only. Any non-"0" value means "on".
    Runtime fallback: even with hybrid on, ingest only computes sparse vectors
    when the target collection was created with sparse support (via
    ``_collection_has_sparse``) — legacy collections remain dense-only.
    """
    return os.environ.get("RAG_HYBRID", "1") != "0"

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

    # Sparse vectors are only computed when hybrid is on AND the target
    # collection was created with sparse support. When either condition fails
    # we produce no sparse vectors and the upsert path takes the legacy
    # dense-only shape (byte-identical to the pre-hybrid behaviour). We use
    # getattr with defaults so test doubles / minimal VectorStore substitutes
    # that don't implement the sparse detection helpers still work.
    sparse_vectors: list[tuple[list[int], list[float]] | None] = [None] * len(paired)
    if _hybrid_enabled():
        refresh = getattr(vector_store, "_refresh_sparse_cache", None)
        has_sparse = getattr(vector_store, "_collection_has_sparse", None)
        if refresh is not None and has_sparse is not None:
            try:
                await refresh(collection)
            except Exception:
                pass  # fall through — has_sparse below will be False
            if has_sparse(collection):
                try:
                    from .sparse_embedder import embed_sparse
                    sparse_vectors = list(embed_sparse(texts))  # type: ignore[assignment]
                except Exception:
                    # fastembed missing or failed — silently skip sparse arm.
                    sparse_vectors = [None] * len(paired)

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

        point: dict = {
            "id": point_id,
            "vector": vec,
            "payload": payload,
        }
        sv = sparse_vectors[gidx]
        if sv is not None:
            point["sparse_vector"] = sv
        points.append(point)
    await vector_store.upsert(collection, points)
    return len(points)
