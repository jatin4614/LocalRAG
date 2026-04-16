"""Relevance smoke — a known-good query finds the right chunk, not noise.

Uses the StubEmbedder (deterministic hash-based): exact text match between query
and chunk → identical vectors → top-1 is always the correct doc. This validates
*pipeline plumbing* (ingest → search → rerank), not model quality.
"""
import pytest
from ext.services.ingest import ingest_bytes
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder
from ext.services.retriever import retrieve
from ext.services.reranker import rerank


@pytest.mark.asyncio
async def test_query_hits_matching_doc_not_noise(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    emb = StubEmbedder(dim=32)
    await vs.ensure_collection("kb_1")

    docs = [
        (b"Python is a programming language used for web development and data science.",
         {"kb_id": 1, "subtag_id": 1, "doc_id": 1}),
        (b"The Eiffel Tower is located in Paris, France and was built in 1889.",
         {"kb_id": 1, "subtag_id": 2, "doc_id": 2}),
        (b"Mount Everest is the tallest mountain on Earth at 8,849 meters.",
         {"kb_id": 1, "subtag_id": 3, "doc_id": 3}),
    ]
    for data, payload in docs:
        await ingest_bytes(
            data=data, mime_type="text/plain", filename="d.txt",
            collection="kb_1", payload_base=payload,
            vector_store=vs, embedder=emb,
            chunk_tokens=30, overlap_tokens=5,
        )

    hits = await retrieve(
        query="Python is a programming language used for web development and data science.",
        selected_kbs=[{"kb_id": 1, "subtag_ids": []}],
        chat_id=None, vector_store=vs, embedder=emb,
    )
    reranked = rerank(hits, top_k=3)
    assert reranked, "no hits returned"
    assert reranked[0].payload["doc_id"] == 1
    await vs.close()
