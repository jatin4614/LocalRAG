import pytest
from ext.services.retriever import retrieve
from ext.services.ingest import ingest_bytes
from ext.services.vector_store import VectorStore
from ext.services.embedder import StubEmbedder

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_retrieve_from_multiple_kbs(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    emb = StubEmbedder(dim=32)

    for kb_id in (1, 2):
        await vs.ensure_collection(f"kb_{kb_id}")
        await ingest_bytes(
            data=f"content for KB {kb_id}: quick brown fox jumps over".encode(),
            mime_type="text/plain", filename=f"kb{kb_id}.txt",
            collection=f"kb_{kb_id}",
            payload_base={"kb_id": kb_id, "subtag_id": kb_id*10, "doc_id": kb_id},
            vector_store=vs, embedder=emb,
            chunk_tokens=20, overlap_tokens=5,
        )

    hits = await retrieve(
        query="quick brown fox",
        selected_kbs=[{"kb_id": 1, "subtag_ids": []}, {"kb_id": 2, "subtag_ids": []}],
        chat_id=None,
        vector_store=vs, embedder=emb,
        per_kb_limit=5, total_limit=10,
    )
    kb_ids = {h.payload["kb_id"] for h in hits}
    assert 1 in kb_ids and 2 in kb_ids
    await vs.close()


@pytest.mark.asyncio
async def test_retrieve_respects_subtag_filter(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=16)
    emb = StubEmbedder(dim=16)
    await vs.ensure_collection("kb_1")
    await ingest_bytes(
        data=b"a a a a a a a a a a a a a a a a",
        mime_type="text/plain", filename="a.txt",
        collection="kb_1",
        payload_base={"kb_id": 1, "subtag_id": 10, "doc_id": 100},
        vector_store=vs, embedder=emb, chunk_tokens=20, overlap_tokens=5,
    )
    await ingest_bytes(
        data=b"b b b b b b b b b b b b b b b b",
        mime_type="text/plain", filename="b.txt",
        collection="kb_1",
        payload_base={"kb_id": 1, "subtag_id": 20, "doc_id": 200},
        vector_store=vs, embedder=emb, chunk_tokens=20, overlap_tokens=5,
    )
    hits = await retrieve(
        query="a a a", selected_kbs=[{"kb_id": 1, "subtag_ids": [10]}], chat_id=None,
        vector_store=vs, embedder=emb,
    )
    assert all(h.payload["subtag_id"] == 10 for h in hits)
    await vs.close()


@pytest.mark.asyncio
async def test_retrieve_includes_chat_private_docs(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=16)
    emb = StubEmbedder(dim=16)
    await vs.ensure_collection("chat_42")
    await ingest_bytes(
        data=b"private words: chapter one of the secret",
        mime_type="text/plain", filename="priv.txt",
        collection="chat_42",
        payload_base={"chat_id": 42, "owner_user_id": 1},
        vector_store=vs, embedder=emb, chunk_tokens=20, overlap_tokens=5,
    )
    hits = await retrieve(
        query="secret chapter", selected_kbs=[], chat_id=42,
        vector_store=vs, embedder=emb,
    )
    assert any(h.payload.get("chat_id") == 42 for h in hits)
    await vs.close()
