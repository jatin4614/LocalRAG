import pytest
from ext.services.vector_store import VectorStore


@pytest.mark.asyncio
async def test_ensure_collection_is_idempotent(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=16)
    await vs.ensure_collection("kb_5")
    await vs.ensure_collection("kb_5")
    cols = await vs.list_collections()
    assert "kb_5" in cols
    await vs.close()


@pytest.mark.asyncio
async def test_upsert_and_search(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=4)
    await vs.ensure_collection("kb_1")

    points = [
        {"id": 1, "vector": [1.0, 0.0, 0.0, 0.0], "payload": {"text": "alpha", "subtag_id": 10}},
        {"id": 2, "vector": [0.0, 1.0, 0.0, 0.0], "payload": {"text": "beta",  "subtag_id": 20}},
    ]
    await vs.upsert("kb_1", points)

    hits = await vs.search("kb_1", [1.0, 0.0, 0.0, 0.0], limit=5)
    assert hits[0].payload["text"] == "alpha"
    assert hits[0].score > 0.9
    await vs.close()


@pytest.mark.asyncio
async def test_search_with_subtag_filter(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=4)
    await vs.ensure_collection("kb_1")
    await vs.upsert("kb_1", [
        {"id": 1, "vector": [1, 0, 0, 0], "payload": {"text": "alpha", "subtag_id": 10}},
        {"id": 2, "vector": [1, 0, 0, 0], "payload": {"text": "beta",  "subtag_id": 20}},
    ])
    hits = await vs.search("kb_1", [1.0, 0, 0, 0], limit=5, subtag_ids=[10])
    assert len(hits) == 1
    assert hits[0].payload["subtag_id"] == 10
    await vs.close()


@pytest.mark.asyncio
async def test_delete_collection(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=4)
    await vs.ensure_collection("chat_42")
    await vs.delete_collection("chat_42")
    assert "chat_42" not in await vs.list_collections()
    await vs.close()
