import httpx
import pytest
from ext.services.embedder import StubEmbedder, TEIEmbedder


@pytest.mark.asyncio
async def test_stub_embedder_deterministic():
    e = StubEmbedder(dim=8)
    a = await e.embed(["hello"])
    b = await e.embed(["hello"])
    assert a == b
    assert len(a[0]) == 8


@pytest.mark.asyncio
async def test_stub_embedder_different_texts_different_vectors():
    e = StubEmbedder(dim=8)
    [va], [vb] = await e.embed(["hello"]), await e.embed(["world"])
    assert va != vb


@pytest.mark.asyncio
async def test_tei_embedder_calls_embed_endpoint():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        return httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4]])

    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)
    vecs = await e.embed(["hello"])
    assert vecs == [[0.1, 0.2, 0.3, 0.4]]
    await e.aclose()


@pytest.mark.asyncio
async def test_tei_embedder_batches():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[0.1]*4, [0.2]*4, [0.3]*4])
    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)
    vecs = await e.embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert len(vecs[0]) == 4
    await e.aclose()
