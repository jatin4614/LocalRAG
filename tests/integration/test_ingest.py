import pytest
from ext.services.ingest import ingest_bytes
from ext.services.embedder import StubEmbedder
from ext.services.vector_store import VectorStore


@pytest.mark.asyncio
async def test_ingest_txt_into_kb(clean_qdrant):
    vs = VectorStore(url=clean_qdrant, vector_size=32)
    await vs.ensure_collection("kb_7")
    e = StubEmbedder(dim=32)

    n = await ingest_bytes(
        data=b"This is the first sentence. Here is more text to index.",
        mime_type="text/plain",
        filename="a.txt",
        collection="kb_7",
        payload_base={"kb_id": 7, "subtag_id": 11, "doc_id": 100},
        vector_store=vs,
        embedder=e,
        chunk_tokens=20, overlap_tokens=5,
    )
    assert n >= 1

    hits = await vs.search("kb_7", [1.0] + [0.0]*31, limit=10)
    assert len(hits) >= 1
    assert all(h.payload["doc_id"] == 100 for h in hits)
    await vs.close()


@pytest.mark.asyncio
async def test_ingest_pdf_empty_ok(clean_qdrant):
    from pypdf import PdfWriter
    import io
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)

    vs = VectorStore(url=clean_qdrant, vector_size=16)
    await vs.ensure_collection("kb_1")
    n = await ingest_bytes(
        data=buf.getvalue(), mime_type="application/pdf", filename="blank.pdf",
        collection="kb_1",
        payload_base={"kb_id": 1, "subtag_id": 2, "doc_id": 3},
        vector_store=vs, embedder=StubEmbedder(dim=16),
    )
    assert n == 0
    await vs.close()
