"""Plan A — Task 3.2: surface ``context_prefix`` separately in Qdrant payload.

The contextualizer currently mutates ``chunk["text"]`` in place to prepend the
LLM-generated prefix. That works for embedding but loses the prefix as a
distinct datum: we can't regenerate prefixes later (e.g. after a prompt
tweak) without re-embedding the originals, and we can't reason about the
prefix as its own field at retrieval / debugging time.

Task 3.2 adds:

1. ``contextualize_chunks_with_prefix`` — async batch helper that mutates
   each chunk dict to include both ``context_prefix=<str>`` and the
   concatenated ``text=f"{prefix}\\n\\n{original}"``. Fail-open per chunk:
   on LLM error, leaves the chunk's ``text`` untouched and sets
   ``context_prefix=None``.
2. ``build_point_payload`` — pure helper that returns the canonical
   Qdrant payload dict including the new ``context_prefix`` field. The
   ingest upsert loop calls this so payload shape lives in one place.

These tests pin the contract before either helper exists.
"""
from unittest.mock import AsyncMock, patch
import asyncio


def test_contextualize_batch_returns_prefix_field():
    from ext.services.contextualizer import contextualize_chunks_with_prefix

    chunks = [
        {"text": "original content", "context_prefix": None},
    ]

    async def fake_llm(messages, *args, **kwargs):
        return "This chunk is from 2024-03-14 OFC roadmap about Feature A rollout."

    with patch("ext.services.contextualizer._chat_call", AsyncMock(side_effect=fake_llm)):
        result = asyncio.run(contextualize_chunks_with_prefix(
            chunks, document_text="<doc>",
            document_metadata={"filename": "a.md", "kb_name": "K", "subtag_name": "S",
                               "document_date": "2024-03-14", "related_doc_titles": []},
        ))

    assert len(result) == 1
    out = result[0]
    assert out["context_prefix"] == "This chunk is from 2024-03-14 OFC roadmap about Feature A rollout."
    assert out["text"].startswith("This chunk is from 2024-03-14 OFC roadmap")
    assert "original content" in out["text"]


def test_ingest_upserts_payload_with_context_prefix():
    from ext.services.ingest import build_point_payload

    chunk_meta = {
        "text": "prefix\n\noriginal", "context_prefix": "prefix",
        "page": 1, "heading_path": ["Intro"], "sheet": None,
        "chunk_index": 0,
    }
    payload = build_point_payload(
        kb_id=1, doc_id=42, subtag_id=5, filename="a.md",
        owner_user_id="u1", chunk_meta=chunk_meta,
    )
    assert payload["context_prefix"] == "prefix"
    assert payload["text"] == "prefix\n\noriginal"


def test_ingest_omits_context_prefix_when_none():
    from ext.services.ingest import build_point_payload
    chunk_meta = {
        "text": "plain", "context_prefix": None,
        "page": 1, "heading_path": [], "sheet": None,
        "chunk_index": 0,
    }
    payload = build_point_payload(
        kb_id=1, doc_id=42, subtag_id=5, filename="a.md",
        owner_user_id="u1", chunk_meta=chunk_meta,
    )
    assert payload.get("context_prefix") is None
