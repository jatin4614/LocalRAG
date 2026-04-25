"""Unit tests for ``build_contextualize_prompt`` (Plan A — Task 3.1).

The contextualizer prompt must:

1. Carry document filename + KB/subtag + temporal + cross-doc hints into the
   prefix the LLM is asked to write — that's what moves Anthropic's
   Contextual Retrieval numbers from generic to the advertised 49% retrieval-
   failure reduction on inter-related corpora.
2. Degrade gracefully when optional metadata is missing — no ``"None"`` /
   ``"[]"`` Python-repr leakage into the prompt body.
3. Explicitly bound the model to a 50-100 token output so the prefix stays
   short enough to be a useful retrieval signal (not a second copy of the
   chunk) and so the embedded chunk doesn't blow past the embedder's window.
4. Be cache-friendly: document-level messages must be byte-identical for all
   chunks of the same document so vllm-chat's automatic prefix caching can
   reuse the KV cache across chunks of the same doc — only the FINAL message
   varies per chunk.
"""
from __future__ import annotations

from ext.services.contextualizer import build_contextualize_prompt


def test_prompt_includes_document_filename():
    msgs = build_contextualize_prompt(
        document_text="<full doc text>",
        chunk_text="specific chunk",
        document_metadata={
            "filename": "2024-03-ofc-roadmap.md",
            "kb_name": "Engineering",
            "subtag_name": "Roadmap",
            "document_date": "2024-03-14",
            "related_doc_titles": ["2024-Q1-planning.md", "2024-02-features.md"],
        },
    )
    content = "\n".join(m["content"] for m in msgs)
    assert "2024-03-ofc-roadmap.md" in content
    assert "2024-03-14" in content
    assert "Roadmap" in content
    assert "2024-Q1-planning.md" in content


def test_prompt_handles_missing_optional_fields():
    msgs = build_contextualize_prompt(
        document_text="<doc>",
        chunk_text="chunk",
        document_metadata={
            "filename": "unknown.txt",
            "kb_name": "General",
            "subtag_name": None,
            "document_date": None,
            "related_doc_titles": [],
        },
    )
    content = "\n".join(m["content"] for m in msgs)
    assert "unknown.txt" in content
    assert "None" not in content
    assert "[]" not in content


def test_prompt_constrains_output_length():
    msgs = build_contextualize_prompt(
        document_text="<doc>", chunk_text="chunk",
        document_metadata={"filename": "x.md", "kb_name": "k", "subtag_name": "s",
                            "document_date": None, "related_doc_titles": []},
    )
    content = "\n".join(m["content"] for m in msgs)
    assert any(n in content for n in ["50-100 tokens", "50 to 100 tokens",
                                       "under 100 tokens", "≤ 100 tokens",
                                       "100 tokens"])


def test_prompt_is_cache_friendly():
    msgs = build_contextualize_prompt(
        document_text="<full doc text>",
        chunk_text="chunk A",
        document_metadata={"filename": "x.md", "kb_name": "k", "subtag_name": "s",
                            "document_date": None, "related_doc_titles": []},
    )
    msgs2 = build_contextualize_prompt(
        document_text="<full doc text>",
        chunk_text="chunk B",
        document_metadata={"filename": "x.md", "kb_name": "k", "subtag_name": "s",
                            "document_date": None, "related_doc_titles": []},
    )
    assert msgs[:-1] == msgs2[:-1]
    assert msgs[-1] != msgs2[-1]
