"""Plan B Phase 6.6 / 6.7 — chunker dispatcher wired into ingest_bytes.

The ingest path now routes ``chunk_text`` calls through the per-KB
dispatcher ``chunk_text_for_kb``. Structured-chunker output dicts may
carry ``chunk_type``, ``language``, and ``continuation`` keys; image
extraction emits ``chunk_type='image_caption'`` chunks. These tests pin
the contract that ``build_point_payload`` propagates the new fields,
and that the legacy default-window strategy stays byte-identical.
"""

from ext.services.ingest import build_point_payload


def _base_meta(**overrides):
    base = {
        "text": "hello",
        "chunk_index": 0,
        "page": 1,
        "heading_path": ["Intro"],
        "sheet": None,
    }
    base.update(overrides)
    return base


def test_payload_includes_chunk_type_when_provided():
    payload = build_point_payload(
        kb_id=1, doc_id=1, subtag_id=None, filename="x.md",
        owner_user_id="u", chunk_meta=_base_meta(chunk_type="table"),
    )
    assert payload["chunk_type"] == "table"


def test_payload_includes_language_when_provided():
    payload = build_point_payload(
        kb_id=1, doc_id=1, subtag_id=None, filename="x.md",
        owner_user_id="u", chunk_meta=_base_meta(
            chunk_type="code", language="python",
        ),
    )
    assert payload["chunk_type"] == "code"
    assert payload["language"] == "python"


def test_payload_includes_continuation_when_provided():
    payload = build_point_payload(
        kb_id=1, doc_id=1, subtag_id=None, filename="x.md",
        owner_user_id="u", chunk_meta=_base_meta(
            chunk_type="table", continuation=True,
        ),
    )
    assert payload["continuation"] is True


def test_payload_omits_new_fields_when_not_provided():
    """Default 'window' chunker emits chunk_type='prose' but ingest_bytes
    skips stamping defaults so legacy payloads stay byte-identical with
    pre-Plan-B-Phase-6 ones (no surprise field additions on existing KBs)."""
    payload = build_point_payload(
        kb_id=1, doc_id=1, subtag_id=None, filename="x.md",
        owner_user_id="u", chunk_meta=_base_meta(),
    )
    # The four new optional keys must NOT be in the payload at all
    # when caller didn't put them in chunk_meta.
    assert "chunk_type" not in payload
    assert "language" not in payload
    assert "continuation" not in payload


def test_payload_image_caption_meta_propagates():
    """Image-caption chunks emitted by Phase 6.7 carry chunk_type='image_caption'
    and a per-image page number (overrides the host block's page)."""
    payload = build_point_payload(
        kb_id=1, doc_id=1, subtag_id=None, filename="x.pdf",
        owner_user_id="u", chunk_meta=_base_meta(
            text="A bar chart titled 'Q1 revenue' showing growth from Jan to Mar.",
            chunk_type="image_caption",
            page=4,  # explicit page from extracted image, not block default
        ),
    )
    assert payload["chunk_type"] == "image_caption"
    assert payload["page"] == 4
    assert "language" not in payload  # not provided
