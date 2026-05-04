from ext.db.qdrant_schema import (
    canonical_payload_schema,
    coerce_to_canonical,
    CANONICAL_INDEXES,
)


def test_canonical_schema_has_required_fields():
    schema = canonical_payload_schema()
    for field in ["kb_id", "doc_id", "subtag_id", "filename",
                   "owner_user_id", "text", "chunk_index", "level"]:
        assert field in schema, f"canonical schema missing field: {field}"


def test_coerce_integer_doc_id_passes_through():
    raw = {"kb_id": 1, "doc_id": 42, "text": "hello", "chunk_index": 0,
           "owner_user_id": "u1", "subtag_id": 5, "filename": "a.md"}
    out = coerce_to_canonical(raw)
    assert out["doc_id"] == 42
    assert isinstance(out["doc_id"], int)


def test_coerce_string_doc_id_is_converted_to_int():
    raw = {"kb_id": 1, "doc_id": "42", "text": "hello", "chunk_index": 0,
           "owner_user_id": "u1", "subtag_id": 5, "filename": "a.md"}
    out = coerce_to_canonical(raw)
    assert out["doc_id"] == 42


def test_coerce_missing_optional_fields_gets_default():
    raw = {"kb_id": 1, "doc_id": 42, "text": "hello", "chunk_index": 0,
           "owner_user_id": "u1", "filename": "a.md"}
    out = coerce_to_canonical(raw)
    assert out.get("subtag_id") is None
    assert out.get("level") == "chunk"


def test_canonical_indexes_list_types():
    for idx in CANONICAL_INDEXES:
        assert "field" in idx and "type" in idx
        assert idx["type"] in {"keyword", "integer", "bool", "float", "text"}


def test_canonical_indexes_includes_entities() -> None:
    """Phase 2 of multi-entity-elaborate-answers spec adds 'entities' index.

    Entity list on level=doc points; text index with lowercase tokenization
    handles "5 PoK" / "5 POK" / "5 PoK Bde" variants the same way the
    per-KB synonym table feeds entity_text_filter on chunk-level points.
    """
    field_names = [idx["field"] for idx in CANONICAL_INDEXES]
    assert "entities" in field_names


def test_entities_index_is_text_lowercased() -> None:
    """Entities index must be text type with lowercase tokenization."""
    entities_idx = next(
        idx for idx in CANONICAL_INDEXES if idx["field"] == "entities"
    )
    assert entities_idx["type"] == "text"
    assert entities_idx.get("lowercase") is True
