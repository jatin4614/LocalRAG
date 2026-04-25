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
        assert idx["type"] in {"keyword", "integer", "bool", "float"}
