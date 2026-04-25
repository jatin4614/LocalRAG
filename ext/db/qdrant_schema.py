"""Canonical Qdrant payload schema and migration helpers.

Today two KB-shaped collections (kb_eval, kb_1_rebuild) have divergent payload
typings: doc_id as keyword vs integer, on_disk_payload on vs off. Phase 3 adds
more fields (context_prefix, colbert vectors) — fixing divergence first
prevents compounding.

This module defines the canonical shape. Migration script in
scripts/reconcile_qdrant_schema.py applies it.
"""
from __future__ import annotations

from typing import Any


# Ordered list — migration applies these as `create_payload_index` calls
CANONICAL_INDEXES = [
    {"field": "kb_id", "type": "integer", "is_tenant": False},
    {"field": "doc_id", "type": "integer", "is_tenant": False},
    {"field": "subtag_id", "type": "integer", "is_tenant": False},
    {"field": "owner_user_id", "type": "keyword", "is_tenant": True},
    {"field": "chat_id", "type": "keyword", "is_tenant": True},
    {"field": "chunk_index", "type": "integer", "is_tenant": False},
    {"field": "level", "type": "keyword", "is_tenant": False},  # 'chunk' | 'doc'
    {"field": "filename", "type": "keyword", "is_tenant": False},
]


def canonical_payload_schema() -> dict[str, type | tuple[type, ...]]:
    return {
        "kb_id": int,
        "doc_id": int,
        "subtag_id": (int, type(None)),
        "chat_id": (str, type(None)),
        "owner_user_id": str,
        "filename": str,
        "text": str,
        "page": (int, type(None)),
        "heading_path": (list, type(None)),
        "sheet": (str, type(None)),
        "chunk_index": int,
        "level": str,
        # Phase 3 additions (may not exist in pre-Phase-3 points — optional)
        "context_prefix": (str, type(None)),
    }


def coerce_to_canonical(raw: dict[str, Any]) -> dict[str, Any]:
    """Best-effort coercion of an existing payload dict into canonical types.

    Used by the migration script to re-upsert divergent points. Unknown fields
    are preserved unchanged (Qdrant accepts arbitrary JSONB).
    """
    out = dict(raw)

    # doc_id → int
    if "doc_id" in out:
        try:
            out["doc_id"] = int(out["doc_id"])
        except (TypeError, ValueError):
            out["doc_id"] = 0  # sentinel for missing

    # kb_id, subtag_id, chunk_index → int
    for k in ("kb_id", "subtag_id", "chunk_index"):
        if k in out and out[k] is not None:
            try:
                out[k] = int(out[k])
            except (TypeError, ValueError):
                out[k] = None

    # owner_user_id, chat_id, filename, text → str
    for k in ("owner_user_id", "chat_id", "filename", "text"):
        if k in out and out[k] is not None:
            out[k] = str(out[k])

    # level default
    if "level" not in out or not out["level"]:
        out["level"] = "chunk"

    return out


CANONICAL_COLLECTION_CONFIG = {
    "on_disk_payload": True,  # pick one: on_disk for RAM savings on large corpora
    "hnsw_config": {
        "m": 16,
        "ef_construct": 200,
    },
}
