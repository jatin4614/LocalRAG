"""Composite pipeline version — stamped on every Qdrant point and every
``kb_documents`` row at ingest time. Used by retrieval / reindex logic to
distinguish chunks produced by different pipeline generations.

Bump any component constant to invalidate previously-ingested chunks:
any reindex path can then filter ``pipeline_version != current_version()``
and drop or re-embed the stale rows.

Single source of truth for what defines "a chunk's provenance".
"""
from __future__ import annotations

# Bumped in P0.3 (O(N) sentence walker).
CHUNKER_VERSION = "v2"

# Bumped in P0.4 (structural extraction — page / heading_path / sheet).
EXTRACTOR_VERSION = "v2"

# Logical embedder name. The actual model is served by TEI; this is the tag
# we stamp so retrieval can reason about "which embeddings am I mixing".
EMBEDDER_MODEL = "bge-m3"

# Flips to "anthropic-ctx" (or similar) in P3.1 when context augmentation
# ships. For now, every chunk is stored verbatim.
CONTEXT_AUGMENTATION = "none"


def current_version() -> str:
    """Return the composite pipeline version string.

    Reads the module-level constants at call time so monkeypatching works
    in tests (``monkeypatch.setattr(pipeline_version, "CHUNKER_VERSION", ...)``).
    """
    return (
        f"chunker={CHUNKER_VERSION}|"
        f"extractor={EXTRACTOR_VERSION}|"
        f"embedder={EMBEDDER_MODEL}|"
        f"ctx={CONTEXT_AUGMENTATION}"
    )


__all__ = [
    "CHUNKER_VERSION",
    "EXTRACTOR_VERSION",
    "EMBEDDER_MODEL",
    "CONTEXT_AUGMENTATION",
    "current_version",
]
