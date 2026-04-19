"""Sparse embeddings via fastembed's Qdrant/bm25 model.

Used for the BM25 arm of hybrid retrieval. Lazy-loads the model; thread-safe
after first init. CPU-only (no GPU needed).

The Qdrant/bm25 model itself is just a stopword list + stemmer (~10 MB).
``model.embed(texts)`` returns TF-weighted sparse vectors (for indexing).
``model.query_embed(texts)`` returns all-ones vectors paired with the matching
token indices — the IDF weighting is applied server-side by Qdrant when the
sparse vector is configured with ``Modifier.IDF``.
"""
from __future__ import annotations

import os
import threading
from typing import Sequence

_LOCK = threading.Lock()
_MODEL = None


class SparseEmbeddingNotAvailable(RuntimeError):
    """Raised when fastembed isn't installed but sparse embeddings were requested."""


def _get_model():
    """Return a lazily-initialized singleton fastembed SparseTextEmbedding.

    Uses double-checked locking so concurrent callers don't double-init. The
    first caller pays the model-download + ONNX-session startup cost.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as e:  # pragma: no cover - covered by tests via importorskip
            raise SparseEmbeddingNotAvailable(
                "fastembed not installed — run `pip install 'fastembed>=0.4'` "
                "or `pip install '.[hybrid]'`"
            ) from e
        model_name = os.environ.get("RAG_SPARSE_MODEL", "Qdrant/bm25")
        _MODEL = SparseTextEmbedding(model_name=model_name)
        return _MODEL


def embed_sparse(texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
    """Embed a batch of texts as sparse (indices, values) pairs for indexing.

    Returns one (indices, values) pair per input, both as plain Python lists
    (safe to serialise into Qdrant ``SparseVector`` models).
    """
    model = _get_model()
    out: list[tuple[list[int], list[float]]] = []
    for emb in model.embed(list(texts)):
        out.append(([int(i) for i in emb.indices], [float(v) for v in emb.values]))
    return out


def embed_sparse_query(text: str) -> tuple[list[int], list[float]]:
    """Sparse-embed a single query.

    Uses fastembed's ``query_embed`` which produces token indices with
    ``values == 1.0`` — Qdrant applies IDF weighting server-side for sparse
    vectors that were configured with ``Modifier.IDF`` at collection creation.
    """
    model = _get_model()
    for emb in model.query_embed([text]):
        return [int(i) for i in emb.indices], [float(v) for v in emb.values]
    return [], []
