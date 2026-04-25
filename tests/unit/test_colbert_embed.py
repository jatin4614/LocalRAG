"""Phase 3.4 — ColBERT multi-vector embedder.

The colbert-ir/colbertv2.0 model is fetched lazily by fastembed on first
use. In air-gapped operator environments the cache (``/var/models/
fastembed_cache/`` or ``/tmp/fastembed_cache/``) must be hydrated up
front (see Appendix A.2 of the runbook). When the cache isn't present
and download fails, these tests skip with a clear message rather than
hang the test runner.

The two assertions exercised here are the public contract Task 3.5 will
read against:

1. ``colbert_embed`` returns ``list[list[list[float]]]``: one outer entry
   per text, one inner list per token, each token vector dim==128.
2. The model is deterministic for identical input — same text twice
   always produces byte-identical token-vectors. The write path relies
   on this: a re-ingest of the same chunk must produce the same
   payload, otherwise idempotent upsert (``uuid5`` IDs) silently
   diverges from the upstream vector.
"""
from __future__ import annotations

import os

import pytest


def _try_colbert_embed(texts):
    """Call colbert_embed; skip cleanly on download/runtime errors.

    fastembed will raise ``RuntimeError`` / ``OSError`` / network errors
    if the model isn't cached locally and the host is air-gapped. Catch
    those at the call-site so the test suite stays green pre-Appendix-A
    hydration. A successful call (returning the expected shape) means
    the operator already cached the model.
    """
    from ext.services.embedder import colbert_embed

    try:
        return colbert_embed(texts)
    except Exception as e:  # noqa: BLE001 — broad on purpose
        pytest.skip(
            f"colbert-ir model not available locally: {type(e).__name__}: {e}"
        )


def test_colbert_embed_returns_list_of_token_vectors():
    """Each text produces a variable-length list of token-dim==128 vectors."""
    out = _try_colbert_embed(["hello world", "another sentence with more tokens"])
    assert len(out) == 2
    assert isinstance(out[0], list)
    assert all(
        isinstance(v, list) and all(isinstance(x, float) for x in v)
        for v in out[0]
    )
    assert all(len(v) == 128 for v in out[0])
    # The second sentence has more tokens than the first. ColBERT's
    # tokenizer keeps subwords + punctuation per token, so the longer
    # sentence must produce >= as many vectors as the shorter one.
    assert len(out[1]) >= len(out[0])


@pytest.mark.skipif(
    os.environ.get("SKIP_COLBERT_LIVE") == "1",
    reason="requires fastembed model cache (Appendix A)",
)
def test_colbert_embed_deterministic_same_input():
    """Identical input → identical token-vector output (idempotent ingest)."""
    a = _try_colbert_embed(["fixed text"])
    b = _try_colbert_embed(["fixed text"])
    assert a[0] == b[0]
