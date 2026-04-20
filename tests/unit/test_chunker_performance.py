"""Performance guard against the O(N^2) regression.

The original implementation computed char offsets via ``enc.decode(ids[:k])``
inside the chunk loop, which is O(N^2) and could wedge the service on
20 MB uploads. The rewrite (P0.3) is O(N); this test keeps it honest.
"""
from __future__ import annotations

import time

from ext.services.chunker import chunk_text


# ~4.5 MB of text — matches the plan's 500 k-token ballpark and produces
# enough iterations to make a quadratic loop blow the budget many times over.
_BIG_TEXT = "The quick brown fox jumps over the lazy dog. " * 100_000


def test_5mb_chunking_completes_in_under_10_seconds() -> None:
    """A ~5 MB input must chunk in well under 10 s.

    Under the pre-fix O(N^2) implementation this took ~19 s on the dev box
    (and worse on prod hardware). The O(N) rewrite should finish in
    roughly a second. We assert < 10 s to keep the gate stable across CI
    hardware variation.
    """
    t0 = time.perf_counter()
    chunks = chunk_text(_BIG_TEXT, chunk_tokens=800, overlap_tokens=100)
    elapsed = time.perf_counter() - t0

    assert len(chunks) > 0, "chunker returned no chunks on a large non-empty input"
    assert elapsed < 10.0, (
        f"chunk_text took {elapsed:.2f}s on ~{len(_BIG_TEXT) / 1_000_000:.1f}MB "
        f"— O(N^2) regression suspected (budget is 10s)"
    )
