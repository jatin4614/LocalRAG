"""Block-coalescing in ingest (sibling-of structured chunker, upstream of it).

The DOCX extractor emits one ExtractedBlock per Word paragraph. On Apr/Mar/
Feb/Jan 26.docx the median paragraph runs ~30 tokens, which means each
``chunk_text_for_kb`` call sees a tiny string and returns a single tiny
chunk — the 800-token window chunker never gets to pack across paragraphs.
Empirically this inflated KB 2's chunk count by ~15x.

``_coalesce_small_blocks`` walks the extracted blocks BEFORE the per-block
chunking loop and merges adjacent small prose blocks (sharing ``heading_path``)
up to a soft cap. Tables / code / image_caption blocks stay atomic so the
structured-chunker contract is preserved.

Defaults:
  * ``RAG_INGEST_BLOCK_MIN_TOKENS=200`` — block under this is "small"
  * ``RAG_INGEST_BLOCK_MAX_TOKENS=600`` — coalesced block stops at this cap

Both env-overridable so an operator can tune for an unusual corpus.
"""
from __future__ import annotations

from ext.services.extractor import ExtractedBlock
from ext.services.ingest import _coalesce_small_blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prose_block(n_tokens: int, *, heading_path=None, page=None) -> ExtractedBlock:
    """Build a prose ExtractedBlock with approximately ``n_tokens`` tokens.

    Uses the same tokenizer the production pipeline uses (via
    ``ext.services.budget.get_tokenizer``) so the size assertions in
    coalesce tests track real token counts, not character estimates.
    """
    from ext.services.budget import get_tokenizer
    enc = get_tokenizer()
    # "word " is 1 token under cl100k and most HF tokenizers; pad until we
    # cross the requested count then trim back to exact size.
    text = ("word " * (n_tokens + 5)).rstrip()
    ids = enc.encode(text)
    if len(ids) > n_tokens:
        text = enc.decode(ids[:n_tokens])
    return ExtractedBlock(
        text=text,
        heading_path=list(heading_path or []),
        page=page,
    )


def _toklen(text: str) -> int:
    from ext.services.budget import get_tokenizer
    return len(get_tokenizer().encode(text))


# ---------------------------------------------------------------------------
# Test 1: single tiny prose block stays unchanged
# ---------------------------------------------------------------------------

def test_single_tiny_block_passes_through_unchanged():
    blocks = [ExtractedBlock(text="hi")]
    out = _coalesce_small_blocks(blocks)
    assert len(out) == 1
    assert out[0].text == "hi"


# ---------------------------------------------------------------------------
# Test 2: two adjacent tiny prose blocks merge
# ---------------------------------------------------------------------------

def test_two_tiny_adjacent_prose_blocks_merge():
    a = _prose_block(50)
    b = _prose_block(50)
    out = _coalesce_small_blocks([a, b])
    assert len(out) == 1, f"expected 1 merged block, got {len(out)}"
    # Merged block contains both source texts, joined.
    assert a.text in out[0].text
    assert b.text in out[0].text
    # Combined token count is roughly the sum (give or take join chars).
    assert 90 <= _toklen(out[0].text) <= 120


# ---------------------------------------------------------------------------
# Test 3: coalesce stops at the soft cap
# ---------------------------------------------------------------------------

def test_coalesce_stops_at_soft_cap_by_default():
    # 10 blocks of ~100 tokens each, all same heading. With cap=600 default,
    # we expect roughly 2 output blocks (~600 + ~400). Never one giant 1000.
    blocks = [_prose_block(100) for _ in range(10)]
    out = _coalesce_small_blocks(blocks)
    assert len(out) >= 2, f"expected ≥2 blocks (cap should split), got {len(out)}"
    # No single output block should exceed ~620 tokens (cap=600 + small slop
    # for the last block we admit before re-checking the cap).
    for ob in out:
        assert _toklen(ob.text) <= 700, (
            f"output block exceeded soft cap: {_toklen(ob.text)} tokens"
        )


# ---------------------------------------------------------------------------
# Test 4: heading boundary stops coalesce
# ---------------------------------------------------------------------------

def test_heading_boundary_stops_coalesce():
    # blocks: A_intro (heading [X]), A_other (heading [Y]), A_intro2 (heading [X])
    # Block 2 has a different heading, so coalesce must NOT merge across it.
    a = _prose_block(50, heading_path=["X"])
    b = _prose_block(50, heading_path=["Y"])
    c = _prose_block(50, heading_path=["X"])
    out = _coalesce_small_blocks([a, b, c])
    # 3 distinct runs, each one block long → 3 outputs.
    assert len(out) == 3
    assert out[0].heading_path == ["X"]
    assert out[1].heading_path == ["Y"]
    assert out[2].heading_path == ["X"]


# ---------------------------------------------------------------------------
# Test 5: non-prose block stops coalesce + stays atomic
# ---------------------------------------------------------------------------

def test_non_prose_block_stays_atomic_and_breaks_coalesce_run():
    # Use the optional ``kind`` field to mark a table block. Adjacent prose
    # blocks must NOT swallow it, and the table itself must pass through
    # byte-identical (no merging of prose into the table either).
    p1 = _prose_block(50)
    table = ExtractedBlock(text="col1\tcol2\nval1\tval2", kind="table")
    p2 = _prose_block(50)
    out = _coalesce_small_blocks([p1, table, p2])
    assert len(out) == 3
    # The table block is byte-identical to the input.
    assert out[1].text == table.text
    assert out[1].kind == "table"


# ---------------------------------------------------------------------------
# Test 6: already-large prose block is passed through unchanged
# ---------------------------------------------------------------------------

def test_already_large_prose_block_unchanged():
    # 800 tokens is well above the 200-token "small" threshold, so this
    # block isn't a coalesce candidate. It should pass through verbatim.
    big = _prose_block(800)
    out = _coalesce_small_blocks([big])
    assert len(out) == 1
    assert out[0].text == big.text


# ---------------------------------------------------------------------------
# Test 7: mixed-size run
# ---------------------------------------------------------------------------

def test_mixed_size_run_merges_only_small_runs():
    # Pattern: tiny, tiny, tiny, LARGE, tiny, tiny.
    # Expected: [merged-3-tinies], [LARGE], [merged-2-tinies] → 3 blocks.
    t1 = _prose_block(50)
    t2 = _prose_block(50)
    t3 = _prose_block(50)
    big = _prose_block(800)
    t4 = _prose_block(50)
    t5 = _prose_block(50)
    out = _coalesce_small_blocks([t1, t2, t3, big, t4, t5])
    assert len(out) == 3
    # First merged block contains t1 + t2 + t3.
    assert all(b.text in out[0].text for b in (t1, t2, t3))
    # Middle block is the large one, untouched.
    assert out[1].text == big.text
    # Trailing merged block contains t4 + t5.
    assert all(b.text in out[2].text for b in (t4, t5))


# ---------------------------------------------------------------------------
# Test 8: env-var overrides
# ---------------------------------------------------------------------------

def test_env_var_overrides_thresholds(monkeypatch):
    # Lower min/max so smaller blocks count as "small" and the cap fires sooner.
    monkeypatch.setenv("RAG_INGEST_BLOCK_MIN_TOKENS", "50")
    monkeypatch.setenv("RAG_INGEST_BLOCK_MAX_TOKENS", "200")
    # 4 blocks of ~75 tokens each. Default cap (600) would merge all 4 into
    # one ~300-token block. With cap=200 we should get ≥2 outputs (each ≤220).
    blocks = [_prose_block(75) for _ in range(4)]
    out = _coalesce_small_blocks(blocks)
    assert len(out) >= 2, f"cap=200 should split; got {len(out)} blocks"
    for ob in out:
        assert _toklen(ob.text) <= 250, (
            f"block exceeded cap=200: {_toklen(ob.text)} tokens"
        )


# ---------------------------------------------------------------------------
# Test 9: metadata preservation — heading_path + page from the leading block
# ---------------------------------------------------------------------------

def test_merged_block_inherits_leading_metadata():
    # First block carries page=5 + heading_path=["Sec"], second is metadata-poor.
    # Merged output must keep heading_path=["Sec"] and page=5 (first non-None).
    a = _prose_block(50, heading_path=["Sec"], page=5)
    b = _prose_block(50, heading_path=["Sec"], page=None)
    out = _coalesce_small_blocks([a, b])
    assert len(out) == 1
    assert out[0].heading_path == ["Sec"]
    assert out[0].page == 5


def test_merged_block_picks_first_non_none_page():
    # Leading block has page=None; second has page=7. Output uses the first
    # non-None value so the citation surface still has a page hint.
    a = _prose_block(50, heading_path=["Sec"], page=None)
    b = _prose_block(50, heading_path=["Sec"], page=7)
    out = _coalesce_small_blocks([a, b])
    assert len(out) == 1
    assert out[0].page == 7
