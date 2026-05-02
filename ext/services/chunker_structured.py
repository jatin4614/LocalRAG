"""Structure-aware chunker for prose + tables + fenced code blocks.

Plan B Phase 6.5. Replaces the window-only chunker for KBs that opt
in via ``rag_config.chunking_strategy="structured"`` (Phase 6.6).

Algorithm:
  1. Split the input into segments. A segment is one of:
     - a fenced code block (``\\`\\`\\`...\\`\\`\\```)
     - a markdown pipe table (header line + dashes + ≥1 row)
     - an HTML <table>...</table>
     - free prose
  2. For code/table segments, emit one chunk per segment with the
     appropriate ``chunk_type`` payload. Oversized segments are split
     by row-group (tables) or line-group (code) with a continuation flag.
  3. For prose segments, fall back to the existing window chunker.

The output is a list of dicts ready for embedding + payload upsert:
  {"text": ..., "chunk_type": "prose|table|code|image_caption",
   "language": "<for code>", "continuation": False}

Token counts are approximate (chars / 4); for KBs that need exact
counts, the caller can re-tokenize after.
"""
from __future__ import annotations

import re


_FENCED_CODE_RE = re.compile(
    r"^```(\w+)?\s*\n(.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)

# Markdown pipe table: header + |---|---| separator + at least one row.
# We use ``[^\n]+`` so each row consumes greedily up to the newline (not
# ``.+?`` which would non-greedy stop at the first whitespace inside the
# row, breaking multi-row tables — see Plan B Phase 6.5 test cases).
_PIPE_TABLE_RE = re.compile(
    r"(\|[^\n]+\|[ \t]*\n\|[-: |]+\|[ \t]*\n(?:\|[^\n]+\|[ \t]*\n?)+)",
    re.MULTILINE,
)

_HTML_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)


def _tok_estimate(s: str) -> int:
    """Real tokenizer count (not the previous ``len(s)//4`` estimate).

    Review §2.2 — dense JSON / table / code under-counts by ~30% under
    ``len/4``, which makes downstream chunks larger than the budget the
    caller supplied (``chunk_size_tokens``). When the chunk is then sent
    through the prompt-budget pass the over-large chunk is evicted.

    Routes through ``ext.services.chunker._encoder()`` — the same
    tokenizer handle the window chunker uses, so structured + window
    chunks share one token vocabulary. Cached for the life of the
    process via ``lru_cache`` inside ``_encoder``.

    Empty / whitespace-only input still returns 1 (preserves the old
    floor — ``max(1, ...)``) so split loops that compare ``cur_tok +
    rtok`` against ``max_tokens`` never get a zero on a blank row.
    """
    from .chunker import _encoder  # local: structured chunker is opt-in path
    return max(1, len(_encoder().encode(s)))


def _split_giant_table(text: str, max_tokens: int) -> list[str]:
    """Split a markdown pipe table by row-groups, repeating the header."""
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return [text]
    header, sep, *rows = lines
    out = []
    cur_rows = []
    cur_tok = _tok_estimate("\n".join([header, sep]))
    for row in rows:
        rtok = _tok_estimate(row)
        if cur_tok + rtok > max_tokens and cur_rows:
            out.append("\n".join([header, sep, *cur_rows]))
            cur_rows = [row]
            cur_tok = _tok_estimate("\n".join([header, sep, row]))
        else:
            cur_rows.append(row)
            cur_tok += rtok
    if cur_rows:
        out.append("\n".join([header, sep, *cur_rows]))
    return out


def _split_giant_code(text: str, language: str | None,
                       max_tokens: int) -> list[tuple[str, bool]]:
    """Split a fenced code block by line-groups. Returns [(text, continuation)]."""
    inner = text
    # Strip the outer fence to get the inner code
    fence_match = re.match(r"^```(\w+)?\s*\n(.*?)^```\s*$",
                            text, re.DOTALL | re.MULTILINE)
    if fence_match:
        inner = fence_match.group(2)
    lang = language or (fence_match.group(1) if fence_match else "")
    lines = inner.split("\n")
    out = []
    cur = []
    cur_tok = 0
    for line in lines:
        ltok = _tok_estimate(line)
        if cur_tok + ltok > max_tokens and cur:
            out.append("\n".join(cur))
            cur = [line]
            cur_tok = ltok
        else:
            cur.append(line)
            cur_tok += ltok
    if cur:
        out.append("\n".join(cur))

    fence_lang = lang or ""
    return [
        (f"```{fence_lang}\n{seg}\n```", i > 0)
        for i, seg in enumerate(out)
    ]


def _window_chunk_prose(text: str, chunk_size_tokens: int,
                         overlap_tokens: int) -> list[str]:
    """Reuse the existing prose chunker logic.

    For Plan B's purposes we approximate via word windows. Production
    callers re-tokenize for exact budgets downstream.
    """
    if not text.strip():
        return []
    # Approximate: 1 token ≈ 0.75 words. Use words for reproducible splits.
    words = text.split()
    target_words = max(50, int(chunk_size_tokens * 0.75))
    overlap_words = max(0, int(overlap_tokens * 0.75))
    out = []
    i = 0
    while i < len(words):
        seg = words[i:i + target_words]
        out.append(" ".join(seg))
        if i + target_words >= len(words):
            break
        i += max(1, target_words - overlap_words)
    return out


def _segments_with_offsets(text: str) -> list[tuple[int, int, str, dict]]:
    """Return [(start, end, type, meta)] sorted by start."""
    segs: list[tuple[int, int, str, dict]] = []
    for m in _FENCED_CODE_RE.finditer(text):
        segs.append((m.start(), m.end(), "code", {"language": m.group(1)}))
    for m in _PIPE_TABLE_RE.finditer(text):
        # Skip if inside a code segment
        if any(s <= m.start() < e for s, e, t, _ in segs if t == "code"):
            continue
        segs.append((m.start(), m.end(), "table", {"format": "markdown"}))
    for m in _HTML_TABLE_RE.finditer(text):
        if any(s <= m.start() < e for s, e, _, _ in segs):
            continue
        segs.append((m.start(), m.end(), "table", {"format": "html"}))
    segs.sort()
    return segs


def chunk_structured(
    text: str, *, chunk_size_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[dict]:
    """Chunk text preserving table + code structure.

    Returns a list of chunk dicts:
        {"text": ..., "chunk_type": "prose|table|code",
         "language": "..." (code only),
         "continuation": False (set True on overflow segments)}
    """
    segments = _segments_with_offsets(text)
    out: list[dict] = []
    cursor = 0
    for start, end, typ, meta in segments:
        # Prose before this structured segment
        if start > cursor:
            prose = text[cursor:start]
            for p in _window_chunk_prose(
                prose, chunk_size_tokens, overlap_tokens,
            ):
                if p.strip():
                    out.append({"text": p, "chunk_type": "prose"})
        # Emit the structured segment
        seg_text = text[start:end]
        if typ == "code":
            if _tok_estimate(seg_text) > chunk_size_tokens:
                for stext, cont in _split_giant_code(
                    seg_text, meta.get("language"), chunk_size_tokens,
                ):
                    out.append({
                        "text": stext, "chunk_type": "code",
                        "language": meta.get("language") or "unknown",
                        "continuation": cont,
                    })
            else:
                out.append({
                    "text": seg_text, "chunk_type": "code",
                    "language": meta.get("language") or "unknown",
                    "continuation": False,
                })
        elif typ == "table":
            if _tok_estimate(seg_text) > chunk_size_tokens \
                    and meta.get("format") == "markdown":
                for stext in _split_giant_table(seg_text, chunk_size_tokens):
                    out.append({
                        "text": stext, "chunk_type": "table",
                        "format": "markdown", "continuation": False,
                    })
            else:
                out.append({
                    "text": seg_text, "chunk_type": "table",
                    "format": meta.get("format"), "continuation": False,
                })
        cursor = end
    # Trailing prose
    if cursor < len(text):
        for p in _window_chunk_prose(
            text[cursor:], chunk_size_tokens, overlap_tokens,
        ):
            if p.strip():
                out.append({"text": p, "chunk_type": "prose"})
    return out


__all__ = ["chunk_structured"]
