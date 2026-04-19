"""Structure- and tokenizer-aware chunker.

Offsets are tracked during a single forward regex walk, so chunking is
O(N) in the input length. Previously the code computed char offsets via
``enc.decode(ids[:k])`` inside the loop, which was O(N^2) and could wedge
the service on 20 MB uploads.

Downstream modules (``ingest.py``, ``budget.py``) depend on:
- ``chunk_text`` signature: (text, *, chunk_tokens, overlap_tokens) -> List[Chunk]
- ``Chunk.text`` and ``Chunk.index`` fields
- ``_encoder`` module-level helper (imported by ``budget.py``)

P1.5 will swap the tokenizer to ``BAAI/bge-m3``; for now we stay on
``tiktoken.cl100k_base`` so the semantics match the existing index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import tiktoken


# Paragraph break: two (or more) newlines, possibly with whitespace between.
_PARA = re.compile(r"\n\s*\n")
# Sentence break: whitespace following a terminator .!?  (English-centric; good enough for P0.)
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int = 0
    end: int = 0
    token_count: int = 0


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    """Shared tokenizer. Also imported by ``budget.py`` — do not rename."""
    return tiktoken.get_encoding("cl100k_base")


def _iter_regions(regex: "re.Pattern[str]", text: str):
    """Yield (abs_start, abs_end, slice) for regions between regex matches.

    The whole input is covered exactly once. Runs in O(N) via ``finditer``.
    """
    pos = 0
    for m in regex.finditer(text):
        yield pos, m.start(), text[pos:m.start()]
        pos = m.end()
    # Always yield the final region (may be empty on trailing separator).
    yield pos, len(text), text[pos:]


def _walk_sentences(text: str):
    """Yield (sentence_text, abs_start, abs_end) in order. O(N) total.

    Splits on paragraph breaks first, then sentence terminators. Preserves
    absolute character offsets in the original ``text`` so chunk offsets
    can be computed without any decode-and-count tricks.
    """
    for para_start, _para_end, para_text in _iter_regions(_PARA, text):
        for rel_start, _rel_end, sent_text in _iter_regions(_SENT, para_text):
            # Strip leading whitespace but keep the offset accurate.
            leading = len(sent_text) - len(sent_text.lstrip())
            stripped = sent_text.strip()
            if not stripped:
                continue
            a = para_start + rel_start + leading
            yield stripped, a, a + len(stripped)


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> List[Chunk]:
    """Split ``text`` into chunks of roughly ``chunk_tokens`` tokens with
    ``overlap_tokens`` of tail-overlap between adjacent chunks.

    Sentence-aware: packs whole sentences until the next one would overflow
    the budget, then emits a chunk. A sentence that is itself larger than
    ``chunk_tokens`` is hard-split on token boundaries (rare — only happens
    for pathological input such as a single huge "word").

    Complexity: O(N) in input length — each token range is decoded at most
    O(1) times (never cumulatively).
    """
    if not text:
        return []
    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must exceed overlap_tokens")

    enc = _encoder()

    # Single forward pass: collect (sentence_text, char_start, char_end, token_count).
    # Each sentence is encoded exactly once.
    meta: list[tuple[str, int, int, int]] = []
    for sent, a, b in _walk_sentences(text):
        meta.append((sent, a, b, len(enc.encode(sent))))
    if not meta:
        return []

    chunks: List[Chunk] = []
    cidx = 0
    i = 0
    n = len(meta)
    while i < n:
        # Greedy pack sentences [i, j) while we fit under the budget.
        j = i
        total = 0
        while j < n and total + meta[j][3] <= chunk_tokens:
            total += meta[j][3]
            j += 1

        if j == i:
            # One sentence is itself larger than chunk_tokens — hard token split.
            # This path also handles sentence-free input (e.g. one giant "word"),
            # which the sentence walker emits as a single region.
            sent, a, _b, _tc = meta[i]
            ids = enc.encode(sent)
            stride = chunk_tokens - overlap_tokens
            for k in range(0, len(ids), stride):
                sub_ids = ids[k : k + chunk_tokens]
                sub_text = enc.decode(sub_ids)
                # Best-effort char offsets: we know the sentence starts at ``a``
                # but without a char-by-char token alignment (expensive on long
                # tokens) we can't pinpoint sub-offsets exactly. Anchor at ``a``
                # for start and step by decoded-length for end.
                sub_start = a + k  # approximation: token index as byte offset
                sub_end = sub_start + len(sub_text)
                chunks.append(
                    Chunk(
                        index=cidx,
                        text=sub_text,
                        start=sub_start,
                        end=sub_end,
                        token_count=len(sub_ids),
                    )
                )
                cidx += 1
            i += 1
            continue

        # Emit a chunk covering sentences [i, j).
        start = meta[i][1]
        end = meta[j - 1][2]
        chunks.append(
            Chunk(
                index=cidx,
                text=text[start:end],
                start=start,
                end=end,
                token_count=total,
            )
        )
        cidx += 1

        if j >= n:
            break

        # Overlap: walk back from j while the trailing sentences fit in the
        # overlap budget. Advance i to that cutoff so the next chunk re-covers
        # those tail sentences. Guarantee forward progress: ``ni > i`` always.
        ni = j
        back = 0
        while ni > i + 1 and back + meta[ni - 1][3] <= overlap_tokens:
            ni -= 1
            back += meta[ni][3]
        i = ni if ni > i else j
    return chunks
