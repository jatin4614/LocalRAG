"""Token-aware chunking with overlap (tiktoken cl100k_base)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import tiktoken


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, *, chunk_tokens: int = 800, overlap_tokens: int = 100) -> List[Chunk]:
    if not text:
        return []
    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must exceed overlap_tokens")

    enc = _encoder()
    ids = enc.encode(text)
    stride = chunk_tokens - overlap_tokens
    chunks: List[Chunk] = []
    idx = 0
    start = 0
    while start < len(ids):
        end = min(start + chunk_tokens, len(ids))
        chunk_ids = ids[start:end]
        chunks.append(Chunk(index=idx, text=enc.decode(chunk_ids)))
        idx += 1
        if end == len(ids):
            break
        start += stride
    return chunks
