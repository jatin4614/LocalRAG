"""Embedder protocol + deterministic StubEmbedder + TEI HTTP client."""
from __future__ import annotations

import hashlib
import os
import struct
from typing import Optional, Protocol

import httpx

from .obs import inject_context_into_headers, span
from .retry_policy import with_transient_retry


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class StubEmbedder:
    """Hash-based deterministic embedder. Same text → same vector, always."""

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    def _hash_vector(self, text: str) -> list[float]:
        data = hashlib.shake_128(text.encode()).digest(self._dim * 4)
        raw = struct.unpack(f"<{self._dim}i", data)
        vec = [x / 2**31 for x in raw]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class TEIEmbedder:
    """HuggingFace Text-Embeddings-Inference client.

    TEI enforces a server-side max batch size (default 32 inputs per
    ``/embed`` request). Longer documents routinely produce more chunks
    than that, so we split the caller's list into batches of
    ``RAG_TEI_MAX_BATCH`` (default 32) and concatenate the results.
    Order is preserved.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        max_batch: Optional[int] = None,
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)
        if max_batch is None:
            try:
                max_batch = int(os.environ.get("RAG_TEI_MAX_BATCH", "32"))
            except (TypeError, ValueError):
                max_batch = 32
        self._max_batch = max(1, max_batch)

    async def aclose(self) -> None:
        await self._client.aclose()

    @with_transient_retry(attempts=3, base_sec=0.5)
    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """POST one TEI batch (≤ ``_max_batch`` inputs).

        Wrapped with the shared transient-error retry policy so a single
        TEI hiccup (GC pause, brief 5xx, network blip) doesn't degrade
        every concurrent request. The decorator is a no-op pass-through
        when ``RAG_TENACITY_RETRY=0``.
        """
        headers = inject_context_into_headers({})
        r = await self._client.post(
            "/embed",
            json={"inputs": batch},
            headers=headers or None,
        )
        r.raise_for_status()
        return r.json()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Heuristic: callers embedding a single input are usually the query
        # path (retriever/hyde); larger batches come from ingest/doc paths.
        _path = "query" if len(texts) == 1 else "doc"
        _bytes = sum(len(t) for t in texts)
        out: list[list[float]] = []
        with span(
            "embed.call",
            path=_path,
            batch_size=len(texts),
            bytes=_bytes,
        ):
            for i in range(0, len(texts), self._max_batch):
                batch = texts[i : i + self._max_batch]
                out.extend(await self._embed_batch(batch))
            return out
