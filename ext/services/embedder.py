"""Embedder protocol + deterministic StubEmbedder + TEI HTTP client."""
from __future__ import annotations

import hashlib
import os
import struct
from functools import lru_cache
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

        Bug-fix campaign §3.5 — when ``RAG_CB_TEI_ENABLED=1`` the call
        also consults a per-process circuit breaker keyed ``"tei"``. If
        the breaker is open (TEI has failed ``RAG_CB_FAIL_THRESHOLD``
        times in ``RAG_CB_WINDOW_SEC``), the breaker raises
        :class:`CircuitOpenError` BEFORE the network call so we don't
        keep hammering a known-broken endpoint. Callers fail-open per
        CLAUDE.md §1.2 (treat ``CircuitOpenError`` as another transient
        upstream failure and degrade retrieval gracefully).

        The breaker only counts terminal failures: success → record_success,
        any exception (including the retry decorator's final reraise) →
        record_failure. The retry decorator runs INSIDE one breaker call
        so a single user-facing embed surfaces as one breaker decision,
        not N (avoids opening the breaker on a single transient blip
        that the retry would have absorbed).
        """
        cb_enabled = os.environ.get("RAG_CB_TEI_ENABLED", "0") == "1"
        breaker = None
        if cb_enabled:
            # Local import — keeps the breaker module out of the import
            # path when the flag is off (cold-start matters here).
            from .circuit_breaker import breaker_for
            breaker = breaker_for("tei")
            breaker.raise_if_open()

        try:
            headers = inject_context_into_headers({})
            r = await self._client.post(
                "/embed",
                json={"inputs": batch},
                headers=headers or None,
            )
            r.raise_for_status()
            out = r.json()
        except Exception:
            if breaker is not None:
                breaker.record_failure()
            raise
        if breaker is not None:
            breaker.record_success()
        return out

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Direct TEI passthrough — bypasses the embed cache. Internal
        only; call sites should use :meth:`embed` (which adds caching
        when enabled). Exposed so the cache layer can fill misses
        without re-entering the cache check (infinite recursion).
        """
        out: list[list[float]] = []
        for i in range(0, len(texts), self._max_batch):
            batch = texts[i : i + self._max_batch]
            out.extend(await self._embed_batch(batch))
        return out

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Heuristic: callers embedding a single input are usually the query
        # path (retriever/hyde); larger batches come from ingest/doc paths.
        _path = "query" if len(texts) == 1 else "doc"
        _bytes = sum(len(t) for t in texts)

        # Bug-fix campaign §3.2 — Redis-backed embed cache. Single-text
        # path (queries) is the natural cache fit; ingest batches are
        # already content-deduped by blob_sha so caching there is mostly
        # redundant. Default-OFF behind RAG_EMBED_CACHE_ENABLED, so this
        # is a passthrough until the operator flips the flag.
        #
        # Pass ``_UncachedView(self)`` to the cache so a miss falls
        # through to ``_embed_uncached`` instead of re-entering this
        # method (which would re-check the cache → infinite recursion).
        if (
            len(texts) == 1
            and os.environ.get("RAG_EMBED_CACHE_ENABLED", "0") == "1"
        ):
            from .embed_cache import get_or_set as _embed_cache_get_or_set

            model_version = os.environ.get("EMBED_MODEL", "unknown")
            with span(
                "embed.call",
                path=_path,
                batch_size=1,
                bytes=_bytes,
                cached=True,
            ):
                vec = await _embed_cache_get_or_set(
                    texts[0],
                    model_version,
                    _UncachedView(self),
                )
                return [vec]

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


class _UncachedView:
    """Embedder protocol facade that delegates to ``_embed_uncached``.

    Lets the embed cache fill misses without re-entering :meth:`TEIEmbedder.embed`
    (which would re-check the cache → infinite recursion).
    """

    def __init__(self, inner: "TEIEmbedder") -> None:
        self._inner = inner

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._inner._embed_uncached(texts)


# --- P3.4 ColBERT multi-vector (late interaction) embedder -----------------
# Used as the third vector slot on hybrid+colbert-shaped collections. Unlike
# dense (one vector / chunk) and sparse (one (idx, val) pair / chunk),
# ColBERT produces a token-level matrix — one 128-dim vector per WordPiece
# token. Qdrant stores this under a named vector with
# ``MultiVectorConfig(comparator=MAX_SIM)`` and computes max-similarity at
# query time. The write path here just produces JSON-serialisable lists;
# read-side fusion lands in Task 3.5.

def _fastembed_providers() -> list[str] | None:
    """Resolve ONNX execution providers for fastembed.

    Priority order:
      1. Explicit override via ``RAG_FASTEMBED_PROVIDERS`` (CSV, e.g.
         ``"CUDAExecutionProvider,CPUExecutionProvider"``).
      2. Auto-detect: if ``CUDAExecutionProvider`` is available in the
         loaded onnxruntime, prefer it with a CPU fallback. Without GPU
         build (only ``CPUExecutionProvider`` available) returns ``None``
         which lets fastembed pick its own default — byte-equivalent to
         pre-Phase-A behaviour.

    fastembed accepts ``providers=[...]`` on ``LateInteractionTextEmbedding``
    and ``SparseTextEmbedding`` constructors. CUDA-bundled distributions
    (``onnxruntime-gpu`` + ``fastembed-gpu``) ship CUDA libs inside the
    wheel so no host-side CUDA toolkit is required as long as the host
    GPU driver is recent enough (550+ for CUDA 12.x).

    Returns None when CPU-only — caller passes ``providers=None`` and
    fastembed uses its built-in default. Returning an explicit ``["CPU"]``
    list would suppress fastembed's debug log line about provider choice
    which is useful during the GPU rollout.
    """
    override = os.environ.get("RAG_FASTEMBED_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _colbert_model():
    """Lazy singleton fastembed LateInteractionTextEmbedding.

    Loaded on first call from the local fastembed cache (operator
    Appendix A.2 hydrates ``/var/models/fastembed_cache/`` ahead of an
    air-gapped deploy). Process-wide single instance — fastembed's ONNX
    session itself isn't free to spin up, and we don't want N concurrent
    ingest workers each holding their own copy.

    GPU acceleration: when ``onnxruntime-gpu`` is installed (typically
    via the ``fastembed-gpu`` distribution), :func:`_fastembed_providers`
    returns ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` and the
    model runs on GPU. CPU-only installs return ``None`` and fastembed
    silently falls back. Set ``RAG_FASTEMBED_PROVIDERS`` to override.

    Failures (model missing, ONNX runtime issue, fastembed not
    installed) propagate to the caller; ``colbert_embed`` does not
    swallow them. Ingest sites wrap colbert_embed in a try/except so a
    stale cache doesn't break the dense + sparse arms.
    """
    from fastembed import LateInteractionTextEmbedding

    model_name = os.environ.get("RAG_COLBERT_MODEL", "colbert-ir/colbertv2.0")
    providers = _fastembed_providers()
    kwargs = {"model_name": model_name}
    if providers is not None:
        kwargs["providers"] = providers
    return LateInteractionTextEmbedding(**kwargs)


def colbert_embed(texts: list[str]) -> list[list[list[float]]]:
    """Compute ColBERT multi-vectors: ``out[text_idx][token_idx][dim]``.

    fastembed's ``embed()`` yields numpy arrays of shape
    ``(n_tokens, 128)``; we convert to nested python lists so the
    result is JSON-serialisable for the Qdrant payload (multi-vector
    POST goes over HTTP / qdrant-client). Token count varies per text;
    Qdrant accepts a list of fixed-dim vectors and computes max-sim at
    query time.

    Empty input returns an empty list — matches the Embedder protocol's
    contract for ``embed([])``.
    """
    if not texts:
        return []
    model = _colbert_model()
    out: list[list[list[float]]] = []
    for arr in model.embed(list(texts)):
        # ``arr`` is np.ndarray of shape (n_tokens, 128) — coerce each
        # token row to a python list of python floats so
        # qdrant-client / json.dumps don't choke on numpy types.
        # Wave 2 (review §3.10): np.ndarray.tolist() is the C-implemented
        # vectorised path; the prior nested list comp dispatched float()
        # per element (~10× slower on 200-tok chunks).
        out.append(arr.tolist())
    return out
