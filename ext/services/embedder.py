"""Embedder protocol + deterministic StubEmbedder + TEI HTTP client."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import struct
from functools import lru_cache
from typing import Optional, Protocol

import httpx

from .obs import inject_context_into_headers, span

log = logging.getLogger(__name__)


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


# --- Embedder redundancy helpers ------------------------------------------
# 2026-05-03: TEIEmbedder.embed gained a retry-with-halving redundancy
# layer to absorb transient TEI 424s (CUDA OOM under shared-GPU
# pressure). These helpers live at module scope so they're trivially
# testable and have no implicit ``self`` dependencies.

# Retryable HTTP status codes from TEI:
#   424 — TEI maps the underlying CUDA OOM to a "Failed Dependency"
#         response. This is THE production trigger for the 2026-05-03
#         redundancy work.
#   429 — rate limit; vLLM and TEI both emit it under load.
#   5xx — generic server error (502/503/504 = upstream / gateway issues
#         that frequently self-heal in seconds).
# 4xx other than 424/429 are NOT retryable: 400/401/403/404/422 are
# permanent input/auth/route problems and retrying just amplifies
# pressure. Surface them so the operator notices.
_RETRYABLE_STATUS = frozenset({424, 429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    """True if an embedder failure may succeed on a subsequent attempt.

    Retryable network exceptions cover the typical transient failure
    modes: ``ReadTimeout`` (TEI stalled past the client timeout),
    ``ConnectError`` (TCP refused — TEI restarting),
    ``RemoteProtocolError`` (connection dropped mid-response),
    ``ReadError`` (peer reset). Other ``httpx.RequestError`` subclasses
    fall through as non-retryable so we don't mask real client bugs.
    """
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


def _classify_retry_reason(exc: BaseException | None) -> str:
    """Bucket an exception into a low-cardinality reason label.

    Buckets: ``"424"`` (TEI OOM, the prod trigger), ``"429"`` (rate
    limit), ``"5xx"`` (other server errors), ``"network"`` (timeouts /
    connection drops), ``"unknown"`` (defensive — never expected since
    callers filter via :func:`_is_retryable` first).
    """
    if exc is None:
        return "unknown"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 424:
            return "424"
        if status == 429:
            return "429"
        if 500 <= status < 600:
            return "5xx"
        return "unknown"
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ),
    ):
        return "network"
    return "unknown"


def _size_bucket(n: int) -> str:
    """Bucket a batch size into a power-of-two label class.

    Buckets: ``"1"``, ``"2-4"``, ``"5-8"``, ``"9-16"``, ``"17-32"``,
    ``"33+"``. Bounded cardinality even if some pathological caller
    embeds huge batches (the upper bound is a single bucket label, not
    a per-size label).
    """
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 8:
        return "5-8"
    if n <= 16:
        return "9-16"
    if n <= 32:
        return "17-32"
    return "33+"


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

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """POST one TEI batch (≤ ``_max_batch`` inputs) — single shot.

        Pure HTTP: no retry, no breaker. The retry + halving wrapper
        :meth:`_embed_with_redundancy` provides redundancy; the
        circuit-breaker integration lives in :meth:`_embed_dispatch`
        (one breaker decision per user-facing :meth:`embed` call, not
        per inner retry).

        2026-05-03: the prior tenacity ``@with_transient_retry``
        decorator was removed because the new
        :meth:`_embed_with_redundancy` superset (retry + halving + 424
        coverage) replaces it. The per-call breaker integration also
        moved out (was here, now wraps the entire redundancy cascade)
        so one user-visible embed surfaces as one breaker decision —
        the old contract from §3.5 is preserved.
        """
        headers = inject_context_into_headers({})
        r = await self._client.post(
            "/embed",
            json={"inputs": batch},
            headers=headers or None,
        )
        r.raise_for_status()
        return r.json()

    async def _embed_dispatch(
        self, batch: list[str]
    ) -> list[list[float]]:
        """Outer entry per per-batch embed — wraps the redundancy
        cascade in a single circuit-breaker decision.

        Bug-fix campaign §3.5 (preserved): when ``RAG_CB_TEI_ENABLED=1``
        the call consults a per-process circuit breaker keyed
        ``"tei"``. If the breaker is open (TEI has failed
        ``RAG_CB_FAIL_THRESHOLD`` times in ``RAG_CB_WINDOW_SEC``), the
        breaker raises :class:`CircuitOpenError` BEFORE any retry chatter
        so we don't keep hammering a known-broken endpoint. Callers
        fail-open per CLAUDE.md §1.2.

        Critical contract (unchanged across the 2026-05-03 rewrite):
        ONE user-facing embed call → ONE breaker decision. The retry +
        halving cascade runs INSIDE the breaker call so a single
        transient blip the redundancy absorbs counts as exactly one
        breaker success, not N (avoids tripping the breaker on intra-
        request retries the redundancy layer was designed to absorb).
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
            out = await self._embed_with_redundancy(batch)
        except Exception:
            if breaker is not None:
                breaker.record_failure()
            raise
        if breaker is not None:
            breaker.record_success()
        return out

    async def _embed_with_redundancy(
        self, batch: list[str]
    ) -> list[list[float]]:
        """Wrap :meth:`_embed_batch` with retry-then-halve redundancy.

        Failure mode being mitigated: GPU 1 (24 GB shared TEI + reranker
        + colbert + fastembed + vllm-qu) runs at ~95% steady-state.
        Under fast-changing GPU pressure TEI's per-forward activation
        can OOM and return ``424``
        (``DriverError(CUDA_ERROR_OUT_OF_MEMORY)``). Pre-fix that single
        424 failed the entire ingest task with no retry.

        Strategy:
        1. Per-call retry budget at the SAME batch size. Up to
           ``RAG_EMBED_MAX_RETRIES`` (default 3) attempts with
           exponential backoff (0.5s, 1s, 2s).
        2. Retryable HTTP statuses: 424, 429, 500, 502, 503, 504.
           Retryable network exceptions: ``ReadTimeout``,
           ``ConnectError``, ``RemoteProtocolError``, ``ReadError``.
           Anything else (400/401/403/404/422 + non-network exceptions)
           surfaces immediately — those are real input/auth bugs, not
           transient pressure.
        3. After the retry budget at the current batch size is exhausted
           AND ``len(batch) > 1``: halve the batch and recurse on each
           half, then concatenate the results in order. Halving
           naturally lowers TEI's per-forward memory pressure.
        4. Recursion floor at ``len(batch) == 1``: if a single text
           still can't go through, raise — that's a real per-chunk
           problem (TEI down, network partition, or a chunk that
           genuinely won't fit).

        Order preservation: the first half of the input always lands at
        positions ``[0:len/2]`` of the output, the second half at
        ``[len/2:]``. Embedding-to-text alignment is load-bearing for
        the downstream Qdrant upsert (deterministic UUIDv5 IDs derive
        from chunk_index), so we MUST not reorder under any retry path.
        """
        if not batch:
            return []
        try:
            max_retries = int(os.environ.get("RAG_EMBED_MAX_RETRIES", "3"))
        except (TypeError, ValueError):
            max_retries = 3
        max_retries = max(1, max_retries)

        # Same-batch retry loop with exponential backoff.
        last_exc: BaseException | None = None
        for attempt in range(1, max_retries + 1):
            try:
                out = await self._embed_batch(batch)
                if attempt > 1:
                    # We recovered after >=1 failed attempt at this batch
                    # size — bump the recovered counter for the original
                    # cause so operators can trend the redundancy
                    # absorbing real pressure.
                    self._record_retry(
                        outcome="recovered",
                        reason=_classify_retry_reason(last_exc),
                    )
                return out
            except Exception as exc:  # noqa: BLE001 — gate on retryability
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                if attempt < max_retries:
                    log.info(
                        "embedder: retry attempt=%d batch=%d reason=%s",
                        attempt + 1, len(batch), type(exc).__name__,
                    )
                    # Exponential backoff: 0.5s, 1s, 2s (capped); the
                    # caller-provided fast_sleep fixture in tests
                    # short-circuits this to ~0.
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        # Retry budget exhausted at this batch size. Decide: halve or
        # raise.
        assert last_exc is not None  # loop only exits via raise above
        reason = _classify_retry_reason(last_exc)
        self._record_retry(outcome="exhausted", reason=reason)

        if len(batch) <= 1:
            # Recursion floor — surface the original retryable error.
            log.warning(
                "embedder: exhausted at batch=1 reason=%s — surfacing",
                type(last_exc).__name__,
            )
            raise last_exc

        # Halve and recurse. Order preserved: first half at [0:mid],
        # second half at [mid:].
        mid = len(batch) // 2
        new_size = mid  # the new per-call batch length
        log.info(
            "embedder: halving batch %d → %d reason=%s",
            len(batch), new_size, type(last_exc).__name__,
        )
        self._record_halving(new_size)

        first = await self._embed_with_redundancy(batch[:mid])
        second = await self._embed_with_redundancy(batch[mid:])
        return first + second

    @staticmethod
    def _record_retry(*, outcome: str, reason: str) -> None:
        """Bump ``embedder_retry_total{outcome,reason}``.

        Fail-open: any metric error is swallowed so the redundancy path
        is never disturbed by an exporter misconfiguration.
        """
        try:
            from .metrics import embedder_retry_total
            embedder_retry_total.labels(outcome=outcome, reason=reason).inc()
        except Exception:  # noqa: BLE001 — metric must never break ingest
            pass

    @staticmethod
    def _record_halving(new_batch_size: int) -> None:
        """Bump ``embedder_halving_total{batch_size_class}``.

        Label is a coarse power-of-two bucket of the NEW batch size so
        the metric stays low-cardinality even under exotic input batches.
        """
        try:
            from .metrics import embedder_halving_total
            embedder_halving_total.labels(
                batch_size_class=_size_bucket(new_batch_size),
            ).inc()
        except Exception:  # noqa: BLE001 — metric must never break ingest
            pass

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Direct TEI passthrough — bypasses the embed cache. Internal
        only; call sites should use :meth:`embed` (which adds caching
        when enabled). Exposed so the cache layer can fill misses
        without re-entering the cache check (infinite recursion).
        """
        out: list[list[float]] = []
        for i in range(0, len(texts), self._max_batch):
            batch = texts[i : i + self._max_batch]
            out.extend(await self._embed_dispatch(batch))
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
                out.extend(await self._embed_dispatch(batch))
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
