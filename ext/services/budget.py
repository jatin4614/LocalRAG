"""Token-budget the reranked chunks — drop from lowest rank end until we fit."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

from .obs import span
from .vector_store import Hit


logger = logging.getLogger("rag.budget")
# Mirror logger named after the module path so caplog filters on
# ``ext.services.budget`` (used by the preflight tests + standard library
# convention) capture preflight messages too.
_module_logger = logging.getLogger(__name__)


class TokenizerPreflightError(RuntimeError):
    """Raised when an explicitly-configured tokenizer fails to load at startup.

    Phase 1.1: silent fallback to cl100k from a non-cl100k alias drifts
    token counts by ~10-15%% which evicts relevant chunks from the budget
    pass. We surface this loudly at startup rather than letting it limp
    along in production.
    """


# Tokenizer registry: maps RAG_BUDGET_TOKENIZER alias -> backend spec.
#   kind="tiktoken" => use a tiktoken encoding by id
#   kind="hf"       => transformers.AutoTokenizer.from_pretrained(id)
#
# Adding a new chat-model family is a 2-line change: pick an alias, point at
# the HF repo. RAG_BUDGET_TOKENIZER_MODEL (if set at lookup time) overrides
# the default id for ALL hf aliases — see _resolve_spec below. Pinned IDs
# remain authoritative when the env var is unset.
_TOKENIZER_REGISTRY: dict[str, dict[str, str]] = {
    "cl100k": {"kind": "tiktoken", "id": "cl100k_base"},
    "qwen": {"kind": "hf", "id": "Qwen/Qwen2.5-14B-Instruct"},
    "qwen2.5": {"kind": "hf", "id": "Qwen/Qwen2.5-14B-Instruct"},
    "gemma": {"kind": "hf", "id": "google/gemma-4-31B-it"},
    "gemma-4": {"kind": "hf", "id": "google/gemma-4-31B-it"},
    "gemma-4-31b": {"kind": "hf", "id": "google/gemma-4-31B-it"},
    "gemma-3": {"kind": "hf", "id": "google/gemma-3-27b-it"},
    "gemma-3-12b": {"kind": "hf", "id": "google/gemma-3-12b-it"},
    "llama": {"kind": "hf", "id": "meta-llama/Llama-3-8B-Instruct"},
}


def _resolve_spec(alias: str) -> dict[str, str] | None:
    """Look up an alias and apply ``RAG_BUDGET_TOKENIZER_MODEL`` override.

    Returns a fresh dict (never mutates the registry). For ``hf`` aliases,
    if ``RAG_BUDGET_TOKENIZER_MODEL`` is set, the resolved ``id`` is the
    operator-supplied value; otherwise the pinned default is used.

    Override applies uniformly to every ``hf`` alias. The env var is read
    at lookup time (not at import) so pytest's ``monkeypatch.setenv`` flips
    behavior without an import reload — the lru_cache on ``get_tokenizer``
    is what pins the resolved tokenizer per process.
    """
    spec = _TOKENIZER_REGISTRY.get(alias)
    if spec is None:
        return None
    if spec.get("kind") != "hf":
        return dict(spec)
    override = os.environ.get("RAG_BUDGET_TOKENIZER_MODEL")
    if override:
        return {"kind": "hf", "id": override}
    return dict(spec)


def _get_tokenizer_alias() -> str:
    return os.environ.get("RAG_BUDGET_TOKENIZER", "cl100k").lower()


class _TokenizerHandle:
    """Uniform encode/decode surface over tiktoken and HF tokenizers.

    The chunker needs ``encode(text) -> list[int]`` and ``decode(ids) -> str``;
    the budget enforcer just needs ``len(encode(text))``. Wrapping both
    backends in a single shape lets ``ext.services.chunker`` and
    ``ext.services.budget`` share one cached instance — chunk boundaries
    and budget counts then use identical token vocabularies.

    For HF backends we always pass ``add_special_tokens=False`` so chunk
    sizes match real-prompt token counts (special tokens are added by the
    chat template, not per chunk).
    """

    __slots__ = ("_kind", "_inner")

    def __init__(self, kind: str, inner) -> None:
        self._kind = kind
        self._inner = inner

    @property
    def kind(self) -> str:
        return self._kind

    def encode(self, text: str) -> list[int]:
        if self._kind == "hf":
            return list(self._inner.encode(text, add_special_tokens=False))
        # tiktoken
        return list(self._inner.encode(text))

    def decode(self, ids) -> str:
        if self._kind == "hf":
            # skip_special_tokens defaults to False on most HF tokenizers;
            # we pass True so a hard-split chunk that grabs a tail token
            # which happens to be <eos>/<bos> doesn't leak control tokens
            # into the chunk text.
            return self._inner.decode(list(ids), skip_special_tokens=True)
        return self._inner.decode(list(ids))


def _make_tiktoken_handle() -> _TokenizerHandle:
    """Load cl100k_base via tiktoken. Owned by budget.py to keep the
    chunker -> budget import direction one-way."""
    import tiktoken  # local import — keeps chunker free of tiktoken at module load
    return _TokenizerHandle("tiktoken", tiktoken.get_encoding("cl100k_base"))


@lru_cache(maxsize=1)
def get_tokenizer() -> _TokenizerHandle:
    """Return the shared tokenizer handle.

    Both the budget enforcer and the ingest-time chunker route through
    this so chunk boundaries are sized in the same token vocabulary the
    prompt-budget pass will count against. Cached for the life of the
    process — tokenizer load is the dominant first-call cost.
    """
    alias = _get_tokenizer_alias()
    spec = _resolve_spec(alias)

    if spec is None:
        logger.warning(
            "RAG_BUDGET_TOKENIZER=%r is not registered; falling back to cl100k. "
            "Known aliases: %s",
            alias,
            ", ".join(sorted(_TOKENIZER_REGISTRY)),
        )
        return _make_tiktoken_handle()

    kind = spec["kind"]
    ident = spec["id"]

    if kind == "hf":
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(ident)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load HF tokenizer %r for alias %r (%s); falling "
                "back to cl100k. Tip: gated repos (e.g. google/gemma-*) "
                "require HF_TOKEN with accepted license.",
                ident, alias, exc,
            )
            # Phase 1.1: surface runtime fallbacks so dashboards can
            # alert. Defensive — preflight should crash before we reach
            # here, but the cache could disappear after startup.
            try:
                from ext.services.metrics import tokenizer_fallback_total
                tokenizer_fallback_total.labels(from_alias=alias, to="cl100k").inc()
            except Exception:  # pragma: no cover - metrics is fail-open
                pass
            return _make_tiktoken_handle()
        return _TokenizerHandle("hf", tok)

    if kind == "tiktoken":
        return _make_tiktoken_handle()

    logger.warning("unknown tokenizer kind %r; falling back to cl100k", kind)
    return _make_tiktoken_handle()


@lru_cache(maxsize=1)
def _budget_tokenizer():
    """Return a callable (text -> int) that counts tokens.

    Alias is chosen via RAG_BUDGET_TOKENIZER (default: cl100k, matches the
    old behavior). Unknown alias logs a warning and falls back to cl100k.

    Counts come from the same handle the chunker uses — see
    :func:`get_tokenizer` — so the prompt-budget pass and ingest-time
    chunk sizing share one token vocabulary.
    """
    handle = get_tokenizer()

    def _count(text: str) -> int:
        return len(handle.encode(text))
    return _count


def _count_tokens(text: str) -> int:
    return _budget_tokenizer()(text)


def preflight_tokenizer() -> None:
    """Validate that the configured tokenizer loads. Called at app startup.

    Rule: if ``RAG_BUDGET_TOKENIZER`` is set to a non-cl100k alias backed
    by an HF tokenizer, failure to load that tokenizer raises
    :class:`TokenizerPreflightError` and crashes the process. Silent
    fallback to cl100k would cause ~10-15%% token-budget drift, which
    evicts relevant chunks. If the operator explicitly asked for
    ``gemma-4`` but we fall back to cl100k, ``budget.py`` then lies
    about how many tokens fit and reranked context gets clipped.

    Non-fatal cases (return cleanly):
    - Unset env var → default cl100k path.
    - Explicit ``cl100k`` → same as default.
    - Unknown alias → operator typo; the runtime fallback to cl100k is
      safer than crashing on a typo (matches existing ``_budget_tokenizer``
      behavior).
    - tiktoken-backed alias → no remote dep, nothing to preload.
    """
    alias = os.environ.get("RAG_BUDGET_TOKENIZER")
    if not alias or alias.lower() == "cl100k":
        msg = "tokenizer preflight: using cl100k (default or explicit)"
        logger.info(msg)
        _module_logger.info(msg)
        return
    alias = alias.lower()
    spec = _resolve_spec(alias)
    if spec is None:
        # Unknown alias — log but don't crash, consistent with the
        # existing _budget_tokenizer fallback path. Mirror to both
        # loggers so caplog filters on either name catch it.
        warn_msg = (
            f"tokenizer preflight: unknown alias {alias!r} — "
            f"will fall back to cl100k at runtime"
        )
        logger.warning(warn_msg)
        _module_logger.warning(warn_msg)
        return
    if spec.get("kind") != "hf":
        # tiktoken (and any future non-network kinds) don't need a
        # network preload — the chunker's encoder loads on first use.
        return
    ident = spec["id"]
    try:
        from transformers import AutoTokenizer  # type: ignore
        AutoTokenizer.from_pretrained(ident)
    except Exception as exc:  # noqa: BLE001
        # Surface to metrics first — even if the raise prevents the
        # process from coming up, an operator scraping /metrics during
        # the failed boot will see the counter tick.
        try:
            from ext.services.metrics import tokenizer_fallback_total
            tokenizer_fallback_total.labels(
                from_alias=alias, to="cl100k_forced_crash"
            ).inc()
        except Exception:  # pragma: no cover - metrics is fail-open
            pass
        raise TokenizerPreflightError(
            f"RAG_BUDGET_TOKENIZER={alias!r} (model={ident!r}) failed to "
            f"load ({type(exc).__name__}: {exc}). "
            f"Silent fallback to cl100k would cause ~10-15% token-budget "
            f"drift. Either (a) ensure the tokenizer is in the HF cache "
            f"at HF_HOME (see Plan A Appendix A for air-gap staging), or "
            f"(b) set RAG_BUDGET_TOKENIZER=cl100k to accept the drift "
            f"explicitly."
        ) from exc
    msg = f"tokenizer preflight: {alias} (model={ident}) loaded successfully"
    logger.info(msg)
    _module_logger.info(msg)


def budget_chunks(hits: List[Hit], *, max_tokens: int = 4000) -> List[Hit]:
    """Assumes hits is pre-sorted best-first. Returns longest prefix that fits."""
    with span("budget.truncate", input_size=len(hits), max_tokens=max_tokens) as _sp:
        kept: list[Hit] = []
        total = 0
        dropped = 0
        for h in hits:
            t = _count_tokens(str(h.payload.get("text", "")))
            if total + t > max_tokens:
                dropped += 1
                continue
            total += t
            kept.append(h)
        if dropped:
            logger.debug("budget dropped %d of %d chunks (used %d/%d tokens)",
                         dropped, len(hits), total, max_tokens)
        try:
            _sp.set_attribute("output_size", len(kept))
            _sp.set_attribute("tokens_used", total)
            _sp.set_attribute("dropped", dropped)
        except Exception:
            pass
        return kept
