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
# the HF repo. RAG_BUDGET_TOKENIZER_MODEL (if set) overrides the default id
# for the family aliases noted below.
_TOKENIZER_REGISTRY: dict[str, dict[str, str]] = {
    "cl100k": {"kind": "tiktoken", "id": "cl100k_base"},
    "qwen": {
        "kind": "hf",
        "id": os.environ.get("RAG_BUDGET_TOKENIZER_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
    },
    "qwen2.5": {"kind": "hf", "id": "Qwen/Qwen2.5-14B-Instruct"},
    "gemma": {
        "kind": "hf",
        "id": os.environ.get("RAG_BUDGET_TOKENIZER_MODEL", "google/gemma-4-31B-it"),
    },
    "gemma-4": {"kind": "hf", "id": "google/gemma-4-31B-it"},
    "gemma-4-31b": {"kind": "hf", "id": "google/gemma-4-31B-it"},
    "gemma-3": {"kind": "hf", "id": "google/gemma-3-27b-it"},
    "gemma-3-12b": {"kind": "hf", "id": "google/gemma-3-12b-it"},
    "llama": {
        "kind": "hf",
        "id": os.environ.get("RAG_BUDGET_TOKENIZER_MODEL", "meta-llama/Llama-3-8B-Instruct"),
    },
}


def _get_tokenizer_alias() -> str:
    return os.environ.get("RAG_BUDGET_TOKENIZER", "cl100k").lower()


@lru_cache(maxsize=1)
def _budget_tokenizer():
    """Return a callable (text -> int) that counts tokens.

    Alias is chosen via RAG_BUDGET_TOKENIZER (default: cl100k, matches the
    old behavior). Unknown alias logs a warning and falls back to cl100k.

    For HF-backed aliases (qwen / gemma / llama), the first call downloads
    tokenizer vocab. When ``RAG_BUDGET_TOKENIZER_MODEL`` is set, it overrides
    the default id for the family alias (qwen / gemma / llama). Exact-version
    aliases like 'qwen2.5', 'gemma-3', 'gemma-3-12b' always use their pinned id.
    """
    alias = _get_tokenizer_alias()
    spec = _TOKENIZER_REGISTRY.get(alias)

    if spec is None:
        logger.warning(
            "RAG_BUDGET_TOKENIZER=%r is not registered; falling back to cl100k. "
            "Known aliases: %s",
            alias,
            ", ".join(sorted(_TOKENIZER_REGISTRY)),
        )
        spec = _TOKENIZER_REGISTRY["cl100k"]

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
            return _cl100k_counter()

        def _count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))
        return _count

    if kind == "tiktoken":
        return _cl100k_counter()

    logger.warning("unknown tokenizer kind %r; falling back to cl100k", kind)
    return _cl100k_counter()


def _cl100k_counter():
    """Shared cl100k counter — reuses the chunker's encoder so we get caching."""
    from ext.services.chunker import _encoder
    enc = _encoder()

    def _count(text: str) -> int:
        return len(enc.encode(text))
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
    spec = _TOKENIZER_REGISTRY.get(alias)
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
