"""Per-KB ``rag_config`` resolution.

Each ``knowledge_bases`` row carries an optional ``rag_config`` JSONB
column (see migration 006) where admins stamp retrieval-quality
preferences — e.g. "this KB is huge, always rerank with a window=2
context expansion". At request time the bridge collects the configs of
all KBs the chat selected, merges them into a single override dict, and
wraps the retrieval pipeline in ``flags.with_overrides(...)`` so each
stage reads the correct values without touching ``os.environ`` (shared
process state, would leak across concurrent requests).

Multi-KB merge policy (UNION / MAX — strictest wins)
----------------------------------------------------
If the user selects multiple KBs for one chat, the merge picks whichever
setting runs the HEAVIER pipeline. The reasoning:

* Rerank/MMR/expand are opt-in quality boosts. If ANY selected KB needs
  rerank (``true``), the chat runs rerank — we can't re-rank only half
  the candidates.
* For numeric thresholds (``context_expand_window``, ``mmr_lambda``,
  ``rerank_top_k``) the larger value wins so the most-demanding KB's
  budget is honoured. The strictest KB effectively "pulls up" the rest.
* Spotlight / semcache: same union rule; treat ``true`` as dominant.

This is conservative on latency (you pay the most-expensive KB's cost)
but correct on quality (no silent downgrade when a small-KB-specific
fast path would otherwise suppress a heavier KB's preference).

Valid keys (anything else is silently dropped — admin UI should refuse
to set unknown keys, but we belt-and-brace at the service layer too)::

    rerank                bool    → RAG_RERANK
    rerank_top_k          int     → RAG_RERANK_TOP_K
    mmr                   bool    → RAG_MMR
    mmr_lambda            float   → RAG_MMR_LAMBDA
    context_expand        bool    → RAG_CONTEXT_EXPAND
    context_expand_window int     → RAG_CONTEXT_EXPAND_WINDOW
    spotlight             bool    → RAG_SPOTLIGHT
    semcache              bool    → RAG_SEMCACHE
    contextualize_on_ingest bool  → RAG_CONTEXTUALIZE_KBS  (ingest-side;
                                     recorded but NOT applied via the
                                     flag overlay — the overlay is
                                     request-scoped and ingest is a
                                     separate process path)
    hyde                  bool    → RAG_HYDE   (P3.3 — enable Hypothetical
                                     Document Embeddings per KB)
    hyde_n                int     → RAG_HYDE_N (number of hypothetical
                                     excerpts to average; default 1)

An empty config ``{}`` means "inherit process defaults" (no overlay
entry emitted — ``flags.get`` falls through to ``os.environ``).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

# Whitelist — any other key is filtered out. Keep this list in sync with
# the PATCH /api/kb/{kb_id}/config validator and the docs in
# ``docs/rag-per-kb-config.md``.
VALID_BOOL_KEYS = frozenset({
    "rerank",
    "mmr",
    "context_expand",
    "spotlight",
    "semcache",
    "contextualize_on_ingest",
    # P3.3: HyDE (Hypothetical Document Embeddings) — per-KB override so a
    # year-long, abstract-query-heavy KB can opt in without flipping the
    # global RAG_HYDE process flag.
    "hyde",
})
VALID_INT_KEYS = frozenset({
    "rerank_top_k",
    "context_expand_window",
    # P3.3: number of hypothetical-doc generations to average. Higher N
    # improves retrieval quality on abstract queries at the cost of N
    # extra chat calls (parallel, so wall-time cost is roughly constant).
    "hyde_n",
})
VALID_FLOAT_KEYS = frozenset({
    "mmr_lambda",
})
VALID_KEYS = VALID_BOOL_KEYS | VALID_INT_KEYS | VALID_FLOAT_KEYS

# Keys that are NOT propagated into the request-scope overlay because the
# underlying flag is read by the ingest process, not the retrieval hot
# path. We still accept them in rag_config (so admin UIs can stamp them
# for future ingest runs) but they do not influence a live chat request.
INGEST_ONLY_KEYS = frozenset({"contextualize_on_ingest"})

# Mapping from JSON config key -> RAG_* env var name. Values are stringified
# at overlay time because ``flags.get`` / ``os.environ.get`` return strings.
_KEY_TO_ENV: dict[str, str] = {
    "rerank": "RAG_RERANK",
    "rerank_top_k": "RAG_RERANK_TOP_K",
    "mmr": "RAG_MMR",
    "mmr_lambda": "RAG_MMR_LAMBDA",
    "context_expand": "RAG_CONTEXT_EXPAND",
    "context_expand_window": "RAG_CONTEXT_EXPAND_WINDOW",
    "spotlight": "RAG_SPOTLIGHT",
    "semcache": "RAG_SEMCACHE",
    "contextualize_on_ingest": "RAG_CONTEXTUALIZE_KBS",
    "hyde": "RAG_HYDE",
    "hyde_n": "RAG_HYDE_N",
}


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Strip unknown keys and coerce values to the expected Python type.

    Returns a fresh dict containing only whitelisted keys with coerced
    values. Does not raise on bad input — unknown keys are dropped,
    un-coerceable values are dropped. Callers that want hard validation
    should check that the returned dict has the same keys as the input.

    Admin PATCH handlers call this before persisting so a caller can't
    sneak arbitrary JSON into the JSONB column.
    """
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in VALID_KEYS:
            continue
        if key in VALID_BOOL_KEYS:
            if isinstance(value, bool):
                out[key] = value
            elif isinstance(value, str):
                out[key] = value.lower() in ("1", "true", "yes", "on")
            elif isinstance(value, (int, float)):
                out[key] = bool(value)
            # else: drop
        elif key in VALID_INT_KEYS:
            try:
                out[key] = int(value)
            except (TypeError, ValueError):
                continue
        elif key in VALID_FLOAT_KEYS:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def merge_configs(configs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge per-KB configs into a single effective config.

    Policy:
      * Booleans: ``any(...)``  (union — if any KB wants it, enable it)
      * Ints:     ``max(...)``  (strictest — largest window wins)
      * Floats:   ``max(...)``  (same reasoning — higher MMR lambda
                                 biases relevance over diversity; we pick
                                 the value closer to the "pure relevance"
                                 extreme to be conservative)

    Each input dict is ``validate_config``-cleaned first so unknown/
    malformed keys are dropped before the merge. Empty/None inputs are
    skipped (treated as "no opinion").

    Returns the merged config. An empty merge (no non-empty inputs) is
    an empty dict — the caller interprets this as "use process defaults".
    """
    merged: dict[str, Any] = {}
    for raw in configs:
        if not raw:
            continue
        cleaned = validate_config(raw)
        for key, value in cleaned.items():
            if key in VALID_BOOL_KEYS:
                merged[key] = bool(merged.get(key, False)) or bool(value)
            elif key in VALID_INT_KEYS:
                prev = merged.get(key)
                merged[key] = max(int(value), int(prev)) if prev is not None else int(value)
            elif key in VALID_FLOAT_KEYS:
                prev = merged.get(key)
                merged[key] = max(float(value), float(prev)) if prev is not None else float(value)
    return merged


def config_to_env_overrides(merged: Mapping[str, Any]) -> dict[str, str]:
    """Convert a merged config dict into ``{RAG_*: "string"}`` for the flag overlay.

    Booleans serialize to ``"1"``/``"0"`` — matching the existing env-var
    conventions read by retriever.py, reranker.py, etc. Ints/floats are
    ``str(...)``-converted. Unknown keys (shouldn't happen after
    ``validate_config``) are dropped defensively.

    Ingest-only keys (``contextualize_on_ingest``) are skipped: the flag
    overlay is request-scoped, so setting it would have no effect and
    would confuse operators reading the overlay in logs.
    """
    out: dict[str, str] = {}
    for key, value in merged.items():
        if key in INGEST_ONLY_KEYS:
            continue
        env = _KEY_TO_ENV.get(key)
        if env is None:
            continue
        if isinstance(value, bool):
            out[env] = "1" if value else "0"
        else:
            out[env] = str(value)
    return out


def with_overrides(overrides: Mapping[str, str]):
    """Re-export of ``ext.services.flags.with_overrides`` for callers that
    only need the KB-config surface.

    This is a courtesy alias — the bridge imports both ``merge_configs``
    and ``with_overrides`` from ``kb_config`` for locality. The actual
    implementation lives in ``flags.py``.
    """
    from .flags import with_overrides as _with_overrides
    return _with_overrides(overrides)


__all__ = [
    "VALID_KEYS",
    "merge_configs",
    "config_to_env_overrides",
    "validate_config",
    "with_overrides",
]
