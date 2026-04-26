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
    # Phase 3.3: short-form per-KB ingest gate consumed directly by
    # ``ingest.should_contextualize``. We keep ``contextualize_on_ingest``
    # accepted (legacy) but the new code path reads ``contextualize`` —
    # explicit ``True``/``False`` here overrides ``RAG_CONTEXTUALIZE_KBS``
    # in either direction (e.g. global ON, KB OFF → skip).
    "contextualize",
    # P3.3: HyDE (Hypothetical Document Embeddings) — per-KB override so a
    # year-long, abstract-query-heavy KB can opt in without flipping the
    # global RAG_HYDE process flag.
    "hyde",
    # Tier 1/2 (doc-summary index + intent router). ``doc_summaries`` is
    # ingest-only (it controls whether ingest emits a per-doc summary
    # point); the other two are request-scope flags read by the bridge.
    "doc_summaries",
    "intent_routing",
    "intent_llm",
})
VALID_INT_KEYS = frozenset({
    "rerank_top_k",
    "context_expand_window",
    # P3.3: number of hypothetical-doc generations to average. Higher N
    # improves retrieval quality on abstract queries at the cost of N
    # extra chat calls (parallel, so wall-time cost is roughly constant).
    "hyde_n",
    # Phase 1a / per-KB chunking. Ingest-only (not propagated to the flag
    # overlay) — see INGEST_ONLY_KEYS below. Corpora differ wildly: daily
    # fact-heavy reports want ~200-300 tokens, whitepapers ~600-800.
    # Bounds enforced in validate_config.
    "chunk_tokens",
    "overlap_tokens",
})
VALID_FLOAT_KEYS = frozenset({
    "mmr_lambda",
})
VALID_KEYS = VALID_BOOL_KEYS | VALID_INT_KEYS | VALID_FLOAT_KEYS

# Keys that are NOT propagated into the request-scope overlay because the
# underlying flag is read by the ingest process, not the retrieval hot
# path. We still accept them in rag_config (so admin UIs can stamp them
# for future ingest runs) but they do not influence a live chat request.
INGEST_ONLY_KEYS = frozenset({
    "contextualize_on_ingest",
    # Phase 3.3 short-form alias — ingest-only, consumed by
    # ``ingest.should_contextualize``. Stripping it from the overlay
    # keeps the request hot path free of an env var that wouldn't
    # influence retrieval anyway and avoids confusing operators reading
    # the overlay in logs.
    "contextualize",
    "chunk_tokens",
    "overlap_tokens",
    # doc_summaries affects what ingest emits (per-doc summary points),
    # not request-time routing — strip it from the flag overlay so it
    # doesn't leak into the retrieval hot path.
    "doc_summaries",
})

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
    "doc_summaries": "RAG_DOC_SUMMARIES",
    # The QU LLM flags (RAG_QU_*) are cluster-wide and not exposed as
    # per-KB rag_config keys.
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
                coerced = int(value)
            except (TypeError, ValueError):
                continue
            # Per-KB chunk-size bounds. 100 floor avoids over-fragmentation;
            # 2000 ceiling keeps chunks below typical embedder max_seq_len.
            # overlap_tokens must be strictly less than chunk_tokens and
            # non-negative. Values outside bounds are dropped silently so
            # the KB inherits process defaults rather than ingesting at a
            # corrupt size.
            if key == "chunk_tokens" and not (100 <= coerced <= 2000):
                continue
            if key == "overlap_tokens":
                # Cross-field check is handled at apply time (ingest reads
                # both); at validate time we only reject absurd values.
                if coerced < 0 or coerced > 1000:
                    continue
            out[key] = coerced
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


def resolve_chunk_params(
    raw_config: Mapping[str, Any] | None,
    *,
    env_chunk_size: int | None = None,
    env_chunk_overlap: int | None = None,
) -> tuple[int, int]:
    """Resolve ``(chunk_tokens, overlap_tokens)`` for a KB ingest call.

    Priority, highest wins:

    1. ``raw_config`` values (after ``validate_config`` cleans them), if the
       KB explicitly stamped ``chunk_tokens`` / ``overlap_tokens``.
    2. Process env defaults ``CHUNK_SIZE`` / ``CHUNK_OVERLAP``.
    3. Hard defaults ``800`` / ``100`` (matches the ``ingest_bytes``
       signature defaults so behaviour is unchanged for KBs without the
       keys and no env override).

    The two args are resolved independently — a KB that sets only
    ``chunk_tokens`` inherits the env overlap, and vice versa.

    Fail-closed: overlap that would exceed ``chunk_tokens // 2`` (the
    chunker's hard contract) is clipped down to ``chunk_tokens // 4`` so
    the call is still valid. A caller should have run ``validate_config``
    already so we never hit that path in practice — this is a safety net
    against hand-edited JSONB.
    """
    import os as _os

    cfg: Mapping[str, Any] = validate_config(raw_config) if raw_config else {}

    if env_chunk_size is None:
        try:
            env_chunk_size = int(_os.environ.get("CHUNK_SIZE", "800"))
        except (TypeError, ValueError):
            env_chunk_size = 800
    if env_chunk_overlap is None:
        try:
            env_chunk_overlap = int(_os.environ.get("CHUNK_OVERLAP", "100"))
        except (TypeError, ValueError):
            env_chunk_overlap = 100

    chunk_tokens = int(cfg.get("chunk_tokens") or env_chunk_size)
    overlap_tokens = int(cfg.get("overlap_tokens") or env_chunk_overlap)

    # Safety clip — chunker raises if overlap >= chunk_tokens. Clip down
    # rather than raise so one bad row doesn't block a whole reingest.
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = max(0, chunk_tokens // 4)

    return chunk_tokens, overlap_tokens


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
    "resolve_chunk_params",
    "with_overrides",
]
