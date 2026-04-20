"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Ingest-time hierarchical clustering. Each uploaded doc becomes a tree:

  level 0 = leaves = original chunks (verbatim)
  level 1 = GMM-clustered groups of leaves, each summarized by the chat model
  level 2 = GMM-clustered groups of level-1 summaries, summarized again
  ...
  root    = single top-level summary (when >1 nodes survive at max_levels)

All nodes (leaves + intermediates) are upserted into the same Qdrant
collection with a ``chunk_level`` payload field. Retrieval is unchanged:
the retriever/reranker/MMR see each node as a "chunk" — if a top hit is an
intermediate node, MMR sees its summary text (which is already dense with
the macro-topic signal that flat retrieval often misses).

This module makes NO retriever changes and NO vector-store schema changes
beyond a 2-field extension to the payload allowlist (``chunk_level``,
``source_chunk_ids``). Flag-off (default) path never imports this module.

Cost: +2-5× ingest time per doc. Win: much better recall on multi-chunk
answers in year-long KBs — policies, specs, contracts where the answer
lives across the whole doc, not in a single 800-token window.

Reference: Sarthi et al. 2024, "RAPTOR: Recursive Abstractive Processing
for Tree-Organized Retrieval" (https://arxiv.org/abs/2401.18059).
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import httpx


@dataclass
class RaptorNode:
    """One node in the RAPTOR tree.

    ``level``: 0 for leaves (= original chunks), 1+ for summary nodes.
    ``source_chunk_ids``: the ORIGINAL leaf chunk indices this node
    ultimately covers. For a leaf, ``[leaf.index]``. For a level-1 summary
    over cluster members, the union of those members' leaf indices. For a
    level-2 summary, the union of level-1 nodes' leaf coverage. The root
    covers every leaf in the doc.
    ``embedding``: vector for this node — leaves reuse their ingest-time
    embedding (caller supplies); summaries are embedded inside
    ``build_tree`` via the passed ``embedder`` so ingest can upsert each
    node directly without re-embedding the full tree.
    """

    text: str
    level: int
    parent_id: Optional[str]  # reserved for future use; not persisted today
    cluster_id: Optional[int]
    source_chunk_ids: list[int]
    embedding: list[float] | None = None


# Summary prompt. Phrased declaratively so the summary embeds as a *reference*,
# not as meta-commentary about "these excerpts" (which would retrieve poorly).
_SUMMARY_PROMPT = """Summarize the following document excerpts into 3-5 sentences. Capture the main themes, named entities, and specific facts. Write declaratively, like a reference document. Do NOT use phrases like 'the document says' or 'these excerpts discuss'.

EXCERPTS:
{excerpts}

Summary:"""


# Hard cap per excerpt so a runaway chunk doesn't blow the request size.
# 2000 chars ≈ 500 tokens; with typical cluster size of 5-10 members this
# keeps each summarize call under ~5000 tokens input.
_MAX_EXCERPT_CHARS = 2000


async def _summarize_cluster(
    texts: Sequence[str],
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout_s: float = 30.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Optional[str]:
    """Call the chat model to summarize a cluster of excerpts.

    Returns the stripped summary string, or None on any error or empty
    reply. Callers use None to drop a cluster (fail-open at the cluster
    level — one failed summary doesn't poison the whole tree).
    """
    if not texts:
        return None
    excerpts = "\n\n---\n\n".join((t or "")[:_MAX_EXCERPT_CHARS] for t in texts)
    body = {
        "model": chat_model,
        "messages": [
            {"role": "user", "content": _SUMMARY_PROMPT.format(excerpts=excerpts)}
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{chat_url.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        return text or None
    except Exception:  # noqa: BLE001 — fail-open by design
        return None


def _cluster_embeddings(
    embeddings: list[list[float]],
    *,
    max_clusters: int = 20,
) -> list[int]:
    """Cluster ``embeddings`` with GMM over the raw high-dim vectors.

    Returns a list of cluster ids, one per input embedding. The number of
    components is ``min(max_clusters, max(2, N//5))`` — roughly one cluster
    per 5 members, capped at 20. Below 3 inputs we just return all zeros
    (single cluster) since GMM can't fit anything useful.

    If scikit-learn isn't importable we also return all zeros. This makes
    the tree "degenerate" (one mega-cluster per level) rather than crashing
    ingest — callers treat that as a signal to stop recursing (cluster
    count of 1 breaks the loop in ``build_tree``).
    """
    n = len(embeddings)
    if n <= 2:
        return [0] * n
    try:
        from sklearn.mixture import GaussianMixture  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — sklearn missing → degenerate clustering
        return [0] * n

    k = min(max_clusters, max(2, n // 5))
    X = np.asarray(embeddings, dtype=np.float32)
    try:
        gmm = GaussianMixture(
            n_components=k,
            random_state=0,
            covariance_type="diag",
            reg_covar=1e-4,
        ).fit(X)
        return [int(c) for c in gmm.predict(X)]
    except Exception:  # noqa: BLE001 — GMM can fail on degenerate inputs
        return [0] * n


async def build_tree(
    leaves: Sequence[dict],
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    embedder,
    max_levels: int = 3,
    cluster_min: int = 5,
    concurrency: int = 4,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> list[RaptorNode]:
    """Build a RAPTOR tree from leaf chunks.

    ``leaves``: sequence of dicts, each ``{"text": str, "index": int,
    "embedding": list[float]}``. The caller has already embedded them
    (typically via TEI on the first ingest pass).

    Returns a flat list of all nodes: every leaf (level=0) plus every
    intermediate summary (level=1, 2, ...). Nodes preserve enough
    provenance (``source_chunk_ids``) to trace summaries back to the
    leaves they cover. The list is ordered: all leaves first (in input
    order), then level-1 nodes, then level-2, ...

    Construction:
      1. Emit leaves at level 0.
      2. Cluster current-level embeddings via GMM.
      3. If only one cluster, stop (can't decompose further).
      4. For each cluster, ask the chat model for a summary. Failed
         clusters (None) are dropped (fail-open per-cluster).
      5. Embed the new summaries in ONE batch call to save time.
      6. Recurse up to ``max_levels`` or until the current level has
         fewer than ``cluster_min`` members.
      7. If after the loop we still have >1 top-level nodes, emit a
         single root summary over all of them.

    ``concurrency``: max simultaneous chat calls per level. Default 4
    matches the chat endpoint's typical batch capacity without choking
    other concurrent ingests.
    """
    nodes: list[RaptorNode] = []

    # --- Level 0: leaves ----------------------------------------------------
    current: list[tuple[list[float], int, str]] = []  # (embedding, leaf_index, text)
    for leaf in leaves:
        idx = int(leaf["index"])
        emb = leaf["embedding"]
        nodes.append(
            RaptorNode(
                text=leaf["text"],
                level=0,
                parent_id=None,
                cluster_id=None,
                source_chunk_ids=[idx],
                embedding=list(emb),
            )
        )
        current.append((emb, idx, leaf["text"]))

    # --- Levels 1..max_levels: cluster, summarize, embed --------------------
    # If the initial leaf count is already below cluster_min, we never
    # cluster at all and return a flat list of leaves. The root-summary
    # emitter below checks ``clustered`` so it only fires when at least
    # one level of tree-building actually ran — avoiding a "degenerate
    # tree" (single root over raw leaves) that adds a summary without
    # the leaf-grouping structure RAPTOR actually needs.
    level = 1
    clustered = False
    sem = asyncio.Semaphore(max(1, concurrency))
    # Track coverage: for summary nodes built from a mix of leaves and
    # prior summaries, we expand back to the original leaf indices so
    # ``source_chunk_ids`` always points at level-0 provenance.
    coverage_by_node_id: dict[int, list[int]] = {
        id(nodes[i]): [int(leaves[i]["index"])] for i in range(len(leaves))
    }
    current_node_refs: list[RaptorNode] = list(nodes)  # parallel to ``current``

    while len(current) >= cluster_min and level <= max_levels:
        vecs = [c[0] for c in current]
        cluster_ids = _cluster_embeddings(vecs)

        # Group members by cluster id. Members are (position-in-current, ref)
        # pairs so we can fetch the underlying node for coverage lookup.
        groups: dict[int, list[tuple[int, RaptorNode, str]]] = defaultdict(list)
        for pos, ((_vec, _idx, text), cid, ref) in enumerate(
            zip(current, cluster_ids, current_node_refs)
        ):
            groups[cid].append((pos, ref, text))

        # If GMM collapsed everything into one cluster (or fewer than 2 distinct
        # groups), we can't make progress at this level — stop the recursion.
        if len(groups) <= 1:
            break

        async def _summarize_one(
            cid: int, members: list[tuple[int, RaptorNode, str]]
        ) -> Optional[RaptorNode]:
            async with sem:
                summary = await _summarize_cluster(
                    [m[2] for m in members],
                    chat_url=chat_url,
                    chat_model=chat_model,
                    api_key=api_key,
                    transport=transport,
                )
            if not summary:
                return None
            # Union of covered leaves across members.
            coverage: list[int] = []
            seen: set[int] = set()
            for _pos, ref, _t in members:
                for leaf_idx in coverage_by_node_id.get(id(ref), []):
                    if leaf_idx not in seen:
                        seen.add(leaf_idx)
                        coverage.append(leaf_idx)
            return RaptorNode(
                text=summary,
                level=level,
                parent_id=None,
                cluster_id=int(cid),
                source_chunk_ids=coverage,
            )

        tasks = [
            _summarize_one(cid, members) for cid, members in groups.items()
        ]
        new_nodes: list[RaptorNode] = [
            n for n in await asyncio.gather(*tasks) if n is not None
        ]
        if not new_nodes:
            # All cluster summaries failed — fall open, stop recursing.
            break

        # Embed all summaries in ONE batched TEI call (cheap vs chat calls
        # and leaves/level-1 embeds shouldn't multiply).
        summary_texts = [n.text for n in new_nodes]
        try:
            summary_vecs = await embedder.embed(summary_texts)
        except Exception:
            # Can't embed → can't recurse further. Still emit the summaries
            # so retrieval can at least match them on text; next level won't run.
            # Note: these nodes will have embedding=None; the caller drops
            # any node without an embedding (leaves/tests/ingest all expect vectors).
            nodes.extend(new_nodes)
            break

        # Attach embeddings to summary nodes so the caller can upsert directly.
        for n, v in zip(new_nodes, summary_vecs):
            n.embedding = list(v)
            coverage_by_node_id[id(n)] = list(n.source_chunk_ids)
        nodes.extend(new_nodes)
        clustered = True

        current = [(summary_vecs[i], -1, new_nodes[i].text) for i in range(len(new_nodes))]
        current_node_refs = list(new_nodes)
        level += 1

    # --- Root: single top summary if >1 nodes survived AFTER clustering -----
    # ``clustered`` guards against the degenerate case where the doc is
    # already below ``cluster_min`` — in that case we emit only the leaves
    # (flat chunking + no summaries) rather than a trivial root over a
    # handful of chunks that adds noise without the RAPTOR grouping.
    if clustered and len(current) > 1:
        root_summary = await _summarize_cluster(
            [c[2] for c in current],
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            transport=transport,
        )
        if root_summary:
            # Coverage = union of every current node's coverage = every leaf.
            coverage: list[int] = []
            seen: set[int] = set()
            for ref in current_node_refs:
                for leaf_idx in coverage_by_node_id.get(id(ref), []):
                    if leaf_idx not in seen:
                        seen.add(leaf_idx)
                        coverage.append(leaf_idx)
            # Embed the root summary so the caller has a complete node.
            try:
                root_vecs = await embedder.embed([root_summary])
                root_emb = list(root_vecs[0]) if root_vecs else None
            except Exception:
                root_emb = None
            nodes.append(
                RaptorNode(
                    text=root_summary,
                    level=level,
                    parent_id=None,
                    cluster_id=0,
                    source_chunk_ids=coverage,
                    embedding=root_emb,
                )
            )

    return nodes


def is_enabled() -> bool:
    """Whether RAPTOR tree-building is enabled for this process/request.

    Reads ``RAG_RAPTOR`` via the flags overlay (so per-KB overrides from
    the chat bridge take effect). Default OFF → byte-identical to the
    pre-RAPTOR ingest flow.
    """
    from ext.services import flags

    return flags.get("RAG_RAPTOR", "0") == "1"


__all__ = [
    "RaptorNode",
    "build_tree",
    "is_enabled",
]
