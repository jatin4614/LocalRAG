#!/usr/bin/env python3
"""Re-embed chunks in-place against a target pipeline version.

Phase 4 deliverable — operational tool for future embedder swaps
(e.g. Phase 1b when Harrier lands). NOT used by the Phase 1a cycle.

What it does:
  1. Scrolls every point in the target KB collection (or all KB
     collections with ``--all``).
  2. For each chunk, reads ``payload.text`` and re-runs it through the
     configured TEI embedder.
  3. Upserts the point back with the new dense vector, preserving
     sparse vectors and payload. Stamps ``payload.pipeline_version`` to
     the CLI-provided ``--pipeline-version``.

What it does NOT do:
  * Re-extract the source document. The original bytes are NOT read.
  * Re-chunk. Chunk boundaries / text are untouched.
  * Re-compute sparse (BM25) vectors — they already reflect the chunk
    text and are stable across embedder swaps.
  * Migrate vector dimensionality. The target collection must have the
    correct size already (handle that by creating a fresh v2 collection
    per ``scripts/reindex_hybrid.py`` and pointing this at it).

Usage:
  # Dry-run against kb_1 — counts points, prints plan, no writes.
  python scripts/reembed_all.py --kb 1 --pipeline-version \\
      'chunker=v3|extractor=v2|embedder=harrier-0.6b|ctx=none'

  # Apply.
  python scripts/reembed_all.py --kb 1 --apply --pipeline-version \\
      'chunker=v3|extractor=v2|embedder=harrier-0.6b|ctx=none'

  # Apply across every KB collection.
  python scripts/reembed_all.py --all --apply --pipeline-version \\
      'chunker=v3|extractor=v2|embedder=harrier-0.6b|ctx=none'

Exit codes:
    0  success (or dry-run finished)
    1  any error — source missing, embedder failure, Qdrant error
    4  invalid arguments

Resumability (review §9.9):
  Pass ``--checkpoint <path>`` (default ``./.reembed_checkpoint.json``)
  to enable atomic per-batch checkpoint writes. On every successful
  batch upsert the script rewrites the checkpoint via tmpfile + rename
  so a kill-9 mid-batch leaves a coherent file (either pre-batch or
  post-batch state, never half-written).

  On a clean exit the file is deleted. On a crash it remains, so the
  next invocation with ``--resume`` can skip ahead. Without ``--resume``
  the script ignores any pre-existing checkpoint (forces a fresh run).

  Skip semantics: points are skipped when their ``payload.doc_id <
  last_doc_id`` OR (``doc_id == last_doc_id`` AND ``chunk_index <=
  last_chunk_index``). Doc-summary points (``chunk_index == None``) are
  always re-processed when their doc_id is reached, then advanced past.
  This mirrors the natural ordering Qdrant returns from scroll within
  a single collection.

  Crash → ``python scripts/reembed_all.py --kb 1 --apply --resume \\``
  ``    --pipeline-version '...'``  (no scratch restart).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Make ``ext.*`` importable when invoked from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_DENSE_NAME = "dense"
_SPARSE_NAME = "bm25"
_DEFAULT_CHECKPOINT_PATH = "./.reembed_checkpoint.json"


def _load_checkpoint(path: Path) -> Optional[dict[str, Any]]:
    """Read a checkpoint dict from ``path`` if it exists, else return None.

    A malformed file is treated as missing — the operator should delete
    it explicitly rather than silently overwrite, so we surface the
    parse error and bail.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"checkpoint at {path} is unreadable / malformed: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"checkpoint at {path} did not parse to a dict (got {type(data).__name__})")
    return data


def _write_checkpoint(
    path: Path,
    *,
    last_doc_id: Any,
    last_chunk_index: Any,
    started_at: str,
    model_version: str,
) -> None:
    """Atomically rewrite the checkpoint via tmpfile + rename.

    The ``os.replace`` call is atomic on POSIX (and on Windows since
    Python 3.3). A SIGKILL between the ``write`` and the ``replace``
    leaves the previous (or absent) file intact, so the next ``--resume``
    sees a coherent prior state. We deliberately fsync the tmp file
    before rename — without that, an OS crash between write+rename and
    the on-disk flush could lose the checkpoint update.
    """
    payload = {
        "last_doc_id": last_doc_id,
        "last_chunk_index": last_chunk_index,
        "started_at": started_at,
        "model_version": model_version,
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    parent = path.parent if str(path.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".reembed_ckpt_", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        # Best-effort tmp cleanup; the original checkpoint is unchanged.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _delete_checkpoint(path: Path) -> None:
    """Remove the checkpoint on clean exit. Missing-file is not an error."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Don't fail the operator over a stale FS state — log to stderr.
        print(f"warning: could not delete checkpoint {path}", file=sys.stderr)


def _point_doc_chunk(point: Any) -> tuple[Optional[Any], Optional[Any]]:
    """Return ``(doc_id, chunk_index)`` from a scrolled point's payload.

    Either may be None for points that pre-date the campaign-§ ingest
    fields. The skip predicate handles None safely (see _should_skip).
    """
    payload = getattr(point, "payload", None) or {}
    return payload.get("doc_id"), payload.get("chunk_index")


def _should_skip(
    *,
    doc_id: Any,
    chunk_index: Any,
    last_doc_id: Any,
    last_chunk_index: Any,
) -> bool:
    """True if this point was already processed in a previous run.

    Predicate: ``doc_id < last_doc_id`` OR
              (``doc_id == last_doc_id`` AND ``chunk_index <= last_chunk_index``).

    Mismatched / missing types short-circuit to False — when we can't
    tell, we re-process (idempotent upsert means worst case is wasted
    work, not corruption).
    """
    if last_doc_id is None or doc_id is None:
        return False
    try:
        if doc_id < last_doc_id:
            return True
        if doc_id == last_doc_id:
            if last_chunk_index is None or chunk_index is None:
                return False
            return chunk_index <= last_chunk_index
    except TypeError:
        # e.g. comparing int/str — treat as different docs, re-process.
        return False
    return False


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-embed existing Qdrant chunks against a new embedder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant REST URL (default: $QDRANT_URL or http://localhost:6333).",
    )
    p.add_argument(
        "--tei-url",
        default=os.environ.get("TEI_URL", "http://localhost:8080"),
        help="TEI base URL (default: $TEI_URL or http://localhost:8080).",
    )

    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--kb",
        type=int,
        help="Single KB id — rebuilds collection kb_<id>.",
    )
    which.add_argument(
        "--all",
        action="store_true",
        help="Every collection whose name starts with kb_.",
    )

    p.add_argument(
        "--pipeline-version",
        required=True,
        help=(
            "Target pipeline version string stamped on re-embedded points "
            "(e.g. 'chunker=v3|extractor=v2|embedder=harrier-0.6b|ctx=none')."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Points scrolled / re-embedded / upserted per round (default: 64).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for Qdrant/TEI calls (default: 60s).",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the plan, make no writes (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Actually re-embed and upsert. Opt-in.",
    )

    # Review §9.9 — checkpointing for crash-safe resumption on 100k+
    # chunk runs. Default path keeps the file alongside the cwd so a
    # follow-up ``--resume`` can find it without remembering the path.
    p.add_argument(
        "--checkpoint",
        default=_DEFAULT_CHECKPOINT_PATH,
        help=(
            "Path to JSON checkpoint file (default: "
            f"{_DEFAULT_CHECKPOINT_PATH}). Rewritten atomically after every "
            "batch upsert; deleted on clean exit; preserved on crash so "
            "``--resume`` can pick up from the last successful batch."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume from the existing checkpoint at ``--checkpoint``. "
            "Without this flag, an existing checkpoint is left untouched "
            "and the run starts from scratch (forces an explicit "
            "decision — accidental restarts shouldn't double-spend the "
            "embedder budget on already-done points)."
        ),
    )

    return p.parse_args(argv)


def _extract_vector_kind(point: Any) -> tuple[Optional[list[float]], Optional[Any]]:
    """Return (dense, sparse_raw) from a scrolled point.

    Hybrid collections store vectors as ``{dense: list, bm25: SparseVector}``.
    Legacy collections store just a list. Sparse is returned as the raw
    SparseVector-ish object (qdrant-client passes them through);
    up-serting accepts the same type so we don't round-trip.
    """
    v = getattr(point, "vector", None)
    if v is None:
        return (None, None)
    if isinstance(v, dict):
        dense = v.get(_DENSE_NAME)
        sparse = v.get(_SPARSE_NAME)
        if isinstance(dense, list):
            return ([float(x) for x in dense], sparse)
        # Fall back to first list value.
        for val in v.values():
            if isinstance(val, list):
                return ([float(x) for x in val], sparse)
        return (None, sparse)
    if isinstance(v, list):
        return ([float(x) for x in v], None)
    return (None, None)


async def _enumerate_collections(qdrant: Any, *, kb: Optional[int], all_kbs: bool) -> list[str]:
    if kb is not None:
        return [f"kb_{kb}"]
    if all_kbs:
        cols = (await qdrant.get_collections()).collections
        return sorted(c.name for c in cols if c.name.startswith("kb_"))
    return []


def _sparse_to_tuple(sparse: Any) -> Optional[tuple[list[int], list[float]]]:
    """Coerce a Qdrant SparseVector-ish object back to ``(indices, values)``.

    VectorStore.upsert expects the tuple form (indices, values); points
    scrolled from Qdrant come back as ``models.SparseVector`` objects
    with ``.indices`` and ``.values`` attributes (or as a dict in some
    qdrant-client versions). Returns None for absent / unrecognized.
    """
    if sparse is None:
        return None
    indices = getattr(sparse, "indices", None)
    values = getattr(sparse, "values", None)
    if indices is None or values is None:
        # Older / mocked shapes may be plain dicts.
        if isinstance(sparse, dict):
            indices = sparse.get("indices")
            values = sparse.get("values")
    if indices is None or values is None:
        return None
    return ([int(i) for i in indices], [float(v) for v in values])


async def _reembed_collection(
    collection: str,
    *,
    qdrant: Any,
    embedder: Any,
    vector_store: Any,
    pipeline_version: str,
    batch_size: int,
    apply: bool,
    checkpoint_path: Optional[Path] = None,
    started_at: Optional[str] = None,
    resume_state: Optional[dict[str, Any]] = None,
) -> tuple[int, int]:
    """Return (total_points, re_embedded). Re-embedded may be 0 in dry-run.

    Bug-fix campaign §3.3: writes route through ``VectorStore.upsert``
    (instead of a raw qdrant client upsert from the original script) so
    custom-sharded targets like ``kb_1_v4`` (live since 2026-04-26)
    auto-derive ``shard_key_selector`` from each point's
    ``payload['shard_key']``. Without that, a raw upsert returns
    ``Shard key not specified`` 400 and the operator script blows up
    after the first scrolled page.

    Review §9.9: when ``checkpoint_path`` is provided AND ``apply`` is
    True, the checkpoint is rewritten atomically after each successful
    batch with the (doc_id, chunk_index) of the last point in the batch.
    ``resume_state`` (loaded from disk) is the predicate input for
    skipping already-done points.
    """
    total = 0
    re_embedded = 0
    offset = None
    print(f"  [{'APPLY' if apply else 'DRY-RUN'}] scrolling {collection} …")
    last_doc_id = (resume_state or {}).get("last_doc_id")
    last_chunk_index = (resume_state or {}).get("last_chunk_index")
    while True:
        page, offset = await qdrant.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not page:
            break

        texts: list[str] = []
        slot_for_idx: list[int] = []
        new_points: list[Any] = []
        skipped_in_batch = 0
        last_seen_doc_id = last_doc_id
        last_seen_chunk_index = last_chunk_index

        for i, pt in enumerate(page):
            doc_id, chunk_index = _point_doc_chunk(pt)
            if _should_skip(
                doc_id=doc_id,
                chunk_index=chunk_index,
                last_doc_id=last_doc_id,
                last_chunk_index=last_chunk_index,
            ):
                # Already covered by a prior run. Don't tally as
                # processed; we want operator-visible counts to reflect
                # actual work done in *this* invocation.
                skipped_in_batch += 1
                continue

            dense, sparse = _extract_vector_kind(pt)
            payload = dict(pt.payload or {})
            text = payload.get("text")
            if not text or dense is None:
                # Nothing to re-embed with or no current dense to replace.
                total += 1
                # Advance the high-water mark even for skipped-because-
                # no-vector points so we don't re-visit them on resume.
                last_seen_doc_id = doc_id if doc_id is not None else last_seen_doc_id
                last_seen_chunk_index = chunk_index if chunk_index is not None else last_seen_chunk_index
                continue
            texts.append(str(text))
            slot_for_idx.append(i)
            new_points.append({
                "id": pt.id,
                "payload": payload,
                "sparse": sparse,
                "_dense_old": dense,
                "_doc_id": doc_id,
                "_chunk_index": chunk_index,
            })
            total += 1

        if skipped_in_batch:
            print(f"    (resume) skipped {skipped_in_batch} already-processed point(s)")

        if not texts:
            if offset is None:
                break
            continue

        if apply:
            new_vecs = await embedder.embed(texts)
            if len(new_vecs) != len(new_points):
                raise RuntimeError(
                    f"embedder returned {len(new_vecs)} vectors for {len(new_points)} texts"
                )

            # VectorStore.upsert takes a list of dicts shaped:
            #   {id, vector (dense list), payload, sparse_vector?, colbert_vector?}
            # It handles named-vector encoding and shard_key derivation
            # internally, so we just hand it the data and let it pick the
            # right path. Custom-sharded collections require each point's
            # payload to carry ``shard_key`` — already preserved here
            # because we copy the original payload through unchanged
            # (only stamping ``pipeline_version`` on top).
            upserts: list[dict] = []
            for np_dict, new_vec in zip(new_points, new_vecs):
                updated_payload = dict(np_dict["payload"])
                updated_payload["pipeline_version"] = pipeline_version
                pt_dict: dict[str, Any] = {
                    "id": np_dict["id"],
                    "vector": list(new_vec),
                    "payload": updated_payload,
                }
                sparse_tup = _sparse_to_tuple(np_dict["sparse"])
                if sparse_tup is not None:
                    pt_dict["sparse_vector"] = sparse_tup
                upserts.append(pt_dict)

            await vector_store.upsert(collection, upserts)
            re_embedded += len(upserts)

            # Advance the checkpoint to the last point in this batch.
            # We pick the LAST upserted point because Qdrant returns
            # scroll pages in a stable internal order — recording the
            # last entry guarantees that on resume we skip the entire
            # successful batch, never half of it.
            tail = new_points[-1]
            last_seen_doc_id = tail["_doc_id"] if tail["_doc_id"] is not None else last_seen_doc_id
            last_seen_chunk_index = tail["_chunk_index"] if tail["_chunk_index"] is not None else last_seen_chunk_index

            if checkpoint_path is not None and started_at is not None:
                _write_checkpoint(
                    checkpoint_path,
                    last_doc_id=last_seen_doc_id,
                    last_chunk_index=last_seen_chunk_index,
                    started_at=started_at,
                    model_version=pipeline_version,
                )
                # Promote the in-memory high-water marks so the next
                # batch's _should_skip() predicate is up-to-date.
                last_doc_id = last_seen_doc_id
                last_chunk_index = last_seen_chunk_index
        else:
            re_embedded += len(texts)

        if offset is None:
            break

    return (total, re_embedded)


async def _main(args: argparse.Namespace) -> int:
    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError as e:
        print(f"error: qdrant-client not installed: {e}", file=sys.stderr)
        return 1

    try:
        from ext.services.embedder import TEIEmbedder
        # Bug-fix campaign §3.3 — route writes through VectorStore so
        # custom-sharded collections (kb_1_v4 onward) auto-derive
        # shard_key_selector from each point's payload. The raw qdrant
        # client upsert path bypassed that and 400'd on first apply.
        from ext.services.vector_store import VectorStore
    except ImportError as e:
        print(f"error: cannot import dependencies: {e}", file=sys.stderr)
        return 1

    apply = bool(args.apply)
    banner = "APPLY" if apply else "DRY-RUN"
    print(f"[{banner}] reembed target pipeline_version={args.pipeline_version!r}")

    # Review §9.9 — checkpoint setup. Only meaningful in --apply mode
    # (dry-run does no writes, so there's nothing to resume). We still
    # accept --checkpoint in dry-run for ergonomic parity, but skip
    # both load+write to keep the file untouched.
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    resume_state: Optional[dict[str, Any]] = None
    if apply and checkpoint_path is not None:
        existing = _load_checkpoint(checkpoint_path)
        if existing is not None:
            if args.resume:
                resume_state = existing
                started_at = str(existing.get("started_at") or started_at)
                print(
                    f"  --resume: continuing from checkpoint {checkpoint_path} "
                    f"(last_doc_id={existing.get('last_doc_id')}, "
                    f"last_chunk_index={existing.get('last_chunk_index')}, "
                    f"started_at={started_at})"
                )
                # Stamp guard: catching an operator who pointed --resume at
                # a checkpoint that was written for a different pipeline.
                old_model = existing.get("model_version")
                if old_model and old_model != args.pipeline_version:
                    print(
                        f"warning: checkpoint model_version={old_model!r} differs from "
                        f"--pipeline-version={args.pipeline_version!r}; resuming anyway",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"warning: checkpoint exists at {checkpoint_path} but --resume "
                    f"was not passed; the file will NOT be touched and the run "
                    f"starts from scratch. Pass --resume to continue, or delete "
                    f"the file to suppress this warning.",
                    file=sys.stderr,
                )

    qdrant = AsyncQdrantClient(url=args.qdrant_url, timeout=args.timeout)
    embedder = TEIEmbedder(base_url=args.tei_url, timeout=args.timeout)
    # vector_size on the wrapper isn't consulted on upsert (the dense
    # vector dimensionality follows the embedder output), so 0 is fine
    # — but pick the real bge-m3 size to keep diagnostic logs sane.
    vector_store = VectorStore(url=args.qdrant_url, vector_size=1024)
    try:
        collections = await _enumerate_collections(
            qdrant, kb=args.kb, all_kbs=args.all,
        )
        if not collections:
            print("no collections matched (pass --kb or --all)", file=sys.stderr)
            return 1
        print(f"  target collections: {collections}")

        total_points = 0
        total_reembedded = 0
        for col in collections:
            try:
                t, r = await _reembed_collection(
                    col,
                    qdrant=qdrant,
                    embedder=embedder,
                    vector_store=vector_store,
                    pipeline_version=args.pipeline_version,
                    batch_size=args.batch_size,
                    apply=apply,
                    checkpoint_path=checkpoint_path if apply else None,
                    started_at=started_at,
                    resume_state=resume_state,
                )
                total_points += t
                total_reembedded += r
                print(
                    f"  {col}: {t} points scanned, "
                    f"{r} {'re-embedded' if apply else 'would be re-embedded'}"
                )
            except Exception as e:
                # Leave the checkpoint in place on crash so --resume picks
                # up where we left off. The operator gets a non-zero exit
                # and a clear pointer to the file.
                if checkpoint_path is not None and apply:
                    print(
                        f"  checkpoint preserved at {checkpoint_path} — "
                        f"re-run with --resume to continue.",
                        file=sys.stderr,
                    )
                print(f"error: collection {col!r} failed: {e}", file=sys.stderr)
                return 1

        # Clean exit: drop the checkpoint so the next operator session
        # doesn't get a misleading "resume?" prompt for a finished run.
        if apply and checkpoint_path is not None:
            _delete_checkpoint(checkpoint_path)

        print(
            f"\n{banner}: total points={total_points}, "
            f"re-embedded={total_reembedded}"
        )
        return 0
    finally:
        try:
            await embedder.aclose()
        except Exception:
            pass
        try:
            await vector_store.close()
        except Exception:
            pass
        try:
            await qdrant.close()
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return asyncio.run(_main(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
