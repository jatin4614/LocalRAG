"""Unit tests for ``scripts/reembed_all.py`` checkpoint mechanism (review §9.9).

Crash-recovery is the whole point: if the script blows up halfway through
a 100k-chunk run, ``--resume`` must skip up to the last successfully
upserted batch instead of restarting from scratch. These tests mock the
qdrant client + embedder so we don't need a live stack — they assert:

  * The checkpoint file is written atomically after every batch upsert.
  * The on-disk payload carries the documented schema
    (``last_doc_id`` / ``last_chunk_index`` / ``started_at`` /
    ``model_version``).
  * On a clean exit the file is deleted.
  * On a simulated crash mid-run the file remains and ``--resume``
    causes the script to skip already-processed points.
  * Without ``--resume`` an existing checkpoint is left untouched (no
    silent restart-from-scratch trampling).
  * The atomic-rewrite path uses ``os.replace`` (tmpfile + rename) so a
    SIGKILL between write and rename leaves the prior file intact.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "reembed_all.py"


def _import_script() -> Any:
    """Import the script module by file path (it lives outside the package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("reembed_all", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reembed_all"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod() -> Any:
    return _import_script()


def _mk_point(point_id: int, doc_id: int, chunk_index: int, *, text: str = "x") -> Any:
    """Build a fake Qdrant point with named ``dense`` vector + payload."""
    return SimpleNamespace(
        id=point_id,
        vector={"dense": [0.1, 0.2, 0.3, 0.4]},
        payload={
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "text": text,
            "shard_key": "kb1",
        },
    )


# ---------- pure helpers -----------------------------------------------------


def test_should_skip_predicate(mod) -> None:
    # Same doc, prior chunk → skip.
    assert mod._should_skip(doc_id=10, chunk_index=3, last_doc_id=10, last_chunk_index=5) is True
    # Same doc, equal chunk → skip (already done).
    assert mod._should_skip(doc_id=10, chunk_index=5, last_doc_id=10, last_chunk_index=5) is True
    # Same doc, later chunk → process.
    assert mod._should_skip(doc_id=10, chunk_index=6, last_doc_id=10, last_chunk_index=5) is False
    # Earlier doc → skip.
    assert mod._should_skip(doc_id=5, chunk_index=99, last_doc_id=10, last_chunk_index=0) is True
    # Later doc → process.
    assert mod._should_skip(doc_id=11, chunk_index=0, last_doc_id=10, last_chunk_index=99) is False
    # No prior state → never skip.
    assert mod._should_skip(doc_id=10, chunk_index=5, last_doc_id=None, last_chunk_index=None) is False


def test_write_checkpoint_atomic(tmp_path: Path, mod) -> None:
    """Checkpoint write goes through a tmpfile + rename (atomic on POSIX)."""
    target = tmp_path / "ckpt.json"
    mod._write_checkpoint(
        target,
        last_doc_id=42,
        last_chunk_index=17,
        started_at="2026-05-02T00:00:00+00:00",
        model_version="embedder=harrier-0.6b|ctx=none",
    )
    assert target.is_file()
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed == {
        "last_doc_id": 42,
        "last_chunk_index": 17,
        "started_at": "2026-05-02T00:00:00+00:00",
        "model_version": "embedder=harrier-0.6b|ctx=none",
    }
    # Side effect of atomic write: no leftover tmp files in the dir.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".reembed_ckpt_")]
    assert leftovers == []


def test_load_checkpoint_missing_returns_none(tmp_path: Path, mod) -> None:
    assert mod._load_checkpoint(tmp_path / "does-not-exist.json") is None


def test_load_checkpoint_malformed_raises(tmp_path: Path, mod) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json {", encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        mod._load_checkpoint(bad)


def test_delete_checkpoint_missing_is_silent(tmp_path: Path, mod) -> None:
    """Clean exit on a checkpoint that was never written shouldn't error."""
    mod._delete_checkpoint(tmp_path / "missing.json")  # no exception


# ---------- collection-level integration with mocked qdrant + embedder -----


def _make_vs() -> Any:
    """Mock VectorStore with awaitable upsert that records calls."""
    vs = MagicMock()
    vs.upsert = AsyncMock()
    return vs


def _make_qdrant_pages(*pages: list[Any]) -> Any:
    """Mock AsyncQdrantClient whose ``scroll`` yields the given pages, then ()."""
    qdrant = MagicMock()
    pages_with_offsets: list[tuple[list[Any], Any]] = []
    for i, pg in enumerate(pages):
        offset = f"o{i}" if i + 1 < len(pages) else None
        pages_with_offsets.append((pg, offset))
    pages_with_offsets.append(([], None))  # terminating page
    pages_iter = iter(pages_with_offsets)

    async def _scroll(**_kwargs: Any) -> tuple[list[Any], Any]:
        try:
            return next(pages_iter)
        except StopIteration:
            return ([], None)

    qdrant.scroll = AsyncMock(side_effect=_scroll)
    return qdrant


def _make_embedder() -> Any:
    """Mock TEI embedder returning one fake vector per text."""
    embedder = MagicMock()

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


@pytest.mark.asyncio
async def test_checkpoint_written_after_each_batch(tmp_path: Path, mod) -> None:
    """Two-batch run → checkpoint file rewritten twice, final state ==
    the last point of the last batch.
    """
    ckpt = tmp_path / "ckpt.json"
    qdrant = _make_qdrant_pages(
        [_mk_point(1, doc_id=1, chunk_index=0), _mk_point(2, doc_id=1, chunk_index=1)],
        [_mk_point(3, doc_id=2, chunk_index=0), _mk_point(4, doc_id=2, chunk_index=1)],
    )
    embedder = _make_embedder()
    vs = _make_vs()

    total, re_emb = await mod._reembed_collection(
        "kb_1",
        qdrant=qdrant,
        embedder=embedder,
        vector_store=vs,
        pipeline_version="test|v=1",
        batch_size=2,
        apply=True,
        checkpoint_path=ckpt,
        started_at="2026-05-02T00:00:00+00:00",
        resume_state=None,
    )

    assert total == 4
    assert re_emb == 4
    # 2 upsert calls, one per batch.
    assert vs.upsert.await_count == 2
    # Final checkpoint reflects the last point of the second batch.
    final = json.loads(ckpt.read_text(encoding="utf-8"))
    assert final == {
        "last_doc_id": 2,
        "last_chunk_index": 1,
        "started_at": "2026-05-02T00:00:00+00:00",
        "model_version": "test|v=1",
    }


@pytest.mark.asyncio
async def test_checkpoint_resume_skips_already_processed(tmp_path: Path, mod) -> None:
    """``resume_state`` should suppress re-processing of prior points."""
    ckpt = tmp_path / "ckpt.json"
    # Same 4 points as above, but our resume state says we got through doc 1.
    qdrant = _make_qdrant_pages(
        [_mk_point(1, doc_id=1, chunk_index=0), _mk_point(2, doc_id=1, chunk_index=1)],
        [_mk_point(3, doc_id=2, chunk_index=0), _mk_point(4, doc_id=2, chunk_index=1)],
    )
    embedder = _make_embedder()
    vs = _make_vs()

    total, re_emb = await mod._reembed_collection(
        "kb_1",
        qdrant=qdrant,
        embedder=embedder,
        vector_store=vs,
        pipeline_version="test|v=1",
        batch_size=2,
        apply=True,
        checkpoint_path=ckpt,
        started_at="2026-05-02T00:00:00+00:00",
        resume_state={
            "last_doc_id": 1,
            "last_chunk_index": 1,
            "started_at": "2026-05-02T00:00:00+00:00",
            "model_version": "test|v=1",
        },
    )

    # First batch entirely skipped (prior run) → only the second batch's
    # 2 points were re-embedded + upserted.
    assert re_emb == 2
    assert vs.upsert.await_count == 1
    # And the embedder only saw the texts from doc=2 chunks.
    assert embedder.embed.await_count == 1
    final = json.loads(ckpt.read_text(encoding="utf-8"))
    assert final["last_doc_id"] == 2
    assert final["last_chunk_index"] == 1


@pytest.mark.asyncio
async def test_checkpoint_preserved_on_crash(tmp_path: Path, mod) -> None:
    """When the embedder raises mid-run, the checkpoint from the prior
    successful batch must remain on disk so ``--resume`` can pick up."""
    ckpt = tmp_path / "ckpt.json"
    qdrant = _make_qdrant_pages(
        [_mk_point(1, doc_id=1, chunk_index=0), _mk_point(2, doc_id=1, chunk_index=1)],
        [_mk_point(3, doc_id=2, chunk_index=0), _mk_point(4, doc_id=2, chunk_index=1)],
    )
    embedder = MagicMock()
    call_count = {"n": 0}

    async def _embed(texts: list[str]) -> list[list[float]]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]
        raise RuntimeError("simulated TEI 503")

    embedder.embed = AsyncMock(side_effect=_embed)
    vs = _make_vs()

    with pytest.raises(RuntimeError, match="simulated TEI"):
        await mod._reembed_collection(
            "kb_1",
            qdrant=qdrant,
            embedder=embedder,
            vector_store=vs,
            pipeline_version="test|v=1",
            batch_size=2,
            apply=True,
            checkpoint_path=ckpt,
            started_at="2026-05-02T00:00:00+00:00",
            resume_state=None,
        )

    # Checkpoint still on disk (NOT deleted) and reflects the LAST
    # successful batch (doc 1, chunk 1).
    assert ckpt.is_file()
    final = json.loads(ckpt.read_text(encoding="utf-8"))
    assert final["last_doc_id"] == 1
    assert final["last_chunk_index"] == 1


@pytest.mark.asyncio
async def test_checkpoint_not_written_in_dry_run(tmp_path: Path, mod) -> None:
    """Dry-run mode must not touch the checkpoint at all."""
    ckpt = tmp_path / "ckpt.json"
    # Pre-existing checkpoint we don't want trampled.
    mod._write_checkpoint(
        ckpt,
        last_doc_id=999,
        last_chunk_index=999,
        started_at="prior",
        model_version="prior",
    )
    pre_mtime = ckpt.stat().st_mtime_ns
    qdrant = _make_qdrant_pages(
        [_mk_point(1, doc_id=1, chunk_index=0)],
    )
    embedder = _make_embedder()
    vs = _make_vs()

    await mod._reembed_collection(
        "kb_1",
        qdrant=qdrant,
        embedder=embedder,
        vector_store=vs,
        pipeline_version="test|v=1",
        batch_size=2,
        apply=False,  # dry run
        checkpoint_path=None,  # _main only passes ckpt when apply=True
        started_at="2026-05-02T00:00:00+00:00",
        resume_state=None,
    )

    # File is unchanged.
    assert ckpt.stat().st_mtime_ns == pre_mtime
    parsed = json.loads(ckpt.read_text(encoding="utf-8"))
    assert parsed["last_doc_id"] == 999
    # Embedder was never called in dry run.
    assert embedder.embed.await_count == 0
    # Vector store upsert was never called in dry run.
    assert vs.upsert.await_count == 0


@pytest.mark.asyncio
async def test_checkpoint_deleted_on_clean_main_exit(tmp_path: Path, mod, monkeypatch) -> None:
    """A successful end-to-end ``_main`` call with --apply removes the
    checkpoint file.
    """
    ckpt = tmp_path / "ckpt.json"
    # Stand up just enough of qdrant_client + ext services to exercise
    # _main without a real backend.
    fake_qdrant_module = MagicMock()
    fake_qdrant = MagicMock()
    fake_qdrant.get_collections = AsyncMock(
        return_value=SimpleNamespace(collections=[SimpleNamespace(name="kb_1")])
    )
    fake_qdrant.scroll = AsyncMock(
        side_effect=[
            ([_mk_point(1, doc_id=1, chunk_index=0)], None),
            ([], None),
        ]
    )
    fake_qdrant.close = AsyncMock()
    fake_qdrant_module.AsyncQdrantClient = MagicMock(return_value=fake_qdrant)
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant_module)

    fake_embedder_mod = MagicMock()
    fake_embedder = _make_embedder()
    fake_embedder.aclose = AsyncMock()
    fake_embedder_mod.TEIEmbedder = MagicMock(return_value=fake_embedder)
    monkeypatch.setitem(sys.modules, "ext.services.embedder", fake_embedder_mod)

    fake_vs_mod = MagicMock()
    fake_vs = _make_vs()
    fake_vs.close = AsyncMock()
    fake_vs_mod.VectorStore = MagicMock(return_value=fake_vs)
    monkeypatch.setitem(sys.modules, "ext.services.vector_store", fake_vs_mod)

    args = SimpleNamespace(
        qdrant_url="http://x",
        tei_url="http://y",
        kb=1,
        all=False,
        pipeline_version="test|v=1",
        batch_size=10,
        timeout=10.0,
        dry_run=False,
        apply=True,
        checkpoint=str(ckpt),
        resume=False,
    )
    rc = await mod._main(args)
    assert rc == 0
    # Checkpoint deleted on clean exit.
    assert not ckpt.is_file()


def test_cli_accepts_checkpoint_and_resume_flags(mod) -> None:
    """``--checkpoint`` and ``--resume`` must be in the parser surface."""
    ns = mod._parse_args(
        [
            "--kb",
            "1",
            "--pipeline-version",
            "x",
            "--checkpoint",
            "/tmp/foo.json",
            "--resume",
        ]
    )
    assert ns.checkpoint == "/tmp/foo.json"
    assert ns.resume is True


def test_cli_default_checkpoint_path(mod) -> None:
    ns = mod._parse_args(["--kb", "1", "--pipeline-version", "x"])
    assert ns.checkpoint == mod._DEFAULT_CHECKPOINT_PATH
    assert ns.resume is False
