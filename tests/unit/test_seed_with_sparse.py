"""Tests for the ``--with-sparse`` flag on scripts/seed_eval_corpus.py.

These tests never hit real Qdrant, TEI, or fastembed — they drive ``seed()``
with a fake Qdrant client and a monkeypatched ``embed_sparse`` so we can
assert the shape of the points that would be upserted. The goal is to prove
the flag:
  * shows up in ``--help``
  * causes the script to attach ``(indices, values)`` sparse vectors to
    each point
  * silently does nothing sparse-related when absent (backwards-compat)
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "seed_eval_corpus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("seed_eval_sparse_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_help_mentions_with_sparse() -> None:
    """The CLI must surface --with-sparse in argparse help."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r.returncode == 0, r.stderr
    assert "--with-sparse" in r.stdout
    # Mention hybrid/dense+bm25 so operators understand what the flag does.
    assert "sparse" in r.stdout.lower()


def _fake_qdrant(*, existing_points: int = 0, existing_sparse: bool | None = None) -> MagicMock:
    """Build a fake AsyncQdrantClient that mimics just enough for seed()."""
    client = MagicMock()

    # get_collections: start empty or with kb_eval already present.
    if existing_sparse is not None:
        info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    sparse_vectors={"bm25": object()} if existing_sparse else None,
                )
            )
        )
        client.get_collection = AsyncMock(return_value=info)
        client.get_collections = AsyncMock(
            return_value=SimpleNamespace(collections=[SimpleNamespace(name="kb_eval")])
        )
    else:
        # Collection does not exist yet.
        client.get_collection = AsyncMock(side_effect=RuntimeError("not found"))
        client.get_collections = AsyncMock(
            return_value=SimpleNamespace(collections=[])
        )

    client.count = AsyncMock(return_value=SimpleNamespace(count=existing_points))
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.delete_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.close = AsyncMock()
    return client


def _fake_tei_client(vector_size: int = 1024) -> MagicMock:
    """Stand-in for httpx.AsyncClient — only needs .post() and .aclose()."""
    client = MagicMock()

    async def _post(url, json=None, **kw):  # noqa: ARG001 - signature-compat
        texts = json.get("inputs", []) if json else []
        # Return a list of fake 1024-dim vectors; values don't matter for
        # these tests (we only assert shape, not content).
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=[[0.0] * vector_size for _ in texts])
        return resp

    client.post = _post
    client.aclose = AsyncMock()
    return client


def _make_args(tmp_path: Path, *, with_sparse: bool, force: bool = True) -> SimpleNamespace:
    """Minimal argparse.Namespace-alike for driving seed() in-process."""
    return SimpleNamespace(
        qdrant_url="http://mock-qdrant",
        tei_url="http://mock-tei",
        collection_name="kb_eval",
        vector_size=1024,
        chunk_tokens=800,
        overlap_tokens=100,
        batch_size=16,
        force=force,
        with_sparse=with_sparse,
        timeout=60.0,
    )


def _install_fakes(
    monkeypatch,
    mod,
    *,
    fake_qdrant: MagicMock,
    fake_tei: MagicMock,
    sparse_call_count: list[int],
) -> None:
    """Swap out AsyncQdrantClient, httpx.AsyncClient, and embed_sparse."""
    # Patch the names *as the script imports them*.
    monkeypatch.setattr(mod, "AsyncQdrantClient", lambda *a, **kw: fake_qdrant)

    class _TEI:
        def __init__(self, *a, **kw):
            pass

        def __new__(cls, *a, **kw):
            return fake_tei

    monkeypatch.setattr(mod.httpx, "AsyncClient", _TEI)

    # embed_sparse stub: records call counts and returns deterministic pairs.
    from ext.services import sparse_embedder as se_mod

    def fake_embed_sparse(texts):
        sparse_call_count.append(len(list(texts)))
        return [([1, 2, 3], [1.0, 1.0, 1.0]) for _ in texts]

    monkeypatch.setattr(se_mod, "embed_sparse", fake_embed_sparse)


@pytest.mark.asyncio
async def test_seed_without_sparse_never_calls_embed_sparse(monkeypatch, tmp_path) -> None:
    """Backwards-compat: no --with-sparse → zero sparse computation."""
    mod = _load_module()
    fake_qdrant = _fake_qdrant(existing_points=0)
    fake_tei = _fake_tei_client()
    sparse_calls: list[int] = []
    _install_fakes(
        monkeypatch,
        mod,
        fake_qdrant=fake_qdrant,
        fake_tei=fake_tei,
        sparse_call_count=sparse_calls,
    )

    args = _make_args(tmp_path, with_sparse=False)
    rc = await mod.seed(args)
    assert rc == 0, "seed must succeed on fresh fake collection"

    # Zero sparse calls — the flag was not set.
    assert sparse_calls == [], f"expected no embed_sparse calls, got {sparse_calls}"

    # Every upsert must use the legacy unnamed-vector shape: vector is a list
    # (not a dict). We don't care which call — just that no point ever was
    # built with a named-dict vector in this mode.
    for call in fake_qdrant.upsert.await_args_list:
        _, kwargs = call
        for pt in kwargs.get("points", []):
            assert not isinstance(pt.vector, dict), (
                f"dense-only mode must use unnamed vector; got dict on point {pt.id}"
            )


@pytest.mark.asyncio
async def test_seed_with_sparse_attaches_sparse_vectors(monkeypatch, tmp_path) -> None:
    """--with-sparse → every upserted point carries a named ``bm25`` sparse vector."""
    mod = _load_module()
    fake_qdrant = _fake_qdrant(existing_points=0)
    fake_tei = _fake_tei_client()
    sparse_calls: list[int] = []
    _install_fakes(
        monkeypatch,
        mod,
        fake_qdrant=fake_qdrant,
        fake_tei=fake_tei,
        sparse_call_count=sparse_calls,
    )

    args = _make_args(tmp_path, with_sparse=True)
    rc = await mod.seed(args)
    assert rc == 0

    # Sparse must have been computed at least once (at least one file had text).
    assert sum(sparse_calls) > 0, "embed_sparse was never called"

    # create_collection must have been called with BOTH vectors_config AND
    # sparse_vectors_config — i.e. the hybrid shape.
    create_calls = fake_qdrant.create_collection.await_args_list
    assert create_calls, "create_collection was not called — no collection created"
    _, kwargs = create_calls[0]
    assert "vectors_config" in kwargs
    assert "sparse_vectors_config" in kwargs
    assert "dense" in kwargs["vectors_config"]
    assert "bm25" in kwargs["sparse_vectors_config"]

    # Every upserted point must use the named-vector dict shape with both
    # ``dense`` and ``bm25`` keys populated.
    upsert_calls = fake_qdrant.upsert.await_args_list
    assert upsert_calls, "upsert was never called — no points attempted"
    any_hybrid_point = False
    for call in upsert_calls:
        _, kwargs = call
        for pt in kwargs.get("points", []):
            assert isinstance(pt.vector, dict), (
                f"hybrid mode must use named-vector dict; got {type(pt.vector)} on {pt.id}"
            )
            assert "dense" in pt.vector and "bm25" in pt.vector, (
                f"point {pt.id} missing named entries: keys={list(pt.vector)}"
            )
            any_hybrid_point = True
    assert any_hybrid_point, "no hybrid-shaped points were upserted"


@pytest.mark.asyncio
async def test_seed_with_sparse_requires_force_when_shape_mismatches(
    monkeypatch, tmp_path
) -> None:
    """An existing dense-only kb_eval + --with-sparse without --force → exit 2.

    Qdrant cannot change vector config in place; forcing the operator to
    pass --force makes the destructive wipe explicit.
    """
    mod = _load_module()
    fake_qdrant = _fake_qdrant(existing_points=42, existing_sparse=False)
    fake_tei = _fake_tei_client()
    sparse_calls: list[int] = []
    _install_fakes(
        monkeypatch,
        mod,
        fake_qdrant=fake_qdrant,
        fake_tei=fake_tei,
        sparse_call_count=sparse_calls,
    )

    args = _make_args(tmp_path, with_sparse=True, force=False)
    rc = await mod.seed(args)
    assert rc == 2
    # Must not have created anything or upserted.
    fake_qdrant.create_collection.assert_not_called()
    fake_qdrant.upsert.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
