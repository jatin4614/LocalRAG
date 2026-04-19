"""CLI sanity + dry-run behaviour tests for scripts/reindex_hybrid.py.

We never touch a real Qdrant or fastembed here. The script is exercised
through argparse + a mocked ``AsyncQdrantClient`` so we can assert it
(a) parses cleanly, (b) refuses bad input early, and (c) makes zero
writes in the default dry-run mode. A real reindex against a live
collection is an opt-in manual step run by the operator.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest  # noqa: F401 - needed for @pytest.mark.asyncio

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "reindex_hybrid.py"


def _load_module():
    """Load the script as a module so we can drive ``main(argv)`` in-process."""
    spec = importlib.util.spec_from_file_location("reindex_hybrid_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_file_exists() -> None:
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"


def test_help_parses_cleanly() -> None:
    """--help exits 0, argparse usage mentions --source / --target / --apply."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, f"stderr={r.stderr!r}"
    assert "--source" in r.stdout
    assert "--target" in r.stdout
    assert "--apply" in r.stdout
    assert "--dry-run" in r.stdout


def test_missing_required_args_exits_nonzero() -> None:
    """Without --source / --target, argparse errors."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--apply"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode != 0
    # argparse emits 'the following arguments are required' or similar.
    combined = r.stderr.lower() + r.stdout.lower()
    assert "required" in combined or "source" in combined


def test_module_imports_cleanly() -> None:
    """Import the script as a module — no side effects, no crashes."""
    mod = _load_module()
    assert hasattr(mod, "main")
    assert callable(mod.main)
    assert hasattr(mod, "_infer_vector_size_and_distance")
    assert hasattr(mod, "_dense_vector_for")
    # The named-vector constants must match VectorStore exactly — a drift
    # here would silently break hybrid retrieval after reindex.
    assert mod._DENSE_NAME == "dense"
    assert mod._SPARSE_NAME == "bm25"


def test_mutually_exclusive_dry_run_and_apply() -> None:
    """argparse refuses both --dry-run and --apply."""
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            "kb_1",
            "--target",
            "kb_1_v2",
            "--dry-run",
            "--apply",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode != 0
    assert "not allowed with" in r.stderr or "mutually exclusive" in r.stderr


def test_infer_vector_size_legacy_unnamed() -> None:
    """Legacy unnamed source → read size + distance off VectorParams directly."""
    mod = _load_module()
    info = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=1024, distance="Cosine"),
            )
        )
    )
    size, dist = mod._infer_vector_size_and_distance(info)
    assert size == 1024
    assert dist == "Cosine"


def test_infer_vector_size_named_dict() -> None:
    """Named-vector source → read the ``dense`` entry."""
    mod = _load_module()
    info = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": SimpleNamespace(size=768, distance="Dot")},
            )
        )
    )
    size, dist = mod._infer_vector_size_and_distance(info)
    assert size == 768
    assert dist == "Dot"


def test_dense_vector_for_legacy_unnamed() -> None:
    """A legacy point carries ``vector`` as a plain list."""
    mod = _load_module()
    pt = SimpleNamespace(vector=[0.1, 0.2, 0.3])
    assert mod._dense_vector_for(pt) == [0.1, 0.2, 0.3]


def test_dense_vector_for_named_dict_pulls_dense() -> None:
    mod = _load_module()
    pt = SimpleNamespace(vector={"dense": [1.0, 2.0], "other": [9.9]})
    assert mod._dense_vector_for(pt) == [1.0, 2.0]


def test_dense_vector_for_missing_vector_returns_none() -> None:
    mod = _load_module()
    pt = SimpleNamespace(vector=None)
    assert mod._dense_vector_for(pt) is None


@pytest.mark.asyncio
async def test_dry_run_makes_no_writes(monkeypatch) -> None:
    """Default (dry-run) must not call create_collection or upsert.

    We drive ``_reindex`` directly rather than ``main`` because ``main``
    wraps with ``asyncio.run`` and pytest-asyncio already owns an event loop.
    """
    mod = _load_module()

    # Fake source collection info (legacy unnamed, 1024 dims).
    info = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=1024, distance="Cosine"),
            )
        )
    )

    fake_client = MagicMock()
    fake_client.get_collection = AsyncMock(return_value=info)
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(collections=[SimpleNamespace(name="kb_1")])
    )
    fake_client.count = AsyncMock(return_value=SimpleNamespace(count=42))
    fake_client.create_collection = AsyncMock()
    fake_client.upsert = AsyncMock()
    fake_client.scroll = AsyncMock()
    fake_client.delete_collection = AsyncMock()
    fake_client.create_payload_index = AsyncMock()
    fake_client.close = AsyncMock()

    # Patch AsyncQdrantClient inside the qdrant_client module so the script's
    # ``from qdrant_client import AsyncQdrantClient`` inside _reindex picks it up.
    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        tei_url="http://mock-tei",
        source="kb_1",
        target="kb_1_v2",
        batch_size=64,
        force=False,
        timeout=60.0,
        dry_run=True,
        apply=False,
    )
    rc = await mod._reindex(args)
    assert rc == 0
    # Critical invariant: zero writes.
    fake_client.create_collection.assert_not_called()
    fake_client.upsert.assert_not_called()
    fake_client.delete_collection.assert_not_called()
    fake_client.scroll.assert_not_called()


@pytest.mark.asyncio
async def test_target_exists_without_force_errors(monkeypatch) -> None:
    """If target already exists and --force is NOT passed, return 1."""
    mod = _load_module()

    info = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=1024, distance="Cosine"),
            )
        )
    )
    fake_client = MagicMock()
    fake_client.get_collection = AsyncMock(return_value=info)
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[
                SimpleNamespace(name="kb_1"),
                SimpleNamespace(name="kb_1_v2"),  # target already exists
            ]
        )
    )
    fake_client.count = AsyncMock(return_value=SimpleNamespace(count=42))
    fake_client.create_collection = AsyncMock()
    fake_client.upsert = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        tei_url="http://mock-tei",
        source="kb_1",
        target="kb_1_v2",
        batch_size=64,
        force=False,
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._reindex(args)
    assert rc == 1
    fake_client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_missing_source_errors(monkeypatch) -> None:
    """If source collection doesn't exist, return 1 and make no writes."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.get_collection = AsyncMock(side_effect=RuntimeError("not found"))
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(collections=[])
    )
    fake_client.create_collection = AsyncMock()
    fake_client.upsert = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        tei_url="http://mock-tei",
        source="kb_missing",
        target="kb_missing_v2",
        batch_size=64,
        force=False,
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._reindex(args)
    assert rc == 1
    fake_client.create_collection.assert_not_called()
    fake_client.upsert.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
