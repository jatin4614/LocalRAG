"""CLI sanity + dry-run/apply tests for scripts/enable_quantization.py.

``AsyncQdrantClient`` is mocked so we can assert dry-run makes zero writes,
apply hits the right collections with the right ScalarQuantization shape, and
the CLI parses cleanly.
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
SCRIPT = ROOT / "scripts" / "enable_quantization.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "enable_quantization_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_file_exists() -> None:
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"


def test_help_parses_cleanly() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, f"stderr={r.stderr!r}"
    assert "--qdrant-url" in r.stdout
    assert "--dry-run" in r.stdout
    assert "--apply" in r.stdout
    assert "--collections" in r.stdout
    assert "--quantile" in r.stdout
    assert "--rescore" in r.stdout


def test_mutually_exclusive_dry_run_and_apply() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--apply"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode != 0
    assert "not allowed with" in r.stderr or "mutually exclusive" in r.stderr


def test_module_imports_cleanly() -> None:
    mod = _load_module()
    assert hasattr(mod, "main")
    assert callable(mod.main)
    # Open WebUI's own collection must always be skipped.
    assert "open-webui_files" in mod._DEFAULT_EXCLUDES


def test_parse_collections_handles_empty_and_whitespace() -> None:
    mod = _load_module()
    assert mod._parse_collections(None) is None
    assert mod._parse_collections("") is None
    assert mod._parse_collections("   ") is None
    assert mod._parse_collections("kb_1") == ("kb_1",)
    assert mod._parse_collections("kb_1,kb_2") == ("kb_1", "kb_2")
    # Whitespace tolerance: ``'  kb_1 , kb_2  '`` → ('kb_1', 'kb_2').
    assert mod._parse_collections("  kb_1 , kb_2  ") == ("kb_1", "kb_2")


def test_should_skip_respects_defaults_and_cli_excludes() -> None:
    mod = _load_module()
    assert mod._should_skip("open-webui_files", ()) is True
    assert mod._should_skip("kb_eval", ("kb_eval",)) is True
    assert mod._should_skip("kb_1", ()) is False


@pytest.mark.asyncio
async def test_dry_run_makes_no_writes(monkeypatch) -> None:
    """Default (dry-run) must not call update_collection."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[
                SimpleNamespace(name="kb_1"),
                SimpleNamespace(name="kb_eval"),
                SimpleNamespace(name="open-webui_files"),
            ]
        )
    )
    fake_client.update_collection = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        collections=None,
        exclude=[],
        quantile=0.99,
        rescore=2.0,
        timeout=60.0,
        dry_run=True,
        apply=False,
    )
    rc = await mod._apply(args)
    assert rc == 0
    # Critical: zero writes in dry-run.
    fake_client.update_collection.assert_not_called()


@pytest.mark.asyncio
async def test_apply_without_collections_targets_all_nonexcluded(monkeypatch) -> None:
    """--apply with no --collections list hits every non-system collection
    with ScalarQuantization(INT8, 0.99, always_ram=True)."""
    mod = _load_module()
    from qdrant_client.http import models as qm

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[
                SimpleNamespace(name="kb_1"),
                SimpleNamespace(name="kb_2"),
                SimpleNamespace(name="open-webui_files"),  # must be skipped
            ]
        )
    )
    fake_client.update_collection = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        collections=None,
        exclude=[],
        quantile=0.99,
        rescore=2.0,
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 0

    calls = fake_client.update_collection.await_args_list
    # kb_1 + kb_2 only; open-webui_files filtered out.
    assert len(calls) == 2
    touched = {c.kwargs["collection_name"] for c in calls}
    assert touched == {"kb_1", "kb_2"}

    # Every call uses the same ScalarQuantization shape.
    for c in calls:
        qc = c.kwargs["quantization_config"]
        assert isinstance(qc, qm.ScalarQuantization)
        assert qc.scalar.type == qm.ScalarType.INT8
        assert qc.scalar.quantile == 0.99
        assert qc.scalar.always_ram is True


@pytest.mark.asyncio
async def test_apply_with_collections_targets_only_listed(monkeypatch) -> None:
    """--collections kb_1,kb_2 must ignore everything else in Qdrant."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[
                SimpleNamespace(name="kb_1"),
                SimpleNamespace(name="kb_2"),
                SimpleNamespace(name="kb_3"),  # NOT in --collections → skipped
            ]
        )
    )
    fake_client.update_collection = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        collections="kb_1,kb_2",
        exclude=[],
        quantile=0.99,
        rescore=2.0,
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 0

    # get_collections must NOT be consulted when --collections is explicit.
    fake_client.get_collections.assert_not_called()

    calls = fake_client.update_collection.await_args_list
    assert len(calls) == 2
    touched = {c.kwargs["collection_name"] for c in calls}
    assert touched == {"kb_1", "kb_2"}


@pytest.mark.asyncio
async def test_apply_honors_custom_quantile(monkeypatch) -> None:
    """--quantile 0.95 flows through to ScalarQuantizationConfig."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.update_collection = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        collections="kb_1",
        exclude=[],
        quantile=0.95,
        rescore=2.0,
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 0

    c = fake_client.update_collection.await_args_list[0]
    assert c.kwargs["quantization_config"].scalar.quantile == 0.95


@pytest.mark.asyncio
async def test_apply_rejects_out_of_range_quantile(monkeypatch) -> None:
    """--quantile outside [0.5, 1.0] must return rc=4 before any write."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.update_collection = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        collections="kb_1",
        exclude=[],
        quantile=1.5,  # invalid
        rescore=2.0,
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 4
    fake_client.update_collection.assert_not_called()


@pytest.mark.asyncio
async def test_apply_returns_1_on_qdrant_errors(monkeypatch) -> None:
    """A genuine update_collection failure bubbles as rc=1."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.update_collection = AsyncMock(
        side_effect=RuntimeError("connection refused")
    )
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        collections="kb_1",
        exclude=[],
        quantile=0.99,
        rescore=2.0,
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 1


@pytest.mark.asyncio
async def test_apply_exclude_subtracts_from_collections_list(monkeypatch) -> None:
    """--exclude kb_2 trims an explicit --collections list."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.update_collection = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        collections="kb_1,kb_2",
        exclude=["kb_2"],
        quantile=0.99,
        rescore=2.0,
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 0

    calls = fake_client.update_collection.await_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["collection_name"] == "kb_1"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
