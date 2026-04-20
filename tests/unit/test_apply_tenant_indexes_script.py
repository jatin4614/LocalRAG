"""CLI sanity + dry-run behaviour tests for scripts/apply_tenant_indexes.py.

Never touches a real Qdrant — ``AsyncQdrantClient`` is mocked so we can
verify that dry-run makes zero writes, and the CLI parses cleanly.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "apply_tenant_indexes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "apply_tenant_indexes_under_test", SCRIPT
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
    assert "--exclude" in r.stdout


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
    # Tenant + filter field lists must stay in sync with VectorStore —
    # a drift here silently misses indexes on one side of the code.
    assert mod._TENANT_FIELDS == ("kb_id", "chat_id", "owner_user_id")
    assert mod._FILTER_FIELDS == ("subtag_id", "doc_id", "deleted")
    # Open WebUI's own collection must always be skipped.
    assert "open-webui_files" in mod._DEFAULT_EXCLUDES


def test_should_skip_respects_defaults_and_cli_excludes() -> None:
    mod = _load_module()
    # Default exclude.
    assert mod._should_skip("open-webui_files", ()) is True
    # CLI exclude.
    assert mod._should_skip("kb_eval", ("kb_eval",)) is True
    # Regular KB collection is NOT skipped.
    assert mod._should_skip("kb_1", ()) is False
    assert mod._should_skip("kb_1_v2", ("kb_eval",)) is False


@pytest.mark.asyncio
async def test_dry_run_makes_no_writes(monkeypatch) -> None:
    """Default (dry-run) must not call create_payload_index at all."""
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
    fake_client.create_payload_index = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        exclude=[],
        timeout=60.0,
        dry_run=True,
        apply=False,
    )
    rc = await mod._apply(args)
    assert rc == 0
    # Critical: zero writes in dry-run.
    fake_client.create_payload_index.assert_not_called()


@pytest.mark.asyncio
async def test_apply_creates_tenant_and_filter_indexes(monkeypatch) -> None:
    """--apply creates tenant-flagged indexes on tenant fields and plain on the rest."""
    mod = _load_module()
    from qdrant_client.http import models as qm

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[
                SimpleNamespace(name="kb_1"),
                SimpleNamespace(name="open-webui_files"),  # must be skipped
            ]
        )
    )
    fake_client.create_payload_index = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        exclude=[],
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 0

    # 6 calls total for kb_1 only (3 tenant + 3 filter); open-webui_files skipped.
    calls = fake_client.create_payload_index.await_args_list
    assert len(calls) == 6
    collections_touched = {c.kwargs["collection_name"] for c in calls}
    assert collections_touched == {"kb_1"}

    by_field = {c.kwargs["field_name"]: c.kwargs["field_schema"] for c in calls}
    for field in ("kb_id", "chat_id", "owner_user_id"):
        schema = by_field[field]
        assert isinstance(schema, qm.KeywordIndexParams)
        assert schema.is_tenant is True
    for field in ("subtag_id", "doc_id", "deleted"):
        assert by_field[field] == qm.PayloadSchemaType.KEYWORD


@pytest.mark.asyncio
async def test_apply_tolerates_duplicate_index_errors(monkeypatch) -> None:
    """Qdrant raises 'already exists' on re-registration — must be counted as noop, not error."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[SimpleNamespace(name="kb_1")]
        )
    )
    fake_client.create_payload_index = AsyncMock(
        side_effect=RuntimeError("index already exists")
    )
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        exclude=[],
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    # 0 (not 1) — duplicates aren't errors.
    assert rc == 0


@pytest.mark.asyncio
async def test_apply_respects_cli_exclude(monkeypatch) -> None:
    """--exclude kb_eval means kb_eval is never touched, even on --apply."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[
                SimpleNamespace(name="kb_1"),
                SimpleNamespace(name="kb_eval"),
            ]
        )
    )
    fake_client.create_payload_index = AsyncMock()
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        exclude=["kb_eval"],
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 0

    collections_touched = {
        c.kwargs["collection_name"]
        for c in fake_client.create_payload_index.await_args_list
    }
    assert collections_touched == {"kb_1"}


@pytest.mark.asyncio
async def test_apply_returns_1_on_genuine_errors(monkeypatch) -> None:
    """A non-'already exists' failure bubbles up as rc=1."""
    mod = _load_module()

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[SimpleNamespace(name="kb_1")]
        )
    )
    fake_client.create_payload_index = AsyncMock(
        side_effect=RuntimeError("connection refused")
    )
    fake_client.close = AsyncMock()

    import qdrant_client as qc_mod

    monkeypatch.setattr(qc_mod, "AsyncQdrantClient", lambda *a, **kw: fake_client)

    args = SimpleNamespace(
        qdrant_url="http://mock",
        exclude=[],
        timeout=60.0,
        dry_run=False,
        apply=True,
    )
    rc = await mod._apply(args)
    assert rc == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
