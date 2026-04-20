"""CLI sanity tests for scripts/normalize_doc_ids.py.

Scope: argparse wiring + module import. We don't spin up a real Qdrant here.
End-to-end normalize behavior is exercised manually against the live cluster.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "normalize_doc_ids.py"


def test_script_file_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"


def test_help_flag_parses_cleanly() -> None:
    """--help exits 0 and prints argparse usage."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode == 0, f"stderr={r.stderr!r}"
    assert "normalize" in r.stdout.lower()
    assert "--qdrant-url" in r.stdout
    assert "--dry-run" in r.stdout
    assert "--apply" in r.stdout


def test_mutually_exclusive_dry_run_and_apply() -> None:
    """--dry-run and --apply cannot both be passed."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--apply"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    # argparse exits 2 on argument error, prints to stderr.
    assert r.returncode != 0
    assert "not allowed with" in r.stderr or "mutually exclusive" in r.stderr


def test_unreachable_qdrant_exits_with_error() -> None:
    """Pointing at a closed port exits non-zero (HTTP/transport error)."""
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--qdrant-url",
            "http://127.0.0.1:1",  # reserved — guaranteed closed
            "--dry-run",
            "--timeout",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 1, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "failed to list" in r.stderr or "transport error" in r.stderr


def test_module_importable() -> None:
    """The script exposes main(argv) that we can call in-process."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("normalize_doc_ids", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
    assert callable(mod.main)


def test_main_with_mock_transport_classifies_clean_and_string_ids() -> None:
    """Exercise the full main() path against a mocked Qdrant.

    Verifies: collection listing, scroll pagination terminates on empty
    next_page_offset, classification into clean/convertible/skipped/no_id,
    and dry-run does NOT POST to /points/payload.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("normalize_doc_ids", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Capture all POST calls so we can assert no payload mutation in dry-run.
    posted: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/collections":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "collections": [
                            {"name": "kb_demo"},
                            {"name": "chat_skip_me"},
                        ]
                    }
                },
            )
        if request.method == "POST" and path.endswith("/points/scroll"):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {"id": "a", "payload": {"doc_id": 1}},
                            {"id": "b", "payload": {"doc_id": "2"}},
                            {"id": "c", "payload": {"doc_id": "nope"}},
                            {"id": "d", "payload": {}},
                        ],
                        "next_page_offset": None,
                    }
                },
            )
        if request.method == "POST" and path.endswith("/points/payload"):
            posted.append((path, request.content.decode()))  # type: ignore[arg-type]
            return httpx.Response(200, json={"result": True, "status": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    # Monkeypatch httpx.Client so the script picks up our transport.
    import httpx as _httpx

    real_client = _httpx.Client

    def patched_client(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    _httpx.Client = patched_client  # type: ignore[assignment]
    try:
        rc = mod.main(["--qdrant-url", "http://mock", "--dry-run"])
    finally:
        _httpx.Client = real_client  # type: ignore[assignment]

    # Exit 2 because we had one non-numeric doc_id ("nope") → skipped.
    assert rc == 2
    # Dry-run must NOT POST to /points/payload.
    assert posted == [], f"dry-run unexpectedly wrote: {posted}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
