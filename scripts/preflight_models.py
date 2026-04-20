#!/usr/bin/env python3
"""Download + verify the four model weights used by the stack.

Runs ONCE on a machine with internet access; results are cached into
$MODEL_CACHE (default: ./volumes/models) and reused by all services
with HF_HUB_OFFLINE=1 thereafter.

Usage:
    python scripts/preflight_models.py                # download everything
    python scripts/preflight_models.py --dry-run      # print plan, download nothing
    python scripts/preflight_models.py --verify-only  # check cache only; exit 2 if missing

Exit codes:
    0  all models present / download succeeded
    2  missing weights or insufficient disk
    3  network / HuggingFace unreachable
    4  invalid arguments
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _plan() -> list[tuple[str, str]]:
    """Returns (kind, identifier) pairs."""
    return [
        ("hf", _env("CHAT_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")),
        ("hf", _env("VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")),
        ("hf", _env("EMBED_MODEL", "BAAI/bge-m3")),
        ("whisper", _env("WHISPER_MODEL", "medium")),
    ]


def _free_disk_gb(path: Path) -> float:
    target = str(path) if path.exists() else str(path.parent)
    total, used, free = shutil.disk_usage(target)
    return free / (1024**3)


def _hf_cache_dirname(repo_id: str) -> str:
    """HuggingFace Hub cache layout: models--<org>--<name>."""
    return "models--" + repo_id.replace("/", "--")


def _whisper_cache_dirname(size: str) -> str:
    """faster_whisper downloads Systran/faster-whisper-<size> by default."""
    return "models--Systran--faster-whisper-" + size


def _is_hf_snapshot_populated(repo_dir: Path) -> bool:
    """True iff snapshots/ has at least one commit dir with at least one file."""
    snap_root = repo_dir / "snapshots"
    if not snap_root.is_dir():
        return False
    for commit_dir in snap_root.iterdir():
        if commit_dir.is_dir():
            # any file (including symlinks to blobs) counts as populated
            for _ in commit_dir.iterdir():
                return True
    return False


def _verify(plan: Iterable[tuple[str, str]], cache_dir: Path) -> list[str]:
    """Return a list of missing model identifiers. Empty list = all good."""
    missing: list[str] = []
    for kind, ident in plan:
        if kind == "hf":
            repo_dir = cache_dir / _hf_cache_dirname(ident)
        elif kind == "whisper":
            repo_dir = cache_dir / _whisper_cache_dirname(ident)
        else:
            missing.append(f"{ident} (unknown kind {kind})")
            continue
        if not _is_hf_snapshot_populated(repo_dir):
            missing.append(ident)
    return missing


def _is_network_error(exc: BaseException) -> bool:
    """Best-effort sniff for 'can't reach HF' errors across hub versions."""
    name = type(exc).__name__
    msg = str(exc).lower()
    net_markers = (
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
    )
    if any(m in name for m in net_markers):
        return True
    if any(
        m in msg
        for m in (
            "temporary failure in name resolution",
            "failed to resolve",
            "connection refused",
            "no route to host",
            "network is unreachable",
            "max retries exceeded",
            "name or service not known",
        )
    ):
        return True
    return False


def _download_hf(repo_id: str, cache_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir), force_download=False)


def _download_whisper(size: str, cache_dir: Path) -> None:
    from faster_whisper import WhisperModel

    WhisperModel(size, download_root=str(cache_dir))


def _network_error_hint(cache_dir: Path) -> str:
    return (
        "!! Cannot reach huggingface.co. If you are on an air-gapped host, "
        f"copy `{cache_dir}/` from a connected machine."
    )


def run(
    plan: Iterable[tuple[str, str]],
    cache_dir: Path,
    dry_run: bool,
    verify_only: bool = False,
) -> int:
    plan = list(plan)
    cache_dir.mkdir(parents=True, exist_ok=True)
    free = _free_disk_gb(cache_dir)
    print(f"cache: {cache_dir}   free disk: {free:.1f} GB")

    # --verify-only: skip disk / plan print; just check and return.
    if verify_only:
        missing = _verify(plan, cache_dir)
        if missing:
            print("!! missing models:", file=sys.stderr)
            for m in missing:
                print(f"   - {m}", file=sys.stderr)
            print(
                "!! run `python scripts/preflight_models.py` from a machine with internet "
                "access, or copy volumes/models/ from a connected machine.",
                file=sys.stderr,
            )
            return 2
        print("preflight OK (verify-only)")
        return 0

    if free < 40 and not dry_run:
        print("!! less than 40 GB free; aborting", file=sys.stderr)
        return 2

    for kind, ident in plan:
        print(f"[{kind}] {ident}")
        if dry_run:
            continue
        try:
            if kind == "hf":
                _download_hf(ident, cache_dir)
            elif kind == "whisper":
                _download_whisper(ident, cache_dir)
            else:
                print(f"!! unknown kind {kind}", file=sys.stderr)
                return 2
        except Exception as exc:  # noqa: BLE001 - we want a single catch-all here
            if _is_network_error(exc):
                print(_network_error_hint(cache_dir), file=sys.stderr)
                print(f"   (underlying error: {type(exc).__name__}: {exc})", file=sys.stderr)
                return 3
            # Re-raise anything we don't recognise so the traceback is useful.
            raise

    print("preflight OK")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Download and verify model weights before first compose up.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without actually fetching anything.",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Don't download; just check that expected model weights exist in the cache. "
            "Exit 0 if all present, 2 if any missing."
        ),
    )
    ap.add_argument(
        "--cache-dir",
        default=_env("MODEL_CACHE", "./volumes/models"),
        help="Directory to store model weights (default: $MODEL_CACHE or ./volumes/models).",
    )
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        # argparse exits with 2 on parse error; remap to our convention (4 = invalid args).
        if exc.code == 2:
            return 4
        raise
    if args.dry_run and args.verify_only:
        print("!! --dry-run and --verify-only are mutually exclusive", file=sys.stderr)
        return 4
    return run(_plan(), Path(args.cache_dir), args.dry_run, args.verify_only)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
