#!/usr/bin/env python3
"""Download + verify the four model weights used by the stack.

Runs ONCE on a machine with internet access; results are cached into
$MODEL_CACHE (default: ./volumes/models) and reused by all services
with HF_HUB_OFFLINE=1 thereafter.

Usage:
    python scripts/preflight_models.py              # download everything
    python scripts/preflight_models.py --dry-run    # print plan, download nothing
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


def _download_hf(repo_id: str, cache_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir), force_download=False)


def _download_whisper(size: str, cache_dir: Path) -> None:
    from faster_whisper import WhisperModel

    WhisperModel(size, download_root=str(cache_dir))


def run(plan: Iterable[tuple[str, str]], cache_dir: Path, dry_run: bool) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    free = _free_disk_gb(cache_dir)
    print(f"cache: {cache_dir}   free disk: {free:.1f} GB")
    if free < 40 and not dry_run:
        print("!! less than 40 GB free; aborting", file=sys.stderr)
        return 2

    for kind, ident in plan:
        print(f"[{kind}] {ident}")
        if dry_run:
            continue
        if kind == "hf":
            _download_hf(ident, cache_dir)
        elif kind == "whisper":
            _download_whisper(ident, cache_dir)
        else:
            print(f"!! unknown kind {kind}", file=sys.stderr)
            return 3

    print("preflight OK")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Download and verify model weights before first compose up."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without actually fetching anything.",
    )
    ap.add_argument(
        "--cache-dir",
        default=_env("MODEL_CACHE", "./volumes/models"),
        help="Directory to store model weights (default: $MODEL_CACHE or ./volumes/models).",
    )
    args = ap.parse_args(argv)
    return run(_plan(), Path(args.cache_dir), args.dry_run)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
