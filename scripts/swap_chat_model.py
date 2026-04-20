#!/usr/bin/env python3
"""Swap the chat model in compose/.env and refresh the HF cache.

Run this from the repo root before restarting vllm-chat. The script:
  1. Backs up compose/.env -> compose/.env.bak
  2. Patches CHAT_MODEL=<new-repo> (idempotent)
  3. Optionally patches RAG_BUDGET_TOKENIZER=<alias>
  4. Runs preflight with HF_HUB_OFFLINE=0 to populate the model cache
  5. Prints the `docker compose restart` line to run next

The script NEVER calls docker itself. It refuses to run if --model is the
same as the current CHAT_MODEL. Use --force to override.

Exit codes:
    0  success (or dry-run success)
    2  invalid args / same-model refusal / preflight failure
    3  .env not found / malformed
    4  preflight script missing
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = REPO_ROOT / "compose" / ".env"
DEFAULT_PREFLIGHT = REPO_ROOT / "scripts" / "preflight_models.py"


def _read_env_var(env_path: Path, key: str) -> str | None:
    """Return the value of KEY=... in a simple dotenv file, or None."""
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, v = stripped.partition("=")
        if k.strip() == key:
            # Strip surrounding quotes (single or double)
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (
                v.startswith("'") and v.endswith("'")
            ):
                v = v[1:-1]
            return v
    return None


def _patch_env_var(env_path: Path, key: str, value: str) -> bool:
    """Idempotently set KEY=VALUE. Returns True if file changed.

    - If KEY exists, replace the whole line (preserves surrounding lines).
    - If KEY is absent, append ``KEY=VALUE`` at the end.
    """
    original = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    lines = original.splitlines()
    found = False
    new_line = f"{key}={value}"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, _ = stripped.partition("=")
        if k.strip() == key:
            lines[i] = new_line
            found = True
            break
    if not found:
        lines.append(new_line)
    updated = "\n".join(lines)
    if original.endswith("\n") or not original:
        updated += "\n"
    if updated == original:
        return False
    env_path.write_text(updated, encoding="utf-8")
    return True


def _tokenizer_hint(hf_repo: str) -> str | None:
    """Best-effort alias suggestion from the repo id."""
    name = hf_repo.lower()
    if "qwen" in name:
        return "qwen"
    if "gemma" in name:
        return "gemma"
    if "llama" in name:
        return "llama"
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Swap CHAT_MODEL in compose/.env and refresh HF cache."
    )
    ap.add_argument(
        "--model",
        required=True,
        help="New HF repo id, e.g. google/gemma-3-12b-it",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "RAG_BUDGET_TOKENIZER alias (qwen / gemma / llama / cl100k / ...). "
            "If omitted, the script suggests one from the repo name but leaves "
            "the .env value unchanged."
        ),
    )
    ap.add_argument(
        "--env",
        default=str(DEFAULT_ENV),
        help=f"Path to compose/.env (default: {DEFAULT_ENV})",
    )
    ap.add_argument(
        "--preflight",
        default=str(DEFAULT_PREFLIGHT),
        help=f"Path to preflight_models.py (default: {DEFAULT_PREFLIGHT})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan; don't modify .env or download anything.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if --model matches the current CHAT_MODEL.",
    )
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Patch .env only; don't run preflight_models.py.",
    )

    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0

    env_path = Path(args.env)
    if not env_path.is_file():
        print(f"!! .env not found at {env_path}", file=sys.stderr)
        return 3

    current_model = _read_env_var(env_path, "CHAT_MODEL")
    if current_model is None:
        print(f"!! CHAT_MODEL not found in {env_path}", file=sys.stderr)
        return 3

    if current_model == args.model and not args.force:
        print(
            f"!! --model {args.model!r} matches current CHAT_MODEL. "
            "Nothing to do. Use --force to override.",
            file=sys.stderr,
        )
        return 2

    suggested = _tokenizer_hint(args.model)
    tokenizer_alias = args.tokenizer or suggested

    print(f"current CHAT_MODEL:  {current_model}")
    print(f"new CHAT_MODEL:      {args.model}")
    if tokenizer_alias:
        current_tok = _read_env_var(env_path, "RAG_BUDGET_TOKENIZER") or "cl100k"
        if args.tokenizer is None:
            print(
                f"tokenizer alias:     {tokenizer_alias}  "
                f"(suggested from repo name; current={current_tok}; "
                "pass --tokenizer to apply)"
            )
        else:
            print(f"tokenizer alias:     {tokenizer_alias}  (will update)")

    if args.dry_run:
        print("dry-run: no changes written, no downloads started.")
        return 0

    backup = env_path.with_suffix(env_path.suffix + ".bak")
    shutil.copy2(env_path, backup)
    print(f"backed up .env -> {backup}")

    changed = _patch_env_var(env_path, "CHAT_MODEL", args.model)
    if changed:
        print(f"patched CHAT_MODEL={args.model}")
    else:
        print("CHAT_MODEL already at target value (no change).")

    if args.tokenizer is not None:
        tchanged = _patch_env_var(env_path, "RAG_BUDGET_TOKENIZER", args.tokenizer)
        if tchanged:
            print(f"patched RAG_BUDGET_TOKENIZER={args.tokenizer}")

    if args.skip_preflight:
        print("--skip-preflight set; skipping model download.")
    else:
        preflight = Path(args.preflight)
        if not preflight.is_file():
            print(f"!! preflight script missing at {preflight}", file=sys.stderr)
            return 4
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "0"
        env["TRANSFORMERS_OFFLINE"] = "0"
        env["CHAT_MODEL"] = args.model
        print(f"running preflight: {preflight} (HF_HUB_OFFLINE=0)")
        r = subprocess.run([sys.executable, str(preflight)], env=env)
        if r.returncode != 0:
            print(
                f"!! preflight exited {r.returncode}; .env was patched but "
                "model cache may be incomplete. Inspect + re-run before "
                "restarting services.",
                file=sys.stderr,
            )
            return 2

    print()
    print("Next step:")
    print("    docker compose restart vllm-chat model-manager")
    print("Verify with:")
    print("    curl http://localhost:8000/v1/models")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
