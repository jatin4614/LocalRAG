#!/usr/bin/env python3
"""Retrofit scalar INT8 quantization onto existing Qdrant collections.

Qdrant supports changing ``quantization_config`` at runtime via
``update_collection``. The engine builds the quantized (INT8) index
asynchronously in the background; original fp32 vectors are untouched. The
change is a settings-only update — zero data rewrites, zero downtime.

This script walks either an explicit ``--collections`` list or every
non-system collection and applies
``ScalarQuantization(INT8, quantile, always_ram=True)`` to each. Re-running
the script is idempotent: Qdrant no-ops when the config already matches.

Scalar INT8 ~= 4× vector-RAM reduction, < 2% recall loss at quantile=0.99.
Binary quantization is intentionally *not* offered here — bge-m3 is 1024-
dimensional, the known borderline below which binary recall collapses per
Qdrant's own research.

Usage:
    # Dry-run (default): print the plan, make no writes.
    python scripts/enable_quantization.py --qdrant-url http://localhost:6333

    # Apply to everything (skips open-webui_files).
    python scripts/enable_quantization.py --qdrant-url http://localhost:6333 --apply

    # Apply to a specific set.
    python scripts/enable_quantization.py \\
        --collections kb_eval,kb_1_v2,chat_private \\
        --apply

    # Tune the outlier-clamp quantile (default 0.99 — safer recall).
    python scripts/enable_quantization.py --quantile 0.95 --apply

Per-query rescoring
-------------------
The rescore behaviour (``QuantizationSearchParams(rescore=True,
oversampling=N)``) is attached at *query* time by the VectorStore, not at
collection-create time. This script's ``--rescore N`` arg is therefore only
advisory — it records the recommended oversampling in the log output and
reminds operators to set ``RAG_QDRANT_OVERSAMPLING`` to match. The default of
2.0 is already a sensible universal setting.

Exit codes:
    0  success (or dry-run finished cleanly)
    1  Qdrant error
    2  qdrant-client not installed
    4  invalid arguments
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Make ``ext.services.*`` importable regardless of how the script is invoked.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Collections we never touch — these are Open WebUI's own data.
_DEFAULT_EXCLUDES = ("open-webui_files",)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant REST URL (default: http://localhost:6333).",
    )
    p.add_argument(
        "--collections",
        default=None,
        help=(
            "Comma-separated collection names to target (e.g. "
            "'kb_eval,kb_1_v2'). When omitted, every non-excluded collection "
            "is targeted."
        ),
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Collection name to skip (exact match). May be passed multiple "
            f"times. {', '.join(_DEFAULT_EXCLUDES)!r} is always skipped."
        ),
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help=(
            "Outlier-clamp quantile for INT8 scaling (default: 0.99 — "
            "Qdrant's recommended safe default; drop to 0.95 for slightly "
            "more speed at ~1%% extra recall loss)."
        ),
    )
    p.add_argument(
        "--rescore",
        type=float,
        default=2.0,
        help=(
            "Advisory: the recommended oversampling ratio for per-query "
            "rescoring. Not written to Qdrant by this script — configure "
            "via RAG_QDRANT_OVERSAMPLING env on the application side. "
            "Default 2.0 (pull 2× candidates, rescore top-N with fp32)."
        ),
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for Qdrant calls (default: 60s).",
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
        help="Actually update quantization_config on each collection. Opt-in.",
    )
    return p.parse_args(argv)


def _parse_collections(raw: Optional[str]) -> Optional[tuple[str, ...]]:
    if not raw:
        return None
    items = [c.strip() for c in raw.split(",") if c.strip()]
    return tuple(items) if items else None


def _should_skip(name: str, exclude: tuple[str, ...]) -> bool:
    if name in _DEFAULT_EXCLUDES:
        return True
    return name in exclude


async def _apply(args: argparse.Namespace) -> int:
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http import models as qm
    except ImportError as e:
        print(f"error: qdrant-client not installed: {e}", file=sys.stderr)
        return 2

    apply = bool(args.apply)
    banner = "APPLY" if apply else "DRY-RUN"
    exclude = tuple(args.exclude or ())
    targets_cli = _parse_collections(args.collections)
    quantile = float(args.quantile)
    if not 0.5 <= quantile <= 1.0:
        print(
            f"error: --quantile must be in [0.5, 1.0]; got {quantile!r}",
            file=sys.stderr,
        )
        return 4

    print(f"[{banner}] enable INT8 scalar quantization @ {args.qdrant_url}")
    print(f"  quantile={quantile} always_ram=True")
    print(f"  rescore oversampling={args.rescore} (advisory — set RAG_QDRANT_OVERSAMPLING)")
    if exclude:
        print(f"  excluded (in addition to defaults): {list(exclude)}")
    if targets_cli is not None:
        print(f"  targets (from --collections): {list(targets_cli)}")

    qdrant = AsyncQdrantClient(url=args.qdrant_url, timeout=args.timeout)
    try:
        if targets_cli is None:
            try:
                cols = [c.name for c in (await qdrant.get_collections()).collections]
            except Exception as e:
                print(f"error: failed to list collections: {e}", file=sys.stderr)
                return 1
            targets = [c for c in cols if not _should_skip(c, exclude)]
            skipped = [c for c in cols if _should_skip(c, exclude)]
            print(
                f"  found {len(cols)} collections "
                f"({len(targets)} targets, {len(skipped)} skipped)"
            )
            for name in skipped:
                print(f"    - skip {name}")
        else:
            # Explicit list: apply excludes on top of it so operators can
            # subtract a subset from the provided list.
            targets = [c for c in targets_cli if not _should_skip(c, exclude)]
            skipped = [c for c in targets_cli if _should_skip(c, exclude)]
            if skipped:
                for name in skipped:
                    print(f"    - skip {name}")

        for name in targets:
            print(f"    - target {name}")

        qc_plan = qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8,
                quantile=quantile,
                always_ram=True,
            )
        )

        if not apply:
            print(
                "  [dry-run] would call update_collection(..., quantization_config="
                f"ScalarQuantization(INT8, quantile={quantile}, always_ram=True)) "
                "per target"
            )
            print("  [dry-run] pass --apply to execute")
            return 0

        ok = 0
        errors = 0
        for name in targets:
            try:
                await qdrant.update_collection(
                    collection_name=name,
                    quantization_config=qc_plan,
                )
                ok += 1
                print(f"    {name} quantization set OK")
            except Exception as e:
                errors += 1
                print(f"    {name} quantization ERROR: {e}", file=sys.stderr)

        print(
            f"\n[done] applied to {ok} collections, {errors} errors. "
            "Qdrant builds the quantized index in the background — check "
            "GET /collections/{name} for config.quantization_config."
        )
        return 0 if errors == 0 else 1
    finally:
        await qdrant.close()


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    return asyncio.run(_apply(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
