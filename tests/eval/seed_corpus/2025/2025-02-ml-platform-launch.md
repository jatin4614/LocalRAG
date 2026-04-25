# ML Platform v1 Launch Notes

**Date:** 2025-02-11
**Author:** ML Platform Team
**Doc ID:** 5001

## What Launched

- Self-serve training jobs on shared GPU pool (8×A100).
- Model registry with semantic versioning; first 12 teams onboarded.
- Automatic dataset snapshotting tied to experiment runs.

## Known Limitations

- No multi-node training yet; single-node only through 2025-Q2.
- Registry webhook fires at most once per minute — not suitable for
  sub-minute CI triggers.
- Budget dashboard updates with a 15-minute lag.

## Roadmap

Multi-node training and fine-grained quota enforcement are planned for the
2025-06 release.
