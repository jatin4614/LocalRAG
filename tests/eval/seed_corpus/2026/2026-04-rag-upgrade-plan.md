# RAG Pipeline Upgrade — Plan A

**Date:** 2026-04-24
**Author:** Retrieval Team
**Doc ID:** 6002

## Motivation

Current eval runs depend on a drifting live KB. Plan A seals a reproducible
seed corpus and adds an offline evaluation gate that blocks regressions in
NDCG@10 and grounded-answer rate.

## Phase 0 Deliverables

- SLO document committed 2026-04-24 with latency, cost, and quality budgets.
- Runbook skeleton covering flag reference and troubleshooting.
- Seed corpus fixtures organized by year bucket with deterministic doc IDs.

## Gating Criteria

A PR merges only when the eval harness reports no regression greater than
1 point on NDCG@10 and no drop in grounded-answer rate below 85%.
