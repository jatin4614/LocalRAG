# LocalRAG Runbook

Operational reference for the LocalRAG retrieval pipeline. Each phase of Plan A/B fills in its own sections.

## Contents
- [SLO document](slo.md) — latency / cost / error / quality budgets
- [Flag reference](flag-reference.md) — every RAG_* env flag, default, runtime-safe-to-toggle status
- [Plan B flag reference](plan-b-flag-reference.md) — Plan B Phase 4/5/6 additions and audit decisions
- [Troubleshooting](troubleshooting.md) — "if X is happening, check Y then Z"
- [Re-ingest checklist](reingest-checklist.md) / [Re-ingest procedure](reingest-procedure.md) — Plan A Phase 3.7
- [Temporal reshard checklist](temporal-reshard-checklist.md) / [Temporal reshard procedure](temporal-reshard-procedure.md) — Plan B Phase 5.4
- [Tiered storage runbook](tiered-storage-runbook.md) — Plan B Phase 5.8 daily tier movement cron
- [QU LLM runbook](qu-llm-runbook.md) — Plan B Phase 4 query-understanding LLM ops

## On-call first 5 minutes

1. Check Grafana dashboard "RAG overview" — red panels tell you the layer.
2. `curl http://localhost:6333/collections/kb_1_rebuild` — Qdrant up and collections present?
3. `docker logs --tail 200 orgchat-open-webui 2>&1 | grep -iE 'error|warn' | tail -50`
4. `nvidia-smi` — either GPU pegged at 100% util or 95%+ VRAM?
5. Escalate path: page RAG on-call via usual channel; include screenshots + timestamps.

## Open follow-ups

- Hand-label `tests/eval/golden_starter.jsonl` (60 queries, stratified per Plan A Task 0.3 Step 3 table).
- Commit `tests/unit/test_golden_starter_shape.py` alongside the labeled JSONL.
