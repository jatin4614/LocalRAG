# Re-ingest Checklist (operator-facing)

Use this checklist in the off-hours window to execute Task 3.7.

**Pre-requisites verified:**

- [ ] Plan A Phase 3.1–3.6 merged to main.
- [ ] Appendix A staging complete; `docker exec orgchat-open-webui ls /models/fastembed_cache/colbert*` returns a directory.
- [ ] Current eval baseline committed at `tests/eval/results/phase-0-baseline.json`.
- [ ] Phase 1.7 schema reconciliation complete for the target KB.
- [ ] Off-peak window is active — confirm via Grafana dashboard that current chat QPS is low.
- [ ] `nvidia-smi` on the host shows GPU 0 VRAM < 90% (base load only).

**Execution:**

- [ ] Step 1 — Snapshot source collection (per docs/runbook/reingest-procedure.md §1).
- [ ] Step 2 — Create target collection (§2).
- [ ] Step 3 — Enable per-KB contextualize + colbert in `rag_config` (§3).
- [ ] Step 4 — Export env for this session (§4).
- [ ] Step 5 — Run `scripts/reingest_kb.py` (§5). Monitor `ChatLatencyDuringIngest` alert in Grafana.
- [ ] Step 6 — Verify point counts match (§6).
- [ ] Step 7 — Run eval against the new collection (§7). Confirm gate passes.
- [ ] Step 8 — Swap alias (§8).
- [ ] Step 9 — Spot-check a live retrieval (§9).
- [ ] Step 10 — Mark source read-only for 14 days (§10).

**Post-window:**

- [ ] Commit the rag_config change (so it persists through restart).
- [ ] Update `docs/runbook/flag-reference.md` with the KB id and its enabled features.
- [ ] Announce completion to the team; set a calendar reminder for Day 14 to drop the rollback collection.

**If any step fails:** follow Rollback in `reingest-procedure.md`. Log the failure mode in `docs/runbook/troubleshooting.md` under "Re-ingest issues."
