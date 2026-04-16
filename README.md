# Org Chat Assistant

Self-hosted, air-gapped, multi-user ChatGPT-like web assistant with hierarchical Knowledge Bases. Hybrid thin-fork of Open WebUI.

See `CLAUDE.md` for the project summary and `docs/superpowers/plans/2026-04-16-org-chat-assistant-master-plan.md` for the implementation plan.

## Quick start (after Phase 1 complete)

```bash
cp compose/.env.example compose/.env
# edit compose/.env: set WEBUI_NAME, ADMIN_EMAIL, ADMIN_PASSWORD, SESSION_SECRET
make preflight   # downloads model weights once (needs internet)
make up          # brings up the stack
make smoke       # runs smoke test
```

## Repo layout

See the master plan's §3 Repo layout.
