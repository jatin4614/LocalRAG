# Cold-start runbook

Walks through the first-time bring-up of LocalRAG on a fresh host.
Captures the order of operations, the one-shot host-side prep, and the
manual steps `scripts/bootstrap.sh` does not yet automate.

CLAUDE.md §8 has the abbreviated checklist; this file is the "why and
how to recover when a step fails" companion.

## Prereqs (host)

- Linux host with NVIDIA driver >= 580 (CUDA 12.9 capable), Docker 24+
  with the `nvidia` container runtime, and ~150 GB free under
  `/var/lib/docker` for image layers + model weights.
- Operator UID 1000 by convention. The compose stack runs every
  service as `1000:1000` (review §10.1 + §11.10) so files written
  to bind-mounted directories (`volumes/uploads`, `volumes/models`,
  `volumes/hf-cache`, `backups/`) show up owned by the host operator.
- HF Hub token at `~/.cache/huggingface/token` if any model in the
  preflight set is gated.

## Step 1 — clone + secrets

```bash
git clone <repo> LocalRAG && cd LocalRAG
cp compose/.env.example compose/.env
$EDITOR compose/.env  # fill in WEBUI_NAME, DOMAIN, ADMIN_*, *_SECRET_KEY, HF_TOKEN
```

The secrets that must NOT remain `change-me-...`:
`SESSION_SECRET`, `WEBUI_SECRET_KEY`, `ADMIN_PASSWORD`,
`POSTGRES_PASSWORD`, `DATABASE_URL` (rebuild with the new pg
password). `bootstrap.sh` rejects the placeholders with an explicit
error.

Optional but recommended for production: `QDRANT_API_KEY` (gen with
`openssl rand -base64 32`).

If `compose/secrets/` exists (Wave 1b §11.5 follow-up), populate the
per-secret files there instead of editing `.env` for the four
secrets that support `_FILE` (`POSTGRES_PASSWORD_FILE`,
`WEBUI_SECRET_KEY_FILE`, `RAG_ADMIN_TOKEN_FILE`,
`QDRANT_API_KEY_FILE`). See `compose/secrets/README.md`.

## Step 2 — host directory prep (one-shot)

The compose stack runs every service as UID 1000. Bind-mounted host
directories must be owned by that UID before the first `up`:

```bash
mkdir -p volumes/{models,uploads,hf-cache,certs} backups
sudo chown -R 1000:1000 volumes/ backups/
```

If the host operator account is already UID 1000 (most common),
`chown -R "$USER":"$USER" volumes/ backups/` is equivalent.

## Step 3 — shared volume UID alignment (one-shot, review §11.10)

The `ingest_blobs` named volume is shared between `open-webui`
(producer) and `celery-worker` (consumer). Docker creates it lazily
on first `up`; if the operator did NOT run as root before changing
`USER` in the Dockerfiles, the volume root may be `0:0` and the
non-root containers can't write/read it.

After the first `docker compose up -d` (or any time you suspect a
permission issue on `/var/ingest`):

```bash
docker compose -p orgchat run --rm --user root open-webui \
    chown -R 1000:1000 /var/ingest /app/backend/data/uploads \
                       /home/orgchat/.cache/huggingface
```

> **2026-05-03 path change.** The HF cache mount target moved from
> `/root/.cache/huggingface` to `/home/orgchat/.cache/huggingface` when
> the post-§10.1 image started shipping `USER 1000:1000` with
> `HOME=/home/orgchat`. `/root/` is mode 700 in the new image so the
> non-root user can't traverse it regardless of mount-point ownership.
> If you're upgrading an older deploy, also update the bind-mount targets
> in `compose/docker-compose.yml` (open-webui service) — search for
> `hf-cache` and `models--QuantTrio--gemma-4-31B-it-AWQ`.

Re-run the same command on celery-worker if the open-webui container
isn't healthy yet:

```bash
docker compose -p orgchat run --rm --user root celery-worker \
    chown -R 1000:1000 /var/ingest /opt/tiktoken-cache /opt/fastembed_cache
```

These are idempotent. Symptoms of skipping them:

- Upload returns `{queued}` but the doc never appears (worker
  rejects with `PermissionError: [Errno 13] /var/ingest/<sha>`).
- open-webui boot logs `PermissionError` reading
  `/app/backend/data/uploads`.
- celery-worker logs `PermissionError` writing the fastembed cache
  (only happens if the image cache wasn't preserved on a rebuild).

## Step 4 — bootstrap

```bash
scripts/bootstrap.sh
```

This:

1. Rejects placeholder secrets in `compose/.env`.
2. Preflights model weights (`scripts/preflight_models.py`).
3. Generates a self-signed cert (idempotent).
4. `docker compose up -d`.
5. Waits for postgres, applies migrations.
6. Seeds the admin user.

If `bootstrap.sh` aborts in Step 5 with `permission denied` on
postgres data dir, the postgres named volume was created with
non-root uid mapping mid-flight; clean it and retry:

```bash
docker compose -p orgchat down
docker volume rm orgchat_postgres_data orgchat_redis_data orgchat_qdrant_data
scripts/bootstrap.sh
```

## Step 5 — analyst config + first KB

```bash
.venv/bin/python scripts/apply_analyst_config.py  # idempotent
```

Then create the first KB via the admin UI at `https://${DOMAIN}/`.

## Step 6 — verify

```bash
docker compose -p orgchat ps                      # all containers Up + (healthy)
docker exec orgchat-open-webui id                 # uid=1000 gid=1000
docker exec orgchat-celery-worker id              # same
docker exec orgchat-vllm-chat nvidia-smi          # GPU 0 visible
docker exec orgchat-tei nvidia-smi                # GPU 1 visible
curl -s http://localhost:6333/cluster | jq .      # Qdrant cluster mode
curl -sf https://${DOMAIN}/health                 # 200 OK
```

If `docker exec orgchat-vllm-chat nvidia-smi` shows "command not
found" or "no devices", revert the USER directive in
`compose/Dockerfile.vllm-chat` (review §10.1) — the upstream image
may have a root-only init step we missed.

## Recovery

- Wipe and restart everything (DESTROYS DATA):
  ```bash
  docker compose -p orgchat down -v
  rm -rf volumes/uploads/*
  scripts/bootstrap.sh
  ```
- Single-service rebuild after Dockerfile change:
  ```bash
  docker compose -p orgchat build celery-worker
  docker compose -p orgchat up -d --force-recreate celery-worker
  ```
- Restore from snapshot: see `docs/runbook/backup-restore.md`.
