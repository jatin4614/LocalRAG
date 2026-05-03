# compose/secrets/

Per-secret files for docker secrets (review §11.5). Mounted into the
relevant containers under `/run/secrets/<name>` by `compose/docker-
compose.yml`'s top-level `secrets:` block.

## Files (NEVER commit any of them — `.gitignore` enforces with `*`)

| File | Used by | Inside-container path | `_FILE` env var supported by service? |
|---|---|---|---|
| `postgres_password` | postgres, celery-worker | `/run/secrets/postgres_password` | YES — postgres reads `POSTGRES_PASSWORD_FILE` natively |
| `webui_secret_key` | open-webui | `/run/secrets/webui_secret_key` | NO — env-var fallback retained |
| `rag_admin_token` | open-webui, celery-worker | `/run/secrets/rag_admin_token` | NO — env-var fallback retained |
| `qdrant_api_key` | qdrant, open-webui, celery-worker | `/run/secrets/qdrant_api_key` | NO — env-var fallback retained |

For the three secrets without native `_FILE` support, the env vars in
`compose/.env` still drive the live config. The secret files are
mounted alongside so a future entrypoint wrapper can `cat` them at
container startup, or so an operator can `docker exec <container>
cat /run/secrets/<name>` during rotation without leaking through
`docker inspect`.

## Populating the dir (one-time)

```bash
cd compose/
mkdir -p secrets
echo -n "$(openssl rand -base64 24)"  > secrets/postgres_password
echo -n "$(openssl rand -base64 32)"  > secrets/webui_secret_key
echo -n "$(openssl rand -base64 32)"  > secrets/rag_admin_token
echo -n "$(openssl rand -base64 32)"  > secrets/qdrant_api_key
chmod 600 secrets/*
```

The trailing newline matters for some tools — note the `-n` flag on
`echo` keeps each file as a single line with no trailing `\n`.

If you're migrating an existing deployment, copy the values you
already have in `compose/.env`:

```bash
set -a; source compose/.env; set +a
echo -n "$POSTGRES_PASSWORD" > compose/secrets/postgres_password
echo -n "$WEBUI_SECRET_KEY"  > compose/secrets/webui_secret_key
echo -n "${RAG_ADMIN_TOKEN:-}" > compose/secrets/rag_admin_token
echo -n "${QDRANT_API_KEY:-}"  > compose/secrets/qdrant_api_key
chmod 600 compose/secrets/*
```

## Bootstrap behaviour

`scripts/bootstrap.sh` checks for `compose/secrets/` existence:

- **Dir exists**: bootstrap verifies each of the four files is
  non-empty (loose check; doesn't validate the value). If any are
  empty/missing, it warns and exits 1.
- **Dir missing**: bootstrap falls back to the legacy env-var-based
  rejection (still refuses `change-me-…` placeholders in
  `compose/.env`). Wave 1a's secret rejection is preserved.

The two paths are mutually exclusive on policy but co-exist on
config: an operator can run with both the env vars AND the secret
files set; whichever the container's entrypoint resolves first wins.

## Opt-out

To revert to env-var-only secrets:

1. Remove the top-level `secrets:` block in `compose/docker-compose.yml`.
2. Remove every per-service `secrets:` mapping (`grep -n "^    secrets:"`).
3. Remove `POSTGRES_PASSWORD_FILE` from the postgres service's `environment:`.

Or just don't create `compose/secrets/`; `bootstrap.sh` will use the
env-var path. `docker compose up` will fail until you remove the
`secrets:` blocks though, because docker resolves the file references
eagerly.

## Security notes

- The dir is `chmod 700` recommended on the host (`chmod 700
  compose/secrets`), with each file `chmod 600`.
- Backups: include `compose/secrets/` in your operator backup of
  `compose/.env` — losing both means a full secret regen + DB
  password rotation.
- Rotation: edit the file, then `docker compose up -d --force-
  recreate <service>`. For postgres password rotation you must also
  `ALTER USER orgchat WITH PASSWORD '...'` inside the DB and update
  any out-of-container clients (admin scripts) before the recreate.
- Never commit a populated file. The `.gitignore` here uses `*` to
  refuse everything but itself + this README.
