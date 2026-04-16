# Org Chat Assistant — Deployment Runbook

This runbook covers everything a system administrator needs to deploy, operate, and maintain
the Org Chat Assistant on a single 32 GB RTX 6000 Ada GPU machine. It is written for Ubuntu
22.04; adjust package names as needed for other distributions.

---

## 1. Prerequisites

### 1.1 Hardware

| Resource | Minimum | Tested / Recommended |
|----------|---------|----------------------|
| GPU VRAM | 24 GB | 32 GB (RTX 6000 Ada 48 GB) |
| CPU cores | 8 | 16+ |
| System RAM | 32 GB | 64–128 GB |
| Disk (SSD) | 100 GB free | 500 GB (models ~40 GB + future data growth) |

The chat model (Qwen2.5-14B-AWQ, ~12 GB) and embedding service (bge-m3, ~3 GB) run
continuously. The vision model (Qwen2-VL-7B, ~8 GB) and Whisper STT (~4 GB) load
on-demand and unload after 5 minutes of inactivity. Peak VRAM usage is approximately
27 GB, leaving headroom for KV-cache batching.

### 1.2 Operating System

- Ubuntu 22.04 LTS (or equivalent systemd-based distro).
- Kernel version 5.15 or later (`uname -r` to check).
- Ensure the NVIDIA proprietary driver is installed and `nvidia-smi` shows the GPU.

### 1.3 Required Software

Install all of the following before running bootstrap:

```bash
# Docker Engine (24+) and Docker Compose v2
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER    # re-login after this

# NVIDIA Container Toolkit (GPU access from Docker)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Python 3.10+ with venv
sudo apt install -y python3 python3-venv python3-pip

# Supporting tools
sudo apt install -y openssl curl git make
```

Verify after installation:

```bash
docker --version             # Docker version 24.x or later
docker compose version       # Docker Compose version v2.x
nvidia-smi                   # GPU visible; driver version printed
python3 --version            # Python 3.10+
```

### 1.4 Network

- One-time outbound internet access is required to download model weights (~40 GB total).
- After bootstrap completes the system can be fully air-gapped; no external API calls
  are made at runtime.
- Internal DNS (or `/etc/hosts` entries on client machines) must resolve `<DOMAIN>` to
  the server's IP address.

---

## 2. First-run procedure

### 2.1 Clone the repository

```bash
git clone https://github.com/your-org/org-chat-assistant.git /opt/orgchat
cd /opt/orgchat
git submodule update --init --recursive   # pulls upstream Open WebUI
```

### 2.2 Create a Python virtual environment

```bash
make install
# Equivalent to: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

### 2.3 Configure environment variables

```bash
cp compose/.env.example compose/.env
$EDITOR compose/.env
```

Every key in `compose/.env` is documented below. **Do not leave any `change-me` value
in place before running bootstrap.**

| Key | Purpose | Notes |
|-----|---------|-------|
| `WEBUI_NAME` | Display name shown in the browser tab and header | e.g. `Acme Internal Assistant` |
| `DOMAIN` | LAN hostname (or FQDN) Caddy will serve on | e.g. `orgchat.acme.lan` |
| `SESSION_SECRET` | HMAC secret for session cookies | Generate with `openssl rand -hex 32` |
| `ADMIN_EMAIL` | Email address of the first admin account | Used by `seed_admin.py` |
| `ADMIN_PASSWORD` | Password for the first admin account | Minimum 12 characters; stored Argon2id-hashed |
| `POSTGRES_USER` | PostgreSQL role name | Default `orgchat`; change if your policy requires |
| `POSTGRES_PASSWORD` | PostgreSQL role password | Generate with `openssl rand -hex 24` |
| `POSTGRES_DB` | PostgreSQL database name | Default `orgchat` |
| `DATABASE_URL` | Full async DSN for SQLAlchemy | Update host/user/password to match the three keys above |
| `REDIS_URL` | Redis connection string | Default `redis://redis:6379/0`; change DB index if needed |
| `QDRANT_URL` | Qdrant HTTP endpoint | Default `http://qdrant:6333`; do not expose publicly |
| `CHAT_MODEL` | HuggingFace model ID for the chat LLM | Default `Qwen/Qwen2.5-14B-Instruct-AWQ` (12 GB) |
| `VISION_MODEL` | HuggingFace model ID for vision | Default `Qwen/Qwen2-VL-7B-Instruct` (8 GB, on-demand) |
| `EMBED_MODEL` | HuggingFace model ID for embeddings | Default `BAAI/bge-m3` (3 GB) |
| `WHISPER_MODEL` | Whisper model size | Default `medium` (4 GB, on-demand); use `small` to save VRAM |
| `ENABLE_SIGNUP` | Allow self-registration | Set `false`; admin creates users manually |
| `DEFAULT_USER_ROLE` | Role assigned to new accounts | Set `pending`; admin must approve |
| `ENABLE_OPENAI_API` | Enable the OpenAI-compatible chat endpoint | Keep `true` |
| `OPENAI_API_BASE_URL` | Internal vllm-chat base URL | Default `http://vllm-chat:8000/v1`; do not change |
| `OPENAI_API_KEY` | Dummy key accepted by vllm | Keep `sk-internal-dummy` or any non-empty string |
| `ENABLE_OLLAMA_API` | Ollama endpoint toggle | Keep `false` (not used) |
| `ENABLE_WEB_SEARCH` | Web search integration | Keep `false` for air-gapped deployments |
| `ENABLE_IMAGE_GENERATION` | Image generation toggle | Keep `false` |
| `ENABLE_RAG_WEB_LOADER` | Allow RAG to fetch web URLs | Keep `false` for air-gapped deployments |
| `AUDIO_STT_ENGINE` | Speech-to-text engine | Keep `whisper-local` |
| `RAG_EMBEDDING_ENGINE` | Embedding provider | Keep `openai` (TEI exposes an OpenAI-compatible API) |
| `RAG_EMBEDDING_OPENAI_API_BASE_URL` | TEI base URL | Default `http://tei:80/v1`; do not change |
| `RAG_EMBEDDING_MODEL` | Model name passed to TEI | Must match `EMBED_MODEL` |
| `VECTOR_DB` | Vector database backend | Keep `qdrant` |
| `MODEL_MANAGER_URL` | Internal model-manager endpoint | Default `http://model-manager:8080` |
| `MODEL_UNLOAD_IDLE_SECS` | Seconds of inactivity before unloading vision/whisper | Default `300` (5 min); lower to `120` on tight VRAM |
| `AUTH_MODE` | Authentication mode for the KB extension | `jwt` for production; `stub` only in test environments |
| `WEBUI_SECRET_KEY` | JWT signing key shared with upstream Open WebUI | Generate with `openssl rand -hex 32`; **must match** the value upstream uses |

### 2.4 Run bootstrap

```bash
source .venv/bin/activate
bash scripts/bootstrap.sh
```

`bootstrap.sh` executes these six steps in order:

1. **Preflight model weights** — runs `scripts/preflight_models.py`, which downloads all
   four model families to `volumes/models/`. This is the step that needs internet access
   (~40 GB). Pass `--skip-pull` if weights are already cached.
2. **Generate self-signed TLS certificate** — runs `scripts/gen_self_signed_cert.sh`,
   writing `volumes/certs/orgchat.crt` and `orgchat.key`. Idempotent: skips if the cert
   already exists and is valid.
3. **Apply upstream patches** — runs `scripts/apply_patches.sh`, which replays the patches
   in `patches/` on top of the Open WebUI submodule. Required before building the Docker
   image.
4. **Start all containers** — runs `docker compose -f compose/docker-compose.yml up -d`.
   All nine services start; vllm-chat and TEI perform their GPU warm-up.
5. **Apply database migrations** — waits for PostgreSQL to be ready, then runs
   `scripts/apply_migrations.py`, which executes `backend/migrations/001_create_kb_schema.sql`
   and any subsequent migration files in order.
6. **Seed the admin account** — runs `scripts/seed_admin.py` using `ADMIN_EMAIL` and
   `ADMIN_PASSWORD` from `compose/.env`. Idempotent: does nothing if the account already
   exists.

Bootstrap typically takes 20–60 minutes on first run (dominated by model download). Subsequent
restarts take under 2 minutes.

### 2.5 Add the hostname to client machines

On each machine that will access the assistant, add an entry to `/etc/hosts`:

```
<SERVER_IP>   orgchat.lan
```

Replace `orgchat.lan` with the value of `DOMAIN` in your `compose/.env`. On macOS/Windows,
the path is `/etc/hosts` and `C:\Windows\System32\drivers\etc\hosts` respectively.

### 2.6 First login

1. Open `https://<DOMAIN>/` in a browser. Caddy uses the self-signed certificate generated
   in step 2.4, so the browser will display a security warning on first access.
2. Click through the browser's "Advanced → Proceed" flow to accept the certificate. To
   suppress the warning permanently, import `volumes/certs/orgchat.crt` into the operating
   system or browser certificate trust store.
3. Log in with the `ADMIN_EMAIL` / `ADMIN_PASSWORD` credentials.
4. Navigate to **Admin Panel → Users** to create additional accounts.

---

## 3. Daily operations

### 3.1 Starting and stopping

```bash
# Start (idempotent — safe to run if already running)
make up
# Equivalent: docker compose -f compose/docker-compose.yml --env-file compose/.env up -d

# Stop (containers down, volumes preserved)
make down

# Tail logs from all services
make logs

# Tail a specific service
docker compose -f compose/docker-compose.yml logs -f vllm-chat
```

### 3.2 Checking service health

```bash
# Overview of all containers
docker compose -f compose/docker-compose.yml ps
# All services should show Status "running" and Health "healthy".

# Individual health checks
curl -sf http://localhost:8080/healthz          # model-manager
docker exec orgchat-postgres pg_isready -U orgchat
curl -sf http://localhost:6333/readyz           # qdrant
curl -sf http://localhost:8001/health           # tei (embeddings)
```

vllm-chat can take 3–8 minutes to reach `healthy` on first start while it loads model weights
into VRAM. This is normal.

### 3.3 Running the test suite

`make test-all` uses testcontainers and spins up ephemeral services in isolation; it does
**not** connect to the production stack. Run it before any deployment:

```bash
source .venv/bin/activate
make test-all
```

---

## 4. Admin tasks

### 4.1 Creating a new user

Users cannot self-register (`ENABLE_SIGNUP=false`). The admin creates accounts in the web UI:

1. Open **Admin Panel → Users → Add User**.
2. Fill in name, email, and a temporary password.
3. The account starts in `pending` status; click **Approve** to activate it.
4. Communicate the temporary password to the user out-of-band; they should change it on
   first login.

### 4.2 Creating a Knowledge Base

Knowledge Bases (KBs) are org-wide collections that admins manage and users query.

```bash
curl -s -X POST https://<DOMAIN>/api/kb \
  -H "Authorization: Bearer <admin_jwt>" \
  -H "Content-Type: application/json" \
  -d '{"name":"Engineering","description":"Internal engineering docs and RFCs"}'
```

Response includes the new `kb_id`. Only users with the `admin` role can call this endpoint.

### 4.3 Creating a subtag inside a KB

Subtags provide a second level of organisation within a KB (e.g., "OFC", "Roadmap", "ADRs").

```bash
curl -s -X POST https://<DOMAIN>/api/kb/<kb_id>/subtags \
  -H "Authorization: Bearer <admin_jwt>" \
  -H "Content-Type: application/json" \
  -d '{"name":"OFC"}'
```

### 4.4 Granting KB access to a group

```bash
curl -s -X POST https://<DOMAIN>/api/kb/<kb_id>/access \
  -H "Authorization: Bearer <admin_jwt>" \
  -H "Content-Type: application/json" \
  -d '{"group_id": 3, "access_type": "read"}'
```

`access_type` is either `"read"` or `"write"`. Users who are members of the group gain
access to the KB immediately; the grant is cached in Redis and expires within 60 seconds.

### 4.5 Uploading a document

```bash
curl -s -X POST https://<DOMAIN>/api/kb/<kb_id>/subtag/<subtag_id>/upload \
  -H "Authorization: Bearer <admin_jwt>" \
  -F "file=@/path/to/document.pdf"
```

Supported file types (Phase 4): `txt`, `md`, `pdf`, `docx`, `xlsx`. Files are chunked into
800-token segments (100-token overlap), embedded, and stored in Qdrant. Check ingest status:

```bash
curl -s https://<DOMAIN>/api/kb/<kb_id>/documents \
  -H "Authorization: Bearer <admin_jwt>"
# Look for "status": "ready" on each document.
```

### 4.6 Viewing the audit log

Structured per-request audit entries are currently written to the container log stream.
Retrieve them with:

```bash
docker compose -f compose/docker-compose.yml logs open-webui | grep '"event":"audit"'
```

A dedicated audit table in PostgreSQL is planned for a future phase.

---

## 5. Troubleshooting

### 5.1 `orgchat-vllm-chat` stuck in `health: starting` beyond 10 minutes

**Symptom:** `docker compose ps` shows `vllm-chat` health as `starting` or `unhealthy`
after more than 10 minutes.

**Cause:** The GPU is not visible inside the container. This can happen if the NVIDIA
Container Toolkit was installed after Docker was last started, or if the user running
Docker is missing the `docker` group membership.

**Fix:**
```bash
# Verify GPU visibility inside the container
docker exec orgchat-vllm-chat nvidia-smi

# If that fails:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker compose -f compose/docker-compose.yml restart vllm-chat
```

Also ensure the NVIDIA driver is loaded: `sudo modprobe nvidia` and check `lsmod | grep nvidia`.

---

### 5.2 `403 Forbidden` on `POST /api/kb`

**Symptom:** Admin API calls to create or manage KBs return `403`.

**Cause:** The authenticated user's role is not `admin` in the database.

**Fix:**
```bash
docker exec -it orgchat-postgres psql -U orgchat orgchat \
  -c "SELECT id, email, role FROM users WHERE email='your@email.com';"

# If role is 'user' or 'pending':
docker exec -it orgchat-postgres psql -U orgchat orgchat \
  -c "UPDATE users SET role='admin' WHERE email='your@email.com';"
```

---

### 5.3 `401 Unauthorized` on `GET /api/kb/available`

**Symptom:** The KB selection endpoint returns `401` even with a valid JWT from the login
flow.

**Cause:** `WEBUI_SECRET_KEY` in `compose/.env` does not match the value used by the Open
WebUI container to sign JWTs. This mismatch arises when the env file is edited after first
boot and the container is not restarted.

**Fix:**
```bash
# Confirm the mismatch
docker exec orgchat-open-webui env | grep WEBUI_SECRET_KEY

# Then ensure compose/.env has the same value and restart
docker compose -f compose/docker-compose.yml --env-file compose/.env \
  restart open-webui model-manager
```

---

### 5.4 File upload returns `422 Unprocessable Entity`

**Symptom:** Uploading a document returns `422`.

**Cause:** The file's extension or MIME type is not supported. Phase 4 accepts only: `.txt`,
`.md`, `.pdf`, `.docx`, `.xlsx`. Files with no extension, `.html`, `.pptx`, etc. are rejected.

**Fix:** Convert the file to a supported format before uploading. For HTML pages, copy the
text content into a `.md` file.

---

### 5.5 File upload returns `413 Request Entity Too Large`

**Symptom:** Uploading returns `413`.

**Cause:** The file exceeds `RAG_MAX_UPLOAD_BYTES` (default 50 MB). Large PDFs with
embedded images are a common trigger.

**Fix:**
```bash
# Temporary: increase the limit in compose/.env
RAG_MAX_UPLOAD_BYTES=104857600   # 100 MB

# Then restart the open-webui service
docker compose -f compose/docker-compose.yml --env-file compose/.env \
  restart open-webui

# Permanent fix: compress or split the document before uploading.
```

---

### 5.6 Port 443 already in use

**Symptom:** `docker compose up` fails with `address already in use` on port 443.

**Cause:** Another service (nginx, apache, another Caddy instance) is bound to port 443.

**Fix:**
```bash
# Find the conflicting process
ss -tlnp | grep ':443'

# Option A: stop the conflicting service
sudo systemctl stop nginx

# Option B: change Caddy to listen on a different port
# Edit compose/docker-compose.yml, change "443:443" to "8443:443"
# Then access via https://<DOMAIN>:8443/
```

---

### 5.7 Qdrant returns `404` on search

**Symptom:** Chat returns no RAG results; `docker compose logs open-webui` shows Qdrant
404 errors for a specific collection.

**Cause:** The Qdrant collection for that KB does not exist yet. Collections are created
automatically on the first document upload. A KB with no uploaded documents has no
collection.

**Fix:**
```bash
# List existing collections
curl -s http://localhost:6333/collections | python3 -m json.tool

# Upload at least one document to the KB to create its collection.
# If the collection should already exist, check that qdrant_data volume is intact:
docker volume inspect orgchat_qdrant_data
```

---

### 5.8 Chat hangs on first image or audio message

**Symptom:** Sending an image or audio file results in a spinner that lasts 2–10 seconds
before the response streams.

**Cause:** This is expected behaviour. The vision model (vllm-vision) and Whisper are
loaded on-demand when the first such message arrives, and unloaded after `MODEL_UNLOAD_IDLE_SECS`
of inactivity. The delay is the model warm-up time.

**Monitor the loading process:**
```bash
curl -s http://localhost:8080/models/status | python3 -m json.tool
# Watch the "state" field transition: "unloaded" → "loading" → "ready"
```

If the model never reaches `ready`, check GPU VRAM headroom:
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
# Need at least 8 GB free for vision, 4 GB for whisper.
```

Reduce `MODEL_UNLOAD_IDLE_SECS` to `60` if VRAM pressure is consistently too high.

---

### 5.9 "Peewee migration failed" on first boot

**Symptom:** `docker compose logs open-webui` shows a Peewee-related traceback during startup.

**Cause:** Upstream Open WebUI uses Peewee for its own internal schema bootstrap. Our
migrations run after upstream's migration completes. If upstream can't initialise its schema
(usually a database connection problem), our migrations never run.

**Fix:**
```bash
# Check that postgres is healthy first
docker exec orgchat-postgres pg_isready -U orgchat

# Check connection from open-webui perspective
docker exec orgchat-open-webui env | grep DATABASE_URL

# If the URL is wrong, fix DATABASE_URL in compose/.env and run:
docker compose -f compose/docker-compose.yml --env-file compose/.env \
  restart open-webui
```

---

### 5.10 TLS certificate expired or untrusted

**Symptom:** Browsers show "Your connection is not private" with `NET::ERR_CERT_DATE_INVALID`.

**Cause:** The self-signed certificate generated by bootstrap has expired (default lifetime
is 365 days) or was never imported into client trust stores.

**Fix:**
```bash
# Regenerate with a longer lifetime
CERT_DAYS=3650 bash scripts/gen_self_signed_cert.sh

# Restart Caddy to pick up the new cert
docker compose -f compose/docker-compose.yml restart caddy

# To permanently trust the cert on Ubuntu clients:
sudo cp volumes/certs/orgchat.crt /usr/local/share/ca-certificates/orgchat.crt
sudo update-ca-certificates
```

On macOS: double-click the `.crt` file, add to System keychain, set to "Always Trust".
On Windows: `certutil -addstore Root volumes\certs\orgchat.crt`.

---

## 6. Upgrading Open WebUI

This project is a thin fork of Open WebUI. Upstream releases are tracked via a Git
submodule in `upstream/` and a small set of patches in `patches/`.

### 6.1 Check the new upstream tag

Before upgrading, review the upstream changelog on GitHub to identify breaking changes,
especially to the auth system, database schema, or API contracts.

### 6.2 Run the rebase script

```bash
source .venv/bin/activate
bash scripts/rebase_upstream.sh v0.9.0   # replace with actual new tag
```

The script:
1. Updates the `upstream/` submodule to the specified tag.
2. Re-applies each patch in `patches/` in order.
3. Prints a summary of which patches applied cleanly.

### 6.3 Resolve patch conflicts

If a patch does not apply cleanly, the script prints the conflicted file path and exits.
Resolve manually:

```bash
# 1. Open the conflicted file and fix by hand
$EDITOR upstream/open_webui/some_file.py

# 2. Regenerate the patch from the fixed state
git -C upstream diff HEAD~1 HEAD -- some_file.py \
  > patches/0001-mount-ext-routers.patch

# 3. Re-run the rebase script to verify all patches apply
bash scripts/rebase_upstream.sh v0.9.0
```

### 6.4 Run the test suite

```bash
make test-all
# All tests must pass before proceeding.
```

### 6.5 Deploy the upgraded version

```bash
docker compose -f compose/docker-compose.yml --env-file compose/.env down
docker compose -f compose/docker-compose.yml --env-file compose/.env \
  build open-webui          # rebuild with patched upstream
docker compose -f compose/docker-compose.yml --env-file compose/.env up -d
```

### 6.6 Verify the upgrade

```bash
docker compose -f compose/docker-compose.yml logs open-webui | head -50
# Look for successful migration messages and no Peewee errors.
curl -sf https://<DOMAIN>/health
```

---

## 7. Backup & restore

### 7.1 PostgreSQL

**Backup:**
```bash
docker exec orgchat-postgres pg_dump -U orgchat orgchat \
  > backup-$(date +%F).sql
```

**Restore to a fresh instance:**
```bash
# Bring up only postgres
docker compose -f compose/docker-compose.yml up -d postgres
# Wait for healthy, then restore
docker exec -i orgchat-postgres psql -U orgchat orgchat < backup-2026-04-16.sql
```

### 7.2 Qdrant vector collections

Qdrant supports snapshot-based backups via its REST API.

**Backup (one collection):**
```bash
KB_ID=5
curl -X POST "http://localhost:6333/collections/kb_${KB_ID}/snapshots"
# The response contains the snapshot name.
# Snapshots land in the qdrant data volume:
ls $(docker volume inspect orgchat_qdrant_data --format '{{.Mountpoint}}')/snapshots/
```

**Restore:**
```bash
curl -X PUT "http://localhost:6333/collections/kb_${KB_ID}/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location":"file:///qdrant/snapshots/<snapshot-name>.snapshot"}'
```

### 7.3 Model weights

Model weights are stored in `volumes/models/` (~40 GB). They are reproducible from
HuggingFace but slow to re-download, so include them in disaster-recovery backups:

```bash
rsync -av --progress volumes/models/ backup-host:/backups/orgchat/models/
```

### 7.4 Raw uploaded documents

In Phase 4 the system does not persist raw uploaded files to disk — only their chunked
vector representations in Qdrant. If the original documents are needed for audit or
re-ingestion purposes, store them separately before uploading.

### 7.5 Full system backup

A complete backup of the persistent state is all data under `volumes/`:

```bash
docker compose -f compose/docker-compose.yml down
tar -czf orgchat-backup-$(date +%F).tar.gz volumes/
docker compose -f compose/docker-compose.yml up -d
```

Contents:
- `volumes/postgres_data/` — all user, chat, KB, and document records
- `volumes/redis_data/` — session cache (can be cleared safely; sessions will expire)
- `volumes/qdrant_data/` — all vector embeddings
- `volumes/models/` — model weights (~40 GB)
- `volumes/certs/` — TLS certificate and private key

---

## 8. Phase 2 — k8s migration (future)

This section describes planned work for Phase 2 and is provided as a reference for
infrastructure planning. None of it is required for Phase 1 operation.

### 8.1 Target environment

- 4 × NVIDIA 48 GB GPUs (e.g., L40S or A6000 Ada)
- 20–200 concurrent users
- Kubernetes 1.28+ (managed cluster or bare-metal kubeadm)

### 8.2 Key architectural differences from Phase 1

| Aspect | Phase 1 (Docker Compose) | Phase 2 (Kubernetes) |
|--------|--------------------------|----------------------|
| Chat model | Qwen2.5-14B-AWQ (1 GPU) | Qwen2.5-72B (tensor-parallel across 2 GPUs) |
| Chat deployment | Single container | StatefulSet with PodDisruptionBudget |
| Open WebUI | Single replica | Deployment + HPA (scale on CPU/latency) |
| Ingress | Caddy (self-managed cert) | cert-manager + Let's Encrypt or internal CA |
| Secrets | `compose/.env` | Kubernetes Secrets + external-secrets operator |
| Storage | Docker named volumes | PersistentVolumeClaims (ReadWriteOnce) |

### 8.3 Migration path

1. Export all data using the backup procedures in Section 7.
2. Provision the Kubernetes cluster with NVIDIA GPU operator installed.
3. Apply manifests from `k8s/` (Phase 2 skeleton — not yet production-wired).
4. Restore PostgreSQL and Qdrant snapshots into the new PVCs.
5. Validate isolation and RBAC tests against the new cluster (`make test-all`).
6. Perform a blue-green cutover by updating DNS to point `DOMAIN` to the new cluster
   ingress IP.

### 8.4 Out of scope

Full Kubernetes manifest authoring, Helm chart development, and multi-GPU tensor-parallel
tuning are out of scope for Phase 7. This section exists to ensure the Docker Compose
deployment does not inadvertently create migration blockers (e.g., hardcoded hostnames,
non-portable volume layouts).
