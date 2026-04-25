# OrgChat Observability Stack — Operator Runbook

> **READ FIRST — Claude / automated-agent safety rule**
>
> This runbook is **documentation only**. **No command in this file may be executed by Claude, an agent, or any automation without explicit user go-ahead for that specific command.** Every `docker`, `curl`, `ss`, `bash`, or `nvidia-smi` invocation below is tagged with one of:
>
> - `RUN THIS WHEN READY — read-only` — safe, no side effects, still requires user OK.
> - `RUN THIS WHEN READY — MAINTENANCE ACTION` — mutates container/network state; requires user OK.
> - `RUN THIS WHEN READY — MAINTENANCE ACTION (requires maintenance window)` — may briefly affect a live OrgChat path; schedule it.
>
> **Nothing in this runbook restarts, rebuilds, or otherwise touches the existing `orgchat-*` or `frams-*` containers**, with the single exception of Phase F (which restarts only `orgchat-open-webui` and, if present, `orgchat-celery-worker`, and is explicitly called out as a maintenance window).

---

## 1. Overview

This is a **side-car observability stack** for OrgChat. It runs as an independent Docker Compose project in `/home/vogic/LocalRAG/observability/` and is wired into the existing `orgchat-net` bridge (read-only scrape + log tail) and a new `orgchat-obs-net` bridge (for OTLP ingress from instrumented app code). It is **separate from the FRAMS observability stack** (different ports, different compose project, different volumes) so that either can be taken down without affecting the other. It instruments: host (node-exporter), containers (cAdvisor), GPU (DCGM), Postgres, Redis, Prometheus metrics scraped from `orgchat-open-webui` and `orgchat-model-manager`, logs (Loki/Promtail over the Docker socket), and traces (OTLP → OTel Collector → Jaeger) emitted by the OrgChat backend when `OBS_ENABLED=true`. Until that flag flips, instrumentation code is dormant and the app path is unchanged.

Design reference: `docs/superpowers/specs/2026-04-21-orgchat-observability-design.md`.

---

## 2. Prerequisites check

Run these in order. **All are read-only.** Do not proceed to bring-up if any fails.

```bash
# RUN THIS WHEN READY — read-only
# 1. Confirm orgchat-* containers are healthy (we will not touch them).
docker ps --format '{{.Names}} {{.Status}}' | grep '^orgchat-'
```

```bash
# RUN THIS WHEN READY — read-only
# 2. Confirm every port we plan to bind is FREE. Output should be empty.
ss -tlnp | grep -E ':(9091|3002|3101|16687|4319|4320|9101|8081|9401|9187|9122)\b'
```

```bash
# RUN THIS WHEN READY — read-only
# 3. Confirm the external orgchat-net exists (we attach to it as external).
docker network inspect orgchat-net >/dev/null && echo OK || echo MISSING
```

```bash
# RUN THIS WHEN READY — read-only
# 4. Confirm GPU is visible for DCGM.
nvidia-smi
```

Also confirm (read-only, visual): `.env` in this directory contains `GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD`, `POSTGRES_EXPORTER_DSN`, `REDIS_EXPORTER_ADDR`. Do **not** commit the file.

---

## 3. Bring-up order (6 phases)

Each phase is scoped to specific services via `docker compose -f observability/docker-compose.yml up -d <services…>`. **Never** run a bare `docker compose up -d` from this file — it would start everything at once and skip the phase gates.

All commands assume `cwd = /home/vogic/LocalRAG`.

### Phase A — Exporters + Prometheus

```bash
# RUN THIS WHEN READY — MAINTENANCE ACTION
docker compose -f observability/docker-compose.yml up -d \
  node-exporter cadvisor dcgm-exporter postgres-exporter redis-exporter prometheus
```

Verify:

```bash
# RUN THIS WHEN READY — read-only
curl -s http://127.0.0.1:9091/api/v1/targets | jq '.data.activeTargets | length'
```

Open http://127.0.0.1:9091/targets and confirm each target is `UP`. Expected target count: 13 (prometheus self + 5 exporters + orgchat scrape jobs).

**Success:** Prometheus UI loads; no target in `down` state for > 60s.

### Phase B — Loki + Promtail

```bash
# RUN THIS WHEN READY — MAINTENANCE ACTION
docker compose -f observability/docker-compose.yml up -d loki promtail
```

Verify:

```bash
# RUN THIS WHEN READY — read-only
curl -s http://127.0.0.1:3101/ready
docker logs --tail=50 obs-promtail | grep -i 'discovered\|orgchat-'
```

**Success:** `/ready` returns `ready`; Promtail logs show orgchat containers discovered via the Docker service-discovery source.

### Phase C — Jaeger + OTel Collector

```bash
# RUN THIS WHEN READY — MAINTENANCE ACTION
docker compose -f observability/docker-compose.yml up -d jaeger otelcol
```

Verify:

```bash
# RUN THIS WHEN READY — read-only
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:16687/
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:4320/v1/traces \
  -H 'Content-Type: application/json' --data '{}'
```

**Success:** Jaeger UI returns 200; OTLP HTTP endpoint responds (any 2xx/4xx is fine — we are just proving the port is live; 415/400 on an empty body is expected).

### Phase D — Grafana

```bash
# RUN THIS WHEN READY — MAINTENANCE ACTION
docker compose -f observability/docker-compose.yml up -d grafana
```

Verify: open http://127.0.0.1:3002, log in with admin creds from `.env`.

- **Connections → Data sources:** `Prometheus`, `Loki`, `Jaeger` should all be present and show "working".
- **Dashboards → OrgChat folder:** exactly 7 dashboards auto-provisioned:
  1. System Overview
  2. Pipeline E2E
  3. RAG Retrieval
  4. Vector Store (Qdrant)
  5. Model Manager / VRAM
  6. Logs Explorer
  7. Traces Explorer

**Success:** all 7 present, datasources green, no provisioning errors in `docker logs obs-grafana`.

### Phase E — Attach networks (no restart)

We need obs-net reachable from orgchat containers so Prometheus can scrape `orgchat-open-webui:9090/metrics` and the app can reach `otelcol:4319`. `docker network connect` attaches a second NIC to a running container — **it does not restart the container**.

```bash
# RUN THIS WHEN READY — MAINTENANCE ACTION
for c in orgchat-open-webui orgchat-vllm-chat orgchat-tei orgchat-qdrant \
         orgchat-model-manager orgchat-postgres orgchat-redis \
         orgchat-whisper kairos-tts; do
  docker network connect orgchat-obs-net "$c" 2>&1 | grep -v "already exists" || true
done
```

Verify (read-only) that each container got a second network and none were restarted — `RestartCount` and `StartedAt` must be unchanged from before the attach:

```bash
# RUN THIS WHEN READY — read-only
for c in orgchat-open-webui orgchat-vllm-chat orgchat-tei orgchat-qdrant \
         orgchat-model-manager orgchat-postgres orgchat-redis \
         orgchat-whisper kairos-tts; do
  docker inspect -f '{{.Name}} networks={{range $k,$_ := .NetworkSettings.Networks}}{{$k}} {{end}}restarts={{.RestartCount}} started={{.State.StartedAt}}' "$c"
done
```

**Success:** each line shows both `orgchat-net` and `orgchat-obs-net`; `restarts` and `started` match the pre-attach snapshot.

### Phase F — Enable instrumentation (MAINTENANCE WINDOW REQUIRED)

This is the only phase that restarts an OrgChat container. Schedule a window. All prior phases are observation-only.

1. Install OTel extras into `orgchat-open-webui` **or** rebuild its image with the `observability` extra (preferred):
   - Rebuild path: bump `ext/requirements.txt` to include the `observability` extra, rebuild image, push to local registry.
   - In-place path (short-term only): `docker exec orgchat-open-webui pip install --no-deps <pinned otel packages>` — this is documented in the design doc §7.
2. Set `OBS_ENABLED=true` and `OTEL_EXPORTER_OTLP_ENDPOINT=http://otelcol:4319` in the env file consumed by `orgchat-open-webui`.
3. Restart ONLY the instrumented service(s):

```bash
# RUN THIS WHEN READY — MAINTENANCE ACTION (requires maintenance window)
docker compose -f compose/docker-compose.yml up -d --no-deps orgchat-open-webui
# If a celery worker exists in your deployment:
docker compose -f compose/docker-compose.yml up -d --no-deps orgchat-celery-worker
```

**Do not** run `docker compose up -d` without `--no-deps` from the OrgChat compose file — it will recreate every service.

**Success:** `docker logs orgchat-open-webui | grep -i 'OTEL\|observability'` shows the tracer initialized; Prometheus target `orgchat-open-webui` flips to `UP`.

---

## 4. Post-bring-up verification checklist

Run through all 10 items. Each should be green before declaring the stack live.

1. `docker ps --filter name=obs-` shows **11 containers**, all `healthy` (or `running` for those without a healthcheck).
2. Prometheus → Status → Targets: **13 targets, all UP**, no scrape errors > 1 min old.
3. node-exporter: Grafana → System Overview shows host CPU %, RAM %, disk % for the OrgChat host.
4. DCGM: Grafana → Model Manager / VRAM shows non-zero `DCGM_FI_DEV_GPU_UTIL` and memory-used gauges.
5. cAdvisor: Grafana → System Overview container panel lists each `orgchat-*` container with CPU/RAM series.
6. Grafana **System Overview** dashboard: no "No data" panels.
7. Loki: Grafana → Logs Explorer, query `{container=~"orgchat-.*"}` — recent log lines stream in.
8. Send **one chat message** through OrgChat, then open Jaeger (http://127.0.0.1:16687) and search service `orgchat-ext`. The trace must contain **≥ 12 spans** covering: HTTP in → auth → KB resolve → embed → qdrant search → rerank → prompt build → vllm call → stream out.
9. Grafana **Pipeline E2E** dashboard: stage-latency breakdown panel shows non-empty bars for embed / retrieve / rerank / generate.
10. Prometheus query `rag_tokens_prompt_total` and `rag_tokens_completion_total` are both > 0 after that one chat.

---

## 5. Rollback (ordered, zero-disruption)

Execute top-to-bottom. Each step is independently safe.

1. **Make instrumentation dormant** (undo Phase F without removing anything):
   ```bash
   # RUN THIS WHEN READY — MAINTENANCE ACTION (requires maintenance window)
   # Set OBS_ENABLED=false in the orgchat-open-webui env, then:
   docker compose -f compose/docker-compose.yml up -d --no-deps orgchat-open-webui
   ```
   Tracer init is skipped; no OTLP egress; metrics endpoint still serves but nobody has to scrape it.

2. **Detach obs-net from orgchat containers** (undoes Phase E, no restart):
   ```bash
   # RUN THIS WHEN READY — MAINTENANCE ACTION
   for c in orgchat-open-webui orgchat-vllm-chat orgchat-tei orgchat-qdrant \
            orgchat-model-manager orgchat-postgres orgchat-redis \
            orgchat-whisper kairos-tts; do
     docker network disconnect orgchat-obs-net "$c" 2>/dev/null || true
   done
   ```

3. **Tear down the obs stack** (undoes Phases A–D):
   ```bash
   # RUN THIS WHEN READY — MAINTENANCE ACTION
   docker compose -f observability/docker-compose.yml down
   ```
   Volumes are preserved under `/home/vogic/LocalRAG/volumes/observability/` (Prometheus TSDB, Loki chunks, Grafana state). Bring-up can resume from Phase A with history intact.

4. **Optional: discard history** (destructive):
   ```bash
   # RUN THIS WHEN READY — MAINTENANCE ACTION (DESTRUCTIVE — deletes metrics/log/trace history)
   rm -rf /home/vogic/LocalRAG/volumes/observability/
   ```

---

## 6. Port reference

All bound to `127.0.0.1` only. Remote access via SSH tunnel (§7).

| Service           | Host port | Notes                                 | FRAMS uses        |
|-------------------|-----------|---------------------------------------|-------------------|
| Prometheus UI     | 9091      | `/targets`, `/graph`                  | 9090              |
| Grafana UI        | 3002      | admin from `.env`                     | 3001              |
| Loki HTTP         | 3101      | `/ready`, push/query                  | 3100              |
| Jaeger UI         | 16687     | search service `orgchat-ext`          | —                 |
| OTel Collector    | 4319      | OTLP gRPC                             | —                 |
| OTel Collector    | 4320      | OTLP HTTP                             | —                 |
| node-exporter     | 9101      | host metrics                          | 9100              |
| cAdvisor          | 8081      | container metrics                     | 8080              |
| DCGM exporter     | 9401      | GPU metrics                           | 9400              |
| postgres-exporter | 9187      | OrgChat Postgres                      | —                 |
| redis-exporter    | 9122      | OrgChat Redis                         | 9121              |

---

## 7. Access model

- **All obs UIs are bound to `127.0.0.1`.** Never publish them on a public interface — they are unauthenticated or weakly authenticated by default.
- Remote access is via SSH local-forward. Example for Grafana:
  ```bash
  # RUN THIS WHEN READY — read-only (client-side)
  ssh -L 3002:127.0.0.1:3002 user@orgchat-host
  ```
  Repeat with the appropriate local port for Prometheus (9091), Jaeger (16687), Loki (3101).
- Grafana admin credentials come from `observability/.env` (`GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD`). Rotate after first login.
- Anonymous viewer is disabled in provisioning. All other UIs (Prometheus, Jaeger, Loki API) have **no auth** — the localhost bind is the only access control. Do not expose via reverse proxy without adding auth.

---

## 8. Troubleshooting

**1. Prometheus target DOWN for an `orgchat-*` service.**
Cause: container is not attached to `orgchat-obs-net`. Fix: re-run Phase E for that container; verify with `docker inspect -f '{{json .NetworkSettings.Networks}}' <container>`.

**2. Grafana dashboards empty / "No data".**
Check datasource URLs under Connections → Data sources: must be `http://prometheus:9090`, `http://loki:3100`, `http://jaeger:16686` (in-network names and in-network ports, not host ports). Then check Prometheus `/targets` — if targets are down, metrics never arrive.

**3. Jaeger shows no traces.**
Check `OBS_ENABLED=true` is actually set in `orgchat-open-webui`'s env (`docker exec orgchat-open-webui env | grep OBS_`). Then check otelcol logs: `docker logs obs-otelcol | tail -100` for exporter errors. Then confirm OTLP reachability from the app container: `docker exec orgchat-open-webui getent hosts otelcol`.

**4. DCGM exporter crashloop.**
Container must run with NVIDIA runtime and GPU capabilities. In `observability/docker-compose.yml`, verify for `dcgm-exporter`:
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```
and the host has the `nvidia-container-toolkit` installed.

**5. postgres-exporter auth error.**
`DATA_SOURCE_NAME` must be `postgresql://user:pass@orgchat-postgres:5432/dbname?sslmode=disable` (Go pq style). Common mistakes: missing `sslmode`, using `postgres://` with special chars un-escaped. Test the DSN with `psql` from inside the exporter container.

**6. cAdvisor port conflict with `frams-cadvisor`.**
Ours binds `127.0.0.1:8081`, FRAMS binds 8080. If port 8081 is already in use, check `ss -tlnp | grep 8081` — it should not be FRAMS. If you changed the port, update the Prometheus scrape config.

**7. Two DCGM exporters on the same GPU — is that OK?**
Yes. DCGM/NVML access is read-only and concurrent readers are supported by the driver. FRAMS (9400) and OrgChat (9401) can coexist without interference. Prometheus dedupe is not needed because each Prom instance only scrapes its own exporter.

**8. Loki: "stream limit" or high-cardinality warnings.**
`docker logs obs-loki | grep -i 'cardinality\|limit'`. Fix: edit `observability/promtail/promtail-config.yml`, drop labels that have high cardinality (e.g., `container_id`, `request_id`, full paths). Keep only `container`, `compose_service`, `level`.

**9. Disk filling up under `volumes/observability/`.**
Inspect: `du -sh /home/vogic/LocalRAG/volumes/observability/*`. Tune retention:
- Prometheus: `--storage.tsdb.retention.time=15d` (adjust in compose).
- Loki: `limits_config.retention_period: 168h` + `compactor` enabled.
- Jaeger all-in-one: in-memory by default; restart drops history. For persistence, switch backend (out of scope for this runbook).

---

## 9. What this runbook does NOT cover

- **Alerting / Alertmanager.** No routes, no receivers. Future work — see design §11.
- **SSO / SAML for Grafana.** Local admin only. Future work.
- **Multi-node / HA scaling.** Single-host Compose only. For HA, migrate to k8s with Mimir + Loki microservices + Tempo.
- **Long-term archival.** No S3 / object-store integration. Retention is local disk only.
- **Log scrubbing / PII redaction.** Promtail pipeline has no redaction stages yet.
- **Security hardening of exporters.** Exporters run with default scrape access; no TLS, no basic-auth. Localhost bind is the only boundary.

---

*End of runbook. If a step is not documented here, it is not authorized. Update this file before running anything new.*
