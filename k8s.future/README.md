# k8s/ — Phase 2 deployment skeleton

**This is a template, not a running chart.** Phase 1 ships on Docker Compose (single host). Phase 2 (future) migrates to Kubernetes across 4×48 GB GPUs for 20–200 concurrent users.

## What's here

- `values.yaml` — Helm values schema covering all nine services from the Phase 1 compose stack, pre-sized for Phase 2 hardware.

## What's NOT here

- Helm chart templates (`templates/*.yaml`) — to be written when Phase 2 starts.
- Ingress + cert-manager wiring.
- Prometheus / Grafana dashboards.
- Backup CronJobs.

## To wire this up when Phase 2 begins

1. Generate a fresh chart: `helm create orgchat-chart && cp values.yaml orgchat-chart/`.
2. Port compose service definitions to `templates/deployment.yaml` + `statefulset.yaml` + `service.yaml`.
3. Replace secrets (`WEBUI_SECRET_KEY`, `ADMIN_PASSWORD`, `POSTGRES_PASSWORD`) with `ExternalSecret` or vault refs — never inline.
4. Install NVIDIA device plugin on the GPU nodes (`kubectl apply -f https://...`) and confirm `nvidia.com/gpu` resource is schedulable.
5. Helm install against the cluster; use `helm lint` before each deploy.
6. Update `docs/runbook.md` § 8 with cluster-specific commands (new section).

## Reference

- CLAUDE.md §1 Phase-2 hardware goals: 4×48 GB GPUs, 70B chat model, higher concurrency.
- Master plan §4 Phase 7 deliverables: k8s skeleton + deployment runbook.
