.PHONY: help venv install test test-unit test-integration test-all test-security test-concurrency test-perf lint preflight up down smoke logs clean

PYTHON ?= python3
VENV   ?= .venv
ACTIVATE = . $(VENV)/bin/activate

help:
	@echo "venv              create .venv"
	@echo "install           install dev deps into venv"
	@echo "test              run all pytest"
	@echo "test-unit         run unit tests only"
	@echo "test-integration  run integration tests only"
	@echo "test-all          lint + unit + integration (SKIP_GPU_SMOKE=1)"
	@echo "test-security     only tests marked @security"
	@echo "test-concurrency  only tests marked @concurrency"
	@echo "test-perf         only tests marked @perf"
	@echo "lint              ruff + mypy"
	@echo "preflight         download + verify model weights (needs internet once)"
	@echo "up                docker compose up -d"
	@echo "down              docker compose down"
	@echo "smoke             curl healthchecks"
	@echo "logs              docker compose logs -f"
	@echo "clean             remove .venv, caches"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(ACTIVATE) && pip install -e ".[dev]"

test:
	$(ACTIVATE) && pytest -v

test-unit:
	$(ACTIVATE) && pytest tests/unit -v

test-integration:
	$(ACTIVATE) && pytest tests/integration -v

test-all:
	$(ACTIVATE) && ruff check . && mypy .
	$(ACTIVATE) && pytest tests/unit -v
	$(ACTIVATE) && SKIP_GPU_SMOKE=1 pytest tests/integration -v

test-security:
	$(ACTIVATE) && pytest -m security -v

test-concurrency:
	$(ACTIVATE) && pytest -m concurrency -v

test-perf:
	$(ACTIVATE) && pytest -m perf -v

lint:
	$(ACTIVATE) && ruff check . && mypy .

preflight:
	$(ACTIVATE) && python scripts/preflight_models.py

up:
	docker compose -f compose/docker-compose.yml --env-file compose/.env up -d

down:
	docker compose -f compose/docker-compose.yml --env-file compose/.env down

smoke:
	$(ACTIVATE) && pytest tests/integration/test_compose_up.py -v

logs:
	docker compose -f compose/docker-compose.yml --env-file compose/.env logs -f

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ---- Eval harness targets (Plan A Phase 0.6) ----

KB_EVAL_ID ?= 1
API_BASE   ?= http://localhost:6100
GOLDEN     ?= tests/eval/golden_starter.jsonl
BASELINE   ?= tests/eval/results/phase-0-baseline.json
LATEST     ?= tests/eval/results/latest.json

.PHONY: eval eval-baseline eval-gate eval-seed

eval-seed:
	@test -n "$$RAG_ADMIN_TOKEN" || { echo "ERROR: export RAG_ADMIN_TOKEN"; exit 2; }
	$(ACTIVATE) && python -m tests.eval.seed_test_kb \
	  --kb-id $(KB_EVAL_ID) \
	  --api-base-url $(API_BASE)

eval:
	@mkdir -p tests/eval/results
	$(ACTIVATE) && python -m tests.eval.harness \
	  --golden $(GOLDEN) \
	  --kb-id $(KB_EVAL_ID) \
	  --api-base-url $(API_BASE) \
	  --out $(LATEST)

eval-baseline: eval
	cp $(LATEST) $(BASELINE)
	@echo "baseline committed to $(BASELINE); include in commit"

eval-gate: eval
	$(ACTIVATE) && python -m tests.eval.gate \
	  --baseline $(BASELINE) \
	  --latest $(LATEST) \
	  --slo docs/runbook/slo.md
