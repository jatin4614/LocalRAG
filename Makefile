.PHONY: help venv install test test-unit test-integration lint preflight up down smoke logs clean

PYTHON ?= python3
VENV   ?= .venv
ACTIVATE = . $(VENV)/bin/activate

help:
	@echo "venv              create .venv"
	@echo "install           install dev deps into venv"
	@echo "test              run all pytest"
	@echo "test-unit         run unit tests only"
	@echo "test-integration  run integration tests only"
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
