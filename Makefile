# ═════════════════════════════════════════════════════════
#  Football Prediction — Makefile
#  Common development and maintenance commands.
# ═════════════════════════════════════════════════════════

.PHONY: install dev-install lint format typecheck test test-cov clean \
        db-up db-down db-migrate db-init docker-build docker-up \
        run-pipeline run-train run-predict setup-hooks help

# ── Environment ─────────────────────────────────────────
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# ── Installation ────────────────────────────────────────
install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

$(VENV)/bin/activate:
	python -m venv $(VENV)

dev-install: install
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black ruff mypy pre-commit
	pre-commit install

# ── Code quality ────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	black --check --diff src/ tests/

format-fix:
	black src/ tests/

typecheck:
	mypy src/ tests/

# ── Testing ─────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Database ────────────────────────────────────────────
db-up:
	docker compose up -d db

db-down:
	docker compose down

db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

db-history:
	alembic history

db-autogen:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

db-init: db-up
	@sleep 3
	alembic upgrade head

# ── Docker ──────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-up-all: db-up docker-up

# ── Application commands ────────────────────────────────
run-pipeline:
	$(PYTHON) -m run_pipeline

run-train:
	$(PYTHON) -m train_worldcup

run-predict:
	$(PYTHON) -m predict_worldcup

run-dashboard:
	$(PYTHON) -m run_dashboard

# ── Setup ──────────────────────────────────────────────
setup-hooks:
	pre-commit install
	pre-commit run --all-files

# ── Cleanup ────────────────────────────────────────────
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf *.egg-info
	rm -rf build dist
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ── Help ────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  install       — Create venv and install dependencies"
	@echo "  dev-install   — Install dev dependencies and pre-commit hooks"
	@echo "  lint          — Run ruff linter"
	@echo "  format        — Check formatting with black"
	@echo "  format-fix    — Auto-fix formatting with black"
	@echo "  typecheck     — Run mypy type checker"
	@echo "  test          — Run pytest suite"
	@echo "  test-cov      — Run tests with coverage report"
	@echo "  db-up         — Start PostgreSQL via docker compose"
	@echo "  db-down       — Stop PostgreSQL"
	@echo "  db-migrate    — Run Alembic migrations"
	@echo "  db-rollback   — Rollback last migration"
	@echo "  db-autogen    — Auto-generate Alembic migration"
	@echo "  docker-build  — Build Docker images"
	@echo "  docker-up     — Start all Docker services"
	@echo "  run-pipeline  — Run the full prediction pipeline"
	@echo "  run-train     — Train the World Cup model"
	@echo "  run-predict   — Generate predictions"
	@echo "  clean         — Remove cache and build artifacts"
