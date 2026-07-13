# Developer Guide

> Everything you need to contribute to the football prediction project.

## Getting Started

### Prerequisites
- Python 3.12+
- PostgreSQL 16+ (or SQLite for local dev)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/football-prediction.git
cd football-prediction

# Create virtual environment
python -m venv .venv

# Activate it
# Windows (Command Prompt):
.venv\Scripts\activate
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements.txt[dev]

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Run database migrations
alembic upgrade head
```

### Verify Installation

```bash
# Run the test suite
python -m pytest tests/ -v --tb=short

# Collect test data
python collect_all_worldcups.py
python collect_leagues.py

# Train a model
python train_xgboost.py

# Launch the dashboard
python run_dashboard.py
```

## Project Structure

```
football-prediction/
├── src/
│   ├── app/              # Streamlit dashboard
│   ├── config/           # Configuration & logging
│   ├── data/             # Data loading & preprocessing
│   ├── database/         # SQLAlchemy models & session
│   ├── data_collection/  # Scrapers (fbref, understat)
│   ├── data_versioning/  # Dataset version control
│   ├── etl/              # ETL pipeline framework
│   ├── experiment_tracking/ # ML experiment tracking
│   ├── feature_store/    # Feature registry & computation
│   ├── importers/        # football-data.org importer
│   ├── models/           # ML model wrappers (ensemble)
│   ├── monitoring/       # System & data quality metrics
│   ├── scheduler/        # Automated task scheduler
│   ├── services/         # Business logic (prediction, training)
│   ├── team_normalizer/  # Team name normalisation
│   ├── utils/            # Shared utilities
│   └── validation/       # Data validation framework
├── tests/                # All test files
├── alembic/              # Database migrations
├── docs/                 # Documentation
└── scripts/              # CI/CD & utility scripts
```

## Coding Standards

### Python Style
- **Formatter:** Black (line-length: 88)
- **Linter:** Ruff (E, F, I, N, W, ARG, B, C4, SIM)
- **Type checker:** MyPy (strict mode)
- **Import sorting:** Ruff's I rule (isort-compatible)

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

Configured hooks:
- `ruff` — linter and import sorter
- `ruff-format` — code formatter
- `mypy` — type checking
- `check-merge-conflict` — merge conflict detection
- `check-json` — JSON syntax validation
- `check-yaml` — YAML syntax validation
- `detect-private-key` — secret detection
- `trailing-whitespace` — trailing whitespace removal
- `end-of-file-fixer` — ensure files end with newline
- `debug-statements` — catch `pdb`/`ipdb`/`breakpoint()`

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Packages | `snake_case` | `src/data_collection/` |
| Modules | `snake_case` | `feature_engineering.py` |
| Classes | `PascalCase` | `EnsembleModel` |
| Functions | `snake_case` | `build_features()` |
| Variables | `snake_case` | `team_elo` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES = 3` |
| Database tables | `snake_case` | `feature_definitions` |
| Database columns | `snake_case` | `team_elo_home` |

### Docstrings

Use NumPy-style docstrings:

```python
def build_features(
    df: pd.DataFrame,
    is_training: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix from raw match data.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed match data with date, teams, goals, etc.
    is_training : bool
        If True, separates the target variable and builds
        all features. If False, only build features (no target).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix ``(X)`` and target series ``(y)``.
        ``y`` is ``None`` when ``is_training=False``.
    """
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test directory
python -m pytest tests/test_etl/ -v

# Run a single test file
python -m pytest tests/test_database/test_session.py -v

# Run tests marked as "slow"
python -m pytest -m slow

# Run tests excluding slow
python -m pytest -m "not slow"

# Run with parallel execution
python -m pytest -n 4
```

### Coverage Goals

| Metric | Target |
|---|---|
| Overall coverage | ≥ 75% |
| Core packages (database, etl, validation) | ≥ 85% |
| New code (in PR) | ≥ 80% |

### Test Structure

```
tests/
├── conftest.py                  # Shared fixtures
├── test_config/
├── test_database/
├── test_data/
├── test_etl/
├── test_importers/
├── test_models/
├── test_services/
├── test_scheduler/
├── test_validation/
├── test_data_versioning/
├── test_experiment_tracking/
├── test_feature_store/
└── test_monitoring/
```

## Git Workflow

### Branch Strategy
- `main` — production-ready code
- `develop` — integration branch
- `feature/*` — new features (branched from `develop`)
- `fix/*` — bug fixes (branched from `main` or `develop`)
- `release/*` — release candidates

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Elo-based team rating feature
fix: correct off-by-one in rolling window calculation
docs: add database ER diagram
test: add integration tests for ETL pipeline
refactor: simplify ensemble weight optimisation
perf: optimise feature matrix construction
chore: update dependencies
```

### PR Checklist

Before submitting a PR:

- [ ] Code follows project style (Black, Ruff, MyPy pass)
- [ ] All tests pass (`python -m pytest`)
- [ ] New code has >80% coverage
- [ ] Docstrings added/updated
- [ ] Changelog updated (if user-facing change)
- [ ] Migration checked (if DB schema changed)
- [ ] Pre-commit hooks pass

## Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "add_elo_indexes"

# Run pending migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# View migration history
alembic history

# Check current version
alembic current
```

## Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Enable SQL echo
export DB_ECHO=true

# Use SQLite for local testing
export DATABASE_URL=sqlite:///data/test.db
```

## Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run tests in container
docker-compose run app python -m pytest

# Access PostgreSQL
docker-compose exec db psql -U postgres football_prediction
```

## Troubleshooting

### Virtual Environment Issues

```bash
# Check Python version
python --version  # Must be 3.12+

# Verify venv is active
which python  # Should point to .venv/Scripts/python

# If numpy/pandas fail to install (MinGW Python):
# Use Microsoft Store Python instead:
"/c/Users/dell/AppData/Local/Microsoft/WindowsApps/python3.exe" -m venv .venv
```

### Database Connection Issues

```bash
# Test connection
python -c "from src.database.session import get_engine; engine = get_engine(); print('OK')"

# Verify PostgreSQL is running
docker-compose ps
```

### Module Not Found

```bash
# Run from project root (not src/)
cd /path/to/football-prediction

# Verify PYTHONPATH
echo $PYTHONPATH  # Should be empty — we use relative imports

# Run as module
python -m src.scheduler.cli list
```
