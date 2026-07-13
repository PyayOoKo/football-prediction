# Testing Guide

> Complete guide for running, writing, and maintaining tests across the football prediction system.

## Test Suite Overview

| Category | Directory | Tests | Focus |
|---|---|---|---|
| Unit tests | `tests/` | 1,269+ | Individual functions, classes |
| Integration | `tests/test_etl/` | ~40 | Pipeline stages, store backends |
| Database | `tests/test_database/` | ~30 | ORM models, session, repositories |
| Validation | `tests/test_validation/` | ~50 | 9 checks, engine, reporter |
| Scheduler | `tests/test_scheduler/` | ~20 | Engine, tasks, models |
| Data Versioning | `tests/test_data_versioning/` | 70 | Create/compare/rollback/verify |
| Experiment Tracking | `tests/test_experiment_tracking/` | ~30 | CRUD, queries |
| Feature Store | `tests/test_feature_store/` | ~30 | Definitions, values, batches |
| Config | `tests/test_config/` | ~10 | Settings, logging |
| Services | `tests/test_services/` | ~10 | Prediction, training |

## Running Tests

### All Tests
```bash
# Full suite
python -m pytest

# With verbose output
python -m pytest -v

# Short traceback
python -m pytest --tb=short

# Quiet mode
python -m pytest -q
```

### By Directory
```bash
# Database tests
python -m pytest tests/test_database/ -v

# ETL tests
python -m pytest tests/test_etl/ -v

# Newer packages
python -m pytest tests/test_data_versioning/ tests/test_experiment_tracking/ tests/test_feature_store/ -v
```

### By Marker
```bash
# Run slow tests
python -m pytest -m slow

# Skip slow tests
python -m pytest -m "not slow"

# Run database tests
python -m pytest -m db

# Run integration tests
python -m pytest -m integration
```

### With Coverage
```bash
# Coverage report
python -m pytest --cov=src --cov-report=term-missing

# HTML coverage report
python -m pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# XML coverage (for CI)
python -m pytest --cov=src --cov-report=xml

# Fail if below threshold
python -m pytest --cov=src --cov-fail-under=75
```

### Parallel Execution
```bash
# Run on 4 CPU cores
python -m pytest -n 4

# Distribute by test module
python -m pytest -n auto
```

## Writing Tests

### Test Structure

```python
# tests/test_feature_store/test_definitions.py
"""Tests for FeatureDefinition ORM model."""

from __future__ import annotations

from src.feature_store.models import FeatureDefinition, FeatureStatus, FeatureCategory


def test_create_feature_definition():
    """Happy path — create a valid feature definition."""
    feat = FeatureDefinition(
        name="home_attack_strength_10",
        feature_type="rolling_stat",
        category=FeatureCategory.ATTACK_STRENGTH,
        entity_type="team",
        description="Avg goals scored in last 10 home matches",
        computation_params={"window": 10, "metric": "goals_scored"},
    )
    assert feat.name == "home_attack_strength_10"
    assert feat.version == 1
    assert feat.status == FeatureStatus.DRAFT
    assert feat.is_active is True


def test_feature_definition_defaults():
    """Verify sensible defaults for missing optional fields."""
    feat = FeatureDefinition(
        name="test_feature",
        feature_type="test",
        category=FeatureCategory.COMPOSITE,
        entity_type="global",
    )
    assert feat.computation_params == {}
    assert feat.validation_rules == {}
    assert feat.dependencies == []
    assert feat.extra_metadata == {}
    assert feat.is_active is True


def test_feature_definition_to_dict():
    """Serialization includes all expected keys."""
    feat = FeatureDefinition(
        name="test",
        feature_type="test",
        category=FeatureCategory.COMPOSITE,
        entity_type="global",
    )
    d = feat.to_dict()
    assert d["name"] == "test"
    assert "created_at" in d
    assert "status" in d
```

### Database-Backed Tests

```python
"""Tests that require a database session."""
import pytest
from src.database.session import get_session


@pytest.mark.db
def test_insert_and_query_feature_definition():
    """DB round-trip — insert, flush, query back."""
    from src.feature_store.models import FeatureDefinition, FeatureCategory

    with get_session() as session:
        feat = FeatureDefinition(
            name="db_test_feature",
            feature_type="test",
            category=FeatureCategory.TEAM_FORM,
            entity_type="team",
        )
        session.add(feat)
        session.flush()

        queried = session.query(FeatureDefinition).filter_by(
            name="db_test_feature"
        ).first()

        assert queried is not None
        assert queried.id == feat.id
        assert queried.category == FeatureCategory.TEAM_FORM
```

### ETL Pipeline Tests

```python
"""Tests for the ETL Pipeline."""
import pytest
from src.etl.models import PipelineStage, StageResult, StageStatus
from src.etl.pipeline import ETLPipeline


def test_etl_result_success_property():
    """ETLResult.success is True when status is SUCCESS or WARNING."""
    from src.etl.models import ETLResult, StageStatus

    result = ETLResult(overall_status=StageStatus.SUCCESS)
    assert result.success is True

    result.overall_status = StageStatus.WARNING
    assert result.success is True

    result.overall_status = StageStatus.FAILED
    assert result.success is False
```

### Fixtures

```python
# tests/conftest.py
"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_match_data() -> list[dict]:
    """Standard match data used across multiple test suites."""
    return [
        {"home_team": "Arsenal", "away_team": "Chelsea",
         "date": "2025-03-15", "home_goals": 2, "away_goals": 1,
         "result": "H", "league": "E0", "id": "1"},
        {"home_team": "Liverpool", "away_team": "Man City",
         "date": "2025-03-16", "home_goals": 1, "away_goals": 1,
         "result": "D", "league": "E0", "id": "2"},
    ]
```

## Markers

| Marker | Description | Usage |
|---|---|---|
| `slow` | Tests that take >5 seconds | `-m "not slow"` for quick runs |
| `db` | Tests requiring a database | `-m db` |
| `integration` | Tests calling external APIs | `-m "not integration"` for offline |

Defined in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "db: marks tests that need a database connection",
    "integration: marks tests that call external APIs",
]
```

## CI Integration

Tests run automatically in CI via GitHub Actions:

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: football_test
          POSTGRES_PASSWORD: postgres
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: python -m pytest --cov=src --cov-fail-under=75 -n 4
```

## Tips

### Debugging Tests

```bash
# Run a single test with stdout
python -m pytest tests/test_etl/test_store.py::test_database_store -v -s

# Enter debugger on failure
python -m pytest --pdb

# Show local variables in traceback
python -m pytest --showlocals
```

### Performance Testing

```bash
# Profile slow tests
python -m pytest --durations=10
```

### Avoiding Common Pitfalls

1. **Never depend on test execution order** — use fixtures, not shared state
2. **Clean up database resources** — use `get_session()` context manager
3. **Mock external APIs** — don't make real HTTP calls in unit tests
4. **Parameterize when testing multiple inputs** — use `@pytest.mark.parametrize`
5. **Test edge cases** — empty data, null values, boundary conditions
