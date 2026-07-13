# Football Prediction Documentation

> AI-powered match outcome prediction, value betting analysis, and backtested performance tracking.

---

## 📚 Documentation Index

### Core References
| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System architecture, layers, sequence/class diagrams |
| [Folder Structure](folder_structure.md) | Complete directory tree + package descriptions |
| [Configuration](configuration.md) | Environment variables, settings, config hierarchy |
| [CLI Reference](cli.md) | All CLI commands across all packages |
| [API Reference](api.md) | Streamlit pages, internal APIs, REST endpoints |

### Data & Infrastructure
| Document | Description |
|----------|-------------|
| [Database Schema](database.md) | 22-table ER diagram, indexes, partitioning, queries |
| [ETL Pipeline](etl.md) | 6-stage pipeline, store backends, checkpointing |
| [Scheduler](scheduler.md) | 6 tasks, engine, DAG, Windows integration |
| [Validation Framework](validation.md) | 9 checks, engine, HTML/JSON/CSV reports |
| [Feature Store](feature_store.md) | Registry, DAG, batch computation, lineage |
| [Experiment Tracking](experiment_tracking.md) | Experiments, runs, best models, artifacts |
| [Database Performance](database_performance.md) | 100M+ row tuning, indexes, migration 006 |
| [PgBouncer Config](pgbouncer_config.md) | Connection pooling guide |

### Guides
| Document | Description |
|----------|-------------|
| [Developer Guide](developer_guide.md) | Setup, coding standards, git workflow |
| [User Guide](user_guide.md) | Dashboard, predictions, value betting |
| [Deployment Guide](deployment_guide.md) | Docker, production DB, backup/restore |
| [Testing Guide](testing_guide.md) | Running tests, writing tests, coverage |
| [Troubleshooting](troubleshooting.md) | Common issues with solutions |
| [FAQ](faq.md) | Frequently asked questions |
| [Production Readiness Audit](production_readiness_audit.md) | 22-category audit, scores, roadmap |
| [CI/CD Pipeline](../.github/workflows/ci.yml) | GitHub Actions, security scans, auto-release |

### Project Files
| File | Description |
|------|-------------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guidelines |
| [README.md](../README.md) | Project overview |
| [GUIDE.md](../GUIDE.md) | Detailed setup guide |
| [CHANGELOG](../.github/workflows/release.yml) | Auto-generated from conventional commits |

## 🏗️ Packages

| Package | Purpose | Docs |
|---------|---------|------|
| `src/config/` | Configuration hierarchy | [Configuration](configuration.md) |
| `src/database/` | SQLAlchemy ORM, 22 models | [Database](database.md) |
| `src/etl/` | Extract-Transform-Load pipeline | [ETL](etl.md) |
| `src/feature_store/` | Feature computation, caching, lineage | [Feature Store](feature_store.md) |
| `src/experiment_tracking/` | ML experiment management | [Experiment Tracking](experiment_tracking.md) |
| `src/data_versioning/` | Immutable dataset versioning | [CLI](cli.md) |
| `src/scheduler/` | Cron/Windows scheduling | [Scheduler](scheduler.md) |
| `src/validation/` | Data validation engine | [Validation](validation.md) |
| `src/monitoring/` | System/ETL/cache monitoring | [Architecture](architecture.md) |
| `src/data_collection/` | Web scrapers (FBref, Understat) | [Architecture](architecture.md) |
| `src/app/` | Streamlit dashboard | [User Guide](user_guide.md) |

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/football-prediction
cd football-prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e .[dev]

# 3. Configure environment
cp .env.example .env
# Edit .env with your database and API credentials

# 4. Setup database
docker compose up -d db
alembic upgrade head

# 5. Install pre-commit hooks
pre-commit install

# 6. Run tests
pytest tests/ -v

# 7. Launch dashboard
streamlit run src/app/dashboard.py
```

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| Lines of Code | ~46,500 |
| Source Files | ~120 |
| Test Files | ~60 |
| Test Cases | 1,269 (97.9% passing) |
| Database Tables | 22 |
| Database Migrations | 6 |
| Python Version | 3.12 |
| Database | PostgreSQL 16 |

## 🔗 External Resources

- [GitHub Repository](https://github.com/yourusername/football-prediction)
- [Issue Tracker](https://github.com/yourusername/football-prediction/issues)
- [Changelog](https://github.com/yourusername/football-prediction/releases)
