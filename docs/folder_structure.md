# Folder Structure

```
football-prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Enterprise CI/CD (lint, test, security, docker, notify)
│       ├── pr-quality.yml      # PR quality gate (conventional commits, changelog)
│       ├── release.yml         # Semantic release (changelog, docker, GitHub Release)
│       └── docs.yml            # Deploy MkDocs to GitHub Pages
├── .pre-commit-config.yaml     # 10 pre-commit hooks (black, ruff, mypy, bandit, etc.)
├── alembic/                    # Database migrations
│   ├── env.py                  # Alembic environment (imports all models)
│   └── versions/
│       ├── 001_initial_schema.py           # 22 tables, FKs, constraints
│       ├── 002_optimize_indexes_and_partitions.py  # BRIN, partial, covering indexes
│       ├── 003_tune_fillfactor_and_sequences.py    # fillfactor 70/90, autovacuum
│       ├── 004_update_connection_config.py         # timeouts, monitoring views
│       ├── 005_fix_foreign_key_types.py            # INT→BIGINT for all FKs
│       └── 006_final_performance_tuning.py         # Missing indexes, materialized views
├── config.py                   # Legacy config (being migrated)
├── data/                       # Data directory (gitignored)
├── docs/                       # Comprehensive documentation
│   ├── index.md                # Landing page
│   ├── architecture.md         # System architecture + diagrams
│   ├── folder_structure.md     # This file
│   ├── database.md             # Schema, ERD, queries
│   ├── database_performance.md # Query optimization for 100M+ rows
│   ├── etl.md                  # ETL pipeline documentation
│   ├── scheduler.md            # Cron/Windows scheduling
│   ├── validation.md           # Data validation engine
│   ├── feature_store.md        # Feature store documentation
│   ├── experiment_tracking.md  # ML experiment management
│   ├── api.md                  # REST API documentation
│   ├── cli.md                  # CLI reference
│   ├── configuration.md        # Environment variables, config hierarchy
│   ├── developer_guide.md      # Development setup, conventions
│   ├── user_guide.md           # End-user documentation
│   ├── deployment_guide.md     # Production deployment
│   ├── testing_guide.md        # Test writing, running
│   ├── troubleshooting.md      # Common issues
│   ├── faq.md                  # Frequently asked questions
│   ├── pgbouncer_config.md     # PgBouncer connection pooling
│   ├── production_readiness_audit.md  # Full 22-category production audit
│   └── benchmarks/
│       ├── benchmark_queries.sql       # 20 benchmark queries
│       └── explain_analyze_templates.sql # 10 EXPLAIN ANALYZE templates
├── docker-compose.yml          # PostgreSQL + app + migration services
├── Dockerfile                  # Multi-stage build (builder + runtime)
├── Makefile                    # 25+ dev commands (test, lint, db, docker)
├── mkdocs.yml                  # MkDocs Material theme config
├── models/                     # Trained model storage
├── pyproject.toml              # Project metadata, dependencies, tool config
├── requirements.txt            # Pinned dependencies
├── run_dashboard.py            # Streamlit dashboard launcher
├── run_pipeline.py             # Automated prediction pipeline
├── scripts/
│   ├── bump_version.py         # Semantic version bumping
│   ├── generate_changelog.py   # Conventional commits → CHANGELOG.md
│   ├── migrate_to_partitions.py # Zero-downtime partition migration
│   └── notify.py               # Slack/email CI notification
├── src/
│   ├── __init__.py             # Package root (v0.1.0)
│   ├── app/                    # Streamlit dashboard
│   │   ├── dashboard.py        # Main dashboard page
│   │   ├── utils.py            # Dashboard caching, model diagnostic
│   │   └── pages/
│   │       ├── 1_Predict.py    # Match prediction page
│   │       ├── 2_Value_Bets.py # Value betting page
│   │       ├── 3_Backtest.py   # Backtesting page
│   │       └── 4_WorldCup.py   # World Cup 2026 page
│   ├── backtesting.py          # Betting strategy backtesting engine
│   ├── calibration.py          # Probability calibration
│   ├── cache/                  # Caching framework
│   ├── confidence_scoring.py   # Prediction confidence scoring
│   ├── config/                 # Configuration hierarchy
│   │   ├── settings.py         # Typed dataclass config (source of truth)
│   │   └── logging.py          # Logging configuration
│   ├── data/                   # Data processing
│   │   ├── cleaners.py         # Data cleaning utilities
│   │   ├── feature_engineering.py # Feature engineering pipeline
│   │   ├── loader.py           # Data loading
│   │   └── preprocessing.py    # Preprocessing pipeline
│   ├── data_collection/        # Web scrapers
│   │   ├── collector.py        # Main collector orchestrator
│   │   └── sources/
│   │       ├── fbref/          # FBref.com scraper
│   │       ├── understat/      # Understat xG data
│   │       ├── football_data_co_uk.py
│   │       ├── football_data_org.py
│   │       ├── transfermarkt.py
│   │       ├── transfermarkt_lineups.py
│   │       └── worldcup.py     # World Cup data
│   ├── data_loader.py          # Unified data loading
│   ├── data_profiling/         # Data profiling utilities
│   ├── data_versioning/        # Immutable dataset versioning
│   │   ├── manager.py          # VersionManager orchestrator
│   │   ├── storage.py          # Parquet-based version storage
│   │   ├── differ.py           # Version diff computation
│   │   ├── integration.py      # ETL auto-versioning patches
│   │   └── cli.py              # Data versioning CLI
│   ├── database/               # SQLAlchemy ORM
│   │   ├── base.py             # Declarative base with naming convention
│   │   ├── session.py          # Engine, session factory, get_session()
│   │   ├── models/             # 22 ORM models
│   │   │   ├── match.py        # Central fact table (BIGINT PK)
│   │   │   ├── team.py         # Team reference
│   │   │   ├── player.py       # Player reference
│   │   │   ├── odds.py         # Multi-bookmaker decimal odds
│   │   │   ├── prediction.py   # Model predictions
│   │   │   ├── competition.py  # League/cup/tournament
│   │   │   ├── season.py       # Time-bound competition grouping
│   │   │   ├── ...             # 15 more models
│   │   └── repositories/       # Repository pattern
│   │       ├── base.py         # Generic CRUD (T get_by_id, find, add, delete)
│   │       ├── match_repository.py  # Match-specific queries
│   │       └── team_repository.py   # Team-specific queries
│   ├── dixon_coles.py          # Dixon-Coles goal model
│   ├── eda.py                  # Exploratory data analysis
│   ├── elo.py                  # Elo rating system
│   ├── ensemble.py             # Ensemble model (XGBoost + LR + Poisson)
│   ├── etl/                    # ETL pipeline
│   │   ├── pipeline.py         # Pipeline orchestrator
│   │   ├── extract.py          # Data extraction (API, scrape)
│   │   ├── transform.py        # Data transformation
│   │   ├── normalize.py        # Schema normalization
│   │   ├── validate.py         # Validation rules
│   │   ├── store.py            # Data persistence (DB, CSV, Parquet)
│   │   ├── clean.py            # Data cleaning
│   │   ├── models.py           # Pipeline stage models
│   │   ├── progress.py         # tqdm progress tracking
│   │   └── tracker.py          # Pipeline run tracking
│   ├── evaluate.py             # Model evaluation metrics
│   ├── experiment_tracking/    # ML experiment management
│   │   ├── tracker.py          # ExperimentTracker (context manager)
│   │   ├── registry.py         # ModelRegistry (leaderboard, best model)
│   │   ├── comparator.py       # ExperimentComparator (diff, ranking)
│   │   ├── export.py           # JSON/CSV/HTML export with Plotly
│   │   ├── api.py              # FastAPI REST API
│   │   ├── cli.py              # 10 CLI subcommands
│   │   └── integrations/       # MLflow, W&B, TensorBoard adapters
│   ├── feature_engineering.py  # Feature matrix building
│   ├── feature_store/          # Feature computation platform
│   │   ├── registry.py         # FeatureRegistry (definitions, dependencies)
│   │   ├── store.py            # FeatureStore (CRUD on feature values)
│   │   ├── computation.py      # FeatureComputationEngine, LazyFeature
│   │   ├── cache.py            # FeatureCache (look-aside with invalidation)
│   │   ├── lineage.py          # FeatureLineage (full provenance tracking)
│   │   ├── validation.py       # FeatureValidator (4 rule types)
│   │   ├── computers.py        # FeatureComputer ABC + registry
│   │   └── cli.py              # 19 CLI subcommands
│   ├── hyperparameter_tuning.py # RandomizedSearchCV tuning
│   ├── importers/              # Data importers
│   ├── models/                 # ML model storage
│   ├── monitoring/             # System/ETL/cache monitoring
│   ├── odds_api.py             # The Odds API client
│   ├── odds_processing.py      # Odds normalization
│   ├── player_info.py          # Player data
│   ├── poisson_model.py        # Poisson goal model
│   ├── predict.py              # Prediction pipeline
│   ├── preprocessing.py        # Data preprocessing
│   ├── scheduler/              # Cross-platform task scheduling
│   ├── scrapers/               # Web scraping base
│   ├── services/               # Business logic orchestration
│   ├── team_normalizer/        # Team name fuzzy matching
│   ├── time_series_cv.py       # Time-series cross validation
│   ├── train.py                # Model training orchestration
│   ├── utils/                  # Exceptions, helpers, validators
│   ├── validate.py             # (placeholder)
│   ├── validation/             # Data validation engine
│   ├── value_betting.py        # Value betting computations
│   └── xg_features.py          # xG feature computation
├── tests/
│   ├── conftest.py             # Shared test fixtures
│   ├── test_config/            # Configuration tests
│   ├── test_database/          # Database model, repository, session tests
│   ├── test_etl/               # Full ETL pipeline tests (10 files)
│   ├── test_fbref/             # FBref scraper tests
│   ├── test_importers/         # Data importer tests
│   ├── test_scheduler/         # Scheduler tests
│   ├── test_services/          # Service layer tests
│   ├── test_team_normalizer/   # Team normalization tests
│   ├── test_understat/         # Understat scraper tests
│   ├── test_utils/             # Utility tests
│   ├── test_validation/        # Validation engine tests
│   ├── test_feature_store/     # Feature store tests (52 tests)
│   ├── test_experiment_tracking/ # Experiment tracking tests (98 tests)
│   └── test_data_versioning/   # Data versioning tests (70 tests)
└── train_*.py                  # Various training scripts
```
