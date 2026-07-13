# Football Match Outcome Prediction — Project Documentation

> **Version:** 0.1.0 | **Python:** 3.12+ | **License:** MIT

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Folder Structure](#2-folder-structure)
3. [Database Schema](#3-database-schema)
4. [ETL Workflow](#4-etl-workflow)
5. [Scheduler Workflow](#5-scheduler-workflow)
6. [Validation Workflow](#6-validation-workflow)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Sequence Diagrams](#8-sequence-diagrams)
9. [Class Diagrams](#9-class-diagrams)
10. [Setup Guide](#10-setup-guide)
11. [Development Guide](#11-development-guide)
12. [Deployment Guide](#12-deployment-guide)
13. [Contribution Guide](#13-contribution-guide)
14. [Coding Standards](#14-coding-standards)
15. [Configuration Reference](#15-configuration-reference)
16. [Testing Strategy](#16-testing-strategy)

---

## 1. System Architecture

### High-Level Overview

The system is a modular, production-oriented ML pipeline for football match outcome prediction. It follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Streamlit Dashboard  │  CLI  │  REST API (FastAPI)          │   │
│  └──────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                         APPLICATION LAYER                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │Predict   │ │Betting   │ │Backtest  │ │Training  │ │Scheduler│  │
│  │Service   │ │Engine    │ │Engine    │ │Service   │ │Engine   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └─────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         FEATURE LAYER                               │
│  ┌────────────────────┐ ┌────────────────────┐ ┌─────────────────┐ │
│  │ Feature Framework  │ │ Feature Store      │ │ Betting Engine  │ │
│  │ (Transformer ABC)  │ │ (Registry + Store) │ │ (12 Modules)    │ │
│  └────────────────────┘ └────────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                     MACHINE LEARNING LAYER                          │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌──────┐ ┌───────────┐  │
│  │XGBoost│ │LightGBM│ │Random │ │Logistic│ │Ensemble│ │Poisson   │  │
│  │       │ │       │ │Forest │ │Regr.  │ │Model  │ │Model     │  │
│  └───────┘ └───────┘ └───────┘ └───────┘ └──────┘ └───────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     DATA LAYER                                      │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │PostgreSQL│ │ Parquet  │ │   CSV    │ │  Cache   │ │  Models  │  │
│  │ (SQLAlch)│ │ Version  │ │  (Raw)   │ │ (SQLite) │ │(joblib)  │  │
│  └─────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     COLLECTION LAYER                                │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │
│  │football-data │ │ The Odds API │ │ Transfermarkt│ │ StatsBomb  │ │
│  │  .co.uk     │ │ (live odds)  │ │ (players)    │ │ (xG data)  │ │
│  └─────────────┘ └──────────────┘ └──────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Plugin-based feature system** | New features register via `FeatureTransformer` ABC — the framework auto-discovers them, builds a DAG, computes in dependency order |
| **Typed configuration dataclasses** | All 19 config sections live in `config.py` with full type hints — no magic strings, IDE autocomplete works everywhere |
| **SQLAlchemy ORM with Alembic** | 22 models with full foreign key constraints, CHECK constraints, Alembic migrations with autogeneration |
| **Chronological train/test split** | Every split is `shuffle=False` — the oldest data goes to training, preventing time-series leakage |
| **Stateless calculators** | EV, Kelly, CLV calculators are pure functions — the betting engine composes them in a 9-step pipeline |
| **tqdm everywhere** | Every batch operation shows progress bars — collection, feature engineering, training, backtesting |

### Core Packages

| Package | Purpose | Key Classes |
|---------|---------|-------------|
| `src/` | Legacy pipeline | `EloSystem`, `PoissonModel`, `add_elo_features` |
| `src/feature_framework/` | Modern feature system | `FeatureTransformer`, `FeaturePipeline`, `FeaturePluginRegistry` |
| `src/feature_store/` | DB-backed feature storage | `FeatureStore`, `FeatureRegistry`, `FeatureComputer`, `FeatureComputationEngine` |
| `src/betting/` | Betting engine | `BettingEngine`, `BettingPipeline` (9-step), `KellyCalculator`, `CLVCalculator` |
| `src/database/` | ORM models | 22 SQLAlchemy models across `core/`, `analytics/`, `betting/` domains |
| `src/etl/` | ETL pipeline | `ETLPipeline` (6-stage orchestration), `DataCleaner`, `DataNormalizer` |
| `src/scheduler/` | Task scheduler | `TaskEngine` (dependency resolution, retry, parallel groups) |
| `src/validation/` | Data validation | `ValidationEngine` (9 built-in checks), `CheckResult`, `Severity` |
| `src/cache/` | Caching framework | `CacheManager`, `SQLiteBackend`, `RedisBackend` |
| `src/data_profiling/` | Dataset profiling | `DataProfiler`, `DataDriftDetector`, `ReportGenerator` |
| `src/data_versioning/` | Dataset versioning | `VersionManager`, `VersionDiff`, Parquet-based snapshots |
| `src/experiment_tracking/` | ML experiment mgmt | `ExperimentTracker`, `ModelRegistry`, `ExperimentComparator` |
| `src/app/` | Streamlit dashboard | 5-page dashboard with probability bars, bracket trees, backtest charts |
| `src/scrapers/` | Data collection | `BaseScraper`, `football-data.co.uk`, `Transfermarkt`, `Understat` |

---

## 2. Folder Structure

```
football-prediction/
│
├── .github/                    # GitHub Actions CI/CD workflows
├── alembic/                    # Database migration scripts
│   ├── versions/               #   Generated migration files
│   └── env.py                  #   Alembic environment configuration
├── data/                       # Datasets (gitignored except .gitkeep)
│   ├── raw/                    #   Original CSV/API dumps
│   ├── processed/              #   Cleaned, feature-engineered data
│   ├── external/               #   Cache files (odds, reference data)
│   ├── scrapers/               #   Scraper-specific data
│   └── versions/               #   Versioned Parquet snapshots
├── docs/                       # Project documentation
│   ├── features/               #   Feature documentation (team_form, elo)
│   ├── vault/                  #   Knowledge base articles
│   ├── er_diagram.md           #   Entity-relationship diagram
│   └── PROJECT_DOCUMENTATION.md #  This file
├── logs/                       # Application logs (gitignored)
├── models/                     # Serialised trained models (gitignored)
├── notebooks/                  # Jupyter notebooks for EDA & prototyping
├── reports/                    # Generated reports (gitignored)
│   ├── predictions/            #   Prediction CSVs
│   ├── backtest/               #   Backtest charts
│   └── profiling/              #   Data profiling reports
├── scripts/                    # Utility scripts
│   ├── auto_commit.ps1         #   Hourly GitHub auto-commit (Windows)
│   ├── backtest_high_conf_away.py
│   ├── debug_lineups.py
│   ├── debug_transfermarkt.py
│   ├── evaluate_existing.py
│   ├── quick_train_eval.py
│   ├── run_training.py
│   ├── today_value_bets.py
│   ├── tune_ensemble.py
│   └── verify_ids.py
├── src/                        # Main source package
│   ├── __init__.py             #   Version: 0.1.0
│   │
│   ├── app/                    #   Streamlit dashboard
│   │   ├── dashboard.py        #     Main dashboard page
│   │   ├── utils.py            #     Caching & shared helpers
│   │   └── pages/              #     Multi-page views
│   │       ├── 1_Predict.py    #       Predict a match
│   │       ├── 2_Value_Bets.py #       Find value bets
│   │       ├── 3_Backtest.py   #       Historical backtesting
│   │       └── 4_WorldCup.py   #       World Cup 2026 bracket
│   │
│   ├── betting/                #   Modular betting engine
│   │   ├── __init__.py         #     40+ exported symbols
│   │   ├── base.py             #     12 typing Protocols
│   │   ├── models.py           #     10 dataclasses + 3 enums
│   │   ├── calculator.py       #     5 stateless calculators
│   │   ├── engine.py           #     BettingEngine orchestrator
│   │   ├── registry.py         #     9 sub-registries
│   │   ├── plugins.py          #     Plugin discovery
│   │   ├── factory.py          #     Config-driven creation
│   │   ├── api.py              #     REST API integration
│   │   ├── cli.py              #     4 CLI commands
│   │   └── decorators.py       #     @timed, @logged, @retry
│   │
│   ├── cache/                  #   Caching framework
│   │   ├── __init__.py
│   │   ├── backend.py          #     SQLiteBackend, RedisBackend
│   │   ├── decorators.py       #     @cached, @invalidate
│   │   ├── manager.py          #     CacheManager
│   │   └── models.py           #     CacheEntry, CacheStats
│   │
│   ├── config/                  #   Application configuration
│   │   ├── __init__.py
│   │   ├── settings.py         #     Env-based settings singleton
│   │   └── logging.py          #     Logging configuration
│   │
│   ├── data/                   #   Data processing
│   │   ├── __init__.py
│   │   ├── cleaners.py         #     Data cleaning utilities
│   │   ├── feature_engineering.py  # Feature builder
│   │   ├── loader.py           #     Data loader
│   │   └── preprocessing.py    #     Preprocessor
│   │
│   ├── data_collection/        #   Data ingestion
│   │   ├── __init__.py
│   │   ├── cleaners.py         #     Collection-specific cleaning
│   │   ├── collector.py        #     Master collector
│   │   └── sources/            #     Source-specific implementations
│   │       ├── football_data_co_uk.py
│   │       ├── football_data_org.py
│   │       ├── transfermarkt.py
│   │       ├── transfermarkt_lineups.py
│   │       ├── understat/
│   │       │   ├── client.py, importer.py, models.py, parser.py
│   │       └── worldcup.py
│   │
│   ├── data_profiling/         #   Dataset profiling
│   │   ├── profiler.py         #     DataProfiler
│   │   ├── reports.py          #     ReportGenerator (HTML/JSON/CSV)
│   │   └── drift.py            #     DataDriftDetector
│   │
│   ├── data_versioning/        #   Dataset versioning
│   │   ├── manager.py          #     VersionManager
│   │   ├── models.py           #     VersionInfo, VersionDiff
│   │   ├── storage.py          #     Parquet version storage
│   │   └── cli.py              #     CLI for version management
│   │
│   ├── database/               #   SQLAlchemy ORM
│   │   ├── __init__.py
│   │   ├── base.py             #     Declarative Base with naming convention
│   │   ├── session.py          #     Engine + session factory + get_session()
│   │   └── models/             #     22 ORM models
│   │       ├── __init__.py     #       Consolidated exports
│   │       ├── match.py        #       Central fact table
│   │       ├── team.py
│   │       ├── competition.py  #       With tier level
│   │       ├── season.py
│   │       ├── country.py
│   │       ├── stadium.py
│   │       ├── referee.py
│   │       ├── player.py
│   │       ├── match_statistics.py
│   │       ├── weather.py
│   │       ├── odds.py
│   │       ├── lineup.py
│   │       ├── team_form.py
│   │       ├── team_elo_history.py
│   │       ├── team_xg_history.py
│   │       ├── prediction.py
│   │       ├── expected_value_bet.py
│   │       ├── closing_line_value.py
│   │       ├── betting_result.py
│   │       ├── player_match_stats.py
│   │       ├── injury.py
│   │       └── transfer.py
│   │
│   ├── etl/                    #   ETL pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py         #     ETLPipeline (6-stage orchestrator)
│   │   ├── extract.py          #     CSVExtractor, APIExtractor
│   │   ├── validate.py         #     DataValidator
│   │   ├── clean.py            #     DataCleaner
│   │   ├── normalize.py        #     DataNormalizer
│   │   ├── transform.py        #     DataTransformer
│   │   ├── store.py            #     DatabaseStore, CSVStore
│   │   ├── models.py           #     ETLConfig, ETLResult, StageResult
│   │   ├── tracker.py          #     JobTracker (checkpoint/resume)
│   │   └── progress.py         #     ProgressReporter (tqdm)
│   │
│   ├── experiment_tracking/    #   ML experiment management
│   │   ├── __init__.py
│   │   ├── models.py           #     Experiment, Run, BestModel
│   │   ├── tracker.py          #     ExperimentTracker
│   │   ├── registry.py         #     ModelRegistry (leaderboard)
│   │   ├── comparator.py       #     ExperimentComparator
│   │   ├── export.py           #     JSON/CSV/HTML export
│   │   ├── cli.py              #     10 subcommands
│   │   ├── api.py              #     FastAPI REST API (20+ endpoints)
│   │   └── integrations/       #     MLflow, W&B, TensorBoard adapters
│   │
│   ├── feature_framework/      #   Feature engineering framework
│   │   ├── __init__.py         #     25+ exported symbols
│   │   ├── base.py             #     FeatureTransformer ABC
│   │   ├── pipeline.py         #     FeaturePipeline orchestrator
│   │   ├── config.py           #     FeatureConfig (YAML/JSON)
│   │   ├── plugins.py          #     FeaturePluginRegistry
│   │   ├── parallel.py         #     ParallelComputer
│   │   ├── models.py           #     ComputationResult, PipelineReport
│   │   ├── decorators.py       #     @timeit, @log_call, @retry
│   │   ├── exceptions.py       #     5 custom exception classes
│   │   └── features/           #     Concrete implementations
│   │       ├── __init__.py
│   │       ├── team_form.py    #       TeamFormTransformer
│   │       └── elo_rating.py   #       EloTransformer + EloEngine
│   │
│   ├── feature_store/          #   Feature Store (DB-backed)
│   │   ├── __init__.py         #     30+ exported symbols
│   │   ├── models.py           #     FeatureDefinition, FeatureValue
│   │   ├── registry.py         #     FeatureRegistry
│   │   ├── store.py            #     FeatureStore CRUD
│   │   ├── validation.py       #     FeatureValidator
│   │   ├── computation.py      #     FeatureComputationEngine
│   │   ├── computers.py        #     FeatureComputer ABC
│   │   ├── cache.py            #     FeatureCache
│   │   ├── lineage.py          #     FeatureLineage
│   │   └── cli.py              #     CLI for feature management
│   │
│   ├── scheduler/              #   Scheduler
│   │   ├── __init__.py
│   │   ├── engine.py           #     TaskEngine (dependency resolution)
│   │   ├── models.py           #     Task, TaskResult, RunReport
│   │   ├── tasks.py            #     6 built-in tasks
│   │   ├── cli.py              #     CLI for scheduler management
│   │   └── windows_scheduler.py #    Windows Task Scheduler integration
│   │
│   ├── scrapers/               #   Scraper base
│   │   ├── __init__.py
│   │   └── base.py             #     BaseScraper ABC
│   │
│   ├── services/               #   Business logic
│   │   ├── __init__.py
│   │   ├── prediction_service.py  # Model inference coordination
│   │   └── training_service.py    # Training lifecycle management
│   │
│   ├── utils/                  #   Cross-cutting utilities
│   │   ├── __init__.py
│   │   ├── exceptions.py       #     Custom exception hierarchy
│   │   ├── helpers.py          #     Shared helper functions
│   │   └── validators.py       #     Input validation utilities
│   │
│   ├── validation/             #   Data validation
│   │   ├── __init__.py
│   │   ├── engine.py           #     ValidationEngine
│   │   ├── checks.py           #     9 validation check functions
│   │   ├── models.py           #     ValidationResult, CheckResult
│   │   └── reporter.py         #     ReportGenerator for validation
│   │
│   │   # ── Core ML modules (legacy, being migrated to feature_framework)
│   ├── backtesting.py          #   Historical betting simulation
│   ├── calibration.py          #   Probability calibration
│   ├── confidence_scoring.py   #   Prediction confidence scores
│   ├── data_loader.py          #   Data loading utilities
│   ├── dixon_coles.py          #   Dixon-Coles MLE model
│   ├── eda.py                  #   Exploratory data analysis
│   ├── elo.py                  #   Legacy Elo system
│   ├── ensemble.py             #   Ensemble model
│   ├── evaluate.py             #   Model evaluation
│   ├── feature_engineering.py  #   Feature engineering pipeline
│   ├── hyperparameter_tuning.py
│   ├── odds_api.py             #   Live odds client
│   ├── odds_processing.py      #   Odds data processing
│   ├── player_info.py          #   Player/lineup data
│   ├── poisson_model.py        #   Poisson regression
│   ├── predict.py              #   Match prediction
│   ├── preprocessing.py        #   Data preprocessing
│   ├── time_series_cv.py       #   Time-series cross-validation
│   ├── train.py                #   Model training
│   ├── value_betting.py        #   Value betting calculations
│   └── xg_features.py          #   Expected Goals features
│
├── tests/                      # Test suite (1,427 tests)
│   ├── conftest.py             #   Shared fixtures
│   ├── test_betting/           #   Betting engine tests
│   ├── test_cache/             #   Cache framework tests
│   ├── test_config/            #   Configuration tests
│   ├── test_database/          #   Database model/repository tests
│   ├── test_data/              #   Data processing tests
│   ├── test_etl/               #   ETL pipeline tests
│   ├── test_feature_framework/ #   Feature framework tests
│   ├── test_feature_store/     #   Feature store tests
│   ├── test_importers/         #   Data import tests
│   ├── test_models/            #   ML model tests
│   ├── test_odds_api.py        #   Odds API client tests
│   ├── test_scheduler/         #   Scheduler tests
│   ├── test_scrapers/          #   Scraper tests
│   ├── test_services/          #   Service layer tests
│   ├── test_team_normalizer/   #   Team name normalizer tests
│   ├── test_understat/         #   Understat scraper tests
│   ├── test_utils/             #   Utility tests
│   └── test_validation/        #   Validation engine tests
│
├── config.py                   # Centralised typed configuration
├── pyproject.toml              # Project metadata, deps, tool config
├── requirements.txt            # Pinned dependencies
├── setup.py                    # Package setup (legacy)
├── Makefile                    # 20+ dev commands
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # PostgreSQL + app services
├── .env.example                # Environment variable template
├── .pre-commit-config.yaml     # Pre-commit hooks
├── .gitignore                  # Git ignore rules
├── CONTRIBUTING.md             # Contributor guide
├── LICENSE                     # MIT license
├── MANIFEST.in                 # Package manifest
├── README.md                   # Quick-start (GitHub front page)
│
│   # ── Top-level scripts ──────────────────────────────
├── run_dashboard.py            # Launch Streamlit dashboard
├── run_pipeline.py             # Run the full prediction pipeline
├── run_backtest.py             # Run historical backtest
├── run_first_model.py          # Train first model (legacy)
├── train_worldcup.py           # Train World Cup model
├── predict_worldcup.py         # Predict World Cup matches
├── refresh_worldcup.py         # Full World Cup refresh cycle
├── find_value_bets.py          # Find value bets
├── collect_leagues.py          # Collect league data
├── collect_all_worldcups.py    # Collect all World Cup data
├── collect_worldcup.py         # Collect single World Cup
├── collect_worldcup_xg.py      # Collect xG data for World Cups
├── collect_lineups.py          # Collect lineup data
├── collect_player_data.py      # Collect player info
├── collect_r16_data.py         # Collect Round of 16 data
├── collect_xag_data.py         # Collect xAG data
├── merge_all_xg_data.py        # Merge all xG data sources
├── merge_xg_data.py            # Merge xG data
├── compare_models_brazil_norway.py
├── what_if_brazil_norway.py    # Brazil vs Norway scenario
├── what_if_canada_morocco.py   # Canada vs Morocco scenario
├── what_if_portugal_spain.py   # Portugal vs Spain scenario
├── analyze_england_norway.py   # England vs Norway analysis
├── train_league.py             # Train league model
├── train_xgboost.py            # Train XGBoost model
├── train_with_xag.py           # Train with xAG data
├── run_combined_pipeline.py    # Combined pipeline run
├── today_value_bets_live.py    # Today's live value bets
├── find_value_bets.py          # Find value bets (fallback)
├── bracket_simulator.py        # Knockout bracket simulation
├── test_2022_worldcup.py       # Test 2022 WC predictions
├── setup_auto_commit.bat       # Install hourly auto-commit
├── setup_value_bets_scheduler.bat
├── scheduler_config.yaml       # Task definitions
└── setup_scheduler.bat         # Install World Cup refresh
```

---

## 3. Database Schema

The database uses **PostgreSQL** with **SQLAlchemy 2.0** ORM and **Alembic** for migrations. There are 22 models organised into three logical domains.

### Entity-Relationship Diagram

```
┌────────────┐     ┌──────────────┐     ┌──────────────┐
│  Country   │──1:N│ Competition  │──1:N│    Season    │
│────────────│     │──────────────│     │──────────────│
│ id (PK)    │     │ id (PK)      │     │ id (PK)      │
│ name       │     │ name         │     │ competition_id(FK)
│ alpha2     │     │ code (unique)│     │ name         │
│ alpha3     │     │ type         │     │ start_date   │
│ fifa_code  │     │ level (tier) │     │ end_date     │
└────────────┘     │ country_id(FK)│     └──────────────┘
                   └──────────────┘            │
                                      ┌────────┴─────────┐
                                      │                  │
┌────────────┐     ┌──────────────┐   │   ┌──────────────┐│
│   Team    │──1:N│    Match     │<──┘   │   Referee    ││
│───────────│     │──────────────│       │──────────────││
│ id (PK)   │     │ id (PK)      │       │ id (PK)      ││
│ name      │     │ season_id(FK)│       │ name         ││
│ country_id│     │ competition  │       │ country_id   ││
│ short_name│     │  _id (FK)    │       └──────────────┘│
└───────────┘     │ home_team    │                  │
                  │  _id (FK)    │──────────────────┘
                  │ away_team    │
                  │  _id (FK)    │
                  │ referee_id   │
                  │ stadium_id   │
                  │ date         │
                  │ result       │── CHECK(H/D/A/NULL)
                  │ home_goals   │
                  │ away_goals   │
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
 ┌────────┴─────┐ ┌──────┴──────┐ ┌────┴─────────┐
 │TeamForm      │ │TeamEloHistory│ │TeamXgHistory │
 │──────────────│ │─────────────│ │──────────────│
 │team_id (FK)  │ │team_id (FK) │ │team_id (FK)  │
 │match_id (FK) │ │match_id(FK) │ │match_id (FK) │
 │form_last5    │ │elo_before   │ │xg_avg5       │
 │form_last10   │ │elo_after    │ │xga_avg5      │
 │goals_avg5    │ │elo_change   │ │xg_avg10      │
 │goals_avg10   │ │k_factor     │ │xga_avg10     │
 │clean_sheets5 │ │side(H/A)    │ └──────────────┘
 │points_total  │ └─────────────┘
 └──────────────┘

┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│MatchStats    │     │      Odds        │     │   Lineup     │
│──────────────│     │──────────────────│     │──────────────│
│match_id (FK) │     │match_id (FK)     │     │match_id (FK) │
│home_shots    │     │bookmaker         │     │team_id (FK)  │
│away_shots    │     │home_odds         │     │formation     │
│home_sot      │     │draw_odds         │     │starting_xi   │
│away_sot      │     │away_odds         │     │substitutes   │
│home_possession│    │timestamp         │     └───────────────┘
│home_corners  │     └──────────────────┘
│home_yellow   │     ┌──────────────────┐
│home_red      │     │   Prediction     │
│away_...      │     │──────────────────│
└──────────────┘     │match_id (FK)     │
                     │model_type        │
┌──────────────┐     │home_prob         │
│   Weather    │     │draw_prob         │
│──────────────│     │away_prob         │
│match_id (FK) │     │confidence        │
│temperature   │     │calibration_method│
│humidity      │     └──────────────────┘
│wind_speed    │
│pitch_condition│
└──────────────┘
```

### Core Tables (22 total)

| Domain | Table | Description | Key Columns |
|--------|-------|-------------|-------------|
| **Core** | `countries` | ISO-coded country reference | alpha2, alpha3, fifa_code |
| | `competitions` | League/cup/tournament | type (CHECK), level (tier) |
| | `seasons` | Time-bound grouping | competition_id (FK), dates |
| | `teams` | Club or national team | country_id (FK), short_name |
| | `stadiums` | Venues | city, capacity, surface |
| | `referees` | Match officials | country_id (FK) |
| | `matches` | Central fact table (7 FKs) | result (CHECK), goals |
| | `match_statistics` | Shot/possession/cards (1:1) | shots, sot, possession |
| | `weather` | Match conditions (1:1) | temperature, humidity |
| | `odds` | Multi-bookmaker odds (1:N) | bookmaker, home/draw/away |
| | `lineups` | Formations + players (1:N per team) | formation, starting_xi |
| **Analytics** | `team_form` | Pre-computed rolling form | form_last5/10, points |
| | `team_elo_history` | Elo snapshots | elo_before/after, k_factor |
| | `team_xg_history` | xG rolling averages | xg_avg5/10, xga_avg5/10 |
| **Betting** | `predictions` | Model probabilities | home/draw/away prob |
| | `expected_value_bets` | EV calculations | ev, edge, stake |
| | `closing_line_value` | Line movement | clv, fair_open/close |
| | `betting_results` | Actual P&L | profit, roi |
| **Players** | `players` | Player info | position, market_value |
| | `player_match_stats` | Per-match performance | goals, xg, rating |
| | `injuries` | Injury tracking | type, severity, return_date |
| | `transfers` | Transfer history | fee, from_club, to_club |

### Naming Conventions

- **Tables**: snake_case, plural (`team_elo_history`)
- **Primary keys**: `id` (auto-increment integer)
- **Foreign keys**: `{table}_id` (e.g., `team_id`)
- **CHECK constraints**: `ck_{table}_{constraint}`
- **Unique constraints**: `uq_{table}_{column}`
- **Indexes**: automatic on all FKs, manual on query-heavy columns

---

## 4. ETL Workflow

The ETL pipeline is a **6-stage composable orchestrator** in `src/etl/pipeline.py`.

### Pipeline Stages

```
Raw Data ──► EXTRACT ──► VALIDATE ──► CLEAN ──► NORMALIZE ──► TRANSFORM ──► STORE ──► Database
                │           │           │           │             │            │
                ▼           ▼           ▼           ▼             ▼            ▼
         CSVExtractor  DataVali-  DataCleaner  DataNorm-   DataTrans-   DatabaseStore
         APIExtractor  dator      (fill/drop)  alizer      former       CSVStore
         DBExtractor              (team names, (feature     (PostgreSQL,
                                  dates)       engineer)    SQLite)
```

### Stage Details

| Stage | Class | Responsibility | Output |
|-------|-------|---------------|--------|
| **1. EXTRACT** | `BaseExtractor` | Read from CSV, API, or database | `list[dict]` raw records |
| **2. VALIDATE** | `DataValidator` | Type coercion, required fields | Validated records |
| **3. CLEAN** | `DataCleaner` | Missing values, outliers | Clean records |
| **4. NORMALIZE** | `DataNormalizer` | Team name standardisation, date parsing | Normalised records |
| **5. TRANSFORM** | `DataTransformer` | Feature engineering | Transformed records |
| **6. STORE** | `DataStore` | Write to DB or CSV | Persisted records |

### Checkpoint & Resume

Each stage can be checkpointed via `JobTracker`. If the pipeline fails mid-way, it can resume from the last successful stage:

```python
pipeline = ETLPipeline(
    name="import_matches",
    source="football-data-co-uk",
    extractor=CSVExtractor("data/raw/results.csv"),
    store=DatabaseStore(Match, unique_columns=["match_id"]),
    checkpoint=True,  # Enable resume support
)
result = pipeline.run()
```

### Parallel Processing

Stages can run in parallel using `concurrent.futures.ThreadPoolExecutor`:

```python
pipeline = ETLPipeline(
    ...,
    parallel=True,
    max_workers=4,
)
```

### Progress Reporting

Every stage shows a `tqdm` progress bar with records-in/records-out counts and elapsed time.

---

## 5. Scheduler Workflow

The scheduler in `src/scheduler/` provides task orchestration with dependency resolution, retry logic, and structured reporting.

### Task Types (6 built-in)

```
download_fixtures ──► validate_data ──► clean_data ──► update_database ──► backup_database
                                                                            │
                                                                       generate_logs
```

| Task | Description | Default Schedule | Retries |
|------|-------------|-----------------|---------|
| `download_fixtures` | Download match fixtures from API | Daily | 3 |
| `validate_data` | Run validation checks on new data | After download | 2 |
| `clean_data` | Clean and normalise data | After validate | 2 |
| `update_database` | Write clean data to PostgreSQL | After clean | 3 |
| `backup_database` | Create database backup | After update | 2 |
| `generate_logs` | Rotate and archive log files | Daily | 1 |

### Execution Flow

```
TaskEngine.run_all()
    │
    ├── 1. Resolve task order (topological sort based on dependencies)
    │
    ├── 2. For each task (in dependency order):
    │       ├── Check dependencies → skip if any failed
    │       ├── Execute with retry (linear backoff: 2s × attempt)
    │       ├── Record TaskResult (SUCCESS / FAILED / SKIPPED / WARNING)
    │       └── If FAILED and abort_on_failure → break pipeline
    │
    └── 3. Return RunReport (succeeded, failed, skipped counts)
```

### Windows Task Scheduler Integration

```batch
setup_scheduler.bat  # Installs a daily Windows Task
setup_auto_commit.bat  # Installs hourly GitHub auto-commit
```

### Configuration (scheduler_config.yaml)

```yaml
pipeline_name: "daily_refresh"
abort_on_failure: true
tasks:
  - name: download_fixtures
    enabled: true
    retry_count: 3
    dependencies: []
  - name: validate_data
    enabled: true
    retry_count: 2
    dependencies: [download_fixtures]
```

---

## 6. Validation Workflow

The validation engine in `src/validation/` runs **9 built-in checks** on any imported dataset.

### Validation Checks

| # | Check | Description | Severity |
|---|-------|-------------|----------|
| 1 | **Duplicate Matches** | Same home/away/date combo | ERROR |
| 2 | **Invalid Dates** | Future dates, null dates, implausible years | ERROR |
| 3 | **Invalid Odds** | Negative odds, extreme odds (>50) | WARNING |
| 4 | **Missing Goals** | Null or negative goal values | ERROR |
| 5 | **Missing Teams** | Unknown team names | ERROR |
| 6 | **Incorrect Leagues** | Unknown league codes | WARNING |
| 7 | **Invalid Statistics** | Shots < sot, negative stats | WARNING |
| 8 | **Duplicate IDs** | Non-unique match IDs | ERROR |
| 9 | **Impossible Scores** | >25 goals, negative scores | ERROR |

### Validation Engine Flow

```
ValidationEngine.run(data)
    │
    ├── For each check (name, function, kwargs):
    │       ├── Execute check function
    │       ├── Catch exceptions → create ERROR result
    │       └── Append CheckResult to results list
    │
    └── Return ValidationResult:
        ├── passed_checks / total_checks
        ├── total_violations
        ├── is_valid (all ERROR checks passed)
        └── per-check violations with row indices
```

### Severity Levels

| Severity | Behaviour | Example |
|----------|-----------|---------|
| `ERROR` | Blocks pipeline | Duplicate match, null goals |
| `WARNING` | Logs warning, continues | Unknown league code |
| `INFO` | Statistical observation | 10% missing odds data |

---

## 7. Data Flow Diagrams

### End-to-End Prediction Pipeline

```
┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌───────────┐
│  DATA    │    │  FEATURE │    │  MODEL      │    │ PREDICT  │    │  REPORT   │
│ COLLECT  │───►│ ENGINEER │───►│  TRAIN      │───►│          │───►│           │
└──────────┘    └──────────┘    └─────────────┘    └──────────┘    └───────────┘
     │               │                │                 │               │
     ▼               ▼                ▼                 ▼               ▼
  football-       Rolling           XGBoost          Match           Prediction
  data.co.uk      averages          Random           outcome         CSV
  Understat       Elo ratings       Forest           probabili-      Dashboard
  StatsBomb       H2H stats         Logistic         ties            Value bets
  Transfermarkt   League pos.       Regression       [H/D/A]         Backtest
                  Team form         Ensemble                          charts
```

### Value Betting Flow

```
Model     ──►  Match        ──►  Expected     ──►  Kelly       ──►  Bet
Probabilities    Odds             Value (EV)        Criterion        Slip
[H/D/A]         [H/D/A]         = P×O - 1         stake %          size
                                    │                 │
                                    ▼                 ▼
                              EV > threshold?    Fraction ×
                                  Yes ──► compute  bankroll %
```

### Feature Engineering Framework Flow

```
YAML Config         Plugin Registry     FeaturePipeline
    │                     │                    │
    ▼                     ▼                    ▼
Feature definitions   Auto-discover       Resolve DAG
(name, type, deps)    transformer         Topological sort
                      classes              │
                                           ▼
                                    Compute in order
                                    (parallel pool)
                                           │
                                           ▼
                                    Validate output
                                    Store via FeatureStore
                                           │
                                           ▼
                                    PipelineReport
                                    (computed/skipped/failed)
```

---

## 8. Sequence Diagrams

### World Cup Refresh Sequence

```
refresh_worldcup.py                    football-data.co.uk    XGBoost Model
       │                                      │                    │
       │  1. Download latest results          │                    │
       │─────────────────────────────────────►│                    │
       │◄─────────────────────────────────────│                    │
       │                                      │                    │
       │  2. Resolve knockout bracket         │                    │
       │     placeholders (W1, R2, ...)       │                    │
       │                                      │                    │
       │  3. Build feature matrix             │                    │
       │     (rolling, Elo, H2H, xG)          │                    │
       │                                      │                    │
       │  4. Train XGBoost model              │                    │
       │──────────────────────────────────────────────────────────►│
       │◄──────────────────────────────────────────────────────────│
       │                                      │                    │
       │  5. Predict upcoming matches         │                    │
       │──────────────────────────────────────────────────────────►│
       │◄──────────────────────────────────────────────────────────│
       │                                      │                    │
       │  6. Save predictions to CSV          │                    │
       │     reports/predictions_worldcup/    │                    │
       │                                      │                    │
       │  7. Auto-commit to GitHub            │                    │
       │     (hourly Windows Task)            │                    │
```

### Betting Engine Sequence

```
User                 BettingEngine        Calculator       BankrollManager
  │                       │                    │                  │
  │  run_pipeline()       │                    │                  │
  │──────────────────────►│                    │                  │
  │                       │                    │                  │
  │                  ┌────┴────┐               │                  │
  │                  │ Step 1: │               │                  │
  │                  │ OddsSrc │               │                  │
  │                  └─────────┘               │                  │
  │                  ┌──────────┐              │                  │
  │                  │ Step 2:  │              │                  │
  │                  │ EV Calc  │─────────────►│                  │
  │                  └──────────┘              │                  │
  │                  ┌──────────┐              │                  │
  │                  │ Step 3:  │              │                  │
  │                  │ Kelly    │─────────────►│                  │
  │                  └──────────┘              │                  │
  │                  ┌───────────┐             │                  │
  │                  │ Step 4:   │             │                  │
  │                  │ BetFilter │             │                  │
  │                  └───────────┘             │                  │
  │                  ┌─────────────────┐       │                  │
  │                  │ Step 5:          │      │                  │
  │                  │ PortfolioOptimize│      │                  │
  │                  └─────────────────┘       │                  │
  │                  ┌──────────────┐          │                  │
  │                  │ Step 6:       │          │                  │
  │                  │ Place Bets    │───────────────────────────►│
  │                  └──────────────┘          │                  │
  │                       │                    │                  │
  │◄──────────────────────│                    │                  │
  │                       │                    │                  │
  │  BettingSessionReport │                    │                  │
```

### Feature Pipeline Sequence

```
User               FeaturePipeline      PluginRegistry     FeatureStore
  │                       │                    │                │
  │  run(entity, df)      │                    │                │
  │──────────────────────►│                    │                │
  │                       │                    │                │
  │                  ┌────┴────┐               │                │
  │                  │ Load    │               │                │
  │                  │ config  │               │                │
  │                  └─────────┘               │                │
  │                  ┌──────────┐              │                │
  │                  │ Resolve  │─────────────►│                │
  │                  │ plugins  │              │                │
  │                  └──────────┘              │                │
  │                  ┌───────────┐             │                │
  │                  │ Build DAG │             │                │
  │                  │ (topo sort)            │                │
  │                  └───────────┘             │                │
  │                       │                    │                │
  │             ┌─────────┴─────────┐          │                │
  │             │  For each feature │          │                │
  │             │  (DAG order):     │          │                │
  │             │                   │          │                │
  │             │  transform(df)    │          │                │
  │             │  validate output  │          │                │
  │             │  store results    │──────────────────────────►│
  │             └───────────────────┘          │                │
  │                       │                    │                │
  │◄──────────────────────│                    │                │
  │                       │                    │                │
  │  PipelineReport       │                    │                │
```

---

## 9. Class Diagrams

### Feature Framework Core

```
┌─────────────────────────────────────────────────────┐
│              FeatureTransformer (ABC)                │
├─────────────────────────────────────────────────────┤
│  + name: str                                        │
│  + version: int                                     │
│  + description: str                                 │
│  + dependencies: list[str]                          │
│  + output_columns: list[str]                        │
│  + data_type: str                                   │
│  + computation_time: str                            │
│  + metadata: FeatureMetadata (property)             │
├─────────────────────────────────────────────────────┤
│  + init(context)                                    │
│  + validate_input(df): list[str]                    │
│  + transform(df, context): DataFrame (abstract)     │
│  + validate_output(df): list[str]                   │
│  + to_dict(): dict                                  │
└───────────────────────┬─────────────────────────────┘
                        │  implements
          ┌─────────────┼─────────────┐
          │             │             │
┌─────────▼─────┐ ┌────▼──────┐ ┌────▼──────────┐
│TeamFormTransf.│ │EloTrans-  │ │(future: H2H,  │
│(rolling stats)│ │former     │ │ league pos,   │
│               │ │(EloEngine)│ │ xG features)  │
└───────────────┘ └───────────┘ └───────────────┘

┌──────────────────────────────────────────────────────┐
│                   FeaturePipeline                     │
├──────────────────────────────────────────────────────┤
│  - config: FeatureConfig                             │
│  - plugins: FeaturePluginRegistry                    │
│  - show_progress: bool                               │
│  - parallel: bool                                    │
├──────────────────────────────────────────────────────┤
│  + run(entity_type, df): PipelineReport              │
│  + resume(batch_id): PipelineReport                  │
│  + register_transformer(transformer)                 │
│  + register_transformer_class(cls)                   │
│  + get_dag(): dict                                   │
│  + print_dag()                                       │
└──────────────────────────────────────────────────────┘
```

### Betting Engine Core

```
┌─────────────────────────────────────────────────────┐
│              BettingEngine                           │
├─────────────────────────────────────────────────────┤
│  - probability_source: ProbabilitySource            │
│  - odds_source: OddsSource                          │
│  - calculators: dict                                │
│  - bankroll: BankrollManager                        │
│  - filters: list                                    │
├─────────────────────────────────────────────────────┤
│  + run_pipeline(matches, staking): Report           │
│  + print_summary()                                  │
└──────────┬──────────────────────────────────────────┘
           │  composes
           │
    ┌──────┴─────────────────────────────────────────┐
    │             12 Module Interfaces (Protocols)    │
    ├────────────────────────────────────────────────┤
    │  ProbabilitySource    → get_probability()      │
    │  OddsSource           → get_odds()             │
    │  ExpectedValueCalc    → calculate()            │
    │  KellyCalculator      → calculate()            │
    │  FractionalKellyCalc  → calculate()            │
    │  FlatStakeCalculator  → calculate()            │
    │  CLVCalculator        → calculate()            │
    │  BankrollManager      → place_bet(), settle()  │
    │  RiskManager          → check_limits()         │
    │  BetFilter            → accept()               │
    │  MarketFilter         → is_suitable()          │
    │  PortfolioOptimizer   → allocate()             │
    └────────────────────────────────────────────────┘
```

### Elo Engine

```
┌──────────────────────────────────────────────────────┐
│                    EloEngine                          │
├──────────────────────────────────────────────────────┤
│  - k: int                                            │
│  - home_advantage: int                               │
│  - initial_rating: float                             │
│  - new_team_rating: float                            │
│  - regression_factor: float                          │
│  - _ratings: dict[str, float]                        │
│  - _history: list[EloMatchRecord]                    │
├──────────────────────────────────────────────────────┤
│  + expected_score(home, away): float                 │
│  + update(home, away, result, ...): EloMatchRecord   │
│  + process_matches(df, ...): DataFrame               │
│  + get_rating(team): float                           │
│  + set_rating(team, rating)                          │
│  + reset()                                           │
│  + get_history_df(): DataFrame                       │
│  + team_trajectory(team): DataFrame                  │
│  + current_snapshot(): EloSnapshot                   │
│  + plot_team_trajectory(team): Figure                │
│  + plot_rating_distribution(): Figure                │
│  + print_standings()                                 │
│  + benchmark_report(): dict                          │
└────────────────────────┬─────────────────────────────┘
                         │  wraps
              ┌──────────▼─────────────┐
              │   EloTransformer        │
              │   (FeatureTransformer)  │
              │   output: h_elo, a_elo,│
              │          elo_diff      │
              └────────────────────────┘

┌───────────────────────────┐
│   EloMatchRecord          │
├───────────────────────────┤
│  match_index: int         │
│  home_elo_before: float   │
│  away_elo_before: float   │
│  home_elo_after: float    │
│  away_elo_after: float    │
│  elo_diff: float          │
│  expected_home: float     │
│  actual_home: float       │
│  k_factor: float          │
│  home_elo_change: float   │
│  away_elo_change: float   │
└───────────────────────────┘

┌───────────────────────────┐
│   EloSnapshot             │
├───────────────────────────┤
│  timestamp: datetime      │
│  ratings: dict[str,float] │
│  total_matches: int       │
└───────────────────────────┘
```

### ETL Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                     ETLPipeline                          │
├─────────────────────────────────────────────────────────┤
│  - name: str                                            │
│  - source: str                                          │
│  - extractor: BaseExtractor                             │
│  - validator: DataValidator                             │
│  - cleaner: DataCleaner                                 │
│  - normalizer: DataNormalizer                           │
│  - transformer: DataTransformer                         │
│  - store: DataStore                                     │
│  - tracker: JobTracker (checkpoint/resume)              │
├─────────────────────────────────────────────────────────┤
│  + run(**kwargs): ETLResult                             │
│  + _run_extract(data): StageResult                      │
│  + _run_validate(data): StageResult                     │
│  + _run_clean(data): StageResult                        │
│  + _run_normalize(data): StageResult                    │
│  + _run_transform(data): StageResult                    │
│  + _run_store(data): StageResult                        │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Setup Guide

### Prerequisites

- **Python 3.12+**
- **PostgreSQL 15+** (or Docker for local development)
- **Git**
- **Make** (optional, for `Makefile` commands)

### Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/football-prediction.git
cd football-prediction

# 2. Create virtual environment
python3.12 -m venv .venv

# Activate:
# Linux/macOS:
source .venv/bin/activate
# Windows (Command Prompt):
.venv\Scripts\activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash):
source .venv/Scripts/activate

# 3. Install dependencies
make install
# OR: pip install -r requirements.txt

# 4. Install dev dependencies
make dev-install
# OR: pip install -r requirements.txt
#     pip install pytest pytest-cov black ruff mypy pre-commit
#     pre-commit install

# 5. Configure environment
cp .env.example .env
# Edit .env with your database credentials and API keys

# 6. Start database (Docker)
make db-up
# OR: docker compose up -d db

# 7. Run database migrations
make db-migrate
# OR: alembic upgrade head

# 8. Verify setup
make test
# Expected: 1,427 tests pass
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | (builds from parts) | Full PostgreSQL URL |
| `THE_ODDS_API_KEY` | No | - | The Odds API key (free tier) |
| `FOOTBALL_DATA_API_KEY` | No | - | football-data.org API key |
| `APP_ENV` | No | `development` | `development/staging/production` |
| `LOG_LEVEL` | No | `INFO` | `DEBUG/INFO/WARNING/ERROR` |
| `SECRET_KEY` | No | `change-me` | Session signing key |

### Docker Setup

```bash
# Build and start all services
make docker-up-all
# OR: docker compose up -d

# Build without compose
make docker-build
# OR: docker build -t football-prediction .
```

---

## 11. Development Guide

### Common Commands (via Makefile)

```bash
make install          # Install production dependencies
make dev-install      # Install dev + pre-commit hooks
make lint             # Run ruff linter
make format           # Check formatting with black
make format-fix       # Auto-fix formatting
make typecheck        # Run mypy strict mode
make test             # Run pytest (1,427 tests)
make test-cov         # Run with coverage report
make db-up            # Start PostgreSQL via Docker
make db-down          # Stop PostgreSQL
make db-migrate       # Run Alembic migrations
make db-autogen       # Auto-generate migration
make run-pipeline     # Run full prediction pipeline
make run-dashboard    # Launch Streamlit dashboard
make clean            # Remove all cache artifacts
```

### Feature Development Workflow

To add a new feature to the framework:

```python
# 1. Create a FeatureTransformer subclass
from src.feature_framework import FeatureTransformer

class NewFeatureTransformer(FeatureTransformer):
    name = "new_feature"
    version = 1
    description = "Description of the new feature"
    output_columns = ["h_new_col", "a_new_col"]
    dependencies = []  # Names of features this depends on
    data_type = "float"
    computation_time = "fast"
    category = "form"

    def transform(self, df, context=None):
        # Compute your feature
        df["h_new_col"] = ...
        df["a_new_col"] = ...
        return df

# 2. Register with the pipeline
pipeline = FeaturePipeline(show_progress=False)
pipeline.plugins.register(NewFeatureTransformer)

# 3. Add YAML config
# features:
#   - name: new_feature
#     type: new_feature
#     category: form
#     version: 1
#     params:
#       custom_param: 42
```

### Running Specific Scripts

```bash
# World Cup pipeline
python refresh_worldcup.py         # Full refresh: download → train → predict
python train_worldcup.py           # Train only

# League data
python collect_leagues.py --train   # Download + train on league data

# Backtesting
python run_backtest.py             # Historical betting simulation

# Value bets
python find_value_bets.py          # With fallback odds
python today_value_bets_live.py    # With live API odds

# Predictions
python predict_worldcup.py         # Generate predictions from saved model

# Merge xG data
python merge_all_xg_data.py        # Merge StatsBomb xG data
```

---

## 12. Deployment Guide

### Production Docker Deployment

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: football_prediction
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s

  app:
    build: .
    depends_on:
      db:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+psycopg2://postgres:${DB_PASSWORD}@db:5432/football_prediction
      APP_ENV: production
      LOG_LEVEL: INFO

  dashboard:
    build: .
    command: streamlit run run_dashboard.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      - db
```

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: make dev-install
      - run: make lint
      - run: make typecheck
      - run: make test
```

### Production Checklist

- [ ] Set `APP_ENV=production`
- [ ] Set strong `SECRET_KEY`
- [ ] Configure PostgreSQL with persistent volume
- [ ] Set up log rotation (handled by `generate_logs` task)
- [ ] Configure The Odds API key for live odds
- [ ] Install Windows Scheduler tasks for automation
- [ ] Set up monitoring (health checks, alerts)
- [ ] Configure backup strategy (handled by `backup_database` task)
- [ ] Review `config.py` for production-appropriate values

---

## 13. Contribution Guide

### Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/your-username/football-prediction.git`
3. **Set up** development environment: `make dev-install`
4. **Create a branch**: `git checkout -b feature/my-feature`

### Pull Request Checklist

- [ ] Code follows style guidelines (black, ruff, mypy)
- [ ] All tests pass (`make test`)
- [ ] New code includes appropriate test coverage
- [ ] Database migrations included (if applicable)
- [ ] Documentation updated (docstrings, feature docs)
- [ ] Changes rebased onto latest `main`
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

### Commit Convention

```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, perf, test, chore, ci
Scope: etl, betting, framework, db, scheduler, dashboard, etc.

Examples:
  feat(framework): add EloTransformer with dynamic K-factor
  fix(betting): correct Kelly stake for negative EV bets
  docs: add ETL workflow diagram
  test(etl): add pipeline resume tests
```

### Reporting Issues

Use GitHub issue templates:
- **Bug report**: Include full error message, steps to reproduce, expected vs actual
- **Feature request**: Describe the problem, proposed solution, alternatives considered

---

## 14. Coding Standards

### Python Style

| Rule | Standard | Enforced By |
|------|----------|-------------|
| Line length | 88 characters | Black, Ruff |
| Indentation | 4 spaces | Black |
| Quotes | Double quotes (`"`) | Black |
| Imports | Grouped: stdlib → third-party → local | Ruff (I) |
| Type hints | Required for all function signatures | mypy (strict) |
| Docstrings | NumPy-style | Manual review |
| Naming | `snake_case` for functions/vars, `PascalCase` for classes | Ruff (N) |

### File Structure Convention

Every module should follow this structure:

```python
"""
Module docstring — one-line summary.

Extended description with usage examples.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.some_module import SomeClass

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────
_DEFAULT_VALUE: int = 42


# ── Classes ────────────────────────────────────────────
class MyClass:
    """Class docstring."""

    def method(self) -> None:
        ...


# ── Functions ──────────────────────────────────────────
def my_function() -> None:
    ...
```

### Pre-commit Hooks

The project uses `pre-commit` with:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.0.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### Type Checking (mypy strict)

All code must pass mypy in strict mode:
- `strict = true` — enables all type-checking features
- `warn_unused_ignores = true` — no unused `# type: ignore` comments
- `no_implicit_reexport = true` — explicit re-exports required
- `ignore_missing_imports = true` — for packages without stubs

### Testing Standards

- **pytest** with `--tb=short` and `-v`
- Tests mirror `src/` structure in `tests/`
- Each test class focuses on one class or function
- Fixtures use `pytest.fixture` (not `unittest.TestCase`)
- Database tests use `pytest.mark.db`
- Slow tests use `pytest.mark.slow`
- Integration tests use `pytest.mark.integration`

```python
# Example test structure
class TestMyClass:
    def test_basic_functionality(self):
        ...
    def test_edge_case_empty_input(self):
        ...
    def test_failure_invalid_argument(self):
        ...
```

---

## 15. Configuration Reference

All settings are centralised in `config.py` under typed dataclasses. The singleton `config` instance is importable anywhere:

```python
from config import config

# Override settings at runtime
config.train.learning_rate = 0.03
config.features.rolling_windows = (5, 10, 20)
```

### Configuration Sections (20 total)

| Section | Dataclass | Key Settings | Default |
|---------|-----------|-------------|---------|
| **Data** | `DataConfig` | `source`, `split_ratios`, `seed` | local, (0.70, 0.15, 0.15), 42 |
| **Data Collection** | `DataCollectionConfig` | `leagues`, `max_seasons`, `missing_strategy` | ("E0",), 10, "fill_zero" |
| **Preprocessing** | `PreprocessingConfig` | `normalise_teams`, `add_temporal_features` | True, True |
| **Features** | `FeatureConfig` | `form_window`, `rolling_windows`, `include_h2h` | 5, (5,10,20), True |
| **Training** | `TrainConfig` | `model_type`, `n_estimators`, `learning_rate` | xgboost, 300, 0.05 |
| **Prediction** | `PredictConfig` | `probability_threshold`, `top_k` | 0.5, 10 |
| **Odds API** | `OddsAPIConfig` | `regions`, `markets`, `cache_ttl` | us,uk,eu, h2h, 3600 |
| **Value Betting** | `ValueBetConfig` | `bankroll`, `kelly_fraction`, `min_ev` | 1000, 0.25, 0.0 |
| **Odds Processing** | `OddsConfig` | `opening_odds_cols`, `compute_consensus` | BbMx*, True |
| **Player Info** | `PlayerInfoConfig` | `enabled`, `placeholder_value` | False, 0.0 |
| **xG Features** | `XgConfig` | `rolling_windows`, `compute_xpts` | (5,10), True |
| **Poisson** | `PoissonConfig` | `min_matches`, `max_goals` | 0, 8 |
| **Dixon-Coles** | `DixonColesConfig` | `enabled`, `refit_every`, `decay_halflife` | False, 500, 1460 |
| **Elo** | `EloConfig` | `k`, `home_advantage`, `initial_rating` | 32, 100, 1500 |
| **Ensemble** | `EnsembleConfig` | `model_names`, `weight_grid_step` | (xgboost, lr, poisson), 0.10 |
| **Hyper Tune** | `HyperTuneConfig` | `n_iter_random`, `cv_folds` | 50, 5 |
| **Confidence** | `ConfidenceConfig` | `weight_spread`, `weight_agreement` | 0.40, 0.35 |
| **Backtesting** | `BacktestConfig` | `initial_bankroll`, `kelly_fraction` | 1000, 0.25 |
| **Evaluation** | `EvalConfig` | `metrics`, `plot_confusion_matrix` | (accuracy, ..., log_loss), True |
| **Paths** | `Paths` | All managed directory paths | data/, models/, logs/ |

---

## 16. Testing Strategy

### Test Suite Overview

| Metric | Value |
|--------|-------|
| Total tests | **1,427** |
| Test files | 90+ files mirroring `src/` |
| Coverage target | 75% (enforced by `pytest-cov`) |
| Slow tests | ~5 (marked `@pytest.mark.slow`) |
| DB tests | 22 (marked `@pytest.mark.db`) |
| Integration tests | 8 (marked `@pytest.mark.integration`) |

### Test Categories

| Category | Description | Location |
|----------|-------------|----------|
| **Unit tests** | Individual classes and functions | `tests/test_*/*.py` |
| **Feature framework** | FeatureTransformer lifecycle, DAG, pipeline | `tests/test_feature_framework/` |
| **Betting engine** | Kelly calc, EV, CLV, pipeline orchestration | `tests/test_betting/` |
| **ETL pipeline** | Extract → Validate → Clean → Normalize → Transform → Store | `tests/test_etl/` |
| **Validation** | 9 validation checks, severity levels | `tests/test_validation/` |
| **Database** | ORM models, repositories, session management | `tests/test_database/` |
| **Scheduler** | Task dependency resolution, retry, reporting | `tests/test_scheduler/` |
| **Cache** | SQLite backend, Redis backend, decorators | `tests/test_cache/` |
| **Services** | Prediction service, training service | `tests/test_services/` |

### Running Tests

```bash
# All tests
make test                       # pytest tests/ -v

# With coverage
make test-cov                   # pytest --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_feature_framework/test_team_form.py -v

# Specific test class
pytest tests/test_feature_framework/test_elo_rating.py::TestEloCoreFormulas -v

# Exclude slow tests
pytest -m "not slow"

# Database tests only
pytest -m db

# Parallel execution (requires pytest-xdist)
pytest -n auto
```

### Test Fixtures

Shared fixtures live in `tests/conftest.py`:

```python
@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """12 match rows with 8 distinct teams."""
    return pd.DataFrame({
        "date": [...],
        "home_team": [...],
        "away_team": [...],
        "home_goals": [...],
        "away_goals": [...],
        "result": [...],
    })

@pytest.fixture
def sample_with_optional(sample_matches) -> pd.DataFrame:
    """Add optional stat columns (xG, shots, etc.)."""
    df = sample_matches.copy()
    df["home_xg"] = [1.8, 0.9, ...]
    return df
```

### Writing New Tests

```python
"""Tests for MyNewFeature."""

import pandas as pd
import pytest

from src.feature_framework.features.my_new_feature import MyNewFeature


class TestMyNewFeature:
    def test_basic_computation(self):
        """Feature should compute expected values."""
        t = MyNewFeature()
        df = pd.DataFrame({"value": [1.0, 2.0]})
        result = t.transform(df)
        assert "new_column" in result.columns
        assert result["new_column"].iloc[0] == pytest.approx(1.0)

    def test_empty_input(self):
        """Empty DataFrame should not crash."""
        t = MyNewFeature()
        df = pd.DataFrame()
        result = t.transform(df)
        assert result is not None
```

---

> **Document Version:** 1.0 | **Last Updated:** July 13, 2026
>
> For questions or corrections, open an issue on GitHub.
