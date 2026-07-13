# ⚽ Football Match Prediction — Project Documentation

> **Version:** 0.1.0 | **Python:** 3.12+ | **License:** MIT

A modular, production-oriented machine learning pipeline for predicting football match outcomes, finding value bets, simulating tournaments, and tracking experiments — with live odds integration.

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

---

## 1. System Architecture

### 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACES                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────────────┐  │
│  │   CLI       │  │  Streamlit   │  │  Scheduler │  │  Jupyter Notebooks │  │
│  │  (python)   │  │  Dashboard   │  │  (cron)    │  │  (EDA/Prototyping) │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  └────────────────────┘  │
└─────────┼─────────────────┼────────────────┼────────────────────────────────┘
          │                 │                │
┌─────────▼─────────────────▼────────────────▼────────────────────────────────┐
│                        APPLICATION LAYER (src/)                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      ORCHESTRATION PIPELINES                         │   │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────┐  ┌────────────┐  │   │
│  │  │ ETL Pipeline │  │ Training      │  │ Scheduler│  │ Backtesting│  │   │
│  │  │ (etl/)       │  │ (train.py)    │  │ (sche-   │  │(backtesting│  │   │
│  │  │              │  │               │  │  duler/) │  │ .py)       │  │   │
│  │  └──────┬───────┘  └──────┬────────┘  └────┬─────┘  └──────┬─────┘  │   │
│  └─────────┼──────────────────┼─────────────────┼──────────────┼────────┘   │
│            │                  │                 │              │            │
│  ┌─────────▼──────────────────▼─────────────────▼──────────────▼────────┐   │
│  │                       CORE SERVICES                                  │   │
│  │  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │   │
│  │  │ Feature   │ │ Ensemble │ │ Poisson  │ │ Elo      │ │ Dixon-  │  │   │
│  │  │ Enginee-  │ │ Model    │ │ Model    │ │ Rating   │ │ Coles   │  │   │
│  │  │ ring      │ │(ensemble)│ │(poisson) │ │(elo.py)  │ │(dixon_) │  │   │
│  │  └───────────┘ └──────────┘ └──────────┘ └──────────┘ └─────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   DATA INFRASTRUCTURE                                 │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │   │
│  │  │ Data     │ │ Feature  │ │ Cache    │ │ Data     │ │ Data      │  │   │
│  │  │ Version- │ │ Store    │ │ Framework│ │ Profiling│ │ Collection│  │   │
│  │  │ ing      │ │(feature_)│ │(cache/)  │ │(data_)  │ │ (data_)   │  │   │
│  │  │          │ │ store/)  │ │          │ │ profiling│ │ collect-/ │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   MONITORING & EXPERIMENT TRACKING                    │   │
│  │  ┌──────────────┐ ┌───────────────────┐ ┌──────────────────────────┐ │   │
│  │  │ Monitoring   │ │ Experiment        │ │ Validation               │ │   │
│  │  │ (monitoring/)│ │ Tracking (experi- │ │ (validation/)            │ │   │
│  │  │              │ │ ment_tracking/)   │ │                          │ │   │
│  │  └──────────────┘ └───────────────────┘ └──────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                         DATA PERSISTENCE                                     │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   PostgreSQL    │  │   CSV Files  │  │   Cache      │  │   Model       │  │
│  │   (ORM Models)  │  │  (data/raw,  │  │   (data/cache│  │   Artifacts   │  │
│  │                 │  │   processed) │  │    /)        │  │   (models/)   │  │
│  └────────────────┘  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                    │                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                        EXTERNAL SOURCES                               │   │
│  │  ┌───────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐  │   │
│  │  │ football-data │  │ The Odds API │  │ StatsBomb  │  │ Transfer-  │  │   │
│  │  │ .co.uk        │  │ (live odds)  │  │ Open Data  │  │ markt      │  │   │
│  │  └───────────────┘  └──────────────┘  └────────────┘  └────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Descriptions

| Layer | Component | Responsibilities |
|-------|-----------|------------------|
| **User Interfaces** | CLI | Run pipelines, manage experiments, compare models, leaderboard |
| | Streamlit Dashboard | Interactive predictions, value bets, backtest charts, World Cup bracket |
| | Scheduler | Automated daily/weekly data refresh, model retraining, backup |
| **Orchestration** | ETL Pipeline | Extract → Validate → Clean → Normalize → Transform → Store |
| | Training Pipeline | Feature engineering, train/test split, ensemble fitting, evaluation |
| | Backtesting | Historical bet simulation, Kelly criterion, ROI analysis |
| **Core Services** | Feature Engineering | Rolling averages, Elo, head-to-head, league position, encoding |
| | Ensemble Model | XGBoost + Logistic Regression + Poisson weighted voting |
| | Poisson Model | Goal distribution model for attack/defence strengths |
| | Elo Rating | Dynamic rating system with goal-margin scaling, home advantage |
| | Dixon-Coles | MLE model with tau correction, recency decay, tournament importance |
| **Infrastructure** | Data Versioning | Parquet-based snapshots, rollback, provenance tracking |
| | Feature Store | Versioned features, registry, validation, batch computation |
| | Cache Framework | SQLite-backed, TTL-based, decorators for memoization |
| | Data Profiling | HTML/JSON/CSV profiling reports, summary statistics |
| **Quality** | Monitoring | ETL metrics, system metrics, data quality, dashboard, alerting |
| | Experiment Tracking | Run lifecycle, hyperparams, metrics, artifacts, compare, export |
| | Validation | 9 check types, engine with severity levels, report generation |
| **Persistence** | PostgreSQL | 22+ ORM models, Alembic migrations, connection pooling |
| | CSV Files | Raw data cache, processed datasets, reports |
| | Model Artifacts | joblib-serialised ensemble models |
| **External** | football-data.co.uk | Historical league match data (CSV downloads) |
| | The Odds API | Live bookmaker odds for value betting |
| | StatsBomb Open Data | xG/xAG/shots data for enhanced features |
| | Transfermarkt | Player data, squad info, lineups |

---

## 2. Folder Structure

```
football_prediction/
│
├── .github/                         # GitHub Actions workflows & templates
├── alembic/                         # Database migration scripts
│   ├── versions/                    #   Generated migration files
│   ├── env.py                       #   Alembic environment config
│   └── script.py.mako               #   Migration template
├── data/                            # Data files (gitignored)
│   ├── raw/                         #   Original CSV downloads
│   ├── processed/                   #   Cleaned, feature-engineered data
│   ├── external/                    #   Cache files, reference data
│   ├── scrapers/                    #   Scraper checkpoints
│   └── cache/                       #   SQLite caches
├── docs/                            # Documentation
│   ├── vault/                       #   Obsidian vault notes
│   └── README.md                    #   This file
├── logs/                            # Application logs
├── models/                          # Serialised trained models (gitignored)
├── notebooks/                       # Jupyter notebooks for EDA
├── reports/                         # Generated reports & charts
│   ├── predictions/                 #   Prediction CSVs
│   ├── backtest/                    #   Backtest results & charts
│   └── predictions_worldcup/        #   World Cup predictions
├── scripts/                         # Utility scripts
│   ├── auto_commit.ps1              #   Hourly GitHub auto-sync
│   ├── backtest_high_conf_away.py   #   High-confidence away backtest
│   └── ...                          #   Other utility/analysis scripts
├── src/                             # Source package
│   ├── __init__.py                  #   Package root
│   ├── config/                      #   Configuration
│   │   ├── settings.py              #     Environment-based settings
│   │   └── logging.py              #     Logging configuration
│   ├── app/                         #   Streamlit dashboard
│   │   ├── dashboard.py             #     Main dashboard
│   │   ├── utils.py                 #     Shared caching & helpers
│   │   └── pages/                   #     Multi-page views
│   │       ├── 1_Predict.py         #     Match prediction UI
│   │       ├── 2_Value_Bets.py      #     Value bet finder UI
│   │       ├── 3_Backtest.py        #     Backtest visualization
│   │       └── 4_WorldCup.py        #     World Cup bracket UI
│   ├── cache/                       #   Caching framework
│   │   ├── decorators.py            #     @cached decorators
│   │   ├── store.py                 #     SQLite-backed cache store
│   │   ├── strategies.py            #     TTL, LRU strategies
│   │   └── __init__.py
│   ├── data/                        #   Data processing
│   │   ├── cleaners.py              #     Data cleaning
│   │   ├── feature_engineering.py   #     Feature engineering
│   │   ├── loader.py                #     Data loading
│   │   ├── preprocessing.py         #     Preprocessing pipeline
│   │   └── __init__.py
│   ├── data_collection/             #   Data ingestion
│   │   ├── collectors.py            #     High-level orchestrator
│   │   ├── cleaners.py              #     Collection-specific cleaning
│   │   └── sources/                 #     Individual source modules
│   │       ├── football_data_co_uk.py   # football-data.co.uk downloader
│   │       ├── football_data_org.py     # football-data.org API
│   │       ├── transfermarkt.py         # Transfermarkt scraper
│   │       ├── worldcup.py              # openfootball worldcup data
│   │       ├── fbref/                   # FBref scraper
│   │       └── understat/              # Understat xG scraper
│   ├── data_profiling/              #   Data profiling
│   │   ├── profiler.py              #     Profiler engine
│   │   ├── reports.py               #     Report generators (HTML/JSON)
│   │   └── __init__.py
│   ├── data_versioning/             #   Dataset version control
│   │   ├── models.py                #     Version metadata models
│   │   ├── storage.py               #     Parquet snapshot storage
│   │   └── __init__.py
│   ├── database/                    #   SQLAlchemy ORM
│   │   ├── base.py                  #     Declarative base
│   │   ├── session.py               #     Session management
│   │   ├── models/                  #     ORM model definitions
│   │   │   ├── match.py             #       Match (fact table)
│   │   │   ├── team.py              #       Team
│   │   │   ├── competition.py       #       Competition/league
│   │   │   ├── season.py            #       Season
│   │   │   ├── country.py           #       Country
│   │   │   ├── stadium.py           #       Stadium
│   │   │   ├── referee.py           #       Referee
│   │   │   ├── player.py            #       Player
│   │   │   ├── lineup.py            #       Lineup
│   │   │   ├── odds.py              #       Odds
│   │   │   ├── prediction.py        #       Model prediction
│   │   │   ├── injury.py            #       Injury tracking
│   │   │   ├── transfer.py          #       Transfer
│   │   │   ├── weather.py           #       Weather
│   │   │   ├── match_statistics.py  #       MatchStatistics
│   │   │   ├── team_form.py         #       TeamForm
│   │   │   ├── team_elo_history.py  #       TeamEloHistory
│   │   │   ├── team_xg_history.py   #       TeamXgHistory
│   │   │   ├── player_match_stats.py #      PlayerMatchStats
│   │   │   ├── betting_result.py    #       BettingResult
│   │   │   ├── closing_line_value.py#       ClosingLineValue
│   │   │   └── expected_value_bet.py#       ExpectedValueBet
│   │   └── repositories/           #     Repository pattern
│   │       ├── base.py              #       Base repository
│   │       ├── match_repository.py  #       Match repository
│   │       └── team_repository.py   #       Team repository
│   ├── etl/                         #   ETL pipeline
│   │   ├── pipeline.py              #     Top-level orchestrator
│   │   ├── extract.py               #     Extract stage
│   │   ├── validate.py              #     Validate stage
│   │   ├── clean.py                 #     Clean stage
│   │   ├── normalize.py             #     Normalize stage
│   │   ├── transform.py             #     Transform stage
│   │   ├── store.py                 #     Store stage
│   │   ├── models.py                #     ETL data models
│   │   ├── tracker.py               #     Job checkpoint tracker
│   │   └── progress.py             #     Progress reporter
│   ├── experiment_tracking/         #   ML Experiment tracking
│   │   ├── models.py                #     Experiment, Run, BestModel, ModelArtifact
│   │   ├── tracker.py               #     ExperimentTracker service
│   │   ├── registry.py              #     ModelRegistry (leaderboard)
│   │   ├── comparator.py            #     ExperimentComparator
│   │   ├── export.py                #     JSON/CSV/HTML export
│   │   ├── cli.py                   #     Experiment CLI
│   │   └── __init__.py
│   ├── feature_store/               #   Feature store
│   │   ├── models.py                #     FeatureDefinition, FeatureValue, etc.
│   │   ├── registry.py              #     FeatureRegistry
│   │   ├── store.py                 #     FeatureStore (CRUD, batches)
│   │   ├── validation.py            #     Feature validation rules
│   │   ├── computers.py             #     FeatureComputer interface
│   │   └── __init__.py
│   ├── importers/                   #   Production importers
│   │   ├── football_data.py         #     FootballDataImporter
│   │   ├── downloader.py            #     DownloadManager
│   │   ├── parser.py                #     CSVParser
│   │   ├── resolver.py              #     EntityResolver
│   │   └── __init__.py
│   ├── monitoring/                  #   Monitoring framework
│   │   ├── models.py                #     ETLMetric, SystemMetric, etc.
│   │   ├── store.py                 #     MonitoringStore (SQLite)
│   │   ├── collectors.py            #     Metric collectors
│   │   ├── monitor.py               #     ETLMonitor service
│   │   ├── dashboard.py             #     HTML dashboard generator
│   │   ├── cli.py                   #     Monitoring CLI
│   │   └── __init__.py
│   ├── models/                      #   Model definitions
│   │   └── __init__.py              #     Re-exports from train.py, ensemble.py
│   ├── scheduler/                   #   Task scheduling
│   │   ├── engine.py                #     TaskEngine
│   │   ├── tasks.py                 #     Pipeline task implementations
│   │   ├── models.py                #     Task, TaskResult, RunReport
│   │   ├── cli.py                   #     Scheduler CLI
│   │   ├── cron_scheduler.py        #     Cron-based scheduling
│   │   └── windows_scheduler.py     #     Windows Task Scheduler
│   ├── scrapers/                    #   Abstract scraper framework
│   │   ├── base.py                  #     BaseScraper, ScrapeResult
│   │   └── __init__.py
│   ├── services/                    #   Business logic services
│   │   ├── prediction_service.py    #     Prediction orchestration
│   │   ├── training_service.py      #     Training orchestration
│   │   └── __init__.py
│   ├── team_normalizer/             #   Team name normalization
│   │   ├── core.py                  #     Team name resolution
│   │   ├── registry.py              #     Team name registry
│   │   └── __init__.py
│   ├── utils/                       #   Cross-cutting utilities
│   │   ├── exceptions.py            #     Custom exception classes
│   │   ├── helpers.py               #     General helpers
│   │   ├── validators.py            #     Input validators
│   │   └── __init__.py
│   ├── validation/                  #   Data validation
│   │   ├── engine.py                #     ValidationEngine
│   │   ├── checks.py                #     Validation check functions
│   │   ├── models.py                #     CheckResult, Severity, ValidationResult
│   │   ├── reporter.py              #     HTML/CSV/JSON report generation
│   │   └── __init__.py
│   ├── backtesting.py               #   Historical bet simulation
│   ├── calibration.py               #   Probability calibration
│   ├── confidence_scoring.py        #   Confidence scoring
│   ├── data_loader.py               #   Data loading utilities
│   ├── dixon_coles.py               #   Dixon-Coles MLE model
│   ├── eda.py                       #   Exploratory data analysis
│   ├── elo.py                       #   Elo rating system
│   ├── ensemble.py                  #   Ensemble model (XGBoost + LR + Poisson)
│   ├── evaluate.py                  #   Model evaluation metrics
│   ├── feature_engineering.py       #   Core feature engineering pipeline
│   ├── hyperparameter_tuning.py     #   Hyper-parameter tuner
│   ├── odds_api.py                  #   The Odds API client
│   ├── odds_processing.py           #   Odds processing & consensus
│   ├── player_info.py               #   Player information features
│   ├── poisson_model.py             #   Poisson goal model
│   ├── predict.py                   #   Match prediction
│   ├── preprocessing.py             #   Data preprocessing
│   ├── time_series_cv.py            #   Time-series cross-validation
│   ├── train.py                     #   Model training
│   ├── value_betting.py             #   Kelly criterion, EV, edge
│   └── xg_features.py              #   Expected Goals features
├── tests/                           # Test suite
│   ├── conftest.py                  #   Global test fixtures
│   ├── test_config/                 #   Configuration tests
│   ├── test_data/                   #   Data processing tests
│   ├── test_database/               #   ORM & repository tests
│   ├── test_etl/                    #   ETL pipeline tests
│   ├── test_fbref/                  #   FBref scraper tests
│   ├── test_feature_store/          #   Feature store tests
│   ├── test_importers/               #   Importer tests
│   ├── test_models/                 #   Model tests
│   ├── test_monitoring/             #   Monitoring tests
│   ├── test_odds_api.py             #   Odds API client tests
│   ├── test_scheduler/              #   Scheduler tests
│   ├── test_scrapers/               #   Scraper tests
│   ├── test_services/               #   Service tests
│   ├── test_team_normalizer/        #   Team normalizer tests
│   ├── test_understat/              #   Understat scraper tests
│   ├── test_utils/                  #   Utility tests
│   └── test_validation/            #   Validation tests
├── config.py                        # Project configuration (dataclasses)
├── pyproject.toml                   # Python project metadata & tool config
├── README.md                        # Public-facing README
├── GUIDE.md                         # User guide
├── CONTRIBUTING.md                  # Contribution guide
├── Makefile                         # Development commands
├── Dockerfile                       # Production Docker image
├── docker-compose.yml               # Docker Compose services
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── alembic.ini                      # Alembic configuration
├── .env.example                     # Environment variable template
├── .pre-commit-config.yaml          # Pre-commit hook configuration
├── scheduler_config.yaml            # Scheduler YAML configuration
│
# ── Standalone Pipeline Scripts ────
├── collect_all_worldcups.py         # Download all World Cup data
├── collect_leagues.py               # Download top 5 league data
├── collect_player_data.py           # Scrape player data from Transfermarkt
├── merge_all_xg_data.py             # Merge StatsBomb xG data
├── train_worldcup.py                # Train World Cup model
├── refresh_worldcup.py              # Automated World Cup refresh
├── predict_worldcup.py              # World Cup prediction
├── bracket_simulator.py             # Knockout bracket simulation
├── run_pipeline.py                  # General prediction pipeline
├── run_first_model.py               # Baseline model runner
├── run_backtest.py                  # Historical backtest runner
├── run_dashboard.py                 # Streamlit dashboard launcher
├── run_combined_pipeline.py         # Full train + cal + eval + backtest
├── find_value_bets.py              # Value bet finder
├── today_value_bets_live.py         # Live value bets
└── train_xgboost.py                 # XGBoost training
```

---

## 3. Database Schema

### 3.1 Entity-Relationship Diagram

The database uses a fully normalised football analytics schema with 22+ tables.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CORE ENTITIES                                      │
│                                                                              │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐                │
│  │ Country │1──N│Competition│1──N│ Season  │1──N│  Match   │ N──1│  Team    │
│  │ (ISO)   │    │(league/  │    │(2024/25)│    │(fact     │     │(club/    │
│  │         │    │ cup)     │    │         │    │ table)   │     │ national) │
│  └─────────┘    └──────────┘    └─────────┘    └────┬─────┘    └──────────┘
│      1                                                │ 1          1        │
│      │                                                │             │        │
│      ▼                                                ▼             ▼        │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Stadium │    │ Referee  │    │Weather  │    │   Lineup │    │   Odds   │ │
│  │(venue)  │    │(official)│    │(match)  │    │(starting │    │(bookmaker│ │
│  │         │    │          │    │         │    │ XI JSON) │    │  odds)   │ │
│  └─────────┘    └──────────┘    └─────────┘    └──────────┘    └──────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      PLAYER ENTITIES                                 │    │
│  │  ┌────────┐ ┌───────────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐  │    │
│  │  │ Player │N│PlayerMatchStats│ │ Injury │ │ Transfer │ │ Match   │  │    │
│  │  │        │1│(perf / match)  │ │        │ │          │ │ Stats   │  │    │
│  │  └────────┘ └───────────────┘ └────────┘ └──────────┘ └─────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ANALYTICS & BETTING                             │    │
│  │  ┌─────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────────┐   │    │
│  │  │ TeamForm│ │ TeamElo      │ │ TeamXg   │ │ Prediction       │   │    │
│  │  │(rolling)│ │ History      │ │ History  │ │ (model output)   │   │    │
│  │  └─────────┘ └──────────────┘ └──────────┘ └──────────────────┘   │    │
│  │  ┌──────────────┐ ┌──────────────────┐ ┌──────────────────┐       │    │
│  │  │ ExpectedValue│ │ ClosingLineValue │ │ BettingResult    │       │    │
│  │  │ Bet          │ │ (CLV analysis)   │ │ (P&L tracking)   │       │    │
│  │  └──────────────┘ └──────────────────┘ └──────────────────┘       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Table Specifications

#### Core Entities

| Table | Description | Key Columns | FK References |
|-------|-------------|-------------|---------------|
| `countries` | ISO-coded country reference | `id`, `name`, `alpha2`, `alpha3`, `fifa_code` | — |
| `competitions` | League/cup/tournament | `id`, `name`, `country_id`, `type`, `tier` | `country_id → countries` |
| `seasons` | Time-bound competition period | `id`, `competition_id`, `name`, `start_date`, `end_date` | `competition_id → competitions` |
| `teams` | Club or national team | `id`, `name`, `short_name`, `country_id`, `founded_year` | `country_id → countries` |
| `stadiums` | Venue | `id`, `name`, `city`, `capacity`, `country_id`, `surface` | `country_id → countries` |
| `referees` | Match official | `id`, `name`, `country_id` | `country_id → countries` |

#### Match Fact Table

| Table | Description | Key Columns | FK References |
|-------|-------------|-------------|---------------|
| `matches` | Central fact table | `id`, `match_date`, `competition_id`, `season_id`, `home_team_id`, `away_team_id`, `stadium_id`, `referee_id`, `home_goals`, `away_goals`, `result`, `round`, `group`, `attendance`, `status` | `competition_id → competitions`, `season_id → seasons`, `home_team_id → teams`, `away_team_id → teams`, `stadium_id → stadiums`, `referee_id → referees` |

#### Match Detail (1:1)

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `match_statistics` | Shots, possession, cards | `match_id`, `home_shots`, `away_shots`, `home_shots_ontarget`, `away_shots_ontarget`, `home_possession`, `away_possession`, `home_corners`, `away_corners`, `home_fouls`, `away_fouls`, `home_yellow_cards`, `away_yellow_cards`, `home_red_cards`, `away_red_cards` |
| `weather` | Match weather conditions | `match_id`, `temperature`, `humidity`, `wind_speed`, `pitch_condition`, `weather_type` |

#### Match Detail (1:N)

| Table | Description | Key Columns | FK References |
|-------|-------------|-------------|---------------|
| `odds` | Multi-bookmaker decimal odds | `match_id`, `bookmaker`, `timestamp`, `odds_home`, `odds_draw`, `odds_away`, `is_opening`, `is_closing` | `match_id → matches` |
| `lineups` | Starting XI + substitutes | `match_id`, `team_id`, `formation`, `starting_xi` (JSON), `substitutes` (JSON), `coach` | `match_id → matches`, `team_id → teams` |

#### Player Entities

| Table | Description | Key Columns | FK References |
|-------|-------------|-------------|---------------|
| `players` | Player biographical info | `id`, `name`, `full_name`, `date_of_birth`, `position`, `nationality_id`, `market_value`, `height`, `foot` | `nationality_id → countries` |
| `player_match_stats` | Per-match performance | `player_id`, `match_id`, `team_id`, `minutes_played`, `goals`, `assists`, `xg`, `xag`, `shots`, `key_passes`, `rating`, `pass_accuracy` | `player_id → players`, `match_id → matches`, `team_id → teams` |
| `injuries` | Injury tracking | `player_id`, `injury_date`, `return_date`, `type`, `severity`, `description` | `player_id → players` |
| `transfers` | Transfer activity | `player_id`, `from_team_id`, `to_team_id`, `transfer_date`, `fee`, `loan_flag` | `player_id → players`, `from_team_id → teams`, `to_team_id → teams` |

#### Analytics & Betting

| Table | Description | Key Columns | FK References |
|-------|-------------|-------------|---------------|
| `team_forms` | Pre-computed rolling form | `team_id`, `match_id`, `last_5_points`, `last_10_points`, `last_20_points`, `avg_goals_scored`, `avg_goals_conceded`, `win_rate_home`, `win_rate_away` | `team_id → teams`, `match_id → matches` |
| `team_elo_histories` | Elo rating snapshots | `team_id`, `match_id`, `elo_before`, `elo_after`, `elo_change`, `home_advantage` | `team_id → teams`, `match_id → matches` |
| `team_xg_histories` | xG history | `team_id`, `match_id`, `xg_for`, `xg_against`, `shots_for`, `shots_against`, `source` | `team_id → teams`, `match_id → matches` |
| `predictions` | Model output | `match_id`, `model_name`, `model_version`, `prob_home`, `prob_draw`, `prob_away`, `predicted_result`, `confidence`, `features_hash` | `match_id → matches` |
| `expected_value_bets` | EV calculations | `match_id`, `bookmaker`, `outcome`, `model_probability`, `odds`, `edge`, `expected_value`, `kelly_stake`, `kelly_fraction` | `match_id → matches` |
| `closing_line_values` | Line movement | `match_id`, `outcome`, `opening_odds`, `closing_odds`, `movement_pct`, `clv` | `match_id → matches` |
| `betting_results` | P&L tracking | `match_id`, `stake`, `odds`, `outcome_bet`, `actual_result`, `profit_loss`, `roi` | `match_id → matches` |

### 3.3 Constraints & Indexes

```sql
-- Match: No team can play itself
CHECK (home_team_id != away_team_id)

-- Match: Valid result values
CHECK (result IN ('H', 'D', 'A'))

-- Match: Goals must be non-negative
CHECK (home_goals >= 0 AND away_goals >= 0)

-- Odds: Unique combination per match+bookmaker+timestamp
UNIQUE (match_id, bookmaker, timestamp)

-- Prediction: Unique per match+model
UNIQUE (match_id, model_name, model_version)

-- Key indexes on frequently queried columns
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_team ON matches(home_team_id);
CREATE INDEX idx_matches_competition ON matches(competition_id);
CREATE INDEX idx_odds_match ON odds(match_id);
```

---

## 4. ETL Workflow

### 4.1 Pipeline Overview

The ETL pipeline follows a strict six-stage architecture:

```
┌──────────┐    ┌──────────┐    ┌────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐
│  EXTRACT │───▶│ VALIDATE │───▶│  CLEAN │───▶│ NORMALIZE │───▶│ TRANSFORM │───▶│  STORE │
│          │    │          │    │        │    │           │    │           │    │        │
│ CSV/API  │    │ 9 checks│    │ Dedup  │    │ Team name │    │ Feature   │    │ DB/CSV │
│ download │    │ Schema  │    │ NaN    │    │ standard  │    │ compute   │    │ upsert │
│          │    │ rules   │    │ fix    │    │ ISO dates │    │           │    │        │
└──────────┘    └──────────┘    └────────┘    └───────────┘    └───────────┘    └────────┘
```

### 4.2 Stage Details

#### Extract
- **Input:** Raw data from external sources (football-data.co.uk CSV, openfootball JSON, API responses)
- **Output:** `list[dict[str, Any]]` — unvalidated rows
- **Components:** `BaseExtractor` (abstract), `CSVExtractor`, `APIExtractor`
- **Features:** Retry logic, integrity checks (row count, column presence), download progress

#### Validate
- **Input:** Raw rows from Extract
- **Output:** Same rows (passed through), with `ValidationResult` attached
- **Checks (9 built-in):**
  1. **Duplicate Matches** — same date + home team + away team
  2. **Invalid Dates** — future dates, impossible dates, missing dates
  3. **Invalid Odds** — negative odds, zero odds, NaN odds
  4. **Missing Goals** — rows with results but no goal data
  5. **Missing Teams** — blank/null team names
  6. **Incorrect Leagues** — league code mismatches
  7. **Invalid Statistics** — negative shots, >100% possession
  8. **Duplicate IDs** — duplicate match IDs in the dataset
  9. **Impossible Scores** — negative scores, unrealistically high scores

#### Clean
- **Input:** Validated rows
- **Output:** Cleaned rows with missing values handled
- **Operations:** Deduplication, missing value imputation (drop/fill_zero/fill_median), outlier removal, data type coercion

#### Normalize
- **Input:** Cleaned rows
- **Output:** Rows with standardised values
- **Operations:** Team name resolution (via `TeamNormalizer` + `EntityResolver`), date standardisation (ISO 8601), league code normalisation, text cleaning (whitespace, casing)

#### Transform
- **Input:** Normalised rows
- **Output:** Feature-engineered rows
- **Operations:** Column mapping, data type conversion, computed fields (goal difference, total goals), feature creation hooks

#### Store
- **Input:** Transformed rows
- **Output:** Persisted rows (database upsert or CSV file)
- **Features:** Batch commit, upsert by unique key, progress tracking, rollback on failure

### 4.3 Pipeline Configuration

```python
from src.etl import ETLPipeline
from src.etl.extract import CSVExtractor
from src.etl.clean import DataCleaner
from src.etl.normalize import DataNormalizer
from src.etl.store import DatabaseStore
from src.database.models import Match

pipeline = ETLPipeline(
    name="import_matches",
    source="football-data-co-uk",
    extractor=CSVExtractor("data/raw/results.csv"),
    cleaner=DataCleaner(fill_strategy="drop"),
    normalizer=DataNormalizer(
        team_name_columns=["home_team", "away_team"],
        date_columns=["date"],
    ),
    store=DatabaseStore(Match, unique_columns=["match_id"]),
    checkpoint=True,     # Enable job checkpointing for resume
    parallel=False,      # Sequential execution
    max_workers=4,
)

result = pipeline.run()
print(result.status)          # StageStatus.SUCCESS
print(result.total_duration_seconds)
```

### 4.4 Error Handling & Checkpoints

- **Per-stage abort:** Failed stages abort the pipeline immediately
- **Checkpoints:** When `checkpoint=True`, each successful stage is persisted to a SQLite tracker, allowing pipeline to resume from the last completed stage on restart
- **Retry:** The Scheduler wraps pipeline execution with configurable retry logic (with linear backoff)
- **Warnings:** Stages returning `StageStatus.WARNING` continue but flag issues
- **Progress bars:** Visual progress via `tqdm` for each stage

```
Pipeline Flow with Checkpoints:

                      ┌──────────────────┐
                      │  Job Created     │
                      │  job_id = UUID   │
                      └────────┬─────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Stage 1: EXTRACT               │
              │  ✓ Success → mark_stage_done()  │
              │  ✗ Fail → mark_stage_failed()   │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Stage 2: VALIDATE              │
              │  ✓ Success → mark_stage_done()  │
              └────────────────┬────────────────┘
                               │
                              ...
                               │
              ┌────────────────▼────────────────┐
              │  Stage 6: STORE                 │
              │  ✓ Success → delete_checkpoint()│
              └────────────────┬────────────────┘
                               │
                      ┌────────▼─────────┐
                      │ Pipeline Complete│
                      └──────────────────┘
```

---

## 5. Scheduler Workflow

### 5.1 Architecture

The scheduler orchestrates automated execution of pipeline tasks with dependency resolution, retry logic, and structured reporting.

```
┌──────────────────────────────────────────────────────────────────┐
│                     SCHEDULER ARCHITECTURE                       │
│                                                                  │
│  ┌───────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │   CLI     │───▶│  Engine  │───▶│      Task Map            │  │
│  │           │    │          │    │  ┌──────────────────────┐ │  │
│  │  run      │    │resolve   │    │  │ download_fixtures   │ │  │
│  │  install  │    │ order    │    │  │ validate_data       │ │  │
│  │  remove   │    │execute   │    │  │ clean_data          │ │  │
│  │  status   │    │ retry    │    │  │ update_database     │ │  │
│  │           │    │ report   │    │  │ backup_database     │ │  │
│  │           │    │          │    │  │ generate_logs       │ │  │
│  └───────────┘    └──────────┘    │  └──────────────────────┘ │  │
│                                    └──────────────────────────┘  │
│                                    ┌──────────────────────────┐  │
│                                    │     ScheduleConfig       │  │
│                                    │  - tasks: Task[]         │  │
│                                    │  - abort_on_failure      │  │
│                                    │  - pipeline_name         │  │
│                                    └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Task Dependencies

Tasks are executed in topological order based on their declared dependencies:

```
download_fixtures
    │
    ▼
validate_data
    │
    ▼
update_database
    │
    ├────────────────────┐
    ▼                    ▼
clean_data        backup_database
    │                    │
    └────────┬───────────┘
             ▼
     generate_logs
```

### 5.3 Task Implementations

| Task | Function | Description |
|------|----------|-------------|
| `download_fixtures` | `download_fixtures()` | Download new match data from football-data.co.uk via `FootballDataImporter` |
| `validate_data` | `validate_data()` | Run all 9 validation checks, generate HTML/CSV/JSON reports |
| `update_database` | `update_database()` | Ingest cleaned CSV into PostgreSQL, retrain ensemble model, save artifact |
| `clean_data` | `clean_data()` | Deduplicate CSVs, archive old raw files, remove stale checkpoints |
| `backup_database` | `backup_database()` | Create DB dump (pg_dump for PostgreSQL, file copy for SQLite), enforce retention |
| `generate_logs` | `generate_logs()` | Rotate log files, archive old reports, write structured JSON run summary |

### 5.4 Retry Logic

Each task has configurable retry settings:

```
Task execution with retry:

    Execute
       │
       ├── Success ──▶ Return TaskResult
       │
       └── Failure (attempt < retry_count)
              │
              ▼
        Wait (2s × attempt)
              │
              ▼
        Retry ──▶ Success ──▶ Return TaskResult
              │
              └── Failure (attempt >= retry_count)
                     │
                     ▼
              Return TaskResult(status=FAILED)
```

### 5.5 Windows Task Scheduler Integration

```bat
setup_scheduler.bat
  Creates a Windows Scheduled Task:
  - Name: "FootballPredictionRefresh"
  - Triggers: Every 6 hours
  - Action: python refresh_worldcup.py --quiet --log-file refresh.log
  - Run as: Current user
  - Start in: Project directory
```

---

## 6. Validation Workflow

### 6.1 Validation Engine

The `ValidationEngine` orchestrates all 9 built-in data quality checks:

```
┌──────────────────────────────────────────────────────────────────┐
│                       VALIDATION ENGINE                          │
│                                                                  │
│  Input: list[dict] + source_name                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   CHECK REGISTRY                          │   │
│  │                                                           │   │
│  │  1. check_duplicate_matches ──▶ Find duplicate fixtures   │   │
│  │  2. check_invalid_dates     ──▶ Bad/missing/empty dates   │   │
│  │  3. check_invalid_odds      ──▶ Negative/zero odds        │   │
│  │  4. check_missing_goals     ──▶ Result but no goals       │   │
│  │  5. check_missing_teams     ──▶ Blank team names          │   │
│  │  6. check_incorrect_leagues ──▶ League code mismatches    │   │
│  │  7. check_invalid_statistics ──▶ Impossible stats         │   │
│  │  8. check_duplicate_ids     ──▶ Duplicate match IDs       │   │
│  │  9. check_impossible_scores ──▶ Negative/crazy scores     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                       │
│                     Execute all                                    │
│                           │                                       │
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │                   VALIDATION RESULT                         │  │
│  │  - source_name, total_rows                                 │  │
│  │  - passed_checks, total_checks, total_violations           │  │
│  │  - passed: bool                                            │  │
│  │  - checks: list[CheckResult]                               │  │
│  │  - get_violations() → list of detailed violations          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                       │
│                    ┌──────┴──────┐                                │
│                    ▼             ▼                                 │
│              report_to_html  report_to_csv  report_to_json        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Check Result Model

Each check function returns a `CheckResult`:

```python
@dataclass
class CheckResult:
    check_name: str
    description: str
    severity: Severity  # ERROR = 3, WARNING = 2, INFO = 1
    passed: bool
    total_rows: int
    violation_count: int
    violations: list[dict]  # Detailed violation records
```

**Severity levels:**
- `ERROR` — Must fix before proceeding (e.g., missing teams)
- `WARNING` — Should investigate but not blocking (e.g., missing odds)
- `INFO` — Informational only (e.g., duplicate matches found)

### 6.3 Report Generation

Validation results can be exported in three formats:

```python
result.to_html("reports/validation_20260101.html")  # Color-coded HTML report
result.to_csv("reports/validation_20260101.csv")    # Flat CSV table
result.to_json("reports/validation_20260101.json")  # Structured JSON
```

---

## 7. Data Flow Diagrams

### 7.1 End-to-End Data Flow (World Cup Pipeline)

```
openfootball ──▶ collect_all_worldcups.py ──▶ data/raw/worldcup_all.csv
                                                   │
                                                   ▼
                                              Preprocessing
                                                   │
                                                   ▼
                                          data/processed/results_clean.csv
                                                   │
                                                   ▼
                                          Feature Engineering
                                           (src/feature_engineering.py)
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          │                        │                        │
                          ▼                        ▼                        ▼
                     Elo Ratings             Rolling Stats           Poisson Model
                     (src/elo.py)            (5/10/20 window)        (src/poisson_model.py)
                          │                        │                        │
                          └────────────────────────┼────────────────────────┘
                                                   │
                                                   ▼
                                          Feature Matrix (X, y)
                                                   │
                                                   ▼
                                          Train/Val/Test Split
                                          (chronological, no shuffle)
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          │                        │                        │
                          ▼                        ▼                        ▼
                     XGBoost                  Logistic              Poisson
                     (80 trees)               Regression            Model
                          │                        │                        │
                          └────────────────────────┼────────────────────────┘
                                                   │
                                          Weight Grid Search
                                          (minimize val log-loss)
                                                   │
                                                   ▼
                                          Ensemble Model
                                          (weighted average)
                                                   │
                                                   ▼
                                          Prediction
                                          (predict_proba → [away, draw, home])
                                                   │
                                                   ▼
                                          reports/predictions_worldcup/
```

### 7.2 Data Flow (FootballData Importer)

```
football-data.co.uk CSV
         │
         ▼
  DownloadManager
  (retry, integrity check)
         │
         ▼
  Raw CSV on disk
  data/raw/football-data/E0_2425.csv
         │
         ▼
  CSVParser
  (strict mode, column validation)
         │
         ▼
  list[dict] — parsed rows
         │
         ▼
  EntityResolver
  (team_id → DB lookup, auto-create)
         │
         ▼
  list[dict] — resolved rows with FK IDs
         │
         ▼
  Incremental Filter
  (skip matches already in DB)
         │
         ▼
  DatabaseStore.upsert()
  (batch commit, 500 rows/batch)
         │
         ▼
  PostgreSQL matches table
```

### 7.3 Data Flow (Feature Engineering)

```
Input DataFrames (df)
    │
    ├── 1. Sort chronologically by date
    │
    ├── 2. Add Elo features
    │      └── Home_Elo, Away_Elo, Elo_Difference, Elo_Home_Advantage
    │
    ├── 3. Add odds processing features
    │      └── implied probabilities, consensus, market movement
    │
    ├── 4. Add player info features (optional)
    │      └── squad depth, injuries, rotation indicators
    │
    ├── 5. Add xG features
    │      └── rolling xG/xGA/xGD, expected points
    │
    ├── 6. Add Poisson-derived expected goals
    │      └── expected goals home/away, expected total goals
    │
    ├── 7. Add Dixon-Coles features (optional)
    │      └── attack/defence strength, home advantage, rho
    │
    ├── 8. Add competition importance
    │      └── numeric weight (0.4–2.5)
    │
    ├── 9. Add rolling team features
    │      └── h_points_avg5, h_goals_scored_avg5, a_goal_diff_avg10, ...
    │
    ├── 10. Add head-to-head features
    │       └── h2h_home_points_avg, h2h_home_win_rate, ...
    │
    ├── 11. Add league position features
    │       └── h_league_position, a_league_position, position_diff
    │
    ├── 12. Encode categoricals
    │       └── label/onehot/target encoding of team names
    │
    ├── 13. Add attack/defence ratios
    │       └── h_attack_ratio5 = h_goals_scored_avg5 / league_avg
    │
    └── 14. Separate X (features) and y (target)
            └── Drop: target, result, home_goals, away_goals, date, season
```

### 7.4 Data Flow (Ensemble Model Training)

```
                  ┌─────────────────────────┐
                  │   X_train, X_val, X_test │
                  │   y_train, y_val, y_test │
                  │   df_train, df_val, df_test │
                  └────────────┬────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌──────────────┐  ┌────────────┐  ┌──────────────┐
     │   XGBoost    │  │  Logistic  │  │   Poisson    │
     │  80 trees    │  │ Regression │  │    Model     │
     │  depth 5     │  │  C=1.0     │  │  min_matches │
     │  lr=0.05     │  │  balanced  │  │  max_goals=8 │
     └──────┬───────┘  └─────┬──────┘  └──────┬───────┘
            │                │                 │
            └────────────────┼─────────────────┘
                             │
                             ▼
              ┌─────────────────────────┐
              │  Validation Predictions  │
              │  per model: (n_val, 3)  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Weight Grid Search    │
              │  enumerate combinations │
              │  (step=0.10, sum=1.0)   │
              │  → min log_loss         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Apply Constraints     │
              │  (min/max per model)    │
              │  → renormalise          │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Trained Ensemble      │
              │  models + weights       │
              └─────────────────────────┘
```

---

## 8. Sequence Diagrams

### 8.1 ETL Pipeline Execution

```
User/CLI              ETLPipeline          Extractor     Validator    Cleaner    Normalizer    Transformer      Store
   │                      │                   │             │           │           │              │            │
   │    pipeline.run()    │                   │             │           │           │              │            │
   │─────────────────────▶│                   │             │           │           │              │            │
   │                      │                   │             │           │           │              │            │
   │                      │  _run_extract()   │             │           │           │              │            │
   │                      │──────────────────▶│             │           │           │              │            │
   │                      │                   │  fetch()    │           │           │              │            │
   │                      │                   │─────────────│──────────▶│           │              │            │
   │                      │                   │◀────────────│───────────│           │              │            │
   │                      │◀──────────────────│             │           │           │              │            │
   │                      │                   │             │           │           │              │            │
   │                      │  _run_validate()  │             │           │           │              │            │
   │                      │───────────────────────────────────────────────▶│           │              │            │
   │                      │◀───────────────────────────────────────────────│           │              │            │
   │                      │                   │             │           │           │              │            │
   │                      │  _run_clean()     │             │           │           │              │            │
   │                      │──────────────────────────────────────────────────────────▶│              │            │
   │                      │◀──────────────────────────────────────────────────────────│              │            │
   │                      │                   │             │           │           │              │            │
   │                      │  _run_normalize() │             │           │           │              │            │
   │                      │──────────────────────────────────────────────────────────────────────▶│            │
   │                      │◀──────────────────────────────────────────────────────────────────────│            │
   │                      │                   │             │           │           │              │            │
   │                      │  _run_transform() │             │           │           │              │            │
   │                      │──────────────────────────────────────────────────────────────────────────────────▶│
   │                      │◀──────────────────────────────────────────────────────────────────────────────────│
   │                      │                   │             │           │           │              │            │
   │                      │  _run_store()     │             │           │           │              │            │
   │                      │──────────────────────────────────────────────────────────────────────────────────────▶
   │                      │                   │             │           │           │              │            │
   │                      │                   │             │           │           │              │   write() │
   │                      │                   │             │           │           │              │◀──────────│
   │                      │◀────────────────────────────────────────────────────────────────────────│───────────│
   │                      │                   │             │           │           │              │            │
   │  ◀─── ETLResult ────│                   │             │           │           │              │            │
   │                      │                   │             │           │           │              │            │
```

### 8.2 Scheduler Pipeline Execution

```
CLI/Scheduler       TaskEngine          download_fixtures   validate_data   update_database   clean_data   backup_db   generate_logs
   │                    │                     │                  │               │              │            │              │
   │  run_all()         │                     │                  │               │              │            │              │
   │───────────────────▶│                     │                  │               │              │            │              │
   │                    │                     │                  │               │              │            │              │
   │                    │  resolve_order()    │                  │               │              │            │              │
   │                    │  ──────────▶        │                  │               │              │            │              │
   │                    │  [download, validate, update, clean, backup, logs]      │              │            │              │
   │                    │                     │                  │               │              │            │              │
   │                    │  download_fixtures  │                  │               │              │            │              │
   │                    │────────────────────▶│                  │               │              │            │              │
   │                    │◀────────────────────│                  │               │              │            │              │
   │                    │                     │                  │               │              │            │              │
   │                    │  validate_data      │                  │               │              │            │              │
   │                    │───────────────────────────────────────▶│               │              │            │              │
   │                    │◀───────────────────────────────────────│               │              │            │              │
   │                    │                     │                  │               │              │            │              │
   │                    │  update_database    │                  │               │              │            │              │
   │                    │──────────────────────────────────────────────────────▶│              │            │              │
   │                    │◀──────────────────────────────────────────────────────│              │            │              │
   │                    │                     │                  │               │              │            │              │
   │                    │  clean_data         │                  │               │              │            │              │
   │                    │─────────────────────────────────────────────────────────────────────▶│            │              │
   │                    │◀─────────────────────────────────────────────────────────────────────│            │              │
   │                    │                     │                  │               │              │            │              │
   │                    │  backup_database    │                  │               │              │            │              │
   │                    │──────────────────────────────────────────────────────────────────────────────────▶│              │
   │                    │◀──────────────────────────────────────────────────────────────────────────────────│              │
   │                    │                     │                  │               │              │            │              │
   │                    │  generate_logs      │                  │               │              │            │              │
   │                    │────────────────────────────────────────────────────────────────────────────────────────────────▶│
   │                    │◀────────────────────────────────────────────────────────────────────────────────────────────────│
   │                    │                     │                  │               │              │            │              │
   │  ◀─── RunReport───│                     │                  │               │              │            │              │
   │                    │                     │                  │               │              │            │              │
```

### 8.3 Value Betting Flow

```
User        ValueBetting        Model           OddsAPI          KellyCalculator
  │              │                 │                │                  │
  │  find_bets() │                 │                │                  │
  │─────────────▶│                 │                │                  │
  │              │  predict()      │                │                  │
  │              │────────────────▶│                │                  │
  │              │◀─── probs ─────│                │                  │
  │              │                 │                │                  │
  │              │  fetch_odds()   │                │                  │
  │              │─────────────────────────────────▶│                  │
  │              │◀──────── odds_with_margin ──────│                  │
  │              │                 │                │                  │
  │              │  extract_margin()               │                  │
  │              │  compute_fair_probs()           │                  │
  │              │                 │                │                  │
  │              │  For each match:                │                  │
  │              │    edge = model_prob - fair_prob│                  │
  │              │    EV = edge * decimal_odds     │                  │
  │              │                 │                │                  │
  │              │  For EV+ bets:  │                │                  │
  │              │    kelly_stake()│                │                  │
  │              │──────────────────────────────────────────────────▶│
  │              │◀──────────── stake ─────────────│────────────────│
  │              │                 │                │                  │
  │  ◀─ List[ValueBet] ──│                │                  │
  │              │                 │                │                  │
```

### 8.4 Experiment Tracking Lifecycle

```
User              ExperimentTracker       Database         Comparator       Exporter
  │                     │                   │                  │               │
  │  create_experiment()│                   │                  │               │
  │────────────────────▶│                   │                  │               │
  │                     │  INSERT INTO      │                  │               │
  │                     │  experiments      │                  │               │
  │                     │──────────────────▶│                  │               │
  │  ◀── Experiment ────│                   │                  │               │
  │                     │                   │                  │               │
  │  start_run()        │                   │                  │               │
  │────────────────────▶│                   │                  │               │
  │                     │  INSERT INTO runs │                  │               │
  │                     │──────────────────▶│                  │               │
  │  ◀── Run (running)──│                   │                  │               │
  │                     │                   │                  │               │
  │  finish_run()       │                   │                  │               │
  │────────────────────▶│                   │                  │               │
  │                     │  UPDATE runs      │                  │               │
  │                     │  SET metrics,     │                  │               │
  │                     │  duration, status │                  │               │
  │                     │──────────────────▶│                  │               │
  │  ◀── Run (completed)│                   │                  │               │
  │                     │                   │                  │               │
  │  register_best()    │                   │                  │               │
  │────────────────────▶│                   │                  │               │
  │                     │  INSERT INTO      │                  │               │
  │                     │  best_models      │                  │               │
  │                     │──────────────────▶│                  │               │
  │                     │                   │                  │               │
  │  list_experiments() │                   │                  │               │
  │────────────────────▶│                   │                  │               │
  │  ◀── list[Experiment]                  │                  │               │
  │                     │                   │                  │               │
  │  compare_runs()     │                   │                  │               │
  │────────────────────────────────────────────────────────▶│               │
  │  ◀── comparison dict                   │                  │               │
  │                     │                   │                  │               │
  │  export_html()      │                   │                  │               │
  │────────────────────────────────────────────────────────────────────────▶│
  │  ◀── HTML report                        │                  │               │
```

---

## 9. Class Diagrams

### 9.1 ETL Pipeline Classes

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ETL PACKAGE (src/etl/)                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────┐           │
│  │                    ETLPipeline                       │           │
│  ├─────────────────────────────────────────────────────┤           │
│  │ - name: str                                          │           │
│  │ - source: str                                        │           │
│  │ - config: ETLConfig                                  │           │
│  │ - extractor: BaseExtractor                           │           │
│  │ - validator: DataValidator                           │           │
│  │ - cleaner: DataCleaner                               │           │
│  │ - normalizer: DataNormalizer                         │           │
│  │ - transformer: DataTransformer                       │           │
│  │ - store: DataStore                                   │           │
│  │ - tracker: JobTracker (optional)                     │           │
│  │ - progress: ProgressReporter                         │           │
│  ├─────────────────────────────────────────────────────┤           │
│  │ + run(**kwargs) → ETLResult                         │           │
│  │ - _run_extract(data, **kwargs) → StageResult        │           │
│  │ - _run_validate(data, **kwargs) → StageResult       │           │
│  │ - _run_clean(data, **kwargs) → StageResult          │           │
│  │ - _run_normalize(data, **kwargs) → StageResult      │           │
│  │ - _run_transform(data, **kwargs) → StageResult      │           │
│  │ - _run_store(data, **kwargs) → StageResult          │           │
│  └─────────────────────────────────────────────────────┘           │
│                              │                                       │
│     ┌────────────────────────┼────────────────────────────┐          │
│     │                        │                            │          │
│     ▼                        ▼                            ▼          │
│  ┌────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│  │ BaseExtract│    │  DataValidator   │    │   DataCleaner    │     │
│  │  (ABC)     │    │                  │    │                  │     │
│  ├────────────┤    │ - checks: list   │    │ - fill_strategy  │     │
│  │ + run()    │    ├──────────────────┤    ├──────────────────┤     │
│  │   → Stage  │    │ + run() → Stage  │    │ + run() → Stage  │     │
│  │   Result   │    │   Result         │    │   Result         │     │
│  └────────────┘    └──────────────────┘    └──────────────────┘     │
│                                                                     │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│   │ DataNormalizer   │    │ DataTransformer  │    │  DataStore   │ │
│   ├──────────────────┤    ├──────────────────┤    ├──────────────┤ │
│   │ - team_cols      │    │ - operations     │    │ - model_class│ │
│   │ - date_cols      │    ├──────────────────┤    │ - batch_size │ │
│   ├──────────────────┤    │ + run() → Stage  │    ├──────────────┤ │
│   │ + run() → Stage  │    │   Result         │    │ + write() →  │ │
│   │   Result         │    └──────────────────┘    │   StageResult│ │
│   └──────────────────┘                           └──────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                        DATA MODELS                           │  │
│  │  ┌───────────┐  ┌──────────────┐  ┌────────────┐  ┌──────┐  │  │
│  │  │ ETLConfig │  │  ETLResult   │  │ StageResult│  │Stage │  │  │
│  │  │           │  │              │  │            │  │Status│  │  │
│  │  └───────────┘  └──────────────┘  └────────────┘  └──────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Database ORM Models

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATABASE MODELS (src/database/models/)         │
│                                                                  │
│  Base (DeclarativeBase)                                          │
│      │                                                           │
│      ├── Country                                                 │
│      ├── Competition                                             │
│      ├── Season                                                  │
│      ├── Team                                                    │
│      ├── Stadium                                                 │
│      ├── Referee                                                 │
│      ├── Player                                                  │
│      ├── Match ◄──────────────────────┐                          │
│      │    ├── MatchStatistics         │                          │
│      │    ├── Weather                 │                          │
│      │    ├── Odds (1:N)              │                          │
│      │    ├── Lineup (1:N)            │                          │
│      │    ├── Prediction              │                          │
│      │    ├── ExpectedValueBet        │                          │
│      │    ├── ClosingLineValue        │                          │
│      │    ├── BettingResult           │                          │
│      │    ├── TeamForm                │                          │
│      │    ├── TeamEloHistory          │                          │
│      │    └── TeamXgHistory           │                          │
│      ├── PlayerMatchStats             │                          │
│      ├── Injury                       │                          │
│      └── Transfer                     │                          │
│                                                                  │
│  ┌──────────────────┐  ┌────────────────────┐                   │
│  │  BaseRepository  │  │   MatchRepository  │                   │
│  │  (ABC)           │  │                    │                   │
│  ├──────────────────┤  ├────────────────────┤                   │
│  │ + get_by_id()    │  │ + get_by_date()    │                   │
│  │ + find()         │  │ + get_by_team()    │                   │
│  │ + create()       │  │ + get_upcoming()   │                   │
│  │ + update()       │  │ + get_competition()│                   │
│  │ + delete()       │  └────────────────────┘                   │
│  └──────────────────┘                                           │
└──────────────────────────────────────────────────────────────────┘
```

### 9.3 Ensemble Model

```
┌────────────────────────────────────────────────────────────────────┐
│                       ENSEMBLE MODEL                               │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    EnsembleModel                              │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ - models: dict[str, Any]                                     │  │
│  │ - weights: dict[str, float]                                  │  │
│  │ - _poisson_model: PoissonModel | None                        │  │
│  │ - _lb: LabelBinarizer | None                                 │  │
│  │ - _train_log_loss: float | None                              │  │
│  │ - _val_log_loss: float | None                                │  │
│  │ - _individual_log_losses: dict[str, float]                   │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ + trained: bool (property)                                   │  │
│  │ + weight_summary: str (property)                             │  │
│  │ + fit(X_train, y_train, X_val, y_val, df_train, df_val)     │  │
│  │ + predict_proba(X, df_raw) → np.ndarray                     │  │
│  │ + predict(X, df_raw) → np.ndarray                           │  │
│  │ + evaluate(X_test, y_test, df_test) → dict                  │  │
│  │ + save(path) → str                                          │  │
│  │ + load(path) → EnsembleModel (classmethod)                   │  │
│  │ - _train_ml_models(X_train, y_train, X_val, y_val)          │  │
│  │ - _train_poisson_model(df_train)                             │  │
│  │ - _get_all_predictions(X, df_raw, y_true, label)            │  │
│  │ - _optimise_weights(preds, y_val) → dict                    │  │
│  │ - _apply_weight_constraints()                                │  │
│  │ - _apply_weights(preds, weights) → np.ndarray (static)      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│                     ┌──────────────────┐                          │
│                     │   PoissonModel   │                          │
│                     ├──────────────────┤                          │
│                     │ - min_matches    │                          │
│                     │ - max_goals      │                          │
│                     │ - team_strengths │                          │
│                     ├──────────────────┤                          │
│                     │ + fit(df)        │                          │
│                     │ + predict_match()│                          │
│                     │ + predict_matches()                          │
│                     │ + add_poisson_features()                     │
│                     └──────────────────┘                          │
└────────────────────────────────────────────────────────────────────┘
```

### 9.4 Scheduler Classes

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SCHEDULER PACKAGE (src/scheduler/)                │
│                                                                      │
│  ┌──────────────────────────┐     ┌────────────────────────────┐    │
│  │       TaskEngine         │     │       TaskResult           │    │
│  ├──────────────────────────┤     ├────────────────────────────┤    │
│  │ - config: ScheduleConfig │     │ - task_name: str           │    │
│  │ - _task_map: dict        │     │ - status: TaskStatus       │    │
│  ├──────────────────────────┤     │ - started_at: datetime     │    │
│  │ + register(name, func)   │     │ - completed_at: datetime   │    │
│  │ + run_all(names) → Report│     │ - output: str              │    │
│  │ - _resolve_order(tasks)  │     │ - error: str | None        │    │
│  │ - _execute_with_retry()  │     │ - warnings: list[str]      │    │
│  │ - _check_dependencies()  │     │ - records_processed: int   │    │
│  └──────────────────────────┘     │ - duration_seconds: float  │    │
│                                    └────────────────────────────┘    │
│  ┌──────────────────────────┐     ┌────────────────────────────┐    │
│  │         Task             │     │        RunReport           │    │
│  ├──────────────────────────┤     ├────────────────────────────┤    │
│  │ - name: str              │     │ - pipeline_name: str       │    │
│  │ - description: str       │     │ - started_at: datetime     │    │
│  │ - enabled: bool          │     │ - completed_at: datetime   │    │
│  │ - dependencies: list[str]│     │ - task_results: dict       │    │
│  │ - retry_count: int       │     │ - total_tasks: int         │    │
│  │ - timeout_seconds: int   │     │ - succeeded/failed/skipped │    │
│  │ - schedule: str (cron)   │     │ - errors: list[str]        │    │
│  └──────────────────────────┘     │ - duration_seconds: float  │    │
│                                    └────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │   ScheduleConfig     │   │   TaskStatus     │   │   Scheduler  │ │
│  │                      │   │   (Enum)         │   │   CLI        │ │
│  │ - tasks: list[Task]  │   │                  │   └──────────────┘ │
│  │ - pipeline_name      │   │ RUNNING          │                     │
│  │ - abort_on_failure   │   │ SUCCESS          │                     │
│  │ - report_dir         │   │ FAILED           │                     │
│  │ - backup_dir         │   │ SKIPPED          │                     │
│  │ - log_dir            │   │ WARNING          │                     │
│  │ - backup_retention   │   └──────────────────┘                     │
│  └──────────────────────┘                                            │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.5 Validation Classes

```
┌──────────────────────────────────────────────────────────────────────┐
│                    VALIDATION PACKAGE (src/validation/)               │
│                                                                      │
│  ┌──────────────────────────────┐                                    │
│  │      ValidationEngine        │                                    │
│  ├──────────────────────────────┤                                    │
│  │ - checks: list[tuple]        │                                    │
│  │ - verbose: bool              │                                    │
│  ├──────────────────────────────┤                                    │
│  │ + run(data, source_name) →   │                                    │
│  │   ValidationResult           │                                    │
│  │ + run_selected(data, names)  │                                    │
│  └──────────────────────────────┘                                    │
│            │                                                         │
│            │ executes                                                │
│            ▼                                                         │
│  ┌──────────────────────────────┐  ┌──────────────────────────┐     │
│  │      Check Functions         │  │      Severity (Enum)     │     │
│  │                              │  │                          │     │
│  │ check_duplicate_matches()    │  │ ERROR   = 3              │     │
│  │ check_invalid_dates()        │  │ WARNING = 2              │     │
│  │ check_invalid_odds()         │  │ INFO    = 1              │     │
│  │ check_missing_goals()        │  └──────────────────────────┘     │
│  │ check_missing_teams()        │                                    │
│  │ check_incorrect_leagues()    │  ┌──────────────────────────┐     │
│  │ check_invalid_statistics()   │  │      CheckResult         │     │
│  │ check_duplicate_ids()        │  ├──────────────────────────┤     │
│  │ check_impossible_scores()    │  │ - check_name: str        │     │
│  └──────────────────────────────┘  │ - description: str       │     │
│                                    │ - severity: Severity     │     │
│  ┌──────────────────────────────┐  │ - passed: bool           │     │
│  │     ValidationResult         │  │ - total_rows: int        │     │
│  ├──────────────────────────────┤  │ - violation_count: int   │     │
│  │ - source_name: str           │  │ - violations: list[dict] │     │
│  │ - total_rows: int            │  └──────────────────────────┘     │
│  │ - checks: list[CheckResult]  │                                    │
│  │ + passed_checks: int         │  ┌──────────────────────────┐     │
│  │ + total_checks: int          │  │   ValidationReporter     │     │
│  │ + total_violations: int      │  ├──────────────────────────┤     │
│  │ + passed: bool               │  │ + to_html(path)          │     │
│  │ + get_violations() → list    │  │ + to_csv(path)           │     │
│  └──────────────────────────────┘  │ + to_json(path)          │     │
│                                    └──────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.6 Feature Store Classes

```
┌──────────────────────────────────────────────────────────────────────┐
│                   FEATURE STORE (src/feature_store/)                  │
│                                                                      │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐   │
│  │    FeatureDefinition     │    │      FeatureRegistry         │   │
│  ├──────────────────────────┤    ├──────────────────────────────┤   │
│  │ id: str (PK)             │    │ - _session: Session          │   │
│  │ name: str                │    ├──────────────────────────────┤   │
│  │ version: str             │    │ + register(...) → FeatureDef │   │
│  │ feature_type: str        │    │ + get(id) → FeatureDef      │   │
│  │ category: FeatureCategory│    │ + search(...) → list         │   │
│  │ entity_type: EntityType  │    │ + new_version(...) → Feature │   │
│  │ status: FeatureStatus    │    │ + activate/deprecate/retire()│   │
│  │ computation_params: dict │    │ + get_dependency_graph()     │   │
│  │ validation_rules: dict   │    │ + topological_sort()         │   │
│  │ dependencies: dict       │    └──────────────────────────────┘   │
│  │ extra_metadata: dict     │                                       │
│  │ created_at: datetime     │    ┌──────────────────────────────┐   │
│  └──────────────────────────┘    │       FeatureStore           │   │
│                                   ├──────────────────────────────┤   │
│  ┌──────────────────────────┐    │ - _session: Session          │   │
│  │      FeatureValue        │    ├──────────────────────────────┤   │
│  ├──────────────────────────┤    │ + set(definition_id, ...)    │   │
│  │ id: str (PK)             │    │ + get(definition_id, ...)    │   │
│  │ definition_id: str (FK)  │    │ + set_many(...)              │   │
│  │ match_id / team_id       │    │ + get_many(...)              │   │
│  │ numeric_value: float     │    │ + get_feature_vector(...)    │   │
│  │ text_value: str          │    │ + needs_update(...)          │   │
│  │ json_value: dict         │    │ + start_batch(...)           │   │
│  │ batch_id: str (FK)       │    │ + complete_batch(...)        │   │
│  │ created_at: datetime     │    └──────────────────────────────┘   │
│  └──────────────────────────┘                                       │
│                                   ┌──────────────────────────────┐  │
│  ┌──────────────────────────┐    │     FeatureValidator          │  │
│  │  FeatureComputationBatch │    ├──────────────────────────────┤  │
│  ├──────────────────────────┤    │ - rules: list[ValidationRule]│  │
│  │ id: str (PK)             │    ├──────────────────────────────┤  │
│  │ batch_label: str         │    │ + validate(...) → list[Viol] │  │
│  │ trigger: str             │    │ + validate_batch(...) → Sum  │  │
│  │ status: str              │    │ + validate_vector(...) → Sum │  │
│  │ timing: dict             │    └──────────────────────────────┘  │
│  │ entity_count: int        │                                       │
│  │ features_computed: dict  │    ┌──────────────────────────────┐  │
│  │ extra_metadata: dict     │    │   FeatureComputer (ABC)      │  │
│  └──────────────────────────┘    ├──────────────────────────────┤  │
│                                   │ + compute(entity, context)  │  │
│  ┌──────────────────────────┐    └──────────────────────────────┘  │
│  │  FeatureVersion          │                                       │
│  ├──────────────────────────┤    ┌──────────────────────────────┐  │
│  │ definition_id: str (FK) │    │    ValidationRule (ABC)       │  │
│  │ version: str             │    ├──────────────────────────────┤  │
│  │ status: str              │    │ RangeRule, NotNullRule,      │  │
│  │ snapshot: dict           │    │ CardinalityRule,             │  │
│  │ changelog: str           │    │ ConsistencyRule              │  │
│  │ is_current: bool         │    └──────────────────────────────┘  │
│  └──────────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.7 Experiment Tracking Classes

```
┌──────────────────────────────────────────────────────────────────────┐
│               EXPERIMENT TRACKING (src/experiment_tracking/)          │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │   Experiment     │    │      Run         │    │  BestModel     │ │
│  ├──────────────────┤    ├──────────────────┤    ├────────────────┤ │
│  │ id: str (PK)     │    │ id: str (PK)     │    │ id: str (PK)   │ │
│  │ name: str        │    │ experiment_id:FK │    │ experiment_id: │ │
│  │ dataset_version  │    │ model_type       │    │   FK           │ │
│  │ feature_version  │    │ hyperparameters  │    │ run_id: FK     │ │
│  │ model_version    │    │ random_seed      │    │ metric_name    │ │
│  │ git_commit       │    │ status           │    │ metric_value   │ │
│  │ notes            │    │ metrics (JSON)   │    │ rank           │ │
│  │ tags (JSON)      │    │ training_duration│    │ is_promoted    │ │
│  │ created_at       │    │ hardware (JSON)  │    │ promoted_at    │ │
│  │ updated_at       │    │ git_commit       │    │ notes          │ │
│  ├──────────────────┤    │ notes            │    ├────────────────┤ │
│  │ runs: list[Run]  │    │ error_message    │    │ to_dict()      │ │
│  │ best_models:list │    │ started_at       │    └────────────────┘ │
│  │ to_dict()        │    │ finished_at      │                       │
│  └──────────────────┘    ├──────────────────┤    ┌────────────────┐ │
│                           │ artifacts:list  │    │ ModelArtifact  │ │
│  ┌──────────────────┐    │ experiment: Exp  │    ├────────────────┤ │
│  │ ExperimentTracker │    │ to_dict()        │    │ run_id: str(FK)│ │
│  ├──────────────────┤    │ create() (static)│    │ name: str      │ │
│  │ + create_experiment│  └──────────────────┘    │ uri: str       │ │
│  │ + get_experiment  │                           │ file_size_bytes│ │
│  │ + list_experiments│    ┌──────────────────┐    │ artifact_type  │ │
│  │ + update_experiment│   │ ModelRegistry    │    │ extra_metadata │ │
│  │ + delete_experiment│   ├──────────────────┤    └────────────────┘ │
│  │ + start_run       │   │ + register()     │                       │
│  │ + finish_run      │   │ + get_leaderboard│    ┌────────────────┐ │
│  │ + fail_run        │   │ + get_best()     │    │ ExperimentCmp  │ │
│  │ + resume_run      │   │ + promote()      │    ├────────────────┤ │
│  │ + log_artifact    │   │ + demote()       │    │ + compare_runs │ │
│  │ + run() [ctx mgr] │   │ + get_promoted() │    │ + compare_in() │ │
│  └──────────────────┘   │ + to_dataframe()  │    │ + compare_acr()│ │
│                           └──────────────────┘    │ + rank_by()    │ │
│  ┌──────────────────┐                              │ + to_dataframe│ │
│  │   Export         │                              └────────────────┘ │
│  │ export_json()    │                                                   │
│  │ export_csv()     │    ┌──────────────────┐                           │
│  │ export_html()    │    │   CLI            │                           │
│  └──────────────────┘    │ list, show,      │                           │
│                           │ compare,         │                           │
│                           │ leaderboard,     │                           │
│                           │ export, promote  │                           │
│                           └──────────────────┘                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. Setup Guide

### 10.1 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | Required for syntax features |
| PostgreSQL | 14+ | Optional; SQLite works for local dev |
| Git | 2.30+ | For version control |
| Docker | 20.10+ | Optional; for containerized deployment |
| Make | 4.0+ | Optional; for dev commands |

### 10.2 Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/football-prediction.git
cd football-prediction

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Install dev dependencies
pip install -r requirements.txt --extra-index-url ...
# or: make dev-install

# 5. Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
# - FOOTBALL_DATA_API_KEY (optional)
# - THE_ODDS_API_KEY (optional)

# 6. Initialize database
# Using SQLite (default, no setup needed):
#   The database is auto-created on first use.

# Using PostgreSQL:
make db-up         # Start PostgreSQL container
make db-init       # Run Alembic migrations

# 7. Collect data
python collect_all_worldcups.py     # World Cup data
# or: python collect_leagues.py     # League data

# 8. Train the model
python train_worldcup.py

# 9. Launch the dashboard
python run_dashboard.py
# Opens at http://localhost:8501
```

### 10.3 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No | SQLite auto | Full PostgreSQL connection string |
| `DB_HOST` | No | `localhost` | Database host |
| `DB_PORT` | No | `5432` | Database port |
| `DB_NAME` | No | `football_prediction` | Database name |
| `DB_USER` | No | `postgres` | Database user |
| `DB_PASSWORD` | No | `postgres` | Database password |
| `FOOTBALL_DATA_API_KEY` | No | `""` | Key for football-data.org |
| `THE_ODDS_API_KEY` | No | `""` | Key for The Odds API |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `APP_ENV` | No | `development` | Environment name |
| `APP_DEBUG` | No | `false` | Debug mode |
| `SECRET_KEY` | No | `change-me` | App secret key |

### 10.4 Dependencies

**Core:**
- `numpy>=1.26.0`, `pandas>=2.2.0`, `scipy>=1.12.0` — Data manipulation
- `scikit-learn>=1.4.0` — ML algorithms (LR, RF, preprocessing)
- `xgboost>=2.0.0`, `lightgbm>=4.3.0` — Gradient boosting
- `sqlalchemy>=2.0.30`, `psycopg2-binary>=2.9.9` — Database ORM
- `alembic>=1.13.0` — Database migrations
- `matplotlib>=3.8.0`, `seaborn>=0.13.0` — Visualization
- `python-dotenv>=1.0.0` — Environment variables
- `requests>=2.31.0`, `httpx>=0.27.0` — HTTP clients
- `pyyaml>=6.0.1` — YAML config parsing
- `tqdm>=4.66.0` — Progress bars

**Optional:**
- `torch>=2.2.0` — Neural network support
- `streamlit>=1.28.0` — Dashboard (if using `run_dashboard.py`)
- `joblib>=1.3.0` — Model serialization

**Dev:**
- `pytest>=8.0.0`, `pytest-cov>=4.1.0` — Testing
- `black>=24.0.0` — Formatting
- `ruff>=0.3.0` — Linting
- `mypy>=1.8.0` — Type checking
- `pre-commit>=3.7.0` — Git hooks

---

## 11. Development Guide

### 11.1 Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/my-feature

# 2. Make changes and run lint/format
make lint          # ruff check
make format-fix   # black auto-format
make typecheck    # mypy strict

# 3. Run tests
make test         # All tests
make test-cov     # With coverage

# 4. Run specific test suites
pytest tests/test_etl/ -v
pytest tests/test_feature_store/ -v

# 5. Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat: add my feature"

# 6. Push and create PR
git push -u origin feature/my-feature
```

### 11.2 Code Organization

| Directory | Purpose | Ownership |
|-----------|---------|-----------|
| `src/etl/` | Data pipeline logic | Data Engineering |
| `src/database/` | ORM models & repositories | Data Engineering |
| `src/data_collection/` | Data ingestion from sources | Data Engineering |
| `src/importers/` | Production data importers | Data Engineering |
| `src/feature_engineering.py` | Feature creation | ML Engineering |
| `src/ensemble.py` | Ensemble model | ML Engineering |
| `src/train.py` | Model training | ML Engineering |
| `src/monitoring/` | Pipeline monitoring | ML Engineering |
| `src/validation/` | Data quality | Data Engineering |
| `src/scheduler/` | Task scheduling | Platform |
| `src/feature_store/` | Feature storage | ML Platform |
| `src/experiment_tracking/` | ML experiment tracking | ML Platform |
| `src/app/` | Streamlit dashboard | Frontend |

### 11.3 Adding a New Feature

1. **Define the feature** in the `config.py` dataclass (if configurable)
2. **Create the computation** in `src/feature_engineering.py` or a new module
3. **Register in the pipeline** — add to `build_features()` in `feature_engineering.py`
4. **Add validation** — add checker in `src/validation/checks.py` if data quality check needed
5. **Write tests** — add test cases in `tests/`
6. **Run formatting & linting** — `make format-fix && make lint && make typecheck`
7. **Run tests** — `make test`

### 11.4 Database Migrations

```bash
# 1. Modify the ORM model in src/database/models/
# 2. Auto-generate a migration
make db-autogen
# Enter a descriptive message

# 3. Review the generated migration in alembic/versions/
# 4. Apply it
make db-migrate
```

---

## 12. Deployment Guide

### 12.1 Docker Deployment

```bash
# Build and start all services
make docker-build
make docker-up

# Or manually:
docker compose build
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f app

# Run migrations
docker compose run --rm migrate

# Stop all services
docker compose down
```

### 12.2 Docker Compose Services

```yaml
services:
  db:            # PostgreSQL 16 (alpine)
  app:           # Application container
  migrate:       # One-off migration runner
```

### 12.3 Manual Deployment

```bash
# 1. Set up PostgreSQL
# 2. Configure environment
export DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/football_prediction"
export APP_ENV="production"
export LOG_LEVEL="INFO"

# 3. Run migrations
alembic upgrade head

# 4. Set up the scheduled pipeline
python -m src.scheduler install --schedule "0 */6 * * *"

# 5. Launch dashboard (optional)
streamlit run src/app/dashboard.py --server.port 8501 --server.headless true
```

### 12.4 Performance Optimization

| Area | Recommendation |
|------|----------------|
| **Database** | Use PostgreSQL with connection pooling (pool_size=10) |
| **Feature Engineering** | Set `time_decay_halflife` to avoid multi-window computation |
| **Model Training** | Use XGBoost (fastest tree model); limit `n_estimators` to 80 |
| **Ensemble** | Use 3-model ensemble (XGBoost + LR + Poisson) instead of 5+ |
| **Data Collection** | Use incremental mode (`incremental=True`) |
| **Caching** | Enable the cache framework for API responses |
| **Parallelism** | Set `ETLPipeline(parallel=True, max_workers=4)` |

---

## 13. Contribution Guide

### 13.1 Getting Started

1. Fork the repository on GitHub.
2. Clone your fork: `git clone https://github.com/your-username/football-prediction.git`
3. Set up the dev environment: `make dev-install`
4. Create a branch: `git checkout -b feature/my-feature`

### 13.2 Pull Request Checklist

- [ ] Code follows project style (black, ruff, mypy)
- [ ] All tests pass (`make test`)
- [ ] New code includes test coverage
- [ ] Database migrations included (if schema changes)
- [ ] Documentation updated (docstrings, README, wiki)
- [ ] Changes rebased onto latest `main`
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

### 13.3 Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`

**Examples:**
```
feat(etl): add checkpoint resume support
fix(scheduler): handle empty dataset in validation
docs: add system architecture diagram
test(monitoring): add 88 tests for monitoring framework
```

### 13.4 Code Review Process

1. Open a PR against the `main` branch.
2. Ensure CI checks pass (tests, lint, typecheck).
3. Request review from maintainers.
4. Address review feedback.
5. Squash-merge once approved.

---

## 14. Coding Standards

### 14.1 Python Style Guide

| Rule | Standard | Enforcement |
|------|----------|-------------|
| Line length | 88 characters | `black` + `ruff` |
| Indentation | 4 spaces | `black` |
| Quotes | Double quotes (`"`) | `black` |
| Imports | `from __future__ import annotations` first | Manual |
| Import order | stdlib → third-party → local | `ruff` (I) |
| Naming | `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants | Manual |
| Type hints | Required for all function signatures | `mypy --strict` |
| Docstrings | NumPy format | `ruff` (D) |

### 14.2 Docstring Format

All public functions and classes must have NumPy-style docstrings:

```python
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> tuple[Any, dict[str, list[float]]]:
    """Train a model on the provided data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_val : pd.DataFrame, optional
        Validation feature matrix.
    y_val : pd.Series, optional
        Validation target vector.

    Returns
    -------
    model : Any
        Trained model object.
    history : dict[str, list[float]]
        Training history (loss, metrics).

    Raises
    ------
    ValueError
        If model_type is not recognized.
    """
```

### 14.3 Project Conventions

- **Error handling:** Use custom exceptions from `src.utils.exceptions` for domain errors
- **Logging:** Use `logging.getLogger(__name__)` at module level; structured logging with context
- **Configuration:** All tunable parameters in `config.py` dataclasses; no magic numbers in code
- **Data leakage:** Rolling features use `.shift(1)` to exclude the current match; splits are chronological (no shuffle)
- **API keys:** Never hardcode — use `.env` + `python-dotenv`
- **Testing:** One test file per module; conftest for shared fixtures; SQLite `:memory:` for DB tests
- **Imports:** Prefer `from __future__ import annotations` for deferred evaluation
- **Module `__init__.py`:** Re-export public API symbols; document package in docstring

### 14.4 Pre-commit Hooks

The project uses `pre-commit` for automated quality checks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

Install hooks:

```bash
pre-commit install
```

### 14.5 Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Create venv and install dependencies |
| `make dev-install` | Install dev dependencies + pre-commit |
| `make lint` | Run ruff linter |
| `make format` | Check formatting (black --check) |
| `make format-fix` | Auto-fix formatting |
| `make typecheck` | Run mypy strict mode |
| `make test` | Run all pytest tests |
| `make test-cov` | Run tests with coverage report |
| `make clean` | Remove cache artifacts |
| `make db-up` | Start PostgreSQL container |
| `make db-down` | Stop PostgreSQL |
| `make db-migrate` | Run Alembic migrations |
| `make db-autogen` | Auto-generate migration |
| `make docker-build` | Build Docker images |
| `make docker-up` | Start all Docker services |
| `make run-pipeline` | Run prediction pipeline |
| `make run-dashboard` | Launch Streamlit |

---

*Generated from the source code. Last updated: 2026-07-13.*
