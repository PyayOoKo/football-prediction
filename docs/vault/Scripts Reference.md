---
tags:
  - football-prediction
  - scripts
  - reference
  - cli
created: 2026-07-12
---

# 📜 Scripts Reference

> Complete reference for all CLI scripts and their purposes.

See also: [[Quick Start Guide]], [[Architecture Overview]]

---

## Main Pipeline Scripts

| Script | What It Does | Est. Time |
|--------|-------------|-----------|
| `python run_pipeline.py` | Full daily pipeline (download → train → predict → report) | ~30-60s |
| `python run_pipeline.py --lightweight` | Predict only (skip download + retrain) | ~5-10s |
| `python run_pipeline.py --skip-download` | Skip download, use existing data | ~20-40s |
| `python run_pipeline.py --skip-train` | Skip retrain, use existing model | ~10-20s |

## Training Scripts

| Script | What It Does | Est. Time |
|--------|-------------|-----------|
| `python train_worldcup.py` | Train + predict on multi-tournament World Cup data | ~20-30s |
| `python train_worldcup.py --skip-train` | Use existing saved model | ~10s |
| `python train_xgboost.py` | XGBoost-specific training | ~20-30s |
| `python train_league.py` | League-specific training | Varies |
| `python train_with_xag.py` | Training using xAG (expected assisted goals) | ~30s |

## Data Collection Scripts

| Script | What It Does | Est. Time |
|--------|-------------|-----------|
| `python collect_all_worldcups.py` | Download 2002–2026 World Cup from openfootball | ~10s |
| `python collect_leagues.py` | Download league data from Football-Data.co.uk | ~10-30s |
| `python collect_worldcup.py` | Download a single World Cup | ~3s |
| `python collect_worldcup_xg.py` | Download World Cup xG data | ~10s |
| `python collect_xag_data.py` | Download xAG (expected assists) data | ~10s |
| `python collect_player_data.py` | Download player/transfer data | ~10-30s |
| `python collect_lineups.py` | Download lineup data | ~10s |

## Analysis & Betting Scripts

| Script | What It Does |
|--------|-------------|
| `python find_value_bets.py` | Historical value bet analysis |
| `python today_value_bets_live.py` | Live value bets from The Odds API |
| `python run_backtest.py` | Run historical backtest simulation |
| `python run_dashboard.py` | Launch Streamlit dashboard |
| `python predict_worldcup.py` | Lightweight World Cup prediction (no training) |

## Merge & Utility Scripts

| Script | What It Does |
|--------|-------------|
| `python merge_xg_data.py` | Merge xG data into main dataset |
| `python merge_all_xg_data.py` | Merge all xG sources together |
| `python refresh_worldcup.py` | Refresh World Cup data |
| `python bracket_simulator.py` | Simulate World Cup knockout bracket |

## What-If & Analysis Scripts

| Script | What It Does |
|--------|-------------|
| `python compare_models_brazil_norway.py` | Model comparison for Brazil vs Norway |
| `python what_if_brazil_norway.py` | What-if scenario: Brazil vs Norway |
| `python what_if_canada_morocco.py` | What-if scenario: Canada vs Morocco |
| `python what_if_portugal_spain.py` | What-if scenario: Portugal vs Spain |
| `python analyze_england_norway.py` | Analysis: England vs Norway |

## Test Scripts

| Script | What It Does |
|--------|-------------|
| `python test_2022_worldcup.py` | Test model on 2022 World Cup data |
| `pytest tests/` | Run full test suite |

## Debug Scripts

| Script | What It Does |
|--------|-------------|
| `python scripts/debug_lineups.py` | Debug lineup data |
| `python scripts/debug_transfermarkt.py` | Debug Transfermarkt data |
| `python scripts/verify_ids.py` | Verify team/player IDs |
| `python scripts/evaluate_existing.py` | Evaluate a saved model |

## Setup Scripts

| Script | What It Does |
|--------|-------------|
| `python setup_auto_commit.bat` | Set up auto git commit |
| `python setup_scheduler.bat` | Set up Windows Task Scheduler |
| `python setup_value_bets_scheduler.bat` | Set up value bets scheduler |

---

## Key Data Files

| Path | Description |
|------|-------------|
| `data/raw/worldcup_all.csv` | Combined World Cup data (2002–2026) |
| `data/raw/results.csv` | League match results |
| `data/processed/results_clean.csv` | Preprocessed clean data |
| `models/ensemble_model.joblib` | Trained ensemble model |
| `models/worldcup_xgboost.joblib` | World Cup-specific XGBoost |
| `reports/predictions/predictions_*.csv` | Generated predictions |
| `reports/predictions_worldcup/worldcup_predictions.csv` | World Cup predictions |
| `reports/backtest/` | Backtest charts (PNG) |
| `reports/figures/` | EDA charts (PNG) |
| `logs/pipeline.log` | Pipeline run logs |

---

## Key Files by Size

| File | Role | Est. Lines |
|------|------|-----------|
| `config.py` | Global configuration | ~500 |
| `src/ensemble.py` | Ensemble model (XGBoost + LR + Poisson) | ~450 |
| `src/feature_engineering.py` | Feature creation hub | ~850 |
| `src/elo.py` | Elo rating system | ~350 |
| `src/poisson_model.py` | Poisson goals model | ~500 |
| `src/dixon_coles.py` | Dixon-Coles MLE model | ~600 |
| `src/backtesting.py` | Backtesting engine | ~600 |
| `src/value_betting.py` | Value bet calculator | ~350 |
| `src/preprocessing.py` | Data cleaning pipeline | ~450 |
| `src/data_collection/collector.py` | Data collection orchestrator | ~250 |
| `run_pipeline.py` | Pipeline entry point | ~600 |
| `train_worldcup.py` | World Cup training script | ~400 |
