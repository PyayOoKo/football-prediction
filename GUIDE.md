# ⚽ Football Prediction System — Complete Guide

## Table of Contents
1. [World Cup Pipeline — Quick Start](#1-world-cup-pipeline--quick-start)
2. [Daily/Weekly Automated Refresh](#2-dailyweekly-automated-refresh)
3. [Full Tournament Bracket Simulation](#3-full-tournament-bracket-simulation)
4. [Transition to Top 5 European Leagues](#4-transition-to-top-5-european-leagues)
5. [Adding New Data & Re-Predicting](#5-adding-new-data--re-predicting)
6. [Dashboard & Visualization](#6-dashboard--visualization)
7. [Auto-Commit to GitHub](#7-auto-commit-to-github)
8. [Performance Tips](#8-performance-tips)
9. [File Reference](#9-file-reference)
10. [Quick Command Reference](#10-quick-command-reference)

---

## ⚡ TL;DR — Quick Start

```bash
# 5 commands, ~2 minutes total:
python -m venv .venv                # 1. Create virtual environment
.venv\Scripts\activate              # 2. Activate it (use source for Git Bash)
pip install -r requirements.txt     # 3. Install packages
python collect_all_worldcups.py     # 4. Download World Cup data
python train_worldcup.py            # 5. Train model & predict!

# To refresh predictions later:
python refresh_worldcup.py --skip-lineups

# For value bets with live odds:
python today_value_bets_live.py --calibrate platt
```

---

## 1. World Cup Pipeline — Quick Start

```bash
# ONE COMMAND — does everything:
python refresh_worldcup.py

# Or step by step:

# Step 1: Download & combine all World Cup data (2002-2026)
python refresh_worldcup.py --skip-train

# Step 2: Train model & predict knockout matches
python train_worldcup.py
```

### What happens:

| Step | Command | What it does |
|------|---------|-------------|
| Download | `refresh_worldcup.py` | Fetches all 7 World Cups from openfootball, resolves knockout bracket placeholders (W89 → actual team name) |
| Train | `train_worldcup.py` | Builds features (Elo, xG, Poisson, rolling stats), trains XGBoost on 658 matches, evaluates on test set |
| Predict | `train_worldcup.py` | Generates R16 predictions with swap-averaged probabilities + Poisson scoreline forecasts |
| Save | `train_worldcup.py` | Saves to `reports/predictions_worldcup/worldcup_predictions.csv` |

### Output files:
- **Predictions CSV**: `reports/predictions_worldcup/worldcup_predictions.csv`
- **Trained model**: `models/worldcup_xgboost.joblib`
- **Combined data**: `data/raw/worldcup_all.csv`

---

## 2. Daily/Weekly Automated Refresh

### Windows Task Scheduler (Recommended)

1. **Setup the scheduler** (one time, Run as Administrator):
   ```
   Right-click setup_scheduler.bat → Run as Administrator
   ```

2. **Manual refresh anytime**:
   ```bash
   python refresh_worldcup.py --quiet --log-file refresh.log
   ```

The task runs every 6 hours and will:
- Download latest match scores
- Resolve newly completed knockout placeholders
- Retrain the model with updated data
- Generate new predictions

### What updates automatically:
- ✅ New match scores (as they're played)
- ✅ Bracket resolution (W89 → actual winner)
- ✅ Model retraining
- ✅ Updated predictions

### Value bets scheduler (daily at 7 AM):
```
Right-click setup_value_bets_scheduler.bat → Run as Administrator
```
Runs `today_value_bets_live.py` every morning with live odds.

---

## 3. Full Tournament Bracket Simulation

To predict the entire knockout bracket (R16 → QF → SF → Final):

```bash
python bracket_simulator.py
```

This script:
1. Loads the R16 predictions
2. Simulates each round using the probabilities
3. Advances winners through QF → SF → Final
4. Saves the full bracket to `reports/predictions_worldcup/bracket_prediction.csv`

### Monte Carlo simulation (champion probabilities):
```bash
python bracket_simulator.py --monte-carlo 10000
```
Runs 10,000 bracket simulations to get champion probabilities.
Output: `reports/predictions_worldcup/monte_carlo_probs.csv`

### What-if analysis:
```bash
python what_if_brazil_norway.py       # Brazil vs Norway hypothetical
python what_if_canada_morocco.py      # Canada vs Morocco
python what_if_portugal_spain.py      # Portugal vs Spain
python compare_models_brazil_norway.py  # Compare model types
```

---

## 4. Transition to Top 5 European Leagues

After the World Cup ends, here's how to switch to betting on Europe's top leagues:

### Step 1: Install league dependencies
```bash
pip install football-data-co-uk  # Already installed
```

### Step 2: Download league data

**Top 5 League codes:**

| League | Code | Data Source |
|--------|------|-------------|
| **Premier League** | `E0` | football-data.co.uk (free, no API key) |
| **La Liga** | `SP1` | football-data.co.uk |
| **Bundesliga** | `D1` | football-data.co.uk |
| **Serie A** | `I1` | football-data.co.uk |
| **Ligue 1** | `F1` | football-data.co.uk |

**Download all 5 leagues:**
```bash
python collect_leagues.py
```
Or manually via Python:
```bash
python -c "
from src.data_collection.sources.football_data_co_uk import download_bulk
import pandas as pd

leagues = ['E0', 'SP1', 'D1', 'I1', 'F1']
all_dfs = []
for league in leagues:
    df = download_bulk(leagues=[league], max_seasons=5)
    all_dfs.append(df)
    print(f'{league}: {len(df)} rows')

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv('data/raw/league_all.csv', index=False)
print(f'Saved {len(combined)} total rows')
"
```

### Step 3: Preprocess & Train

```bash
# Train on league data
python train_league.py

# Or train XGBoost specifically
python train_xgboost.py

# Or use the existing pipeline:
python run_pipeline.py --skip-download
```

### Step 4: View predictions in Dashboard

```bash
python run_dashboard.py
```

The existing dashboard pages handle league data:
- **🔮 Predict a Match** — pick any two teams from the Top 5 leagues
- **💰 Find Value Bets** — enter bookmaker odds to find EV+ opportunities
- **📊 View Backtest** — see historical performance

### What's different about leagues vs World Cup

| Feature | World Cup | League |
|---------|-----------|--------|
| Home advantage | Neutral (50 Elo pts) | Strong (100 Elo pts) |
| Training data | 658 matches (7 tournaments) | 2000+ matches per league |
| xG data | StatsBomb (314 matches) | Not available (zero-filled) |
| Head-to-head | Disabled | ✅ Enabled |
| League position | Disabled | ✅ Enabled |
| Odds processing | Disabled | ✅ Enabled |
| Player info | Disabled | ✅ Optional |
| Betting value | Manual | ✅ Built-in value bet finder |

---

## 5. Adding New Data & Re-Predicting

### For World Cup (automatic)

```bash
python refresh_worldcup.py
```

This automatically downloads new scores, resolves brackets, retrains, and predicts.

### For League (manual update)

```bash
# 1. Download latest match results
python collect_leagues.py

# 2. Run the full pipeline
python run_pipeline.py
```

### To add completely new data (any competition)

1. Get the data in CSV format with these minimum columns:
   ```
   date, home_team, away_team, result, home_goals, away_goals
   ```

2. Append to existing CSV:
   ```bash
   python -c "
   import pandas as pd
   existing = pd.read_csv('data/raw/results.csv')
   new = pd.read_csv('your_new_data.csv')
   combined = pd.concat([existing, new], ignore_index=True)
   combined.to_csv('data/raw/results.csv', index=False)
   "
   ```

3. Re-run pipeline:
   ```bash
   python run_pipeline.py
   ```

---

## 6. Dashboard & Visualization

```bash
python run_dashboard.py
```

Opens your browser to the Streamlit dashboard with:

| Page | What it shows |
|------|--------------|
| **Home** | Overview metrics, recent matches, model info |
| **🔮 Predict** | Pick two teams → instant prediction |
| **💰 Value Bets** | Live value bets (cached) + manual odds entry |
| **📊 Backtest** | Historical performance metrics & charts |
| **🏆 World Cup 2026** | Bracket tree, probability bars, Poisson scorelines, confidence analysis, champion probs |

---

## 7. Auto-Commit to GitHub

The project has a built-in system to automatically commit and push changes
to GitHub every hour, so you never forget to back up your work.

### One-time setup (10 seconds):
```
Right-click setup_auto_commit.bat → Run as Administrator
```
This installs a Windows Task Scheduler job that runs `scripts/auto_commit.ps1`
every hour. It:
- Detects changed files
- Stages them (`git add -A`)
- Creates a timestamped commit (e.g. `Auto-commit: 2026-07-14 [+5 ~12 -1]`)
- Pushes to the current branch on GitHub
- Logs everything to `auto_commit.log`

### Useful commands:
| Action | Command |
|--------|---------|
| Run auto-commit now | `schtasks /run /tn "FootballPredictionAutoCommit"` |
| Check if installed | `schtasks /query /tn "FootballPredictionAutoCommit"` |
| View the log | `type auto_commit.log` |
| Remove the task | `schtasks /delete /tn "FootballPredictionAutoCommit" /f` |

---

## 8. Performance Tips

Most scripts finish in under 60 seconds. If yours is taking longer:

| Cause | Solution |
|-------|----------|
| **MinGW Python** (Inkscape/Git Bash bundled) can't use pre-compiled wheels, tries to build from source | Use Microsoft Store Python instead (see setup guide) |
| **Downloading data** (no internet / slow source) | `python run_pipeline.py --skip-download` |
| **Collecting lineups** (adds 30-60s) | `python refresh_worldcup.py --skip-lineups` |
| **Hyper-parameter tuning** enabled | Set `config.train.tune_base_models = False` |

### Expected run times:

| Command | Expected time |
|---------|---------------|
| `python train_worldcup.py` | 20-30 sec |
| `python predict_worldcup.py` | 15-25 sec |
| `python run_pipeline.py --lightweight` | 5-10 sec |
| `python run_pipeline.py --skip-download` | 30-60 sec |
| `python today_value_bets_live.py` | 10-20 sec |
| `python run_dashboard.py` | < 5 sec |
| `python bracket_simulator.py` | 10-60 sec |
| `python refresh_worldcup.py` | 30-120 sec |
| `scripts/auto_commit.ps1` | < 5 sec |

---

## 9. File Reference

### Main scripts

| File | Purpose |
|------|---------|
| `train_worldcup.py` | World Cup: train XGBoost + Poisson + predict |
| `predict_worldcup.py` | Lightweight World Cup prediction (no tuning) |
| `refresh_worldcup.py` | Automated data refresh & re-train (scheduler-friendly) |
| `bracket_simulator.py` | Simulate full knockout bracket + Monte Carlo |
| `today_value_bets_live.py` | Live value bets with calibration (primary) |
| `run_dashboard.py` | Launch Streamlit dashboard |
| `run_pipeline.py` | General pipeline (download → preprocess → train → predict) |
| `run_combined_pipeline.py` | Full ensemble pipeline (LR + RF + XGB + Poisson) |
| `run_backtest.py` | Run betting backtest |
| `run_first_model.py` | Train & evaluate first model |
| `test_2022_worldcup.py` | Test model on 2022 World Cup data |

### Data collection

| File | Purpose |
|------|---------|
| `collect_all_worldcups.py` | Download all World Cup data (2002-2026) |
| `collect_worldcup.py` | Download single World Cup edition |
| `collect_worldcup_xg.py` | Extract xG from StatsBomb (2018/2022) |
| `collect_xag_data.py` | Extract xAG from StatsBomb key passes |
| `collect_r16_data.py` | Collect international tournament data (Euro, Copa, AFCON) |
| `collect_leagues.py` | Download top 5 European leagues |
| `collect_player_data.py` | Squad info from Transfermarkt |
| `collect_lineups.py` | Lineup data from Transfermarkt |
| `merge_all_xg_data.py` | Merge multiple xG sources |
| `merge_xg_data.py` | Merge xG with main dataset |
| `find_value_bets.py` | Legacy value bet finder |

### Training & analysis

| File | Purpose |
|------|---------|
| `train_league.py` | Train on league data |
| `train_xgboost.py` | Train league predictor (XGBoost) |
| `train_with_xag.py` | Enhanced training with xAG features |
| `analyze_england_norway.py` | Analyze specific matchups |
| `compare_models_brazil_norway.py` | Compare model types on matchup |
| `what_if_brazil_norway.py` | What-if: Brazil vs Norway |
| `what_if_canada_morocco.py` | What-if: Canada vs Morocco |
| `what_if_portugal_spain.py` | What-if: Portugal vs Spain |

### Core modules (`src/`)

| File | Purpose |
|------|---------|
| `src/elo.py` | Enhanced Elo system (xG-margin K-factor, host-nation bonus) |
| `src/feature_engineering.py` | Leakage-free feature pipeline (140+ features) |
| `src/xg_features.py` | Rolling xG/xAG features |
| `src/poisson_model.py` | Poisson goal distribution model |
| `src/dixon_coles.py` | Dixon-Coles joint attack/defence model |
| `src/preprocessing.py` | Data preprocessing pipeline |
| `src/ensemble.py` | Ensemble model (LR + RF + XGB + Poisson) |
| `src/calibration.py` | Platt/Isotonic probability calibration |
| `src/value_betting.py` | Kelly criterion value betting |
| `src/backtesting.py` | Betting backtest engine |
| `src/odds_api.py` | The Odds API integration |
| `src/odds_processing.py` | Odds processing & normalization |
| `src/player_info.py` | Player data integration |
| `src/confidence_scoring.py` | Confidence scoring |
| `src/evaluate.py` | Model evaluation metrics |
| `src/hyperparameter_tuning.py` | Grid/random search tuning |
| `src/time_series_cv.py` | Time-series cross-validation |

### Utility scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `scripts/auto_commit.ps1` | PowerShell: auto-commit to GitHub |
| `scripts/today_value_bets.py` | Value bets with hardcoded odds |
| `scripts/quick_train_eval.py` | Fast train + evaluate |
| `scripts/run_training.py` | Full training pipeline runner |
| `scripts/tune_ensemble.py` | Grid-search ensemble weights |
| `scripts/train_baseline.py` | Train a baseline model |
| `scripts/evaluate_existing.py` | Evaluate saved model on test set |
| `scripts/backtest_high_conf_away.py` | Backtest high-confidence away bets |
| `scripts/audit_leakage.py` | Check for data leakage in features |
| `scripts/validate_features.py` | Validate computed features |
| `scripts/verify_ids.py` | Verify Transfermarkt team IDs |
| `scripts/verify_feature_store.py` | Verify feature store integrity |
| `scripts/data_quality_dashboard.py` | Data quality metrics dashboard |
| `scripts/test_end_to_end.py` | End-to-end pipeline test |
| `scripts/test_time_validation.py` | Validate time-series correctness |
| `scripts/pre_phase3_checklist.py` | Pre-deployment checklist |
| `scripts/run_benchmark.py` | Run performance benchmarks |
| `scripts/benchmark_database.py` | Benchmark database queries |
| `scripts/feature_importance_analysis.py` | Feature importance analysis |
| `scripts/migrate_to_partitions.py` | Migrate DB to partitioned tables |
| `scripts/bump_version.py` | Bump project version number |
| `scripts/generate_changelog.py` | Auto-generate changelog |
| `scripts/notify.py` | Send notifications |
| `scripts/debug_lineups.py` | Debug lineup scraping |
| `scripts/debug_lineups2.py` | Lineup scraping (alt method) |
| `scripts/debug_lineups3.py` | Lineup scraping (alt method 2) |
| `scripts/debug_transfermarkt.py` | Debug Transfermarkt scraping |
| `scripts/debug_transfermarkt2.py` | Transfermarkt scraping (alt method) |

### Batch files (scheduler setup)

| File | Purpose |
|------|---------|
| `setup_auto_commit.bat` | Install hourly GitHub auto-commit task |
| `setup_scheduler.bat` | Install World Cup refresh scheduler (6hr) |
| `setup_value_bets_scheduler.bat` | Install daily value bets scheduler (7AM) |

---

## 10. Quick Command Reference

| Task | Command |
|------|---------|
| Predict tomorrow's matches | `python today_value_bets_live.py --days 1` |
| Predict this week's matches | `python today_value_bets_live.py` |
| Train + predict all World Cup | `python train_worldcup.py` |
| Quick predict (no tuning) | `python predict_worldcup.py` |
| Full data refresh | `python refresh_worldcup.py --skip-lineups` |
| Simulate bracket | `python bracket_simulator.py --monte-carlo 10000` |
| Launch dashboard | `python run_dashboard.py` |
| Full ensemble pipeline | `python run_combined_pipeline.py` |
| What-if Brazil vs Norway | `python what_if_brazil_norway.py` |
| Collect player data | `python collect_player_data.py` |
| Download World Cup data | `python collect_all_worldcups.py` |
| Download league data | `python collect_leagues.py` |
| Train on league data | `python train_league.py` |
| Auto-commit to GitHub | `Right-click setup_auto_commit.bat → Run as Admin` |
| Manual git push | `git add . && git commit -m "msg" && git push` |
| Run tests | `pytest` |
| Check test coverage | `pytest --cov=src` |
| Data leakage audit | `python scripts/audit_leakage.py` |
| Feature importance | `python scripts/feature_importance_analysis.py` |
