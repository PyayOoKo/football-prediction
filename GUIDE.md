# ⚽ Football Prediction System — Complete Guide

## Table of Contents
1. [World Cup Pipeline — Quick Start](#1-world-cup-pipeline--quick-start)
2. [Daily/Weekly Automated Refresh](#2-dailyweekly-automated-refresh)
3. [Full Tournament Bracket Simulation](#3-full-tournament-bracket-simulation)
4. [Transition to Top 5 European Leagues](#4-transition-to-top-5-european-leagues)
5. [Adding New Data & Re-Predicting](#5-adding-new-data--re-predicting)
6. [Dashboard & Visualization](#6-dashboard--visualization)
7. [File Reference](#7-file-reference)

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
# Preprocess the league data
python -c "
from src.preprocessing import run_preprocessing
run_preprocessing()
"

# Train a model for league predictions
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
python -c "
from src.data_collection.sources.football_data_co_uk import download_bulk
df = download_bulk(leagues=['E0', 'SP1', 'D1', 'I1', 'F1'], max_seasons=5)
df.to_csv('data/raw/league_all.csv', index=False)
"

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
| **💰 Value Bets** | Enter odds → find EV+ opportunities |
| **📊 Backtest** | Historical performance metrics |
| **🏆 World Cup 2026** | Bracket tree, probability bars, Poisson scorelines, confidence analysis |

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `train_worldcup.py` | World Cup: train XGBoost + Poisson + predict |
| `refresh_worldcup.py` | Automated data refresh & re-train (scheduler-friendly) |
| `bracket_simulator.py` | Simulate full knockout bracket |
| `collect_worldcup_xg.py` | Extract xG from StatsBomb (2018/2022) |
| `collect_xag_data.py` | Extract xAG from StatsBomb key passes |
| `collect_r16_data.py` | Collect international tournament data (Euro, Copa, AFCON) |
| `train_with_xag.py` | Full enhanced training (xAG + improved Elo) |
| `run_pipeline.py` | General league pipeline (download → preprocess → train → predict) |
| `train_xgboost.py` | Train league predictor |
| `run_dashboard.py` | Launch Streamlit dashboard |
| `src/elo.py` | Enhanced Elo system (xG-margin K-factor, host-nation bonus) |
| `src/feature_engineering.py` | Leakage-free feature pipeline |
| `src/xg_features.py` | Rolling xG/xAG features |
| `src/poisson_model.py` | Poisson goal distribution model |
