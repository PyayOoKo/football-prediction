# User Guide

> End-user documentation for running predictions, finding value bets, using the dashboard, and interpreting results.

## Getting Started

### Launch the Dashboard

```bash
python run_dashboard.py
```

This starts the Streamlit dashboard at **http://localhost:8501**.

### Run a Quick Prediction

```python
# Command line (after collecting data and training a model)
python predict_worldcup.py
```

## Dashboard Overview

The dashboard has 4 pages:

```
⚽ Football Match Predictor
├── 📊 Dashboard (Home)    — Overview, key metrics, model performance
├── 🔮 Predict             — Predict match outcomes
├── 💰 Value Bets          — Find betting opportunities
├── 📊 Backtest            — Historical strategy performance
└── 🏆 World Cup 2026      — World Cup predictions
```

### Dashboard (Home)
- **Historical Matches** — total matches in the dataset
- **Teams** — unique teams tracked
- **Active Model** — currently loaded model
- **Latest Match** — most recent match date
- **Recent Matches** — last 15 matches with scores
- **Model Performance** — accuracy, log-loss, per-class metrics
- **Balance Check** — detects draw-blindness and class imbalance

### 🔮 Predict Page

1. Select **Home Team** and **Away Team** from dropdowns
2. Click **Predict Match**
3. View:
   - Win/Draw/Lose probabilities
   - Predicted outcome
   - Head-to-head history
   - Expected goals

### 💰 Value Bets Page

1. View all upcoming matches with betting opportunities
2. Columns:
   - Match (home vs away)
   - Model probability (home/draw/away)
   - Best odds from bookmakers
   - **Expected Value** (EV > 0 = value bet)
   - Kelly stake (recommended bet size)
3. Filter by:
   - Minimum expected value
   - League
   - Date range

### 📊 Backtest Page

Historical simulation of betting strategy:
- **Equity curve** — bankroll over time
- **Key metrics**: ROI, Sharpe ratio, max drawdown, win rate
- **Per-season breakdown** — how strategy performed each season
- **Confusion matrix** — model prediction accuracy

### 🏆 World Cup 2026 Page

- Group stage predictions
- Bracket visualization
- Match-by-match probabilities
- Tournament winner odds

## Interpreting Predictions

### Probability Distribution

```
Home Win:  52%  ████████████████████████████░░░░░░░░░░░
Draw:      28%  █████████████░░░░░░░░░░░░░░░░░░░░░░░░░
Away Win:  20%  █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

- **High confidence** (>65%): Strong prediction
- **Medium confidence** (45-65%): Moderate confidence
- **Low confidence** (<45%): Near coin-flip

### Value Betting

A **value bet** exists when the model's implied probability is higher than the bookmaker's implied probability:

```
Model probability: 52% → Fair odds: 1.92
Bookmaker odds:    2.10
Expected value:    (0.52 × 2.10) - 1 = 0.092 → 9.2% edge → VALUE ✅
```

### Model Performance Metrics

| Metric | What It Means | Good Value |
|---|---|---|
| **Accuracy** | % of correct predictions | >60% |
| **Log-Loss** | Prediction uncertainty (lower = better) | <0.65 |
| **Precision** | % of predicted wins that actually won | >0.60 |
| **Recall** | % of actual wins that were predicted | >0.60 |
| **F1-Score** | Harmonic mean of precision & recall | >0.60 |
| **ROI** | Return on investment from value bets | >10% |

## Batch Prediction Pipeline

```bash
# Full pipeline (download → train → predict → report)
python run_pipeline.py

# Lightweight (predict only, skip download & retrain)
python run_pipeline.py --lightweight

# Custom config
python run_pipeline.py --config my_config.yaml
```

## Data Collection

```bash
# Collect all World Cup historical data
python collect_all_worldcups.py

# Collect league data
python collect_leagues.py

# Collect xG data
python collect_worldcup_xg.py
```

## Training

```bash
# Train XGBoost model
python train_xgboost.py

# Train World Cup model
python train_worldcup.py

# Train per-league model
python train_league.py --league E0

# Run backtest
python run_backtest.py
```

## Value Betting

```bash
# Find value bets using live odds
python find_value_bets.py

# Today's live value bets (requires THE_ODDS_API_KEY)
python today_value_bets_live.py

# Run scheduled value bets
python -m src.scheduler.cli run --tasks find_value_bets
```

## What-If Analysis

```bash
# Simulate hypothetical matchups
python what_if_brazil_norway.py
python what_if_canada_morocco.py
python what_if_portugal_spain.py
```

## Scheduled Execution

```bash
# Run all scheduled tasks
python -m src.scheduler.cli run

# Install as Windows scheduled task
python -m src.scheduler.cli install-windows
```

## Reports and Outputs

After running the pipeline, check these directories:

| Directory | Contents |
|---|---|
| `reports/predictions/` | Match predictions CSV |
| `reports/backtest/` | Backtest HTML reports |
| `reports/validation/` | Data validation HTML reports |
| `reports/value_bets/` | Value betting CSV reports |
| `reports/scheduler/` | Pipeline run reports |
| `logs/` | Application logs |

## System Requirements

- **Python:** 3.12+
- **RAM:** 4GB minimum, 8GB+ recommended (for 100K+ match datasets)
- **Disk:** 1GB for data + models
- **Database:** PostgreSQL 16+ (or SQLite for local dev)
- **OS:** Windows, macOS, or Linux
