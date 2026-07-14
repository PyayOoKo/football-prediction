<div align="center">

# ⚽ Football Match Outcome Prediction

**A machine learning pipeline for predicting football match outcomes, finding value bets, and simulating tournaments — with live odds integration.**

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange?logo=xgboost)
![Tests](https://img.shields.io/badge/tests-1.4k%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Dashboard](#-dashboard) • [World Cup 2026](#-world-cup-2026) • [Configuration](#-configuration)

</div>

---

## 📚 Documentation

Looking for **complete project documentation**? See the full guide:

➡️ **[`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md)** — system architecture, folder structure, database schema, ETL/scheduler/validation workflows, diagrams, setup/deployment/development guides, coding standards, and testing strategy.

---

## 📋 Overview

A modular, production-oriented machine learning pipeline that:

- **Trains** XGBoost models on 7 World Cups of data (2002–2026) with **65.3% test accuracy**
- **Predicts** match outcomes (home win / draw / away win) with probability distributions
- **Integrates xG/xAG data** from StatsBomb Open Data for richer feature engineering
- **Finds value bets** by comparing model probabilities against live bookmaker odds
- **Backtests** betting strategies using Kelly criterion, ROI, drawdown analysis
- **Streamlit dashboard** for interactive predictions, bracket simulation, and performance charts
- **Auto-syncs** to GitHub hourly via Windows Task Scheduler

---

## ✨ Features

### 🤖 Model Pipeline

| Component | Description |
| :--- | :--- |
| **Feature Engineering** | Rolling averages (5/10 match windows), head-to-head stats, Elo ratings, league position, temporal features |
| **Expected Goals (xG)** | StatsBomb xG & xAG data merged into training — **+4.2pp accuracy improvement** |
| **Models** | XGBoost, Random Forest, Logistic Regression, LightGBM, Poisson, Neural Network |
| **Ensemble** | Weighted voting across multiple models with grid-search optimisation |
| **Hyper-parameter Tuning** | RandomisedSearchCV across all model types |

### 🏆 World Cup 2026

- **Automated refresh** — downloads latest results, resolves knockout bracket placeholders, retrains, repredicts
- **Round of 16 predictions** with full probability breakdowns (home/draw/away)
- **Knockout bracket simulator** — Monte Carlo simulations through QF → SF → Final
- **Swap-averaging** — neutral-venue handling for tournament knockout matches

### 💰 Value Betting

- **Live odds** via [The Odds API](https://the-odds-api.com/) (free tier — 500 requests/month)
- **Hardcoded fallback** when API is unavailable
- **Kelly criterion** stake sizing (configurable fraction)
- **Expected Value (EV)** and **edge** calculations per outcome
- Bookmaker margin extraction and fair probability computation

### 📊 Historical Backtesting

- Chronological train/test split (no data leakage)
- ROI, yield, win rate, max drawdown, profit factor
- Cumulative profit & drawdown charts
- Streak analysis (longest win/loss streaks)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Git
- (Optional) [The Odds API](https://the-odds-api.com/) key for live odds

### Setup

```bash
# Clone the repository
git clone https://github.com/PyayOoKo/football-prediction.git
cd football-prediction

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Collect World Cup data, train XGBoost, generate predictions
python refresh_worldcup.py

# Find value bets (hardcoded odds fallback)
python find_value_bets.py

# Find value bets with live odds (requires THE_ODDS_API_KEY)
export THE_ODDS_API_KEY='your_key_here'
python find_value_bets.py

# Merge StatsBomb xG data for improved accuracy
python merge_all_xg_data.py

# Run historical backtest on league data
python collect_leagues.py --train
python run_backtest.py

# Launch the Streamlit dashboard
python run_dashboard.py
```

---

## 🏆 World Cup 2026 Predictions

The pipeline is currently focused on the **2026 FIFA World Cup**. The refresh cycle:

```
refresh_worldcup.py
    ├── 1. Downloads latest match results from openfootball
    ├── 2. Resolves knockout bracket placeholders
    ├── 3. Retrains XGBoost model on 488 matches (7 World Cups)
    ├── 4. Generates predictions for all upcoming matches
    └── 5. Saves to reports/predictions_worldcup/
```

### Tonight's Round of 16 (July 5, 2026)

| Match | Prediction | Home | Draw | Away |
| :--- | :--- | :---: | :---: | :---: |
| 🇧🇷 **Brazil** vs 🇳🇴 Norway | **Brazil wins** (53% after xG) | **53.0%** | 16.4% | 30.6% |
| 🇲🇽 Mexico vs 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **England** | **England wins** (45% confidence) | 44.6% | 20.5% | 34.9% |

---

## 🖥️ Dashboard

A dark-themed Streamlit dashboard provides interactive visualisation:

```bash
python run_dashboard.py
# Opens at http://localhost:8501
```

### Pages

| Page | Content |
| :--- | :--- |
| **🏠 Dashboard** | Overview, recent matches, model performance |
| **🔮 Predict a Match** | Head-to-head prediction with probability bars |
| **💰 Find Value Bets** | Live vs model odds comparison with Kelly stakes |
| **📊 Backtest** | Historical simulation with ROI, drawdown charts |
| **🏆 World Cup 2026** | Bracket tree, scoreline distributions, confidence analysis |

---

## 📁 Project Structure

```
football_prediction/
├── data/                  # Raw, processed & external datasets (gitignored)
│   ├── raw/               #   Original CSV / API dumps
│   ├── processed/         #   Cleaned, feature-engineered data
│   └── external/          #   Cache files (odds, reference data)
├── models/                # Serialised trained models (gitignored)
├── notebooks/             # Jupyter notebooks for EDA & prototyping
├── reports/               # Backtest charts, predictions CSV (gitignored)
├── scripts/               # Utility scripts
│   └── auto_commit.ps1    #   Hourly GitHub auto-commit
├── src/                   # Source package
│   ├── app/               #   Streamlit dashboard
│   │   ├── dashboard.py   #     Main dashboard
│   │   ├── utils.py       #     Shared caching & helpers
│   │   └── pages/         #     Multi-page views
│   ├── data_collection/   #   Data ingestion modules
│   │   └── sources/       #     openfootball, StatsBomb, football-data
│   ├── odds_api.py        #   The Odds API client (cached, live odds)
│   ├── value_betting.py   #   Kelly criterion, EV, edge computation
│   ├── backtesting.py     #   Historical simulation engine
│   ├── feature_engineering.py  # Rolling averages, H2H, encodings
│   ├── train.py           #   Model training & cross-validation
│   ├── predict.py         #   Match-outcome prediction
│   ├── xg_features.py    #   Expected Goals feature pipeline
│   ├── elo.py            #   Elo rating system
│   ├── poisson_model.py  #   Poisson regression for goals
│   └── evaluate.py       #   Metrics, plots, reports
├── tests/                 # Unit tests
│   └── test_odds_api.py   #   56 tests for the Odds API client
├── config.py              # Centralised typed configuration
├── refresh_worldcup.py    # World Cup data refresh & predict pipeline
├── find_value_bets.py     # Live/hardcoded value bet analysis
├── merge_all_xg_data.py   # xG & xAG data integration
├── run_dashboard.py       # Streamlit dashboard launcher
├── run_backtest.py        # Historical backtest runner
├── setup_auto_commit.bat  # Install hourly GitHub auto-sync
└── setup_scheduler.bat    # Install World Cup refresh scheduler
```

---

## 🔧 Configuration

All settings are centralised in `config.py` under typed dataclasses:

```python
from config import config

# Override training hyper-params
config.train.learning_rate = 0.03
config.train.n_estimators = 500

# Set up live odds
config.odds_api.regions = "uk,ie,eu"

# Kelly betting settings
config.value_betting.kelly_fraction = 0.25  # 25% Kelly
config.value_betting.bankroll = 1000.0
```

### Key Config Sections

| Section | Controls |
| :--- | :--- |
| `config.data` | Data source, split ratios, seed |
| `config.train` | Model type (XGBoost/RF/LR/LGBM/NN), hyper-params, CV folds |
| `config.features` | Rolling windows, H2H, encoding strategy |
| `config.odds_api` | Live odds API key, regions, cache TTL |
| `config.value_betting` | Bankroll, Kelly fraction, min EV |
| `config.backtesting` | Initial bankroll, odds column sets |
| `config.xg` | xG rolling windows, expected points computation |
| `config.elo` | K-factor, home advantage, regression settings |

---

## 🤖 CI & Automation

### Auto-commit to GitHub

An **hourly scheduled task** automatically commits and pushes any changes:

```bash
# Install the task (right-click → Run as Administrator)
setup_auto_commit.bat

# Manual run
schtasks /run /tn "FootballPredictionAutoCommit"
```

### World Cup Data Refresh

```bash
# Install the task (right-click → Run as Administrator)
setup_scheduler.bat
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test suite
pytest tests/test_odds_api.py -v
```

**1,427 tests** across the full project covering feature engineering, betting engine, ETL pipeline, validation, database models, scheduler, cache, services, and more.

---

## 📄 License

MIT — feel free to use, modify, and share.

---

<div align="center">
  <sub>Built with ❤️ using XGBoost, pandas, scikit-learn, Streamlit, and StatsBomb Open Data.</sub>
</div>
