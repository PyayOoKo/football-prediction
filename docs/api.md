# API Reference

> The project currently uses a Streamlit dashboard UI. This page documents the internal interfaces and planned REST API.

## Streamlit Dashboard Pages

The primary user interface is a multi-page Streamlit application.

### Main Dashboard
**Route:** `/` (`src/app/dashboard.py`)

Displays:
- Key metrics (match count, teams, model type, latest date)
- Recent matches table (last 15)
- Quick action links to other pages
- Model performance diagnostic
- Per-class balance check (confusion matrix, precision/recall/f1)
- Draw blindness detection

### Page 1 — Predict
**Route:** `/pages/1_Predict.py`

Match outcome prediction:
- Select home team and away team from dropdowns
- Display model probabilities (Home Win / Draw / Away Win)
- Show head-to-head history
- Expected goals

### Page 2 — Value Bets
**Route:** `/pages/2_Value_Bets.py`

Value betting analysis:
- Compare model probabilities vs bookmaker odds
- Filter by minimum expected value
- Bankroll management with Kelly criterion
- Sortable table of value opportunities

### Page 3 — Backtest
**Route:** `/pages/3_Backtest.py`

Historical performance simulation:
- Backtest betting strategy on historical data
- Equity curve visualization
- Sharpe ratio, max drawdown, ROI
- Per-season breakdown

### Page 4 — World Cup
**Route:** `/pages/4_WorldCup.py`

World Cup predictions:
- Bracket visualization
- Match-by-match predictions
- Group stage probabilities
- Knockout round simulation

## Internal API (Python)

### Model Loading

```python
from src.app.utils import load_model, load_clean_data

# Cached model loading
model = load_model()  # Tries ensemble → xgboost → league model
model = load_model("my_model.joblib")  # Specific file

# Data loading
df = load_clean_data()  # Cached results_clean.csv
```

### Feature Engineering

```python
from src.feature_engineering import build_features, train_val_test_split

X, y = build_features(df, is_training=True)
splits = train_val_test_split(X, y)
```

### Prediction

```python
from src.predict import predict_match

probs = predict_match(model, home_team="Arsenal", away_team="Chelsea", df=df)
print(probs)  # {"home_win": 0.52, "draw": 0.28, "away_win": 0.20}
```

### Value Betting

```python
from src.value_betting import find_value_bets

bets = find_value_bets(
    model=model,
    df=df,
    odds_df=df_with_odds,
    min_ev=0.05,
    kelly_fraction=0.25,
)
```

### Backtesting

```python
from src.backtesting import run_backtest

result = run_backtest(
    model=model,
    X_test=X_test,
    y_test=y_test,
    odds_df=odds_df,
    initial_bankroll=1000,
)
```

## Shared Utilities (src/app/utils.py)

| Function | Description | Caching |
|---|---|---|
| `load_model()` | Load trained model | `@st.cache_resource` |
| `load_clean_data()` | Load preprocessed data | `@st.cache_resource` |
| `build_feature_matrix(df)` | Build feature matrix | `@st.cache_data` |
| `get_available_teams(df)` | Team list from data | — |
| `get_latest_matches(df, n=20)` | N most recent matches | — |
| `get_matchup_stats(df, home, away)` | H2H stats | — |
| `run_model_diagnostic(model, df)` | Full model evaluation | `@st.cache_data` |
| `run_backtest_cached(...)` | Cached backtest | `@st.cache_data` |

## Planned REST API (Future)

The following endpoints are planned for Phase Two:

```
GET  /api/v1/health                  → Health check
GET  /api/v1/matches                 → List matches (paginated, filterable)
GET  /api/v1/matches/{id}            → Single match details
GET  /api/v1/teams                   → List teams
GET  /api/v1/teams/{id}/form        → Team form history
POST /api/v1/predict                 → Predict match outcome
GET  /api/v1/value-bets              → Current value betting opportunities
GET  /api/v1/backtest                → Backtest results
POST /api/v1/experiments             → Create experiment
GET  /api/v1/experiments/{id}/runs   → Run history for experiment
```

### Projected API Schema

```json
POST /api/v1/predict
{
  "home_team": "Arsenal",
  "away_team": "Chelsea"
}

Response 200:
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "probabilities": {
    "home_win": 0.52,
    "draw": 0.28,
    "away_win": 0.20
  },
  "predicted_outcome": "home_win",
  "confidence": 0.72,
  "model": "ensemble",
  "model_version": "v3"
}
```

## Database Session API

```python
from src.database import get_session

# Context manager — auto-commits on success, rolls back on error
with get_session() as session:
    matches = session.query(Match).filter_by(league="E0").all()
```
