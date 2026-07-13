---
tags:
  - football-prediction
  - betting
  - value
  - backtesting
created: 2026-07-12
---

# 💰 Value Betting & Backtesting

> Identify betting opportunities and simulate strategies on historical data.

See also: [[Ensemble Model]], [[Config System]], [[Runtime Sequence Diagrams]]

---

## Value Betting System

**File:** [[value_betting.py]]

### How It Works

```mermaid
graph LR
    subgraph "Input"
        ODDS_IN["Bookmaker Odds<br/>(decimal: H, D, A)"]
        PROBS_IN["Model Probabilities<br/>(away, draw, home)"]
    end
    
    subgraph "Calculations"
        IP["Implied Probability<br/>IP = 1 / odds"]
        MARGIN["Bookmaker Margin<br/>margin = Σ IP - 1"]
        FAIR["Fair Probability<br/>fair = IP / (1 + margin)"]
        EV["Expected Value<br/>EV = (model × odds) - 1"]
        KELLY["Kelly Stake<br/>K = EV / (odds - 1) × fraction"]
    end
    
    ODDS_IN --> IP
    IP --> MARGIN
    MARGIN --> FAIR
    PROBS_IN --> EV
    FAIR --> EV
    EV --> KELLY
    
    KELLY --> OUTPUT["Value Bet DataFrame"]
    OUTPUT --> C1["positive_ev: bool"]
    OUTPUT --> C2["ev: expected value"]
    OUTPUT --> C3["kelly_pct: stake %"]
    OUTPUT --> C4["prob_edge: model - fair"]
    OUTPUT --> C5["recommendation: text"]
```

### Calculations Explained

| Step | Formula | Example |
|------|---------|---------|
| **Implied Probability** | `IP = 1 / decimal_odds` | Odds 2.10 → IP = 47.6% |
| **Bookmaker Margin** | `margin = Σ IP - 1` | 47.6% + 29.4% + 26.3% - 1 = 3.3% |
| **Fair Probability** | `fair = IP / (1 + margin)` | 47.6% / 1.033 = 46.1% |
| **Expected Value** | `EV = (model × odds) - 1` | (0.52 × 2.10) - 1 = +9.2% |
| **Kelly Stake** | `k = EV / (odds - 1) × fraction` | 9.2% / 1.10 × 0.25 = 2.1% |

### API

```python
from src.value_betting import compute_value_bets

bets = compute_value_bets(
    odds=[[2.10, 3.40, 3.80], [1.95, 3.50, 4.00]],
    model_probs=[[0.52, 0.28, 0.20], [0.48, 0.30, 0.22]],
    team_matches=[("Arsenal", "Chelsea"), ("Liverpool", "Man City")],
    bankroll=1000.0,
    kelly_fraction=0.25,
    min_ev=0.0,
)

good_bets = bets[bets["positive_ev"]]
```

---

## Confidence Scoring

**File:** [[confidence_scoring.py]]

```mermaid
graph LR
    PROBS["Model Probabilities<br/>(n, 3)"] --> SPREAD["Spread Score (40%)<br/>1 - entropy / log₂(3)"]
    ENS_PROBS["Individual Model Probs"] --> AGREE["Agreement Score (35%)<br/>1 - σ_models / 0.5"]
    BRIER["Calibration Brier Score"] --> CALIB["Calibration Score (25%)<br/>1 - Brier / 2.0"]
    SPREAD & AGREE & CALIB --> COMPOSITE["Composite Confidence<br/>0 - 100"]
```

**3 components:** Spread (40%) + Agreement (35%) + Calibration (25%)

---

## Backtesting Engine

**File:** [[backtesting.py]]

```mermaid
graph TD
    subgraph "Setup"
        INIT["BacktestEngine<br/>(model, bankroll=1000, kelly=0.25)"]
    end
    
    subgraph "Run"
        INIT --> RUN["engine.run(X_test, y_test, odds_df)"]
        RUN --> LOOP["For each test match:"]
        LOOP --> GET_ODDS["Get odds + model probs"]
        GET_ODDS --> COMPUTE["Compute EV per outcome"]
        COMPUTE --> DECIDE{"EV > min_ev?"}
        DECIDE -->|YES| PLACE["Place bet (Kelly stake)"]
        DECIDE -->|NO| SKIP["Skip match"]
        PLACE --> TRACK["Record BetRecord"]
        SKIP --> TRACK
        TRACK --> NEXT["Next match"]
    end
    
    subgraph "Metrics"
        NEXT --> METRICS["calculate_metrics()"]
        METRICS --> ROI["ROI %"]
        METRICS --> YIELD["Yield %"]
        METRICS --> WINRATE["Win Rate %"]
        METRICS --> DRAWDOWN["Max Drawdown %"]
        METRICS --> PROFIT["Profit Factor"]
        METRICS --> STREAKS["Longest Streaks"]
    end
```

### Metrics Explained

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **ROI** | `(final - initial) / initial × 100` | Total return on bankroll |
| **Yield** | `profit / staked × 100` | Return per unit staked |
| **Win Rate** | `wins / total × 100` | % of bets won |
| **Max Drawdown** | `max(peak - trough) / peak × 100` | Worst losing streak |
| **Profit Factor** | `gross_profit / gross_loss` | Risk/reward ratio |

### API

```python
from src.backtesting import BacktestEngine

engine = BacktestEngine(
    model=model,
    initial_bankroll=1000.0,
    kelly_fraction=0.25,
    min_ev=0.0,
)

metrics = engine.run(X_test, y_test, odds_df=odds_df)

engine.print_report()
chart_paths = engine.plot_results(output_dir="reports/backtest")
```
