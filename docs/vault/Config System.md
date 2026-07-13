---
tags:
  - football-prediction
  - config
  - settings
created: 2026-07-12
---

# ⚙️ Config System

> The central configuration hub. All sub-modules import `config` from `config.py`.

See also: [[Architecture Overview]], [[Feature Engineering Pipeline]], [[Ensemble Model]]

---

## Architecture

```mermaid
graph TD
    ENV[".env file"] -->|load_dotenv| CFG["config.Config (singleton)"]
    
    CFG --> PATHS["config.paths"]
    CFG --> DATA_CFG["config.data"]
    CFG --> DC_CFG["config.data_collection"]
    CFG --> FEAT_CFG["config.features"]
    CFG --> TRAIN_CFG["config.train"]
    CFG --> PRED_CFG["config.predict"]
    CFG --> XG_CFG["config.xg"]
    CFG --> POI_CFG["config.poisson"]
    CFG --> ELO_CFG["config.elo"]
    CFG --> ENS_CFG["config.ensemble"]
    CFG --> DC_CFG["config.dixon_coles"]
    CFG --> ODDS_CFG["config.odds"]
    CFG --> VB_CFG["config.value_betting"]
    CFG --> BT_CFG["config.backtesting"]
    
    PATHS --> RM["data/raw/"]
    PATHS --> PM["data/processed/"]
    PATHS --> EXTM["data/external/"]
    PATHS --> MODM["models/"]
    
    classDef cfg fill:#2c3e50,stroke:#34495e,color:white
    class CFG cfg
```

**File:** [[config.py]]

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Singleton pattern** | Single `Config` dataclass with 18 nested sub-configs |
| **Auto `.env` loading** | `load_dotenv()` runs at import time — no manual setup |
| **Auto directory creation** | `Paths.__post_init__()` creates all dirs on first import |
| **Full type hints** | All fields annotated (`Literal`, `tuple`, `int`, `float`, etc.) |
| **Mutable at runtime** | Scripts can override any field (e.g. `config.elo.home_advantage = 50`) |

---

## Key Config Sections

| Config Object | Controls | Key Fields |
|--------------|----------|------------|
| `config.paths` | All file/directory paths | `raw`, `processed`, `models`, `external` |
| `config.data_collection` | Data downloading | `leagues`, `max_seasons`, `missing_strategy` |
| `config.features` | Feature engineering | `form_window`, `rolling_windows`, `include_h2h`, `categorical_encoding` |
| `config.train` | Model training | `model_type`, `n_estimators`, `max_depth`, `learning_rate` |
| `config.elo` | Elo rating system | `k`, `home_advantage`, `initial_rating`, `adjustments` |
| `config.poisson` | Poisson model | `min_matches`, `max_goals` |
| `config.ensemble` | Ensemble model | `model_names`, `weight_grid_step` |
| `config.odds` | Odds processing | `opening_odds_cols`, `closing_odds_cols` |
| `config.value_betting` | Value bet detection | `bankroll`, `kelly_fraction`, `min_ev` |
| `config.backtesting` | Backtesting engine | `initial_bankroll`, `kelly_fraction`, `min_ev` |

---

## Usage Example

```python
from config import config

# Read
print(config.elo.home_advantage)    # 100
print(config.train.model_type)      # 'xgboost'

# Override at runtime
config.elo.home_advantage = 50       # neutral venue
config.train.n_estimators = 300
config.ensemble.model_names = (
    "xgboost", "logistic_regression", "poisson"
)
```

---

## `.env` File

Auto-loaded by `config.py`. Located in project root.

```ini
APP_ENV=development
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/football_prediction
THE_ODDS_API_KEY=your_key_here
FOOTBALL_DATA_API_KEY=your_key_here
LOG_LEVEL=INFO
```
