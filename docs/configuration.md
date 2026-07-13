# Configuration

> Centralised configuration hierarchy — loaded from environment variables via `.env`.

## Config Hierarchy

```
Environment Variables (.env)
         │
         ▼
    Config class (Python dataclass)
         │
         ▼
    Singleton: from src.config.settings import config
         │
         ├── config.app           # Application settings
         ├── config.paths         # Directory paths
         ├── config.db            # Database connection
         ├── config.logging       # Logging configuration
         └── config.api           # External API keys
```

## Quick Start

```bash
# Copy the example env file
cp .env.example .env

# Edit with your settings
# .env
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/football_prediction
THE_ODDS_API_KEY=your_key_here
FOOTBALL_DATA_API_KEY=your_key_here
LOG_LEVEL=INFO
```

```python
from src.config.settings import config

# Access any config value
print(config.db.sa_url)          # PostgreSQL connection URL
print(config.paths.data)         # Path("data")
print(config.logging.level)      # "INFO"
print(config.api.odds_api_key)   # "your_key_here"
```

## Environment Variables

### Application

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development`, `staging`, `production` |
| `APP_DEBUG` | `false` | Enable debug mode |
| `SECRET_KEY` | `change-me-in-production` | Session signing key |

### Database

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(composed)* | Full PostgreSQL URL (takes precedence) |
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `football_prediction` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `postgres` | Database password |
| `DB_POOL_SIZE` | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Max overflow connections |
| `DB_POOL_PRE_PING` | `true` | Verify connections before use |
| `DB_ECHO` | `false` | Log all SQL statements |

### Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE` | `true` | Enable file logging |

### External APIs

| Variable | Description |
|---|---|
| `FOOTBALL_DATA_API_KEY` | API key for football-data.org |
| `THE_ODDS_API_KEY` | API key for The Odds API |

## Paths

All paths are relative to the project root. Created automatically on startup.

| Property | Default Path | Description |
|---|---|---|
| `config.paths.root` | *(project root)* | Project root directory |
| `config.paths.data` | `data/` | All data storage |
| `config.paths.raw` | `data/raw/` | Raw imported data |
| `config.paths.processed` | `data/processed/` | Cleaned/processed data |
| `config.paths.external` | `data/external/` | Third-party data |
| `config.paths.models` | `models/` | Trained model files |
| `config.paths.logs` | `logs/` | Log files |
| `config.paths.reports` | `reports/` | Generated reports |

## Logging Configuration

```python
# Default format
"%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"

# Sample output
# 2025-06-10 06:00:00,000 | INFO     | src.etl.pipeline:run:142 | Pipeline complete
```

Log rotation: daily at midnight, retaining 30 days.

## API Configuration

### The Odds API (the-odds-api.com)
- Free tier: 500 requests/month
- Used for: live betting odds, value bet analysis
- Sign up: [https://the-odds-api.com/](https://the-odds-api.com/)

### Football-Data.org
- Free tier: 10 requests/min, limited leagues
- Used for: historical league match data
- Sign up: [https://www.football-data.org/](https://www.football-data.org/)

## Model Configuration

The `config.py` module provides additional model-specific settings:

```python
from config import config

# Training config
config.train.model_type          # "xgboost"
config.train.n_estimators        # 300
config.train.max_depth           # 6
config.train.learning_rate       # 0.01

# Ensemble config
config.ensemble.model_names      # ("xgboost", "logistic_regression", "poisson")
config.ensemble.weight_grid_step # 0.10

# Value betting config
config.value_betting.bankroll         # 1000.0
config.value_betting.kelly_fraction   # 0.25
config.value_betting.min_ev          # 0.05

# Backtesting config
config.backtest.initial_capital       # 1000
config.backtest.commission           # 0.02
```

## .env.example

```ini
# ── Application ────────────────────────────────
APP_ENV=development
APP_DEBUG=true
SECRET_KEY=change-me-in-production

# ── Database (PostgreSQL) ──────────────────────
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/football_prediction

# ── Logging ────────────────────────────────────
LOG_LEVEL=INFO
LOG_FILE=true

# ── External APIs ──────────────────────────────
FOOTBALL_DATA_API_KEY=
THE_ODDS_API_KEY=
```
