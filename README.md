<div align="center">

# Football Match Outcome Prediction

**A machine learning pipeline for predicting football match outcomes, finding value bets, and simulating tournaments — with live odds integration.**

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## Installation

### Prerequisites

- Python 3.12+
- Git

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd football-prediction

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

# Install core dependencies
pip install --upgrade pip
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install API dependencies (optional, for REST API)
pip install -e ".[api]"
```

## Quick Start

```bash
# Train a model
football-predict train

# Generate predictions
football-predict predict

# Launch the dashboard
football-predict dashboard

# Start the REST API
football-predict api

# Run the full pipeline
football-predict pipeline
```

### Alternative entry points

```bash
# Run module directly
python -m src

# Start API server
python -m uvicorn api.main:app --reload --port 8000

# Start Streamlit monitoring dashboard
python -m streamlit run dashboard/app.py
```

---

## Commands

| Command | Description |
| :--- | :--- |
| `football-predict train` | Train a prediction model |
| `football-predict predict` | Generate match predictions |
| `football-predict evaluate` | Evaluate model performance |
| `football-predict collect` | Download match data |
| `football-predict backtest` | Run betting backtest |
| `football-predict dashboard` | Launch Streamlit dashboard |
| `football-predict api` | Start REST API server |
| `football-predict desktop` | Launch desktop app |
| `football-predict pipeline` | Run the full pipeline |

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage (minimum 75%)
pytest --cov=src --cov-report=term-missing --cov-fail-under=75

# Run specific test file
pytest tests/test_phase2_core_fixes.py -v

# Run fast tests only (skip slow)
pytest -m "not slow"
```

## Linting & Type Checking

```bash
# Format check
black --check src/ tests/

# Lint
ruff check src/

# Type check
mypy src/
```

---

## API

Start the REST API:

```bash
# Set authentication key (required in production)
export PREDICTION_API_KEY='your-secret-key'

# Start server
football-predict api

# Or directly with uvicorn
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Authentication

- **Production**: Set `PREDICTION_API_KEY` environment variable.
  Requests must include `Authorization: Bearer <key>` header.
- **Development**: Set `APP_ENV=development` or `API_AUTH_DISABLED=true`
  to disable authentication.

### Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/predict` | Predict match outcomes |

### Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer $PREDICTION_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fixtures": [{"home_team": "Brazil", "away_team": "Argentina"}]}'
```

---

## Docker

### Build

```bash
docker build -t football-prediction .
```

### Run (API mode)

```bash
docker run --rm -it \
  -e PREDICTION_API_KEY=your-key \
  -p 8000:8000 \
  football-prediction api
```

### docker-compose (API + PostgreSQL)

```bash
# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d

# Run migrations only
docker-compose run --rm migrate
```

---

## Configuration

All settings are centralised in `config.py`. Key sections:

| Section | Environment Variable | Description |
| :--- | :--- | :--- |
| `config.data` | `DATABASE_URL` | Data source, database connection |
| `config.train` | — | Model type, hyper-parameters |
| `config.features` | — | Rolling windows, H2H, encoding |
| `config.odds_api` | `THE_ODDS_API_KEY` | Live odds API |
| `config.value_betting` | — | Bankroll, Kelly fraction |
| `config.elo` | — | K-factor, home advantage |

### Environment Variables

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `APP_ENV` | No | `production` | `development` or `production` |
| `PREDICTION_API_KEY` | Production | — | API authentication key |
| `DATABASE_URL` | Production | — | Database connection string |
| `THE_ODDS_API_KEY` | No | — | Live odds API key |
| `LOG_LEVEL` | No | `INFO` | Logging level |

---

## Model Artifact Format

Trained models are saved as `.joblib` files in `models/`. The artifact
may include:

- Trained model object (with `predict()` and `predict_proba()`)
- Feature names (`feature_names_in_`)
- Encoder state (for target encoding during inference)
- Training metadata

## Feature Selection

Feature selection (optional) reduces dimensionality before training.
Configured via `config.feature_selection`:

- `method`: `"mutual_info"`, `"rfe"`, `"l1"`, `"threshold"`
- `n_features`: Number of features to keep

When enabled, selection is **train-only** — no leakage from validation/test data.

## Leakage Prevention

- Rolling features use `.shift(1)` so current match data never
  influences its own features.
- Target encoding stores priors from training data only.
- Chronological train/val/test split (no shuffling).
- League positions are computed from prior matches only.
- `DataPreprocessor.fit()` learns statistics from training data;
  `transform()` only applies stored state.

---

## Project Structure

```
football_prediction/
├── data/                  # Raw, processed & external datasets (gitignored)
├── models/                # Serialised trained models (gitignored)
├── reports/               # Backtest charts, predictions (gitignored)
├── src/                   # Source package
│   ├── data/              #   Data loading, cleaning, preprocessing
│   ├── features/          #   Feature engineering sub-package
│   ├── services/          #   Business logic (training, prediction)
│   └── cli.py             #   CLI entry point
├── api/                   # REST API (FastAPI)
├── dashboard/             # Streamlit monitoring dashboard
├── app/                   # Desktop application
├── tests/                 # Unit tests
├── config.py              # Centralised configuration
├── pyproject.toml         # Project metadata & dependencies
└── Dockerfile             # Container build
```

---

## License

MIT — feel free to use, modify, and share.
