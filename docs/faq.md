# Frequently Asked Questions

## General

### What does this project do?
It predicts football (soccer) match outcomes using an ensemble machine learning model (XGBoost + Logistic Regression + Poisson). It collects historical match data, engineers features (ELO ratings, rolling stats, team form, xG data), trains models, evaluates them, and helps find value betting opportunities.

### What leagues are supported?
Over 50 leagues across 10+ countries including:
- **England:** Premier League, Championship, League One, League Two, FA Cup, EFL Cup
- **Germany:** Bundesliga, 2. Bundesliga, 3. Liga
- **Spain:** La Liga, La Liga 2
- **Italy:** Serie A, Serie B
- **France:** Ligue 1, Ligue 2
- **Netherlands:** Eredivisie
- **International:** World Cup, Champions League, Europa League

### How accurate are the predictions?
Typical test accuracy: **62-68%** (vs ~45% baseline for always-predicting-home). Log-loss typically ranges from **0.58-0.65**. Accuracy varies by league — larger leagues with more data tend to perform better.

### Do I need an API key?
- **The Odds API key** — optional, required only for live betting odds in value bet analysis (free tier: 500 req/month)
- **football-data.org key** — optional, used for historical league data (free tier: 10 req/min, limited leagues)
- The basic functionality works with the included scrapers without any API keys

## Setup

### What Python version do I need?
Python 3.12 or higher.

### How do I set up on Windows?
See the [GUIDE.md](../GUIDE.md) or [Developer Guide](developer_guide.md#setup). The key steps:
1. Install Python 3.12+ (from python.org or Microsoft Store — **not** Inkscape's MinGW Python)
2. Clone the repo
3. Create venv: `python -m venv .venv`
4. Activate and install: `pip install -r requirements.txt`

### Can I use SQLite instead of PostgreSQL?
Yes. For local development:
```bash
export DATABASE_URL=sqlite:///data/football.db
```
PostgreSQL is recommended for production with 100K+ rows.

### How long does the first setup take?
- Installing dependencies: 2-5 minutes
- Collecting World Cup data: 2-5 minutes
- Collecting league data: 5-30 minutes (depending on leagues)
- Training first model: 20-30 seconds
- **Total:** 10-40 minutes

## Models

### What model does the system use?
An **ensemble model** combining:
1. **XGBoost** — gradient boosted trees (primary learner)
2. **Logistic Regression** — linear baseline
3. **Poisson Model** — goal-based scoring model

Weights are optimised via grid search on a validation set.

### Why not use deep learning?
Football match data is relatively small (tens of thousands of matches vs millions of images/text samples). Gradient boosted trees consistently outperform deep learning on this type of tabular data with strong feature engineering. The project has optional PyTorch support for experimentation.

### How often should I retrain?
The pipeline retrains automatically when:
- Data is stale (>7 days old, configurable)
- Every 10 runs (configurable)
- You can force retrain with `--skip-retrain` flag

For best results, retrain **weekly** or after collecting significant new data.

## Data

### Where does the data come from?
- **football-data.co.uk** — Historical match results with odds (CSV, manually downloaded or via football-data.org API)
- **FBref** — Advanced stats (xG, possession, passes) via scraping
- **Transfermarkt** — Player values, lineups via scraping
- **The Odds API** — Live betting odds

### How much data is collected?
- World Cup: every tournament from 1930 to present (~900 matches)
- Leagues: varies by league, typically 5-20 seasons (5K-20K matches per league)
- Total: 50K-500K+ matches depending on configured leagues

### Can I add my own data?
Yes! The ETL pipeline accepts CSV files with standard columns (date, home_team, away_team, home_goals, away_goals, result, league). Place your CSV in `data/raw/` and run:
```python
from src.etl.pipeline import ETLPipeline
pipeline.run()
```

## Dashboard

### How do I start the dashboard?
```bash
python run_dashboard.py
```
Then open http://localhost:8501 in your browser.

### The dashboard shows "No Model Loaded"
Run a training script first:
```bash
python train_xgboost.py
```

### Can I access the dashboard remotely?
By default, Streamlit runs on localhost only. For remote access:
```bash
streamlit run src/app/dashboard.py --server.address=0.0.0.0
```
**⚠️ Do this only in a trusted network or with authentication enabled.**

## Value Betting

### What is a "value bet"?
A bet where the model's estimated probability of an outcome is **higher** than the probability implied by the bookmaker's odds. If the model is well-calibrated, these bets have positive expected value.

### How is the Kelly criterion used?
The system uses fractional Kelly (default: 25% of full Kelly) to calculate optimal bet size:
```
stake = bankroll × kelly_fraction × (fair_prob × odds - 1) / (odds - 1)
```
This balances growth with risk management.

### Do I need live odds?
The Odds API key is required for live odds. Without it, you can use historical odds from football-data.co.uk for backtesting.

## Scheduler

### How do I automate daily runs?
```bash
# Linux/macOS: Add to crontab
0 6 * * * cd /path/to/football-prediction && .venv/bin/python run_pipeline.py

# Windows: Install scheduled task
python -m src.scheduler.cli install-windows
```

### What tasks run automatically?
1. Download new match data
2. Validate data quality
3. Clean and deduplicate
4. Update database and retrain model
5. Backup database
6. Rotate logs and archive reports

## Performance

### Why is my pipeline slow?
The most common causes:
1. **Data download** — slow internet or source throttling
2. **Hyperparameter tuning** — set `tune_base_models = False` in `config.py`
3. **Player data collection** — use `--skip-lineups`
4. **Large dataset** — 100K+ rows takes longer

### Fastest way to test:
```bash
python run_pipeline.py --lightweight  # ~5-10 seconds
```

### Memory usage seems high
- Feature engineering on 500K+ matches: ~2GB RAM
- XGBoost training: ~500MB-1GB
- Streamlit dashboard: ~200MB
- **Total:** 2-4GB

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`
Run all commands from the project root directory, not from `src/`.

### `ImportError` in production
Make sure the virtual environment is activated and requirements are installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt --no-deps
```

### Database connection fails
```bash
# Check if PostgreSQL is running
docker-compose ps db

# Verify .env settings
grep DATABASE_URL .env
```

## Contributing

### How can I contribute?
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests: `python -m pytest`
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

### What should I work on?
Check the GitHub Issues page for:
- Bug fixes
- New features (more league support, different model types)
- Documentation improvements
- Performance optimizations
- Test coverage

## Technical

### What database schema does the system use?
22 tables across 4 domains: Core (matches, odds, teams), ML Ops (experiment tracking), Feature Store, and Monitoring. See [Database Schema](database.md) for full ER diagram.

### How does data versioning work?
Imports are tracked as versioned snapshots with:
- SHA256 checksums for integrity
- Delta computation (inserted/updated/deleted records)
- Rollback capability
- Git commit tracking
- See [Data Versioning CLI](cli.md#data-versioning-cli)

### Can I run this in Docker?
Yes! Docker Compose is provided:
```bash
docker-compose up --build -d
```
See [Deployment Guide](deployment_guide.md).
