# Troubleshooting

> Common issues and their solutions.

## Installation Issues

### `ModuleNotFoundError: No module named 'pandas'`

**Cause:** Virtual environment not activated.

**Solution:**
```bash
# Activate venv
source .venv/Scripts/activate  # Git Bash
.venv\Scripts\activate          # Command Prompt

# Install requirements
pip install -r requirements.txt
```

### `Failed to build numpy` / `Failed to build wheel`

**Cause:** Using MinGW Python (common with Inkscape, Git Bash for Windows) which can't use pre-compiled Windows wheels.

**Solution:** Use Microsoft Store Python instead:
```bash
# Find Microsoft Store Python
"/c/Users/dell/AppData/Local/Microsoft/WindowsApps/python3.exe" -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### `SSL: CERTIFICATE_VERIFY_FAILED`

**Cause:** SSL certificate verification failure, often due to corporate proxy or outdated certificates.

**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Data Issues

### `FileNotFoundError: data/raw/worldcup_all.csv`

**Cause:** Data has not been collected yet.

**Solution:**
```bash
# Collect all World Cup data
python collect_all_worldcups.py

# Or collect specific leagues
python collect_leagues.py
```

### No data in dashboard

**Cause:** Preprocessed data file missing.

**Solution:**
```bash
# Check if data exists
ls data/processed/results_clean.csv

# If missing, run preprocessing
python -c "from src.preprocessing import run_preprocessing; run_preprocessing()"
```

## Model Issues

### `No Model Loaded` warning in dashboard

**Cause:** No trained model found in `models/` directory.

**Solution:**
```bash
# Train a model
python train_xgboost.py
# or
python train_worldcup.py
```

### Model training is slow

**Cause:** Multiple factors — see [performance guide](troubleshooting.md#performance).

**Solution:**
```bash
# Fastest path — skip tuning, use defaults
python train_worldcup.py  # ~20-30 seconds

# If still slow, check: is hyperparameter tuning enabled?
# Set tune_base_models = False in config.py
```

## Database Issues

### `Connection refused` to PostgreSQL

**Cause:** PostgreSQL is not running.

**Solution:**
```bash
# With Docker Compose
docker-compose up -d db

# Verify
docker-compose ps db

# Check if PostgreSQL is running locally
pg_isready
```

### `psycopg2.OperationalError: could not connect to server`

**Cause:** Wrong host, port, credentials, or PostgreSQL not started.

**Solution:**
```bash
# Test connection
python -c "from src.database.session import get_engine; print(get_engine())"

# Check .env settings
cat .env | grep DATABASE_URL
```

### Alembic migration fails

**Cause:** Migration conflict or missing dependencies.

**Solution:**
```bash
# Check current migration version
alembic current

# Manually fix: stamp to a known version
alembic stamp head

# Or rollback and retry
alembic downgrade -1
alembic upgrade head
```

## Performance Issues

### Full pipeline takes >5 minutes

**Cause:** Possible issues with data download, hyperparameter tuning, or player data collection.

**Solution:**
```bash
# Fastest command
python run_pipeline.py --lightweight  # ~5-10 seconds

# Skip download if data is recent
python run_pipeline.py --skip-download  # ~30-60 seconds

# Skip retrain if model is current
python run_pipeline.py --skip-retrain

# Skip lineup collection (adds 30-60s)
python refresh_worldcup.py --skip-lineups
```

### Dashboard loads slowly

**Cause:** Large dataset or uncached model diagnostic.

**Solution:**
- Wait for the initial cache to populate (only slow on first load)
- Reduce dataset size for development:
  ```bash
  # Use a subset of data
  head -1000 data/processed/results_clean.csv > data/processed/results_clean_dev.csv
  ```

### MemoryError when processing data

**Cause:** Dataset too large for available RAM.

**Solution:**
```bash
# Use chunked processing
python run_pipeline.py --batch-size 1000

# Or reduce data scope
python train_league.py --league E0  # Single league is faster
```

## Python/Environment Issues

### Wrong Python version

```bash
# Check version
python --version  # Must be 3.12+

# If wrong version, install Python 3.12 from python.org or Microsoft Store
```

### `ImportError` when running scripts

**Cause:** Running from wrong directory or missing `__init__.py`.

**Solution:**
```bash
# Always run from project root, not from src/
cd /path/to/football-prediction

# Run scripts directly
python run_pipeline.py

# Or as module
python -m src.scheduler.cli list
```

### Git hooks not running

**Cause:** Pre-commit not installed.

**Solution:**
```bash
# Install hooks
pre-commit install

# Run once to verify
pre-commit run --all-files
```

## Scheduler Issues

### Windows scheduled task doesn't run

**Solution:**
```bash
# Check task exists
schtasks /query /TN "FootballPredictionPipeline"

# Reinstall
python -m src.scheduler.cli install-windows

# Ensure Python path in task points to .venv
```

### Cron job not executing

```bash
# Check cron logs
grep CRON /var/log/syslog

# Verify script path in crontab is absolute
crontab -l

# Test manually
/opt/football-prediction/.venv/bin/python /opt/football-prediction/run_pipeline.py
```

## Data Versioning Issues

### `Version not found`

**Solution:**
```bash
# List all versions
python -m src.data_versioning.cli list-versions

# Check storage directory
ls data/versions/
```

### Integrity check fails

**Solution:**
```bash
# Recreate version from source data
python -m src.data_versioning.cli create-version \
    --file data/raw/results.csv --source football-data

# Verify specific version
python -m src.data_versioning.cli verify --version v003
```

## Common Error Messages

| Error | Likely Cause | Solution |
|---|---|---|
| `No module named 'src'` | Running from wrong directory | `cd` to project root |
| `psycopg2.OperationalError` | PostgreSQL not running | `docker-compose up -d db` |
| `FileNotFoundError: data/raw/...` | Data not collected | Run `collect_*.py` scripts |
| `ModelNotFoundError` | No trained model | Run `train_*.py` scripts |
| `alembic.util.exc.CommandError` | Migration conflict | `alembic stamp head` |
| `ValueError: cannot create version from empty DataFrame` | Empty CSV | Check source file |

## Still Stuck?

1. **Check the logs:** `tail -f logs/football_prediction.log`
2. **Enable debug logging:** Set `LOG_LEVEL=DEBUG` in `.env`
3. **Enable SQL echo:** Set `DB_ECHO=true` in `.env`
4. **Run with verbose output:** `python -m src.scheduler.cli run -v`
5. **Open an issue:** GitHub Issues page
