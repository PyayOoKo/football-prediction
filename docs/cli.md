# CLI Reference

> All command-line interfaces across the project — data versioning, scheduler, ETL, and training.

## Data Versioning CLI

```bash
python -m src.data_versioning.cli [options] <command> [args]
```

### Commands

#### `create-version`
Create a new dataset version from a CSV or Parquet file.

```bash
python -m src.data_versioning.cli create-version \
    --file data/raw/results.csv \
    --source football-data \
    --league E0 \
    --season 2025/2026 \
    --schema-version v2 \
    --pipeline-version v3 \
    --user analyst \
    --notes "Initial import for 2025/26 season" \
    --tag environment=production
```

| Flag | Default | Description |
|---|---|---|
| `--file, -f` | *(required)* | Path to CSV or Parquet |
| `--source, -s` | filename stem | Data source identifier |
| `--league, -l` | `""` | League code |
| `--season` | `""` | Season identifier |
| `--schema-version` | `""` | Schema version tag |
| `--pipeline-version` | `""` | Pipeline version tag |
| `--user, -u` | `cli` | User creating the version |
| `--notes, -n` | `""` | Optional description |
| `--tag` | *(repeatable)* | Key=value tags |

**Output:**
```
  ✅ Version created: v012
     Rows:      15,234
     Columns:   42
     Source:    football-data
     League:    E0
     Season:    2025/2026
     Schema:    v2
     Pipeline:  v3
     Hash:      a1b2c3d4e5f6...
     Path:      data/versions/v012/results.parquet
```

#### `list-versions`
List all dataset versions with summary metadata.

```bash
python -m src.data_versioning.cli list-versions
```

**Output:**
```
  ID       Created                 Source               League Season       Rows  Hash
  ─────────────────────────────────────────────────────────────────────────────────────
  v001     2025-01-15 10:30:00 UTC football-data         E0     2024/25    14,821  a1b2...  ← CURRENT
  v002     2025-03-01 08:15:00 UTC football-data         E0     2024/25    15,102  c3d4...
  v003     2025-06-10 06:00:00 UTC football-data         E0     2025/26    15,234  e5f6...

  3 version(s) total
  Active: v003
```

#### `compare`
Show row-level diff between two versions.

```bash
python -m src.data_versioning.cli compare v001 v003
```

**Output:**
```
  ══════════════════════════════════════════════════════════════════════
  DIFF: v001 → v003
  ══════════════════════════════════════════════════════════════════════

  Metric                    Count
  ────────────────────────────────
  Unchanged                 14,821
  Inserted                    413
  Updated                       0
  Deleted                       0
  Total changed               413
```

#### `rollback`
Restore a previous version as current.

```bash
python -m src.data_versioning.cli rollback v002 --user admin
```

| Flag | Default | Description |
|---|---|---|
| `version_id` | *(required)* | Target version |
| `--no-backup` | `False` | Skip backup before rollback |
| `--user` | `cli` | User performing rollback |

#### `verify`
Verify data integrity via SHA256 hash comparison.

```bash
python -m src.data_versioning.cli verify --version v003
```

| Flag | Default | Description |
|---|---|---|
| `--version` | *(all)* | Specific version to verify |

#### `info`
Show detailed metadata for a specific version.

```bash
python -m src.data_versioning.cli info v003
```

**Output:**
```
  ──────────────────────────────────────────────────
  Version: v003
  ──────────────────────────────────────────────────
  Created:        2025-06-10 06:00:00 UTC
  Source:         football-data
  League:         E0
  Season:         2025/2026
  Schema version: v2
  Pipeline v.:    v3
  Git commit:     a1b2c3d4e5f6
  Rows:           15,234
  Columns:        42
  Hash:           a1b2c3d4e5f6...
  User:           system
  Import time:    3.45s
  Added records:  413
  Data path:      data/versions/v003/results.parquet
  Tags:           {"environment": "production"}
```

## Scheduler CLI

```bash
python -m src.scheduler.cli [command] [options]
```

| Command | Description |
|---|---|
| `run` | Execute all or selected tasks |
| `list` | List all registered tasks |
| `status` | Show status of last run |
| `install-windows` | Install as Windows scheduled task |

### `run` options

| Flag | Default | Description |
|---|---|---|
| `--tasks` | *(all)* | Comma-separated task names |
| `--config` | *(default)* | Path to YAML config |
| `--abort-on-failure` | `True` | Stop on task failure |

## Run Pipeline Script

```bash
python run_pipeline.py [options]
```

| Flag | Description |
|---|---|
| `--skip-download` | Skip data download step |
| `--skip-retrain` | Skip model retraining |
| `--lightweight` | Skip download + retrain (predict only) |
| `--config` | Path to custom config file |

## Dashboard

```bash
python run_dashboard.py
```

Launches the Streamlit dashboard at `http://localhost:8501`.

## Training Scripts

| Script | Description |
|---|---|
| `python train_xgboost.py` | Train XGBoost model on collected data |
| `python train_worldcup.py` | Train model specifically for World Cup data |
| `python train_league.py` | Train per-league models |
| `python train_with_xag.py` | Train using xAG (expected assisted goals) features |
| `python run_first_model.py` | Train and evaluate the first baseline model |
| `python run_backtest.py` | Run backtesting simulation |
| `python run_combined_pipeline.py` | Full combined training pipeline |

## Data Collection Scripts

| Script | Description |
|---|---|
| `python collect_all_worldcups.py` | Collect all World Cup historical data |
| `python collect_leagues.py` | Collect league-specific data |
| `python collect_lineups.py` | Collect lineup data from Transfermarkt |
| `python collect_player_data.py` | Collect player-level statistics |
| `python collect_r16_data.py` | Collect Round of 16 / knockout data |
| `python collect_worldcup.py` | Collect World Cup tournament data |
| `python collect_worldcup_xg.py` | Collect World Cup xG data |
| `python collect_xag_data.py` | Collect xAG data |

## Prediction Scripts

| Script | Description |
|---|---|
| `python predict_worldcup.py` | Predict World Cup matches |
| `python find_value_bets.py` | Find value betting opportunities |
| `python today_value_bets_live.py` | Today's live value bets |
| `python what_if_brazil_norway.py` | What-if scenario: Brazil × Norway |
| `python what_if_canada_morocco.py` | What-if scenario: Canada × Morocco |
| `python what_if_portugal_spain.py` | What-if scenario: Portugal × Spain |

## Common Options (all scripts)

| Flag | Description |
|---|---|
| `--help, -h` | Show help message and exit |
| `--data-dir DIR` | Custom data directory path (where applicable) |
| `--verbose, -v` | Enable verbose logging |
| `--skip-train` | Skip model training (use existing) |
