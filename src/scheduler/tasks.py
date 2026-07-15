"""
Task implementations — executable functions for each pipeline step.

Each task function accepts a ``ScheduleConfig`` and returns a ``TaskResult``.
They are designed to be composable, testable, and safe to run independently.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.scheduler.models import (
    ScheduleConfig,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  1. DOWNLOAD FIXTURES
# ══════════════════════════════════════════════════════════


def download_fixtures(cfg: ScheduleConfig) -> TaskResult:
    """Download new match data from configured data sources.

    Delegates to the active importer (football-data.co.uk).
    Logs what was downloaded and how many new rows were added.
    """
    result = TaskResult(task_name="download_fixtures", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    try:
        from src.importers import FootballDataImporter

        importer = FootballDataImporter(
            auto_create_teams=True,
            auto_create_competitions=True,
            auto_create_seasons=True,
            incremental=True,
        )

        # Fetch historical + current for all known leagues
        leagues = list(importer.league_map.keys())
        logger.info("Downloading fixtures for %d leagues ...", len(leagues))

        reports = importer.import_historical(leagues, max_seasons=2)

        total_rows = sum(r.rows_imported for r in reports)
        success_count = sum(1 for r in reports if r.success)

        errors = [e for r in reports for e in r.errors if e]
        # Build summary
        summary_lines = []
        for r in reports:
            if r.rows_imported > 0 or r.status == "failed":
                summary_lines.append(
                    f"  {r.league}/{r.season}: {r.status} ({r.rows_imported} rows)"
                )

        result.output = (
            f"Downloaded {total_rows} rows across {len(reports)} "
            f"league-seasons ({success_count} succeeded)"
        )
        if summary_lines:
            result.output += "\n" + "\n".join(summary_lines[-10:])

        result.records_processed = total_rows
        result.status = TaskStatus.SUCCESS if len(errors) == 0 else TaskStatus.WARNING

        if errors:
            result.warnings = errors[:5]
            logger.warning("Download completed with %d warnings", len(errors))

    except Exception as exc:
        logger.exception("Download fixtures failed")
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.duration_seconds = time.perf_counter() - start
    result.completed_at = datetime.now(timezone.utc)
    return result


# ══════════════════════════════════════════════════════════
#  2. UPDATE DATABASE
# ══════════════════════════════════════════════════════════


def update_database(cfg: ScheduleConfig) -> TaskResult:
    """Ingest cleaned data into the database and retrain the ensemble model.

    Steps:
    1. Verify DB connection
    2. Load processed results CSV
    3. Build feature matrix
    4. Store match data via ORM
    5. Retrain ensemble model with fresh data
    6. Save trained model
    """
    result = TaskResult(task_name="update_database", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    try:
        # Step 1: Check DB connection
        from src.database.session import get_session
        from sqlalchemy import text

        with get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("Database connection OK")

        import pandas as pd

        data_path = Path("data/processed/results_clean.csv")
        if not data_path.exists():
            result.output = "No processed data file found — skipping DB update"
            result.status = TaskStatus.SKIPPED
            result.duration_seconds = time.perf_counter() - start
            result.completed_at = datetime.now(timezone.utc)
            return result

        df = pd.read_csv(data_path, low_memory=False)
        logger.info("Processing %d rows for database update ...", len(df))

        # Step 2: Build features
        from src.feature_engineering import build_features, train_val_test_split

        X, y = build_features(df, is_training=True)
        splits = train_val_test_split(X, y)

        # Step 3: Store data via ORM (upsert matches)
        records = df.to_dict(orient="records")
        from src.database.models.match import Match
        from src.etl.store import DatabaseStore

        store = DatabaseStore(
            model_class=Match,
            unique_columns=["id"],
            batch_size=500,
        )
        store_result = store.write(records)

        # Step 4: Retrain ensemble model
        from src.ensemble import EnsembleModel

        df_sorted = df.loc[X.index] if hasattr(X, "index") else df
        n_train = len(splits["X_train"])
        n_val = len(splits["X_val"])
        df_train = df_sorted.iloc[:n_train] if len(df_sorted) >= n_train else df_sorted
        df_val = df_sorted.iloc[n_train:n_train + n_val] if len(df_sorted) >= n_train + n_val else pd.DataFrame()

        ensemble = EnsembleModel()
        fit_report = ensemble.fit(
            splits["X_train"], splits["y_train"],
            splits["X_val"], splits["y_val"],
            df_train=df_train, df_val=df_val,
        )

        model_path = Path("models") / "ensemble_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble.save(str(model_path))

        inserted = getattr(store_result, "records_out", 0) or 0
        weights_str = ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(fit_report.get("weights", {}).items())
        )

        result.records_processed = len(df)
        result.output = (
            f"DB updated: {len(X)} feature rows, "
            f"{len(splits['X_train'])} train / {len(splits['X_val'])} val | "
            f"Model val log-loss={fit_report.get('val_log_loss', '?'):.4f} | "
            f"Weights: {{{weights_str}}}"
        )
        result.status = TaskStatus.SUCCESS

    except Exception as exc:
        logger.exception("Database update failed")
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.duration_seconds = time.perf_counter() - start
    result.completed_at = datetime.now(timezone.utc)
    return result


# ══════════════════════════════════════════════════════════
#  3. VALIDATE DATA
# ══════════════════════════════════════════════════════════


def validate_data(cfg: ScheduleConfig) -> TaskResult:
    """Run validation checks on imported data.

    Uses the ValidationEngine to detect duplicate matches,
    invalid dates, missing goals, impossible scores, etc.
    Generates HTML/CSV/JSON reports.
    """
    result = TaskResult(task_name="validate_data", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    try:
        from src.validation import ValidationEngine

        # Load the most recent import data
        import pandas as pd

        data_path = Path("data/processed/results_clean.csv")
        if not data_path.exists():
            result.output = "No data file to validate"
            result.status = TaskStatus.SKIPPED
            result.duration_seconds = time.perf_counter() - start
            result.completed_at = datetime.now(timezone.utc)
            return result

        df = pd.read_csv(data_path, low_memory=False)
        data = df.to_dict(orient="records")

        engine = ValidationEngine()
        validation_result = engine.run(data, source_name="scheduler_validate")

        # Save reports
        report_dir = Path(cfg.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        validation_result.to_html(str(report_dir / f"validation_{timestamp}.html"))
        validation_result.to_csv(str(report_dir / f"validation_{timestamp}.csv"))
        validation_result.to_json(str(report_dir / f"validation_{timestamp}.json"))

        result.records_processed = len(data)
        result.output = (
            f"Validation: {validation_result.passed_checks}/{validation_result.total_checks} "
            f"checks passed, {validation_result.total_violations} violations"
        )

        if validation_result.total_violations > 0:
            result.warnings = [
                f"{v['check']}: {v.get('message', '')[:100]}"
                for v in validation_result.get_violations()[:5]
            ]

        result.status = (
            TaskStatus.WARNING if validation_result.total_violations > 0
            else TaskStatus.SUCCESS
        )

    except Exception as exc:
        logger.exception("Data validation failed")
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.duration_seconds = time.perf_counter() - start
    result.completed_at = datetime.now(timezone.utc)
    return result


# ══════════════════════════════════════════════════════════
#  4. CLEAN DATA
# ══════════════════════════════════════════════════════════


def clean_data(cfg: ScheduleConfig) -> TaskResult:
    """Clean and archive data artifacts.

    Actions:
    1. Deduplicate the processed results CSV
    2. Archive old raw CSV files (older than 7 days)
    3. Remove stale checkpoint files
    4. Remove empty or corrupted cache files
    """
    result = TaskResult(task_name="clean_data", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    actions: list[str] = []

    try:
        # 1. Deduplicate processed CSV
        import pandas as pd

        data_path = Path("data/processed/results_clean.csv")
        if data_path.exists():
            df = pd.read_csv(data_path, low_memory=False)
            before = len(df)
            df = df.drop_duplicates(subset=["date", "home_team", "away_team", "league"],
                                    keep="last")
            after = len(df)
            if after < before:
                df.to_csv(data_path, index=False)
                actions.append(f"Deduplicated results_clean.csv: {before} -> {after} rows")
                logger.info("Cleaned %d duplicate rows from results", before - after)
            else:
                actions.append("No duplicates found in results_clean.csv")

        # 2. Archive old raw files
        raw_dirs = [
            Path("data/raw/football-data"),
            Path("data/raw/archive"),
        ]
        archive_dir = Path("data/raw/archive")
        archive_dir.mkdir(parents=True, exist_ok=True)

        cutoff = datetime.now() - timedelta(days=7)
        archived_count = 0
        for raw_dir in raw_dirs:
            if raw_dir.exists():
                for f in raw_dir.glob("*.csv"):
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        dest = archive_dir / f.name
                        shutil.move(str(f), str(dest))
                        archived_count += 1
        if archived_count:
            actions.append(f"Archived {archived_count} old raw CSV files")

        # 3. Remove stale checkpoints
        checkpoint_dirs = [
            Path("data/scrapers"),
            Path("data/cache"),
        ]
        removed_checkpoints = 0
        for cd in checkpoint_dirs:
            if cd.exists():
                for f in cd.rglob("*.checkpoint"):
                    f.unlink()
                    removed_checkpoints += 1
        if removed_checkpoints:
            actions.append(f"Removed {removed_checkpoints} stale checkpoint files")

        # 4. Remove empty cache files
        cache_dir = Path("data/cache")
        empty_caches = 0
        if cache_dir.exists():
            for f in cache_dir.rglob("*.cache"):
                if f.stat().st_size == 0:
                    f.unlink()
                    empty_caches += 1
        if empty_caches:
            actions.append(f"Removed {empty_caches} empty cache files")

        result.records_processed = archived_count + removed_checkpoints
        result.output = "\n".join(actions) if actions else "No cleanup needed"
        result.status = TaskStatus.SUCCESS

    except Exception as exc:
        logger.exception("Data cleanup failed")
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.duration_seconds = time.perf_counter() - start
    result.completed_at = datetime.now(timezone.utc)
    return result


# ══════════════════════════════════════════════════════════
#  5. BACKUP DATABASE
# ══════════════════════════════════════════════════════════


def backup_database(cfg: ScheduleConfig) -> TaskResult:
    """Create a database backup with configurable retention.

    For PostgreSQL: uses pg_dump.
    For SQLite: copies the database file.
    Retains only the N most recent backups (configurable).
    """
    result = TaskResult(task_name="backup_database", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    backup_dir = Path(cfg.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.database.session import get_engine

        engine = get_engine()
        url = str(engine.url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"football_db_{timestamp}"

        if url.startswith("sqlite"):
            # SQLite: simple file copy
            # Extract the path from sqlite:///path/to/db
            db_path = url.replace("sqlite:///", "")
            if db_path:
                src = Path(db_path)
                if src.exists():
                    dest = backup_path.with_suffix(".sqlite")
                    shutil.copy2(str(src), str(dest))
                    file_size = dest.stat().st_size
                    result.output = (
                        f"SQLite backup created: {dest.name} ({file_size / 1024 / 1024:.1f} MB)"
                    )
                    result.records_processed = 1
                else:
                    result.output = "SQLite database file not found"
                    result.status = TaskStatus.SKIPPED
                    result.duration_seconds = time.perf_counter() - start
                    result.completed_at = datetime.now(timezone.utc)
                    return result
            else:
                result.output = "Unknown SQLite path — skipping backup"
                result.status = TaskStatus.SKIPPED
                result.duration_seconds = time.perf_counter() - start
                result.completed_at = datetime.now(timezone.utc)
                return result

        elif "postgresql" in url or "postgres" in url:
            # PostgreSQL: use pg_dump
            backup_path = backup_path.with_suffix(".sql.gz")
            # Extract connection params from engine URL
            try:
                cmd = [
                    "pg_dump",
                    "--no-owner",
                    "--no-acl",
                    "--compress=9",
                    "-f", str(backup_path),
                    url,
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                file_size = backup_path.stat().st_size
                result.output = (
                    f"PostgreSQL backup created: {backup_path.name} "
                    f"({file_size / 1024 / 1024:.1f} MB)"
                )
                result.records_processed = 1
            except FileNotFoundError:
                result.output = "pg_dump not found — install PostgreSQL client tools"
                result.status = TaskStatus.FAILED
                result.duration_seconds = time.perf_counter() - start
                result.completed_at = datetime.now(timezone.utc)
                return result
        else:
            result.output = f"Unsupported database type: {url.split(':')[0]}"
            result.status = TaskStatus.SKIPPED
            result.duration_seconds = time.perf_counter() - start
            result.completed_at = datetime.now(timezone.utc)
            return result

        # Enforce retention policy
        retention = cfg.backup_retention_days
        cutoff = datetime.now() - timedelta(days=retention)
        removed = 0
        for f in sorted(backup_dir.iterdir()):
            if f.name.startswith("football_db_"):
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    removed += 1
        if removed:
            result.output += f" | Removed {removed} old backups (retention: {retention}d)"

        result.status = TaskStatus.SUCCESS

    except Exception as exc:
        logger.exception("Database backup failed")
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.duration_seconds = time.perf_counter() - start
    result.completed_at = datetime.now(timezone.utc)
    return result



# ══════════════════════════════════════════════════════════
#  7. DAILY DATA PIPELINE
# ══════════════════════════════════════════════════════════


def daily_data_pipeline(cfg: ScheduleConfig) -> TaskResult:
    """Fetch new match data from all configured sources, clean and merge.

    Delegates to ``scripts/daily_data_pipeline.py``.
    """
    result = TaskResult(task_name="daily_data_pipeline", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    try:
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "daily_data_pipeline.py"
        if not script_path.exists():
            result.output = "Script not found: scripts/daily_data_pipeline.py"
            result.status = TaskStatus.SKIPPED
            result.duration_seconds = time.perf_counter() - start
            result.completed_at = datetime.now(timezone.utc)
            return result

        proc = subprocess.run(
            [sys.executable, str(script_path), "--quiet"],
            capture_output=True, text=True, check=False, timeout=600,
        )

        result.output = proc.stdout.strip()[:500] or "Completed"
        result.duration_seconds = time.perf_counter() - start

        if proc.returncode == 0:
            result.status = TaskStatus.SUCCESS
        else:
            result.status = TaskStatus.FAILED
            result.error = proc.stderr.strip()[:500] or f"Exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result.status = TaskStatus.FAILED
        result.error = "Timed out after 600s"
    except Exception as exc:
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.completed_at = datetime.now(timezone.utc)
    return result


# ══════════════════════════════════════════════════════════
#  8. DAILY FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════


def daily_feature_computation(cfg: ScheduleConfig) -> TaskResult:
    """Load new matches and compute all features via Feature Store.

    Delegates to ``scripts/daily_feature_computation.py``.
    """
    result = TaskResult(task_name="daily_feature_computation", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    try:
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "daily_feature_computation.py"
        if not script_path.exists():
            result.output = "Script not found: scripts/daily_feature_computation.py"
            result.status = TaskStatus.SKIPPED
            result.duration_seconds = time.perf_counter() - start
            result.completed_at = datetime.now(timezone.utc)
            return result

        proc = subprocess.run(
            [sys.executable, str(script_path), "--quiet"],
            capture_output=True, text=True, check=False, timeout=600,
        )

        result.output = proc.stdout.strip()[:500] or "Completed"
        result.duration_seconds = time.perf_counter() - start

        if proc.returncode == 0:
            result.status = TaskStatus.SUCCESS
        else:
            result.status = TaskStatus.FAILED
            result.error = proc.stderr.strip()[:500] or f"Exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result.status = TaskStatus.FAILED
        result.error = "Timed out after 600s"
    except Exception as exc:
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.completed_at = datetime.now(timezone.utc)
    return result


# ══════════════════════════════════════════════════════════
#  9. DAILY MODEL RETRAINING
# ══════════════════════════════════════════════════════════


def daily_model_retraining(cfg: ScheduleConfig) -> TaskResult:
    """Check for new data, retrain models, validate, and save the best.

    Delegates to ``scripts/daily_model_retraining.py``.
    """
    result = TaskResult(task_name="daily_model_retraining", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    try:
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "daily_model_retraining.py"
        if not script_path.exists():
            result.output = "Script not found: scripts/daily_model_retraining.py"
            result.status = TaskStatus.SKIPPED
            result.duration_seconds = time.perf_counter() - start
            result.completed_at = datetime.now(timezone.utc)
            return result

        proc = subprocess.run(
            [sys.executable, str(script_path), "--quiet"],
            capture_output=True, text=True, check=False, timeout=1200,
        )

        result.output = proc.stdout.strip()[:500] or "Completed"
        result.duration_seconds = time.perf_counter() - start

        if proc.returncode == 0:
            result.status = TaskStatus.SUCCESS
        else:
            result.status = TaskStatus.FAILED
            result.error = proc.stderr.strip()[:500] or f"Exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result.status = TaskStatus.FAILED
        result.error = "Timed out after 1200s"
    except Exception as exc:
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.completed_at = datetime.now(timezone.utc)
    return result


# ══════════════════════════════════════════════════════════
#  10. DAILY PREDICTIONS
# ══════════════════════════════════════════════════════════


def daily_predictions(cfg: ScheduleConfig) -> TaskResult:
    """Load upcoming fixtures and generate predictions using best model.

    Delegates to ``scripts/daily_predictions.py``.
    """
    result = TaskResult(task_name="daily_predictions", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    try:
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "daily_predictions.py"
        if not script_path.exists():
            result.output = "Script not found: scripts/daily_predictions.py"
            result.status = TaskStatus.SKIPPED
            result.duration_seconds = time.perf_counter() - start
            result.completed_at = datetime.now(timezone.utc)
            return result

        proc = subprocess.run(
            [sys.executable, str(script_path), "--quiet"],
            capture_output=True, text=True, check=False, timeout=300,
        )

        result.output = proc.stdout.strip()[:500] or "Completed"
        result.duration_seconds = time.perf_counter() - start

        if proc.returncode == 0:
            result.status = TaskStatus.SUCCESS
        else:
            result.status = TaskStatus.FAILED
            result.error = proc.stderr.strip()[:500] or f"Exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result.status = TaskStatus.FAILED
        result.error = "Timed out after 300s"
    except Exception as exc:
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.completed_at = datetime.now(timezone.utc)
    return result


# ══════════════════════════════════════════════════════════
#  6. GENERATE LOGS
# ══════════════════════════════════════════════════════════


def generate_logs(cfg: ScheduleConfig) -> TaskResult:
    """Rotate logs, archive old reports, and write a run summary.

    Actions:
    1. Rotate log files (compress logs older than max_log_age_days)
    2. Archive old reports (move to archive subdirectory)
    3. Write a structured JSON summary of the run
    """
    result = TaskResult(task_name="generate_logs", status=TaskStatus.RUNNING)
    result.started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    actions: list[str] = []

    try:
        # 1. Rotate old log files
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        max_age = cfg.max_log_age_days
        cutoff = datetime.now() - timedelta(days=max_age)
        rotated = 0

        for f in log_dir.glob("*.log"):
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                # Compress and rename
                archive_name = f.with_suffix(f"{f.suffix}.{mtime.strftime('%Y%m%d')}.old")
                f.rename(archive_name)
                rotated += 1

        if rotated:
            actions.append(f"Rotated {rotated} old log files (> {max_age}d)")

        # 2. Archive old reports
        report_dir = Path(cfg.report_dir)
        archive_subdir = report_dir / "archive"
        archived_reports = 0

        if report_dir.exists():
            archive_subdir.mkdir(parents=True, exist_ok=True)
            for f in report_dir.glob("*.html"):
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    dest = archive_subdir / f.name
                    shutil.move(str(f), str(dest))
                    archived_reports += 1

        if archived_reports:
            actions.append(f"Archived {archived_reports} old reports")

        # 3. Write run summary
        run_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline": cfg.pipeline_name,
            "actions": actions,
            "log_dir": str(log_dir),
            "report_dir": str(report_dir),
            "backup_dir": str(cfg.backup_dir),
        }
        summary_path = log_dir / f"run_summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(summary_path, "w") as f:
            json.dump(run_summary, f, indent=2)
        actions.append(f"Run summary written to {summary_path.name}")

        result.output = "\n".join(actions) if actions else "No log maintenance needed"
        result.records_processed = rotated + archived_reports
        result.status = TaskStatus.SUCCESS

    except Exception as exc:
        logger.exception("Log generation failed")
        result.status = TaskStatus.FAILED
        result.error = str(exc)

    result.duration_seconds = time.perf_counter() - start
    result.completed_at = datetime.now(timezone.utc)
    return result
