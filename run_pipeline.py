"""
Automated Prediction Pipeline — run daily to download, train, predict, and report.

Usage
-----
::

    python run_pipeline.py                    # Full daily run
    python run_pipeline.py --skip-download    # Skip data download (use existing)
    python run_pipeline.py --skip-train       # Skip retraining (use existing model)
    python run_pipeline.py --lightweight      # Skip download + train (predict only)

Schedule (cron)
---------------
Add to crontab to run daily at 8 AM::

    0 8 * * * cd /path/to/project && python run_pipeline.py >> logs/pipeline.log 2>&1

Or using Windows Task Scheduler / schtasks::

    schtasks /create /tn "FootballPredictor" /tr "python run_pipeline.py" /sc daily /st 08:00

Pipeline steps
--------------
1. **Download** — fetch latest match results and upcoming fixtures
2. **Preprocess** — clean, normalise, and validate the updated dataset
3. **Check retrain** — compare model file age vs data update time; retrain if stale
4. **Build features** — generate leakage-free feature matrix from updated data
5. **Predict** — run the model on the most recent / upcoming matches
6. **Save** — write predictions to CSV in ``reports/predictions/``
7. **Report** — print / save a summary of the run

Failure handling
----------------
- Every step is wrapped in a try/except with detailed error logging.
- If any step fails, the pipeline continues to the next step (unless it's a
  hard dependency — e.g. prediction requires features).
- A final status table shows which steps succeeded or failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import config

# ── Ensure logs directory exists ────────────────────────
_log_dir = Path("logs")
_log_dir.mkdir(parents=True, exist_ok=True)

# ── Logging configuration ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_log_dir / "pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════
#  Pipeline configuration
# ═══════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """Settings for the automated prediction pipeline.

    Attributes
    ----------
    retrain_if_stale_days : int
        Retrain the model if the saved model file is older than this many
        days (default 7 — retrain weekly).
    force_retrain_every_n_runs : int
        Force a full retrain every N pipeline runs regardless of staleness
        (default 10).  0 = never force.
    predictions_dir : str
        Directory to save prediction CSVs (relative to project root).
    keep_last_n_predictions : int
        Number of old prediction CSVs to keep (default 30).
    report_dir : str
        Directory to save pipeline reports (relative to project root).
    model_file : str
        File name of the trained model in ``models/`` (default ``xgboost_model.joblib``).
    """
    retrain_if_stale_days: int = 7
    force_retrain_every_n_runs: int = 10
    predictions_dir: str = "reports/predictions"
    keep_last_n_predictions: int = 30
    report_dir: str = "reports"
    model_file: str = "xgboost_model.joblib"


# Default config instance
_pipeline_cfg = PipelineConfig()

# Runtime state
_start_time: float = 0.0
_run_counter_path = Path("logs/.pipeline_run_count")


# ═══════════════════════════════════════════════════════════
#  Step 1 — Download data
# ═══════════════════════════════════════════════════════════


def step_download() -> dict[str, Any]:
    """Download latest match results and upcoming fixtures.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``new_rows``, ``source``, ``error``.
    """
    logger.info("─" * 60)
    logger.info("STEP 1: Download data")
    logger.info("─" * 60)

    try:
        from src.data_collection import update

        result = update()
        success = result.get("new_rows", 0) > 0 or True  # Not a failure if 0 new rows
        logger.info(
            "Download complete — %d new rows (total: %d)",
            result.get("new_rows", 0),
            result.get("total_rows", 0),
        )
        return {
            "success": True,
            "new_rows": result.get("new_rows", 0),
            "total_rows": result.get("total_rows", 0),
            "source": result.get("path", "unknown"),
        }
    except Exception as exc:
        logger.error("Download failed: %s", exc, exc_info=True)
        return {"success": False, "new_rows": 0, "error": str(exc)}


# ═══════════════════════════════════════════════════════════
#  Step 2 — Preprocess data
# ═══════════════════════════════════════════════════════════


def step_preprocess() -> dict[str, Any]:
    """Run the preprocessing pipeline on the raw data.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``rows``, ``columns``, ``error``.
    """
    logger.info("─" * 60)
    logger.info("STEP 2: Preprocess data")
    logger.info("─" * 60)

    try:
        from src.preprocessing import run_preprocessing

        report = run_preprocessing(save=True)
        success = report.get("total_rows", 0) > 0
        logger.info(
            "Preprocessing complete — %d rows, %d columns",
            report.get("total_rows", 0),
            report.get("total_columns", 0),
        )
        return {
            "success": success,
            "rows": report.get("total_rows", 0),
            "columns": report.get("total_columns", 0),
            "output_path": report.get("saved_to", ""),
        }
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc, exc_info=True)
        return {"success": False, "rows": 0, "error": str(exc)}


# ═══════════════════════════════════════════════════════════
#  Step 3 — Check if retrain is needed
# ═══════════════════════════════════════════════════════════


def _should_retrain() -> bool:
    """Determine if the model should be retrained.

    Checks:
    1. Model file exists? → no model → must train.
    2. Model file stale? → older than ``retrain_if_stale_days``.
    3. Force retrain? → every ``force_retrain_every_n_runs`` runs.
    """
    model_path = config.paths.models / _pipeline_cfg.model_file

    # No model exists — must train
    if not model_path.exists():
        logger.info("  Model file not found — retrain required")
        return True

    # Check staleness
    model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - model_mtime).days
    if age_days >= _pipeline_cfg.retrain_if_stale_days:
        logger.info(
            "  Model is %d days old (threshold: %d) — retrain required",
            age_days,
            _pipeline_cfg.retrain_if_stale_days,
        )
        return True

    # Force retrain counter
    if _pipeline_cfg.force_retrain_every_n_runs > 0:
        count = _get_run_count()
        if count > 0 and count % _pipeline_cfg.force_retrain_every_n_runs == 0:
            logger.info("  Force retrain trigger (run #%d)", count)
            return True

    logger.info("  Model is current (%d days old) — skipping retrain", age_days)
    return False


def _get_run_count() -> int:
    """Read the persistent run counter from ``logs/.pipeline_run_count``."""
    try:
        if _run_counter_path.exists():
            return int(_run_counter_path.read_text().strip())
    except (ValueError, OSError):
        pass
    return 0


def _increment_run_count() -> int:
    """Increment and save the run counter."""
    count = _get_run_count() + 1
    _run_counter_path.parent.mkdir(parents=True, exist_ok=True)
    _run_counter_path.write_text(str(count))
    return count


def step_retrain() -> dict[str, Any]:
    """Retrain the model if needed.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``retrained``, ``model_path``, ``error``.
    """
    logger.info("─" * 60)
    logger.info("STEP 3: Check & retrain model")
    logger.info("─" * 60)

    if not _should_retrain():
        return {"success": True, "retrained": False, "model_path": str(config.paths.models / _pipeline_cfg.model_file)}

    try:
        from src.feature_engineering import build_features, train_val_test_split
        from src.train import train_model, save_model

        # Load preprocessed data
        data_path = config.paths.processed / "results_clean.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {data_path}")

        logger.info("  Loading preprocessed data ...")
        df = pd.read_csv(data_path, low_memory=False)

        logger.info("  Building features ...")
        X, y = build_features(df, is_training=True)

        logger.info("  Splitting chronologically ...")
        splits = train_val_test_split(X, y)

        logger.info("  Training model (type=%s) ...", config.train.model_type)
        model, history = train_model(
            splits["X_train"], splits["y_train"],
            splits["X_val"], splits["y_val"],
        )

        model_path = save_model(model, _pipeline_cfg.model_file)
        logger.info("  Model saved to %s", model_path)

        return {
            "success": True,
            "retrained": True,
            "model_path": model_path,
            "train_loss": history.get("train_loss", [None])[0],
            "val_loss": history.get("val_loss", [None])[0],
            "val_accuracy": history.get("val_accuracy", [None])[0],
        }
    except Exception as exc:
        logger.error("Retrain failed: %s", exc, exc_info=True)
        return {"success": False, "retrained": False, "error": str(exc)}


# ═══════════════════════════════════════════════════════════
#  Step 4 — Predict upcoming matches
# ═══════════════════════════════════════════════════════════


def step_predict() -> dict[str, Any]:
    """Generate predictions for upcoming / most recent matches.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``n_predictions``, ``output_path``, ``error``.
    """
    logger.info("─" * 60)
    logger.info("STEP 4: Predict upcoming matches")
    logger.info("─" * 60)

    try:
        from src.train import load_model
        from src.feature_engineering import build_features

        # Load the trained model
        model = load_model(_pipeline_cfg.model_file)

        # Load preprocessed data for feature building
        data_path = config.paths.processed / "results_clean.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {data_path}")

        df = pd.read_csv(data_path, low_memory=False)

        # Build features (is_training=False means no target column separation
        # needed, but we need the full feature pipeline for the last rows)
        logger.info("  Building feature matrix for prediction ...")
        X, _ = build_features(df, is_training=True)

        # Predict on the last N rows (most recent matches)
        # In a real scenario, we'd use upcoming fixtures. Here we use the
        # most recent matches from the dataset as a proxy.
        n_recent = min(50, len(X) // 2)  # Predict on last 50 or 50% of data
        X_recent = X.iloc[-n_recent:]

        logger.info("  Generating predictions for %d matches ...", n_recent)
        probs = model.predict_proba(X_recent)
        preds = model.predict(X_recent)

        # Build output DataFrame
        # We need timestamps / team names from the original data.
        # Align by index: the last n_recent rows of X correspond to the
        # last n_recent rows of the original DataFrame.
        output_df = df.iloc[-n_recent:][
            [c for c in ["date", "home_team", "away_team", "result", "league"]
             if c in df.columns]
        ].copy()
        output_df["prediction"] = preds
        output_df["home_win_prob"] = probs[:, 2]
        output_df["draw_prob"] = probs[:, 1]
        output_df["away_win_prob"] = probs[:, 0]
        output_df["max_prob"] = probs.max(axis=1)

        # Map prediction to label
        label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        output_df["prediction_label"] = output_df["prediction"].map(label_map)

        # Save to timestamped CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_dir = Path(_pipeline_cfg.predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        out_path = predictions_dir / f"predictions_{timestamp}.csv"
        output_df.to_csv(out_path, index=False)

        # Clean up old predictions
        _cleanup_old_files(predictions_dir, _pipeline_cfg.keep_last_n_predictions, "predictions_*.csv")

        logger.info("  Predictions saved: %s (%d matches)", out_path, len(output_df))
        return {
            "success": True,
            "n_predictions": len(output_df),
            "output_path": str(out_path),
        }
    except Exception as exc:
        logger.error("Prediction step failed: %s", exc, exc_info=True)
        return {"success": False, "n_predictions": 0, "error": str(exc)}


# ═══════════════════════════════════════════════════════════
#  Step 5 — Generate summary report
# ═══════════════════════════════════════════════════════════


def step_report(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generate and save a summary report of the pipeline run.

    Parameters
    ----------
    results : dict[str, dict[str, Any]]
        Results from each pipeline step.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``report_path``, ``error``.
    """
    logger.info("─" * 60)
    logger.info("STEP 5: Generate summary report")
    logger.info("─" * 60)

    elapsed = time.time() - _start_time

    try:
        # ── Build report text ────────────────────────────
        lines: list[str] = []
        sep = "=" * 70

        lines.append("")
        lines.append(sep)
        lines.append("  PIPELINE RUN REPORT".center(68))
        lines.append(sep)
        lines.append(f"")
        lines.append(f"  Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Duration:      {elapsed:.1f} seconds")
        lines.append(f"  Run #:         {_get_run_count()}")
        lines.append("")

        # ── Step status table ────────────────────────────
        lines.append(f"  {'─' * 50}")
        lines.append(f"  {'Step':<30s} {'Status':<15s} {'Details':>40s}")
        lines.append(f"  {'─' * 50}")

        for step_name, result in results.items():
            status = "✅ PASS" if result.get("success") else "❌ FAIL"
            detail = result.get("error", "OK")
            # Truncate long errors but keep the meaningful part
            detail_str = str(detail)[:40] if len(str(detail)) > 40 else str(detail)
            lines.append(f"  {step_name:<30s} {status:<15s} {detail_str:>40s}")
        lines.append(f"  {'─' * 50}")
        lines.append("")

        # ── Detailed step info ───────────────────────────
        dl = results.get("download", {})
        if dl.get("success"):
            lines.append(f"  Download:  {dl.get('new_rows', 0)} new rows "
                         f"(total: {dl.get('total_rows', 0)})")

        pp = results.get("preprocess", {})
        if pp.get("success"):
            lines.append(f"  Preprocess: {pp.get('rows', 0)} rows, {pp.get('columns', 0)} cols")

        tr = results.get("retrain", {})
        if tr.get("retrained"):
            lines.append(f"  Retrain:   YES — model saved to {tr.get('model_path', '?')}")
            if tr.get("val_loss"):
                lines.append(f"             Val log-loss: {tr['val_loss']:.4f}")
            if tr.get("val_accuracy"):
                lines.append(f"             Val accuracy: {tr['val_accuracy']:.2%}")
        else:
            lines.append(f"  Retrain:   NO (model is current)")

        pr = results.get("predict", {})
        if pr.get("success"):
            lines.append(f"  Predict:   {pr.get('n_predictions', 0)} matches → "
                         f"{pr.get('output_path', '?')}")

        lines.append("")
        lines.append(sep)
        lines.append("")

        report_text = "\n".join(lines)

        # ── Print to console ─────────────────────────────
        print(report_text)

        # ── Save to file ─────────────────────────────────
        report_dir = Path(_pipeline_cfg.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"pipeline_report_{timestamp}.txt"
        report_path.write_text(report_text, encoding="utf-8")

        logger.info("Report saved to %s", report_path)

        # Clean up old reports
        _cleanup_old_files(report_dir, 30, "pipeline_report_*.txt")

        return {"success": True, "report_path": str(report_path)}
    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════


def _cleanup_old_files(directory: Path, keep_n: int, pattern: str) -> None:
    """Remove all but the *keep_n* most recent files matching *pattern*."""
    if keep_n <= 0:
        return
    try:
        files = sorted(directory.glob(pattern))
        if len(files) > keep_n:
            for f in files[:-keep_n]:
                f.unlink(missing_ok=True)
                logger.debug("Cleaned up old file: %s", f)
    except OSError as exc:
        logger.warning("File cleanup failed: %s", exc)


def _all_steps_succeeded(results: dict[str, dict[str, Any]]) -> bool:
    """Return True if every step in the results dict succeeded."""
    return all(r.get("success", False) for r in results.values())


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Football Prediction — automated pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the data download step",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model retraining (use the existing saved model)",
    )
    parser.add_argument(
        "--lightweight", action="store_true",
        help="Skip download + train (predict only using existing data and model)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full prediction pipeline.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments.  Uses ``sys.argv`` if not provided.

    Returns
    -------
    int
        0 if all steps succeeded, 1 if any step failed.
    """
    global _start_time
    _start_time = time.time()
    args = parse_args(argv)

    # Increment run counter
    run_count = _increment_run_count()

    print()
    print("=" * 70)
    print(f"  FOOTBALL PREDICTION PIPELINE — Run #{run_count}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    results: dict[str, dict[str, Any]] = {}

    # ── Step 1: Download ─────────────────────────────────
    if args.lightweight or args.skip_download:
        logger.info("Skipping download (--skip-download or --lightweight)")
        results["download"] = {"success": True, "skipped": True}
    else:
        results["download"] = step_download()

    # ── Step 2: Preprocess ───────────────────────────────
    if args.lightweight:
        logger.info("Skipping preprocess (--lightweight)")
        results["preprocess"] = {"success": True, "skipped": True}
    else:
        results["preprocess"] = step_preprocess()

    # ── Step 3: Retrain ──────────────────────────────────
    if args.lightweight or args.skip_train:
        logger.info("Skipping retrain (--skip-train or --lightweight)")
        results["retrain"] = {"success": True, "retrained": False, "skipped": True}
    else:
        results["retrain"] = step_retrain()

    # ── Step 4: Predict ──────────────────────────────────
    results["predict"] = step_predict()

    # ── Step 5: Report ───────────────────────────────────
    results["report"] = step_report(results)

    # ── Final status ─────────────────────────────────────
    ok = _all_steps_succeeded(results)
    if ok:
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE — all steps passed (%.1f s)", time.time() - _start_time)
        logger.info("=" * 70)
    else:
        logger.warning("=" * 70)
        logger.warning(
            "PIPELINE COMPLETE — some steps FAILED (%.1f s)",
            time.time() - _start_time,
        )
        logger.warning("=" * 70)
        for step_name, result in results.items():
            if not result.get("success"):
                logger.warning("  ✗ %s: %s", step_name, result.get("error", "Unknown error"))

    # ── Print summary to stdout ────────────────────────
    status_icon = "✅" if ok else "⚠️"
    print(f"\n  {status_icon} Pipeline finished in {time.time() - _start_time:.1f}s")
    print(f"  Log: logs/pipeline.log")
    print()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
