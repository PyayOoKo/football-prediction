"""
Automated Prediction Pipeline — run daily to download, train, predict, and report.

Uses the **3-model blend** (Poisson + Elo + XGBoost) for predictions by default,
which also predicts Over/Under and BTTS markets alongside 1X2 match outcomes.
Optionally falls back to the ensemble model (XGBoost + Logistic Regression +
Poisson) with ``--ensemble``.

Usage
-----
::

    python run_pipeline.py                    # Full daily run (blend default)
    python run_pipeline.py --ensemble         # Use ensemble model instead of blend
    python run_pipeline.py --skip-blend       # Skip blend training (ensemble only)
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
4. **Train ensemble** — train XGBoost + Logistic Regression + Poisson, optimise weights on val set
5. **Build 3-model blend** — combine Poisson + Elo + XGBoost for 1X2, O/U, BTTS markets
6. **Predict** — run the blend (or ensemble) on the most recent / upcoming matches
7. **Save** — write predictions to CSV in ``reports/predictions/``
8. **Report** — print / save a summary of the run

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
        File name of the trained ensemble model in ``models/``
        (default ``ensemble_model.joblib``).
    blend_file : str
        File name of the 3-model blend in ``models/``
        (default ``three_model_blend.joblib``).
    """
    retrain_if_stale_days: int = 7
    force_retrain_every_n_runs: int = 10
    predictions_dir: str = "reports/predictions"
    keep_last_n_predictions: int = 30
    report_dir: str = "reports"
    model_file: str = "ensemble_model.joblib"
    blend_file: str = "three_model_blend.joblib"


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
    logger.info("-" * 60)
    logger.info("STEP 1: Download data")
    logger.info("-" * 60)

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
    logger.info("-" * 60)
    logger.info("STEP 2: Preprocess data")
    logger.info("-" * 60)

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
    """Retrain the ensemble model if needed.        Trains the full ensemble (XGBoost + Logistic Regression + Poisson),
        optimises weights on the validation set, and saves the trained
        ensemble to disk.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``retrained``, ``model_path``,
        ``weights``, ``val_loss``, ``error``.
    """
    logger.info("-" * 60)
    logger.info("STEP 3: Retrain model")
    logger.info("-" * 60)

    if not _should_retrain():
        return {"success": True, "retrained": False, "model_path": str(config.paths.models / _pipeline_cfg.model_file)}

    try:
        from src.feature_engineering import build_features, train_val_test_split
        from src.ensemble import EnsembleModel

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

        # Align raw df for Poisson model (needs home_team, away_team, goals)
        df_sorted = df.loc[X.index] if hasattr(X, "index") else df
        n_train = len(splits["X_train"])
        n_val = len(splits["X_val"])
        df_train = df_sorted.iloc[:n_train] if len(df_sorted) >= n_train else df_sorted
        df_val = df_sorted.iloc[n_train:n_train + n_val] if len(df_sorted) >= n_train + n_val else pd.DataFrame()

        logger.info("  Training ensemble (XGBoost + Logistic Regression + Poisson) ...")
        ensemble = EnsembleModel()
        fit_report = ensemble.fit(
            splits["X_train"], splits["y_train"],
            splits["X_val"], splits["y_val"],
            df_train=df_train, df_val=df_val,
        )

        # Save ensemble using its built-in save method
        model_path = config.paths.models / _pipeline_cfg.model_file
        ensemble.save(str(model_path))
        logger.info("  Ensemble saved to %s", model_path)

        weights_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(fit_report["weights"].items()))
        logger.info("  Ensemble val log-loss: %.4f | Weights: %s", fit_report["val_log_loss"], weights_str)

        return {
            "success": True,
            "retrained": True,
            "model_path": str(model_path),
            "val_loss": fit_report.get("val_log_loss"),
            "weights": fit_report.get("weights", {}),
            "individual_losses": fit_report.get("individual_log_losses", {}),
        }
    except Exception as exc:
        logger.error("Retrain failed: %s", exc, exc_info=True)
        return {"success": False, "retrained": False, "error": str(exc)}


# ═══════════════════════════════════════════════════════════
#  Step 3b — Retrain 3-model blend
# ═══════════════════════════════════════════════════════════


def step_retrain_blend() -> dict[str, Any]:
    """Train and save the 3-model blend (Poisson + Elo + XGBoost).

    Loads preprocessed data, fits Poisson and Elo from scratch, extracts
    the XGBoost model from the saved ensemble (``ensemble_model.joblib``),
    builds the ``ThreeModelBlend``, and persists it.

    Falls back to standalone ``xgboost_model.joblib`` / ``worldcup_xgboost.joblib``
    if the ensemble model is not available.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``retrained``, ``blend_path``, ``error``.
    """
    logger.info("-" * 60)
    logger.info("STEP 3b: Retrain 3-model blend")
    logger.info("-" * 60)

    blend_path = config.paths.models / "three_model_blend.joblib"

    try:
        import joblib

        # Load preprocessed data
        data_path = config.paths.processed / "results_clean.csv"
        if not data_path.exists():
            data_path = Path("data/raw/worldcup_all.csv")
            if not data_path.exists():
                raise FileNotFoundError("No data found for blend training")

        logger.info("  Loading data from %s ...", data_path.name)
        df = pd.read_csv(data_path, low_memory=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        logger.info("  %d rows loaded", len(df))

        # Fit Poisson model
        logger.info("  Fitting Poisson model ...")
        from src.poisson_model import PoissonModel
        poisson = PoissonModel()
        poisson.fit(df)

        # Fit Elo system
        logger.info("  Fitting Elo system ...")
        from src.elo import EloSystem
        elo = EloSystem()
        elo.process_matches(df)

        # ── Load XGBoost model ────────────────────────────────────
        # Priority 1: Extract from the saved EnsembleModel (just trained by step_retrain)
        xgb = None
        ensemble_path = config.paths.models / _pipeline_cfg.model_file
        if ensemble_path.exists():
            try:
                ensemble_payload = joblib.load(ensemble_path)
                # EnsembleModel.save() stores models dict with "xgboost" key
                models_dict = ensemble_payload.get("models", {})
                if "xgboost" in models_dict:
                    xgb = models_dict["xgboost"]
                    logger.info("  XGBoost extracted from %s (key='xgboost')", ensemble_path.name)
            except Exception as exc:
                logger.warning("  Failed to extract XGBoost from ensemble: %s", exc)

        # Priority 2: Standalone XGBoost model file
        if xgb is None:
            for candidate in ["xgboost_model.joblib", "worldcup_xgboost.joblib", "xgboost_model"]:
                p = config.paths.models / candidate
                if p.exists():
                    xgb = joblib.load(p)
                    logger.info("  XGBoost model loaded: %s", candidate)
                    break

        if xgb is None:
            raise FileNotFoundError(
                "No XGBoost model found for blend training. "
                f"Searched ensemble model ({ensemble_path}) and standalone files "
                "(xgboost_model.joblib, worldcup_xgboost.joblib)."
            )

        # Build blend
        from src.models.three_model_blend import ThreeModelBlend, ConditionalRates, DEFAULT_WEIGHTS
        cond_rates = ConditionalRates.from_data(df)

        blend = ThreeModelBlend(
            poisson_model=poisson,
            elo_model=elo,
            xgb_model=xgb,
            weights=DEFAULT_WEIGHTS,
            conditional_rates=cond_rates,
            historical_df=df,
        )

        # Persist
        blend.save(str(blend_path))
        logger.info("  ThreeModelBlend saved to %s", blend_path)

        # ── Fit and save 1X2 probability calibrator ──────────
        try:
            logger.info("  Fitting 1X2 calibrator on training data...")
            # Use last 20% of data chronologically for calibration fitting
            n = len(df)
            val_start = int(n * 0.80)
            df_val = df.iloc[val_start:].copy()
            df_val = df_val[df_val["result"].isin(["H", "D", "A"])].copy()
            if len(df_val) > 50:
                ppm = blend.precompute(df_val)
                w = blend.weights["1X2"]
                val_probs = (
                    w["poisson"] * ppm.pois_1x2
                    + w["elo"] * ppm.elo_1x2
                    + w["xgb"] * ppm.xgb_1x2
                )
                row_sums = val_probs.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                val_probs = val_probs / row_sums

                y_val = df_val["result"].map({"A": 0, "D": 1, "H": 2}).values

                from src.calibration import HybridTailCalibrator
                calibrator = HybridTailCalibrator(n_classes=3, tail_threshold=0.10)
                calibrator.fit(val_probs, y_val)

                cal_path = config.paths.models / "blend_calibrator_hybrid.joblib"
                import joblib
                joblib.dump(calibrator, cal_path)
                logger.info("  1X2 calibrator saved to %s", cal_path)
            else:
                logger.warning("  Not enough validation data (%d rows) to fit calibrator", len(df_val))
        except Exception as cal_exc:
            logger.warning("  Calibrator fitting failed: %s — calibration skipped", cal_exc)

        return {
            "success": True,
            "retrained": True,
            "blend_path": str(blend_path),
            "markets": blend.available_markets,
            "calibrator_fitted": True,
        }
    except Exception as exc:
        logger.error("Blend retrain failed: %s", exc, exc_info=True)
        return {"success": False, "retrained": False, "error": str(exc)}


# ═══════════════════════════════════════════════════════════
#  Step 4 — Predict upcoming matches
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
#  Step 4b — Value bets
# ═══════════════════════════════════════════════════════════


def step_value_bets(calibrate: str, max_odds: float) -> dict[str, Any]:
    """Run today_value_bets_live.py with the given calibration and max-odds settings.

    Parameters
    ----------
    calibrate : str
        Calibration method ('platt', 'isotonic', 'hybrid', 'none').
    max_odds : float
        Maximum decimal odds to accept.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``n_value_bets``, ``output_path``, ``error``.
    """
    logger.info("-" * 60)
    logger.info("STEP 4b: Value bets (today_value_bets_live.py)")
    logger.info(f"  Calibration: {calibrate}  |  Max odds: {max_odds}")
    logger.info("-" * 60)

    try:
        import subprocess
        import sys as _sys

        # Use the same Python interpreter to call the script
        python_exe = _sys.executable
        script_path = Path(__file__).resolve().parent / "today_value_bets_live.py"

        if not script_path.exists():
            raise FileNotFoundError(f"{script_path} not found")

        cmd = [
            python_exe, str(script_path),
            "--calibrate", calibrate,
            "--max-odds", str(max_odds),
            "--days", "14",
            "--quiet",
        ]

        logger.info("  Running: %s", " ".join(str(c) for c in cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("Value bets step failed (exit %d)", result.returncode)
            if result.stderr:
                logger.error("  stderr: %s", result.stderr.strip()[-500:])
            return {"success": False, "n_value_bets": 0, "error": result.stderr.strip()[:200] or "subprocess failed"}

        # Try to read the latest report to count value bets
        from pathlib import Path as _Path
        reports_dir = _Path(__file__).resolve().parent / "reports" / "value_bets"
        latest_csv = reports_dir / "latest.csv"
        n_value_bets = 0
        if latest_csv.exists():
            import pandas as _pd
            df = _pd.read_csv(latest_csv)
            if "positive_ev" in df.columns:
                n_value_bets = int(df["positive_ev"].sum())

        logger.info(
            "Value bets complete — %d value bets found",
            n_value_bets,
        )
        return {
            "success": True,
            "n_value_bets": n_value_bets,
            "calibration": calibrate,
            "max_odds": max_odds,
            "output_path": str(latest_csv) if latest_csv.exists() else "",
        }
    except Exception as exc:
        logger.error("Value bets step failed: %s", exc, exc_info=True)
        return {"success": False, "n_value_bets": 0, "error": str(exc)}


def step_predict(use_blend: bool = True) -> dict[str, Any]:
    """Generate predictions for upcoming / most recent matches.

    By default uses the saved ``ThreeModelBlend`` (Poisson + Elo + XGBoost)
    which predicts 1X2, Over/Under, and BTTS probabilities.  When
    ``use_blend=False`` loads the ``EnsembleModel`` (XGBoost + Logistic
    Regression + Poisson) for 1X2 predictions only.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``n_predictions``, ``output_path``, ``error``.
    """
    logger.info("-" * 60)
    logger.info("STEP 4: Predict upcoming matches")
    logger.info("-" * 60)

    try:
        from src.feature_engineering import build_features

        # Load preprocessed data for feature building
        data_path = config.paths.processed / "results_clean.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {data_path}")

        df = pd.read_csv(data_path, low_memory=False)

        # Predict using the 3-model blend or EnsembleModel
        if use_blend:
            logger.info("  Using 3-model blend for predictions ...")
            from src.models.three_model_blend import ThreeModelBlend

            blend_path = config.paths.models / "three_model_blend.joblib"
            if not blend_path.exists():
                raise FileNotFoundError(f"ThreeModelBlend not found at {blend_path} — run 'python run_pipeline.py' first to train the blend")

            blend = ThreeModelBlend.load(str(blend_path), historical_df=df)

            # Predict on last N rows using the blend's predict_matches
            n_recent = min(50, len(df) // 2)
            df_recent = df.iloc[-n_recent:]

            logger.info("  Generating blend predictions for %d matches ...", n_recent)
            blend_results = blend.predict_matches(df_recent)

            # Build output DataFrame from blend results
            output_df = df_recent[
                [c for c in ["date", "home_team", "away_team", "result", "league"]
                 if c in df.columns]
            ].copy()
            output_df["home_win_prob"] = blend_results["home_win_prob"].values
            output_df["draw_prob"] = blend_results["draw_prob"].values
            output_df["away_win_prob"] = blend_results["away_win_prob"].values
            output_df["max_prob"] = blend_results["confidence"].values
            output_df["model"] = "3_model_blend"
            output_df["prediction_label"] = blend_results["predicted_outcome"].values
            output_df["over_2_5_prob"] = blend_results["over_2_5_prob"].values
            output_df["under_2_5_prob"] = blend_results["under_2_5_prob"].values
            output_df["over_3_5_prob"] = blend_results["over_3_5_prob"].values
            output_df["under_3_5_prob"] = blend_results["under_3_5_prob"].values
            output_df["btts_prob"] = blend_results["btts_prob"].values
            output_df["btts_no_prob"] = blend_results["btts_no_prob"].values

        else:
            from src.ensemble import EnsembleModel

            # Load the trained ensemble model
            model_path = config.paths.models / _pipeline_cfg.model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Ensemble model not found at {model_path}")
            ensemble = EnsembleModel.load(str(model_path))

            # Build features
            logger.info("  Building feature matrix for prediction ...")
            X, _ = build_features(df, is_training=True)

            # Predict on the last N rows (most recent matches)
            n_recent = min(50, len(X) // 2)
            X_recent = X.iloc[-n_recent:]

            # Align raw df for Poisson model (needs home_team, away_team, etc.)
            df_sorted = df.loc[X.index] if hasattr(X, "index") else df
            df_raw_recent = df_sorted.iloc[-n_recent:] if len(df_sorted) >= n_recent else df_sorted

            logger.info("  Generating ensemble predictions for %d matches ...", n_recent)
            probs = ensemble.predict_proba(X_recent, df_raw=df_raw_recent)
            preds = ensemble.predict(X_recent, df_raw=df_raw_recent)

            # Build output DataFrame
            output_df = df.iloc[-n_recent:][
                [c for c in ["date", "home_team", "away_team", "result", "league"]
                 if c in df.columns]
            ].copy()
            output_df["prediction"] = preds
            output_df["home_win_prob"] = probs[:, 2]
            output_df["draw_prob"] = probs[:, 1]
            output_df["away_win_prob"] = probs[:, 0]
            output_df["max_prob"] = probs.max(axis=1)
            output_df["model"] = "ensemble"

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
    logger.info("-" * 60)
    logger.info("STEP 5: Generate summary report")
    logger.info("-" * 60)

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
        # Determine which model type was used
        br = results.get("retrain_blend", {})
        if br.get("success") and br.get("retrained"):
            model_label = "3-Model Blend (Poisson + Elo + XGBoost)"
        elif br.get("success") and br.get("skipped"):
            model_label = "Ensemble (XGBoost + Logistic Regression + Poisson)"
        else:
            model_label = "Ensemble (XGBoost + Logistic Regression + Poisson)"
        lines.append(f"  Model:         {model_label}")
        lines.append("")

        # ── Step status table ────────────────────────────
        lines.append(f"  {'-' * 50}")
        lines.append(f"  {'Step':<30s} {'Status':<15s} {'Details':>40s}")
        lines.append(f"  {'-' * 50}")

        for step_name, result in results.items():
            status = "PASS" if result.get("success") else "FAIL"
            detail = result.get("error", "OK")
            detail_str = str(detail)[:40] if len(str(detail)) > 40 else str(detail)
            lines.append(f"  {step_name:<30s} {status:<15s} {detail_str:>40s}")
        lines.append(f"  {'-' * 50}")
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
            lines.append(f"  Retrain:   YES — ensemble saved to {tr.get('model_path', '?')}")
            if tr.get("val_loss"):
                lines.append(f"             Val log-loss: {tr['val_loss']:.4f}")
            if tr.get("weights"):
                w_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(tr["weights"].items()))
                lines.append(f"             Weights: {w_str}")
        else:
            lines.append(f"  Retrain:   NO (ensemble is current)")

        br = results.get("retrain_blend", {})
        if br.get("retrained"):
            lines.append(f"  Blend:     YES — 3-model blend saved to {br.get('blend_path', '?')}")
            if br.get("markets"):
                lines.append(f"             Markets: {', '.join(br['markets'])}")
        elif br.get("success", True):
            lines.append(f"  Blend:     Skipped ({br.get('message', 'not requested')})")
        else:
            lines.append(f"  Blend:     FAILED ({br.get('error', 'unknown')})")

        pr = results.get("predict", {})
        if pr.get("success"):
            lines.append(f"  Predict:   {pr.get('n_predictions', 0)} matches -> "
                         f"{pr.get('output_path', '?')}")

        lines.append("")
        lines.append(sep)
        lines.append("")

        report_text = "\n".join(lines)

        # ── Print to console (handle Windows encoding gracefully) ──
        safe_text = report_text.replace("—", "-").replace("─", "-")
        try:
            print(safe_text)
        except UnicodeEncodeError:
            print(safe_text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))

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
    parser.add_argument(
        "--calibrate", type=str, default="hybrid",
        choices=["platt", "isotonic", "hybrid", "none"],
        help="Calibration method for value bets (default: hybrid)",
    )
    parser.add_argument(
        "--max-odds", type=float, default=30.0,
        help="Maximum decimal odds for value bets (default: 30.0). "
             "Bets above this are rejected to manage variance.",
    )
    parser.add_argument(
        "--skip-value-bets", action="store_true",
        help="Skip the value bets step",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use the EnsembleModel (XGBoost + Logistic Regression + Poisson) "
             "instead of the 3-model blend.  Predictions will only include "
             "1X2 probabilities (no Over/Under or BTTS).",
    )
    parser.add_argument(
        "--skip-blend", action="store_true",
        help="Skip training the 3-model blend (ensemble model only)",
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

    # ── Step 3b: Build 3-blend (default, skip if --ensemble or --skip-blend) ─
    if args.ensemble or args.skip_blend:
        if args.skip_blend:
            logger.info("Skipping blend training (--skip-blend)")
        else:
            logger.info("Using ensemble model (--ensemble) — skipping 3-model blend")
        results["retrain_blend"] = {"success": True, "skipped": True, "message": "not requested"}
    elif args.lightweight or args.skip_train:
        results["retrain_blend"] = {"success": True, "retrained": False, "skipped": True, "message": "train skipped"}
    else:
        results["retrain_blend"] = step_retrain_blend()

    # ── Step 4: Predict ──────────────────────────────────
    results["predict"] = step_predict(use_blend=not args.ensemble)

    # ── Step 4b: Value bets ──────────────────────────────
    if args.skip_value_bets:
        logger.info("Skipping value bets (--skip-value-bets)")
        results["value_bets"] = {"success": True, "skipped": True}
    else:
        results["value_bets"] = step_value_bets(args.calibrate, args.max_odds)

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
                logger.warning("  %s: %s", step_name, result.get("error", "Unknown error"))

    # ── Print summary to stdout ────────────────────────
    status_icon = "OK" if ok else "WARN"
    print(f"\n  [{status_icon}] Pipeline finished in {time.time() - _start_time:.1f}s")
    print(f"  Log: logs/pipeline.log")
    print()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
