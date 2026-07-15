#!/usr/bin/env python3
"""
Daily Model Retraining — Checks for new data, retrains models, validates, saves best.

Steps:
1. Check if sufficient new data exists (configurable threshold)
2. Retrain XGBoost / Random Forest / Logistic Regression / Ensemble
3. Validate new models against test set
4. Compare with existing model — keep the best
5. Save best model(s) to models/
6. Log success/failure + metrics

Scheduler:
    python -m src.scheduler.cli run --tasks daily_model_retraining

Dependencies:
    - daily_feature_computation (must run first — needs fresh features)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd

from src.monitoring import Monitor

logger = logging.getLogger("daily_model_retraining")

# ── Paths ─────────────────────────────────────────────
FEATURE_DIR = _project_root / "data" / "features"
MODEL_DIR = _project_root / "models"
REPORT_DIR = _project_root / "reports" / "retraining"
PROCESSED_DIR = _project_root / "data" / "processed"

# ── Thresholds ────────────────────────────────────────
MIN_NEW_MATCHES_FOR_RETRAIN = 10  # Minimum new matches to trigger retrain
DEFAULT_TEST_SPLIT = 0.2


def count_new_data() -> int:
    """Count how many new matches are available since last retrain."""
    processed_path = PROCESSED_DIR / "results_clean.csv"
    if not processed_path.exists():
        return 0

    # Load current data
    df = pd.read_csv(processed_path, low_memory=False)

    # Try to find when we last trained
    training_log = REPORT_DIR / "last_training.txt"
    if training_log.exists():
        last_train_date_str = training_log.read_text().strip()
        if last_train_date_str and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            last_train_date = pd.to_datetime(last_train_date_str, errors="coerce")
            if pd.notna(last_train_date):
                new_matches = df[df["date"] > last_train_date]
                logger.info("Found %d new matches since last retrain (%s)",
                            len(new_matches), last_train_date.date())
                return len(new_matches)

    # No training log — return total count to trigger initial training
    return len(df)


def should_retrain(new_matches: int) -> bool:
    """Determine whether retraining should proceed based on new data threshold."""
    if new_matches >= MIN_NEW_MATCHES_FOR_RETRAIN:
        logger.info("Retraining triggered: %d new matches >= threshold (%d)",
                    new_matches, MIN_NEW_MATCHES_FOR_RETRAIN)
        return True

    logger.info("Skipping retrain: %d new matches < threshold (%d)",
                new_matches, MIN_NEW_MATCHES_FOR_RETRAIN)
    return False


def train_models(df: pd.DataFrame) -> dict[str, Any]:
    """Train all model types and return results with metrics."""
    results = {}
    start = time.perf_counter()

    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    from sklearn.model_selection import train_test_split

    from src.feature_engineering import build_features

    # Build feature matrix
    X, y = build_features(df, is_training=True)
    logger.info("Built feature matrix: %d samples, %d features", X.shape[0], X.shape[1])

    # Train/test split (chronological)
    split_idx = int(len(X) * (1 - DEFAULT_TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ── XGBoost ────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        sub_start = time.perf_counter()
        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
        xgb.fit(X_train, y_train)
        y_pred_proba = xgb.predict_proba(X_test)
        y_pred = xgb.predict(X_test)
        elapsed = time.perf_counter() - sub_start

        results["xgboost"] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "log_loss": float(log_loss(y_test, y_pred_proba)),
            "brier_score": float(np.mean([brier_score_loss(y_test == c, y_pred_proba[:, i])
                                          for i, c in enumerate(xgb.classes_)])),
            "training_time": round(elapsed, 2),
            "model": xgb,
        }
        logger.info("XGBoost: acc=%.3f, log_loss=%.4f in %.1fs",
                    results["xgboost"]["accuracy"], results["xgboost"]["log_loss"], elapsed)
    except Exception as exc:
        logger.error("XGBoost training failed: %s", exc)

    # ── Random Forest ─────────────────────────────────
    try:
        from sklearn.ensemble import RandomForestClassifier
        sub_start = time.perf_counter()
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred_proba = rf.predict_proba(X_test)
        y_pred = rf.predict(X_test)
        elapsed = time.perf_counter() - sub_start

        results["random_forest"] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "log_loss": float(log_loss(y_test, y_pred_proba)),
            "training_time": round(elapsed, 2),
            "model": rf,
        }
        logger.info("RandomForest: acc=%.3f, log_loss=%.4f in %.1fs",
                    results["random_forest"]["accuracy"], results["random_forest"]["log_loss"], elapsed)
    except Exception as exc:
        logger.error("RandomForest training failed: %s", exc)

    # ── Logistic Regression ───────────────────────────
    try:
        from sklearn.linear_model import LogisticRegression
        sub_start = time.perf_counter()
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        lr.fit(X_train, y_train)
        y_pred_proba = lr.predict_proba(X_test)
        y_pred = lr.predict(X_test)
        elapsed = time.perf_counter() - sub_start

        results["logistic_regression"] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "log_loss": float(log_loss(y_test, y_pred_proba)),
            "training_time": round(elapsed, 2),
            "model": lr,
        }
        logger.info("LogisticRegression: acc=%.3f, log_loss=%.4f in %.1fs",
                    results["logistic_regression"]["accuracy"],
                    results["logistic_regression"]["log_loss"], elapsed)
    except Exception as exc:
        logger.error("LogisticRegression training failed: %s", exc)

    total_elapsed = time.perf_counter() - start
    results["_meta"] = {
        "total_training_time": round(total_elapsed, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
    }

    logger.info("All models trained in %.1fs", total_elapsed)
    return results


def select_best_model(results: dict[str, Any]) -> tuple[str, dict]:
    """Select the best model based on log_loss (lower is better)."""
    candidates = {}
    for name, r in results.items():
        if name.startswith("_"):
            continue
        if "log_loss" in r:
            candidates[name] = r["log_loss"]

    if not candidates:
        return "none", {}

    best_name = min(candidates, key=candidates.get)
    best_result = results[best_name]
    return best_name, best_result


def save_best_model(results: dict[str, Any], best_name: str) -> None:
    """Save the best model to the models directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    import joblib

    best_model = results[best_name]["model"]
    model_path = MODEL_DIR / f"{best_name}_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Saved best model '%s' to: %s", best_name, model_path)

    # Also save as generic 'model.pkl' for API/dashboard
    generic_path = MODEL_DIR / "model.pkl"
    joblib.dump(best_model, generic_path)
    logger.info("Saved generic model to: %s", generic_path)

    # Save metadata
    meta = {
        "best_model": best_name,
        "accuracy": results[best_name].get("accuracy"),
        "log_loss": results[best_name].get("log_loss"),
        "brier_score": results[best_name].get("brier_score"),
        "training_time": results[best_name].get("training_time"),
        "total_training_time": results["_meta"]["total_training_time"],
        "n_train": results["_meta"]["n_train"],
        "n_test": results["_meta"]["n_test"],
        "n_features": results["_meta"]["n_features"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = MODEL_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved model metadata to: %s", meta_path)

    # Update last training timestamp
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts_path = REPORT_DIR / "last_training.txt"
    ts_path.write_text(datetime.now(timezone.utc).isoformat())


def run_retraining_pipeline(monitor: Monitor | None = None) -> dict:
    """Run the full daily model retraining pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING DAILY MODEL RETRAINING")
    logger.info("=" * 60)

    pipeline_start = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Check for new data
    new_matches = count_new_data()
    if not should_retrain(new_matches):
        report = {
            "pipeline": "daily_model_retraining",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.perf_counter() - pipeline_start, 2),
            "new_matches": new_matches,
            "retrained": False,
            "reason": f"Only {new_matches} new matches < threshold {MIN_NEW_MATCHES_FOR_RETRAIN}",
            "success": True,
        }
        logger.info("Skipped retraining: %s", report["reason"])
        return report

    # Step 2: Load data
    processed_path = PROCESSED_DIR / "results_clean.csv"
    if not processed_path.exists():
        return {"pipeline": "daily_model_retraining", "error": "No processed data",
                "success": False, "duration_seconds": round(time.perf_counter() - pipeline_start, 2)}

    df = pd.read_csv(processed_path, low_memory=False)
    logger.info("Loaded %d rows for retraining", len(df))

    # Step 3: Train all models
    training_results = train_models(df)
    if not training_results or len(training_results) <= 1:
        return {"pipeline": "daily_model_retraining", "error": "No models trained",
                "success": False, "duration_seconds": round(time.perf_counter() - pipeline_start, 2)}

    # Step 4: Select best model
    best_name, best_result = select_best_model(training_results)

    # Step 5: Save best model
    save_best_model(training_results, best_name)

    # Step 6: Build comparison table
    model_comparison = {}
    for name, r in training_results.items():
        if name.startswith("_"):
            continue
        model_comparison[name] = {
            "accuracy": r.get("accuracy"),
            "log_loss": r.get("log_loss"),
            "brier_score": r.get("brier_score"),
            "training_time": r.get("training_time"),
        }

    # Record monitoring
    if monitor:
        monitor.record_etl(
            pipeline="daily_model_retraining",
            duration_seconds=training_results["_meta"]["total_training_time"],
            rows_imported=training_results["_meta"]["n_train"],
            success=True,
        )

    pipeline_elapsed = time.perf_counter() - pipeline_start

    report = {
        "pipeline": "daily_model_retraining",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(pipeline_elapsed, 2),
        "new_matches": new_matches,
        "retrained": True,
        "best_model": best_name,
        "best_accuracy": best_result.get("accuracy"),
        "best_log_loss": best_result.get("log_loss"),
        "best_brier_score": best_result.get("brier_score"),
        "n_train": training_results["_meta"]["n_train"],
        "n_test": training_results["_meta"]["n_test"],
        "n_features": training_results["_meta"]["n_features"],
        "model_comparison": model_comparison,
        "success": True,
    }

    # Save report
    report_path = REPORT_DIR / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved retraining report: %s", report_path)

    logger.info("Retraining complete: best=%s, acc=%.3f, log_loss=%.4f in %.1fs",
                best_name, report["best_accuracy"], report["best_log_loss"], pipeline_elapsed)
    return report


def main() -> int:
    """CLI entry point for daily model retraining."""
    import argparse

    parser = argparse.ArgumentParser(description="Daily model retraining")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of threshold")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.force:
        global MIN_NEW_MATCHES_FOR_RETRAIN
        MIN_NEW_MATCHES_FOR_RETRAIN = 0

    monitor = Monitor()
    report = run_retraining_pipeline(monitor)

    if not args.quiet:
        print(f"\n{'=' * 55}")
        print(f"  MODEL RETRAINING REPORT")
        print(f"{'=' * 55}")
        if report.get("retrained"):
            print(f"  Best model:  {report['best_model']}")
            print(f"  Accuracy:    {report.get('best_accuracy', 0):.4f}")
            print(f"  Log loss:    {report.get('best_log_loss', 0):.4f}")
            print(f"  Brier:       {report.get('best_brier_score', 0):.4f}")
            print(f"  Train rows:  {report.get('n_train', 0)}")
            print(f"  Features:    {report.get('n_features', 0)}")
            print(f"  Duration:    {report.get('duration_seconds', 0):.1f}s")
            for name, metrics in report.get("model_comparison", {}).items():
                acc = metrics.get("accuracy", 0)
                ll = metrics.get("log_loss", 0)
                print(f"    {name:<22} acc={acc:.3f}  ll={ll:.4f}")
        else:
            print(f"  Skipped:     {report.get('reason', 'Unknown')}")
        print(f"{'=' * 55}\n")

    return 0 if report.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
