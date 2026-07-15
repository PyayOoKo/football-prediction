#!/usr/bin/env python3
"""
Daily Feature Computation — Loads new matches, computes all features using Feature Store.

Steps:
1. Load new/updated matches from database or processed CSV
2. Compute rolling features (goals, xG, form, Elo, H2H, etc.)
3. Store computed features in Feature Store
4. Log success/failure + metrics

Scheduler:
    python -m src.scheduler.cli run --tasks daily_feature_computation

Dependencies:
    - daily_data_pipeline (must run first — needs fresh data)
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.monitoring import Monitor

logger = logging.getLogger("daily_feature_computation")

# ── Data paths ────────────────────────────────────────
RAW_DIR = _project_root / "data" / "raw"
PROCESSED_DIR = _project_root / "data" / "processed"
FEATURE_DIR = _project_root / "data" / "features"
REPORT_DIR = _project_root / "reports" / "feature_computation"


def load_latest_data() -> pd.DataFrame | None:
    """Load the most recent processed match data."""
    # Try processed CSV first
    processed_path = PROCESSED_DIR / "results_clean.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path, low_memory=False)
        logger.info("Loaded %d rows from processed data: %s", len(df), processed_path)
        return df

    # Fallback: try raw worldcup data
    raw_path = RAW_DIR / "worldcup_all.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path, low_memory=False)
        logger.info("Loaded %d rows from raw data: %s", len(df), raw_path)
        return df

    # Fallback: try database
    try:
        from src.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_database("matches")
        if df is not None and not df.empty:
            logger.info("Loaded %d rows from database", len(df))
            return df
    except Exception as exc:
        logger.warning("Could not load from database: %s", exc)

    logger.error("No data sources available for feature computation")
    return None


def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute all feature sets from the match data.

    Returns (feature_matrix, feature_report).
    """
    report = {"features_computed": {}, "duration": 0.0}
    start = time.perf_counter()

    from src.feature_engineering import build_features

    X, y = build_features(df, is_training=True)

    elapsed = time.perf_counter() - start
    report["duration"] = elapsed
    report["feature_count"] = X.shape[1] if hasattr(X, "shape") else 0
    report["row_count"] = X.shape[0] if hasattr(X, "shape") else 0
    report["features_computed"] = list(X.columns) if hasattr(X, "columns") else []

    logger.info("Computed %d features for %d rows in %.1fs",
                report["feature_count"], report["row_count"], elapsed)
    return X, report


def store_features(X, report: dict) -> bool:
    """Store computed features to disk and feature store."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Save as Parquet for efficient storage
        if hasattr(X, "to_parquet"):
            feature_path = FEATURE_DIR / f"features_{datetime.now().strftime('%Y%m%d')}.parquet"
            X.to_parquet(feature_path)
            report["feature_path"] = str(feature_path)
            logger.info("Saved features to: %s", feature_path)

        # Also save as CSV for inspection
        if hasattr(X, "to_csv"):
            csv_path = FEATURE_DIR / f"features_latest.csv"
            X.to_csv(csv_path, index=False)
            report["feature_csv_path"] = str(csv_path)

    except Exception as exc:
        logger.warning("Could not save features to disk: %s", exc)
        return False

    # Try Feature Store if available
    try:
        from src.feature_store import FeatureStore
        store = FeatureStore()
        store.store_batch(X)
        logger.info("Stored features in Feature Store")
    except Exception as exc:
        logger.debug("Feature Store not available: %s", exc)

    return True


def compute_elo(df: pd.DataFrame) -> dict:
    """Compute Elo ratings separately for validation."""
    start = time.perf_counter()
    result = {"success": False, "teams_rated": 0, "duration": 0.0}

    try:
        from src.elo import EloSystem
        elo = EloSystem()
        elo.fit(df)
        result["teams_rated"] = len(elo.ratings) if hasattr(elo, "ratings") else 0
        result["success"] = True
        elapsed = time.perf_counter() - start
        result["duration"] = elapsed
        logger.info("Computed Elo ratings for %d teams in %.1fs",
                    result["teams_rated"], elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        result["duration"] = elapsed
        result["error"] = str(exc)
        logger.warning("Elo computation failed: %s", exc)

    return result


def run_feature_pipeline(monitor: Monitor | None = None) -> dict:
    """Run the full daily feature computation pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING DAILY FEATURE COMPUTATION")
    logger.info("=" * 60)

    pipeline_start = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load latest data
    df = load_latest_data()
    if df is None:
        return {"pipeline": "daily_feature_computation", "error": "No data available",
                "success": False, "duration_seconds": time.perf_counter() - pipeline_start}

    # Step 2: Compute features
    X, feature_report = compute_features(df)

    # Step 3: Store features
    stored = store_features(X, feature_report)

    # Step 4: Compute Elo (validation)
    elo_result = compute_elo(df)

    # Step 5: Record monitoring
    if monitor:
        monitor.record_etl(
            pipeline="daily_feature_computation",
            duration_seconds=feature_report["duration"],
            rows_imported=feature_report["row_count"],
            success=elo_result["success"],
        )

    pipeline_elapsed = time.perf_counter() - pipeline_start

    report = {
        "pipeline": "daily_feature_computation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(pipeline_elapsed, 2),
        "feature_count": feature_report.get("feature_count", 0),
        "row_count": feature_report.get("row_count", 0),
        "features": feature_report.get("features_computed", []),
        "feature_path": feature_report.get("feature_path"),
        "elo_teams_rated": elo_result.get("teams_rated", 0),
        "elo_duration": round(elo_result.get("duration", 0), 2),
        "stored": stored,
        "success": True,
    }

    logger.info("Feature computation complete: %d features for %d rows in %.1fs",
                report["feature_count"], report["row_count"], pipeline_elapsed)
    return report


def main() -> int:
    """CLI entry point for daily feature computation."""
    import argparse

    parser = argparse.ArgumentParser(description="Daily feature computation")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--skip-elo", action="store_true", help="Skip Elo computation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    monitor = Monitor()
    report = run_feature_pipeline(monitor)

    if not args.quiet:
        print(f"\n{'=' * 50}")
        print(f"  FEATURE COMPUTATION REPORT")
        print(f"{'=' * 50}")
        print(f"  Duration:    {report.get('duration_seconds', 0):.1f}s")
        print(f"  Rows:        {report.get('row_count', 0)}")
        print(f"  Features:    {report.get('feature_count', 0)}")
        print(f"  Elo teams:   {report.get('elo_teams_rated', 0)}")
        print(f"  Stored:      {'✅' if report.get('stored') else '❌'}")
        if report.get("feature_path"):
            print(f"  Output:      {report['feature_path']}")
        print(f"{'=' * 50}\n")

    return 0 if report.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
