"""
Backtesting — simulate value betting on historical match data.

Walks through test-set matches chronologically, compares model probabilities
against actual bookmaker odds, places Kelly-sized stakes on positive-EV bets,
and reports ROI, Yield, Profit, Win Rate, and Maximum Drawdown.

Usage:
    python run_backtest.py

Requires:
    - Preprocessed data at ``data/processed/results_clean.csv``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_backtest")


def main() -> None:
    print("=" * 90)
    print("  FOOTBALL PREDICTION — VALUE BETTING BACKTEST".center(88))
    print("=" * 90)

    # ── 1. Load preprocessed data ───────────────────────
    data_path = config.paths.processed / "results_clean.csv"

    if not data_path.exists():
        print(f"\n  [X] Preprocessed data not found at {data_path}")
        print("    Run:  python -c \"from src.preprocessing import run_preprocessing; run_preprocessing()\"")
        sys.exit(1)

    print(f"\n  Loading preprocessed data ...")
    df = pd.read_csv(data_path, low_memory=False)

    # Parse dates for sorting
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    print(f"  [OK] {len(df):,} rows x {len(df.columns)} columns")

    # ── Detect available odds columns ────────────────────
    # The preprocessed CSV preserves odds columns from the raw data.
    # Try multiple column sets in preference order.
    odds_candidates = [
        ("BbAvA", "BbAvD", "BbAvH"),      # Average odds (preferred)
        ("B365A", "B365D", "B365H"),      # Bet365
        ("BWA", "BWD", "BWH"),            # Bet&Win
        ("IWA", "IWD", "IWH"),            # Interwetten
        ("PSA", "PSD", "PSH"),            # Pinnacle
    ]

    odds_cols = None
    for cols in odds_candidates:
        if all(c in df.columns for c in cols):
            non_null = df[list(cols)].notna().all(axis=1).mean()
            if non_null > 0.5:
                odds_cols = cols
                logger.info("Using odds columns: %s (%.0f%% non-null)", cols, non_null * 100)
                break

    if odds_cols:
        print(f"  [OK] Found odds columns: {odds_cols}")
        # Create a sorted copy of the relevant columns for odds alignment.
        # build_features() sorts by date + home_team and resets index, so we
        # pre-sort the odds data identically for positional alignment.
        odds_raw = df[["date", "home_team", "away_team"] + list(odds_cols)].copy()
        odds_raw.sort_values(["date", "home_team"], inplace=True)
        odds_raw.reset_index(drop=True, inplace=True)
    else:
        odds_raw = None
        print(f"  [!] No odds columns found in preprocessed data")
        available = [c for c in df.columns if "Bb" in c or "B3" in c or "B365" in c
                     or c in ("PSH", "PSD", "PSA", "BWH", "BWD", "BWA")]
        print(f"    Available betting columns: {available[:10]}")

    # ── 2. Build features ───────────────────────────────
    from src.feature_engineering import build_features

    print(f"\n  Building features (rolling stats, H2H, league position) ...")
    X, y = build_features(df, is_training=True)
    print(f"  [OK] Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  [OK] Target distribution: {dist}")

    # ── 3. Split chronologically ─────────────────────────
    from src.feature_engineering import train_val_test_split

    print(f"\n  Splitting chronologically (70/15/15) ...")
    splits = train_val_test_split(X, y)
    print(f"  [OK] Train: {len(splits['X_train']):,}  |  "
          f"Val: {len(splits['X_val']):,}  |  "
          f"Test: {len(splits['X_test']):,}")

    # ── 4. Extract test-set odds ────────────────────────
    # build_features() sorts by date and does NOT drop any rows.
    # The test set is the last 15% of rows positionally.
    # Our pre-sorted odds_raw aligns positionally with the sorted feature matrix.
    odds_df = None
    if odds_raw is not None:
        n_total = len(X)
        n_test = len(splits["X_test"])
        n_val = len(splits["X_val"])
        n_train = len(splits["X_train"])

        # build_features sorts + resets index, and train_val_test_split
        # uses positional iloc: test is last 15% of rows.
        test_start = n_total - n_test
        odds_df = odds_raw.iloc[test_start:test_start + n_test].copy()

        # Also add team names for display
        odds_df["home_team"] = odds_raw["home_team"].iloc[
            test_start:test_start + n_test
        ].values
        odds_df["away_team"] = odds_raw["away_team"].iloc[
            test_start:test_start + n_test
        ].values

        n_with_odds = odds_df[list(odds_cols)].notna().all(axis=1).sum()
        print(f"  [OK] Extracted odds for {len(odds_df)} test matches "
              f"({n_with_odds} with complete odds)")

    # ── 5. Train model on train + val ──────────────────
    from src.train import train_model

    print(f"\n  Training XGBoost on train + val ...")
    X_train_full = pd.concat([splits["X_train"], splits["X_val"]], axis=0)
    y_train_full = pd.concat([splits["y_train"], splits["y_val"]], axis=0)

    model, history = train_model(X_train_full, y_train_full)
    print(f"  [OK] Training log-loss: {history['train_loss'][0]:.4f}")

    # ── 6. Run backtest ─────────────────────────────────
    from src.backtesting import run_backtest

    print(f"\n{'=' * 90}")
    print("  RUNNING BACKTEST...")
    print(f"{'=' * 90}")

    result = run_backtest(
        model=model,
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        odds_df=odds_df,
        odds_cols=odds_cols or ("BbAvA", "BbAvD", "BbAvH"),
        team_cols=("home_team", "away_team"),
        initial_bankroll=config.value_betting.bankroll,
        kelly_fraction=config.value_betting.kelly_fraction,
        min_ev=config.value_betting.min_ev,
        output_dir=config.paths.data.parent / "reports" / "backtest",
        print_report=True,
        show_charts=False,
    )

    metrics = result["metrics"]
    chart_paths = result["chart_paths"]

    # ── 7. Print summary ────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  SUMMARY".center(88))
    print(f"{'=' * 90}")
    print(f"\n    Model:        XGBoost")
    print(f"    Bankroll:     £{metrics.initial_bankroll:.2f} -> £{metrics.final_bankroll:.2f}")
    print(f"    Bets placed:  {metrics.total_bets}")
    print(f"    Win rate:     {metrics.win_rate_pct:.1f}%")
    print(f"    ROI:          {metrics.roi_pct:+.2f}%")
    print(f"    Yield:        {metrics.yield_pct:+.2f}%")
    print(f"    Max DD:       {metrics.max_drawdown_pct:.1f}%")

    print(f"\n    Charts saved:")
    for name, path in chart_paths.items():
        print(f"      • {name}: {path}")

    print(f"\n{'=' * 90}")
    print("  To re-run with different parameters:")
    print("    python run_backtest.py")
    print("    # Or change settings in config.py under value_betting")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
