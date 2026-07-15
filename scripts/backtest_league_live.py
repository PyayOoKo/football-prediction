"""
backtest_league_live.py — Backtest on 5 European Leagues with REAL bookmaker odds.

Uses the data/raw/league_all.csv (17,937 matches from 2016-2026) which includes
BetBrain average odds (BbAvH, BbAvD, BbAvA) from football-data.co.uk.

Usage:
    python scripts/backtest_league_live.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from config import config
from src.feature_engineering import build_features
from src.feature_engineering import train_val_test_split
from src.train import train_model
from src.backtesting import run_backtest


def main() -> None:
    print("=" * 90)
    print("  LEAGUE BACKTEST — REAL BOOKMAKER ODDS".center(88))
    print("  Top 5 European Leagues  |  LightGBM model")
    print("=" * 90)

    # ── 1. Load raw league data ──────────────────────────
    raw_path = Path("data/raw/league_all.csv")
    print(f"\n  Loading league data from {raw_path} ...")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"  [OK] {len(df):,} rows x {len(df.columns)} cols")

    # ── 2. Filter to matches with BbAv odds ──────────────
    odds_cols = ("bbavh", "bbavd", "bbava")
    has_odds = df[list(odds_cols)].notna().all(axis=1)
    print(f"\n  Matches with BbAv odds: {has_odds.sum():,} / {len(df):,} ({100 * has_odds.mean():.0f}%)")

    if has_odds.sum() < 100:
        print("  [X] Not enough odds data — need at least 100 matches")
        return

    # Keep only rows with odds
    df_clean = df[has_odds].copy()
    df_clean.sort_values(["date", "home_team"], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    # Parse date
    df_clean["date"] = pd.to_datetime(df_clean["date"], dayfirst=True, errors="coerce")
    df_clean.dropna(subset=["date", "result"], inplace=True)

    # Add 'target' column that build_features() expects: H=2, D=1, A=0
    # Keep 'result' column for internal processing (Elo, H2H, etc.)
    df_clean["target"] = df_clean["result"].map({"H": 2, "D": 1, "A": 0})
    df_clean.dropna(subset=["target"], inplace=True)
    df_clean["target"] = df_clean["target"].astype(int)

    print(f"  [OK] Cleaned: {len(df_clean):,} rows with odds + results")
    print(f"       Date range: {df_clean['date'].min():%Y-%m-%d} to {df_clean['date'].max():%Y-%m-%d}")

    # Save odds separately (aligned by sorted order)
    odds_raw = df_clean[["date", "home_team", "away_team"] + list(odds_cols)].copy()

    # ── 3. Build features ────────────────────────────────
    print(f"\n  Building features (rolling stats, form, H2H) ...")
    X, y = build_features(df_clean, is_training=True)
    print(f"  [OK] Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  Target distribution: {dist}")

    # ── 4. Split chronologically ─────────────────────────
    print(f"\n  Splitting chronologically (70/15/15) ...")
    splits = train_val_test_split(X, y)
    print(f"  [OK] Train: {len(splits['X_train']):,}  |  "
          f"Val: {len(splits['X_val']):,}  |  "
          f"Test: {len(splits['X_test']):,}")

    # ── 5. Align test-set odds ───────────────────────────
    n_total = len(X)
    n_test = len(splits["X_test"])
    n_val = len(splits["X_val"])
    n_train = len(splits["X_train"])
    test_start = n_total - n_test

    odds_test = odds_raw.iloc[test_start:test_start + n_test].copy()
    odds_test.rename(columns={
        "bbavh": "BbAvH",
        "bbavd": "BbAvD",
        "bbava": "BbAvA",
    }, inplace=True)

    n_with_odds = odds_test[["BbAvH", "BbAvD", "BbAvA"]].notna().all(axis=1).sum()
    print(f"  [OK] Test set odds: {n_with_odds}/{len(odds_test)} matches have complete odds")

    # ── 6. Train LightGBM on train + val ────────────────
    print(f"\n  Training LightGBM on train + val ...")
    X_train_full = pd.concat([splits["X_train"], splits["X_val"]], axis=0)
    y_train_full = pd.concat([splits["y_train"], splits["y_val"]], axis=0)

    model, history = train_model(X_train_full, y_train_full)
    print(f"  [OK] Training log-loss: {history['train_loss'][0]:.4f}")

    # ── 7. Run backtest ──────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  RUNNING BACKTEST WITH REAL ODDS...")
    print(f"{'=' * 90}")

    result = run_backtest(
        model=model,
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        odds_df=odds_test,
        odds_cols=("BbAvA", "BbAvD", "BbAvH"),
        team_cols=("home_team", "away_team"),
        initial_bankroll=1000.0,
        kelly_fraction=config.value_betting.kelly_fraction,
        min_ev=config.value_betting.min_ev,
        output_dir=Path("reports") / "backtest",
        print_report=True,
        show_charts=False,
    )

    metrics = result["metrics"]
    chart_paths = result["chart_paths"]

    # ── 8. Final summary ────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  LEAGUE BACKTEST — FINAL RESULTS".center(88))
    print(f"{'=' * 90}")
    print(f"\n    Model:        LightGBM (config default)")
    print(f"    Dataset:      Top 5 European Leagues ({len(df_clean):,} matches)")
    print(f"    Test period:  {odds_test['date'].min():%Y-%m-%d} to {odds_test['date'].max():%Y-%m-%d}")
    print(f"    Odds source:  BetBrain Average (BbAv) — REAL market odds")
    print(f"    Bankroll:     ${metrics.initial_bankroll:.2f} -> ${metrics.final_bankroll:.2f}")
    print(f"    Bets placed:  {metrics.total_bets}")
    print(f"    Win rate:     {metrics.win_rate_pct:.1f}%")
    print(f"    ROI:          {metrics.roi_pct:+.2f}%")
    print(f"    Yield:        {metrics.yield_pct:+.2f}%")
    print(f"    Max DD:       {metrics.max_drawdown_pct:.1f}%")
    if hasattr(metrics, 'profit_factor'):
        print(f"    Profit factor: {metrics.profit_factor:.2f}")
    if hasattr(metrics, 'sharpe_ratio'):
        print(f"    Sharpe ratio:  {metrics.sharpe_ratio:.2f}")

    print(f"\n    Charts saved:")
    for name, path in chart_paths.items():
        print(f"      * {name}: {path}")

    # Verdict
    print(f"\n{'=' * 90}")
    if metrics.roi_pct > 0:
        print(f"  VERDICT: PROFITABLE [YES]  (+{metrics.roi_pct:.2f}% ROI over {metrics.total_bets} bets)")
    else:
        print(f"  VERDICT: NOT PROFITABLE [NO]  ({metrics.roi_pct:.2f}% ROI over {metrics.total_bets} bets)")
    print(f"{'=' * 90}")
    print()


if __name__ == "__main__":
    main()
