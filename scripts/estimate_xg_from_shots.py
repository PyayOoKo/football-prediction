"""
estimate_xg_from_shots.py — Estimate xG from shots-on-target for leagues without real xG.

For leagues where we have shots/shots-on-target data but no xG (all top 5 leagues),
estimates xG using per-league Poisson regression: SOT → expected goals.

Usage
-----
    python scripts/estimate_xg_from_shots.py --league E0     # Single league
    python scripts/estimate_xg_from_shots.py --all            # All top 5 leagues
    python scripts/estimate_xg_from_shots.py --all --train-only  # Skip DB update
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("estimate_xg_from_shots")

DB_PATH = Path("data/football_data.db")
MODELS_DIR = Path("models/per_league")

TOP_LEAGUES = ["E0", "SP1", "F1", "I1", "D1"]

# Minimum shots-on-target to consider for conversion rate estimation
MIN_SOT = 5


def estimate_league_conversion_rate(df: pd.DataFrame) -> float:
    """Estimate the league-average conversion rate (goals / SOT).

    Uses all available historical data to compute a stable per-league rate.
    """
    all_sot = pd.concat([df["home_shots_target"], df["away_shots_target"]]).dropna()
    all_goals = pd.concat([df["home_goals"], df["away_goals"]]).dropna()
    total_sot = all_sot.sum()
    total_goals = all_goals.sum()
    if total_sot <= 0:
        return 0.25  # fallback
    return min(total_goals / total_sot, 0.50)  # cap at 50%


def estimate_league_overdispersion(df: pd.DataFrame) -> float:
    """Estimate overdispersion (variance/mean ratio) for the league.

    Used to add realistic noise to xG estimates.
    Uses a combined DataFrame approach to ensure same-length arrays.
    """
    # Build combined DataFrame from home and away rows
    home = df[["home_shots_target", "home_goals"]].rename(
        columns={"home_shots_target": "sot", "home_goals": "goals"}
    )
    away = df[["away_shots_target", "away_goals"]].rename(
        columns={"away_shots_target": "sot", "away_goals": "goals"}
    )
    combined = pd.concat([home, away], ignore_index=True).dropna()

    if len(combined) < MIN_SOT or combined["sot"].std() == 0:
        return 1.2  # default overdispersion

    # Group by SOT value and compute variance/mean ratio
    grouped = combined.groupby("sot")["goals"].agg(["mean", "var"]).dropna()
    if len(grouped) < 3 or grouped["mean"].sum() == 0:
        return 1.2
    ratio = (grouped["var"] / grouped["mean"]).median()
    return max(ratio, 0.8)  # floor at 0.8


def estimate_xg_for_league(league: str, df: pd.DataFrame) -> pd.DataFrame:
    """Estimate xG from shots-on-target for all matches in a league.

    Uses a simple Poisson model: xG = SOT × conversion_rate
    where conversion_rate = total_goals / total_shots_on_target for the league.

    Also adds realistic variance based on league overdispersion.

    Returns a copy of df with home_xg and away_xg filled in.
    """
    df = df.copy()
    conv_rate = estimate_league_conversion_rate(df)
    overdispersion = estimate_league_overdispersion(df)

    logger.info(
        "  League %s: conversion_rate=%.4f, overdispersion=%.2f",
        league, conv_rate, overdispersion,
    )

    # Estimate xG from SOT
    home_sot = df["home_shots_target"].fillna(0).values
    away_sot = df["away_shots_target"].fillna(0).values

    home_xg = home_sot * conv_rate
    away_xg = away_sot * conv_rate

    # Add small amount of noise proportional to overdispersion
    rng = np.random.default_rng(42)
    noise_scale = overdispersion * conv_rate * 0.1
    home_xg = home_xg + rng.normal(0, noise_scale, size=len(home_xg))
    away_xg = away_xg + rng.normal(0, noise_scale, size=len(away_xg))

    # Clip to realistic bounds
    home_xg = np.clip(home_xg, 0.0, 8.0)
    away_xg = np.clip(away_xg, 0.0, 8.0)

    # Round to 3 decimal places
    df["home_xg"] = np.round(home_xg, 3)
    df["away_xg"] = np.round(away_xg, 3)

    return df


def update_db(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Write estimated xG values back to the database."""
    updated = 0
    for _, row in df.iterrows():
        hxg = float(row["home_xg"])
        axg = float(row["away_xg"])
        cur = conn.execute(
            """UPDATE matches
               SET home_xg = ?, away_xg = ?
               WHERE league = ?
                 AND date = ?
                 AND home_team = ?
                 AND away_team = ?
                 AND (home_xg IS NULL OR away_xg IS NULL)""",
            (hxg, axg, row["league"], row["date"], row["home_team"], row["away_team"]),
        )
        updated += cur.rowcount
    return updated


def get_league_conversion_rates() -> dict[str, float]:
    """Compute conversion rates for all top 5 leagues."""
    conn = sqlite3.connect(str(DB_PATH))
    rates = {}
    for league in TOP_LEAGUES:
        df = pd.read_sql_query(
            """SELECT home_goals, away_goals, home_shots_target, away_shots_target
               FROM matches
               WHERE league = ?
                 AND home_goals IS NOT NULL
                 AND home_shots_target IS NOT NULL""",
            conn, params=(league,),
        )
        if len(df) > MIN_SOT:
            rate = estimate_league_conversion_rate(df)
            rates[league] = round(rate, 4)
    conn.close()
    return rates


def train_xg_strength_model(league: str):
    """Train xG strength model for a league (requires home_xg/away_xg in DB)."""
    from scripts.train_xg_model import load_matches_with_real_xg, fit_xg_strength_model

    logger.info("  Training xG strength model for %s...", league)
    df = load_matches_with_real_xg(league)
    if len(df) < 50:
        logger.warning("  Only %d matches with xG data — need 50+, skipping xG model", len(df))
        return None

    model = fit_xg_strength_model(df)

    import joblib
    league_dir = MODELS_DIR / league
    league_dir.mkdir(parents=True, exist_ok=True)
    model_path = league_dir / "xg_strength_model.joblib"
    joblib.dump(model, model_path)
    logger.info("  xG strength model saved to %s (%d matches, %d teams)",
                 model_path, model.n_matches, len(model.team_list))
    return model


def main():
    parser = argparse.ArgumentParser(description="Estimate xG from shots-on-target")
    parser.add_argument("--league", "-l", help="Single league code")
    parser.add_argument("--all", action="store_true", help="Process all top 5 leagues")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip DB update, only train xG strength model")
    parser.add_argument("--show-rates", action="store_true",
                        help="Show conversion rates for all leagues then exit")
    args = parser.parse_args()

    if args.show_rates:
        rates = get_league_conversion_rates()
        print("\n  League Conversion Rates (goals / SOT):")
        print("  " + "=" * 40)
        for league, rate in sorted(rates.items()):
            print(f"    {league}: {rate:.4f}")
        print()
        return

    leagues = []
    if args.all:
        leagues = TOP_LEAGUES
    elif args.league:
        leagues = [args.league.upper()]
    else:
        parser.print_help()
        return

    conn = sqlite3.connect(str(DB_PATH))

    try:
        for league in leagues:
            print(f"\n  {'='*50}")
            print(f"  League: {league}")
            print(f"  {'='*50}")

            # Load matches with shots data
            df = pd.read_sql_query(
                """SELECT date, home_team, away_team, home_goals, away_goals, result,
                          home_shots, away_shots, home_shots_target, away_shots_target,
                          home_xg, away_xg, league
                   FROM matches
                   WHERE league = ?
                     AND home_goals IS NOT NULL
                  ORDER BY date ASC""",
                conn, params=(league,),
            )

            logger.info("  Loaded %d matches for %s", len(df), league)

            # Check how many already have xG
            has_xg = (df["home_xg"].notna() | df["away_xg"].notna()).sum()
            logger.info("  Matches already with xG: %d", has_xg)

            if not args.train_only and has_xg < len(df):
                # Estimate xG for rows that don't have it
                missing_mask = df["home_xg"].isna() & df["away_xg"].isna()
                missing_df = df[missing_mask].copy()
                logger.info("  Estimating xG for %d matches without xG...", len(missing_df))

                if len(missing_df) > 0:
                    estimated = estimate_xg_for_league(league, missing_df)
                    updated = update_db(conn, estimated)
                    conn.commit()
                    logger.info("  DB updated: %d rows", updated)
            elif args.train_only:
                logger.info("  Skipping DB update (--train-only)")

            # Train xG strength model
            train_xg_strength_model(league)
    finally:
        conn.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
