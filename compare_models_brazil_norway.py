"""
compare_models_brazil_norway.py — Compare XGBoost vs Poisson Model for Brazil vs Norway.

Loads historical World Cup data, fits the Poisson model, and compares
its predictions with the trained XGBoost model's output.

Usage:
    python compare_models_brazil_norway.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# Import project Poisson model
sys.path.insert(0, str(PROJECT_ROOT))
from src.poisson_model import PoissonModel


def load_historical_data() -> pd.DataFrame:
    """Load all World Cup matches (2002-2026) for Poisson fitting."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter to completed matches (have score data)
    df = df[df["home_goals"].notna() & (df["home_goals"] != "")].copy()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Loaded {len(df):,} completed World Cup matches (2002-2026)")
    return df


def load_xgboost_prediction() -> dict:
    """Load the XGBoost model's prediction for Brazil vs Norway."""
    csv_path = PROJECT_ROOT / "reports" / "predictions_worldcup" / "worldcup_predictions.csv"
    df = pd.read_csv(csv_path)

    bra_row = df[df["home_team"] == "Brazil"].iloc[0]
    return {
        "home_team": bra_row["home_team"],
        "away_team": bra_row["away_team"],
        "home_win_prob": bra_row["home_win_prob"],
        "draw_prob": bra_row["draw_prob"],
        "away_win_prob": bra_row["away_win_prob"],
        "confidence": bra_row["confidence"],
    }


def analyze_team_strengths(model: PoissonModel) -> None:
    """Print team strengths for Brazil and Norway relative to the field."""
    strengths = model.team_strengths

    print(f"\n  ── TEAM STRENGTHS (attack α, defense β) ──")
    print(f"  {'Team':<20} {'Attack (α)':>12} {'Defense (β)':>14} {'Rating':>10}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 14} {'─' * 10}")

    # Sort all teams by "rating" (attack / defense — higher is better)
    rated = [(team, α, β, α / β) for team, (α, β) in strengths.items()]
    rated.sort(key=lambda x: -x[3])

    # Show top 8 + Brazil + Norway specifically
    shown = set()
    for team, α, β, rating in rated:
        if team in ("Brazil", "Norway") or rating >= rated[7][3]:
            if team not in shown:
                print(f"  {team:<20} {α:>10.3f}   {β:>10.3f}   {rating:>7.2f}")
                shown.add(team)

    # Ensure Norway is shown if not already
    for team, α, β, rating in rated:
        if team == "Norway" and team not in shown:
            print(f"  {team:<20} {α:>10.3f}   {β:>10.3f}   {rating:>7.2f}")
            break


def display_poisson_prediction(model: PoissonModel) -> dict:
    """Get and display Poisson model's full prediction for Brazil vs Norway."""
    result = model.predict("Brazil", "Norway", max_goals=6)

    print(f"\n  ── POISSON MODEL PREDICTION ──")
    print(f"  Match: {result['home_team']} vs {result['away_team']}")
    print(f"  Expected goals: Brazil {result['expected_home_goals']:.3f} — Norway {result['expected_away_goals']:.3f}")
    print(f"  Most likely score: {result['most_likely_score']} ({result['most_likely_prob']:.1%})")
    print(f"")
    print(f"  Outcome probabilities:")
    print(f"    Brazil win:  {result['home_win_prob']:.4f} ({result['home_win_prob']:.1%})")
    print(f"    Draw:        {result['draw_prob']:.4f} ({result['draw_prob']:.1%})")
    print(f"    Norway win:  {result['away_win_prob']:.4f} ({result['away_win_prob']:.1%})")
    print(f"")
    print(f"  Goal markets:")
    print(f"    Over 2.5 goals: {result['over_2_5_prob']:.1%}")
    print(f"    Under 2.5 goals: {result['under_2_5_prob']:.1%}")
    print(f"    BTTS: {result['btts_prob']:.1%}")
    print(f"    BTTS No: {result['btts_no_prob']:.1%}")

    return result


def display_scoreline_table(model: PoissonModel) -> None:
    """Show the top 10 most likely scorelines."""
    table = model.scoreline_table("Brazil", "Norway", max_goals=6)

    print(f"\n  ── TOP 10 MOST LIKELY SCORELINES ──")
    print(f"  {'Score':>8}  {'Probability':>12}  {'Cumulative':>12}")
    print(f"  {'─' * 8}  {'─' * 12}  {'─' * 12}")

    cumulative = 0.0
    for _, row in table.head(10).iterrows():
        cumulative += row["probability"]
        print(f"  {row['scoreline']:>8}  {row['probability']:>10.2%}  {cumulative:>10.2%}")

    print(f"\n  Remaining scorelines: {1 - cumulative:.2%}")


def compare_models(xgb: dict, poisson: dict) -> None:
    """Side-by-side comparison of both models."""
    print(f"\n{'=' * 72}")
    print(f"  MODEL COMPARISON: XGBoost vs Poisson")
    print(f"  Brazil vs Norway — MetLife Stadium, July 5, 2026")
    print(f"{'=' * 72}")

    print(f"\n  {'Metric':<30} {'XGBoost':>14} {'Poisson':>14} {'Diff':>10}")
    print(f"  {'─' * 30} {'─' * 14} {'─' * 14} {'─' * 10}")

    metrics = [
        ("Brazil win prob", xgb["home_win_prob"], poisson["home_win_prob"]),
        ("Draw prob", xgb["draw_prob"], poisson["draw_prob"]),
        ("Norway win prob", xgb["away_win_prob"], poisson["away_win_prob"]),
    ]

    for label, x_val, p_val in metrics:
        diff = (x_val - p_val) * 100
        print(f"  {label:<30} {x_val:>10.1%}   {p_val:>10.1%}   {diff:>+7.1f}pp")

    # Add expected goals comparison
    print(f"\n  {'Metric':<30} {'XGBoost':>14} {'Poisson':>14}")
    print(f"  {'─' * 30} {'─' * 14} {'─' * 14}")
    print(f"  {'Expected Brazil goals':<30} {'—':>14} {poisson['expected_home_goals']:>10.3f}")
    print(f"  {'Expected Norway goals':<30} {'—':>14} {poisson['expected_away_goals']:>10.3f}")
    print(f"  {'Expected total goals':<30} {'—':>14} {poisson['expected_home_goals'] + poisson['expected_away_goals']:>10.3f}")

    print(f"\n  ── ANALYSIS ──")

    # Compare win probs
    bra_diff = (xgb["home_win_prob"] - poisson["home_win_prob"]) * 100
    nor_diff = (xgb["away_win_prob"] - poisson["away_win_prob"]) * 100
    draw_diff = (xgb["draw_prob"] - poisson["draw_prob"]) * 100

    print(f"  Brazil: XGBoost ({xgb['home_win_prob']:.1%}) vs Poisson ({poisson['home_win_prob']:.1%})")
    if bra_diff > 5:
        print(f"    → XGBoost is {bra_diff:.1f}pp MORE confident in Brazil. This suggests the 80+ engineered")
        print(f"      features (form, Elo, rolling stats) see Brazil stronger than the raw historical")
        print(f"      goal averages captured by the simple Poisson model.")
    elif bra_diff < -5:
        print(f"    → Poisson is {abs(bra_diff):.1f}pp MORE confident in Brazil. The raw historical")
        print(f"      goal data suggests Brazil dominates more than the ML model's features indicate.")
    else:
        print(f"    → Both models closely agree on Brazil's chances (within {abs(bra_diff):.1f}pp).")

    print(f"  Norway: XGBoost ({xgb['away_win_prob']:.1%}) vs Poisson ({poisson['away_win_prob']:.1%})")
    if nor_diff > 5:
        print(f"    → XGBoost is {nor_diff:.1f}pp MORE confident in Norway — likely from recent")
        print(f"      form signals or Elo momentum that the Poisson model doesn't capture.")
    elif nor_diff < -5:
        print(f"    → Poisson is {abs(nor_diff):.1f}pp MORE confident in Norway — raw historical rates")
        print(f"      suggest Norway is more dangerous than the ML model's features indicate.")
    else:
        print(f"    → Both models closely agree on Norway's chances (within {abs(nor_diff):.1f}pp).")

    print(f"\n  Draw: XGBoost ({xgb['draw_prob']:.1%}) vs Poisson ({poisson['draw_prob']:.1%})")
    if draw_diff > 3:
        print(f"    → XGBoost sees a draw as much more likely. The feature set may be detecting")
        print(f"      defensive solidity that suppresses goal-scoring.")
    elif draw_diff < -3:
        print(f"    → Poisson sees a draw as more likely. The independence assumption of Poisson")
        print(f"      naturally inflates draw probabilities in closely-matched fixtures.")
    else:
        print(f"    → Both models agree on draw probability (within {abs(draw_diff):.1f}pp).")

    # Over/Under comparison
    print(f"\n  Over/Under:")
    print(f"    Poisson: Over 2.5 = {poisson['over_2_5_prob']:.1%}, Under 2.5 = {poisson['under_2_5_prob']:.1%}")
    print(f"    Market:  Over 2.5 @ 1.68 (59.5% implied), Under 2.5 @ 2.14 (46.7% implied)")
    if poisson['under_2_5_prob'] > 0.50:
        print(f"    → Poisson favours UNDER 2.5 — expected total {poisson['expected_home_goals'] + poisson['expected_away_goals']:.2f} is below the 2.5 threshold.")
    else:
        print(f"    → Poisson favours OVER 2.5 — expected total {poisson['expected_home_goals'] + poisson['expected_away_goals']:.2f} is above the 2.5 threshold.")

    print()


def main() -> int:
    print(f"\n{'#' * 72}")
    print(f"  #  MODEL COMPARISON: Brazil vs Norway")
    print(f"{'#' * 72}")

    # 1. Load XGBoost prediction
    print(f"\n  Loading XGBoost prediction ...")
    xgb = load_xgboost_prediction()
    print(f"  Brazil {xgb['home_win_prob']:.1%} | Draw {xgb['draw_prob']:.1%} | Norway {xgb['away_win_prob']:.1%}")

    # 2. Load historical data and fit Poisson model
    print(f"\n  Fitting Poisson model ...")
    df = load_historical_data()
    model = PoissonModel(min_matches=3)
    model.fit(df)
    print(f"  League averages: μ_home = {model.league_avg_home:.3f}, μ_away = {model.league_avg_away:.3f}")

    # 3. Analyze team strengths
    analyze_team_strengths(model)

    # 4. Poisson prediction
    poisson = display_poisson_prediction(model)

    # 5. Scoreline table
    display_scoreline_table(model)

    # 6. Side-by-side comparison
    compare_models(xgb, poisson)

    return 0


if __name__ == "__main__":
    sys.exit(main())
