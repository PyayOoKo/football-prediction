"""
train_xg_model.py — Train an xG-based team strength model for SE1 using
real SofaScore xG data.

Trains attack/defence parameters via quasi-Poisson regression on observed xG
values (home_xg, away_xg) from SofaScore. The resulting model estimates
expected goals based on chance creation rather than actual results.

Usage
-----
    python scripts/train_xg_model.py --league SE1

Saves model to:  models/per_league/{league}/xg_strength_model.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xg_strength import XGStrengthModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_xg_model")

DB_PATH = Path("data/football_data.db")
MODELS_DIR = Path("models/per_league")


def fit_xg_strength_model(
    matches: pd.DataFrame,
    decay_halflife_days: float = 730.0,
    prior_strength: float = 0.02,
) -> XGStrengthModel:
    """Fit quasi-Poisson regression model on real xG data.

    Loss:  λ - xG * ln(λ) + 0.5 * prior_strength * (α² + β²)
    This is Poisson NLL without the ln(Γ(xG+1)) constant term (which
    does not affect optimisation).
    """
    df = matches.copy()
    all_teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    n_teams = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    home_idx = np.array([team_to_idx[t] for t in df["home_team"]], dtype=int)
    away_idx = np.array([team_to_idx[t] for t in df["away_team"]], dtype=int)
    home_xg = df["home_xg"].values.astype(float)
    away_xg = df["away_xg"].values.astype(float)

    # Recency weights
    dates = pd.to_datetime(df["date"])
    ref_date = dates.max() + pd.Timedelta(days=1)
    days_ago = (ref_date - dates).dt.days.clip(lower=0)
    if decay_halflife_days > 0:
        weights = np.exp(-np.log(2) * days_ago / decay_halflife_days)
    else:
        weights = np.ones(len(df))
    weights = np.clip(weights, 1e-6, None)

    n_est = n_teams - 1
    n_total = 2 * n_est + 1  # α, β, γ (no ρ)

    def _build_full(est: np.ndarray):
        alpha_full = np.zeros(n_teams)
        beta_full = np.zeros(n_teams)
        alpha_full[1:] = est[:n_est]
        beta_full[1:] = est[n_est:2 * n_est]
        return alpha_full, beta_full

    def quasi_poisson_nll(est: np.ndarray) -> float:
        alpha_full, beta_full = _build_full(est)
        gamma = est[-1]

        lam_h = np.exp(alpha_full[home_idx] + beta_full[away_idx] + gamma)
        lam_a = np.exp(alpha_full[away_idx] + beta_full[home_idx])

        nll_h = lam_h - home_xg * np.log(np.clip(lam_h, 1e-12, None))
        nll_a = lam_a - away_xg * np.log(np.clip(lam_a, 1e-12, None))

        loss = float(np.sum(weights * (nll_h + nll_a)))

        prior = prior_strength * float(
            np.sum(alpha_full ** 2) + np.sum(beta_full ** 2)
        )
        return loss + prior

    # Initial values
    rng = np.random.default_rng(42)
    x0 = np.zeros(n_total)
    x0[:n_est] = rng.normal(0, 0.1, size=n_est)
    x0[n_est:2 * n_est] = rng.normal(0, 0.1, size=n_est)
    x0[-1] = 0.05

    bounds = [(-3.0, 3.0)] * n_est + [(-3.0, 3.0)] * n_est + [(-2.0, 2.0)]

    logger.info(
        "Fitting xG strength model: %d teams, %d matches (%.0f with xG)",
        n_teams, len(df), len(df),
    )
    result = minimize(
        quasi_poisson_nll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-8, "gtol": 1e-6},
    )

    logger.info(
        "Optimisation %s: neg-LL=%.1f",
        "converged" if result.success else "partial",
        result.fun,
    )

    alpha_full = np.zeros(n_teams)
    beta_full = np.zeros(n_teams)
    alpha_full[1:] = result.x[:n_est]
    beta_full[1:] = result.x[n_est:2 * n_est]

    model = XGStrengthModel(
        alpha={t: float(alpha_full[i]) for i, t in enumerate(all_teams)},
        beta={t: float(beta_full[i]) for i, t in enumerate(all_teams)},
        gamma=float(result.x[-1]),
        team_list=all_teams,
        fitted=True,
        n_matches=len(df),
    )
    return model


def load_matches_with_real_xg(league: str) -> pd.DataFrame:
    """Load matches that have real xG data (from SofaScore)."""
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_xg, away_xg, season
        FROM matches
        WHERE league = ?
          AND home_goals IS NOT NULL
          AND home_xg IS NOT NULL
          AND home_shots IS NOT NULL  -- real xG marker
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    logger.info("Loaded %d matches with real xG for %s", len(df), league)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train xG-based strength model")
    parser.add_argument("--league", "-l", default="SE1", help="League code")
    args = parser.parse_args()

    league = args.league.upper()

    df = load_matches_with_real_xg(league)
    if len(df) < 50:
        logger.error("Need at least 50 matches with real xG, got %d", len(df))
        sys.exit(1)

    model = fit_xg_strength_model(df)

    # Save
    league_dir = MODELS_DIR / league
    league_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    model_path = league_dir / "xg_strength_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved xG strength model to %s", model_path)

    # Save metadata
    meta_path = league_dir / "xg_model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "league": league,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_matches": model.n_matches,
            "n_teams": len(model.team_list),
            "gamma": round(model.gamma, 4),
            "decay_halflife_days": 730.0,
            "prior_strength": 0.02,
        }, f, indent=2)
    logger.info("Saved xG metadata to %s", meta_path)

    # Quick sanity check: show expected goals for a few matches
    print(f"\n  xG Strength Model for {league} — {model.n_matches} matches, {len(model.team_list)} teams")
    print(f"  Home advantage gamma = {model.gamma:.4f}")
    print()
    print("  Top 10 attack strengths:")
    top_attack = sorted(model.alpha.items(), key=lambda x: -x[1])[:10]
    for team, val in top_attack:
        print(f"    {team:<30s} alpha = {val:+.4f}")

    print()
    print("  Top 10 defence weaknesses (higher = more goals conceded):")
    top_defence = sorted(model.beta.items(), key=lambda x: -x[1])[:10]
    for team, val in top_defence:
        print(f"    {team:<30s} beta = {val:+.4f}")

    # Show expected goals for next fixtures
    conn = sqlite3.connect(str(DB_PATH))
    fixtures = pd.read_sql_query(
        """SELECT date, home_team, away_team
           FROM matches
           WHERE league = ? AND home_goals IS NULL
           ORDER BY date ASC
           LIMIT 10""",
        conn, params=(league,),
    )
    conn.close()

    if not fixtures.empty:
        print("\n  Expected goals for upcoming fixtures:")
        for _, row in fixtures.iterrows():
            lam, mu = model.expected_goals(row["home_team"], row["away_team"])
            print(f"    {row['home_team']:<30s} vs {row['away_team']:<30s}  {lam:.2f} - {mu:.2f}")


if __name__ == "__main__":
    main()
