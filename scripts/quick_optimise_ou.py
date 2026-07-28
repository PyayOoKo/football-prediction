"""
quick_optimise_ou.py — Fast F1 Over/Under 2.5 parameter optimisation.

Precomputes blend predictions ONCE, then scans all EV/Kelly combos
(24 combinations) without re-precomputing, reducing run time from
~50 min to ~2.5 min for a single league.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("quick_optimise")

# ── Ensure project root is on path ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Paths ──
DB_PATH = _PROJECT_ROOT / "data/football_data.db"
MODELS_DIR = _PROJECT_ROOT / "models/per_league"
CSV_PATH = _PROJECT_ROOT / "data/raw/league_all.csv"

INITIAL_BANKROLL = 10_000.0
BACKTEST_FRAC = 0.15  # last 15% for backtesting


# ── Data loading (same as backtest_ou_btts.py) ──

def load_csv_odds(league: str) -> pd.DataFrame:
    import csv
    rows: list[dict[str, Any]] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("league", "").strip() == league:
                try:
                    hg = float(r["home_goals"])
                    ag = float(r["away_goals"])
                except (ValueError, KeyError):
                    continue
                over_odds_str = (r.get("bbav>2.5") or "").strip()
                under_odds_str = (r.get("bbav<2.5") or "").strip()
                if not over_odds_str:
                    over_odds_str = (r.get("avg>2.5") or "").strip()
                if not under_odds_str:
                    under_odds_str = (r.get("avg<2.5") or "").strip()
                if not over_odds_str or not under_odds_str:
                    continue
                try:
                    over_odds = float(over_odds_str)
                    under_odds = float(under_odds_str)
                except ValueError:
                    continue
                if over_odds <= 1 or under_odds <= 1:
                    continue
                row_out = {
                    "date": r.get("date", "").strip()[:10],
                    "home_team": r.get("home_team", "").strip(),
                    "away_team": r.get("away_team", "").strip(),
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "result": r.get("result", "").strip(),
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                }
                rows.append(row_out)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_db_data(league: str) -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    return df


def load_league_models(league: str) -> dict[str, Any] | None:
    import joblib
    league_dir = MODELS_DIR / league
    dc_path = league_dir / "dixon_coles.joblib"
    elo_path = league_dir / "elo.joblib"
    xgb_path = league_dir / "xgboost.joblib"
    lgb_path = league_dir / "lightgbm.joblib"
    if not dc_path.exists() or not elo_path.exists():
        return None
    models: dict[str, Any] = {
        "dc": joblib.load(dc_path),
        "elo": joblib.load(elo_path),
    }
    if xgb_path.exists():
        models["xgb"] = joblib.load(xgb_path)
    if lgb_path.exists():
        models["lgb"] = joblib.load(lgb_path)
    return models


# ── Single precompute ──

def precompute_blend_probs(backtest_df: pd.DataFrame, all_data_df: pd.DataFrame,
                           models: dict[str, Any]) -> np.ndarray:
    """Build blend once, precompute predictions, return Over 2.5 probabilities array."""
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    cr = ConditionalRates.from_data(all_data_df)
    blend = ThreeModelBlend(
        dc_model=models.get("dc"),
        elo_model=models.get("elo"),
        xgb_model=models.get("xgb"),
        lgb_model=models.get("lgb"),
        conditional_rates=cr,
        historical_df=all_data_df,
    )

    logger.info("  Pre-computing blend predictions for %d matches...", len(backtest_df))
    ppm = blend.precompute(backtest_df, cache_key=f"opt_{len(backtest_df)}")

    w_ou = blend.weights.get("Over2.5", {})
    return blend._blend_binary(ppm, w_ou, "Over2.5")


# ── Fast scan (precomputed probs, no re-precompute) ──

def fast_scan(backtest_df: pd.DataFrame, over_probs: np.ndarray,
              initial_bankroll: float = INITIAL_BANKROLL,
              over_only: bool = False) -> list[dict[str, Any]]:
    """Scan EV/Kelly combos using precomputed Over 2.5 probabilities.

    If over_only=True, only take Over 2.5 value bets (skip Under 2.5 entirely).
    """
    ev_values = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    kelly_values = [0.10, 0.25, 0.50, 1.0]

    results = []
    n = len(backtest_df)

    for min_ev in ev_values:
        for kelly_frac in kelly_values:
            bankroll = initial_bankroll
            total_staked = 0.0
            total_profit = 0.0
            n_bets = 0
            n_over = 0
            n_won = 0
            peak = initial_bankroll
            max_dd = 0.0

            for i in range(n):
                row = backtest_df.iloc[i]
                hg = int(row["home_goals"])
                ag = int(row["away_goals"])
                actual_over = (hg + ag) > 2.5

                model_over_prob = float(over_probs[i])
                over_odds = float(row["over_odds"])

                outcomes = [
                    ("Over 2.5", over_odds, model_over_prob, actual_over),
                ]
                if not over_only:
                    model_under_prob = 1.0 - model_over_prob
                    under_odds = float(row["under_odds"])
                    outcomes.append(
                        ("Under 2.5", under_odds, model_under_prob, not actual_over),
                    )

                for label, odds, model_prob, actual_won in outcomes:
                    if odds <= 1 or model_prob <= 0:
                        continue
                    implied = 1.0 / odds
                    ev = model_prob / implied - 1.0
                    if ev < min_ev:
                        continue
                    if bankroll <= 1.0:
                        continue
                    full_kelly = (model_prob * odds - 1.0) / (odds - 1.0)
                    stake_pct = max(0.0, full_kelly * kelly_frac)
                    if stake_pct <= 0:
                        continue
                    stake_amount = bankroll * stake_pct
                    profit = stake_amount * (odds - 1.0) if actual_won else -stake_amount

                    bankroll += profit
                    total_staked += stake_amount
                    total_profit += profit
                    n_bets += 1
                    if actual_won:
                        n_won += 1
                    if label == "Over 2.5":
                        n_over += 1

                # Track drawdown
                if bankroll > peak:
                    peak = bankroll
                dd = (peak - bankroll) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            yield_pct = (total_profit / total_staked * 100) if total_staked > 0 else 0
            win_rate = n_won / n_bets if n_bets > 0 else 0

            results.append({
                "min_ev": min_ev,
                "kelly_frac": kelly_frac,
                "n_bets": n_bets,
                "n_over": n_over,
                "n_won": n_won,
                "win_rate": round(win_rate, 4),
                "total_staked": round(total_staked, 2),
                "total_profit": round(total_profit, 2),
                "yield_pct": round(yield_pct, 2),
                "max_dd": round(max_dd * 100, 2),
                "final_bankroll": round(bankroll, 2),
            })

    return results


def report(league: str, results: list[dict[str, Any]], league_name: str,
          over_only: bool = False):
    """Print optimisation report."""
    strategy = "Over 2.5 ONLY" if over_only else "Over + Under Combined"
    print()
    print("=" * 80)
    print(f"  OPTIMISATION RESULTS — {league} ({league_name})")
    print(f"  Strategy: {strategy}")
    print("=" * 80)

    # Table header
    print(f"  {'min_ev':>7} {'Kelly':>7} {'Bets':>6} {'WR':>5} {'Yield':>8} {'Profit':>10} {'DD':>6} {'Final':>10}")
    print(f"  {'─'*7} {'─'*7} {'─'*6} {'─'*5} {'─'*8} {'─'*10} {'─'*6} {'─'*10}")

    for r in results:
        if r["n_bets"] == 0:
            continue
        wr_str = f"{r['win_rate']*100:.0f}%"
        yield_str = f"{r['yield_pct']:+.1f}%" if r["yield_pct"] != 0 else "  0.0%"
        profit_str = f"GBP {r['total_profit']:+,.0f}"
        dd_str = f"{r['max_dd']:.0f}%"
        final_str = f"GBP {r['final_bankroll']:,.0f}"
        print(f"  {r['min_ev']:>7.0%} {r['kelly_frac']:>7.0%} {r['n_bets']:>6d} {wr_str:>5s} {yield_str:>8s} {profit_str:>10s} {dd_str:>6s} {final_str:>10s}")

    # Best findings
    viable = [r for r in results if r["n_bets"] >= 20]
    best_profit = max(results, key=lambda r: r["final_bankroll"]) if results else {}
    best_yield = max(viable, key=lambda r: r["yield_pct"]) if viable else best_profit

    def safety_score(r):
        return r["yield_pct"] / max(r["max_dd"], 0.1) if r["n_bets"] >= 10 else 0
    safest = max(results, key=safety_score) if results else {}

    print()
    print(f"  {'─'*50}")
    print(f"  RECOMMENDATIONS")
    print(f"  {'─'*50}")
    if best_profit:
        print(f"  🏆 Best profit:   min_ev={best_profit['min_ev']:.0%} kelly={best_profit['kelly_frac']:.0%} "
              f"→ {best_profit['n_bets']} bets, {best_profit['yield_pct']:+.1f}% yield, "
              f"GBP {best_profit['total_profit']:+,.0f} profit")
    if best_yield and best_yield != best_profit:
        print(f"  ⚡ Best yield:    min_ev={best_yield['min_ev']:.0%} kelly={best_yield['kelly_frac']:.0%} "
              f"→ {best_yield['n_bets']} bets, {best_yield['yield_pct']:+.1f}% yield")
    if safest:
        print(f"  🛡️  Safest:        min_ev={safest['min_ev']:.0%} kelly={safest['kelly_frac']:.0%} "
              f"→ {safest['n_bets']} bets, max DD {safest['max_dd']:.1f}%, "
              f"yield {safest['yield_pct']:+.1f}%")

    if over_only:
        # Compare Combined vs Over-only
        total_profits = [r["total_profit"] for r in results]
        print(f"\n  {'─'*50}")
        print(f"  STRATEGY COMPARISON")
        print(f"  {'─'*50}")
        print(f"  All results with Over 2.5 ONLY strategy")
        print(f"  Range: profit GBP {min(total_profits):+,.0f} to GBP {max(total_profits):+,.0f}")
        print(f"  Best yield: {best_yield['yield_pct']:+.1f}% (@ min_ev={best_yield['min_ev']:.0%}, kelly={best_yield['kelly_frac']:.0%})")


def main():
    import argparse
    from football_data.config import LEAGUE_NAMES

    parser = argparse.ArgumentParser(description="Fast OU parameter optimisation")
    parser.add_argument("--leagues", nargs="+", default=["F1"],
                        help="Leagues to optimise (default: F1)")
    parser.add_argument("--over-only", action="store_true",
                        help="Only take Over 2.5 bets (skip Under 2.5)")
    args = parser.parse_args()

    for league in args.leagues:
        league_name = LEAGUE_NAMES.get(league, league)
        strategy = " (Over 2.5 ONLY)" if args.over_only else ""
        print(f"\n{'─'*60}")
        print(f"  {league} — {league_name}{strategy}")

        models = load_league_models(league)
        if models is None:
            logger.warning("No trained models for %s — skipping", league)
            continue

        csv_df = load_csv_odds(league)
        if len(csv_df) < 50:
            logger.warning("Only %d OU odds rows — need 50+, skipping", len(csv_df))
            continue

        db_df = load_db_data(league)
        if len(db_df) < 100:
            logger.warning("Only %d DB matches — need 100+, skipping", len(db_df))
            continue

        logger.info("CSV OU odds: %d | DB matches: %d", len(csv_df), len(db_df))

        # Merge CSV + DB
        csv_df["_key"] = csv_df["home_team"] + "|" + csv_df["away_team"] + "|" + csv_df["date"].astype(str)
        db_df["_key"] = db_df["home_team"] + "|" + db_df["away_team"] + "|" + db_df["date"].astype(str)

        merged = csv_df.merge(
            db_df[["_key", "date", "home_team", "away_team", "home_goals", "away_goals", "result"]],
            on="_key", suffixes=("", "_db"), how="inner",
        )
        for col in ["date_db", "home_team_db", "away_team_db", "home_goals_db", "away_goals_db", "result_db"]:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True)
        merged = merged.sort_values("date").reset_index(drop=True)
        logger.info("Merged matches: %d", len(merged))

        # Split
        split_idx = int(len(merged) * (1 - BACKTEST_FRAC))
        backtest_df = merged.iloc[split_idx:].copy()
        logger.info("Backtest set: %d matches (last %d%%)", len(backtest_df), int(BACKTEST_FRAC * 100))

        # Add result codes for conditional rates
        all_data = merged.copy()
        all_data["result"] = all_data.apply(
            lambda r: "H" if r["home_goals"] > r["away_goals"]
            else "A" if r["away_goals"] > r["home_goals"]
            else "D", axis=1
        )

        # ⭐ Precompute ONCE
        logger.info("Precomputing blend predictions (once for all param combos)...")
        over_probs = precompute_blend_probs(backtest_df, all_data, models)

        # ⭐ Fast scan
        logger.info("Scanning 24 parameter combinations (over_only=%s)...", args.over_only)
        results = fast_scan(backtest_df, over_probs, over_only=args.over_only)

        report(league, results, league_name, over_only=args.over_only)

    print("\nDone.")


if __name__ == "__main__":
    main()
