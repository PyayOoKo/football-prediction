"""
compare_backtest_before_after.py — Compare backtest performance:
  OLD: min_ev=0.0, no max_odds, no hybrid cal, no progressive Kelly
  NEW: min_ev=0.05, max_odds=30.0, hybrid cal, progressive Kelly

Uses compute_value_bets() directly (which has all fixes properly implemented)
rather than monkey-patching the BacktestEngine.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_backtest_simulation(
    model_probs: np.ndarray,
    odds_array: np.ndarray,
    y_true: np.ndarray,
    team_matches: list,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_ev: float = 0.0,
    max_odds: float | None = None,
    label: str = "",
) -> dict:
    """Run a value-betting simulation using compute_value_bets.

    Walks through matches chronologically, evaluates value bets,
    places Kelly-sized stakes, and tracks bankroll.

    Returns dict with all metrics.
    """
    from src.value_betting import compute_value_bets

    current_bankroll = bankroll
    bankroll_history = [current_bankroll]
    bet_history: list[dict] = []
    peak = bankroll

    n_matches = len(odds_array)

    for i in range(n_matches):
        match_odds = odds_array[i]
        match_probs = model_probs[i]

        # Skip matches with NaN odds
        if np.any(np.isnan(match_odds)):
            bankroll_history.append(current_bankroll)
            continue

        # Compute value bets for a single match
        result = compute_value_bets(
            odds=[match_odds.tolist()],
            model_probs=[match_probs.tolist()],
            team_matches=[team_matches[i]] if team_matches and i < len(team_matches) else None,
            bankroll=current_bankroll,
            kelly_fraction=kelly_fraction,
            min_ev=min_ev,
            max_odds=max_odds,
        )

        # Check if we got a value bet
        pos = result[result["positive_ev"]]

        if len(pos) > 0:
            # Take the highest-EV bet for this match
            best = pos.iloc[0]
            outcome_idx = {"A": 0, "D": 1, "H": 2}.get(best["outcome"], -1)
            decimal_odds = best["decimal_odds"]
            kelly_pct = best["kelly_pct"]
            stake = best["kelly_stake"]

            if stake > 0 and outcome_idx >= 0 and outcome_idx < 3:
                actual_idx = int(y_true[i])
                won = outcome_idx == actual_idx

                if won:
                    profit = stake * (decimal_odds - 1.0)
                else:
                    profit = -stake

                current_bankroll += profit
                if current_bankroll > peak:
                    peak = current_bankroll

                bet_history.append({
                    "match_idx": i,
                    "match": best["match"],
                    "outcome": best["outcome_label"],
                    "odds": decimal_odds,
                    "prob": best["model_prob"],
                    "ev": best["ev"],
                    "stake": stake,
                    "stake_pct": kelly_pct,
                    "profit": profit,
                    "won": won,
                    "bankroll": current_bankroll,
                })

        bankroll_history.append(current_bankroll)

    # ── Compute metrics ──────────────────────────────
    total_bets = len(bet_history)
    if total_bets == 0:
        return {
            "label": label,
            "total_bets": 0,
            "winning_bets": 0,
            "win_rate_pct": 0.0,
            "total_profit": 0.0,
            "roi_pct": 0.0,
            "yield_pct": 0.0,
            "final_bankroll": current_bankroll,
            "max_drawdown_pct": 0.0,
            "avg_odds": 0.0,
            "avg_ev": 0.0,
            "profit_factor": 0.0,
            "longest_win_streak": 0,
            "longest_lose_streak": 0,
            "bankroll_history": bankroll_history,
            "bet_history": [],
        }

    winning = [b for b in bet_history if b["won"]]
    losing = [b for b in bet_history if not b["won"]]
    total_staked = sum(b["stake"] for b in bet_history)
    total_profit = sum(b["profit"] for b in bet_history)

    roi_pct = ((current_bankroll - bankroll) / bankroll) * 100 if bankroll > 0 else 0.0
    yield_pct = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (len(winning) / total_bets) * 100
    avg_odds = float(np.mean([b["odds"] for b in bet_history]))
    avg_ev = float(np.mean([b["ev"] for b in bet_history]))

    gross_profit = sum(b["profit"] for b in bet_history if b["profit"] > 0)
    gross_loss = abs(sum(b["profit"] for b in bet_history if b["profit"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    dd_peak = bankroll_history[0]
    max_dd = 0.0
    for v in bankroll_history:
        if v > dd_peak:
            dd_peak = v
        dd = (dd_peak - v) / dd_peak * 100 if dd_peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Streaks
    longest_win = 0
    longest_loss = 0
    current_win = 0
    current_loss = 0
    for b in bet_history:
        if b["won"]:
            current_win += 1
            current_loss = 0
            longest_win = max(longest_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            longest_loss = max(longest_loss, current_loss)

    return {
        "label": label,
        "total_bets": total_bets,
        "winning_bets": len(winning),
        "win_rate_pct": round(win_rate, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi_pct, 2),
        "yield_pct": round(yield_pct, 2),
        "final_bankroll": round(current_bankroll, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_odds": round(avg_odds, 2),
        "avg_ev": round(avg_ev, 4),
        "profit_factor": round(profit_factor, 2),
        "longest_win_streak": longest_win,
        "longest_lose_streak": longest_loss,
        "bankroll_history": bankroll_history,
        "bet_history": bet_history,
    }


def main() -> int:
    print("=" * 90)
    print("  BACKTEST COMPARISON: OLD vs NEW PROFITABILITY FIXES".center(88))
    print("=" * 90)

    # Use league_all.csv which has actual bookmaker odds columns
    from config import config

    data_path = config.paths.raw / "league_all.csv"
    if not data_path.exists():
        print(f"\n  [X] Data not found at {data_path}")
        return 1

    # ── 1. Load & prepare data ──────────────────────────
    print(f"\n  Loading data ...")
    df = pd.read_csv(data_path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    print(f"  [OK] {len(df):,} rows x {len(df.columns)} columns")

    # Detect odds columns — league_all.csv has BbAvH / BbAvD / BbAvA
    odds_candidates = [
        ("bbavh", "bbavd", "bbava"),
        ("b365h", "b365d", "b365a"),
        ("psh", "psd", "psa"),
        ("bwh", "bwd", "bwa"),
    ]
    # Normalise column names to lowercase for matching
    col_lower = {c.lower(): c for c in df.columns}

    odds_cols = None
    for cand in odds_candidates:
        if all(c in col_lower for c in cand):
            actual = tuple(col_lower[c] for c in cand)
            non_null = df[list(actual)].notna().all(axis=1).mean()
            if non_null > 0.5:
                odds_cols = actual
                break
    if odds_cols is None:
        print("  [X] No odds columns found.")
        available = [c for c in df.columns if any(x in c.lower() for x in ['bb', 'b3', 'b365', 'bw', 'ps', 'odd'])]
        print(f"  Available betting columns: {available[:15]}")
        return 1
    print(f"  [OK] Using odds columns: {odds_cols}")

    # ── 2. Build features ───────────────────────────────
    # ── 2. Map result to numeric target ────────────────
    if "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0})
        # Drop rows with missing results for training
        has_result = df["target"].notna()
        print(f"  [OK] {has_result.sum():,} rows with results ({len(df) - has_result.sum():,} without)")
    else:
        print("  [X] No 'result' column in data")
        return 1

    # ── 3. Build features ───────────────────────────────
    from src.feature_engineering import build_features, train_val_test_split

    print(f"\n  Building features ...")
    X, y = build_features(df, is_training=True)
    print(f"  [OK] Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    print(f"\n  Splitting chronologically (70/15/15) ...")
    splits = train_val_test_split(X, y)
    n_total = len(X)
    n_test = len(splits["X_test"])
    n_train = len(splits["X_train"])
    n_val = len(splits["X_val"])
    print(f"  [OK] Train: {n_train:,}  |  Val: {n_val:,}  |  Test: {n_test:,}")

    # build_features sorts by date + home_team internally, so the feature matrix
    # rows are in a specific order. We need to align odds with the sorted data.
    # Create a sorted copy of odds data matching build_features() sorting
    odds_sorted = df[["date", "home_team", "away_team"] + list(odds_cols)].copy()
    odds_sorted.sort_values(["date", "home_team"], inplace=True)
    odds_sorted.reset_index(drop=True, inplace=True)

    # Extract test-set odds & team names (last 15% of sorted data)
    test_start = n_total - n_test
    odds_df = odds_sorted.iloc[test_start:test_start + n_test].copy()
    odds_df["home_team"] = odds_sorted["home_team"].iloc[test_start:test_start + n_test].values
    odds_df["away_team"] = odds_sorted["away_team"].iloc[test_start:test_start + n_test].values
    n_with_odds = odds_df[list(odds_cols)].notna().all(axis=1).sum()
    print(f"  [OK] Extracted odds for {len(odds_df)} test matches ({n_with_odds} with odds)")

    # Team matches for display
    team_matches = list(zip(
        odds_df["home_team"].values,
        odds_df["away_team"].values,
    ))

    # ── 3. Train model once ─────────────────────────────
    from src.train import train_model

    print(f"\n  Training XGBoost on train + val ...")
    X_train_full = pd.concat([splits["X_train"], splits["X_val"]], axis=0)
    y_train_full = pd.concat([splits["y_train"], splits["y_val"]], axis=0)
    model, history = train_model(X_train_full, y_train_full)
    print(f"  [OK] Training log-loss: {history['train_loss'][0]:.4f}")

    # ── 4. Fit HybridTail calibration for NEW runs ──────
    print(f"\n  Fitting HybridTail calibration ...")
    from src.calibration import HybridTailCalibrator
    cal_split = int(len(X_train_full) * 0.8)
    X_cal = X_train_full.iloc[cal_split:].copy()
    y_cal = y_train_full.iloc[cal_split:].copy()
    X_fit = X_train_full.iloc[:cal_split].copy()
    y_fit = y_train_full.iloc[:cal_split].copy()

    cal_model, _ = train_model(X_fit, y_fit)
    cal = HybridTailCalibrator(n_classes=3)
    cal.fit(cal_model.predict_proba(X_cal), y_cal.values)
    print(f"  [OK] HybridTail calibrator fitted on {len(y_cal)} validation samples")

    # ── 5. Get predictions ──────────────────────────────
    print(f"\n  Generating predictions ...")
    raw_probs = model.predict_proba(splits["X_test"])  # [away, draw, home]
    cal_probs = cal.transform(raw_probs)

    # Build odds array (column order: away_odds, draw_odds, home_odds for compute_value_bets)
    # League data columns are: home_odds, draw_odds, away_odds (H, D, A)
    odds_array = odds_df[list(odds_cols)].values.astype(float)
    # Reorder to [away_odds, draw_odds, home_odds] if cols are [home, draw, away]
    # BbAvH, BbAvD, BbAvA -> need reorder
    # Reorder odds from (home, draw, away) to (away, draw, home) for compute_value_bets
    # The odds columns are in (home_odds, draw_odds, away_odds) order
    # compute_value_bets expects (away_odds, draw_odds, home_odds)
    odds_array = odds_array[:, [2, 1, 0]]
    print(f"  [OK] Reordered odds from (H,D,A) to (A,D,H) for compute_value_bets")
    y_test = splits["y_test"].values

    print(f"  [OK] {len(raw_probs)} test predictions generated")

    # ── 6. Run simulations ──────────────────────────────
    simulations = [
        ("OLD (min_ev=0.0, platt cal)", dict(min_ev=0.0, max_odds=None)),
        ("NEW (min_ev=0.05, max_odds=30)", dict(min_ev=0.05, max_odds=30.0)),
        ("NEW+HYBRID (hybrid cal + all fixes)", dict(min_ev=0.05, max_odds=30.0)),
    ]

    results = []
    for sim_label, sim_cfg in simulations:
        use_cal = "HYBRID" in sim_label
        probs = cal_probs if use_cal else raw_probs

        print(f"\n  Running: {sim_label} ...", end=" ", flush=True)
        r = run_backtest_simulation(
            model_probs=probs,
            odds_array=odds_array,
            y_true=y_test,
            team_matches=team_matches,
            bankroll=1000.0,
            kelly_fraction=0.25,
            **sim_cfg,
        )
        results.append(r)
        direction = "+" if r["roi_pct"] > 0 else "-" if r["roi_pct"] < 0 else "="
        print(f"ROI: {direction} {abs(r['roi_pct']):.2f}%  |  "
              f"Profit: ${r['total_profit']:+.2f}  |  "
              f"Bets: {r['total_bets']}  |  "
              f"Win: {r['win_rate_pct']:.1f}%")

    # ── 7. Comparison table ─────────────────────────────
    print(f"\n{'=' * 90}")
    print("  COMPARISON: OLD vs NEW vs NEW+HYBRID".center(88))
    print(f"{'=' * 90}")

    _print_comparison_table(results)

    # ── 8. Summary verdict ──────────────────────────────
    print(f"\n{'=' * 90}")
    print("  VERDICT".center(88))
    print(f"{'=' * 90}")
    if len(results) >= 3:
        old = results[0]
        best = max(results, key=lambda r: r["roi_pct"])
        print(f"\n  Best configuration: {best['label']}")
        print(f"  ROI improvement:    {old['roi_pct']:.2f}% → {best['roi_pct']:.2f}%")
        dd_change = old["max_drawdown_pct"] - best["max_drawdown_pct"]
        print(f"  Drawdown change:   {old['max_drawdown_pct']:.1f}% → {best['max_drawdown_pct']:.1f}%"
              f" ({'✅ improved' if dd_change > 0 else '❌ worsened' if dd_change < 0 else 'same'})")
        print(f"  Bets filtered out: {old['total_bets']} → {best['total_bets']}"
              f" ({'✅ fewer extreme odds' if best['total_bets'] < old['total_bets'] else ''})")
        print(f"  Avg odds:          {old['avg_odds']:.1f}x → {best['avg_odds']:.1f}x"
              f" ({'✅ healthier' if best['avg_odds'] < old['avg_odds'] else ''})")
    print(f"\n{'=' * 90}")
    print("  COMPARISON COMPLETE".center(88))
    print(f"{'=' * 90}")

    return 0


def _print_comparison_table(results: list[dict]) -> None:
    """Print side-by-side comparison of all simulation results."""
    headers = ["Metric"] + [r["label"] for r in results]

    # Find improvement column (last - first)
    def _improvement(old_val, new_val, better="higher"):
        try:
            if better == "higher":
                delta = ((new_val - old_val) / abs(old_val) * 100) if old_val != 0 else float("inf")
            else:
                delta = ((old_val - new_val) / abs(old_val) * 100) if old_val != 0 else float("inf")
            if delta == float("inf"):
                return "NEW"
            return f"{delta:+.1f}%"
        except:
            return "—"

    # Header
    header_str = f"{'Metric':<24}"
    for r in results:
        header_str += f" {r['label'][:18]:>18}"
    header_str += " % Change"
    print(f"\n  {header_str}")
    print(f"  {'-' * (24 + 19 * len(results) + 12)}")

    rows = [
        ("Total bets", "total_bets", "higher"),
        ("Win rate %", "win_rate_pct", "higher"),
        ("ROI %", "roi_pct", "higher"),
        ("Yield %", "yield_pct", "higher"),
        ("Profit $", "total_profit", "higher"),
        ("Final bankroll", "final_bankroll", "higher"),
        ("Max DD %", "max_drawdown_pct", "lower"),
        ("Avg odds", "avg_odds", "lower"),
        ("Avg EV", "avg_ev", "higher"),
        ("Profit factor", "profit_factor", "higher"),
        ("Worst streak", "longest_lose_streak", "lower"),
    ]

    for name, key, better in rows:
        vals = [r.get(key, 0) for r in results]
        row_str = f"{name:<24}"
        for v in vals:
            if isinstance(v, float):
                row_str += f" {v:>18.2f}"
            else:
                row_str += f" {str(v):>18}"
        if len(vals) >= 2:
            row_str += f" {_improvement(vals[0], vals[-1], better):>9}"
        print(f"  {row_str}")

    # Highlight note
    print(f"\n  * % Change = (last - first) / |first| * 100")
    print(f"    Positive = improvement (for metrics where 'higher' is better)")
    print(f"    Negative = improvement (for metrics where 'lower' is better)")


if __name__ == "__main__":
    sys.exit(main())
