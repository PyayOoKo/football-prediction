"""
Feature Validation — range, distribution, NaN, outlier, and consistency checks.

Loads the full feature matrix from ``build_features()`` and runs every
feature through a suite of validation checks. Produces a structured report
and exits with code 0 if all checks pass, 1 otherwise.

Usage::

    python scripts/validate_features.py
    python scripts/validate_features.py --report reports/feature_validation.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate")

DATA_PATH = ROOT / "data" / "processed" / "results_clean.csv"
REPORT_PATH = next(
    (Path(a) for a in sys.argv[1:] if a.startswith("--report=")),
    ROOT / "reports" / "feature_validation.json",
)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


# ═══════════════════════════════════════════════════════════
#  Expected ranges per feature group (prefix / substring)
# ═══════════════════════════════════════════════════════════

RANGE_RULES: list[tuple[str, float, float, str]] = [
    # (column_substring, min, max, description)
    # Elo
    ("Home_Elo", 1000.0, 2000.0, "Elo ratings: typical range 1000–2000"),
    ("Away_Elo", 1000.0, 2000.0, "Elo ratings: typical range 1000–2000"),
    ("Elo_Difference", -1000.0, 1000.0, "Elo diff: max spread ~1000 points"),
    # Rolling stats — points
    ("_points_avg", 0.0, 3.0, "Avg points per match: 0–3"),
    ("_points_total", 0.0, None, "Total points: ≥0"),
    # Rolling stats — goals
    ("_goals_scored_avg", 0.0, 10.0, "Avg goals scored: 0–10"),
    ("_goals_conceded_avg", 0.0, 10.0, "Avg goals conceded: 0–10"),
    ("_goal_diff_avg", -10.0, 10.0, "Avg goal diff: -10 to +10"),
    # Win rates
    ("_win_rate", 0.0, 1.0, "Win rate: 0–1"),
    # Match counts
    ("_matches_this_season", 0, 60, "Matches per season: 0–60"),
    ("_home_matches", 0, None, "Home match count: ≥0"),
    ("_away_matches", 0, None, "Away match count: ≥0"),
    ("h2h_matches_played", 0, 100, "H2H count: 0–100"),
    ("h2h_home_win_rate", 0.0, 1.0, "H2H win rate: 0–1"),
    ("h2h_home_points_avg", 0.0, 3.0, "H2H avg points: 0–3"),
    ("h2h_away_points_avg", 0.0, 3.0, "H2H avg points: 0–3"),
    ("h2h_home_goals_avg", 0.0, 10.0, "H2H avg goals: 0–10"),
    ("h2h_away_goals_avg", 0.0, 10.0, "H2H avg goals: 0–10"),
    ("h2h_total_goals_avg", 0.0, 10.0, "H2H avg total goals: 0–10"),
    # Rest days — can be large between seasons (off-season gaps of years)
    ("h_days_since_last_match", 0, 10000, "Home rest days: 0–10000"),
    ("a_days_since_last_match", 0, 10000, "Away rest days: 0–10000"),
    # Odds — allow 0 (placeholders when no odds data available)
    ("odds_home_", 0.0, 100.0, "Decimal odds: ≥0 (0=no data)"),
    ("odds_draw_", 0.0, 100.0, "Decimal odds: ≥0 (0=no data)"),
    ("odds_away_", 0.0, 100.0, "Decimal odds: ≥0 (0=no data)"),
    ("fair_prob_", 0.0, 1.0, "Fair probability: 0–1"),
    ("clv_", -1.0, 1.0, "Closing line value: typically -1 to +1"),
    ("market_confidence", 0.0, 1.0, "Market confidence: 0–1"),
    ("bookmaker_margin_", 0.0, 1.0, "Bookmaker margin: 0–1"),
    ("odds_movement_", -100.0, 100.0, "Odds movement: unbounded in extreme cases"),
    # xG
    ("_xg_avg", 0.0, 5.0, "xG average: 0–5 per match"),
    ("_xga_avg", 0.0, 5.0, "xG against avg: 0–5"),
    ("_xgd_avg", -5.0, 5.0, "xG difference: -5 to +5"),
    ("Expected_Home_Goals", 0.0, 10.0, "Expected home goals: 0–10"),
    ("Expected_Away_Goals", 0.0, 10.0, "Expected away goals: 0–10"),
    ("Expected_Total_Goals", 0.0, 15.0, "Expected total goals: 0–15"),
    ("Expected_Goal_Difference", -10.0, 10.0, "Expected goal diff: -10 to +10"),
    ("Home_Attack_Strength", 0.0, 6.0, "Attack strength: 0–6"),
    ("Home_Defense_Strength", 0.0, 6.0, "Defense strength: 0–6"),
    ("Away_Attack_Strength", 0.0, 6.0, "Attack strength: 0–6"),
    ("Away_Defense_Strength", 0.0, 6.0, "Defense strength: 0–6"),
    ("_xpts", 0.0, 3.0, "Expected points: 0–3"),
    ("xgd", -10.0, 10.0, "xG difference: -10 to +10"),
    # League positions
    ("_league_position", 1, 80, "League position: 1+"),
    ("_matches_played_league", 0, 60, "Matches played in league: 0–60"),
    ("position_diff", -79, 79, "Position diff between teams"),
    # League averages
    ("league_avg_goals_scored", 0.0, 5.0, "League avg goals scored: 0–5"),
    ("league_avg_goals_conceded", 0.0, 5.0, "League avg goals conceded: 0–5"),
    # Attack/defence ratios
    ("_attack_ratio", 0.0, 10.0, "Attack ratio: 0–10"),
    ("_defence_ratio", 0.0, 10.0, "Defence ratio: 0–10"),
    # Temporal (exact match / careful substrings)
    ("\byear\b", 1990, 2030, "Calendar year"),  # word boundary to avoid day_of_year
    ("month", 1, 12, "Calendar month"),
    ("day_of_week", 0, 6, "Day of week: 0=Mon, 6=Sun"),
    ("day_of_year", 1, 366, "Day of year"),
    ("week_of_season", 1, 60, "Week of season"),
    ("is_midweek", 0, 1, "Midweek flag: binary"),
    ("home_goals_ht", 0, 10, "Half-time home goals: ≥0"),
    ("away_goals_ht", 0, 10, "Half-time away goals: ≥0"),
    ("competition_importance", 0.0, 5.0, "Competition importance: typically 0–5"),
]

# Features that are expected to have a high proportion of NaN (legitimate)
EXPECTED_SPARSE: list[tuple[str, float, str]] = [
    ("h2h_", 0.95, "H2H features: ~92% NaN for teams that never met"),
    ("_xg_", 0.70, "xG features: ~53% NaN when no xG data available"),
    ("_xga_", 0.70, "xG against features: ~53% NaN when no xG data"),
    ("_xgd_", 0.70, "xG diff features: ~53% NaN when no xG data"),
    ("Expected_", 0.70, "Expected goal features: ~53% NaN when no xG data"),
    ("Home_Attack_", 0.70, "Attack strength: ~53% NaN when no xG data"),
    ("Home_Defense_", 0.70, "Defense strength: ~53% NaN when no xG data"),
    ("Away_Attack_", 0.70, "Attack strength: ~53% NaN when no xG data"),
    ("Away_Defense_", 0.70, "Defense strength: ~53% NaN when no xG data"),
    ("_xpts", 0.70, "Expected points: ~53% NaN when no xG data"),
    ("xgd", 0.70, "xG difference: ~53% NaN when no xG data"),
    ("match_id_x", 0.70, "Match ID from xG source: NaN when no xG data"),
    ("match_id_y", 0.70, "Match ID from xG source: NaN when no xG data"),
    ("is_home_x", 0.70, "Is-home flag from xG source: NaN when no xG data"),
    ("is_home_y", 0.70, "Is-home flag from xG source: NaN when no xG data"),
    # Rolling features — NaN for first matches of each team/season
    ("_points_avg", 0.15, "Rolling avg points: NaN at season start (<5 matches)"),
    ("_goals_scored_avg", 0.15, "Rolling avg goals: NaN at season start"),
    ("_goals_conceded_avg", 0.15, "Rolling avg conceded: NaN at season start"),
    ("_goal_diff_avg", 0.15, "Rolling goal diff: NaN at season start"),
    ("_matches_this_season", 0.15, "Match count: 0 for first match of season"),
    ("h_win_rate", 0.15, "Home win rates: NaN before first match"),
    ("a_win_rate", 0.15, "Away win rates: NaN before first match"),
    ("_home_matches", 0.15, "Home match count: NaN before first home match"),
    ("_away_matches", 0.15, "Away match count: NaN before first away match"),
    ("_days_since_last_match", 0.25, "Rest days: NaN for first match of each team"),
    ("_attack_ratio", 0.15, "Attack ratio: NaN at season start"),
    ("_defence_ratio", 0.15, "Defence ratio: NaN at season start"),
    ("h_league_position", 0.15, "League position: NaN before first tracked match"),
    ("a_league_position", 0.15, "League position: NaN before first tracked match"),
    ("h_points_total", 0.15, "Points total: NaN before first tracked match"),
    ("a_points_total", 0.15, "Points total: NaN before first tracked match"),
    ("_matches_played_league", 0.15, "Matches played: NaN before first tracked match"),
    ("position_diff", 0.15, "Position diff: NaN before first tracked match"),
]


# ═══════════════════════════════════════════════════════════
#  Validation logic
# ═══════════════════════════════════════════════════════════

def validate_features(
    X: pd.DataFrame,
    y: pd.Series | None = None,
) -> dict:
    """Run all validation checks. Returns a structured report dict."""
    results: list[dict] = []
    n = len(X)

    # ── 1. Range checks ──────────────────────────────────
    import re
    for substr, vmin, vmax, desc in RANGE_RULES:
        if substr.startswith("\\b"):
            matching = [c for c in X.columns if re.search(substr, c)]
        else:
            matching = [c for c in X.columns if substr in c]

        for col in matching:
            series = X[col].dropna()
            if len(series) == 0:
                results.append(_r(col, WARN, "range", f"All NaN — cannot check range"))
                continue

            actual_min = float(series.min())
            actual_max = float(series.max())
            violations = (series < vmin).sum() + (series > vmax).sum() if vmax is not None else (series < vmin).sum()
            if vmax is None:
                violations = int((series < vmin).sum())

            ok = actual_min >= vmin and (vmax is None or actual_max <= vmax)

            if ok:
                results.append(_r(col, PASS, "range",
                    f"[{actual_min:.2f}, {actual_max:.2f}] ⊆ [{vmin}, {vmax or '∞'}]"))
            else:
                results.append(_r(col, FAIL, "range",
                    f"[{actual_min:.2f}, {actual_max:.2f}] ⊈ [{vmin}, {vmax or '∞'}] — {violations} violations"))

    # ── 2. NaN checks ─────────────────────────────────────
    nan_pct = X.isna().mean()
    for col in X.columns:
        pct = float(nan_pct[col])
        # Check if this column belongs to an expected-sparse group
        expected_max = None
        for substr, max_nan, _ in EXPECTED_SPARSE:
            if substr in col:
                expected_max = max_nan
                break

        if pct == 0.0:
            results.append(_r(col, PASS, "nan", f"0.0% NaN"))
        elif expected_max is not None:
            if pct <= expected_max:
                results.append(_r(col, PASS, "nan",
                    f"{pct*100:.1f}% NaN (within expected ≤{expected_max*100:.0f}% for this group)"))
            else:
                results.append(_r(col, FAIL, "nan",
                    f"{pct*100:.1f}% NaN — exceeds expected {expected_max*100:.0f}% for this group"))
        elif pct > 0.05:
            results.append(_r(col, FAIL, "nan",
                f"{pct*100:.1f}% NaN — exceeds 5% threshold"))

    # ── 3. Distribution checks (rough normality for rolling avgs) ──
    for col in X.columns:
        series = X[col].dropna()
        if len(series) < 10:
            continue
        if any(p in col for p in ["_avg", "_ratio", "Elo", "Expected_", "_xpts"]):
            skew = float(series.skew())
            if abs(skew) > 5:
                results.append(_r(col, WARN, "distribution",
                    f"Highly skewed: skew={skew:.2f} (|skew|>5)"))

    # ── 4. Outlier checks (IQR-based) ─────────────────────
    for col in X.select_dtypes(include=[np.number]).columns:
        series = X[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            # Constant column — check if that makes sense
            if series.nunique() <= 2:
                continue
            results.append(_r(col, WARN, "outlier", f"Zero IQR — constant column"))
            continue
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        outlier_pct = n_outliers / len(series) * 100
        if outlier_pct > 10:
            results.append(_r(col, WARN, "outlier",
                f"{n_outliers} outliers ({outlier_pct:.1f}%) beyond 3×IQR"))
        elif outlier_pct > 0:
            pass

    # ── 5. Consistency checks ─────────────────────────────
    # 5a: league_avg_goals_scored and league_avg_goals_conceded should be similar
    if "league_avg_goals_scored" in X.columns and "league_avg_goals_conceded" in X.columns:
        diff = (X["league_avg_goals_scored"] - X["league_avg_goals_conceded"]).dropna()
        mean_diff = float(diff.mean())
        if abs(mean_diff) > 0.5:
            results.append(_r("league_avg", WARN, "consistency",
                f"Scored vs conceded avg differ by {mean_diff:.3f} (should be near 0)"))

    # 5b: Home_Attack_Strength and Home_Defense_Strength should be roughly inverse
    if "Home_Attack_Strength" in X.columns and "Home_Defense_Strength" in X.columns:
        corr = X[["Home_Attack_Strength", "Home_Defense_Strength"]].dropna().corr().iloc[0, 1]
        if corr > 0.3:
            results.append(_r("Home_Strength", WARN, "consistency",
                f"Attack-Defense correlation = {corr:.3f} (expected weakly negative)"))

    # 5c: Expected_Goal_Difference should approx equal Expected_Home_Goals - Expected_Away_Goals
    if all(c in X.columns for c in ["Expected_Goal_Difference", "Expected_Home_Goals", "Expected_Away_Goals"]):
        err = (X["Expected_Goal_Difference"] - (X["Expected_Home_Goals"] - X["Expected_Away_Goals"])).dropna()
        max_err = float(err.abs().max())
        if max_err > 0.01:
            results.append(_r("Expected_Goal_Difference", WARN, "consistency",
                f"Max internal inconsistency: {max_err:.4f}"))

    # 5d: is_midweek should match day_of_week
    if "is_midweek" in X.columns and "day_of_week" in X.columns:
        expected = X["day_of_week"].isin([1, 2, 3]).astype(int)
        mismatches = (X["is_midweek"] != expected).sum()
        if mismatches > 0:
            results.append(_r("is_midweek", WARN, "consistency",
                f"{mismatches} rows where is_midweek ≠ (day_of_week ∈ {1,2,3})"))

    # Compile summary
    total = len(results)
    passed = sum(1 for r in results if r["status"] == PASS)
    failed = sum(1 for r in results if r["status"] == FAIL)
    warned = sum(1 for r in results if r["status"] == WARN)

    return {
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "warnings": warned,
            "pass_rate": round(passed / max(total, 1), 4),
            "status": PASS if failed == 0 else FAIL,
        },
        "results": results,
        "feature_matrix": {
            "n_rows": X.shape[0],
            "n_columns": X.shape[1],
            "numeric_columns": int(X.select_dtypes(include=[np.number]).shape[1]),
            "categorical_columns": int(X.select_dtypes(exclude=[np.number]).shape[1]),
            "total_nan": int(X.isna().sum().sum()),
            "nan_cells_pct": round(float(X.isna().sum().sum() / (X.shape[0] * X.shape[1]) * 100), 2),
            "infinite_values": int(np.isinf(X.select_dtypes(include=[np.number]).values).sum()),
        },
        "expected_sparse_groups": [
            {"substring": s, "max_nan_pct": m * 100, "description": d}
            for s, m, d in EXPECTED_SPARSE
        ],
    }


def _r(feature: str, status: str, check: str, message: str) -> dict:
    return {
        "feature": feature,
        "status": status,
        "check": check,
        "message": message,
    }


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("  FEATURE VALIDATION")
    print("=" * 72)

    print("\n  Loading data and building features ...")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    X, y = build_features(df, is_training=True)
    log.info("Feature matrix: %s", X.shape)
    print(f"      {X.shape[0]} rows × {X.shape[1]} cols  ({time.time()-t0:.1f}s)")

    print("\n  Running validation checks ...")
    t0 = time.time()
    report = validate_features(X, y)
    elapsed = time.time() - t0

    s = report["summary"]
    print(f"\n  {'CHECK':<10} {'TOTAL':>8} {'PASS':>8} {'FAIL':>8} {'WARN':>8}")
    print(f"  {'─' * 42}")
    print(f"  {'all':<10} {s['total_checks']:>8} {s['passed']:>8} {s['failed']:>8} {s['warnings']:>8}")
    print(f"\n  Pass rate: {s['pass_rate']:.2%}")

    fm = report["feature_matrix"]
    print(f"\n  Matrix: {fm['n_rows']} rows, {fm['n_columns']} cols")
    print(f"  NaN:    {fm['total_nan']} cells ({fm['nan_cells_pct']}%)")
    print(f"  Inf:    {fm['infinite_values']} cells")

    # Show failures
    failures = [r for r in report["results"] if r["status"] == FAIL]
    if failures:
        print(f"\n  ❌ FAILURES ({len(failures)}):")
        for r in failures[:20]:
            print(f"    [{r['check']}] {r['feature']}: {r['message']}")
        if len(failures) > 20:
            print(f"    ... and {len(failures)-20} more")

    # Show warnings
    warnings = [r for r in report["results"] if r["status"] == WARN]
    if warnings:
        print(f"\n  ⚠ WARNINGS ({len(warnings)}):")
        for r in warnings[:15]:
            print(f"    [{r['check']}] {r['feature']}: {r['message']}")
        if len(warnings) > 15:
            print(f"    ... and {len(warnings)-15} more")

    print(f"\n  Validation took {elapsed:.1f}s")

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to {REPORT_PATH}")

    print(f"\n  Status: {s['status']}")
    print("=" * 72)

    return 0 if s["status"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
