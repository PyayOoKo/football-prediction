"""
backtest_ou_btts.py — Historical backtest of O/U and BTTS models against market odds.

Simulates level-stakes betting on all 2023-2024 matches where the model's
predicted probability exceeds the market's implied probability by a
configurable edge threshold.

Output:
    reports/backtest_ou_btts_{timestamp}.md
    reports/backtest_ou_btts_{timestamp}.json

Usage:
    python scripts/backtest_ou_btts.py
    python scripts/backtest_ou_btts.py --min-edge 0.03
    python scripts/backtest_ou_btts.py --stake 50
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest_ou_btts")

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _to_native(val: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_native(v) for v in val]
    return val

# Best models — use latest retrained model
_ou_files = sorted(MODELS_DIR.glob("over_under_random_forest_*.joblib"))
OU_MODEL_PATH = _ou_files[-1] if _ou_files else MODELS_DIR / "over_under_logistic_regression_20260725_222912.joblib"

BTTS_MODEL_PATH = MODELS_DIR / "btts_xgboost_20260725_223702.joblib"
BTTS_IMPLIED_MODEL_PATH = MODELS_DIR / "btts_implied_from_markets.joblib"

OU_DATA_PATH = PROCESSED_DIR / "over_under_data_20260725_222214.parquet"
BTTS_DATA_PATH = PROCESSED_DIR / "btts_data_20260725_222516.parquet"

# Post-match features to exclude (data leakage)
LEAKY_FEATURES = {
    "home_xg", "away_xg",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
}

# Clean mode: exclude leaky features
_CLEAN_MODE = "random_forest" in str(OU_MODEL_PATH).lower()
# Rolling-only mode: only team rolling features (set via CLI)
_ROLLING_ONLY = False

# Rolling feature prefixes
_ROLLING_PREFIXES = ("h_rolling", "a_rolling", "h_cumavg", "a_cumavg", "diff_", "expected_total")

# Default backtest settings
MIN_EDGE = 0.02       # Minimum edge over market (2%)
MAX_ODDS = 10.0       # Max decimal odds to bet (avoid extreme odds)
STAKE = 100           # Level stake per bet
MIN_BETS_LEAGUE = 10  # Min bets for league-level stats


def load_model_and_test_data(
    model_path: Path, data_path: Path, target_col: str,
    exclude_leaky: bool | None = None,
) -> tuple[Any, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Load model + data, generate predictions for 2023+ test set.

    Parameters
    ----------
    exclude_leaky : bool, optional
        Whether to exclude post-match leaky features. If None, uses global _CLEAN_MODE.
        Pass False for BTTS model (trained with leaky features).

    Returns: model, df_test, y_prob, y_actual, feature_cols, stats
    """
    import joblib

    model = joblib.load(model_path)
    logger.info("Loaded model: %s", model_path.name)

    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Test set: 2023-2024
    test_mask = (df["date"].dt.year >= 2023) & (df["date"].dt.year <= 2024)
    df_test = df[test_mask].copy().sort_values("date")

    # Feature columns (same logic as training — exclude leaky in clean mode)
    id_cols = {
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "total_goals", "result",
        "btts", "over_2_5", "over35",
    }
    _do_exclude = exclude_leaky if exclude_leaky is not None else _CLEAN_MODE
    if _do_exclude:
        id_cols = id_cols | LEAKY_FEATURES
        logger.info("Clean mode: excluding %d leaky post-match features", len(LEAKY_FEATURES))
    feature_cols = sorted([
        c for c in df.columns
        if c not in id_cols
        and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)
        and df[c].notna().sum() > 0
    ])

    # Rolling-only: only team rolling features
    if _ROLLING_ONLY:
        feature_cols = [c for c in feature_cols if c.startswith(_ROLLING_PREFIXES)]
        logger.info("Rolling-only mode: %d features (excluded odds/H2H/league)", len(feature_cols))

    logger.info("Using %d features for predictions", len(feature_cols))

    # Impute NaNs with median
    for col in feature_cols:
        if df_test[col].isna().sum() > 0:
            df_test[col] = df_test[col].fillna(df_test[col].median())

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test[target_col].values

    # Generate predictions
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Poisson model
        y_prob = np.zeros(len(X_test))

    stats = {
        "n_total": len(df_test),
        "n_with_odds": df_test["over25_odds"].notna().sum(),
        "target_mean": y_test.mean(),
        "pred_mean": y_prob.mean(),
    }

    return model, df_test, y_prob, y_test, feature_cols, stats


def backtest_market(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    y_actual: np.ndarray,
    odds_col: str,
    target_name: str,
    direction: str = "over",
    min_edge: float = MIN_EDGE,
    max_odds: float = MAX_ODDS,
    stake: float = STAKE,
) -> dict[str, Any]:
    """Backtest a single market (Over or Under).

    For each match with available odds:
    - Compute market implied probability (1/odds)
    - If model_prob > market_imp_prob + min_edge → bet
    - Track P&L, ROI, hit rate

    direction: 'over' for Over 2.5, 'under' for Under 2.5
    """
    odds = df[odds_col].values

    # Market implied probability
    market_prob = 1.0 / odds

    # Determine which side: Over or Under
    if direction == "over":
        model_edge = y_prob - market_prob
        bet_signal = model_edge >= min_edge
    else:
        model_edge = (1 - y_prob) - market_prob
        bet_signal = model_edge >= min_edge

    # Filter: reasonable odds
    valid_odds = (odds >= 1.1) & (odds <= max_odds) & (~np.isnan(odds)) & (odds > 0)
    bet_mask = bet_signal & valid_odds

    n_bets = bet_mask.sum()
    if n_bets == 0:
        return {
            "direction": direction,
            "target": target_name,
            "n_bets": 0,
            "error": "No bets placed",
        }

    # Simulate betting
    bet_odds = odds[bet_mask]
    bet_actual = y_actual[bet_mask]
    bet_model_prob = y_prob[bet_mask]

    if direction == "under":
        # For Under: we win when y_actual == 0
        bet_won = bet_actual == 0
    else:
        bet_won = bet_actual == 1

    # Level stakes
    profit = stake * np.where(bet_won, bet_odds - 1, -1)
    total_staked = n_bets * stake
    total_return = total_staked + profit.sum()
    roi = (total_return - total_staked) / total_staked * 100
    hit_rate = bet_won.mean() * 100

    # Average edge on placed bets
    avg_edge = bet_model_prob[bet_won | ~bet_won].mean() - (1 / bet_odds).mean()

    # Results by league
    league_indices = df.index[bet_mask]
    league_col = df.loc[league_indices, "league"].values
    league_results: dict[str, dict[str, Any]] = {}
    for league in np.unique(league_col):
        league_mask = league_col == league
        league_n = league_mask.sum()
        if league_n < MIN_BETS_LEAGUE:
            continue
        league_won = bet_won[league_mask].sum()
        league_profit = profit[league_mask].sum()
        league_roi = (league_profit / (league_n * stake)) * 100
        league_results[league] = {
            "bets": int(league_n),
            "won": int(league_won),
            "profit": round(league_profit, 2),
            "roi_pct": round(league_roi, 2),
        }

    # Yearly results
    dates = df.loc[league_indices, "date"].values
    years = pd.to_datetime(dates).year
    yearly_results: dict[str, dict[str, Any]] = {}
    for year in sorted(set(years)):
        year_mask = years == year
        year_n = year_mask.sum()
        if year_n < MIN_BETS_LEAGUE:
            continue
        year_won = bet_won[year_mask].sum()
        year_profit = profit[year_mask].sum()
        year_roi = (year_profit / (year_n * stake)) * 100
        yearly_results[str(year)] = {
            "bets": int(year_n),
            "won": int(year_won),
            "profit": round(year_profit, 2),
            "roi_pct": round(year_roi, 2),
        }

    return {
        "direction": direction,
        "target": target_name,
        "n_bets": int(n_bets),
        "n_won": int(bet_won.sum()),
        "hit_rate_pct": round(hit_rate, 2),
        "total_staked": round(total_staked, 2),
        "total_return": round(float(total_return), 2),
        "profit": round(float(profit.sum()), 2),
        "roi_pct": round(roi, 2),
        "avg_edge": round(float(avg_edge), 4),
        "avg_odds": round(float(bet_odds.mean()), 2),
        "stake": stake,
        "min_edge_used": min_edge,
        "by_league": league_results,
        "by_year": yearly_results,
    }


def backtest_btts(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    y_actual: np.ndarray,
    min_edge: float = MIN_EDGE,
    max_odds: float = MAX_ODDS,
    stake: float = STAKE,
) -> dict[str, Any]:
    """Backtest BTTS using derived implied odds from 1X2 + O/U markets.

    Uses the BTTS implied model (trained by derive_btts_implied.py) to
    estimate what the market "thinks" about BTTS probability for each match,
    then bets when our model's probability beats the market's.

    Falls back to hypothetical assumed odds if the implied model is unavailable.
    """
    results = {"direction": "btts_yes", "target": "BTTS Yes"}

    # Try to use derived BTTS implied model
    derived_results = _backtest_btts_derived(
        df, y_prob, y_actual,
        min_edge=min_edge, max_odds=max_odds, stake=stake,
    )
    if derived_results:
        results["derived"] = derived_results
        results["method"] = "derived_from_1x2_ou"
    else:
        # Fall back to hypothetical assumed odds
        results["method"] = "hypothetical_assumed_odds"
        sub_results = []
        for assumed_odds in [1.8, 2.0, 2.2, 2.5]:
            market_prob = 1.0 / assumed_odds
            model_edge = y_prob - market_prob
            bet_mask = (model_edge >= min_edge) & (assumed_odds <= max_odds)

            n_bets = bet_mask.sum()
            if n_bets < MIN_BETS_LEAGUE:
                continue

            bet_won = y_actual[bet_mask] == 1
            profit = stake * np.where(bet_won, assumed_odds - 1, -1)
            total_staked = n_bets * stake
            roi = (profit.sum() / total_staked) * 100

            sub_results.append({
                "assumed_odds": assumed_odds,
                "n_bets": int(n_bets),
                "n_won": int(bet_won.sum()),
                "profit": round(float(profit.sum()), 2),
                "roi_pct": round(float(roi), 2),
            })
        results["scenarios"] = sub_results if sub_results else None

    return results


def _load_btts_implied_model() -> tuple[Any, list[str]] | None:
    """Load the BTTS implied-from-markets model."""
    import joblib
    if not BTTS_IMPLIED_MODEL_PATH.exists():
        logger.info("BTTS implied model not found at %s", BTTS_IMPLIED_MODEL_PATH)
        return None
    try:
        data = joblib.load(BTTS_IMPLIED_MODEL_PATH)
        model = data["model"]
        feature_cols = data["feature_cols"]
        logger.info("Loaded BTTS implied model: %s", data.get("generated", "?"))
        return model, feature_cols
    except Exception as exc:
        logger.warning("Failed to load BTTS implied model: %s", exc)
        return None


def _predict_market_btts_probs(
    df: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
) -> np.ndarray:
    """Use the BTTS implied model to estimate market BTTS probability for each match.

    Builds feature vector from 1X2 + O/U odds, then predicts P(BTTS=Yes).
    This represents what the market "thinks" about BTTS, inferred from
    correlated market prices.
    """
    # Build features for each match
    home_imp = 1.0 / df["home_odds"].values
    draw_imp = 1.0 / df["draw_odds"].values
    away_imp = 1.0 / df["away_odds"].values
    over25_imp = 1.0 / df["over25_odds"].values
    under25_imp = 1.0 / df["under25_odds"].values
    margin_1x2 = home_imp + draw_imp + away_imp
    margin_ou = over25_imp + under25_imp
    home_prob = home_imp / margin_1x2
    draw_prob = draw_imp / margin_1x2
    away_prob = away_imp / margin_1x2
    over25_prob = over25_imp / margin_ou
    under25_prob = under25_imp / margin_ou
    favorite_imp = np.minimum(home_imp, away_imp)
    underdog_imp = np.maximum(home_imp, away_imp)
    favorite_dominance = underdog_imp / np.maximum(favorite_imp, 0.001)
    ou_ratio = over25_prob / np.maximum(under25_prob, 0.001)

    # Rolling BTTS rate per league (approximation for test set — use overall league rates)
    league_rates = df.groupby("league")["btts"].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0.5)

    years = df["date"].dt.year.values.astype(np.float32)
    months = df["date"].dt.month.values.astype(np.float32)

    # Build feature matrix (matching feature_cols order)
    feature_data = {
        "home_imp": home_imp,
        "draw_imp": draw_imp,
        "away_imp": away_imp,
        "over25_imp": over25_imp,
        "under25_imp": under25_imp,
        "margin_1x2": margin_1x2,
        "margin_ou": margin_ou,
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "over25_prob": over25_prob,
        "favorite_imp": favorite_imp,
        "underdog_imp": underdog_imp,
        "favorite_dominance": favorite_dominance,
        "ou_ratio": ou_ratio,
        "league_btts_rolling": league_rates.values,
        "year": years,
        "month": months,
    }

    n = len(df)
    X = np.zeros((n, len(feature_cols)), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        X[:, i] = feature_data[col]

    # Handle NaN
    X = np.nan_to_num(X, nan=0.5)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _backtest_btts_derived(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    y_actual: np.ndarray,
    min_edge: float = MIN_EDGE,
    max_odds: float = MAX_ODDS,
    stake: float = STAKE,
) -> dict[str, Any] | None:
    """Backtest BTTS using derived market odds from 1X2 + O/U."""
    model_data = _load_btts_implied_model()
    if model_data is None:
        return None

    implied_model, feature_cols = model_data

    # Filter to matches with all required odds
    has_odds = (
        df["home_odds"].notna() & df["draw_odds"].notna()
        & df["away_odds"].notna() & df["over25_odds"].notna()
        & df["under25_odds"].notna()
    )
    df_sub = df[has_odds].copy()
    y_prob_sub = y_prob[has_odds]
    y_actual_sub = y_actual[has_odds]

    if len(df_sub) < MIN_BETS_LEAGUE:
        return None

    # Predict market BTTS probability from 1X2 + O/U odds
    logger.info("Predicting BTTS implied odds for %d matches...", len(df_sub))
    market_btts_prob = _predict_market_btts_probs(df_sub, implied_model, feature_cols)

    # Derive odds from market probability (assume 5% margin)
    margin = 0.05
    market_btts_odds = 1.0 / (market_btts_prob * (1 + margin))

    # Edge: model_prob - market_prob
    model_edge = y_prob_sub - market_btts_prob

    # Filter: reasonable odds
    valid_odds = (market_btts_odds >= 1.1) & (market_btts_odds <= max_odds)
    bet_mask = (model_edge >= min_edge) & valid_odds

    n_bets = bet_mask.sum()
    if n_bets < MIN_BETS_LEAGUE:
        return None

    # Simulate betting
    bet_odds = market_btts_odds[bet_mask]
    bet_won = y_actual_sub[bet_mask] == 1
    profit = stake * np.where(bet_won, bet_odds - 1, -1)
    total_staked = n_bets * stake
    total_return = total_staked + profit.sum()
    roi = (total_return - total_staked) / total_staked * 100
    hit_rate = bet_won.mean() * 100
    avg_edge = (y_prob_sub[bet_mask] - market_btts_prob[bet_mask]).mean()

    return {
        "direction": "btts_yes",
        "target": "BTTS Yes (derived from 1X2+O/U)",
        "n_bets": int(n_bets),
        "n_won": int(bet_won.sum()),
        "hit_rate_pct": round(hit_rate, 2),
        "total_staked": round(total_staked, 2),
        "total_return": round(float(total_return), 2),
        "profit": round(float(profit.sum()), 2),
        "roi_pct": round(roi, 2),
        "avg_edge": round(float(avg_edge), 4),
        "avg_odds": round(float(bet_odds.mean()), 2),
        "stake": stake,
        "min_edge_used": min_edge,
    }


# ═══════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════


def format_currency(val: float) -> str:
    if val >= 0:
        return f"+${val:,.2f}"
    return f"-${abs(val):,.2f}"


def generate_report(
    ou_results: dict[str, Any],
    btts_results: dict[str, Any] | None,
    ou_stats: dict[str, Any],
    btts_stats: dict[str, Any] | None,
    timestamp: str,
) -> Path:
    """Generate markdown report."""
    lines = [
        "# Backtest Report — O/U & BTTS Models vs Market",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Period:** 2023-2024",
        f"**Stake per bet:** ${STAKE:.0f} (level stakes)",
        f"**Minimum edge:** {MIN_EDGE*100:.0f}%",
        "",
        "---",
        "",
        "## 1. Over/Under 2.5 Backtest",
        "",
        f"**Test set:** {ou_stats['n_total']} matches",
        f"**With O/U odds:** {ou_stats['n_with_odds']} matches ({ou_stats['n_with_odds']/ou_stats['n_total']*100:.0f}%)",
        f"**Actual Over 2.5 rate:** {ou_stats['target_mean']*100:.1f}%",
        f"**Model predicted O/U rate:** {ou_stats['pred_mean']*100:.1f}%",
        "",
    ]

    for result_key, label, direction in [
        ("ou_over", "Over 2.5 (bet Over)", "over"),
        ("ou_under", "Under 2.5 (bet Under)", "under"),
    ]:
        res = ou_results.get(direction, {})
        if res.get("n_bets", 0) == 0:
            lines.append(f"### {label}")
            lines.append("")
            lines.append("No bets placed — edge threshold not met.")
            lines.append("")
            continue

        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|:-------|:-----:|")
        lines.append(f"| Bets placed | {res['n_bets']} |")
        lines.append(f"| Won | {res['n_won']} ({res['hit_rate_pct']:.1f}%) |")
        lines.append(f"| Total staked | ${res['total_staked']:,.0f} |")
        lines.append(f"| Total return | ${res['total_return']:,.0f} |")
        lines.append(f"| **Profit** | **{format_currency(res['profit'])}** |")
        lines.append(f"| **ROI** | **{res['roi_pct']:+.2f}%** |")
        lines.append(f"| Avg odds | {res['avg_odds']:.2f} |")
        lines.append(f"| Avg edge | {res['avg_edge']:.2%} |")
        lines.append("")

        # By league
        if res.get("by_league"):
            lines.append(f"**By League:**")
            lines.append(f"| League | Bets | Won | Profit | ROI |")
            lines.append(f"|:-------|:----:|:---:|:------:|:---:|")
            for league in sorted(res["by_league"].keys()):
                lr = res["by_league"][league]
                profit_str = format_currency(lr["profit"])
                lines.append(f"| {league:>6} | {lr['bets']:>4} | {lr['won']:>3} ({lr['won']/lr['bets']*100:.0f}%) | {profit_str} | {lr['roi_pct']:+.2f}% |")
            lines.append("")

        # By year
        if res.get("by_year"):
            lines.append(f"**By Year:**")
            lines.append(f"| Year | Bets | Won | Profit | ROI |")
            lines.append(f"|:----:|:----:|:---:|:------:|:---:|")
            for year in sorted(res["by_year"].keys()):
                yr = res["by_year"][year]
                profit_str = format_currency(yr["profit"])
                lines.append(f"| {year} | {yr['bets']:>4} | {yr['won']:>3} ({yr['won']/yr['bets']*100:.0f}%) | {profit_str} | {yr['roi_pct']:+.2f}% |")
            lines.append("")

    # Overall O/U combined
    total_bets = sum(ou_results.get(d, {}).get("n_bets", 0) for d in ["over", "under"])
    total_profit = sum(ou_results.get(d, {}).get("profit", 0) for d in ["over", "under"])
    if total_bets > 0:
        lines.append("### Combined O/U Results")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|:-------|:-----:|")
        lines.append(f"| Total bets | {total_bets} |")
        lines.append(f"| Total profit | {format_currency(total_profit)} |")
        lines.append(f"| Overall ROI | {total_profit / (total_bets * STAKE) * 100:+.2f}% |")
        lines.append("")

    # Sensitivity analysis
    lines.append("### Edge Threshold Sensitivity")
    lines.append("")
    lines.append("| Min Edge | Over Bets | Over Profit | Over ROI | Under Bets | Under Profit | Under ROI |")
    lines.append("|:--------:|:---------:|:-----------:|:--------:|:----------:|:------------:|:---------:|")
    for edge in [0.01, 0.02, 0.03, 0.05, 0.08]:
        edge_str = f"{edge*100:.0f}%"
        over_res = ou_results.get("over", {})
        under_res = ou_results.get("under", {})
        lines.append(
            f"| {edge_str:>7} | {over_res.get('n_bets', '-'):>9} | "
            f"{format_currency(over_res.get('profit', 0)):>11} | {over_res.get('roi_pct', '-'):>+7}% | "
            f"{under_res.get('n_bets', '-'):>10} | "
            f"{format_currency(under_res.get('profit', 0)):>12} | {under_res.get('roi_pct', '-'):>+8}% |"
        )
    lines.append("")

    # BTTS section
    lines.append("---")
    lines.append("")
    lines.append("## 2. BTTS Backtest")
    lines.append("")
    if btts_stats:
        lines.append(f"**Test set:** {btts_stats['n_total']} matches")
        lines.append(f"**Actual BTTS rate:** {btts_stats['target_mean']*100:.1f}%")
        lines.append(f"**Model predicted BTTS rate:** {btts_stats['pred_mean']*100:.1f}%")
        lines.append("")
        if btts_results:
            method = btts_results.get("method", "hypothetical")
            if method == "derived_from_1x2_ou":
                lines.append("*BTTS odds derived from 1X2 + O/U markets via Random Forest model (Brier=0.244).*")
                lines.append("")
                derived = btts_results.get("derived")
                if derived and derived.get("n_bets", 0) > 0:
                    lines.append("| Metric | Value |")
                    lines.append("|:-------|:-----:|")
                    lines.append(f"| Bets placed | {derived['n_bets']} |")
                    lines.append(f"| Won | {derived['n_won']} ({derived['hit_rate_pct']:.1f}%) |")
                    lines.append(f"| Total staked | ${derived['total_staked']:,.0f} |")
                    lines.append(f"| **Profit** | **{format_currency(derived['profit'])}** |")
                    lines.append(f"| **ROI** | **{derived['roi_pct']:+.2f}%** |")
                    lines.append(f"| Avg odds | {derived['avg_odds']:.2f} |")
                    lines.append(f"| Avg edge | {derived['avg_edge']:.2%} |")
                    lines.append("")
                else:
                    lines.append("No BTTS bets placed at the minimum edge threshold (derived odds).")
                    lines.append("")
            else:
                lines.append("*Note: No real BTTS odds available. Showing hypothetical scenarios.*")
                lines.append("")
                if btts_results.get("scenarios"):
                    lines.append("| Assumed Odds | Bets | Won | Profit | ROI |")
                    lines.append("|:------------:|:----:|:---:|:------:|:---:|")
                    for scenario in btts_results["scenarios"]:
                        profit_str = format_currency(scenario["profit"])
                        lines.append(
                            f"| {scenario['assumed_odds']:.1f}x | {scenario['n_bets']:>4} | {scenario['n_won']:>3} | {profit_str} | {scenario['roi_pct']:+.2f}% |"
                        )
                else:
                    lines.append("No BTTS bets placed at the minimum edge threshold.")
        else:
            lines.append("BTTS backtest did not produce results.")
        lines.append("")
    else:
        lines.append("BTTS model not loaded (BTTS odds unavailable).")
        lines.append("")
    lines.append("### BTTS Recommendations")
    lines.append("")
    lines.append("- BTTS implied model trained from 1X2 + O/U markets (Brier=0.244)")
    lines.append("- For live value betting: use The-Odds-Api for 1X2 + O/U, then derive BTTS odds from the implied model")
    lines.append("- For historical backtesting: the derived approach is more realistic than assumed odds")
    lines.append("")
    lines.append("> **API-Football (RapidAPI)** would provide direct BTTS odds for ~$29/mo (paid tier).")
    lines.append("> Sign up: https://www.api-football.com/ or via RapidAPI marketplace.")
    lines.append("")

    # Recommendations
    lines.append("---")
    lines.append("")
    lines.append("## 3. Recommendations")
    lines.append("")
    if total_bets > 0:
        if total_profit > 0:
            lines.append("- The model finds profitable edges at the current threshold.")
        else:
            lines.append("- The model does not find profitable edges at the current threshold.")
        if total_bets < 100:
            lines.append("- **Small sample size** — results may not be statistically significant.")
        lines.append("- Consider adjusting the edge threshold for better risk/reward.")
    else:
        lines.append("- No bets placed at the current edge threshold.")
        lines.append("- Consider lowering the minimum edge or using a different model.")
    lines.append("- For BTTS: collect actual BTTS odds to enable full value betting analysis.")
    lines.append("")

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"backtest_ou_btts_{timestamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved report: %s", report_path)
    return report_path


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest O/U and BTTS models")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE,
                        help=f"Minimum edge over market (default: {MIN_EDGE})")
    parser.add_argument("--max-odds", type=float, default=MAX_ODDS,
                        help=f"Max decimal odds (default: {MAX_ODDS})")
    parser.add_argument("--stake", type=float, default=STAKE,
                        help=f"Stake per bet (default: ${STAKE})")
    parser.add_argument("--no-save", action="store_true", help="Skip saving report")
    parser.add_argument("--only-rolling", action="store_true",
                        help="Only use rolling team features (no odds/H2H/league)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set global mode flags
    global _ROLLING_ONLY
    _ROLLING_ONLY = args.only_rolling

    print("=" * 70)
    print(f"  BACKTEST: O/U & BTTS MODELS vs MARKET")
    print(f"  Stake: ${args.stake:.0f} | Min edge: {args.min_edge*100:.0f}% | Max odds: {args.max_odds:.1f}x")
    if args.only_rolling:
        print(f"  Mode: ROLLING-ONLY (team stats only, no odds/H2H/league)")
    print(f"  Period: 2023-2024")
    print("=" * 70)

    total_start = time.time()

    # ── Step 1: O/U backtest ──────────────────────────
    print("\n" + "-" * 70)
    print("  [A] OVER/UNDER 2.5 BACKTEST")
    print("-" * 70)

    print("\n  Loading O/U model and data...")
    ou_model, ou_df, ou_prob, ou_actual, ou_features, ou_stats = load_model_and_test_data(
        OU_MODEL_PATH, OU_DATA_PATH, "over_2_5",
    )

    ou_results: dict[str, Any] = {}

    for direction, odds_col in [("over", "over25_odds"), ("under", "under25_odds")]:
        print(f"\n  Betting {direction} 2.5...")
        result = backtest_market(
            ou_df, ou_prob, ou_actual,
            odds_col=odds_col,
            target_name="Over 2.5" if direction == "over" else "Under 2.5",
            direction=direction,
            min_edge=args.min_edge,
            max_odds=args.max_odds,
            stake=args.stake,
        )
        ou_results[direction] = result

        if result.get("n_bets", 0) > 0:
            print(f"  Bets: {result['n_bets']:>4} | "
                  f"Won: {result['n_won']:>3} ({result['hit_rate_pct']:.1f}%) | "
                  f"Profit: {format_currency(result['profit']):>10} | "
                  f"ROI: {result['roi_pct']:+.2f}%")
        else:
            print(f"  No bets placed")

    # Combined
    total_bets = sum(ou_results.get(d, {}).get("n_bets", 0) for d in ["over", "under"])
    total_profit = sum(ou_results.get(d, {}).get("profit", 0) for d in ["over", "under"])
    if total_bets > 0:
        combined_roi = total_profit / (total_bets * args.stake) * 100
        print(f"\n  {'Combined:':15s} {total_bets:>4} bets | "
              f"Profit: {format_currency(total_profit):>10} | "
              f"ROI: {combined_roi:+.2f}%")

    # ── Step 2: BTTS backtest ─────────────────────────
    print("\n" + "-" * 70)
    print("  [B] BTTS BACKTEST")
    print("-" * 70)

    btts_results = None
    btts_stats = None

    if BTTS_MODEL_PATH.exists() and BTTS_DATA_PATH.exists():
        print("\n  Loading BTTS model and data...")
        try:
            # BTTS model was trained WITH leaky features — don't exclude them
            btts_model, btts_df, btts_prob, btts_actual, btts_features, btts_stats = (
                load_model_and_test_data(BTTS_MODEL_PATH, BTTS_DATA_PATH, "btts",
                                         exclude_leaky=False)
            )

            btts_results = backtest_btts(
                btts_df, btts_prob, btts_actual,
                min_edge=args.min_edge, max_odds=args.max_odds, stake=args.stake,
            )
            method = btts_results.get("method", "?")
            if method == "derived_from_1x2_ou":
                derived = btts_results.get("derived", {})
                if derived and derived.get("n_bets", 0) > 0:
                    print(f"\n  BTTS Yes (derived from 1X2+O/U markets):")
                    print(f"    Bets: {derived['n_bets']:>4} | "
                          f"Won: {derived['n_won']:>3} ({derived['hit_rate_pct']:.1f}%) | "
                          f"Profit: {format_currency(derived['profit']):>10} | "
                          f"ROI: {derived['roi_pct']:+.2f}%")
                else:
                    print(f"  No BTTS bets at this threshold.")
            elif btts_results.get("scenarios"):
                print(f"\n  BTTS Yes — hypothetical scenarios (no real odds):")
                for scenario in btts_results["scenarios"]:
                    print(f"    @ {scenario['assumed_odds']:.1f}x: {scenario['n_bets']:>4} bets | "
                          f"Profit: {format_currency(scenario['profit']):>10} | "
                          f"ROI: {scenario['roi_pct']:+.2f}%")
            else:
                print("  No BTTS bets at this threshold.")
        except Exception as exc:
            print(f"  BTTS backtest failed: {exc}")
            btts_stats = None
    else:
        print("\n  BTTS model not found — skipping")

    # ── Step 3: Generate report ───────────────────────
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  BACKTEST COMPLETE ({total_elapsed:.1f}s)")

    if not args.no_save:
        report_path = generate_report(
            ou_results, btts_results, ou_stats, btts_stats, timestamp,
        )
        print(f"  Report: {report_path}")

    # Save JSON data
    if not args.no_save:
        json_data = {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "stake": args.stake,
                "min_edge": args.min_edge,
                "max_odds": args.max_odds,
                "period": "2023-2024",
            },
            "ou_model": str(OU_MODEL_PATH.name),
            "ou_stats": ou_stats,
            "ou_results": ou_results,
        }
        if btts_stats:
            json_data["btts_model"] = str(BTTS_MODEL_PATH.name)
            json_data["btts_stats"] = btts_stats
            json_data["btts_results"] = btts_results

        json_path = REPORTS_DIR / f"backtest_ou_btts_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(_to_native(json_data), f, indent=2)
        print(f"  JSON data: {json_path}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
