"""
Compare backtest results across all models.

Loads the most recent batch of per-model backtest JSONs, builds a
comparison table (models × metrics), identifies best performers for
each key metric, and saves to CSV.

Usage:
    python scripts/compare_backtest_results.py

Output:
    reports/backtest_comparison_{timestamp}.csv
    reports/backtest_comparison_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("compare_backtest_results")

# ── Settings ─────────────────────────────────────────

REPORTS_DIR = Path("reports")

# Metrics to include in comparison, in display order
# (label_in_csv, field_in_json, higher_is_better, precision)
COMPARISON_METRICS = [
    ("total_bets", "total_bets", False, 0),       # more data is better
    ("winning_bets", "winning_bets", True, 0),
    ("losing_bets", "losing_bets", False, 0),
    ("total_profit", "total_profit", True, 2),
    ("roi_pct", "roi_pct", True, 2),
    ("yield_pct", "yield_pct", True, 2),
    ("win_rate_pct", "win_rate_pct", True, 2),
    ("sharpe_ratio", "sharpe_ratio", True, 4),
    ("sortino_ratio", "sortino_ratio", True, 4),
    ("max_drawdown_pct", "max_drawdown_pct", False, 2),
    ("profit_factor", "profit_factor", True, 4),
    ("avg_odds", "avg_odds", False, 4),
    ("avg_clv", "avg_clv", True, 6),
    ("positive_clv_pct", "positive_clv_pct", True, 2),
    ("avg_stake_pct", "avg_stake_pct", False, 2),
    ("longest_win_streak", "longest_win_streak", True, 0),
    ("longest_lose_streak", "longest_lose_streak", False, 0),
]

# ── Load data ────────────────────────────────────────


def find_latest_backtest_files() -> list[Path]:
    """Find the most recent batch of per-model backtest JSONs.

    Scans ``reports/backtest_*.json`` and groups by timestamp suffix.
    Returns the paths for the most populous group (the latest run).
    """
    # Glob per-model backtest results (exclude summary & comparison files)
    all_files = sorted(
        f for f in REPORTS_DIR.glob("backtest_*.json")
        if "summary" not in f.name and "comparison" not in f.name
    )
    if not all_files:
        logger.error("No backtest result files found in reports/")
        sys.exit(1)

    # Group by timestamp
    groups: dict[str, list[Path]] = defaultdict(list)
    for fp in all_files:
        # filename: backtest_ModelName_YYYYMMDD_HHMMSS.json
        # Extract the timestamp suffix (after last underscore before .json)
        match = re.search(r"_(\d{8}_\d{6})\.json$", fp.name)
        if match:
            groups[match.group(1)].append(fp)

    if not groups:
        # Fall back to just using all non-summary files
        result = [f for f in all_files if "summary" not in f.name]
        if not result:
            logger.error("No per-model backtest files found")
            sys.exit(1)
        logger.info("Using %d files (no timestamp grouping)", len(result))
        return result

    # Pick the most recent timestamp (lexicographic sort works for YYYYMMDD)
    latest_ts = max(groups.keys())
    result = groups[latest_ts]
    logger.info(
        "Found %d files for timestamp %s (from %d groups)",
        len(result), latest_ts, len(groups),
    )
    return result


def load_all_results(file_paths: list[Path]) -> list[dict]:
    """Load and flatten per-model backtest JSONs into a list of dicts.

    Skips any file that doesn't have a ``model_name`` key (safeguard
    against non-model files matching the glob).
    """
    results = []
    for fp in sorted(file_paths):
        with open(fp) as f:
            data = json.load(f)
        if "model_name" not in data:
            logger.debug("Skipping %s — no model_name field", fp.name)
            continue
        # Flatten: merge backtest_metrics into top-level
        metrics = data.pop("backtest_metrics", {})
        flat = {**data, **metrics}
        flat["source_file"] = fp.name
        results.append(flat)
    return results


# ── Build comparison ─────────────────────────────────


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build a models × metrics comparison DataFrame.

    Columns are the metrics defined in COMPARISON_METRICS plus
    model metadata (model_name, phase, calibration, brier).
    """
    rows = []
    for r in results:
        row = {
            "model_name": r.get("model_name", "?"),
            "phase": r.get("phase", ""),
            "calibration": r.get("calibration", ""),
            "brier": r.get("brier", 0.0),
        }
        for label, field, *_ in COMPARISON_METRICS:
            row[label] = r.get(field, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by Sharpe ratio descending (best model first)
    if "sharpe_ratio" in df.columns:
        df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    return df


def identify_best_performers(
    df: pd.DataFrame,
) -> dict[str, dict[str, str | float]]:
    """Identify the best model for each key metric.

    Returns a dict mapping metric labels to {model, value} dicts.
    """
    best: dict[str, dict[str, str | float]] = {}

    for label, field, higher_better, *_ in COMPARISON_METRICS:
        if field not in df.columns:
            continue
        if higher_better:
            best_idx = df[field].idxmax()
        else:
            best_idx = df[field].idxmin()
        best[label] = {
            "model": str(df.loc[best_idx, "model_name"]),
            "value": float(df.loc[best_idx, field]),
        }

    # Also add a "total_profit" winner
    if "total_profit" in df.columns:
        best["overall_profit"] = {
            "model": str(df.loc[df["total_profit"].idxmax(), "model_name"]),
            "value": float(df["total_profit"].max()),
        }

    return best


# ── Save ──────────────────────────────────────────────


def save_results(
    df: pd.DataFrame,
    best: dict,
    timestamp: str,
) -> None:
    """Save comparison table as CSV + JSON with best-performer highlights."""
    prefix = f"backtest_comparison_{timestamp}"

    # CSV
    csv_path = REPORTS_DIR / f"{prefix}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    logger.info("CSV saved to %s", csv_path)

    # JSON — full comparison + best performers
    json_path = REPORTS_DIR / f"{prefix}.json"
    data = {
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "total_models": len(df),
        "configuration": "FractionalKelly(0.50), min_ev=0.05, "
        "min_confidence=0.6, max_stake=0.05, bookmaker_margin=0.05",
        "comparison_table": df.to_dict(orient="records"),
        "best_performers": best,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("JSON saved to %s", json_path)


# ── Console report ───────────────────────────────────


def print_report(df: pd.DataFrame, best: dict) -> None:
    """Print a formatted comparison report to the console."""
    print("\n" + "=" * 100)
    print("  BACKTEST COMPARISON — All Models".center(98))
    print("=" * 100)

    # ── Comparison table ──
    cols = [
        "model_name", "brier", "total_bets", "total_profit",
        "roi_pct", "yield_pct", "win_rate_pct",
        "sharpe_ratio", "max_drawdown_pct", "profit_factor",
        "avg_clv",
    ]
    display_cols = [c for c in cols if c in df.columns]

    if not df.empty:
        print(f"\n  {'Model':<22} {'Brier':<8} {'Bets':<6} {'P&L':<10} "
              f"{'ROI%':<9} {'Yield%':<9} {'Win%':<7} {'Sharpe':<9} "
              f"{'DD%':<8} {'PF':<8} {'CLV':<8}")
        print(f"  {'-' * 98}")

        for _, row in df.iterrows():
            model = str(row.get("model_name", "?"))
            brier = row.get("brier", 0)
            bets = int(row.get("total_bets", 0))
            pnl = row.get("total_profit", 0)
            roi = row.get("roi_pct", 0)
            yld = row.get("yield_pct", 0)
            win = row.get("win_rate_pct", 0)
            sharpe = row.get("sharpe_ratio", 0)
            dd = row.get("max_drawdown_pct", 0)
            pf = row.get("profit_factor", 0)
            clv = row.get("avg_clv", 0)

            print(
                f"  {model:<22} {brier:<8.4f} {bets:<6} {pnl:>+8.2f}  "
                f"{roi:>+7.2f}% {yld:>+7.2f}% "
                f"{win:<6.1f}% {sharpe:<8.4f} {dd:<7.2f}% "
                f"{pf:<7.4f} {clv:<8.6f}"
            )

    # ── Best performers ──
    print(f"\n{'=' * 100}")
    print("  BEST PERFORMERS".center(98))
    print(f"{'=' * 100}\n")

    # Define the key highlights to show
    highlights = [
        ("overall_profit", "Total P&L (least losses)"),
        ("roi_pct", "Best ROI %"),
        ("yield_pct", "Best Yield %"),
        ("sharpe_ratio", "Best Sharpe Ratio"),
        ("avg_clv", "Best Average CLV"),
        ("max_drawdown_pct", "Most Consistent (lowest drawdown)"),
        ("win_rate_pct", "Highest Win Rate"),
        ("profit_factor", "Best Profit Factor"),
    ]

    for key, label in highlights:
        entry = best.get(key)
        if entry:
            model = str(entry["model"])
            value = entry["value"]
            if key == "overall_profit":
                print(f"  ** {label:<38s}  {model:<22s}  GBP{value:>+9.2f}")
            elif "pct" in key or "rate" in key:
                print(f"  ** {label:<38s}  {model:<22s}  {value:>+9.2f}%")
            elif key == "avg_clv":
                print(f"  ** {label:<38s}  {model:<22s}  {value:>+9.6f}")
            else:
                print(f"  ** {label:<38s}  {model:<22s}  {value:>9.4f}")

    print(f"\n{'=' * 100}")

    # Summary line
    print()
    top_roi = best.get("roi_pct", {}).get("model", "?")
    top_sharpe = best.get("sharpe_ratio", {}).get("model", "?")
    top_clv = best.get("avg_clv", {}).get("model", "?")
    safest = best.get("max_drawdown_pct", {}).get("model", "?")
    print(f"  Summary: {top_roi} leads in ROI, {top_sharpe} leads in "
          f"Sharpe,")
    print(f"           {safest} is most consistent (lowest drawdown), "
          f"{top_clv} has best CLV.")
    print()


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════


def main() -> None:
    file_paths = find_latest_backtest_files()
    logger.info("Loading %d backtest result files...", len(file_paths))

    results = load_all_results(file_paths)
    logger.info(
        "Loaded %d models: %s",
        len(results),
        ", ".join(r.get("model_name", "?") for r in results),
    )

    df = build_comparison_table(results)
    best = identify_best_performers(df)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_results(df, best, timestamp)
    print_report(df, best)

    logger.info("Comparison complete!")


if __name__ == "__main__":
    main()
