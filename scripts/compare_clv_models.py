"""
Compare CLV performance across all models.

Loads per-model CLV JSONs, computes comparison metrics,
identifies best performers, and saves results to CSV + console report.

Output: reports/clv_comparison_{timestamp}.csv
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("compare_clv_models")

REPORTS_DIR = Path("reports")

CLV_METRICS = ["avg_clv", "max_clv", "min_clv", "positive_clv_pct"]
COMPUTED_METRICS = ["gt_5_pct", "gt_10_pct", "median_clv", "std_clv"]


def find_per_model_files() -> list[Path]:
    """Find the latest batch of per-model CLV JSONs."""
    all_files = sorted(f for f in REPORTS_DIR.glob("clv_*.json")
                       if not any(x in f.name for x in ["summary", "analysis", "tracking", "report", "comparison"]))
    groups: dict[str, list[Path]] = {}
    import re
    for fp in all_files:
        m = re.search(r"_(\d{8}_\d{6})\.json$", fp.name)
        if m:
            groups.setdefault(m.group(1), []).append(fp)
    if not groups:
        logger.error("No timestamped CLV files found")
        sys.exit(1)
    latest = max(groups.keys())
    result = groups[latest]
    logger.info("Found %d CLV files for timestamp %s", len(result), latest)
    return result


def load_all_clv(files: list[Path]) -> dict[str, dict[str, Any]]:
    """Load all per-model CLV data and compute derived metrics."""
    models: dict[str, dict[str, Any]] = {}
    for fp in sorted(files):
        with open(fp) as f:
            raw = json.load(f)
        name = raw.get("model_name", fp.stem.replace("clv_", ""))
        cs = raw.get("clv_summary", {})

        clv_vals = [b.get("clv", 0) for b in raw.get("bets", [])]
        total = raw.get("total_bets", len(clv_vals))
        arr = np.array(clv_vals, dtype=np.float64)

        models[name] = {
            "total_bets": total,
            "n_winners": raw.get("n_winners", 0),
            "n_losers": raw.get("n_losers", 0),
            "avg_clv": cs.get("avg_clv", 0.0),
            "median_clv": round(float(np.median(arr)), 6) if len(arr) > 0 else 0.0,
            "std_clv": round(float(np.std(arr, ddof=1)), 6) if len(arr) > 1 else 0.0,
            "max_clv": cs.get("max_clv", 0.0),
            "min_clv": cs.get("min_clv", 0.0),
            "positive_clv_pct": cs.get("positive_clv_pct", 0.0),
            "gt_5_pct": round(np.sum(arr > 0.05) / total * 100, 2) if total > 0 else 0.0,
            "gt_10_pct": round(np.sum(arr > 0.10) / total * 100, 2) if total > 0 else 0.0,
            "n_with_closing": cs.get("n_with_closing", 0),
        }
    return models


def save_csv(models: dict[str, dict[str, Any]], timestamp: str) -> Path:
    """Save comparison table as CSV."""
    all_metrics = ["total_bets", "avg_clv", "median_clv", "std_clv",
                    "positive_clv_pct", "gt_5_pct", "gt_10_pct",
                    "max_clv", "min_clv", "n_with_closing"]

    header = "model," + ",".join(all_metrics)
    rows = [header]
    # Sort by avg_clv descending
    sorted_names = sorted(models, key=lambda n: models[n]["avg_clv"], reverse=True)
    for name in sorted_names:
        row = [name]
        for m in all_metrics:
            row.append(str(models[name].get(m, 0)))
        rows.append(",".join(row))

    path = REPORTS_DIR / f"clv_comparison_{timestamp}.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    logger.info("Saved comparison CSV: %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def print_report(models: dict[str, dict[str, Any]]) -> None:
    """Print formatted comparison report."""
    sorted_names = sorted(models, key=lambda n: models[n]["avg_clv"], reverse=True)

    print(f"\n{'=' * 85}")
    print("  CLV COMPARISON — All Models")
    print(f"{'=' * 85}")
    print(f"  {'Rank':<5} {'Model':<22} {'Bets':<5} {'Avg CLV':<10} {'Med CLV':<10} "
          f"{'+CLV%':<8} {'>5%':<7} {'>10%':<7} {'Win%':<6}")
    print(f"  {'-' * 80}")

    for rank, name in enumerate(sorted_names, 1):
        m = models[name]
        wr = m["n_winners"] / m["total_bets"] * 100 if m["total_bets"] > 0 else 0.0
        rank_tag = f"#{rank}"
        print(f"  {rank_tag:<5} {name:<22} {m['total_bets']:<5} "
              f"{m['avg_clv']:<+8.6f}  {m['median_clv']:<+8.6f}  "
              f"{m['positive_clv_pct']:<6.1f}%  {m['gt_5_pct']:<5.1f}%  "
              f"{m['gt_10_pct']:<5.1f}%  {wr:<4.1f}%")

    print(f"{'=' * 85}")

    # Best performers
    print()
    print("  🏆 BEST PERFORMERS BY METRIC")
    print(f"  {'-' * 60}")
    best_avg = max(models, key=lambda n: models[n]["avg_clv"])
    best_pos = max(models, key=lambda n: models[n]["positive_clv_pct"])
    best_gt5 = max(models, key=lambda n: models[n]["gt_5_pct"])
    best_gt10 = max(models, key=lambda n: models[n]["gt_10_pct"])
    best_med = max(models, key=lambda n: models[n]["median_clv"])

    print(f"  {'Highest Avg CLV':25s} → {best_avg} ({models[best_avg]['avg_clv']:+.6f})")
    print(f"  {'Highest +CLV%':25s} → {best_pos} ({models[best_pos]['positive_clv_pct']:.1f}%)")
    print(f"  {'Highest >5% CLV':25s} → {best_gt5} ({models[best_gt5]['gt_5_pct']:.1f}%)")
    print(f"  {'Highest >10% CLV':25s} → {best_gt10} ({models[best_gt10]['gt_10_pct']:.1f}%)")
    print(f"  {'Highest Median CLV':25s} → {best_med} ({models[best_med]['median_clv']:+.6f})")
    print()


def main() -> None:
    files = find_per_model_files()
    models = load_all_clv(files)
    if not models:
        logger.error("No model data loaded")
        sys.exit(1)
    logger.info("Loaded %d models", len(models))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = save_csv(models, timestamp)
    print_report(models)

    print(f"  CSV: {csv_path}")
    print()


if __name__ == "__main__":
    main()
