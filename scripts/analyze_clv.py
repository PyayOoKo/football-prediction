"""
Analyze CLV performance across all models.

Loads per-model CLV result JSONs (from calculate_clv_backtests.py),
computes comprehensive metrics, and saves analysis results.

Analysis per model
------------------
- Average CLV, Median CLV, Std Dev CLV
- % of bets with positive CLV
- CLV distribution (percentiles, histogram bins)
- CLV by market (all markets currently 1X2)
- CLV over time (by match date)

Cross-model analysis
--------------------
- Models with consistently positive CLV
- Markets with best CLV
- Time periods with best/worst CLV

Output: reports/clv_analysis_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("analyze_clv")

REPORTS_DIR = Path("reports")


# ═══════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════


def find_latest_clv_files() -> list[Path]:
    """Find the most recent batch of per-model CLV JSONs (exclude summary)."""
    all_files = sorted(
        f for f in REPORTS_DIR.glob("clv_*.json")
        if "summary" not in f.name and "analysis" not in f.name
    )
    if not all_files:
        logger.error("No CLV result files found in reports/")
        sys.exit(1)

    # Group by timestamp
    groups: dict[str, list[Path]] = defaultdict(list)
    for fp in all_files:
        match = re.search(r"_(\d{8}_\d{6})\.json$", fp.name)
        if match:
            groups[match.group(1)].append(fp)

    if not groups:
        logger.error("No timestamped CLV files found")
        sys.exit(1)

    latest_ts = max(groups.keys())
    result = groups[latest_ts]
    logger.info("Found %d CLV files for timestamp %s", len(result), latest_ts)
    return result


def load_clv_data(file_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Load all per-model CLV data into a dict keyed by model name."""
    all_data: dict[str, list[dict[str, Any]]] = {}
    for fp in sorted(file_paths):
        with open(fp) as f:
            data = json.load(f)
        name = data.get("model_name", fp.stem.replace("clv_", ""))
        bets = data.get("bets", [])
        if bets:
            all_data[name] = bets
            logger.debug("  %s: %d bets", name, len(bets))
    logger.info("Loaded %d models with CLV data", len(all_data))
    return all_data


# ═══════════════════════════════════════════════════════════
#  Per-model analysis
# ═══════════════════════════════════════════════════════════


def analyze_model_clv(
    model_name: str,
    bets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute all CLV metrics for a single model."""
    if not bets:
        return {"model_name": model_name, "total_bets": 0, "error": "no bets"}

    clv_values = np.array([b.get("clv", 0.0) for b in bets])

    # Basic statistics
    avg_clv = float(np.mean(clv_values))
    median_clv = float(np.median(clv_values))
    std_clv = float(np.std(clv_values, ddof=1)) if len(clv_values) > 1 else 0.0

    # Positive CLV stats
    n_positive = int(np.sum(clv_values > 0))
    n_negative = int(np.sum(clv_values < 0))
    n_zero = int(np.sum(clv_values == 0))
    pct_positive = (n_positive / len(clv_values)) * 100
    pct_negative = (n_negative / len(clv_values)) * 100

    # Distribution percentiles
    percentiles = {
        "p1": float(np.percentile(clv_values, 1)),
        "p5": float(np.percentile(clv_values, 5)),
        "p10": float(np.percentile(clv_values, 10)),
        "p25": float(np.percentile(clv_values, 25)),
        "p50": float(np.percentile(clv_values, 50)),
        "p75": float(np.percentile(clv_values, 75)),
        "p90": float(np.percentile(clv_values, 90)),
        "p95": float(np.percentile(clv_values, 95)),
        "p99": float(np.percentile(clv_values, 99)),
    }

    # Histogram bins (for charting)
    hist, bin_edges = np.histogram(
        clv_values,
        bins=20,
        range=(-2.0, 2.0),
    )
    histogram = {
        "counts": hist.tolist(),
        "bin_edges": [round(float(e), 4) for e in bin_edges],
    }

    # Min/max
    max_clv = float(np.max(clv_values))
    min_clv = float(np.min(clv_values))

    # CLV by market
    market_groups: dict[str, list[float]] = defaultdict(list)
    for b in bets:
        mkt = b.get("market", "Unknown")
        market_groups[mkt].append(b.get("clv", 0.0))

    clv_by_market: dict[str, dict[str, float]] = {}
    for mkt, vals in sorted(market_groups.items()):
        arr = np.array(vals)
        clv_by_market[mkt] = {
            "count": len(vals),
            "avg_clv": round(float(np.mean(arr)), 6),
            "median_clv": round(float(np.median(arr)), 6),
            "std_clv": round(float(np.std(arr, ddof=1)), 6),
            "pct_positive": round(
                (int(np.sum(arr > 0)) / len(arr)) * 100, 2,
            ),
        }

    # CLV by outcome (Home Win, Draw, Away Win)
    outcome_groups: dict[str, list[float]] = defaultdict(list)
    for b in bets:
        outcome = b.get("outcome", "Unknown")
        outcome_groups[outcome].append(b.get("clv", 0.0))

    clv_by_outcome: dict[str, dict[str, float]] = {}
    for out, vals in sorted(outcome_groups.items()):
        arr = np.array(vals)
        clv_by_outcome[out] = {
            "count": len(vals),
            "avg_clv": round(float(np.mean(arr)), 6),
        }

    # CLV by win/loss
    won_clv = np.array([b.get("clv", 0.0) for b in bets if b.get("won")])
    lost_clv = np.array([b.get("clv", 0.0) for b in bets if not b.get("won")])

    clv_by_result = {
        "won": {
            "count": int(len(won_clv)),
            "avg_clv": round(float(np.mean(won_clv)), 6) if len(won_clv) > 0 else 0.0,
            "median_clv": round(float(np.median(won_clv)), 6) if len(won_clv) > 0 else 0.0,
        },
        "lost": {
            "count": int(len(lost_clv)),
            "avg_clv": round(float(np.mean(lost_clv)), 6) if len(lost_clv) > 0 else 0.0,
            "median_clv": round(float(np.median(lost_clv)), 6) if len(lost_clv) > 0 else 0.0,
        },
    }

    # Consistency score: % of bets with positive CLV × avg CLV
    # Models with high positive pct AND high avg CLV are most consistent
    consistency_score = round((pct_positive / 100) * max(avg_clv, 0), 6)

    return {
        "model_name": model_name,
        "total_bets": len(bets),
        "winners": int(np.sum([b.get("won", False) for b in bets])),
        "losers": int(np.sum([not b.get("won", False) for b in bets])),

        "clv_stats": {
            "avg_clv": round(avg_clv, 6),
            "median_clv": round(median_clv, 6),
            "std_clv": round(std_clv, 6),
            "min_clv": round(min_clv, 6),
            "max_clv": round(max_clv, 6),
        },

        "positive_clv": {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_zero": n_zero,
            "pct_positive": round(pct_positive, 2),
            "pct_negative": round(pct_negative, 2),
        },

        "distribution": {
            "percentiles": percentiles,
            "histogram": histogram,
        },

        "clv_by_market": clv_by_market,
        "clv_by_outcome": clv_by_outcome,
        "clv_by_result": clv_by_result,

        "consistency_score": consistency_score,
    }


# ═══════════════════════════════════════════════════════════
#  Cross-model analysis
# ═══════════════════════════════════════════════════════════


def cross_model_analysis(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute cross-model comparative analysis."""
    if not results:
        return {}

    # Models with consistently positive CLV (positive avg AND > 50% positive bets)
    consistently_positive = [
        name
        for name, r in results.items()
        if r.get("clv_stats", {}).get("avg_clv", 0) > 0
        and r.get("positive_clv", {}).get("pct_positive", 0) > 50
    ]

    # Markets with best CLV (aggregate across models)
    market_aggregates: dict[str, list[float]] = defaultdict(list)
    for name, r in results.items():
        for mkt, mkt_data in r.get("clv_by_market", {}).items():
            market_aggregates[mkt].append(mkt_data["avg_clv"])

    best_markets: dict[str, dict[str, float]] = {}
    for mkt, clvs in market_aggregates.items():
        arr = np.array(clvs)
        best_markets[mkt] = {
            "models_analyzed": len(clvs),
            "avg_clv_across_models": round(float(np.mean(arr)), 6),
            "best_model_clv": round(float(np.max(arr)), 6),
            "worst_model_clv": round(float(np.min(arr)), 6),
        }

    # Top models by consistency
    ranked_by_consistency = sorted(
        results.items(),
        key=lambda kv: kv[1].get("consistency_score", 0),
        reverse=True,
    )

    top_consistent = [
        {
            "rank": i + 1,
            "model": name,
            "consistency_score": r.get("consistency_score", 0),
            "avg_clv": r.get("clv_stats", {}).get("avg_clv", 0),
            "pct_positive": r.get("positive_clv", {}).get("pct_positive", 0),
        }
        for i, (name, r) in enumerate(ranked_by_consistency[:5])
    ]

    # Overall average across all models
    all_avg_clv = np.array([
        r.get("clv_stats", {}).get("avg_clv", 0)
        for r in results.values()
    ])
    all_median_clv = np.array([
        r.get("clv_stats", {}).get("median_clv", 0)
        for r in results.values()
    ])
    all_pct_pos = np.array([
        r.get("positive_clv", {}).get("pct_positive", 0)
        for r in results.values()
    ])

    return {
        "models_with_consistently_positive_clv": consistently_positive,
        "n_models_consistently_positive": len(consistently_positive),
        "n_models_total": len(results),
        "markets_ranked_by_clv": dict(
            sorted(
                best_markets.items(),
                key=lambda kv: kv[1]["avg_clv_across_models"],
                reverse=True,
            )
        ),
        "top_models_by_consistency": top_consistent,
        "cross_model_averages": {
            "avg_clv": round(float(np.mean(all_avg_clv)), 6),
            "median_clv": round(float(np.median(all_median_clv)), 6),
            "std_clv": round(float(np.std(all_avg_clv, ddof=1)), 6),
            "avg_positive_pct": round(float(np.mean(all_pct_pos)), 2),
            "min_avg_clv": round(float(np.min(all_avg_clv)), 6),
            "max_avg_clv": round(float(np.max(all_avg_clv)), 6),
        },
    }


# ═══════════════════════════════════════════════════════════
#  Trend analysis
# ═══════════════════════════════════════════════════════════


def trend_analysis(
    all_bets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Analyze CLV trends over time across all models."""
    # Flatten all bets with model name and date
    flat: list[dict[str, Any]] = []
    for model_name, bets in all_bets.items():
        for b in bets:
            match_label = b.get("match_label", "")
            flat.append({
                "model": model_name,
                "clv": b.get("clv", 0.0),
                "match_label": match_label,
                "won": b.get("won", False),
            })

    if not flat:
        return {"error": "no data for trend analysis"}

    clv_values = np.array([f["clv"] for f in flat])
    won_array = np.array([f["won"] for f in flat])

    # Overall trend
    overall = {
        "total_bets": len(flat),
        "avg_clv": round(float(np.mean(clv_values)), 6),
        "median_clv": round(float(np.median(clv_values)), 6),
        "pct_positive": round(
            (int(np.sum(clv_values > 0)) / len(clv_values)) * 100, 2,
        ),
        "pct_won": round(
            (int(np.sum(won_array)) / len(won_array)) * 100, 2,
        ),
    }

    # Model-level trend summary
    model_trends: dict[str, dict[str, Any]] = {}
    for model_name, bets in all_bets.items():
        vals = np.array([b.get("clv", 0.0) for b in bets])
        won = np.array([b.get("won", False) for b in bets])
        model_trends[model_name] = {
            "bets": len(vals),
            "avg_clv": round(float(np.mean(vals)), 6),
            "pct_positive": round(
                (int(np.sum(vals > 0)) / len(vals)) * 100, 2,
            ),
            "pct_won": round(
                (int(np.sum(won)) / len(won)) * 100, 2,
            ),
            # Simple moving average of CLV (last 5 bets)
            "recent_avg_clv": (
                round(float(np.mean(vals[-5:])), 6)
                if len(vals) >= 5
                else round(float(np.mean(vals)), 6)
            ),
            "trend": (
                "Improving"
                if len(vals) >= 10
                and float(np.mean(vals[-5:])) > float(np.mean(vals[:5]))
                else "Declining"
                if len(vals) >= 10
                and float(np.mean(vals[-5:])) < float(np.mean(vals[:5]))
                else "Stable"
            ),
        }

    return {
        "overall": overall,
        "model_trends": model_trends,
    }


# ═══════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════


def save_analysis(
    per_model: dict[str, dict[str, Any]],
    cross: dict[str, Any],
    trends: dict[str, Any],
    timestamp: str,
) -> Path:
    """Save the full CLV analysis to a JSON file."""
    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_models": len(per_model),
        "per_model": per_model,
        "cross_model": cross,
        "trend_analysis": trends,
    }

    path = REPORTS_DIR / f"clv_analysis_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Analysis saved to %s (%d bytes)", path, len(json.dumps(data)))
    return path


# ═══════════════════════════════════════════════════════════
#  Console report
# ═══════════════════════════════════════════════════════════


def print_analysis(
    per_model: dict[str, dict[str, Any]],
    cross: dict[str, Any],
    trends: dict[str, Any],
) -> None:
    """Print a formatted CLV analysis report to the console."""
    print(f"\n{'=' * 90}")
    print("  CLV PERFORMANCE ANALYSIS — All Models".center(88))
    print(f"{'=' * 90}")

    # ── Per-model summary table ──
    print(f"\n  {'Model':<20} {'Bets':<6} {'Avg CLV':<12} {'Median':<12} "
          f"{'+CLV%':<9} {'Win%':<7} {'Consistency':<12}")
    print(f"  {'-' * 80}")

    sorted_models = sorted(
        per_model.values(),
        key=lambda r: r.get("consistency_score", 0),
        reverse=True,
    )

    for r in sorted_models:
        name = r["model_name"]
        bets = r["total_bets"]
        stats = r["clv_stats"]
        pos = r["positive_clv"]
        cons = r["consistency_score"]
        wr = (
            (r["winners"] / r["total_bets"] * 100)
            if r["total_bets"] > 0 else 0.0
        )
        print(
            f"  {name:<20} {bets:<6} "
            f"{stats['avg_clv']:<+10.6f}  "
            f"{stats['median_clv']:<+10.6f}  "
            f"{pos['pct_positive']:<7.1f}% "
            f"{wr:<5.1f}% "
            f"{cons:<+10.6f}"
        )

    # ── Cross-model summary ──
    print(f"\n{'=' * 90}")
    print("  CROSS-MODEL ANALYSIS".center(88))
    print(f"{'=' * 90}")

    ca = cross.get("cross_model_averages", {})
    print(f"\n  Average CLV across all models:         {ca.get('avg_clv', 0):+9.6f}")
    print(f"  Average +CLV %% across all models:      {ca.get('avg_positive_pct', 0):.1f}%")
    print(f"  Models with consistently positive CLV: {cross.get('n_models_consistently_positive', 0)} / {cross.get('n_models_total', 0)}")
    print()

    # Markets
    markets = cross.get("markets_ranked_by_clv", {})
    if markets:
        print(f"  {'Market':<20} {'Avg CLV':<12} {'Best Model':<12} {'Models':<8}")
        print(f"  {'-' * 52}")
        for mkt, data in markets.items():
            print(
                f"  {mkt:<20} {data['avg_clv_across_models']:<+10.6f}  "
                f"{data['best_model_clv']:<+10.6f}  "
                f"{data['models_analyzed']:<6d}"
            )

    # Top consistent models
    top = cross.get("top_models_by_consistency", [])
    if top:
        print(f"\n  Top models by consistency:")
        for m in top:
            print(
                f"    #{m['rank']} {m['model']:<20s}  "
                f"(score={m['consistency_score']:.4f}, "
                f"avg CLV={m['avg_clv']:+.6f}, "
                f"+CLV={m['pct_positive']:.1f}%)"
            )

    # ── Trends ──
    print(f"\n{'=' * 90}")
    print("  TREND ANALYSIS".center(88))
    print(f"{'=' * 90}")

    overall_trend = trends.get("overall", {})
    print(f"\n  Total bets analyzed:       {overall_trend.get('total_bets', 0)}")
    print(f"  Overall avg CLV:           {overall_trend.get('avg_clv', 0):+9.6f}")
    print(f"  Overall win rate:          {overall_trend.get('pct_won', 0):.1f}%")
    print()

    model_trends = trends.get("model_trends", {})
    if model_trends:
        print(f"  {'Model':<20} {'Bets':<6} {'Avg CLV':<12} {'Recent(5)':<12} {'Trend':<12}")
        print(f"  {'-' * 62}")
        for name, mt in sorted(model_trends.items()):
            print(
                f"  {name:<20} {mt['bets']:<6} "
                f"{mt['avg_clv']:<+10.6f}  "
                f"{mt['recent_avg_clv']:<+10.6f}  "
                f"{mt['trend']:<12}"
            )

    print(f"\n{'=' * 90}\n")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main() -> None:
    file_paths = find_latest_clv_files()
    all_bets = load_clv_data(file_paths)

    if not all_bets:
        logger.error("No CLV data loaded")
        sys.exit(1)

    # Per-model analysis
    logger.info("Analyzing per-model CLV...")
    per_model: dict[str, dict[str, Any]] = {}
    for name, bets in sorted(all_bets.items()):
        per_model[name] = analyze_model_clv(name, bets)

    # Cross-model analysis
    logger.info("Computing cross-model analysis...")
    cross = cross_model_analysis(per_model)

    # Trend analysis
    logger.info("Computing CLV trend analysis...")
    trends = trend_analysis(all_bets)

    # Save
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_analysis(per_model, cross, trends, timestamp)

    # Print report
    print_analysis(per_model, cross, trends)

    logger.info("CLV analysis complete!")


if __name__ == "__main__":
    main()
