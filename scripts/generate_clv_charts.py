"""
Generate CLV visualizations from per-model CLV result files.

Produces:
  - Per-model: CLV distribution histogram, CLV over time
  - Combined:  CLV comparison across models, CLV by market, positive CLV %

All figures saved to reports/figures/clv_*.png
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("clv_charts")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def find_latest_clv_files() -> list[Path]:
    """Find the most recent batch of per-model CLV JSONs."""
    all_files = sorted(
        f for f in REPORTS_DIR.glob("clv_*.json")
        if not any(x in f.name for x in ["summary", "analysis", "tracking", "report", "comparison"])
    )
    if not all_files:
        logger.error("No CLV files found")
        sys.exit(1)
    groups: dict[str, list[Path]] = defaultdict(list)
    for fp in all_files:
        m = re.search(r"_(\d{8}_\d{6})\.json$", fp.name)
        if m:
            groups[m.group(1)].append(fp)
    if not groups:
        logger.error("No timestamped CLV files")
        sys.exit(1)
    latest = max(groups.keys())
    result = groups[latest]
    logger.info("Found %d CLV files for timestamp %s", len(result), latest)
    return result


def load_clv_data(file_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Load all per-model CLV data."""
    data: dict[str, list[dict[str, Any]]] = {}
    for fp in sorted(file_paths):
        with open(fp) as f:
            raw = json.load(f)
        name = raw.get("model_name", fp.stem.replace("clv_", ""))
        bets = raw.get("bets", [])
        if bets:
            data[name] = bets
    logger.info("Loaded %d models", len(data))
    return data


# ══════════════════════════════════════════════════════════
#  Per-model charts
# ══════════════════════════════════════════════════════════


def plot_clv_distribution(model: str, bets: list[dict[str, Any]]) -> Path:
    """Histogram of CLV values with key stats annotated."""
    fig, ax = plt.subplots(figsize=(10, 5))
    clv = np.array([b.get("clv", 0.0) for b in bets])

    # Compute dynamic range (cap at 99th percentile to avoid extreme outliers)
    p99 = float(np.percentile(clv, 99))
    p1 = float(np.percentile(clv, 1))
    lo = max(p1, -5.0)
    hi = min(p99, 5.0)
    clipped = np.clip(clv, lo, hi)

    ax.hist(clipped, bins=30, color="#1f77b4", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="CLV=0")
    ax.axvline(float(np.mean(clv)), color="green", linestyle="-", linewidth=1.5,
               label=f"Mean={np.mean(clv):+.4f}")
    ax.axvline(float(np.median(clv)), color="orange", linestyle=":", linewidth=1.5,
               label=f"Median={np.median(clv):+.4f}")

    pct_pos = np.sum(clv > 0) / len(clv) * 100
    ax.set_xlabel("CLV")
    ax.set_ylabel("Frequency")
    ax.set_title(f"CLV Distribution — {model}  (+CLV={pct_pos:.1f}%)")
    ax.legend(fontsize=9)

    # Stats box
    outliers = int(np.sum((clv < lo) | (clv > hi)))
    ax.text(0.98, 0.95, f"n={len(clv)}\noutliers clipped={outliers}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / f"clv_dist_{model}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_clv_over_time(model: str, bets: list[dict[str, Any]]) -> Path:
    """CLV value for each bet in chronological order with running average."""
    fig, ax = plt.subplots(figsize=(10, 5))
    clv = np.array([b.get("clv", 0.0) for b in bets])
    x = range(len(clv))

    # Bar chart: green for positive, red for negative
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in clv]
    ax.bar(x, clv, color=colors, width=0.7, alpha=0.6, edgecolor="none")

    # Running average (window=5)
    if len(clv) >= 5:
        running = np.convolve(clv, np.ones(5) / 5, mode="valid")
        ax.plot(range(4, len(clv)), running, color="black", linewidth=2,
                label="5-bet MA", zorder=5)

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.axhline(float(np.mean(clv)), color="green", linestyle="--", linewidth=1,
               label=f"Avg={np.mean(clv):+.4f}")

    ax.set_xlabel("Bet Number")
    ax.set_ylabel("CLV")
    ax.set_title(f"CLV Over Time — {model}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIGURES_DIR / f"clv_over_time_{model}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════
#  Combined charts
# ══════════════════════════════════════════════════════════


def plot_clv_comparison(all_data: dict[str, list[dict[str, Any]]]) -> Path:
    """Horizontal bar chart: avg CLV per model (sorted), with +CLV% annotation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names: list[str] = []
    avg_clvs: list[float] = []
    pct_pos: list[float] = []

    for name, bets in sorted(all_data.items()):
        clv = np.array([b.get("clv", 0.0) for b in bets])
        names.append(name)
        avg_clvs.append(float(np.mean(clv)))
        pct_pos.append(float(np.sum(clv > 0) / len(clv) * 100))

    # Sort by avg CLV descending
    idx = np.argsort(avg_clvs)[::-1]
    names_sorted = [names[i] for i in idx]
    avg_sorted = [avg_clvs[i] for i in idx]
    pct_sorted = [pct_pos[i] for i in idx]

    colors_bar = ["#2ecc71" if v >= 0 else "#e74c3c" for v in avg_sorted]
    bars = ax.barh(names_sorted, avg_sorted, color=colors_bar, alpha=0.7, edgecolor="gray")

    for i, (bar, pct) in enumerate(zip(bars, pct_sorted)):
        ax.text(
            bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2,
            f"+CLV={pct:.1f}%",
            va="center", fontsize=8, color="gray",
        )

    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Average CLV")
    ax.set_title("CLV Comparison Across Models")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = FIGURES_DIR / "clv_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_clv_by_market(all_data: dict[str, list[dict[str, Any]]]) -> Path:
    """Grouped bar: CLV by market per model (or aggregate if only 1X2)."""
    # Collect CLV by market across all models
    market_clvs: dict[str, list[float]] = defaultdict(list)
    for name, bets in all_data.items():
        for b in bets:
            mkt = b.get("market", "Unknown")
            market_clvs[mkt].append(b.get("clv", 0.0))

    fig, ax = plt.subplots(figsize=(10, 5))

    markets = sorted(market_clvs.keys())
    x = range(len(markets))
    averages = [float(np.mean(market_clvs[m])) for m in markets]
    positives = [
        float(np.sum(np.array(market_clvs[m]) > 0) / len(market_clvs[m]) * 100)
        for m in markets
    ]
    counts = [len(market_clvs[m]) for m in markets]

    colors_bar = ["#2ecc71" if v >= 0 else "#e74c3c" for v in averages]
    bars = ax.bar(x, averages, color=colors_bar, alpha=0.7, edgecolor="gray", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(markets)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Market")
    ax.set_ylabel("Average CLV")
    ax.set_title("CLV by Market")

    for bar, pct, cnt in zip(bars, positives, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.01 if bar.get_height() >= 0 else -0.03),
                f"+CLV={pct:.1f}%\nn={cnt}", ha="center", fontsize=9)

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIGURES_DIR / "clv_by_market.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_clv_positive_pct(all_data: dict[str, list[dict[str, Any]]]) -> Path:
    """Bar chart of % positive CLV per model with threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names: list[str] = []
    pct_pos: list[float] = []
    avg_clvs: list[float] = []

    for name, bets in sorted(all_data.items()):
        clv = np.array([b.get("clv", 0.0) for b in bets])
        names.append(name)
        pct_pos.append(float(np.sum(clv > 0) / len(clv) * 100))
        avg_clvs.append(float(np.mean(clv)))

    # Sort by pct_pos descending
    idx = np.argsort(pct_pos)[::-1]
    names_sorted = [names[i] for i in idx]
    pct_sorted = [pct_pos[i] for i in idx]
    avg_sorted = [avg_clvs[i] for i in idx]

    bars = ax.bar(names_sorted, pct_sorted, color=[
        "#2ecc71" if p >= 50 else "#f39c12" if p >= 40 else "#e74c3c"
        for p in pct_sorted
    ], alpha=0.7, edgecolor="gray")

    # Threshold lines
    ax.axhline(50, color="green", linestyle="--", linewidth=1, alpha=0.6, label="50% (skill threshold)")
    ax.axhline(40, color="orange", linestyle=":", linewidth=1, alpha=0.6, label="40% (random)")

    for bar, pct, avg in zip(bars, pct_sorted, avg_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n(avg={avg:+.3f})", ha="center", fontsize=7, color="gray")

    ax.set_ylabel("Bets with Positive CLV (%)")
    ax.set_title("Positive CLV Rate by Model")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(pct_sorted) + 15 if pct_sorted else 100)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIGURES_DIR / "clv_positive_pct.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    file_paths = find_latest_clv_files()
    all_data = load_clv_data(file_paths)

    if not all_data:
        logger.error("No CLV data loaded")
        sys.exit(1)

    generated: list[Path] = []

    # Per-model charts
    for model_name, bets in sorted(all_data.items()):
        if not bets:
            continue
        logger.info("Charts for %s (%d bets)...", model_name, len(bets))
        generated.append(plot_clv_distribution(model_name, bets))
        generated.append(plot_clv_over_time(model_name, bets))

    # Combined charts
    logger.info("Combined charts...")
    generated.append(plot_clv_comparison(all_data))
    generated.append(plot_clv_by_market(all_data))
    generated.append(plot_clv_positive_pct(all_data))

    # Summary
    print(f"\n{'=' * 60}")
    print("  GENERATED CLV CHARTS")
    print(f"{'=' * 60}")
    for p in generated:
        sz = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  [OK] {p.name:<42s} ({sz:.1f} KB)")
    n_per_model = sum(1 for p in generated if "clv_dist_" in p.name or "clv_over_time_" in p.name)
    n_combined = len(generated) - n_per_model
    print(f"{'=' * 60}")
    print(f"  {len(generated)} total: {n_per_model} per-model + {n_combined} combined")
    print()


if __name__ == "__main__":
    main()
