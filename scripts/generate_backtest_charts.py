"""
Generate backtest visualizations from per-model backtest results.

Produces:
  - Per-model: bankroll growth, P&L per bet, drawdown
  - Combined:  bankroll comparison, ROI comparison, drawdown comparison

All figures saved to reports/figures/*.png
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

# matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("generate_backtest_charts")

# ── Paths ────────────────────────────────────────────
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
INITIAL_BANKROLL = 1000.0

# ── Colour palette (10 distinct colours) ─────────────
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ══════════════════════════════════════════════════════
#  Data loading (reusing the grouping logic)
# ══════════════════════════════════════════════════════

def find_latest_backtest_files() -> list[Path]:
    """Find the most recent batch of per-model backtest JSONs."""
    all_files = sorted(
        f for f in REPORTS_DIR.glob("backtest_*.json")
        if "summary" not in f.name and "comparison" not in f.name
    )
    if not all_files:
        logger.error("No backtest files found in reports/")
        sys.exit(1)

    groups: dict[str, list[Path]] = defaultdict(list)
    for fp in all_files:
        match = re.search(r"_(\d{8}_\d{6})\.json$", fp.name)
        if match:
            groups[match.group(1)].append(fp)

    if not groups:
        result = [
            f for f in all_files
            if "summary" not in f.name and "comparison" not in f.name
        ]
        if not result:
            logger.error("No per-model backtest files found")
            sys.exit(1)
        return result

    latest_ts = max(groups.keys())
    result = groups[latest_ts]
    logger.info("Found %d files for timestamp %s", len(result), latest_ts)
    return result


def load_model_data(file_paths: list[Path]) -> list[dict]:
    """Load and parse all model backtest data."""
    models = []
    for fp in sorted(file_paths):
        with open(fp) as f:
            data = json.load(f)
        name = data.get("model_name", fp.stem.replace("backtest_", ""))
        brier = data.get("brier", 0)
        phase = data.get("phase", "")
        cal = data.get("calibration", "")
        m = data.get("backtest_metrics", {})

        # Equity curve (cumulative P&L from synthetic bankroll)
        equity = m.get("equity_curve", [])
        if not equity or not isinstance(equity, list):
            logger.warning("  %s: no equity_curve — skipping", name)
            continue

        # Derived bankroll: initial_bankroll + equity
        bankroll = [INITIAL_BANKROLL + v for v in equity]

        # Derived drawdown from bankroll
        drawdown = compute_drawdown(bankroll)

        # P&L per bet from equity differences (skip leading zero)
        pl_per_bet = [
            round(equity[i] - equity[i - 1], 2)
            for i in range(1, len(equity))
        ]

        models.append({
            "name": name,
            "brier": brier,
            "phase": phase,
            "calibration": cal,
            "equity": equity,
            "bankroll": bankroll,
            "drawdown": drawdown,
            "pl_per_bet": pl_per_bet,
            "total_bets": m.get("total_bets", 0),
            "roi_pct": m.get("roi_pct", 0),
            "yield_pct": m.get("yield_pct", 0),
            "win_rate_pct": m.get("win_rate_pct", 0),
            "sharpe_ratio": m.get("sharpe_ratio", 0),
            "max_drawdown_pct": m.get("max_drawdown_pct", 0),
            "profit_factor": m.get("profit_factor", 0),
            "avg_odds": m.get("avg_odds", 0),
        })

    # Sort by Sharpe descending
    models.sort(key=lambda m: m["sharpe_ratio"], reverse=True)
    logger.info("Loaded %d models: %s", len(models), ", ".join(m["name"] for m in models))
    return models


def compute_drawdown(bankroll: list[float]) -> list[float]:
    """Compute drawdown % from a bankroll time series."""
    if not bankroll:
        return []
    peak = bankroll[0]
    dd = []
    for v in bankroll:
        if v > peak:
            peak = v
        dd.append((peak - v) / peak * 100 if peak > 0 else 0.0)
    return dd


# ══════════════════════════════════════════════════════
#  Per-model plots
# ══════════════════════════════════════════════════════


def plot_bankroll_growth(model: dict) -> Path:
    """Bankroll over time (equity curve shifted by initial bankroll)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bankroll = model["bankroll"]
    ax.plot(bankroll, color=COLORS[0], linewidth=2, label=model["name"])
    ax.axhline(INITIAL_BANKROLL, color="gray", linestyle="--", alpha=0.5, label="Initial")
    ax.fill_between(range(len(bankroll)), INITIAL_BANKROLL, bankroll,
                     where=(np.array(bankroll) >= INITIAL_BANKROLL),
                     color="green", alpha=0.08)
    ax.fill_between(range(len(bankroll)), bankroll, INITIAL_BANKROLL,
                     where=(np.array(bankroll) < INITIAL_BANKROLL),
                     color="red", alpha=0.08)
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Bankroll (GBP)")
    ax.set_title(f"Bankroll Growth — {model['name']} (Brier={model['brier']:.4f})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / f"bankroll_{model['name']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pl_per_bet(model: dict) -> Path:
    """Bar chart of P&L for each individual bet."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pl = model["pl_per_bet"]
    bets = range(len(pl))
    colors = ["green" if v >= 0 else "red" for v in pl]
    ax.bar(bets, pl, color=colors, width=0.7, edgecolor="none", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("P&L (GBP)")
    ax.set_title(f"P&L Per Bet — {model['name']}")
    # Show total at end
    total = sum(pl)
    ax.text(0.98, 0.95, f"Total: GBP{total:+.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = FIGURES_DIR / f"pl_per_bet_{model['name']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_drawdown(model: dict) -> Path:
    """Drawdown over time (from bankroll peak)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    dd = model["drawdown"]
    ax.fill_between(range(len(dd)), dd, color="red", alpha=0.3)
    ax.plot(dd, color="darkred", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown — {model['name']}")
    # Annotate max drawdown
    max_dd = max(dd)
    max_idx = dd.index(max_dd)
    ax.annotate(f"Max: {max_dd:.1f}%", xy=(max_idx, max_dd),
                xytext=(max_idx + len(dd) * 0.05, max_dd * 1.1),
                arrowprops=dict(arrowstyle="->", color="darkred"),
                fontsize=9, color="darkred")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / f"drawdown_{model['name']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════
#  Combined plots
# ══════════════════════════════════════════════════════


def plot_bankroll_comparison(models: list[dict]) -> Path:
    """Overlay all bankroll curves on one chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    n_models = len(models)
    for i, m in enumerate(models):
        color = COLORS[i % len(COLORS)]
        bk = m["bankroll"]
        ax.plot(bk, color=color, linewidth=1.5, label=f"{m['name']} (Sharpe={m['sharpe_ratio']:.2f})")
    ax.axhline(INITIAL_BANKROLL, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Bankroll (GBP)")
    ax.set_title("Bankroll Comparison — All Models")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "bankroll_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roi_comparison(models: list[dict]) -> Path:
    """Horizontal bar chart of ROI % for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [m["name"] for m in models]
    roi = [m["roi_pct"] for m in models]
    yield_pct = [m["yield_pct"] for m in models]
    sharpe = [m["sharpe_ratio"] for m in models]

    y = range(len(names))
    height = 0.35

    bars_roi = ax.barh([i + height / 2 for i in y], roi, height,
                       color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                       alpha=0.8, label="ROI %")
    bars_yield = ax.barh([i - height / 2 for i in y], yield_pct, height,
                         color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                         alpha=0.3, label="Yield %", hatch="//")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Return (%)")
    ax.set_title("ROI & Yield Comparison — All Models")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(loc="lower right")

    # Annotate with Sharpe
    for i, (r, s) in enumerate(zip(roi, sharpe)):
        label = f"Sharpe={s:.2f}"
        ax.text(r + (1 if r >= 0 else -1), i, label,
                va="center", fontsize=7, color="gray")

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = FIGURES_DIR / "roi_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_drawdown_comparison(models: list[dict]) -> Path:
    """Bar chart of max drawdown for all models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [m["name"] for m in models]
    dd = [m["max_drawdown_pct"] for m in models]
    colors = ["darkgreen" if v < 50 else "orange" if v < 90 else "darkred" for v in dd]

    bars = ax.barh(names, dd, color=colors, alpha=0.7, edgecolor="gray")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_title("Maximum Drawdown — All Models")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Legend for risk levels
    legend_elements = [
        Patch(facecolor="darkgreen", alpha=0.7, label="Low (<50%)"),
        Patch(facecolor="orange", alpha=0.7, label="Moderate (50-90%)"),
        Patch(facecolor="darkred", alpha=0.7, label="High (>90%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    for bar, val in zip(bars, dd):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "drawdown_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    file_paths = find_latest_backtest_files()
    models = load_model_data(file_paths)

    if not models:
        logger.error("No models loaded — nothing to plot")
        sys.exit(1)

    generated: list[Path] = []

    # ── Per-model plots ──
    for m in models:
        logger.info("Generating charts for %s...", m["name"])
        generated.append(plot_bankroll_growth(m))
        generated.append(plot_pl_per_bet(m))
        generated.append(plot_drawdown(m))

    # ── Combined plots ──
    logger.info("Generating combined charts...")
    generated.append(plot_bankroll_comparison(models))
    generated.append(plot_roi_comparison(models))
    generated.append(plot_drawdown_comparison(models))

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  GENERATED BACKTEST CHARTS")
    print(f"{'=' * 60}")
    for p in generated:
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  [OK] {p.name:<40s} ({size_kb:.1f} KB)")
    print(f"{'=' * 60}")
    print(f"  Total: {len(generated)} charts saved to {FIGURES_DIR}/")
    print()


if __name__ == "__main__":
    main()
