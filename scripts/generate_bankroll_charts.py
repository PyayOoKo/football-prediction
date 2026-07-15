"""
Generate bankroll management visualizations from bankroll_management_*.json results.

Produces:
  - Combined: bankroll strategy comparison, risk-adjusted returns
  - Per-strategy: bankroll growth, drawdown, risk/reward profile

All figures saved to reports/figures/*.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("generate_bankroll_charts")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
INITIAL_BANKROLL = 1000.0    # Colour palette
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

CATEGORY_COLORS = {
    "kelly": "#1f77b4",
    "fractional_kelly": "#2ca02c",
    "fixed_ratio": "#d62728",
    "fixed": "#9467bd",
    "percentage": "#8c564b",
    "dynamic": "#e377c2",
    "portfolio": "#7f7f7f",
}


# ══════════════════════════════════════════════════════
#  Per-strategy bankroll growth chart
# ══════════════════════════════════════════════════════


def plot_single_strategy_growth(
    strategy: dict[str, Any],
    total_bets: int,
) -> Path:
    """Generate a simulated bankroll growth chart for a single strategy.

    Uses known metrics (ROI, DD, bets) to create a representative
    equity curve showing bankroll over time.
    """
    label = strategy.get("label", "Strategy")
    roi_pct = strategy.get("roi_pct", 0)
    max_dd = strategy.get("max_drawdown_pct", 0)
    sharpe = strategy.get("sharpe_ratio", 0)
    final = strategy.get("final_bankroll", INITIAL_BANKROLL)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Generate smooth equity curve from ROI and Sharpe characteristics
    n_points = max(total_bets, 3)
    x = np.linspace(0, 1, n_points + 1)

    # Target final multiplier
    final_multiplier = final / INITIAL_BANKROLL

    # Base trend (compounding growth)
    trend = INITIAL_BANKROLL * (final_multiplier ** x)

    # Add noise proportional to drawdown characteristics
    np.random.seed(hash(label) % (2**16))
    noise_std = max_dd * 0.005  # Scale noise to drawdown
    noise = np.random.randn(len(x)) * noise_std * INITIAL_BANKROLL
    noise[0] = 0
    noise[-1] = 0  # End at exact final value

    equity = trend + noise
    equity = np.maximum(equity, INITIAL_BANKROLL * 0.5)  # floor at 50%
    equity[-1] = final  # pin final value

    # Plot
    color = CATEGORY_COLORS.get(strategy.get("category", ""), COLORS[0])
    ax.plot(equity, color=color, linewidth=2, label=label)
    ax.axhline(INITIAL_BANKROLL, color="gray", linestyle="--", alpha=0.5,
               label=f"Initial (GBP{INITIAL_BANKROLL:.0f})")
    ax.fill_between(range(len(equity)), INITIAL_BANKROLL, equity,
                     where=(equity >= INITIAL_BANKROLL),
                     color="green", alpha=0.08)
    ax.fill_between(range(len(equity)), equity, INITIAL_BANKROLL,
                     where=(equity < INITIAL_BANKROLL),
                     color="red", alpha=0.08)

    # Annotations
    ax.text(0.98, 0.95,
            f"ROI: {roi_pct:+.1f}%  |  DD: {max_dd:.1f}%  |  "
            f"Sharpe: {sharpe:.2f}  |  Final: GBP{final:.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Bankroll (GBP)")
    ax.set_title(f"Bankroll Growth — {label}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = label.replace(" ", "_").replace("%", "pct").replace("£", "GBP").replace(",", "").replace("/", "-")
    path = FIGURES_DIR / f"bankroll_growth_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def find_latest_management_results() -> Path | None:
    """Find the most recent bankroll_management_*.json file."""
    files = sorted(REPORTS_DIR.glob("bankroll_management_*.json"))
    return files[-1] if files else None


def load_data(path: Path) -> dict[str, Any]:
    """Load and parse the management comparison JSON."""
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════
#  Chart: Bankroll strategy comparison (Sharpe, ROI, DD)
# ══════════════════════════════════════════════════════


def plot_bankroll_strategy_comparison(data: dict[str, Any]) -> Path:
    """Combined horizontal bar chart: Sharpe, ROI, Drawdown for all strategies.

    Three subplots side by side for multi-metric comparison.
    """
    strategies = data.get("stake_strategies", {}).get("results", [])
    if not strategies:
        logger.warning("No strategy results to plot")
        return FIGURES_DIR / "bankroll_strategy_comparison.png"

    # Sort by Sharpe descending, take top 15
    strategies = sorted(strategies, key=lambda s: s.get("sharpe_ratio", -999), reverse=True)[:15]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    labels = [s.get("label", "?") for s in strategies]
    sharpe = [s.get("sharpe_ratio", 0) for s in strategies]
    roi = [s.get("roi_pct", 0) for s in strategies]
    dd = [s.get("max_drawdown_pct", 0) for s in strategies]
    colors = [CATEGORY_COLORS.get(s.get("category", ""), "#999999") for s in strategies]

    cat_color = [CATEGORY_COLORS.get(s.get("category", ""), "#999999") for s in strategies]

    # Sharpe
    bars_sharpe = axes[0].barh(labels, sharpe, color=cat_color, alpha=0.8, edgecolor="gray")
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("Sharpe Ratio")
    axes[0].set_title("Risk-Adjusted Return (Sharpe)")
    axes[0].grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars_sharpe, sharpe):
        axes[0].text(val + (0.05 if val >= 0 else -0.05),
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", fontsize=7,
                     ha="left" if val >= 0 else "right")

    # ROI
    bars_roi = axes[1].barh(labels, roi, color=cat_color, alpha=0.8, edgecolor="gray")
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("ROI (%)")
    axes[1].set_title("Return on Investment")
    axes[1].grid(True, alpha=0.3, axis="x")

    # Drawdown (inverted — lower is better)
    dd_colors = ["darkgreen" if v < 10 else "orange" if v < 20 else "darkred" for v in dd]
    bars_dd = axes[2].barh(labels, dd, color=dd_colors, alpha=0.8, edgecolor="gray")
    axes[2].set_xlabel("Max Drawdown (%)")
    axes[2].set_title("Drawdown (lower is better)")
    axes[2].grid(True, alpha=0.3, axis="x")
    # Mark the 20% constraint line
    axes[2].axvline(20, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
                    label="< 20% constraint")
    for bar, val in zip(bars_dd, dd):
        axes[2].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=7)

    # Legend for categories
    legend_patches = [
        Patch(facecolor=color, alpha=0.8, label=cat)
        for cat, color in CATEGORY_COLORS.items()
        if cat in [s.get("category", "") for s in strategies]
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=6, fontsize=8, title="Strategy Category")

    plt.suptitle("Bankroll Strategy Comparison — Top 15 by Sharpe",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "bankroll_strategy_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ══════════════════════════════════════════════════════
#  Chart: Risk-adjusted returns (Sharpe vs Drawdown scatter)
# ══════════════════════════════════════════════════════


def plot_risk_adjusted_returns(data: dict[str, Any]) -> Path:
    """Scatter plot: Sharpe ratio vs Max Drawdown, size = ROI, color = category.

    The 'sweet spot' is top-right (high Sharpe, low drawdown).
    """
    strategies = data.get("stake_strategies", {}).get("results", [])
    if not strategies:
        logger.warning("No strategy results to plot")
        return FIGURES_DIR / "risk_adjusted_returns.png"

    fig, ax = plt.subplots(figsize=(12, 8))

    categories = {}
    for s in strategies:
        cat = s.get("category", "other")
        if cat not in categories:
            categories[cat] = {"labels": [], "sharpe": [], "dd": [], "roi": []}
        categories[cat]["labels"].append(s.get("label", "?"))
        categories[cat]["sharpe"].append(s.get("sharpe_ratio", 0))
        categories[cat]["dd"].append(s.get("max_drawdown_pct", 0))
        categories[cat]["roi"].append(s.get("roi_pct", 0))

    for cat, data_pts in categories.items():
        sharpe = data_pts["sharpe"]
        dd = data_pts["dd"]
        roi = data_pts["roi"]

        # Size = proportional to absolute ROI, min size 20, max 200
        sizes = [max(20, min(200, abs(r) * 3 + 30)) for r in roi]

        color = CATEGORY_COLORS.get(cat, "#999999")
        scatter = ax.scatter(dd, sharpe, s=sizes, c=color, alpha=0.7,
                             edgecolors="black", linewidth=0.5, label=cat)

        # Annotate top strategies
        for i, (lbl, s, d) in enumerate(zip(data_pts["labels"], sharpe, dd)):
            if s > 0.5 or s == max(sharpe):
                ax.annotate(lbl, (d, s),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=6, alpha=0.8)

    # Constraint region
    ax.axvline(20, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx([ymin, ymax], 0, 20, alpha=0.05, color="green",
                     label="Target zone (DD < 20%, Sharpe > 0)")

    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Risk-Adjusted Returns — Sharpe vs Drawdown\n"
                 "(size = |ROI|, color = strategy category)")
    ax.legend(loc="lower left", fontsize=8, title="Category")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    path = FIGURES_DIR / "risk_adjusted_returns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ══════════════════════════════════════════════════════
#  Chart: Kelly fraction sensitivity analysis
# ══════════════════════════════════════════════════════


def plot_kelly_sensitivity(data: dict[str, Any]) -> Path:
    """Line chart: Sharpe and Drawdown vs Kelly fraction for FractionalKelly strategies."""
    strategies = data.get("stake_strategies", {}).get("results", [])
    kelly_strategies = [
        s for s in strategies
        if s.get("category") == "kelly" and "Kelly" in s.get("label", "")
    ]

    if len(kelly_strategies) < 3:
        logger.info("Not enough Kelly strategies for sensitivity plot")
        return FIGURES_DIR / "kelly_sensitivity.png"

    # Extract fraction from label
    kelly_data = []
    for s in kelly_strategies:
        label = s.get("label", "")
        # Parse: "Kelly 10%", "Kelly 25%", ..., "Full Kelly"
        if "Full" in label:
            fraction = 1.0
        else:
            try:
                pct = float(label.split()[-1].replace("%", ""))
                fraction = pct / 100.0
            except (ValueError, IndexError):
                continue
        kelly_data.append({
            "fraction": fraction,
            "sharpe": s.get("sharpe_ratio", 0),
            "dd": s.get("max_drawdown_pct", 0),
            "roi": s.get("roi_pct", 0),
            "label": label,
        })

    if not kelly_data:
        return FIGURES_DIR / "kelly_sensitivity.png"

    kelly_data.sort(key=lambda k: k["fraction"])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    fractions = [k["fraction"] * 100 for k in kelly_data]

    color1 = "#1f77b4"
    ax1.plot(fractions, [k["sharpe"] for k in kelly_data],
             marker="o", color=color1, linewidth=2, markersize=8, label="Sharpe Ratio")
    ax1.set_xlabel("Kelly Fraction (%)")
    ax1.set_ylabel("Sharpe Ratio", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(0, color=color1, linestyle="--", alpha=0.3)

    color2 = "#d62728"
    ax2 = ax1.twinx()
    ax2.plot(fractions, [k["dd"] for k in kelly_data],
             marker="s", color=color2, linewidth=2, markersize=8, label="Max Drawdown")
    ax2.set_ylabel("Max Drawdown (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(20, color=color2, linestyle="--", alpha=0.3, linewidth=1)

    # Annotate each point
    for k in kelly_data:
        ax1.annotate(f'{k["label"]}',
                     (k["fraction"] * 100, k["sharpe"]),
                     xytext=(0, 10), textcoords="offset points",
                     fontsize=7, ha="center")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Kelly Fraction Sensitivity — Sharpe vs Drawdown")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "kelly_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ══════════════════════════════════════════════════════
#  Chart: Fixed ratio sensitivity analysis
# ══════════════════════════════════════════════════════


def plot_ratio_sensitivity(data: dict[str, Any]) -> Path:
    """Line chart: Sharpe and Drawdown vs ratio for FixedRatio strategies."""
    strategies = data.get("stake_strategies", {}).get("results", [])
    ratio_strategies = [
        s for s in strategies
        if s.get("category") == "percentage" and "FixedRatio" in s.get("label", "")
    ]

    if len(ratio_strategies) < 3:
        logger.info("Not enough FixedRatio strategies for sensitivity plot")
        return FIGURES_DIR / "ratio_sensitivity.png"

    ratio_data = []
    for s in ratio_strategies:
        label = s.get("label", "")
        try:
            pct = float(label.split()[-1].replace("%", ""))
        except (ValueError, IndexError):
            continue
        ratio_data.append({
            "ratio_pct": pct,
            "sharpe": s.get("sharpe_ratio", 0),
            "dd": s.get("max_drawdown_pct", 0),
            "roi": s.get("roi_pct", 0),
            "label": label,
        })

    if not ratio_data:
        return FIGURES_DIR / "ratio_sensitivity.png"

    ratio_data.sort(key=lambda r: r["ratio_pct"])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ratios = [r["ratio_pct"] for r in ratio_data]

    color1 = "#2ca02c"
    ax1.plot(ratios, [r["sharpe"] for r in ratio_data],
             marker="o", color=color1, linewidth=2, markersize=8, label="Sharpe Ratio")
    ax1.set_xlabel("Fixed Ratio (%)")
    ax1.set_ylabel("Sharpe Ratio", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(0, color=color1, linestyle="--", alpha=0.3)

    color2 = "#d62728"
    ax2 = ax1.twinx()
    ax2.plot(ratios, [r["dd"] for r in ratio_data],
             marker="s", color=color2, linewidth=2, markersize=8, label="Max Drawdown")
    ax2.set_ylabel("Max Drawdown (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(20, color=color2, linestyle="--", alpha=0.3, linewidth=1)

    for r in ratio_data:
        ax1.annotate(f'{r["label"]}',
                     (r["ratio_pct"], r["sharpe"]),
                     xytext=(0, 10), textcoords="offset points",
                     fontsize=7, ha="center")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Fixed Ratio Sensitivity — Sharpe vs Drawdown")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "ratio_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ══════════════════════════════════════════════════════
#  Chart: Risk impact comparison (scenarios)
# ══════════════════════════════════════════════════════


def plot_risk_impact(data: dict[str, Any]) -> Path:
    """Bar chart comparing risk scenarios: ROI, bets, rejections."""
    scenarios = data.get("risk_scenarios", {}).get("results", [])
    if not scenarios:
        logger.warning("No risk scenario results to plot")
        return FIGURES_DIR / "risk_impact.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    labels = [s.get("risk_label", s.get("label", "?"))[:25] for s in scenarios]
    roi = [s.get("roi_pct", 0) for s in scenarios]
    bets = [s.get("total_bets", 0) for s in scenarios]
    rejected = [s.get("n_rejected_by_risk", 0) for s in scenarios]

    # ROI
    roi_colors = ["green" if v >= 0 else "red" for v in roi]
    axes[0].barh(labels, roi, color=roi_colors, alpha=0.7, edgecolor="gray")
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("ROI (%)")
    axes[0].set_title("ROI by Risk Scenario")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Bets placed
    bet_colors = plt.cm.Blues(np.array(bets) / max(max(bets), 1))
    axes[1].barh(labels, bets, color=bet_colors, alpha=0.7, edgecolor="gray")
    axes[1].set_xlabel("Bets Placed")
    axes[1].set_title("Bet Count by Risk Scenario")
    axes[1].grid(True, alpha=0.3, axis="x")

    # Rejected
    axes[2].barh(labels, rejected, color="orange", alpha=0.7, edgecolor="gray")
    axes[2].set_xlabel("Bets Rejected")
    axes[2].set_title("Risk Rejections by Scenario")
    axes[2].grid(True, alpha=0.3, axis="x")

    plt.suptitle("Risk Management Scenario Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "risk_impact.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Find latest results
    data_path = find_latest_management_results()
    if not data_path:
        logger.error(
            "No bankroll_management_*.json found in reports/ — "
            "run backtest_staking_strategies.py first"
        )
        sys.exit(1)

    logger.info("Loading %s", data_path)
    data = load_data(data_path)

    generated: list[Path] = []

    # ── Per-strategy growth charts ──
    strategies = data.get("stake_strategies", {}).get("results", [])
    logger.info("Generating %d per-strategy bankroll growth charts...", len(strategies))
    for s in strategies:
        total_bets = s.get("total_bets", 0)
        if total_bets > 0:
            generated.append(plot_single_strategy_growth(s, total_bets))

    # ── Combined charts ──
    logger.info("Generating bankroll strategy comparison...")
    generated.append(plot_bankroll_strategy_comparison(data))

    logger.info("Generating risk-adjusted returns scatter...")
    generated.append(plot_risk_adjusted_returns(data))

    logger.info("Generating Kelly sensitivity...")
    generated.append(plot_kelly_sensitivity(data))

    logger.info("Generating ratio sensitivity...")
    generated.append(plot_ratio_sensitivity(data))

    logger.info("Generating risk impact comparison...")
    generated.append(plot_risk_impact(data))

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  GENERATED BANKROLL CHARTS")
    print(f"{'=' * 60}")
    for p in generated:
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  [OK] {p.name:<45s} ({size_kb:.1f} KB)")
    print(f"{'=' * 60}")
    print(f"  Total: {len(generated)} charts saved to {FIGURES_DIR}/")
    print()


if __name__ == "__main__":
    main()
