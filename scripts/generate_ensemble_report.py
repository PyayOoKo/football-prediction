#!/usr/bin/env python3
"""
Generate Comprehensive Ensemble Report — markdown with embedded visualizations.

Loads latest ensemble_selection, ensemble_weights_optimised,
ensemble_validation, and ensemble_optimisation data to produce a complete
report with comparison table, performance charts, and recommendations.

Usage:
    python scripts/generate_ensemble_report.py
    python scripts/generate_ensemble_report.py --quiet
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


# ═══════════════════════════════════════════════════════════
#  Load latest data
# ═══════════════════════════════════════════════════════════


def load_latest(pattern: str, exclude: str | None = None) -> dict[str, Any]:
    files = sorted(REPORT_DIR.glob(pattern))
    if exclude:
        files = [f for f in files if exclude not in f.stem]
    f = files[-1] if files else None
    if not f:
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════
#  Visualizations
# ═══════════════════════════════════════════════════════════


def plot_performance_comparison(
    comparison_table: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Grouped bar chart: Brier, LogLoss, Accuracy for all models + ensembles."""
    models = [row["model"] for row in comparison_table]
    brier = [row["brier_score"] for row in comparison_table]
    logloss = [row["log_loss"] for row in comparison_table]
    accuracy = [row["accuracy"] for row in comparison_table]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_b = ax.bar(x - width, brier, width, label="Brier Score (↓)", color="#3498db", alpha=0.85)
    bars_ll = ax.bar(x, logloss, width, label="Log Loss (↓)", color="#e74c3c", alpha=0.85)
    bars_a = ax.bar(x + width, accuracy, width, label="Accuracy (↑)", color="#2ecc71", alpha=0.85)

    # Best-model highlight
    best_brier_idx = int(np.argmin(brier))
    bars_b[best_brier_idx].set_edgecolor("black")
    bars_b[best_brier_idx].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ensemble Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6, alpha=0.7)
    for bar in bars_ll:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6, alpha=0.7)
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6, alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weight_distribution(
    weights: dict[str, float],
    individual_briers: dict[str, float],
    save_path: Path,
) -> None:
    """Horizontal bar chart showing optimised weight per model with Brier overlay."""
    names = list(weights.keys())
    w_vals = [weights[n] for n in names]
    b_vals = [individual_briers.get(n, 0) for n in names]

    # Remove zero-weight models from pie
    non_zero = [(n, w) for n, w in zip(names, w_vals) if w > 0]
    if len(non_zero) < len(names):
        # Bar chart for all
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors_weights = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
        bar_colors = [colors_weights[i % len(colors_weights)] for i in range(len(names))]

        ax1.barh(names, w_vals, color=bar_colors, alpha=0.85, edgecolor="white")
        ax1.set_xlabel("Weight", fontsize=11)
        ax1.set_title("Optimised Ensemble Weights", fontsize=12, fontweight="bold")
        ax1.spines[["top", "right"]].set_visible(False)
        for i, (n, w) in enumerate(zip(names, w_vals)):
            if w > 0:
                ax1.text(w + 0.01, i, f"{w:.1%}", va="center", fontsize=9, fontweight="bold")

        # Brier side-by-side
        ax2.barh(names, b_vals, color="#e74c3c", alpha=0.7, edgecolor="white")
        ax2.set_xlabel("Brier Score (lower is better)", fontsize=11)
        ax2.set_title("Individual Model Brier Scores", fontsize=12, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        for i, (n, b) in enumerate(zip(names, b_vals)):
            ax2.text(b + 0.005, i, f"{b:.4f}", va="center", fontsize=8)

    else:
        # All models have weight → use pie chart for weights
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        labels = [n for n, _ in non_zero]
        sizes = [w for _, w in non_zero]
        colors_pie = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
        explode = [0.03] * len(non_zero)

        ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
                colors=colors_pie[:len(non_zero)], explode=explode,
                shadow=False, textprops={"fontsize": 10})
        ax1.set_title("Optimised Ensemble Weights", fontsize=12, fontweight="bold")

        ax2.barh(names, b_vals, color="#e74c3c", alpha=0.7, edgecolor="white")
        ax2.set_xlabel("Brier Score (lower is better)", fontsize=11)
        ax2.set_title("Individual Model Brier Scores", fontsize=12, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        for i, (n, b) in enumerate(zip(names, b_vals)):
            ax2.text(b + 0.005, i, f"{b:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  Markdown report
# ═══════════════════════════════════════════════════════════


def generate_report(
    selection: dict[str, Any],
    optimisation: dict[str, Any],
    validation: dict[str, Any],
    perf_chart_rel: str | None,
    weight_chart_rel: str | None,
    timestamp: str,
) -> str:
    """Generate the comprehensive markdown report."""
    selected_models = selection.get("selected_models", [])
    optimised_weights = optimisation.get("optimised_weights", {})
    opt_brier = optimisation.get("optimised_brier", 0)
    opt_logloss = optimisation.get("optimised_log_loss", 0)
    opt_accuracy = optimisation.get("optimised_accuracy", 0)
    best_single = optimisation.get("best_single_model", {})
    best_name = best_single.get("name", "?")
    best_brier = best_single.get("brier", 0)
    improvement = optimisation.get("improvement_vs_best_single", 0)
    individual_briers = optimisation.get("individual_briers", {})
    comparison = validation.get("comparison_table", [])
    initial_brier = optimisation.get("initial_brier", 0)
    n_val = optimisation.get("n_validation", 0)

    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Ensemble Report",
        "",
        f"**Generated:** {now}",
        f"**Validation set:** Last {n_val} matches with known outcomes",
        f"**Models selected:** {len(selected_models)}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"An ensemble of **{len(selected_models)} models** was built and optimised via "
        f"grid search over weight combinations to minimise the multi-class Brier score. "
        f"The optimised ensemble achieved a Brier score of **{opt_brier:.4f}** "
        f"(LogLoss: {opt_logloss:.4f}, Accuracy: {opt_accuracy:.2%}), "
        f"compared to the best single model ({best_name}) at **{best_brier:.4f}**.",
        "",
        f"**Improvement vs best single model:** {improvement:+.4f} Brier",
        f"**Improvement vs initial (inverse-Brier) weights:** {initial_brier - opt_brier:+.4f} Brier",
        "",
        "---",
        "",
        "## Selected Base Models",
        "",
        "| Model | Type | Family | Calibrated Brier | Calibration Method |",
        "|-------|------|--------|:----------------:|:------------------:|",
    ]

    for m in selected_models:
        name = m["name"]
        mtype = m["type"]
        family = m["family"]
        brier = m["brier_score"]
        method = m.get("calibration_method") or "raw"
        lines.append(f"| **{name}** | {mtype} | {family} | {brier:.4f} | {method} |")

    lines.extend([
        "",
        "## Optimised Ensemble Weights",
        "",
        "| Model | Weight | Brier Score | Contribution",
        "|-------|:-----:|:-----------:|:-----------:|",
    ])

    for name in optimised_weights:
        w = optimised_weights[name]
        b = individual_briers.get(name, 0)
        contrib = w * (1.0 / max(b, 0.001))
        lines.append(f"| **{name}** | {w:.4f} | {b:.4f} | {contrib:.4f} |")

    lines.append("")
    lines.append(f"Weighted average Brier: **{opt_brier:.4f}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Weight Distribution")
    lines.append("")

    if weight_chart_rel:
        lines.append(f"![Weight Distribution]({weight_chart_rel})")
        lines.append("")
        lines.append("*Left: Optimised ensemble weights. Right: Individual model Brier scores.*")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Performance Comparison")
    lines.append("")

    if perf_chart_rel:
        lines.append(f"![Performance Comparison]({perf_chart_rel})")
        lines.append("")
        lines.append("*Grouped bar chart: Brier Score (blue, lower better), Log Loss (red, lower better), "
                     "Accuracy (green, higher better). Best Brier model highlighted with black border.*")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Detailed Metrics Table")
    lines.append("")
    lines.append("| Model | Brier Score | Log Loss | Accuracy | BTTS Acc | O/U 2.5 Acc |")
    lines.append("|-------|:-----------:|:--------:|:--------:|:--------:|:-----------:|")

    for row in comparison:
        name = row["model"]
        b = row["brier_score"]
        ll = row["log_loss"]
        a = row["accuracy"]
        ba = row.get("btts_accuracy", 0)
        oa = row.get("over25_accuracy", 0)
        suffix = ""
        if name == best_name:
            suffix = "  ← best single"
        elif "Ensemble" in name:
            suffix = "  ← ensemble"
        lines.append(f"| {name}{suffix:<30s} | {b:.4f} | {ll:.4f} | {a:.4f} | {ba:.4f} | {oa:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Analysis",
        "",
        "### Why the ensemble didn't beat the best single model",
        "",
        "The grid search converged to a **Poisson-dominated** weight distribution "
        f"({optimised_weights.get('Poisson', 0)*100:.0f}% Poisson, "
        f"{optimised_weights.get('Elo', 0)*100:.0f}% Elo). This happened because:",
        "",
        "1. **Poisson** (Brier {best_brier:.4f}) significantly outperforms the ML models "
        f"(XGBoost {individual_briers.get('XGBoost', 0):.4f}, "
        f"RF {individual_briers.get('Random Forest', 0):.4f}) on this validation window.",
        "2. The ML models' errors are **not complementary** enough to overcome their worse "
        "individual Brier scores.",
        "3. Adding any non-zero weight to worse models inevitably increases the ensemble's "
        "overall Brier score.",
        "",
        "### When ensembles help most",
        "",
        "Ensembles provide the greatest benefit when:",
        "- Individual models have **comparable performance** but make **different errors**",
        "- Models capture **different signal sources** (statistical vs ML vs rating)",
        "- The validation set is **large enough** to detect subtle complementary patterns",
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "### 1. Rebalance Model Selection",
        "",
        "Consider dropping XGBoost and Random Forest from the ensemble if they consistently "
        "underperform Poisson/Elo on your test window. A **2-model ensemble (Poisson + Elo)** "
        "may perform better with weights proportional to their relative Brier scores.",
        "",
        "### 2. Improve ML Models for This Domain",
        "",
        f"The ML models (Brier {individual_briers.get('XGBoost', 0):.4f}-{individual_briers.get('Random Forest', 0):.4f}) "
        f"underperform statistical models (Brier {individual_briers.get('Elo', 0):.4f}-{best_brier:.4f}) "
        f"on this validation window. Consider:",
        "- Training ML models on the **same features** that make Poisson/Elo successful",
        "- Adding **tournament-specific features** (World Cup vs league, knockout vs group)",
        "- **Calibrating** ML model outputs before ensemble integration",
        "",
        "### 3. Sliding-Window Validation",
        "",
        "Test the ensemble across **multiple validation windows** (different time periods) "
        "to verify the weight distribution is stable and not overfit to a specific window. "
        "Use the `--val-size` flag to test different window sizes.",
        "",
        "### 4. Monitor Weight Drift",
        "",
        "As new match data arrives, model performance may shift. Periodically re-run the "
        "optimisation pipeline to adjust weights. Set up a monthly or quarterly recalibration "
        "schedule.",
        "",
        "---",
        "",
        "## Files Generated",
        "",
        "- `reports/ensemble_selection_*.json` — model selection data",
        "- `reports/ensemble_weights_*.json` — inverse-Brier initial weights",
        "- `reports/ensemble_weights_optimised_*.json` — grid-search optimised weights",
        "- `reports/ensemble_validation_*.json` — full validation comparison table",
        "- `reports/ensemble_optimisation_*.json` — optimisation report with all metrics",
        "- `reports/ensemble_report_*.md` — this report",
        "",
        "---",
        "",
        f"*Report generated automatically by `scripts/generate_ensemble_report.py`*",
        "",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def run_report(quiet: bool = False) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not quiet:
        print("\n" + "=" * 75)
        print("  GENERATE ENSEMBLE REPORT")
        print("=" * 75)

    # ── Load data ──
    if not quiet:
        print("\n  Loading data...")

    selection = load_latest("ensemble_selection_*.json", exclude="_meta")
    optimisation = load_latest("ensemble_optimisation_*.json")
    validation = load_latest("ensemble_validation_*.json")

    if not selection or not optimisation or not validation:
        print("  [FAIL] Missing required data files. Run the full ensemble pipeline first.")
        print(f"    selection={'OK' if selection else 'MISSING'}")
        print(f"    optimisation={'OK' if optimisation else 'MISSING'}")
        print(f"    validation={'OK' if validation else 'MISSING'}")
        sys.exit(1)

    if not quiet:
        print(f"    Selection:     {len(selection.get('selected_models', []))} models")
        print(f"    Optimisation:  Brier={optimisation.get('optimised_brier', '?')}")
        print(f"    Validation:    {len(validation.get('comparison_table', []))} entries")

    # Inject optimised ensemble metrics into comparison table
    # (validation was run before _detect_model_type fix, so its ensemble rows are stale)
    comparison = validation.get("comparison_table", [])
    opt_brier = optimisation.get("optimised_brier")
    if opt_brier is not None:
        # Remove stale ensemble rows
        comparison = [r for r in comparison if "Ensemble" not in r["model"]]
        # Add fresh optimised metrics
        comparison.append({
            "model": "Ensemble (grid-search optimised)",
            "brier_score": opt_brier,
            "log_loss": optimisation.get("optimised_log_loss", 0),
            "accuracy": optimisation.get("optimised_accuracy", 0),
            "btts_accuracy": 0,
            "over25_accuracy": 0,
        })
        # Also add equal-weight reference from validation if available
        for row in validation.get("comparison_table", []):
            if "Ensemble (equal)" in row["model"] or "Ensemble" in row["model"]:
                comparison.append(row)
                break
        validation["comparison_table"] = comparison

    # ── Generate visualizations ──
    if not quiet:
        print("\n  Generating visualizations...")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    perf_chart = FIGURE_DIR / f"ensemble_report_performance_{timestamp}.png"
    if comparison:
        plot_performance_comparison(comparison, perf_chart)
        if not quiet:
            print(f"    Performance chart: {perf_chart.name}")
    else:
        perf_chart = None

    weight_chart = FIGURE_DIR / f"ensemble_report_weights_{timestamp}.png"
    opt_weights = optimisation.get("optimised_weights", {})
    ind_briers = optimisation.get("individual_briers", {})
    if opt_weights:
        plot_weight_distribution(opt_weights, ind_briers, weight_chart)
        if not quiet:
            print(f"    Weight chart: {weight_chart.name}")
    else:
        weight_chart = None

    # ── Generate report ──
    if not quiet:
        print("\n  Generating report...")

    # In generate_report, only embed charts that exist
    def _rel(p: Path | None) -> str | None:
        if p and p.exists():
            return str(p.relative_to(PROJECT_ROOT))
        return None

    report_md = generate_report(
        selection=selection,
        optimisation=optimisation,
        validation=validation,
        perf_chart_rel=_rel(perf_chart),
        weight_chart_rel=_rel(weight_chart),
        timestamp=timestamp,
    )

    report_path = REPORT_DIR / f"ensemble_report_{timestamp}.md"
    report_path.write_text(report_md, encoding="utf-8")

    if not quiet:
        print(f"    Report: {report_path.name}")
        print(f"\n  Report saved with {len(report_md.split(chr(10)))} lines")
        print(f"\n  {'=' * 75}")
        print("  REPORT GENERATED")
        print(f"  {report_path}")
        print("=" * 75)

    return {
        "report_path": str(report_path),
        "perf_chart": str(perf_chart) if perf_chart else "",
        "weight_chart": str(weight_chart) if weight_chart else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate comprehensive ensemble report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    try:
        run_report(quiet=args.quiet)
        return 0
    except Exception as e:
        print(f"\n[FAIL] Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
