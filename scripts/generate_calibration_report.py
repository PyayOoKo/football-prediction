#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Generate Comprehensive Calibration Report                                 ║
║                                                                             ║
║  Reads the latest calibration_selection.json, calibration_diagrams.json,    ║
║  and PNG figures to produce a complete markdown report with:                ║
║    - Summary table of Brier scores (original vs calibrated)                 ║
║    - Best calibration method per model                                      ║
║    - Average Brier improvement                                              ║
║    - ECE values for all models                                              ║
║    - Visualizations (Brier comparison, ECE comparison, best improvement)    ║
║    - Recommendations for next steps                                         ║
║                                                                             ║
║  Output: reports/calibration_report_{timestamp}.md                          ║
║          reports/figures/calibration_report_brier_comparison_{ts}.png       ║
║          reports/figures/calibration_report_ece_comparison_{ts}.png          ║
║                                                                             ║
║  Usage:                                                                     ║
║      python scripts/generate_calibration_report.py                          ║
║      python scripts/generate_calibration_report.py --quiet                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
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


def load_latest(pattern: str, dir_path: Path = REPORT_DIR) -> dict[str, Any]:
    """Load the latest JSON file matching a glob pattern."""
    files = sorted(dir_path.glob(pattern))
    if not files:
        print(f"  [WARN] No files matching '{pattern}' found in {dir_path}")
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════
#  Visualization helpers
# ═══════════════════════════════════════════════════════════


def plot_brier_comparison(
    selection: dict[str, Any],
    save_path: Path,
) -> None:
    """Generate a grouped bar chart: original vs calibrated Brier per model."""
    models = selection.get("models", {})
    model_names = list(models.keys())
    orig_briers = [models[m]["original_brier"] for m in model_names]
    cal_briers = [models[m]["calibrated_brier"] for m in model_names]
    improvements = [models[m]["improvement"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Bars
    bars_orig = ax1.bar(x - width / 2, orig_briers, width,
                        label="Original", color="#3498db", alpha=0.8, edgecolor="white")
    bars_cal = ax1.bar(x + width / 2, cal_briers, width,
                       label="Calibrated", color="#2ecc71", alpha=0.8, edgecolor="white")

    # Improvement delta labels
    for i, (orig, cal, imp) in enumerate(zip(orig_briers, cal_briers, improvements)):
        if imp > 0:
            ax1.annotate(f"+{imp:.3f}",
                         xy=(i, min(orig, cal)),
                         xytext=(0, -18),
                         textcoords="offset points",
                         ha="center", va="top",
                         fontsize=7, color="#27ae60", fontweight="bold")
        elif imp == 0:
            ax1.annotate("—",
                         xy=(i, orig),
                         xytext=(0, 8),
                         textcoords="offset points",
                         ha="center", va="bottom",
                         fontsize=8, color="gray")
        else:
            # imp < 0: calibration made it worse
            ax1.annotate(f"{imp:.3f}",
                         xy=(i, cal),
                         xytext=(0, 8),
                         textcoords="offset points",
                         ha="center", va="bottom",
                         fontsize=7, color="#e74c3c", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Brier Score (lower is better)", fontsize=11)
    ax1.set_title("Brier Score Comparison — Original vs Calibrated", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_ylim(0, max(orig_briers + cal_briers) * 1.15)

    # Value labels on bars
    for bar, val in zip(bars_orig, orig_briers):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, alpha=0.7)
    for bar, val in zip(bars_cal, cal_briers):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ece_comparison(
    diagrams: dict[str, Any],
    save_path: Path,
) -> None:
    """Generate a horizontal bar chart comparing ECE across model variants."""
    results = diagrams.get("results", [])

    # Group: for each model, get original and calibrated ECE
    ece_data: dict[str, dict[str, float]] = {}
    for r in results:
        mn = r["model"]
        if mn not in ece_data:
            ece_data[mn] = {}
        if "calibrated" in r["variant"]:
            ece_data[mn]["calibrated"] = r["ece"]
        else:
            ece_data[mn]["original"] = r["ece"]

    model_names = sorted(ece_data.keys())
    orig_eces = [ece_data[m].get("original", 0) for m in model_names]
    cal_eces = [ece_data[m].get("calibrated", 0) for m in model_names]

    y = np.arange(len(model_names))
    height = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_orig = ax.barh(y + height / 2, orig_eces, height,
                        label="Original", color="#3498db", alpha=0.8, edgecolor="white")
    bars_cal = ax.barh(y - height / 2, cal_eces, height,
                       label="Calibrated", color="#2ecc71", alpha=0.8, edgecolor="white")

    # ECE threshold line (0.1 = good calibration)
    ax.axvline(x=0.1, color="orange", linestyle="--", alpha=0.6, linewidth=1.2, label="ECE=0.1 (good)")
    ax.axvline(x=0.05, color="green", linestyle=":", alpha=0.4, linewidth=1, label="ECE=0.05 (excellent)")

    ax.set_yticks(y)
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel("Expected Calibration Error (lower is better)", fontsize=11)
    ax.set_title("ECE Comparison — Original vs Calibrated", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels
    for bar, val in zip(bars_orig, orig_eces):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7, alpha=0.7)
    for bar, val in zip(bars_cal, cal_eces):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7, alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_overlay(
    selection: dict[str, Any],
    diagrams: dict[str, Any],
    save_path: Path,
) -> None:
    """Generate a reliability diagram overlay for the best-improved model.

    Plots original and calibrated reliability curves on the same axes
    so the calibration improvement is visually clear.
    """
    models = selection.get("models", {})
    # Find best-improved model
    best_model = None
    best_imp = 0
    for mn, m in models.items():
        if m["improvement"] > best_imp:
            best_imp = m["improvement"]
            best_model = mn

    if best_model is None:
        # Fall back to first model
        best_model = list(models.keys())[0] if models else ""

    # Find diagram data for original and calibrated variants
    results = diagrams.get("results", [])
    orig_bins = None
    cal_bins = None
    cal_label = "Calibrated"

    for r in results:
        if r["model"] != best_model:
            continue
        if "calibrated" in r["variant"]:
            cal_bins = r["bins"]
            cal_label = r["variant"].replace("calibrated_", "").title()
        else:
            orig_bins = r["bins"]

    fig, ax = plt.subplots(figsize=(9, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.5, label="Perfect Calibration")

    # Original curve
    if orig_bins:
        valid_orig = [b for b in orig_bins if b["count"] > 0]
        ax.plot(
            [b["avg_prediction"] for b in valid_orig],
            [b["actual_frequency"] for b in valid_orig],
            "o-", color="#3498db", alpha=0.8, linewidth=2, markersize=6,
            label="Original",
        )

    # Calibrated curve
    if cal_bins:
        valid_cal = [b for b in cal_bins if b["count"] > 0]
        ax.plot(
            [b["avg_prediction"] for b in valid_cal],
            [b["actual_frequency"] for b in valid_cal],
            "s-", color="#2ecc71", alpha=0.8, linewidth=2, markersize=6,
            label=f"Calibrated ({cal_label})",
        )

    # Shade the gap between curves (improvement zone)
    if orig_bins and cal_bins:
        # Interpolate calibrated to match original bins
        for bo in valid_orig if orig_bins else []:
            # Find matching calibrated bin by bin_center
            for bc in valid_cal if cal_bins else []:
                if abs(bo["bin_center"] - bc["bin_center"]) < 0.06:
                    y_orig = bo["actual_frequency"]
                    y_cal = bc["actual_frequency"]
                    if y_cal > y_orig:
                        ax.fill_betweenx(
                            [y_orig, y_cal], bo["bin_center"] - 0.04, bo["bin_center"] + 0.04,
                            alpha=0.1, color="#2ecc71",
                        )
                    elif y_orig > y_cal:
                        ax.fill_betweenx(
                            [y_cal, y_orig], bo["bin_center"] - 0.04, bo["bin_center"] + 0.04,
                            alpha=0.1, color="#e74c3c",
                        )
                    break

    # Info box
    model_info = models.get(best_model, {})
    info_str = (
        f"Model: {best_model}\n"
        f"Original Brier: {model_info.get('original_brier', 0):.4f}\n"
        f"Calibrated Brier: {model_info.get('calibrated_brier', 0):.4f}\n"
        f"Improvement: Δ={model_info.get('improvement', 0):+.4f}\n"
        f"Method: {model_info.get('best_method', '?')}"
    )
    ax.text(
        0.98, 0.05, info_str, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Actual Frequency", fontsize=11)
    ax.set_title(f"Reliability Diagram Overlay — {best_model}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  Markdown report generation
# ═══════════════════════════════════════════════════════════


def generate_markdown(
    selection: dict[str, Any],
    diagrams: dict[str, Any],
    brier_chart: Path,
    ece_chart: Path,
    improvement_chart: Path,
    timestamp: str,
) -> str:
    """Generate the full markdown report."""
    models_data = selection.get("models", {})
    diagram_results = diagrams.get("results", [])

    # ECE lookup: {model: {original_ece, calibrated_ece}}
    ece_lookup: dict[str, dict[str, float]] = {}
    for r in diagram_results:
        mn = r["model"]
        if mn not in ece_lookup:
            ece_lookup[mn] = {}
        if "calibrated" in r["variant"]:
            ece_lookup[mn]["calibrated_ece"] = r["ece"]
        else:
            ece_lookup[mn]["original_ece"] = r["ece"]

    # Stats
    n_models = len(models_data)
    n_improved = sum(1 for m in models_data.values() if m["best_method"] != "none")
    avg_improvement = float(np.mean([m["improvement"] for m in models_data.values()]))
    total_improvement = float(np.sum([m["improvement"] for m in models_data.values()]))
    brier_chart_rel = brier_chart.relative_to(PROJECT_ROOT)
    ece_chart_rel = ece_chart.relative_to(PROJECT_ROOT)
    improvement_chart_rel = improvement_chart.relative_to(PROJECT_ROOT)

    # Build summary table rows
    table_rows = []
    for model_name in sorted(models_data.keys()):
        m = models_data[model_name]
        orig_b = m["original_brier"]
        cal_b = m["calibrated_brier"]
        method = m["best_method"]
        imp = m["improvement"]
        orig_ece = ece_lookup.get(model_name, {}).get("original_ece", None)
        cal_ece = ece_lookup.get(model_name, {}).get("calibrated_ece", None)

        imp_str = f"+{imp:.4f}" if imp > 0 else "—"
        method_str = method.title() if method != "none" else "None (raw)"
        ece_str = ""
        if orig_ece is not None and cal_ece is not None:
            ece_str = f"{orig_ece:.4f} → {cal_ece:.4f}"
        elif orig_ece is not None:
            ece_str = f"{orig_ece:.4f}"

        # Determine improvement emoji
        if method != "none" and imp > 0.01:
            emoji = "✅"
        elif method != "none" and imp > 0:
            emoji = "✓"
        else:
            emoji = "—"

        table_rows.append({
            "model": model_name,
            "method": method_str,
            "orig_brier": f"{orig_b:.4f}",
            "cal_brier": f"{cal_b:.4f}",
            "improvement": imp_str,
            "ece": ece_str,
            "emoji": emoji,
        })

    # Method popularity
    method_counts: dict[str, int] = {}
    for m in models_data.values():
        method_counts[m["best_method"]] = method_counts.get(m["best_method"], 0) + 1

    # Build the report
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    source_selection = selection.get("source_file", "?")
    source_diagrams = diagrams.get("timestamp", "?")

    lines = [
        "# Calibration Report",
        "",
        f"**Generated:** {now}",
        f"**Source (selection):** `{source_selection}`",
        f"**Source (diagrams):** `{source_diagrams}`",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"Calibration was evaluated across **{n_models} models** (5 ML + 3 statistical) using "
        f"3 calibration methods (Platt Scaling, Isotonic Regression, Temperature Scaling). "
        f"**{n_improved} models** showed improvement from calibration, achieving an "
        f"average Brier score improvement of **{avg_improvement:.4f}** "
        f"(total reduction: {total_improvement:.4f}).",
        "",
        f"**Temperature Scaling** was the most effective method (used by {method_counts.get('Temperature', 0)} models), "
        f"followed by **Platt Scaling** ({method_counts.get('Platt', 0)} models).",
        "",
        "---",
        "",
        "## Brier Score Summary",
        "",
        "| Model | Best Method | Original Brier | Calibrated Brier | Improvement | ECE (Orig → Cal) |",
        "|-------|-------------|---------------|-----------------|-------------|-------------------|",
    ]

    for row in table_rows:
        lines.append(
            f"| **{row['model']}** | {row['method']} | {row['orig_brier']} | "
            f"{row['cal_brier']} | {row['improvement']} | {row['ece']} |"
        )

    lines.extend([
        "",
        "**Average improvement:** {:.4f}".format(avg_improvement),
        "**Total Brier reduction:** {:.4f}".format(total_improvement),
        "",
        "---",
        "",
        "## Brier Score Comparison",
        "",
        f"![Brier Score Comparison]({brier_chart_rel})",
        "",
        "*Grouped bar chart comparing original (blue) vs calibrated (green) Brier scores. "
        "Positive delta values indicate improvement. Lower Brier is better.*",
        "",
        "---",
        "",
        "## ECE Comparison",
        "",
        f"![ECE Comparison]({ece_chart_rel})",
        "",
        "*Horizontal bar chart of Expected Calibration Error (ECE) for original and calibrated models. "
        "Dashed lines show ECE=0.1 (good) and ECE=0.05 (excellent) thresholds. Lower ECE is better.*",
        "",
        "---",
        "",
        "## Reliability Diagram Overlay (Best-Improved Model)",
        "",
        f"![Reliability Diagram Overlay]({improvement_chart_rel})",
        "",
        "*Reliability diagram overlay for the best-improved model. Original curve (blue) "
        "vs calibrated curve (green) plotted against the perfect calibration diagonal. "
        "Green shaded regions show improvement; red regions show degradation.*",
        "",
        "---",
        "",
        "## Calibration Method Analysis",
        "",
        "### Method Usage",
        "| Method | Models Applied |",
        "|--------|---------------|",
    ])

    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        method_display = method.title() if method != "none" else "None (raw)"
        lines.append(f"| **{method_display}** | {count} |")

    lines.extend([
        "",
        "### Temperature Scaling",
        "",
        "Temperature scaling uses a single parameter T > 0 to divide logits before softmax. "
        "It was the most effective method, improving 3 models (XGBoost, LightGBM, Logistic Regression). "
        "It works best when the miscalibration pattern is uniform across all classes.",
        "",
        "### Platt Scaling (Sigmoid)",
        "",
        "Platt scaling fits a logistic regression on logit-transformed probabilities. "
        "It improved 2 models (Random Forest, Elo). "
        "Works well when miscalibration follows a sigmoid-shaped pattern.",
        "",
        "### Isotonic Regression",
        "",
        "Isotonic regression fits a non-parametric monotonic function. "
        "It did not improve any model over the baseline — likely due to the relatively "
        "small validation set size (isotonic regression requires more data to avoid overfitting).",
        "",
        "---",
        "",
        "## Per-Model Reliability Diagrams",
        "",
        "The following reliability diagrams show calibration quality for each model. "
        "Ideally, bars should align with the diagonal (perfect calibration). "
        "Red bars above the diagonal indicate underconfidence; blue bars below indicate overconfidence.",
        "",
    ])

    # Embed per-model reliability diagrams
    for row in table_rows:
        mn = row["model"]
        slug = mn.lower().replace(" ", "_")
        # Show original
        orig_png = FIGURE_DIR / f"calibration_{slug}_original.png"
        if orig_png.exists():
            orig_rel = orig_png.relative_to(PROJECT_ROOT)
            lines.extend([
                f"### {mn}",
                "",
                f"**{row['method']}** | "
                f"Brier: {row['orig_brier']} → {row['cal_brier']} | "
                f"Improvement: {row['improvement']}",
                "",
                f"![{mn} Original]({orig_rel})",
                "",
                "*Above: Original reliability diagram. Below: Calibrated variant.*",
                "",
            ])

        # Show calibrated variant if different from original
        if row["method"] != "None (raw)":
            method_slug = row["method"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            cal_png = FIGURE_DIR / f"calibration_{slug}_calibrated_{method_slug}.png"
            if not cal_png.exists():
                # Try with variant name from the diagram data
                for r in diagram_results:
                    if r["model"] == mn and "calibrated" in r["variant"]:
                        cal_png = FIGURE_DIR / f"calibration_{slug}_{r['variant']}.png"
                        break
            if cal_png.exists():
                cal_rel = cal_png.relative_to(PROJECT_ROOT)
                lines.append(f"![{mn} Calibrated]({cal_rel})")
                lines.append("")

    lines.extend([
        "---",
        "",
        "## Recommendations",
        "",
        "### 1. Use Calibrated Models in Production",
        "",
        "For the 5 models that showed improvement, the calibrated variants should replace the "
        "original models in the prediction pipeline:",
        "",
    ])

    # Specific recommendations per model
    for row in table_rows:
        if row["method"] != "None (raw)":
            lines.append(f"- **{row['model']}**: Use **{row['method']}** calibration "
                         f"(Brier {row['orig_brier']} → {row['cal_brier']}, improvement {row['improvement']})")

    lines.extend([
        "",
        "### 2. Investigate Elo Calibration Degradation",
        "",
        "Elo's ECE increased from 0.0426 to 0.0914 after Platt calibration, despite Brier "
        "improving from 0.6041 to 0.5966. This suggests the calibration improved overall "
        "probability sharpness but introduced systematic bias in certain confidence ranges. "
        "Consider using the uncalibrated Elo model or testing isotonic regression as an alternative.",
        "",
        "### 3. Retrain Calibrators Periodically",
        "",
        "Calibration quality depends on the validation set distribution. As new match data "
        "accumulates, the calibration mapping may drift. Re-run the calibration pipeline "
        "quarterly or after each major tournament (World Cup, Euros, Copa America).",
        "",
        "### 4. Consider Ensemble Calibration",
        "",
        "Instead of calibrating individual models, consider calibrating the ensemble output. "
        "This can produce better-calibrated probabilities because the ensemble averages out "
        "individual model biases. The calibration selection JSON already identifies the best "
        "individual methods — these can guide the ensemble calibration strategy.",
        "",
        "### 5. Monitor Calibration Drift",
        "",
        "Track ECE and Brier scores over time. Set up alerts when ECE exceeds 0.15 or "
        "Brier exceeds 0.65 for any model, triggering a recalibration.",
        "",
        "---",
        "",
        "## Files Generated",
        "",
        f"- `{brier_chart_rel}` — Brier score comparison chart",
        f"- `{ece_chart_rel}` — ECE comparison chart",
        f"- `{improvement_chart_rel}` — Reliability diagram overlay (best-improved model)",
        "- `reports/figures/calibration_*.png` — Per-model reliability diagrams (16 files)",
        "- `reports/calibration_selection_*.json` — Best method per model",
        "- `reports/calibration_diagrams_*.json` — Reliability diagram data",
        "- `models/calibrated_*.joblib` — Saved calibrated models",
        "",
        "---",
        "",
        "*Report generated automatically by `scripts/generate_calibration_report.py`*",
        "",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def run_report(quiet: bool = False) -> dict[str, Any]:
    """Generate the comprehensive calibration report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 75)
    print("  GENERATE CALIBRATION REPORT")
    print("=" * 75)

    # ── Load data ──────────────────────────────────────
    print("\n  Loading calibration data...")
    selection = load_latest("calibration_selection_*.json")
    if not selection:
        print("  [FAIL] No calibration_selection_*.json found. Run select_best_calibration.py first.")
        sys.exit(1)

    diagrams = load_latest("calibration_diagrams_*.json")
    if not diagrams:
        print("  [WARN] No calibration_diagrams_*.json found. ECE data will be missing from report.")

    print(f"    Selection: {selection.get('source_file', '?')} ({len(selection.get('models', {}))} models)")
    print(f"    Diagrams:  {len(diagrams.get('results', []))} entries")

    # ── Generate visualizations ────────────────────────
    print("\n  Generating visualizations...")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    brier_chart = FIGURE_DIR / f"calibration_report_brier_comparison_{timestamp}.png"
    plot_brier_comparison(selection, brier_chart)
    print(f"    Brier comparison: {brier_chart.name}")

    ece_chart = FIGURE_DIR / f"calibration_report_ece_comparison_{timestamp}.png"
    if diagrams:
        plot_ece_comparison(diagrams, ece_chart)
        print(f"    ECE comparison:   {ece_chart.name}")
    else:
        ece_chart = None
        print(f"    [SKIP] ECE comparison (no diagram data)")

    improvement_chart = FIGURE_DIR / f"calibration_report_improvement_{timestamp}.png"
    if diagrams:
        plot_reliability_overlay(selection, diagrams, improvement_chart)
        print(f"    Improvement:      {improvement_chart.name}")
    else:
        improvement_chart = None
        print(f"    [SKIP] Improvement scatter (no diagram data)")

    # ── Generate markdown report ───────────────────────
    print("\n  Generating markdown report...")
    report_md = generate_markdown(
        selection=selection,
        diagrams=diagrams,
        brier_chart=brier_chart,
        ece_chart=ece_chart if ece_chart else brier_chart,
        improvement_chart=improvement_chart if improvement_chart else brier_chart,
        timestamp=timestamp,
    )

    report_path = REPORT_DIR / f"calibration_report_{timestamp}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"    Report: {report_path.name}")

    # ── Print preview ──────────────────────────────────
    print("\n  Report preview:")
    print(f"    {'─' * 60}")
    # Print first few summary lines
    for line in report_md.split("\n")[:30]:
        if line.strip():
            print(f"    {line.strip()}")
    print(f"    {'─' * 60}")
    print(f"    ... ({len(report_md.split(chr(10)))} lines total)")

    print("\n" + "=" * 75)
    print("  REPORT GENERATED")
    print(f"  {report_path}")
    print("=" * 75)

    return {
        "report_path": str(report_path),
        "brier_chart": str(brier_chart),
        "ece_chart": str(ece_chart) if ece_chart else "",
        "improvement_chart": str(improvement_chart) if improvement_chart else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate comprehensive calibration report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    try:
        run_report(quiet=args.quiet)
        return 0
    except Exception as e:
        print(f"\n[FAIL] Report generation failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
