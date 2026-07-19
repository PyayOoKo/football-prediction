#!/usr/bin/env python3
"""
generate_calibration_report.py — Detailed calibration analysis with
reliability diagrams, ECE per probability bin, and per-class breakdown.

Generates:
  - Reliability diagrams (PNG) for each calibrator + per-class
  - Detailed bin-by-bin calibration table (10 bins x 3 classes)
  - ECE, MCE, Brier, Log-loss for raw vs each calibrator
  - Full HTML report with embedded charts
  - JSON report with machine-readable metrics

Usage:
    python scripts/generate_calibration_report.py
    python scripts/generate_calibration_report.py --data league
    python scripts/generate_calibration_report.py --no-plots  # text-only report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────


def _ece_from_bins(counts: np.ndarray, acc: np.ndarray, conf: np.ndarray) -> float:
    """Weighted ECE: sum(bin_weight * |acc - conf|)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    return float(np.sum(counts / total * np.abs(acc - conf)))


def _compute_bin_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute per-bin calibration metrics.

    Returns dict with keys:
        bin_edges, bin_centers, bin_counts, bin_accuracy, bin_confidence,
        bin_gap, ece, mce, n_total, class_counts, total_correct
    """
    pred_class = np.argmax(y_prob, axis=1)
    pred_conf = np.max(y_prob, axis=1)
    correct = (pred_class == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    bin_counts = np.zeros(n_bins, dtype=float)
    bin_accuracy = np.zeros(n_bins, dtype=float)
    bin_confidence = np.zeros(n_bins, dtype=float)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Last bin includes the right edge
        if i == n_bins - 1:
            mask = (pred_conf >= lo) & (pred_conf <= hi)
        else:
            mask = (pred_conf >= lo) & (pred_conf < hi)
        cnt = mask.sum()
        bin_counts[i] = cnt
        if cnt > 0:
            bin_accuracy[i] = float(correct[mask].mean())
            bin_confidence[i] = float(pred_conf[mask].mean())

    ece = _ece_from_bins(bin_counts, bin_accuracy, bin_confidence)
    gaps = np.abs(bin_accuracy - bin_confidence)
    mce = float(gaps.max()) if len(gaps) > 0 else 0.0

    return {
        "bin_edges": bins.tolist(),
        "bin_centers": bin_centers.tolist(),
        "bin_counts": bin_counts.tolist(),
        "bin_accuracy": bin_accuracy.tolist(),
        "bin_confidence": bin_confidence.tolist(),
        "bin_gap": gaps.tolist(),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "n_total": int(len(y_true)),
        "n_correct": int(correct.sum()),
    }


def _compute_per_class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    n_bins: int = 10,
) -> dict[str, dict]:
    """Compute per-class calibration metrics (one-vs-rest)."""
    n_classes = len(class_names)
    results: dict[str, dict] = {}
    for c in range(n_classes):
        y_binary = (y_true == c).astype(float)
        probs_c = y_prob[:, c]

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_counts = np.zeros(n_bins, dtype=float)
        bin_accuracy = np.zeros(n_bins, dtype=float)
        bin_confidence = np.zeros(n_bins, dtype=float)

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs_c >= lo) & (probs_c < hi) if i < n_bins - 1 else (probs_c >= lo) & (probs_c <= hi)
            cnt = mask.sum()
            bin_counts[i] = cnt
            if cnt > 0:
                bin_accuracy[i] = float(y_binary[mask].mean())
                bin_confidence[i] = float(probs_c[mask].mean())

        ece_class = _ece_from_bins(bin_counts, bin_accuracy, bin_confidence)
        brier = float(np.mean((probs_c - y_binary) ** 2))

        results[class_names[c]] = {
            "ece": round(ece_class, 4),
            "brier": round(brier, 4),
            "n_samples": int(y_binary.sum()),
            "bin_counts": bin_counts.tolist(),
            "bin_accuracy": bin_accuracy.tolist(),
            "bin_confidence": bin_confidence.tolist(),
        }
    return results


def _compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    label: str = "",
) -> dict:
    """Compute comprehensive calibration metrics."""
    from sklearn.metrics import log_loss as sk_log_loss

    y_true_int = y_true.astype(np.int64)
    y_onehot = np.eye(y_prob.shape[1])[y_true_int]
    ll = float(sk_log_loss(y_true, y_prob))
    brier = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

    bin_metrics = _compute_bin_metrics(y_true, y_prob, n_bins=10)
    per_class = _compute_per_class_metrics(y_true, y_prob, class_names, n_bins=10)

    return {
        "label": label,
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "ece": bin_metrics["ece"],
        "mce": bin_metrics["mce"],
        "n_total": bin_metrics["n_total"],
        "n_correct": bin_metrics["n_correct"],
        "bin_centers": bin_metrics["bin_centers"],
        "bin_counts": bin_metrics["bin_counts"],
        "bin_accuracy": bin_metrics["bin_accuracy"],
        "bin_confidence": bin_metrics["bin_confidence"],
        "bin_gap": bin_metrics["bin_gap"],
        "per_class": per_class,
    }


# ── Plotting ─────────────────────────────────────────────────────────


def _plot_reliability_diagram(
    metrics_list: list[dict],
    save_path: Path,
    class_names: list[str],
) -> None:
    """Create a 2-panel figure: reliability curve + histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cals = len(metrics_list)
    fig, axes = plt.subplots(2, n_cals, figsize=(6 * n_cals, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    if n_cals == 1:
        axes = axes.reshape(2, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_cals))

    for idx, m in enumerate(metrics_list):
        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]

        centers = np.array(m["bin_centers"])
        acc = np.array(m["bin_accuracy"])
        conf = np.array(m["bin_confidence"])
        counts = np.array(m["bin_counts"])

        # Reliability curve
        ax_top.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
        ax_top.plot(centers, acc, "o-", color=colors[idx], lw=2, ms=6, label=f"{m['label']}")
        ax_top.fill_between(centers, conf, acc, alpha=0.15, color=colors[idx])

        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax_top.set_ylabel("Observed Frequency", fontsize=11)
        ax_top.set_title(
            f"{m['label']}\nECE={m['ece']:.4f}  MCE={m['mce']:.4f}  Brier={m['brier']:.4f}",
            fontsize=12, fontweight="bold",
        )
        ax_top.legend(loc="lower right", fontsize=9)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_aspect("equal")

        # Histogram
        nonzero = counts > 0
        if nonzero.any():
            ax_bot.bar(centers[nonzero], counts[nonzero], width=0.09,
                       color=colors[idx], alpha=0.7, edgecolor="white", lw=0.5)
        ax_bot.set_xlim(0, 1)
        ax_bot.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax_bot.set_ylabel("Count", fontsize=11)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_reliability(
    metrics_list: list[dict],
    save_path: Path,
    class_names: list[str],
) -> None:
    """Plot per-class reliability curves in a single figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_list)))

    for c_idx, cname in enumerate(class_names):
        ax = axes[c_idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")

        for m_idx, m in enumerate(metrics_list):
            pc = m["per_class"].get(cname, {})
            centers = np.array(m["bin_centers"])
            acc = np.array(pc.get("bin_accuracy", []))
            if len(acc) > 0:
                ax.plot(centers, acc, "o-", color=colors[m_idx], lw=2,
                        ms=5, label=f"{m['label']} (ECE={pc.get('ece', '?'):})")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Probability", fontsize=11)
        ax.set_ylabel("Observed Frequency", fontsize=11)
        ax.set_title(f"Class: {cname}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gap_bar_chart(
    metrics_list: list[dict],
    save_path: Path,
) -> None:
    """Plot per-bin calibration gap (|acc - conf|) as grouped bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not metrics_list:
        return

    centers = np.array(metrics_list[0]["bin_centers"])
    n_bins = len(centers)
    n_cals = len(metrics_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / n_cals
    colors = plt.cm.tab10(np.linspace(0, 1, n_cals))

    for idx, m in enumerate(metrics_list):
        gaps = np.array(m["bin_gap"])
        offset = (idx - (n_cals - 1) / 2) * width
        ax.bar(centers + offset, gaps, width, label=m["label"],
               color=colors[idx], alpha=0.8, edgecolor="white", lw=0.5)

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("|Accuracy - Confidence|", fontsize=11)
    ax.set_title("Per-Bin Calibration Error (Gap)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── HTML Report ──────────────────────────────────────────────────────


def _html_image_section(title: str, rel_path: str | None) -> str:
    """Render an <img> tag with the given title and relative path, or empty string."""
    if not rel_path:
        return ""
    return (
        f"\n<h2>{title}</h2>\n"
        f'<img class="plot" src="{rel_path}" alt="{title}">\n'
    )


def _generate_html_report(
    raw_metrics: dict,
    cal_metrics_list: list[dict],
    class_names: list[str],
    plot_paths: dict[str, Path],
    hybrid_eval: dict | None = None,
) -> str:
    """Generate a self-contained HTML report."""
    # ── Plot image relative paths (for HTML embedding) ──
    rel_reliability = plot_paths.get("reliability")
    rel_reliability = rel_reliability.name if rel_reliability else None
    rel_per_class = plot_paths.get("per_class")
    rel_per_class = rel_per_class.name if rel_per_class else None
    rel_gap = plot_paths.get("gap_chart")
    rel_gap = rel_gap.name if rel_gap else None

    # ── Build comparison table ──
    rows_html = ""
    all_metrics = [raw_metrics] + cal_metrics_list
    for m in all_metrics:
        tag = m["label"]
        cls = ' class="raw-row"' if tag == "Raw" else ""
        rows_html += f"""
            <tr{cls}>
                <td><strong>{tag}</strong></td>
                <td>{m['log_loss']}</td>
                <td>{m['brier']}</td>
                <td>{m['ece']}</td>
                <td>{m['mce']}</td>
                <td>{m['n_correct']}/{m['n_total']} ({m['n_correct']/m['n_total']*100:.1f}%)</td>
            </tr>"""

    # ── Per-class breakdown ──
    per_class_rows = ""
    for m in all_metrics:
        tag = m["label"]
        for cname in class_names:
            pc = m["per_class"].get(cname, {})
            per_class_rows += f"""
            <tr>
                <td>{tag}</td>
                <td>{cname}</td>
                <td>{pc.get('ece', 'N/A')}</td>
                <td>{pc.get('brier', 'N/A')}</td>
                <td>{pc.get('n_samples', 0)}</td>
            </tr>"""

    # ── Bin table ──
    bin_rows = ""
    for i in range(len(raw_metrics["bin_centers"])):
        center = raw_metrics["bin_centers"][i]
        count = int(raw_metrics["bin_counts"][i])
        raw_acc = raw_metrics["bin_accuracy"][i]
        raw_conf = raw_metrics["bin_confidence"][i]
        raw_gap = raw_metrics["bin_gap"][i]

        # Best calibrator for this bin
        best_gap = raw_gap
        best_cal = "—"
        for m in cal_metrics_list:
            g = m["bin_gap"][i]
            if g < best_gap:
                best_gap = g
                best_cal = m["label"]

        gap_color = "green" if raw_gap < 0.05 else ("orange" if raw_gap < 0.15 else "red")
        bin_rows += f"""
            <tr>
                <td>{center:.1%}</td>
                <td>{count}</td>
                <td>{raw_acc:.1%}</td>
                <td>{raw_conf:.1%}</td>
                <td style="color:{gap_color}; font-weight:bold">{raw_gap:.1%}</td>
                <td>{best_cal}</td>
            </tr>"""

    # ── Hybrid tail evaluation ──
    hybrid_html = ""
    if hybrid_eval:
        hybrid_html = f"""
        <h2>5. HybridTailCalibrator — Tail Analysis</h2>
        <p>Detailed evaluation of the <strong>HybridTailCalibrator</strong> across probability regions.</p>
        <table class="dataframe">
            <tr><th>Metric</th><th>Raw</th><th>Calibrated</th><th>Improvement</th></tr>
            <tr><td>Log-loss</td><td>{hybrid_eval.get('raw_log_loss', '—')}</td>
                <td>{hybrid_eval.get('calibrated_log_loss', '—')}</td>
                <td style="color:green">+{hybrid_eval.get('log_loss_improvement', '—')}</td></tr>
            <tr><td>Brier</td><td>{hybrid_eval.get('raw_brier', '—')}</td>
                <td>{hybrid_eval.get('calibrated_brier', '—')}</td>
                <td style="color:green">+{hybrid_eval.get('brier_improvement', '—')}</td></tr>
            <tr><th colspan="4">Per-Region ECE (Expected Calibration Error)</th></tr>
            <tr><td>Low tail (p &lt; {hybrid_eval.get('tail_threshold', 0.1):.0%})</td>
                <td colspan="3">{hybrid_eval.get('low_tail_ece', '—')} &nbsp;(n={hybrid_eval.get('low_tail_samples', 0)})</td></tr>
            <tr><td>Mid-range ({hybrid_eval.get('tail_threshold', 0.1):.0%} ≤ p ≤ {1 - hybrid_eval.get('tail_threshold', 0.1):.0%})</td>
                <td colspan="3">{hybrid_eval.get('mid_ece', '—')} &nbsp;(n={hybrid_eval.get('mid_samples', 0)})</td></tr>
            <tr><td>High tail (p &gt; {1 - hybrid_eval.get('tail_threshold', 0.1):.0%})</td>
                <td colspan="3">{hybrid_eval.get('high_tail_ece', '—')} &nbsp;(n={hybrid_eval.get('high_tail_samples', 0)})</td></tr>
        </table>
        <p><em>Mid-region isotonic weight: {hybrid_eval.get('mid_isotonic_weight', 0.3):.0%}</em></p>
        """

    # ── Build full HTML ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Calibration Report — Football Predictions</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1100px; margin: 20px auto; padding: 0 20px;
           background: #f8f9fa; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 8px; }}
    h2 {{ color: #16213e; margin-top: 30px; }}
    .subtitle {{ color: #666; font-size: 14px; margin-top: -10px; }}
    table.dataframe {{ border-collapse: collapse; width: 100%; margin: 15px 0;
                       box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    table.dataframe th {{ background: #16213e; color: white; padding: 10px 12px;
                          text-align: left; font-size: 13px; }}
    table.dataframe td {{ padding: 8px 12px; border-bottom: 1px solid #ddd; font-size: 13px; }}
    table.dataframe tr:hover {{ background: #f1f3f5; }}
    .raw-row {{ background: #fff3cd; }}
    img.plot {{ width: 100%; max-width: 1100px; margin: 20px 0;
                border: 1px solid #ddd; border-radius: 4px; }}
    .summary-box {{ background: #e8f4f8; border-left: 4px solid #16213e;
                    padding: 15px; margin: 20px 0; border-radius: 4px; }}
    .metric-big {{ font-size: 24px; font-weight: bold; color: #16213e; }}
    .verdict {{ background: #d4edda; border-left: 4px solid #28a745; padding: 15px;
                margin: 20px 0; border-radius: 4px; }}
    .footer {{ margin-top: 40px; color: #999; font-size: 12px; text-align: center; }}
</style>
</head>
<body>

<h1>📊 Calibration Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &mdash; {raw_metrics['n_total']:,} test samples</p>

<div class="summary-box">
    <h3>Summary</h3>
    <p>
        <span class="metric-big">{raw_metrics['ece']:.4f}</span> Raw ECE &rarr;
        <span class="metric-big">{min(m['ece'] for m in cal_metrics_list):.4f}</span> Best calibrated ECE
        &nbsp;|&nbsp;
        <span class="metric-big">{raw_metrics['log_loss']:.4f}</span> Raw log-loss &rarr;
        <span class="metric-big">{min(m['log_loss'] for m in cal_metrics_list):.4f}</span> Best calibrated
    </p>
</div>

<h2>1. Overall Calibration Metrics</h2>
<table class="dataframe">
    <tr><th>Method</th><th>Log-loss</th><th>Brier</th><th>ECE</th><th>MCE</th><th>Accuracy</th></tr>
    {rows_html}
</table>

{_html_image_section('2. Reliability Diagram', rel_reliability)}
{_html_image_section('3. Per-Class Calibration', rel_per_class)}
{_html_image_section('4. Per-Bin Calibration Gap', rel_gap)}

<h2>5. Per-Bin Breakdown (Raw — 10 bins)</h2>
<table class="dataframe">
    <tr><th>Bin</th><th>Count</th><th>Accuracy</th><th>Confidence</th><th>Gap</th><th>Best Cal.</th></tr>
    {bin_rows}
</table>

<h2>6. Per-Class & Per-Method Breakdown</h2>
<table class="dataframe">
    <tr><th>Method</th><th>Class</th><th>ECE</th><th>Brier</th><th>Samples</th></tr>
    {per_class_rows}
</table>

{hybrid_html}

<h2>7. Improvement Summary</h2>
<table class="dataframe">
    <tr><th>Metric</th><th>Raw</th><th>Best Calibrated</th><th>Δ</th><th>% Improvement</th></tr>
"""
    raw_ll = raw_metrics["log_loss"]
    best_ll = min(m["log_loss"] for m in cal_metrics_list)
    raw_brier = raw_metrics["brier"]
    best_brier = min(m["brier"] for m in cal_metrics_list)
    raw_ece = raw_metrics["ece"]
    best_ece = min(m["ece"] for m in cal_metrics_list)

    for metric, raw_v, best_v in [
        ("Log-loss", raw_ll, best_ll),
        ("Brier", raw_brier, best_brier),
        ("ECE", raw_ece, best_ece),
    ]:
        delta = raw_v - best_v
        pct = f"+{delta/raw_v*100:.1f}%" if raw_v > 0 else "—"
        color = "green" if delta > 0 else "red"
        html += f"""
    <tr><td>{metric}</td><td>{raw_v:.4f}</td><td>{best_v:.4f}</td>
        <td style="color:{color}">{delta:+.4f}</td>
        <td style="color:{color}">{pct}</td></tr>"""

    # Best calibrator
    best_cal = min(cal_metrics_list, key=lambda m: m["ece"])
    html += f"""
</table>

<div class="verdict">
    <strong>✅ Best Calibrator: {best_cal['label']}</strong>
    (ECE: {best_cal['ece']:.4f} &bull; Log-loss: {best_cal['log_loss']:.4f} &bull; Brier: {best_cal['brier']:.4f})
</div>

<p class="footer">
    Generated by generate_calibration_report.py &mdash; Football Prediction Project
</p>

</body>
</html>"""

    return html


# ── Main ─────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate detailed calibration report")
    p.add_argument("--data", choices=["worldcup", "league"], default="worldcup",
                   help="Dataset to use (default: worldcup)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip generating plots (text/HTML only)")
    p.add_argument("--n-bins", type=int, default=10,
                   help="Number of probability bins (default: 10)")
    p.add_argument("--output-dir", type=str, default="reports/calibration",
                   help="Output directory for reports (default: reports/calibration)")
    p.add_argument("--days", type=int, default=14,
                   help="Day window for upcoming matches (default: 14)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    print("=" * 70)
    print("  CALIBRATION REPORT GENERATOR")
    print(f"  Data: {args.data}  |  Bins: {args.n_bins}  |  Plots: {'yes' if not args.no_plots else 'no'}")
    print("=" * 70)

    # Output directory
    report_dir = PROJECT_ROOT / args.output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Load data ──
    print("\n  Loading data...")
    if args.data == "worldcup":
        data_path = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
    else:
        data_path = PROJECT_ROOT / "data" / "raw" / "league_all.csv"

    if not data_path.exists():
        print(f"  [X] Data not found at {data_path}")
        return 1

    df = pd.read_csv(data_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df):,} matches from {data_path.name}")

    # ── 2. Prepare data ──
    df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0})
    completed = df[df["result"].notna()].copy()
    if len(completed) < 100:
        print(f"  [X] Only {len(completed)} completed matches — need at least 100")
        return 1

    print(f"  Completed matches: {len(completed):,}")

    # ── 3. Build features ──
    print("\n  Building features...")
    from config import config
    config.dixon_coles.enabled = False
    config.features.include_h2h = True
    config.elo.regress_to_mean = True
    config.player_info.enabled = True

    from src.feature_engineering import build_features
    from src.train import train_model

    X_all, y_all = build_features(completed)

    # Chronological split: 80% train, 10% val, 10% test
    n = len(X_all)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    X_train = X_all.iloc[:n_train]
    y_train = y_all.iloc[:n_train]
    X_val = X_all.iloc[n_train:n_train + n_val]
    y_val = y_all.iloc[n_train:n_train + n_val]
    X_test = X_all.iloc[n_train + n_val:]
    y_test = y_all.iloc[n_train + n_val:]

    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ── 4. Train model ──
    print("\n  Training XGBoost model...")
    model, _ = train_model(X_train, y_train)

    # Get raw probabilities on test set
    raw_probs_test = model.predict_proba(X_test)
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    class_names = ["Away Win", "Draw", "Home Win"]

    # ── 5. Compute raw metrics ──
    print("\n  Computing raw (uncalibrated) metrics...")
    raw_metrics = _compute_all_metrics(y_test_arr, raw_probs_test, class_names, "Raw")
    print(f"    Log-loss: {raw_metrics['log_loss']:.4f}  |  "
          f"ECE: {raw_metrics['ece']:.4f}  |  "
          f"Brier: {raw_metrics['brier']:.4f}")

    # ── 6. Fit all calibration methods ──
    from src.calibration import (
        CalibratedModel,
        HybridTailCalibrator,
        _fit_calibrators,
    )

    cal_probs_list: list[dict] = []
    methods = [
        ("Platt", "platt"),
        ("Isotonic", "isotonic"),
        ("HybridTail", "hybrid"),
    ]

    val_probs = model.predict_proba(X_val)
    y_val_arr = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)

    for label, method_name in methods:
        print(f"\n  Fitting {label} calibration...")
        cal = CalibratedModel(base_model=model, method=method_name, n_classes=3)
        cal._calibrators = _fit_calibrators(val_probs, y_val_arr, 3, method_name)
        cal._fitted = True
        cal_probs = cal.predict_proba(X_test)

        m = _compute_all_metrics(y_test_arr, cal_probs, class_names, label)
        cal_probs_list.append(m)
        print(f"    Log-loss: {m['log_loss']:.4f}  |  "
              f"ECE: {m['ece']:.4f}  |  "
              f"Brier: {m['brier']:.4f}")

    # ── 7. HybridTail detailed evaluation ──
    # Fit on VAL data but evaluate on TEST data to avoid biased optimism
    print("\n  Computing HybridTail tail analysis...")
    hybrid_cal = HybridTailCalibrator(n_classes=3)
    hybrid_cal.fit(val_probs, y_val_arr)
    hybrid_eval = hybrid_cal.evaluate_calibration(raw_probs_test, y_test_arr)
    print(f"    Low-tail ECE:  {hybrid_eval.get('low_tail_ece', '—')}  "
          f"(n={hybrid_eval.get('low_tail_samples', 0)})")
    print(f"    Mid ECE:       {hybrid_eval.get('mid_ece', '—')}  "
          f"(n={hybrid_eval.get('mid_samples', 0)})")
    print(f"    High-tail ECE: {hybrid_eval.get('high_tail_ece', '—')}  "
          f"(n={hybrid_eval.get('high_tail_samples', 0)})")

    # ── 8. Generate plots ──
    plot_paths: dict[str, Path] = {}
    if not args.no_plots:
        print("\n  Generating plots...")
        all_metrics = [raw_metrics] + cal_probs_list

        rel_path = report_dir / f"reliability_diagram_{timestamp}.png"
        _plot_reliability_diagram(all_metrics, rel_path, class_names)
        plot_paths["reliability"] = rel_path
        print(f"    Reliability diagram: {rel_path}")

        pc_path = report_dir / f"per_class_reliability_{timestamp}.png"
        _plot_per_class_reliability(all_metrics, pc_path, class_names)
        plot_paths["per_class"] = pc_path
        print(f"    Per-class reliability: {pc_path}")

        gap_path = report_dir / f"gap_chart_{timestamp}.png"
        _plot_gap_bar_chart(all_metrics, gap_path)
        plot_paths["gap_chart"] = gap_path
        print(f"    Gap chart: {gap_path}")

    # ── 9. Generate HTML report ──
    print("\n  Generating HTML report...")
    html = _generate_html_report(
        raw_metrics, cal_probs_list, class_names,
        plot_paths, hybrid_eval,
    )
    html_path = report_dir / f"calibration_report_{timestamp}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"    HTML report: {html_path}")

    # ── 10. Save JSON report ──
    json_report = {
        "generated_at": datetime.now().isoformat(),
        "dataset": args.data,
        "n_samples": raw_metrics["n_total"],
        "n_bins": args.n_bins,
        "raw": {k: v for k, v in raw_metrics.items()
                if k not in ("bin_centers", "bin_counts", "bin_accuracy",
                             "bin_confidence", "bin_gap", "per_class")},
        "calibrators": [
            {k: v for k, v in m.items()
             if k not in ("bin_centers", "bin_counts", "bin_accuracy",
                          "bin_confidence", "bin_gap", "per_class")}
            for m in cal_probs_list
        ],
        "hybrid_tail_evaluation": hybrid_eval,
        "best_calibrator": min(cal_probs_list, key=lambda m: m["ece"])["label"],
    }
    json_path = report_dir / f"calibration_report_{timestamp}.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    print(f"    JSON report: {json_path}")

    # ── 11. Print text summary ──
    print("\n" + "=" * 70)
    print("  CALIBRATION REPORT SUMMARY")
    print("=" * 70)
    print(f"\n  Dataset:          {args.data} ({raw_metrics['n_total']:,} test samples)")
    print(f"  Raw ECE:          {raw_metrics['ece']:.4f}")
    print(f"  Raw Log-loss:     {raw_metrics['log_loss']:.4f}")
    print(f"  Raw Brier:        {raw_metrics['brier']:.4f}")
    print()

    best_ece = min(cal_probs_list, key=lambda m: m["ece"])
    best_ll = min(cal_probs_list, key=lambda m: m["log_loss"])
    best_brier = min(cal_probs_list, key=lambda m: m["brier"])

    for m in cal_probs_list:
        ece_arrow = " <- BEST" if m["ece"] == best_ece["ece"] else ""
        ll_arrow = " <- BEST" if m["log_loss"] == best_ll["log_loss"] else ""
        br_arrow = " <- BEST" if m["brier"] == best_brier["brier"] else ""
        print(f"  {m['label']:<14}  ECE: {m['ece']:.4f}{ece_arrow:>10}  "
              f"LL: {m['log_loss']:.4f}{ll_arrow:>9}  "
              f"Brier: {m['brier']:.4f}{br_arrow:>8}")

    print()
    print(f"  Best overall: {best_ece['label']} (ECE: {best_ece['ece']:.4f})")
    print(f"  HybridTail low-tail ECE:  {hybrid_eval.get('low_tail_ece', '---')}")
    print(f"  HybridTail mid ECE:       {hybrid_eval.get('mid_ece', '---')}")
    print(f"  HybridTail high-tail ECE: {hybrid_eval.get('high_tail_ece', '---')}")
    print(f"\n  Reports saved to: {report_dir}/")
    print(f"  Time elapsed: {time.time() - t0:.1f}s")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
