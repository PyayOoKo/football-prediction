"""
Generate a comprehensive backtest report as a Markdown document.

Loads the comparison JSON (all models metrics + best performers),
the summary JSON (rankings), and links to all figure PNGs,
assembling everything into reports/backtest_report_{timestamp}.md.
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

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("generate_backtest_report")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


# ══════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════


def find_latest(pattern: str) -> Path | None:
    """Find the most recent file matching a glob pattern in reports/."""
    files = sorted(REPORTS_DIR.glob(pattern))
    return files[-1] if files else None


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_pct(v: float) -> str:
    """Format a percentage value (e.g. -99.47% or 0.00%)."""
    return f"{v:+.2f}%" if abs(v) >= 0.005 else "0.00%"


def fmt_gbp(v: float) -> str:
    """Format as GBP currency."""
    return f"GBP{v:+,.2f}"


def fmt_round(v: float, dp: int = 4) -> str:
    return f"{v:.{dp}f}"


# ══════════════════════════════════════════════════════
#  Markdown builder
# ══════════════════════════════════════════════════════


def build_report(
    comparison: dict[str, Any],
    summary: dict[str, Any] | None,
    timestamp: str,
) -> str:
    """Assemble the full Markdown report."""
    lines: list[str] = []
    table = comparison.get("comparison_table", [])
    best = comparison.get("best_performers", {})
    config = comparison.get("configuration", "")
    models = summary.get("models", []) if summary else []
    # Build a lookup by model name
    model_ranks: dict[str, int] = {
        m["model_name"]: m.get("rank", 0) for m in models
    }

    # ── Header ────────────────────────────────────────
    lines.append(f"# Backtest Report — All Models")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")
    lines.append(f"**Configuration:** {config}")
    lines.append("")
    lines.append(
        f"**Models evaluated:** {comparison.get('total_models', len(table))}"
    )
    lines.append("")

    # ── 1. Executive Summary ─────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")

    # Key findings
    top_sharpe = best.get("sharpe_ratio", {})
    top_roi = best.get("roi_pct", {})
    top_yield_ = best.get("yield_pct", {})
    safest = best.get("max_drawdown_pct", {})
    top_clv = best.get("avg_clv", {})
    top_pf = best.get("profit_factor", {})
    top_wr = best.get("win_rate_pct", {})
    top_profit = best.get("overall_profit", {})

    lines.append("### Key Findings")
    lines.append("")
    lines.append(
        f"- **Best Sharpe ratio:** {top_sharpe.get('model', '?')} "
        f"({fmt_round(top_sharpe.get('value', 0), 2)})"
    )
    lines.append(
        f"- **Best ROI:** {top_roi.get('model', '?')} "
        f"({fmt_pct(top_roi.get('value', 0))})"
    )
    lines.append(
        f"- **Best Yield:** {top_yield_.get('model', '?')} "
        f"({fmt_pct(top_yield_.get('value', 0))})"
    )
    lines.append(
        f"- **Most consistent (lowest drawdown):** {safest.get('model', '?')} "
        f"({fmt_pct(safest.get('value', 0))})"
    )
    lines.append(
        f"- **Best profit factor:** {top_pf.get('model', '?')} "
        f"({fmt_round(top_pf.get('value', 0), 4)})"
    )
    lines.append(
        f"- **Highest win rate:** {top_wr.get('model', '?')} "
        f"({fmt_pct(top_wr.get('value', 0))})"
    )
    lines.append(
        f"- **Best CLV:** {top_clv.get('model', '?')} "
        f"({fmt_round(top_clv.get('value', 0), 6)})"
    )
    lines.append(
        f"- **Least total loss:** {top_profit.get('model', '?')} "
        f"({fmt_gbp(top_profit.get('value', 0))})"
    )
    lines.append("")

    # Overall assessment
    all_roi = [m.get("roi_pct", 0) for m in table]
    all_sharpe = [m.get("sharpe_ratio", 0) for m in table]
    all_dd = [m.get("max_drawdown_pct", 0) for m in table]
    avg_roi = sum(all_roi) / len(all_roi) if all_roi else 0
    avg_sharpe = sum(all_sharpe) / len(all_sharpe) if all_sharpe else 0
    avg_dd = sum(all_dd) / len(all_dd) if all_dd else 0

    lines.append("### Portfolio-Level Metrics")
    lines.append("")
    lines.append(f"- **Average ROI across all models:** {fmt_pct(avg_roi)}")
    lines.append(
        f"- **Average Sharpe ratio:** {fmt_round(avg_sharpe, 2)}"
    )
    lines.append(
        f"- **Average max drawdown:** {fmt_pct(avg_dd)}"
    )
    lines.append(
        f"- **Best-to-worst ROI spread:** "
        f"{fmt_pct(max(all_roi))} to {fmt_pct(min(all_roi))}"
    )
    lines.append("")

    # ── 2. Performance Table ─────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 2. Performance Comparison — All Models")
    lines.append("")

    # Table header
    lines.append(
        "| Rank | Model | Phase | Calibration | Brier | Bets | "
        "W/L | P&L | ROI% | Yield% | Win% | Sharpe | DD% | PF | CLV |"
    )
    lines.append(
        "|------|-------|-------|-------------|-------|------|"
        "-----|------|------|--------|------|--------|------|------|------|"
    )

    # Sort by rank if available, else by Sharpe descending
    sorted_models = sorted(
        table,
        key=lambda m: (
            -model_ranks.get(m.get("model_name", ""), 999),
        ),
    )

    for m in sorted_models:
        rank = model_ranks.get(m.get("model_name", ""), "-")
        name = m.get("model_name", "?")
        phase = m.get("phase", "")
        cal = m.get("calibration", "")
        brier = m.get("brier", 0)
        bets = m.get("total_bets", 0)
        wins = m.get("winning_bets", 0)
        losses = m.get("losing_bets", 0)
        pnl = m.get("total_profit", 0)
        roi = m.get("roi_pct", 0)
        yld = m.get("yield_pct", 0)
        wr = m.get("win_rate_pct", 0)
        sharpe = m.get("sharpe_ratio", 0)
        dd = m.get("max_drawdown_pct", 0)
        pf = m.get("profit_factor", 0)
        clv = m.get("avg_clv", 0)

        lines.append(
            f"| {rank} | {name} | {phase} | {cal} | "
            f"{brier:.4f} | {bets} | {wins}/{losses} | "
            f"{fmt_gbp(pnl)} | {fmt_pct(roi)} | {fmt_pct(yld)} | "
            f"{wr:.1f}% | {fmt_round(sharpe, 2)} | "
            f"{fmt_pct(dd)} | {fmt_round(pf, 2)} | "
            f"{clv:+.6f} |"
        )

    lines.append("")

    # ── 3. Best Performers ───────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 3. Best Performing Models by Metric")
    lines.append("")
    lines.append(
        "| Category | Model | Value | Description |"
    )
    lines.append(
        "|----------|-------|-------|-------------|"
    )

    categories = [
        ("Sharpe Ratio", "sharpe_ratio", "Risk-adjusted return (higher=better)", fmt_round),
        ("ROI %", "roi_pct", "Return on investment over whole period", fmt_pct),
        ("Yield %", "yield_pct", "Profit per unit staked", fmt_pct),
        ("Win Rate %", "win_rate_pct", "Percentage of bets won", fmt_pct),
        ("Profit Factor", "profit_factor", "Gross profit / gross loss", lambda v: fmt_round(v, 2)),
        ("Least Losses", "overall_profit", "Least negative total P&L", fmt_gbp),
        ("Lowest Drawdown", "max_drawdown_pct", "Most capital preservation", fmt_pct),
        ("Best CLV", "avg_clv", "Highest closing line value", lambda v: fmt_round(v, 6)),
        ("Sortino Ratio", "sortino_ratio", "Downside risk-adjusted return", lambda v: fmt_round(v, 2)),
    ]

    for label, key, desc, formatter in categories:
        entry = best.get(key)
        if entry:
            model = entry.get("model", "?")
            value = entry.get("value", 0)
            lines.append(
                f"| {label} | {model} | {formatter(value)} | {desc} |"
            )

    lines.append("")

    # ── 4. Risk Analysis ─────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 4. Risk Analysis")
    lines.append("")

    # Drawdown discussion
    lines.append("### 4.1 Maximum Drawdown")
    lines.append("")
    lines.append(
        "Maximum drawdown measures the largest peak-to-trough decline "
        "in the bankroll — a key indicator of downside risk. "
        "Lower values indicate better capital preservation."
    )
    lines.append("")

    # Risk tier summary
    low_dd = [m for m in table if m.get("max_drawdown_pct", 100) < 50]
    mod_dd = [
        m
        for m in table
        if 50 <= m.get("max_drawdown_pct", 100) < 90
    ]
    high_dd = [m for m in table if m.get("max_drawdown_pct", 0) >= 90]

    lines.append("| Risk Tier | Threshold | Models |")
    lines.append("|-----------|-----------|--------|")
    lines.append(
        f"| Low | < 50% | {len(low_dd)} |"
    )
    lines.append(
        f"| Moderate | 50% – 90% | {len(mod_dd)} |"
    )
    lines.append(
        f"| High | > 90% | {len(high_dd)} |"
    )
    lines.append("")

    if high_dd:
        worst_dd = max(high_dd, key=lambda m: m.get("max_drawdown_pct", 0))
        best_dd = min(high_dd, key=lambda m: m.get("max_drawdown_pct", 0))
        lines.append(
            f"**Note:** All {len(high_dd)} models fall into the high-risk "
            f"tier (drawdown > 90%). The best in this group is "
            f"**{best_dd.get('model_name', '?')}** at "
            f"{fmt_pct(best_dd.get('max_drawdown_pct', 0))}, "
            f"while the worst is **{worst_dd.get('model_name', '?')}** "
            f"at {fmt_pct(worst_dd.get('max_drawdown_pct', 0))}."
        )
        lines.append("")

    # Sharpe discussion
    lines.append("### 4.2 Sharpe Ratio")
    lines.append("")
    lines.append(
        "The Sharpe ratio measures risk-adjusted return: how much "
        "return is earned per unit of risk (volatility). "
        "Conventional interpretation:"
    )
    lines.append("")
    lines.append("| Range | Interpretation |")
    lines.append("|-------|----------------|")
    lines.append("| > 2.0 | Excellent |")
    lines.append("| 1.0 – 2.0 | Good |")
    lines.append("| 0.5 – 1.0 | Adequate |")
    lines.append("| 0.0 – 0.5 | Poor |")
    lines.append("| < 0.0 | Negative — losing money |")
    lines.append("")

    n_positive_sharpe = sum(1 for m in table if m.get("sharpe_ratio", 0) > 0)
    n_negative_sharpe = len(table) - n_positive_sharpe
    lines.append(
        f"- **{n_positive_sharpe} model(s)** have positive Sharpe ratio "
        f"(better than risk-free)"
    )
    lines.append(
        f"- **{n_negative_sharpe} model(s)** have negative Sharpe ratio "
        f"(underperforming risk-free)"
    )
    lines.append("")

    best_sharpe_entry = best.get("sharpe_ratio", {})
    lines.append(
        f"**Best:** {best_sharpe_entry.get('model', '?')} "
        f"(Sharpe={fmt_round(best_sharpe_entry.get('value', 0), 2)})"
    )
    lines.append("")

    # ── 5. CLV Analysis ──────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 5. CLV Analysis")
    lines.append("")
    lines.append(
        "Closing Line Value (CLV) measures the difference between "
        "the odds at bet placement and the closing odds. "
        "Positive CLV is a strong indicator of predictive skill: "
        "it means your model's estimate was better than the market's "
        "final consensus."
    )
    lines.append("")

    clv_models = [m for m in table if abs(m.get("avg_clv", 0)) > 0]
    if not clv_models:
        lines.append(
            "No CLV data available — all models had average CLV of 0.000000, "
            "which is expected when using synthetic odds derived from the "
            "same underlying true results. Real market data would show "
            "nonzero closing line values."
        )
        lines.append("")
    else:
        lines.append("| Model | Avg CLV | Positive CLV % |")
        lines.append("|-------|---------|-----------------|")
        for m in clv_models:
            name = m.get("model_name", "?")
            clv = m.get("avg_clv", 0)
            pos_clv_pct = m.get("positive_clv_pct", 0)
            lines.append(f"| {name} | {clv:+.6f} | {pos_clv_pct:.1f}% |")
        lines.append("")

    total_bets_all = sum(m.get("total_bets", 0) for m in table)
    total_clv_pos = sum(
        m.get("positive_clv_pct", 0) * m.get("total_bets", 0) / 100
        for m in table
    )
    if total_bets_all > 0:
        avg_pos_clv = total_clv_pos / total_bets_all * 100
        lines.append(
            f"Across all models, the average positive CLV rate is "
            f"**{avg_pos_clv:.1f}%** — this measures how often "
            f"bettors improved their position vs closing odds."
        )
        lines.append("")

    # ── 6. Visualizations ────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 6. Visualizations")
    lines.append("")

    # Combined charts
    combined_charts = [
        ("Bankroll Comparison", "bankroll_comparison.png"),
        ("ROI & Yield Comparison", "roi_comparison.png"),
        ("Drawdown Comparison", "drawdown_comparison.png"),
    ]

    lines.append("### 6.1 Combined Charts")
    lines.append("")
    for title, filename in combined_charts:
        img_rel = f"figures/{filename}"
        lines.append(f"**{title}**")
        lines.append("")
        lines.append(f"![{title}]({img_rel})")
        lines.append("")

    # Per-model bankroll charts
    lines.append("### 6.2 Bankroll Growth (Per Model)")
    lines.append("")
    for m in table:
        name = m.get("model_name", "?")
        img = f"figures/bankroll_{name}.png"
        lines.append(f"<details>")
        lines.append(f"<summary><b>{name}</b> — Bankroll Growth</summary>")
        lines.append("")
        lines.append(f"![Bankroll_{name}]({img})")
        lines.append("")
        lines.append(f"</details>")
        lines.append("")

    # Per-model P&L per bet charts
    lines.append("### 6.3 P&L Per Bet (Per Model)")
    lines.append("")
    for m in table:
        name = m.get("model_name", "?")
        img = f"figures/pl_per_bet_{name}.png"
        lines.append(f"<details>")
        lines.append(f"<summary><b>{name}</b> — P&L Per Bet</summary>")
        lines.append("")
        lines.append(f"![PL_{name}]({img})")
        lines.append("")
        lines.append(f"</details>")
        lines.append("")

    # Per-model drawdown charts
    lines.append("### 6.4 Drawdown (Per Model)")
    lines.append("")
    for m in table:
        name = m.get("model_name", "?")
        img = f"figures/drawdown_{name}.png"
        lines.append(f"<details>")
        lines.append(f"<summary><b>{name}</b> — Drawdown</summary>")
        lines.append("")
        lines.append(f"![Drawdown_{name}]({img})")
        lines.append("")
        lines.append(f"</details>")
        lines.append("")

    # ── 7. Data Dictionary ───────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 7. Data Dictionary")
    lines.append("")
    lines.append(
        "| Metric | Description |"
    )
    lines.append(
        "|--------|-------------|"
    )
    lines.append(
        "| **Brier Score** | Mean squared error of probability predictions (0=perfect, 1=worst) |"
    )
    lines.append(
        "| **Total Bets** | Number of bets placed after filtering |"
    )
    lines.append(
        "| **W/L** | Winning vs losing bets |"
    )
    lines.append(
        "| **P&L** | Total profit/loss in GBP |"
    )
    lines.append(
        "| **ROI %** | Return on Investment = P&L / Initial Bankroll × 100 |"
    )
    lines.append(
        "| **Yield %** | P&L / Total Staked × 100 (profit per unit risked) |"
    )
    lines.append(
        "| **Win Rate %** | Winning Bets / Total Bets × 100 |"
    )
    lines.append(
        "| **Sharpe Ratio** | (Mean Return / Std Dev) × √500 (annualised) |"
    )
    lines.append(
        "| **Max Drawdown %** | Largest peak-to-trough decline in bankroll |"
    )
    lines.append(
        "| **Profit Factor** | Gross Profit / Gross Loss (≥ 1.0 is profitable) |"
    )
    lines.append(
        "| **Avg CLV** | Average closing line value (positive = skill) |"
    )
    lines.append("")

    # ── 8. Recommendations ───────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 8. Recommendations for Production Use")
    lines.append("")

    # Assess overall viability
    n_positive = sum(1 for m in table if m.get("total_profit", 0) >= 0)

    lines.append("### 8.1 Model Selection")
    lines.append("")
    if n_positive == 0:
        lines.append(
            "⚠️ **All models produced negative returns in this backtest.** "
            "This is expected when using synthetic predictions derived from "
            "Brier scores rather than real trained models. "
            "The RELATIVE ranking of models is still meaningful:"
        )
    else:
        lines.append(
            f"{n_profitable} model(s) generated positive returns. ..."
        )
    lines.append("")

    # Rank models by overall quality (composite score)
    lines.append("| Rank | Model | Composite Score | Strengths | Weaknesses |")
    lines.append("|------|-------|----------------|-----------|------------|")
    for i, m in enumerate(sorted_models[:5], 1):
        name = m.get("model_name", "?")
        sharpe_v = m.get("sharpe_ratio", 0)
        roi_v = m.get("roi_pct", 0)
        dd_v = m.get("max_drawdown_pct", 0)
        pf_v = m.get("profit_factor", 0)

        # Simple composite: normalised Sharpe + normalised PF (higher=better)
        composite = max(sharpe_v, -10) / 10 + min(pf_v, 2) / 2

        strengths = []
        weaknesses = []
        if sharpe_v > 0:
            strengths.append("Positive risk-adjusted return")
        else:
            weaknesses.append("Negative Sharpe")
        if pf_v > 0.5:
            strengths.append("Good profit factor")
        elif pf_v < 0.1:
            weaknesses.append("Very low profit factor")
        if dd_v < 99:
            strengths.append("Better-than-average drawdown")
        else:
            weaknesses.append("Extreme drawdown > 99%")
        if m.get("win_rate_pct", 0) > 12:
            strengths.append("Win rate > 12%")
        else:
            weaknesses.append("Low win rate")

        lines.append(
            f"| {i} | {name} | {composite:.2f} | "
            f"{'; '.join(strengths[:2]) or '—'} | "
            f"{'; '.join(weaknesses[:2]) or '—'} |"
        )
    lines.append("")

    lines.append("### 8.2 Strategy Recommendations")
    lines.append("")

    best_model_name = best.get("sharpe_ratio", {}).get("model", "?")
    lines.append(
        f"1. **Primary model:** **{best_model_name}** ranked best by "
        f"Sharpe ratio. Use as the primary prediction engine."
    )
    lines.append("")
    lines.append(
        "2. **Ensemble approach:** Combine top 2–3 models via weighted "
        "averaging to reduce variance and improve robustness."
    )
    lines.append("")
    lines.append(
        "3. **Stake sizing:** Continue using Fractional Kelly (25–50%) "
        "to manage risk. Avoid full Kelly to reduce variance."
    )
    lines.append("")
    lines.append(
        "4. **Filter tuning:** Tighten filters as the bankroll grows. "
        "Consider dynamic thresholds based on recent performance."
    )
    lines.append("")

    lines.append("### 8.3 Improvements for Next Iteration")
    lines.append("")
    lines.append(
        "1. **Feature engineering:** Add more granular features "
        "(player-level data, weather, referee tendencies)."
    )
    lines.append(
        "2. **Calibration:** Ensure all models are calibrated "
        "(Platt scaling or Temperature scaling) before ensembling."
    )
    lines.append(
        "3. **Multi-market:** Expand to BTTS and Over/Under markets "
        "to diversify bet types and reduce correlation."
    )
    lines.append(
        "4. **Walk-forward validation:** Test on rolling windows "
        "rather than a single fixed test set for more robust results."
    )
    lines.append(
        "5. **Real-time monitoring:** Track SLI metrics (bankroll, "
        "drawdown, Sharpe) in production and set stop-loss thresholds."
    )
    lines.append("")

    # ── Footer ────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Report auto-generated at {timestamp} "
        f"by `generate_backtest_report.py`*"
    )
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════


def main() -> None:
    # Find latest files
    comp_path = find_latest("backtest_comparison_*.json")
    summary_path = find_latest("backtest_summary_*.json")

    if not comp_path:
        logger.error("No comparison JSON found — run compare_backtest_results.py first")
        sys.exit(1)

    logger.info("Loading comparison: %s", comp_path)
    comparison = load_json(comp_path)

    summary = None
    if summary_path:
        logger.info("Loading summary: %s", summary_path)
        summary = load_json(summary_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_md = build_report(comparison, summary, timestamp)

    out_path = REPORTS_DIR / f"backtest_report_{timestamp}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Report saved to %s (%d bytes)", out_path, len(report_md))

    # Print summary
    print(f"\n{'=' * 60}")
    print("  BACKTEST REPORT GENERATED")
    print(f"{'=' * 60}")
    print(f"  File: {out_path.name}")
    print(f"  Size: {len(report_md):,} bytes")
    print(f"  Models: {comparison.get('total_models', '?')}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
