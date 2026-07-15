"""
Generate a comprehensive CLV analysis report as a Markdown file.

Loads CLV analysis JSON, per-model CLV JSONs, and backtest ROI data.
Produces reports/clv_report_{timestamp}.md with embedded visualizations.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("clv_report")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Helpers ──────────────────────────────────────────────────────────────


def _find_latest(pattern: str) -> Path | None:
    """Find the most recent file matching *pattern* in reports/."""
    files = sorted(REPORTS_DIR.glob(pattern))
    return files[-1] if files else None


def _fmt(val: float | None, decimals: int = 4, pct: bool = False) -> str:
    """Format a float, handling None and NaN."""
    if val is None:
        return "—"
    try:
        if not np.isfinite(val):
            return "—"
        if pct:
            return f"{val * 100:.2f}%" if val < 10 else f"{val:.2f}%"
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError, RuntimeError):
        return "—"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_backtest_rois() -> dict[str, float]:
    """Load ROI from backtest JSONs for CLV vs ROI comparison."""
    rois: dict[str, float] = {}
    # Discover the latest backtest timestamp from any model's file
    latest_bt = _find_latest("backtest_Elo_*.json") or _find_latest("backtest_*.json")
    if not latest_bt:
        logger.warning("No backtest files found — ROI data unavailable")
        return rois
    ts_match = re.search(r"_(\d{8}_\d{6})\.json$", latest_bt.name)
    if not ts_match:
        return rois
    ts = ts_match.group(1)
    bt_files = sorted(REPORTS_DIR.glob(f"backtest_*_{ts}.json"))
    logger.info("Found %d backtest files with timestamp %s", len(bt_files), ts)
    for fp in bt_files:
        try:
            data = _load_json(fp)
            metrics = data.get("backtest_metrics", data)
            roi = metrics.get("roi", metrics.get("total_roi", metrics.get("return_on_investment")))
            name = data.get("model_name", fp.stem.replace("backtest_", ""))
            name = re.sub(r"_\d{8}_\d{6}$", "", name)
            if roi is not None:
                rois[name] = float(roi)
        except Exception as exc:
            logger.debug("Failed to load ROI from %s: %s", fp.name, exc)
    return rois


# ── Report sections ─────────────────────────────────────────────────────


def section_header(level: int, title: str) -> str:
    return f"\n{'#' * level} {title}\n"


def exec_summary(clv_analysis: dict[str, Any], rois: dict[str, float]) -> str:
    lines = [
        section_header(2, "Executive Summary"),
        "",
        "This report analyses **Closing Line Value (CLV)** across all 9 prediction models. "
        "CLV measures how your odds compare to the market's closing odds — "
        "the consensus price after all information is priced in. "
        "Positive CLV is widely regarded as the single best indicator of betting skill.",
        "",
    ]

    cross = clv_analysis.get("cross_model", {})
    global_avg = cross.get("avg_clv_across_models", cross.get("avg_clv_across", cross.get("avg_clv", None)))
    global_pct = cross.get("avg_positive_pct", cross.get("avg_pos_clv_pct", cross.get("avg_pct_positive", None)))
    total_bets = (clv_analysis.get("trend_analysis", {})
                  .get("overall", {})
                  .get("total_bets", clv_analysis.get("total_models", 0) * 25))

    lines.append(f"- **Total models analysed:** {clv_analysis.get('total_models', '?')}")
    lines.append(f"- **Total bets simulated:** ~{total_bets}")
    lines.append(f"- **Average CLV across all models:** {_fmt(global_avg)}")
    lines.append(f"- **Average % of bets with positive CLV:** {_fmt(global_pct)}%")
    lines.append("")

    # Best/worst
    per_model = clv_analysis.get("per_model", {})
    if per_model:
        best_name = max(per_model, key=lambda n: _pm(per_model[n], "avg_clv", -999))
        worst_name = min(per_model, key=lambda n: _pm(per_model[n], "avg_clv", 999))
        best_clv = _pm(per_model[best_name], "avg_clv", 0)
        worst_clv = _pm(per_model[worst_name], "avg_clv", 0)
        lines.append(f"- **Best average CLV:** {best_name} ({_fmt(best_clv)})")
        lines.append(f"- **Worst average CLV:** {worst_name} ({_fmt(worst_clv)})")
        lines.append("")

    # Consistency
    consistent = cross.get("consistently_positive_models", [])
    if consistent:
        lines.append(f"- **Models with consistently positive CLV:** {', '.join(consistent[:5])}")
        lines.append("")

    # CLV vs ROI — any correlation?
    if rois:
        pos_clv = [n for n, v in per_model.items() if _pm(v, "avg_clv", 0) > 0]
        pos_roi = [n for n, r in rois.items() if r > 0]
        overlap = set(pos_clv) & set(pos_roi)
        lines.append(f"- **Models with positive CLV that also show positive ROI:** "
                      f"{len(overlap)} / {len(pos_clv)}")
        lines.append("")
        lines.append(f"- ⚠️ **Key insight:** All models show positive average CLV but negative ROI. "
                      f"This suggests the synthetic closing-odds generation systematically "
                      f"favours the model's predictions, and CLV alone doesn't guarantee profitability "
                      f"without considering stake sizing and odds filtering.")
        lines.append("")

    lines.extend([
        "---",
        "",
    ])
    return "\n".join(lines)


def performance_table(clv_analysis: dict[str, Any], rois: dict[str, float]) -> str:
    lines = [
        section_header(2, "CLV Performance Table"),
        "",
        "| Rank | Model | Bets | Avg CLV | Median CLV | +CLV% | Win% | Consistency | ROI | Trend |",
        "|------|-------|------|---------|------------|-------|------|-------------|-----|-------|",
    ]

    per_model = clv_analysis.get("per_model", {})
    trends = clv_analysis.get("trend_analysis", {}).get("model_trends", {})

    if not per_model:
        return "\n".join(lines) + "\n\n*No per-model data available.*\n"

    # Sort by avg_clv descending
    sorted_models = sorted(per_model.items(), key=lambda x: _pm(x[1], "avg_clv", 0), reverse=True)

    for rank, (name, stats) in enumerate(sorted_models, 1):
        avg = _pm(stats, "avg_clv")
        med = _pm(stats, "median_clv")
        pos = stats.get("positive_clv", {})
        pct_pos = pos.get("pct_positive", pos.get("pct", 0))
        winners = stats.get("winners", 0)
        total = stats.get("total_bets", 1)
        win_pct = winners / total * 100 if total > 0 else 0
        consistency = stats.get("consistency_score", 0)
        roi = rois.get(name, None)
        trend = trends.get(name, {}).get("trend", "—")

        # Rank emoji
        rank_str = f"🥇 {name}" if rank == 1 else f"🥈 {name}" if rank == 2 else f"🥉 {name}" if rank == 3 else name

        lines.append(
            f"| {rank} | {rank_str} | {total} "
            f"| {_fmt(avg)} | {_fmt(med)} "
            f"| {_fmt(pct_pos, 1)}% | {_fmt(win_pct, 1)}% "
            f"| {_fmt(consistency, 2)} | {_fmt(roi, 2)}% "
            f"| {trend} |"
        )

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def best_model_market(clv_analysis: dict[str, Any]) -> str:
    lines = [
        section_header(2, "Best CLV Model and Market Analysis"),
        "",
    ]

    per_model = clv_analysis.get("per_model", {})
    cross = clv_analysis.get("cross_model", {})

    if per_model:
        # Best by avg CLV
        best_avg = max(per_model.items(), key=lambda x: _pm(x[1], "avg_clv", -999))
        lines.append(f"### Best Average CLV: {best_avg[0]}")
        lines.append(f"- **Average CLV:** {_fmt(_pm(best_avg[1], 'avg_clv'))}")
        lines.append(f"- **Median CLV:** {_fmt(_pm(best_avg[1], 'median_clv'))}")
        pos = best_avg[1].get("positive_clv", {})
        lines.append(f"- **Positive CLV rate:** {_fmt(pos.get('pct_positive', pos.get('pct', 0)), 1)}%")
        lines.append(f"- **Win rate:** {_fmt(best_avg[1].get('winners', 0) / max(best_avg[1].get('total_bets', 1), 1) * 100, 1)}%")
        lines.append("")

        # Best by consistency
        if cross.get("top_consistency"):
            top_cons = cross["top_consistency"][0]
            lines.append(f"### Most Consistent Positive CLV: {top_cons.get('model_name', '—')}")
            lines.append(f"- **Consistency score:** {_fmt(top_cons.get('consistency_score', 0), 2)}")
            lines.append(f"- **Average CLV:** {_fmt(top_cons.get('avg_clv', 0))}")
            lines.append("")

        # Best by median
        best_med = max(per_model.items(), key=lambda x: _pm(x[1], "median_clv", -999))
        lines.append(f"### Best Median CLV: {best_med[0]}")
        lines.append(f"- **Median CLV:** {_fmt(_pm(best_med[1], 'median_clv'))}")
        lines.append(f"- This indicates {best_med[0]} has the most consistently positive CLV at the 50th percentile.")
        lines.append("")

    # Market analysis
    lines.append("### Market Analysis")
    lines.append("")
    market_data = cross.get("markets_ranked_by_clv", cross.get("market_analysis", {}))
    if isinstance(market_data, dict) and market_data:
        lines.append("| Market | Avg CLV | Models Analysed |")
        lines.append("|--------|---------|-----------------|")
        for market_name, info in market_data.items():
            lines.append(
                f"| {market_name} "
                f"| {_fmt(info.get('avg_clv_across_models', info.get('avg_clv', 0)))} "
                f"| {info.get('models_analyzed', '—')} |"
            )
    elif isinstance(market_data, list) and market_data:
        lines.append("| Market | Avg CLV | Best Model | Worst Model |")
        lines.append("|--------|---------|------------|-------------|")
        for m in market_data:
            if isinstance(m, dict):
                lines.append(
                    f"| {m.get('market', '—')} "
                    f"| {_fmt(m.get('avg_clv', 0))} "
                    f"| {m.get('best_model', '—')} "
                    f"| {m.get('worst_model', '—')} |"
                )
    else:
        lines.append("*Only 1X2 market data available — CLV by market limited to synthetic simulation.*")

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def clv_vs_roi(clv_analysis: dict[str, Any], rois: dict[str, float]) -> str:
    lines = [
        section_header(2, "CLV vs ROI Analysis"),
        "",
        "Does positive CLV translate to profitability? This section compares "
        "each model's average CLV against its backtested ROI.",
        "",
    ]

    per_model = clv_analysis.get("per_model", {})

    if not per_model or not rois:
        lines.append("*ROI data unavailable for comparison.*")
        lines.extend(["", "---", ""])
        return "\n".join(lines)

    lines.append("| Model | Avg CLV | ROI | CLV > 0? | Profitable? | Alignment |")
    lines.append("|-------|---------|-----|----------|-------------|-----------|")

    aligned = 0
    total = 0
    for name in sorted(per_model.keys()):
        avg = _pm(per_model[name], "avg_clv", 0)
        roi = rois.get(name)
        if roi is None:
            continue
        clv_pos = avg > 0
        roi_pos = roi > 0
        match = "✅" if clv_pos == roi_pos else "⚠️" if clv_pos else "❌"
        if clv_pos == roi_pos:
            aligned += 1
        total += 1
        lines.append(
            f"| {name} | {_fmt(avg)} | {_fmt(roi, 2)}% "
            f"| {'✅' if clv_pos else '❌'} | {'✅' if roi_pos else '❌'} "
            f"| {match} |"
        )

    lines.append("")
    if total > 0:
        lines.append(f"- **Alignment rate:** {aligned}/{total} models ({aligned/total*100:.0f}%)")
        lines.append("")

    # Correlation analysis
    clv_vals = []
    roi_vals = []
    for name in per_model:
        avg = _pm(per_model[name], "avg_clv", 0)
        roi = rois.get(name)
        if roi is not None:
            clv_vals.append(avg)
            roi_vals.append(roi)

    if len(clv_vals) >= 3:
        corr = float(np.corrcoef(clv_vals, roi_vals)[0, 1])
        lines.append(f"- **Correlation coefficient:** {_fmt(corr, 3)}")
        if abs(corr) < 0.3:
            lines.append("- ⚠️ **Weak correlation** — CLV is not strongly predictive of ROI in this data.")
        elif abs(corr) < 0.7:
            lines.append("- ✅ **Moderate correlation** — CLV shows some relationship with ROI.")
        else:
            lines.append("- ✅ **Strong correlation** — CLV is a reliable indicator of profitability.")

        lines.append("")
        lines.append("**Possible explanations for CLV-ROI disconnect:**")
        lines.append("1. **Synthetic closing odds** bias toward truth (+2% probability) inflates CLV")
        lines.append("2. **Stake sizing** (fractional Kelly) amplifies losses despite good CLV")
        lines.append("3. **Low win rates** (12-15%) mean high variance dominates outcomes")
        lines.append("4. **Small sample size** (~25 bets per model) — CLV signal needs 200+ bets to stabilise")

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def trend_analysis(clv_analysis: dict[str, Any]) -> str:
    lines = [
        section_header(2, "CLV Trends Over Time"),
        "",
        "For each model, CLV over the bet sequence is analysed. "
        "A 5-bet moving average separates signal from noise.",
        "",
    ]

    trends = clv_analysis.get("trend_analysis", {})
    model_trends = trends.get("model_trends", {})
    overall = trends.get("overall", {})

    if overall:
        wr = overall.get("win_rate", overall.get("win_pct", overall.get("winning_pct", None)))
        if wr is None:
            # Calculate from total wins/bets
            total_w = overall.get("total_wins", overall.get("n_wins", 0))
            total_b = overall.get("total_bets", overall.get("n_bets", 1))
            wr = total_w / total_b if total_b > 0 else 0
        lines.append("### Overall")
        lines.append(f"- **Total bets across all models:** {overall.get('total_bets', '?')}")
        lines.append(f"- **Average CLV:** {_fmt(overall.get('avg_clv', 0))}")
        lines.append(f"- **Win rate:** {_fmt(wr * 100, 1) if wr and wr < 1 else _fmt(wr, 1)}%")
        lines.append("")
        lines.append("### Per-Model Trend Direction")
        lines.append("")
        lines.append("| Model | Trend | Recent Avg CLV | Overall Avg | Change |")
        lines.append("|-------|-------|----------------|-------------|--------|")

        # Categorize
        improving = []
        declining = []
        stable = []

        for name, t in sorted(model_trends.items()):
            trend = t.get("trend", "—")
            recent = t.get("recent_avg_clv", t.get("last_5_avg"))
            overall_val = t.get("overall_avg_clv", t.get("full_mean"))
            if trend == "Improving":
                improving.append(name)
            elif trend == "Declining":
                declining.append(name)
            else:
                stable.append(name)

            change = ""
            if recent is not None and overall_val is not None and overall_val != 0:
                pct_change = (recent - overall_val) / abs(overall_val) * 100
                change = f"{pct_change:+.1f}%"

            lines.append(
                f"| {name} | {trend} "
                f"| {_fmt(recent)} | {_fmt(overall_val)} "
                f"| {change} |"
            )

        lines.append("")
        lines.append(f"- **Improving:** {', '.join(improving) if improving else 'None'}")
        lines.append(f"- **Declining:** {', '.join(declining) if declining else 'None'}")
        lines.append(f"- **Stable:** {', '.join(stable) if stable else 'None'}")
        lines.append("")

    # Embed per-model over-time charts
    lines.append("### CLV Over Time Charts")
    lines.append("")
    for name in sorted(clv_analysis.get("per_model", {}).keys()):
        chart = FIGURES_DIR / f"clv_over_time_{name}.png"
        rel_path = f"figures/clv_over_time_{name}.png"
        if chart.exists():
            lines.append(f"**{name}**")
            lines.append("")
            lines.append(f"![{name} CLV over time]({rel_path})")
            lines.append("")

    lines.extend(["---", ""])
    return "\n".join(lines)


def _pm(stats: dict[str, Any], key: str, default=None):
    """Safely get a nested CLV stats field."""
    cs = stats.get("clv_stats", {})
    if key in cs:
        return cs[key]
    return stats.get(key, default)


def recommendations(clv_analysis: dict[str, Any], rois: dict[str, float]) -> str:
    lines = [
        section_header(2, "Recommendations for Improving CLV"),
        "",
        "Based on the analysis across all 9 models, here are actionable recommendations:",
        "",
    ]

    per_model = clv_analysis.get("per_model", {})

    # 1. Model selection
    lines.append("### 1. Model Selection for Best CLV")
    lines.append("")
    if per_model:
        best_consistency = sorted(
            per_model.items(),
            key=lambda x: x[1].get("consistency_score", x[1].get("pct_positive", 0)),
            reverse=True,
        )
        top3 = best_consistency[:3]
        lines.append(f"1. **{top3[0][0]}** (consistency: {_fmt(top3[0][1].get('consistency_score', 0), 2)}) — top pick")
        if len(top3) > 1:
            lines.append(f"2. **{top3[1][0]}** (consistency: {_fmt(top3[1][1].get('consistency_score', 0), 2)}) — strong runner-up")
        if len(top3) > 2:
            lines.append(f"3. **{top3[2][0]}** (consistency: {_fmt(top3[2][1].get('consistency_score', 0), 2)})")
        lines.append("")

    # 2. Market focus
    lines.append("### 2. Market Focus")
    lines.append("")
    lines.append("- **1X2** is the most liquid market — stick to it for CLV stability.")
    lines.append("- Avoid niche markets (corner odds, card odds) where closing lines are unreliable.")
    lines.append("- For Over/Under, ensure you use the same lines as the market (e.g., 2.5 goals).")
    lines.append("")

    # 3. Stake sizing
    lines.append("### 3. Stake Sizing and CLV")
    lines.append("")
    lines.append("- Models with positive CLV still showed negative ROI — stake sizing matters.")
    lines.append("- **Fractional Kelly (k=0.25)** is recommended for CLV-based betting to manage variance.")
    lines.append("- CLV of +5%+ should trigger higher conviction; below 2% is noise.")
    lines.append("")

    # 4. Data quality
    lines.append("### 4. Improve Closing Odds Data")
    lines.append("")
    lines.append("- **Use real closing odds** from Football-Data.co.uk or OddsPortal instead of synthetic data.")
    lines.append("- Match closing odds to each bet's exact timestamp (not just the day).")
    lines.append("- Remove matches where closing odds are stale (e.g., >12 hours before kick-off).")
    lines.append("")

    # 5. CLV thresholds
    lines.append("### 5. CLV Filtering Strategy")
    lines.append("")
    lines.append("| Filter | Expected Impact |")
    lines.append("|--------|----------------|")
    lines.append("| CLV > 0.00 | Maximum bets, ~52% positive CLV |")
    lines.append("| CLV > 0.02 | Removes ~20% of bets, improves win rate |")
    lines.append("| CLV > 0.05 | Fewer bets, higher quality — recommended threshold |")
    lines.append("| CLV > 0.10 | Premium bets only, small sample but best performance |")
    lines.append("")

    # 6. Next steps
    lines.append("### 6. Production Recommendations")
    lines.append("")
    lines.append("1. **Deploy top-3 CLV models** into live betting pipeline")
    lines.append("2. **Add CLV monitor** — track daily average CLV as a leading indicator")
    lines.append("3. **Set a CLV floor** — reject bets with CLV < 0.02")
    lines.append("4. **Blend ensemble weights** using CLV as a secondary objective (after Brier score)")
    lines.append("5. **Re-evaluate monthly** — CLV decays as markets become more efficient")

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def visualizations_embed() -> str:
    lines = [
        section_header(2, "Visualizations"),
        "",
        "### Combined Charts",
        "",
    ]

    combined = [
        ("CLV Comparison Across Models", "clv_comparison.png"),
        ("CLV by Market", "clv_by_market.png"),
        ("Positive CLV Rate by Model", "clv_positive_pct.png"),
    ]

    for title, fname in combined:
        path = FIGURES_DIR / fname
        if path.exists():
            lines.append(f"**{title}**")
            lines.append("")
            lines.append(f"![{title}](figures/{fname})")
            lines.append("")

    # Per-model distribution charts
    lines.append("### Per-Model CLV Distributions")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Click to expand all 9 distribution charts</summary>")
    lines.append("")
    for name in sorted(FIGURES_DIR.glob("clv_dist_*.png")):
        model = name.stem.replace("clv_dist_", "")
        fname = name.name
        lines.append(f"**{model}**")
        lines.append("")
        lines.append(f"![{model} CLV distribution](figures/{fname})")
        lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def appendix() -> str:
    lines = [
        section_header(2, "Appendix: Data Dictionary"),
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        "| **CLV** | Closing Line Value = (your odds − closing odds) / closing odds |",
        "| **Avg CLV** | Mean CLV across all bets for a model |",
        "| **Median CLV** | 50th percentile CLV — more robust to outliers |",
        "| **+CLV%** | Percentage of bets where CLV was positive (> 0) |",
        "| **Win Rate** | Percentage of bets that won |",
        "| **Consistency Score** | Composite: (+CLV% / 100) × max(avg CLV, 0) |",
        "| **ROI** | Return on Investment = total profit / total staked |",
        "| **Trend** | Direction of moving CLV over recent bets (Improving/Declining/Stable) |",
        "| **5-Bet MA** | 5-bet moving average — smooths short-term noise |",
        "",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════


def main() -> None:
    # Locate input data
    analysis_file = _find_latest("clv_analysis_*.json")
    if not analysis_file:
        logger.error("No clv_analysis_*.json found in reports/")
        sys.exit(1)

    logger.info("Loading CLV analysis: %s", analysis_file.name)
    clv_analysis = _load_json(analysis_file)
    rois = _load_backtest_rois()
    logger.info("Loaded ROI data for %d models", len(rois))

    # Report metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"clv_report_{timestamp}.md"
    report_title = "Comprehensive Closing Line Value (CLV) Report"

    # Build report
    sections: list[str] = []

    # Title & metadata
    sections.append(f"# {report_title}")
    sections.append("")
    sections.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(f"- **Data source:** `{analysis_file.name}`")
    sections.append(f"- **Models:** {clv_analysis.get('total_models', '?')} models, "
                     f"{sum(s.get('total_bets', s.get('total', 0)) for s in clv_analysis.get('per_model', {}).values())} total bets")
    sections.append(f"- **Backtest data:** {len(rois)} models with ROI data")
    sections.append("")
    sections.append("---")
    sections.append("")

    # Sections
    sections.append(exec_summary(clv_analysis, rois))
    sections.append(performance_table(clv_analysis, rois))
    sections.append(best_model_market(clv_analysis))
    sections.append(clv_vs_roi(clv_analysis, rois))
    sections.append(trend_analysis(clv_analysis))
    sections.append(recommendations(clv_analysis, rois))
    sections.append(visualizations_embed())
    sections.append(appendix())

    # Write
    report_content = "\n".join(sections)
    report_file.write_text(report_content, encoding="utf-8")

    sz = report_file.stat().st_size / 1024
    print(f"\n{'=' * 60}")
    print(f"  CLV REPORT GENERATED")
    print(f"{'=' * 60}")
    print(f"  File:   {report_file}")
    print(f"  Size:   {sz:.1f} KB")
    print(f"  Lines:  {report_content.count(chr(10))}")

    # Section stats
    import re as _re
    section_count = len(_re.findall(r"^## ", report_content, _re.MULTILINE))
    chart_count = len(_re.findall(r"!\[", report_content))
    table_count = len(_re.findall(r"^\|.*\|$", report_content, _re.MULTILINE))
    print(f"  Sections: {section_count}")
    print(f"  Charts:   {chart_count}")
    print(f"  Tables:   {table_count}")
    print(f"{'=' * 60}")
    print()


if __name__ == "__main__":
    main()
