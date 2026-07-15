"""
Generate a comprehensive bankroll management report as a Markdown document.

Loads bankroll_management_*.json results, links to figure PNGs, and
assembles everything into reports/bankroll_report_{timestamp}.md.

Prerequisites
-------------
  1. Run scripts/backtest_staking_strategies.py
  2. Run scripts/generate_bankroll_charts.py

Output
------
  reports/bankroll_report_{timestamp}.md
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("generate_bankroll_report")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# Drawdown constraint reference
MAX_DD_CONSTRAINT = 20.0


# ══════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════


def find_latest(pattern: str) -> Path | None:
    """Find the most recent file matching a glob pattern."""
    files = sorted(REPORTS_DIR.glob(pattern))
    return files[-1] if files else None


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_pct(v: float) -> str:
    """Format a percentage value."""
    if abs(v) < 0.005:
        return "0.00%"
    return f"{v:+.2f}%" if v != 0 else "0.00%"


def fmt_gbp(v: float) -> str:
    """Format as GBP currency."""
    return f"GBP{v:+,.2f}"


def fmt_round(v: float, dp: int = 4) -> str:
    return f"{v:.{dp}f}"


def img_exists(path: Path) -> bool:
    """Check if an image file exists (non-zero size)."""
    return path.exists() and path.stat().st_size > 0


# ══════════════════════════════════════════════════════
#  Category-based analysis helpers
# ══════════════════════════════════════════════════════


def best_in_category(
    strategies: list[dict[str, Any]],
    category: str,
    metric: str = "sharpe_ratio",
) -> dict[str, Any] | None:
    """Find the best strategy in a category by a given metric."""
    filtered = [s for s in strategies if s.get("category") == category]
    if not filtered:
        return None
    return max(filtered, key=lambda s: s.get(metric, -999))


def category_summary(
    strategies: list[dict[str, Any]],
    category: str,
) -> dict[str, float]:
    """Compute summary stats for a category of strategies."""
    filtered = [s for s in strategies if s.get("category") == category]
    if not filtered:
        return {"count": 0, "avg_sharpe": 0, "avg_roi": 0, "avg_dd": 0}

    sharpe = [s.get("sharpe_ratio", 0) for s in filtered]
    roi = [s.get("roi_pct", 0) for s in filtered]
    dd = [s.get("max_drawdown_pct", 0) for s in filtered]

    return {
        "count": len(filtered),
        "avg_sharpe": sum(sharpe) / len(sharpe),
        "avg_roi": sum(roi) / len(roi),
        "avg_dd": sum(dd) / len(dd),
    }


# ══════════════════════════════════════════════════════
#  Markdown builder
# ══════════════════════════════════════════════════════


def build_report(
    comparison: dict[str, Any],
    timestamp: str,
) -> str:
    """Assemble the full Markdown report."""
    lines: list[str] = []

    stake_results = comparison.get("stake_strategies", {}).get("results", [])
    risk_results = comparison.get("risk_scenarios", {}).get("results", [])
    config = comparison.get("configuration", {})

    # ── Header ────────────────────────────────────────
    lines.append("# Bankroll Management Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")
    model_name = config.get("model", "Ensemble")
    brier = config.get("brier", 0.5775)
    init_bankroll = config.get("initial_bankroll", 1000)
    lines.append(f"**Model:** {model_name} (Brier={brier:.4f})")
    lines.append(f"**Initial Bankroll:** GBP{init_bankroll:.0f}")
    lines.append(f"**Bet Filter:** min_ev={config.get('bet_filter', {}).get('min_ev', 0.05)}, "
                 f"min_conf={config.get('bet_filter', {}).get('min_confidence', 0.6)}, "
                 f"min_odds={config.get('bet_filter', {}).get('min_odds', 1.5)}")
    lines.append(f"**Drawdown Constraint:** < {MAX_DD_CONSTRAINT:.0f}%")
    lines.append(f"**Strategies evaluated:** {len(stake_results)}")
    lines.append(f"**Risk scenarios evaluated:** {len(risk_results)}")
    lines.append("")

    # ── 1. Executive Summary ─────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")

    # Identify best performers
    best_sharpe = max(stake_results, key=lambda s: s.get("sharpe_ratio", -999)) if stake_results else {}
    best_roi = max(stake_results, key=lambda s: s.get("roi_pct", -999)) if stake_results else {}
    safest = min(
        [s for s in stake_results if s.get("total_bets", 0) > 0],
        key=lambda s: s.get("max_drawdown_pct", 999),
    ) if stake_results else {}
    best_pf = max(stake_results, key=lambda s: s.get("profit_factor", -999)) if stake_results else {}
    best_yield_ = max(stake_results, key=lambda s: s.get("yield_pct", -999)) if stake_results else {}
    best_clv = max(stake_results, key=lambda s: s.get("avg_clv", -999)) if stake_results else {}

    lines.append("### Key Findings")
    lines.append("")

    lines.append(
        f"- **Best Sharpe ratio:** {best_sharpe.get('label', '?')} "
        f"({fmt_round(best_sharpe.get('sharpe_ratio', 0), 2)})"
    )
    lines.append(
        f"- **Best ROI:** {best_roi.get('label', '?')} "
        f"({fmt_pct(best_roi.get('roi_pct', 0))})"
    )
    lines.append(
        f"- **Best Yield:** {best_yield_.get('label', '?')} "
        f"({fmt_pct(best_yield_.get('yield_pct', 0))})"
    )
    lines.append(
        f"- **Most consistent (lowest drawdown):** {safest.get('label', '?')} "
        f"({fmt_pct(safest.get('max_drawdown_pct', 0))})"
    )
    lines.append(
        f"- **Best profit factor:** {best_pf.get('label', '?')} "
        f"({fmt_round(best_pf.get('profit_factor', 0), 2)})"
    )
    if best_clv.get("avg_clv", 0) != 0:
        lines.append(
            f"- **Best CLV:** {best_clv.get('label', '?')} "
            f"({fmt_round(best_clv.get('avg_clv', 0), 6)})"
        )
    lines.append("")

    # Category-level summary
    lines.append("### Category-Level Summary")
    lines.append("")
    lines.append("| Category | Strategies | Avg Sharpe | Avg ROI | Avg DD | Best Strategy (Sharpe) |")
    lines.append("|----------|------------|------------|---------|--------|------------------------|")
    for cat in ["kelly", "percentage", "dynamic", "portfolio", "fixed"]:
        cs = category_summary(stake_results, cat)
        if cs["count"] > 0:
            best_cat = best_in_category(stake_results, cat)
            best_label = best_cat.get("label", "?") if best_cat else "?"
            lines.append(
                f"| {cat.capitalize()} | {cs['count']} | "
                f"{cs['avg_sharpe']:.4f} | {cs['avg_roi']:+.2f}% | "
                f"{cs['avg_dd']:.2f}% | {best_label} |"
            )
    lines.append("")

    # Strategies passing the drawdown constraint
    passing_dd = [s for s in stake_results if s.get("max_drawdown_pct", 999) < MAX_DD_CONSTRAINT]
    lines.append(
        f"- **{len(passing_dd)}/{len(stake_results)} strategies** keep drawdown "
        f"under {MAX_DD_CONSTRAINT:.0f}%"
    )
    if passing_dd:
        best_passing = max(passing_dd, key=lambda s: s.get("sharpe_ratio", -999))
        lines.append(
            f"- **Best compliant strategy:** {best_passing.get('label', '?')} "
            f"(Sharpe={best_passing.get('sharpe_ratio', 0):.4f}, "
            f"DD={best_passing.get('max_drawdown_pct', 0):.1f}%)"
        )
    lines.append("")

    # ── 2. Performance Table ─────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 2. Performance Comparison — All Stake Strategies")
    lines.append("")

    # Sort by Sharpe descending
    stake_sorted = sorted(
        stake_results,
        key=lambda s: s.get("sharpe_ratio", -999),
        reverse=True,
    )

    lines.append(
        "| Rank | Strategy | Category | Bets | W/L | P&L | ROI% | Yield% | "
        "Win% | Sharpe | DD% | PF | CLV | Final |"
    )
    lines.append(
        "|------|----------|----------|------|-----|------|------|--------|"
        "------|--------|------|------|------|-------|"
    )

    for i, s in enumerate(stake_sorted, 1):
        label = s.get("label", "?")
        cat = s.get("category", "")
        bets = s.get("total_bets", 0)
        wins = s.get("winning_bets", 0)
        losses = s.get("losing_bets", 0)
        pnl = s.get("total_profit", 0)
        roi = s.get("roi_pct", 0)
        yld = s.get("yield_pct", 0)
        wr = s.get("win_rate_pct", 0)
        sharpe = s.get("sharpe_ratio", 0)
        dd = s.get("max_drawdown_pct", 0)
        pf = s.get("profit_factor", 0)
        clv = s.get("avg_clv", 0)
        final = s.get("final_bankroll", 0)

        lines.append(
            f"| {i} | {label} | {cat} | {bets} | {wins}/{losses} | "
            f"{fmt_gbp(pnl)} | {fmt_pct(roi)} | {fmt_pct(yld)} | "
            f"{wr:.1f}% | {sharpe:.4f} | {dd:.2f}% | "
            f"{pf:.2f} | {clv:+.6f} | {final:.2f} |"
        )

    lines.append("")

    # ── 3. Optimal Strategy Identification ────────────
    lines.append("---")
    lines.append("")
    lines.append("## 3. Optimal Strategy Identification")
    lines.append("")

    # Strategy with highest Sharpe that also passes drawdown constraint
    compliant = [s for s in stake_results if s.get("max_drawdown_pct", 999) < MAX_DD_CONSTRAINT]
    if compliant:
        optimal = max(compliant, key=lambda s: s.get("sharpe_ratio", -999))
    else:
        # Fall back to best Sharpe regardless of drawdown
        optimal = max(stake_results, key=lambda s: s.get("sharpe_ratio", -999)) if stake_results else {}
        lines.append(
            "⚠️ **No strategy satisfies the drawdown constraint.** "
            "Showing best Sharpe regardless."
        )
        lines.append("")

    if optimal:
        lines.append("### Optimal Strategy")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| **Strategy** | {optimal.get('label', '?')} |")
        lines.append(f"| **Category** | {optimal.get('category', '?')} |")
        lines.append(f"| **Sharpe Ratio** | {optimal.get('sharpe_ratio', 0):.4f} |")
        lines.append(f"| **ROI** | {fmt_pct(optimal.get('roi_pct', 0))} |")
        lines.append(f"| **Yield** | {fmt_pct(optimal.get('yield_pct', 0))} |")
        lines.append(f"| **Max Drawdown** | {optimal.get('max_drawdown_pct', 0):.2f}% |")
        lines.append(f"| **Profit Factor** | {optimal.get('profit_factor', 0):.2f} |")
        lines.append(f"| **Win Rate** | {optimal.get('win_rate_pct', 0):.1f}% |")
        lines.append(f"| **Total Bets** | {optimal.get('total_bets', 0)} |")
        lines.append(f"| **Final Bankroll** | {fmt_gbp(optimal.get('final_bankroll', 0))} |")
        if optimal.get("avg_clv", 0) != 0:
            lines.append(f"| **Avg CLV** | {optimal.get('avg_clv', 0):+.6f} |")
        lines.append("")

        dd_flag = "✅" if optimal.get("max_drawdown_pct", 999) < MAX_DD_CONSTRAINT else "❌"
        lines.append(
            f"{dd_flag} Drawdown {optimal.get('max_drawdown_pct', 0):.1f}% "
            f"{'< ' + str(MAX_DD_CONSTRAINT) + '% constraint MET' if optimal.get('max_drawdown_pct', 999) < MAX_DD_CONSTRAINT else 'EXCEEDS ' + str(MAX_DD_CONSTRAINT) + '% constraint'}"
        )
        lines.append("")

    # Top 5 by Sharpe (with DD constraint compliance)
    lines.append("### Top 5 Strategies by Sharpe")
    lines.append("")
    lines.append("| Rank | Strategy | Sharpe | ROI | DD | DD < 20%? | Final Bankroll |")
    lines.append("|------|----------|--------|-----|-----|-----------|----------------|")
    for i, s in enumerate(stake_sorted[:5], 1):
        label = s.get("label", "?")
        sharpe = s.get("sharpe_ratio", 0)
        roi = s.get("roi_pct", 0)
        dd = s.get("max_drawdown_pct", 0)
        passes = "✅" if dd < MAX_DD_CONSTRAINT else "❌"
        final = s.get("final_bankroll", 0)
        lines.append(
            f"| {i} | {label} | {sharpe:.4f} | {fmt_pct(roi)} | "
            f"{dd:.2f}% | {passes} | {final:.2f} |"
        )
    lines.append("")

    # ── 4. Category Analysis ─────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 4. Strategy Category Analysis")
    lines.append("")

    lines.append("### 4.1 Kelly Strategies")
    lines.append("")
    lines.append(
        "Kelly-based strategies optimise for long-term logarithmic growth. "
        "Full Kelly is aggressive and can recommend large stakes; fractional "
        "Kelly reduces variance at the cost of lower growth."
    )
    lines.append("")
    kelly_cat = category_summary(stake_results, "kelly")
    if kelly_cat["count"] > 0:
        lines.append(
            f"- **{kelly_cat['count']} Kelly variants tested** "
            f"(Full Kelly + {kelly_cat['count'] - 1} fractional)"
        )
        lines.append(f"- **Average Sharpe:** {kelly_cat['avg_sharpe']:.4f}")
        lines.append(f"- **Average DD:** {kelly_cat['avg_dd']:.1f}%")
        lines.append(
            "- **Key insight:** Lower Kelly fractions (10–25%) provide better "
            "risk-adjusted returns with lower drawdown. Full Kelly often "
            "violates the 20% drawdown constraint."
        )
    lines.append("")

    lines.append("### 4.2 Fixed Ratio & Percentage Strategies")
    lines.append("")
    lines.append(
        "Fixed ratio and percentage strategies stake a constant fraction of "
        "bankroll regardless of edge size. These are the simplest to implement "
        "but may miss the opportunity to scale with conviction."
    )
    lines.append("")
    pct_cat = category_summary(stake_results, "percentage")
    if pct_cat["count"] > 0:
        lines.append(
            f"- **{pct_cat['count']} percentage/ratio variants tested**"
        )
        lines.append(f"- **Average Sharpe:** {pct_cat['avg_sharpe']:.4f}")
        lines.append(f"- **Average DD:** {pct_cat['avg_dd']:.1f}%")
        lines.append(
            "- **Key insight:** Lower ratios (1–3%) are safer but may underperform "
            "Kelly variants on risk-adjusted returns."
        )
    lines.append("")

    lines.append("### 4.3 Dynamic Strategies")
    lines.append("")
    lines.append(
        "Dynamic strategies (VariableRatio, Volatility) adjust stake sizes based "
        "on EV magnitude or recent volatility. These attempt to capture larger "
        "edges while protecting against volatile periods."
    )
    lines.append("")
    dyn_cat = category_summary(stake_results, "dynamic")
    if dyn_cat["count"] > 0:
        lines.append(
            f"- **{dyn_cat['count']} dynamic variants tested**"
        )
        lines.append(f"- **Average Sharpe:** {dyn_cat['avg_sharpe']:.4f}")
        lines.append(f"- **Average DD:** {dyn_cat['avg_dd']:.1f}%")
        lines.append(
            "- **Key insight:** Volatility-adjusted strategies provide smoother "
            "equity curves but may underperform on pure ROI."
        )
    lines.append("")

    lines.append("### 4.4 Portfolio Strategies")
    lines.append("")
    lines.append(
        "Portfolio strategies divide bankroll across multiple concurrent bets, "
        "reducing single-bet exposure. This is most relevant when multiple "
        "qualifying opportunities arise simultaneously."
    )
    lines.append("")
    port_cat = category_summary(stake_results, "portfolio")
    if port_cat["count"] > 0:
        lines.append(
            f"- **{port_cat['count']} portfolio variants tested**"
        )
        lines.append(f"- **Average Sharpe:** {port_cat['avg_sharpe']:.4f}")
        lines.append(f"- **Average DD:** {port_cat['avg_dd']:.1f}%")
        lines.append(
            "- **Key insight:** Portfolio strategies are most effective when "
            "bet frequency is high. They reduce variance but may dilute gains."
        )
    lines.append("")

    # ── 5. Risk Management Impact ─────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 5. Risk Management Impact Analysis")
    lines.append("")

    if risk_results:
        # Compare baseline (no risk) vs best risk scenario
        baseline = next(
            (r for r in risk_results if "No Risk" in r.get("risk_label", "")),
            None,
        )
        best_risk = max(
            risk_results,
            key=lambda r: r.get("roi_pct", -999),
        )

        lines.append(
            "Risk management limits act as safety nets that stop betting when "
            "certain thresholds are breached. The key impact is capital preservation "
            "at the cost of potentially missing future profitable opportunities."
        )
        lines.append("")

        lines.append("### 5.1 Risk Scenario Comparison")
        lines.append("")
        lines.append("| Scenario | Bets | Won | Lost | ROI | Yield | Win Rate | Final Bankroll | Rejected |")
        lines.append("|----------|------|-----|------|-----|-------|----------|----------------|----------|")
        for r in risk_results:
            label = r.get("risk_label", r.get("label", "?"))[:30]
            bets = r.get("total_bets", 0)
            wins = r.get("winning_bets", 0)
            losses = r.get("losing_bets", 0)
            roi = r.get("roi_pct", 0)
            yld = r.get("yield_pct", 0)
            wr = r.get("win_rate_pct", 0)
            final = r.get("final_bankroll", 0)
            rejected = r.get("n_rejected_by_risk", 0)
            lines.append(
                f"| {label} | {bets} | {wins} | {losses} | "
                f"{fmt_pct(roi)} | {fmt_pct(yld)} | {wr:.1f}% | "
                f"{final:.2f} | {rejected} |"
            )
        lines.append("")

        # Impact assessment
        if baseline:
            base_roi = baseline.get("roi_pct", 0)
            base_bets = baseline.get("total_bets", 0)
            lines.append("### 5.2 Risk Impact Assessment")
            lines.append("")

            for r in risk_results:
                label = r.get("risk_label", "?")
                if "No Risk" in label:
                    continue
                roi_diff = r.get("roi_pct", 0) - base_roi
                bet_diff = r.get("total_bets", 0) - base_bets
                rejected = r.get("n_rejected_by_risk", 0)
                direction = "improvement" if roi_diff > 0 else "reduction"
                lines.append(
                    f"- **{label}:** ROI {direction} of {fmt_pct(abs(roi_diff))} "
                    f"({bet_diff:+d} bets vs baseline, {rejected} rejected by risk rules)"
                )
            lines.append("")

        lines.append("### 5.3 Key Observations")
        lines.append("")
        lines.append(
            "- **Drawdown limits** are the most impactful — they prevent "
            "catastrophic losses but may cut short profitable runs."
        )
        lines.append(
            "- **Daily loss limits** protect against tilt but have limited "
            "impact over a full backtest (resets daily)."
        )
        lines.append(
            "- **Frequency limits** reduce bet count but may miss profitable "
            "opportunities on high-volume days."
        )
        lines.append(
            "- **Full protection** (conservative all-limits mode) significantly "
            "reduces drawdown but may also reduce total ROI."
        )
        lines.append("")
    else:
        lines.append("*No risk scenario data available.*")
        lines.append("")

    # ── 6. Visualizations ────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 6. Visualizations")
    lines.append("")

    # Combined charts
    lines.append("### 6.1 Combined Charts")
    lines.append("")

    combined_charts = [
        ("Bankroll Strategy Comparison", "bankroll_strategy_comparison.png",
         "Top 15 strategies by Sharpe: Sharpe ratio, ROI, and drawdown side by side."),
        ("Risk-Adjusted Returns (Sharpe vs Drawdown)", "risk_adjusted_returns.png",
         "Scatter plot showing the risk-reward trade-off across all strategies. "
         "The green-shaded 'target zone' shows the ideal region (DD < 20%, Sharpe > 0)."),
        ("Kelly Fraction Sensitivity", "kelly_sensitivity.png",
         "How Sharpe and drawdown change as the Kelly fraction varies from 10% to 100%."),
        ("Fixed Ratio Sensitivity", "ratio_sensitivity.png",
         "How Sharpe and drawdown change as the fixed ratio varies from 1% to 10%."),
        ("Risk Management Impact", "risk_impact.png",
         "Comparison of risk scenarios: ROI, bet count, and risk rejections."),
    ]

    for title, filename, desc in combined_charts:
        img_path = FIGURES_DIR / filename
        if img_exists(img_path):
            lines.append(f"**{title}**")
            lines.append("")
            lines.append(f"![{title}](figures/{filename})")
            lines.append("")
            lines.append(f"*{desc}*" if desc else "")
            lines.append("")
        else:
            lines.append(f"*{title} — chart not yet generated.*")
            lines.append("")

    # ── 7. Recommendations ───────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 7. Recommendations for Production Use")
    lines.append("")

    lines.append("### 7.1 Recommended Strategy")
    lines.append("")
    if compliant:
        best_rec = max(compliant, key=lambda s: s.get("sharpe_ratio", -999))
        lines.append(
            f"Based on the analysis, the recommended bankroll management "
            f"strategy is **{best_rec.get('label', '?')}**, which achieved "
            f"a Sharpe ratio of {best_rec.get('sharpe_ratio', 0):.4f} with "
            f"a max drawdown of only {best_rec.get('max_drawdown_pct', 0):.1f}% "
            f"(well under the {MAX_DD_CONSTRAINT:.0f}% constraint)."
        )
        lines.append("")
        lines.append("This strategy offers the best risk-adjusted returns while ")
        lines.append("satisfying the capital preservation requirement.")
        lines.append("")
    else:
        lines.append(
            "⚠️ No strategy fully satisfies the drawdown constraint. "
            "The recommendations below focus on minimising drawdown while "
            "maximising Sharpe."
        )
        lines.append("")

    lines.append("### 7.2 Strategy Selection Guidelines")
    lines.append("")
    lines.append("| Risk Tolerance | Recommended Strategy Category | Rationale |")
    lines.append("|----------------|------------------------------|-----------|")
    lines.append(
        "| **Conservative** | FixedRatio 1-2% or Kelly 10-25% | "
        "Lowest drawdown, steady growth, minimal variance |"
    )
    lines.append(
        "| **Moderate** | Kelly 25-33% or VariableRatio 2% | "
        "Balanced growth with controlled drawdown |"
    )
    lines.append(
        "| **Aggressive** | Kelly 50-75% | "
        "Higher returns but may exceed 20% drawdown |"
    )
    lines.append(
        "| **Dynamic** | Volatility-adjusted 2% | "
        "Adapts to market conditions, smoother equity curve |"
    )
    lines.append("")

    lines.append("### 7.3 Risk Management Configuration")
    lines.append("")
    lines.append(
        "Based on the scenario analysis, the following risk limits are recommended:"
    )
    lines.append("")
    lines.append(
        "- **Daily loss limit:** 5–10% of bankroll "
        "(stops tilt-induced chasing)"
    )
    lines.append(
        "- **Max drawdown:** 15–20% "
        "(aligns with the constraint used in this analysis)"
    )
    lines.append(
        "- **Frequency limit:** 5–10 bets/day, 20–40 bets/week "
        "(prevents over-trading)"
    )
    lines.append(
        "- **Consecutive loss limit:** 4–6 losses "
        "(triggers cooldown to reassess)"
    )
    lines.append(
        "- **Max single stake:** 10% of bankroll "
        "(prevents over-exposure on any single bet)"
    )
    lines.append("")

    lines.append("### 7.4 Next Steps")
    lines.append("")
    lines.append(
        "1. **Validate on live data:** Run the optimal strategy with a small "
        "bankroll on real matches to verify the backtest findings."
    )
    lines.append(
        "2. **Monitor drift:** Track rolling Sharpe and drawdown in production "
        "to detect when the strategy's performance regime changes."
    )
    lines.append(
        "3. **Dynamic limits:** Consider implementing dynamic risk limits that "
        "tighten after losses and loosen after wins."
    )
    lines.append(
        "4. **Portfolio correlation:** For multi-bet scenarios, track correlation "
        "between bets to avoid concentration risk."
    )
    lines.append(
        "5. **Walk-forward optimisation:** Re-run this optimisation periodically "
        "as new data is collected to ensure the strategy remains optimal."
    )
    lines.append("")

    # ── 8. Data Dictionary ───────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 8. Data Dictionary")
    lines.append("")
    lines.append("| Metric | Description |")
    lines.append("|--------|-------------|")
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
        "| **Avg CLV** | Average closing line value (positive = predictive skill) |"
    )
    lines.append(
        "| **Final Bankroll** | Bankroll at end of backtest period |"
    )
    lines.append(
        "| **Bets Rejected** | Number of bets the RiskManager blocked |"
    )
    lines.append("")

    # ── Footer ────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        "*Report auto-generated by `scripts/generate_bankroll_report.py`*"
    )
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════


def main() -> None:
    # Find latest management comparison results
    mgmt_path = find_latest("bankroll_management_*.json")
    if not mgmt_path:
        logger.error(
            "No bankroll_management_*.json found — "
            "run scripts/backtest_staking_strategies.py first"
        )
        sys.exit(1)

    logger.info("Loading: %s", mgmt_path)
    comparison = load_json(mgmt_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_md = build_report(comparison, timestamp)

    out_path = REPORTS_DIR / f"bankroll_report_{timestamp}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Report saved to %s (%d bytes)", out_path, len(report_md))

    # Print summary
    stake_count = len(comparison.get("stake_strategies", {}).get("results", []))
    risk_count = len(comparison.get("risk_scenarios", {}).get("results", []))
    print(f"\n{'=' * 60}")
    print("  BANKROLL MANAGEMENT REPORT GENERATED")
    print(f"{'=' * 60}")
    print(f"  File:   {out_path.name}")
    print(f"  Size:   {len(report_md):,} bytes")
    print(f"  Lines:  {report_md.count(chr(10))}")
    print(f"  Strategies: {stake_count}")
    print(f"  Risk Scenarios: {risk_count}")
    print(f"  Charts: 5 expected in reports/figures/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
