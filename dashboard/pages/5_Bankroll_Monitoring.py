"""
Bankroll Monitoring Dashboard — track bankroll growth, drawdown, risk metrics,
optimal stake sizing, and run what-if simulations with rich Plotly visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    init_theme,
    sidebar_theme_radio,
    render_custom_css,
    render_hero,
    render_footer,
    section_header,
    section_header_sm,
    metric_card,
    status_badge,
    gauge_chart,
    bankroll_growth_chart,
    drawdown_chart,
    comparison_bar_chart,
    info_row,
    Colors,
)

st.set_page_config(page_title="Bankroll Monitoring", page_icon="🏦", layout="wide")

# ── Theme initialisation ───────────────────────────────
init_theme()

# ── Sidebar theme toggle ───────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    sidebar_theme_radio()
    st.markdown("---")

render_custom_css()

render_hero(
    title="🏦 Bankroll Monitoring",
    subtitle="Monitor bankroll growth, drawdown, risk metrics, and optimal stake sizing. "
             "Run what-if simulations to stress-test your betting strategy.",
    badges=[("Risk management", "🛡️"), ("What-if simulator", "🔬")],
)


# ── Data helpers ──────────────────────────────────────
@st.cache_data(ttl=60)
def load_reports() -> list[dict]:
    """Load bankroll-related report files."""
    results = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return results
    for pattern in [
        "bankroll_management_*.json", "bankroll_optimization_*.json",
        "bankroll_report*.json", "bankroll_*.json",
    ]:
        for f in sorted(reports_dir.glob(pattern), reverse=True):
            try:
                with open(f) as fh:
                    results.append({"file": f.name, "data": json.load(fh)})
            except Exception:
                pass
    return results


reports = load_reports()

if not reports:
    st.info(
        '<div style="background:#141824;border:1px solid #1e2235;border-radius:12px;'
        'padding:2rem;text-align:center">'
        '<div style="font-size:3rem;margin-bottom:0.5rem">📭</div>'
        '<div style="color:#9ca3af;font-size:1rem">No bankroll reports found.</div>'
        '<div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">'
        'Run backtest/optimize scripts first.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Extract optimal strategy ──────────────────────────
best_strategy = None
best_sharpe = -999

for r in reports:
    data = r["data"]
    if "best_strategy" in data:
        bs = data["best_strategy"]
        sharpe = bs.get("sharpe_ratio", -999)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_strategy = bs
            break

    strategies = []
    for section in ["stake_strategies", "risk_scenarios"]:
        if section in data:
            strategies = data[section].get("results", [])
            for s in strategies:
                if s.get("total_bets", 0) > 0:
                    sh = s.get("sharpe_ratio", -999)
                    if sh > best_sharpe:
                        best_sharpe = sh
                        best_strategy = s


# ── Optimal Strategy Section ──────────────────────────
section_header("🏆 Optimal Strategy", "🏆")

if best_strategy:
    strategy_name = best_strategy.get("strategy", best_strategy.get("name", best_strategy.get("model_name", "Optimal")))
    status_badge(f"Strategy: {strategy_name}", "green")
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    sharpe = float(best_strategy.get("sharpe_ratio", 0))
    total_profit = float(best_strategy.get("total_profit", 0))
    drawdown_val = float(best_strategy.get("max_drawdown_pct", 0))
    win_rate_val = float(best_strategy.get("win_rate", best_strategy.get("win_rate_pct", 0)))

    metric_card(col1, f"{sharpe:.2f}", "Sharpe Ratio", up=sharpe > 0)
    metric_card(col2, f"£{total_profit:+,.2f}", "Total P&L", up=total_profit > 0)
    metric_card(col3, f"{drawdown_val:.1f}%", "Max Drawdown", up=drawdown_val < 20)
    metric_card(col4, f"{win_rate_val:.1f}%", "Win Rate", up=win_rate_val > 50)

    # Performance gauges
    st.markdown("### Performance Gauges")
    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    gauge_chart(gcol1, "Sharpe Ratio", sharpe, target=1.0, lower_better=False)
    gauge_chart(gcol2, "Win Rate", win_rate_val, target=50.0, unit="%")
    gauge_chart(gcol3, "Max Drawdown", drawdown_val, target=20.0, unit="%", lower_better=True)
    gauge_chart(gcol4, "Profit Factor", float(best_strategy.get("profit_factor", 0)), target=1.5, lower_better=False)

    # Extended metrics
    smcol1, smcol2, smcol3, smcol4 = st.columns(4)
    metric_card(smcol1, str(int(best_strategy.get("total_bets", 0))), "Total Bets")
    roi_val = float(best_strategy.get("roi", best_strategy.get("roi_pct", 0)))
    metric_card(smcol2, f"{roi_val:+.2f}%", "ROI", up=roi_val > 0)
    metric_card(smcol3, f"{float(best_strategy.get('profit_factor', 0)):.2f}", "Profit Factor",
                up=float(best_strategy.get('profit_factor', 0)) > 1)


# ── Reports browser ──────────────────────────────────
section_header("📋 Reports Browser", "📋")
report_names = sorted(set(r["file"] for r in reports))
sel_file = st.selectbox("Choose a report:", report_names)
sel = next(r for r in reports if r["file"] == sel_file)
data = sel["data"]

# ── Strategy comparison ────────────────────────────────
strategies = []
for section in ["stake_strategies", "risk_scenarios"]:
    if section in data:
        strategies = data[section].get("results", data.get(section, []))

if not strategies and isinstance(data, dict):
    for key in ["results", "strategies", "scenarios"]:
        items = data.get(key, [])
        if isinstance(items, list):
            strategies = items
            break

if strategies:
    s_df = pd.DataFrame(strategies)
    if "total_bets" in s_df.columns:
        s_df = s_df[s_df["total_bets"] > 0]

    if len(s_df) > 0:
        section_header("⚖️ Strategy Comparison", "⚖️")
        sort_col = "sharpe_ratio" if "sharpe_ratio" in s_df.columns else None
        if sort_col:
            s_df = s_df.sort_values(sort_col, ascending=False)

        cols = [c for c in [
            "strategy", "name", "category", "sharpe_ratio", "roi_pct", "roi",
            "total_profit", "max_drawdown_pct", "win_rate_pct", "win_rate",
            "total_bets", "profit_factor",
        ] if c in s_df.columns]

        display_df = s_df[cols].copy() if cols else s_df

        # Format
        for c in display_df.columns:
            if c in ("roi_pct", "win_rate_pct", "max_drawdown_pct"):
                display_df[c] = display_df[c].apply(
                    lambda x: f"{float(x):+.2f}%" if "roi" in c else f"{float(x):.1f}%"
                )
            if c == "total_profit":
                display_df[c] = display_df[c].apply(lambda x: f"£{float(x):+,.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Sharpe vs Drawdown scatter ───────────────────
        if "sharpe_ratio" in s_df.columns and "max_drawdown_pct" in s_df.columns:
            section_header("🎯 Risk-Adjusted Return", "🎯")
            name_col = "strategy" if "strategy" in s_df.columns else s_df.columns[0]
            fig = px.scatter(
                s_df, x="max_drawdown_pct", y="sharpe_ratio",
                size="total_bets" if "total_bets" in s_df.columns else None,
                color="sharpe_ratio",
                hover_name=name_col,
                color_continuous_scale="RdYlGn",
                title="Each dot = one strategy. Top-left = best risk-adjusted returns.",
                labels={"max_drawdown_pct": "Max Drawdown (%)", "sharpe_ratio": "Sharpe Ratio"},
            )
            fig.add_hline(y=1, line_dash="dash", line_color=Colors.WARNING,
                          annotation_text="Sharpe=1")
            fig.add_vline(x=20, line_dash="dash", line_color=Colors.DANGER,
                          annotation_text="DD=20%")
            fig.update_layout(
                height=400,
                **{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                   "font": {"color": Colors.TEXT_PRIMARY}},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Highlight best strategy
            if sort_col in s_df.columns:
                best_idx = s_df[sort_col].idxmax()
                best_row = s_df.loc[best_idx]
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#141824 0%,#1a1f2e 100%);'
                    f'border:1px solid #4fc3f7;border-radius:12px;padding:1rem 1.5rem;margin-top:0.5rem">'
                    f'<div style="color:#4fc3f7;font-weight:600;font-size:0.85rem">🏆 RECOMMENDED STRATEGY</div>'
                    f'<div style="color:#e0e0e0;font-size:1.1rem;font-weight:700;margin-top:0.2rem">'
                    f'{best_row.get(name_col, "?")}</div>'
                    f'<div style="color:#6b7280;font-size:0.8rem;margin-top:0.3rem">'
                    f'Sharpe: {best_row.get("sharpe_ratio", 0):.2f} · '
                    f'ROI: {best_row.get("roi_pct", 0):+.2f}% · '
                    f'Win Rate: {best_row.get("win_rate_pct", 0):.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
else:
    with st.expander("📄 Raw Report Data"):
        st.json(data)


# ── What-If Bankroll Simulator ─────────────────────────
section_header("🔬 What-If Bankroll Simulator", "🔬")
st.markdown(
    '<p style="color:#9ca3af;font-size:0.9rem">'
    'Simulate a bankroll with custom parameters to test different betting strategies '
    'before deploying with real money.</p>',
    unsafe_allow_html=True,
)

sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
initial_br = sim_col1.number_input("Initial Bankroll (£)", min_value=100, value=1000, step=100)
stake_pct = sim_col2.slider("Stake (% of bankroll)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
n_bets_sim = sim_col3.number_input("Number of Bets", min_value=10, max_value=10000, value=500, step=100)

win_rate_sim = sim_col4.slider("Win Rate (%)", min_value=20, max_value=80, value=52, step=1)
avg_odds_sim = st.slider("Average Odds", min_value=1.2, max_value=15.0, value=2.0, step=0.1)

if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running Monte Carlo simulation..."):
        rng = np.random.default_rng()
        bankroll = initial_br
        history = [bankroll]
        peak = bankroll
        max_dd = 0

        for _ in range(n_bets_sim):
            stake = bankroll * (stake_pct / 100)
            stake = min(stake, bankroll * 0.5)  # Safety cap
            won = rng.random() < (win_rate_sim / 100)
            if won:
                profit = stake * (avg_odds_sim - 1)
            else:
                profit = -stake
            bankroll += profit
            bankroll = max(bankroll, 0)
            history.append(bankroll)

            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        sim_df = pd.DataFrame({"Bet": range(len(history)), "Bankroll": history})

        # Metrics
        final_br = history[-1]
        total_pnl = final_br - initial_br
        roi_sim = (total_pnl / initial_br) * 100 if initial_br > 0 else 0

        st.markdown("### Simulation Results")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)

        metric_card(sc1, f"£{final_br:,.2f}", "Final Bankroll",
                    delta=f"£{total_pnl:+,.2f}", up=total_pnl > 0)
        metric_card(sc2, f"{roi_sim:+.1f}%", "ROI", up=roi_sim > 0)
        metric_card(sc3, f"{max_dd:.1f}%", "Max Drawdown", up=max_dd < 20)
        metric_card(sc4, f"{win_rate_sim}%", "Assumed Win Rate")
        metric_card(sc5, f"{stake_pct:.1f}%", "Stake Size")

        # Bankroll growth chart
        fig1 = bankroll_growth_chart(history, initial=initial_br, height=350)
        st.plotly_chart(fig1, use_container_width=True)

        # Drawdown chart
        if len(history) > 1:
            fig2 = drawdown_chart(history, height=180)
            st.plotly_chart(fig2, use_container_width=True)

        # Final verdict
        if max_dd > 30:
            status_badge("⚠️ High Drawdown Risk — Consider reducing stake size", "red")
        elif max_dd > 15:
            status_badge("⚠️ Moderate Drawdown Risk — Monitor closely", "yellow")
        else:
            status_badge("✅ Low Drawdown Risk — Strategy looks healthy", "green")

        if final_br > initial_br * 2:
            status_badge(f"🚀 Portfolio doubled from £{initial_br:,.0f} to £{final_br:,.0f}", "green")
        st.markdown("")


render_footer()
