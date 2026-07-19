"""
Betting Results Dashboard — P&L charts, win rates, ROI analysis, streaks,
with rich Plotly visualizations and interactive filtering.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

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
    status_dot,
    gauge_chart,
    bankroll_growth_chart,
    drawdown_chart,
    comparison_bar_chart,
    area_trend_chart,
    info_row,
    Colors,
)

st.set_page_config(page_title="Betting Results", page_icon="💰", layout="wide")

# ── Theme initialisation ───────────────────────────────
init_theme()

# ── Sidebar theme toggle ───────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    sidebar_theme_radio()
    st.markdown("---")

render_custom_css()

render_hero(
    title="💰 Betting Results",
    subtitle="Analyse betting P&L, win rates, ROI, and streak performance across "
             "all models, markets, and strategies.",
    badges=[("Backtest analysis", "📊"), ("Strategy comparison", "🎯")],
)


# ── Data helpers ──────────────────────────────────────
@st.cache_data(ttl=60)
def load_backtest_data() -> list[dict]:
    """Load all backtest result JSON files."""
    results = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return results
    for pattern in [
        "backtest_*.json", "backtest_summary_*.json",
        "bankroll_management_*.json", "bankroll_optimization_*.json",
        "betting_report*.json", "realistic_backtest*.json",
    ]:
        for f in reports_dir.glob(pattern):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                results.append({"file": f.name, "data": data})
            except Exception:
                pass
    return results


def extract_metrics(d: dict) -> dict:
    """Normalise metrics from various backtest report formats."""
    m = {}
    if "metrics" in d:
        m = d["metrics"]
    elif "stake_strategies" in d:
        strategies = d["stake_strategies"].get("results", [])
        if strategies:
            best = max(strategies, key=lambda s: s.get("sharpe_ratio", 0) if s.get("total_bets", 0) > 0 else -999)
            m = best
            m["model_name"] = best.get("strategy", best.get("name", "Unknown"))
    elif "results" in d and isinstance(d["results"], list):
        if d["results"]:
            m = d["results"][0]
    else:
        m = d
    return m


backtests = load_backtest_data()

if not backtests:
    st.info(
        '<div style="background:#141824;border:1px solid #1e2235;border-radius:12px;'
        'padding:2rem;text-align:center">'
        '<div style="font-size:3rem;margin-bottom:0.5rem">📭</div>'
        '<div style="color:#9ca3af;font-size:1rem">No backtest results found.</div>'
        '<div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">'
        'Run backtest scripts first.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Overview ──────────────────────────────────────────
# Aggregate metrics across all reports
all_metrics = []
for bt in backtests:
    m = extract_metrics(bt["data"])
    m["_file"] = bt["file"]
    all_metrics.append(m)

overall_df = pd.DataFrame(all_metrics)

section_header("📊 Aggregate Overview", "📊")

col1, col2, col3, col4 = st.columns(4)
avg_roi = overall_df.get("roi_pct", overall_df.get("roi", pd.Series([0]))).mean()
avg_win_rate = overall_df.get("win_rate_pct", overall_df.get("win_rate", pd.Series([0]))).mean()
avg_sharpe = overall_df.get("sharpe_ratio", pd.Series([0])).mean()
avg_profit = overall_df.get("total_profit", pd.Series([0])).mean()

metric_card(col1, f"{avg_roi:+.2f}%", "Avg ROI", delta=f"Across {len(backtests)} reports", up=avg_roi > 0)
metric_card(col2, f"{avg_win_rate:.1f}%", "Avg Win Rate", delta=f"{'Above' if avg_win_rate > 50 else 'Below'} 50%", up=avg_win_rate > 50)
metric_card(col3, f"{avg_sharpe:.2f}", "Avg Sharpe", delta=f"{'Healthy' if avg_sharpe > 1 else 'Needs work'}", up=avg_sharpe > 0)
metric_card(col4, f"£{avg_profit:+,.2f}", "Avg Total P&L", up=avg_profit > 0)


# ── Reports browser ──────────────────────────────────
section_header("📋 Backtest Reports", "📋")
report_names_clean = [b["file"] for b in backtests]
selected_file = st.selectbox("Choose a report:", report_names_clean, label_visibility="collapsed")

report = next((b for b in backtests if b["file"] == selected_file), None)
if report is None:
    st.error(f"Report file '{selected_file}' not found.")
    st.stop()

data = report["data"]
metrics = extract_metrics(data)

section_header_sm(f"Report: {report['file']}")

# ── Metrics cards ─────────────────────────────────────
model_name = metrics.get("model_name", metrics.get("strategy", metrics.get("name", "N/A")))
status_badge(f"Model: {model_name}", "blue")
st.markdown("")

mcol1, mcol2, mcol3, mcol4 = st.columns(4)

# Profit
profit = metrics.get("total_profit", metrics.get("pnl", metrics.get("profit", 0)))
metric_card(mcol1, f"£{float(profit):+,.2f}", "Total P&L", up=float(profit) >= 0)

# ROI
roi = metrics.get("roi", metrics.get("roi_pct", metrics.get("roi_on_bankroll_pct", 0)))
metric_card(mcol2, f"{float(roi):+.2f}%", "ROI", up=float(roi) >= 0)

# Win Rate
win_rate = metrics.get("win_rate", metrics.get("win_rate_pct", 0))
metric_card(mcol3, f"{float(win_rate):.1f}%", "Win Rate", up=float(win_rate) > 50)

# Sharpe
sharpe = metrics.get("sharpe_ratio", 0)
metric_card(mcol4, f"{float(sharpe):.2f}", "Sharpe", up=float(sharpe) > 0)

# Secondary metrics
smcol1, smcol2, smcol3, smcol4 = st.columns(4)
total_bets = metrics.get("total_bets", metrics.get("n_bets", 0))
metric_card(smcol1, str(int(total_bets)), "Total Bets")
drawdown = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0))
metric_card(smcol2, f"{float(drawdown):.1f}%", "Max Drawdown", up=float(drawdown) < 20)
profit_factor = metrics.get("profit_factor", 0)
metric_card(smcol3, f"{float(profit_factor):.2f}", "Profit Factor", up=float(profit_factor) > 1)
avg_ev = metrics.get("avg_ev", metrics.get("average_ev", 0))
metric_card(smcol4, f"{float(avg_ev):+.2%}", "Avg EV", up=float(avg_ev) > 0)

# ── Gauges ────────────────────────────────────────────
st.markdown("### Performance Gauges")
gcol1, gcol2, gcol3 = st.columns(3)
gauge_chart(gcol1, "Sharpe Ratio", float(sharpe), target=1.0, lower_better=False)
gauge_chart(gcol2, "Win Rate", float(win_rate), target=50.0, unit="%")
gauge_chart(gcol3, "Max Drawdown", float(drawdown), target=20.0, unit="%", lower_better=True)


# ── Bankroll chart ─────────────────────────────────────
bankroll_history = metrics.get("bankroll_history", data.get("bankroll_history", []))
if not bankroll_history:
    # Try to find bankroll history in various nested formats
    for key in ["bankroll", "equity_curve", "pnl_history", "capital_curve"]:
        bh = data.get(key, [])
        if bh:
            bankroll_history = bh if isinstance(bh, list) else bh.get("values", [])
            break

if bankroll_history:
    st.markdown("### 📈 Bankroll Growth & Drawdown")
    initial_br = bankroll_history[0] if bankroll_history else 1000
    fig1 = bankroll_growth_chart(bankroll_history, initial=initial_br, height=350)
    st.plotly_chart(fig1, use_container_width=True)

    # Drawdown chart
    if len(bankroll_history) > 1:
        fig2 = drawdown_chart(bankroll_history, height=180)
        st.plotly_chart(fig2, use_container_width=True)


# ── Strategy Comparison ───────────────────────────────
strategies = []
if "stake_strategies" in data:
    strategies = data["stake_strategies"].get("results", [])
elif "risk_scenarios" in data:
    strategies = data["risk_scenarios"].get("results", [])
elif isinstance(data, list):
    strategies = data

if strategies:
    section_header("⚖️ Strategy Comparison", "⚖️")
    s_df = pd.DataFrame(strategies)

    plot_cols = [
        c for c in [
            "strategy", "name", "sharpe_ratio", "roi_pct", "roi",
            "win_rate_pct", "win_rate", "max_drawdown_pct",
            "total_profit", "total_bets", "profit_factor",
        ]
        if c in s_df.columns
    ]

    if plot_cols:
        name_col = "strategy" if "strategy" in s_df.columns else "name" if "name" in s_df.columns else plot_cols[0]
        sort_by = "sharpe_ratio" if "sharpe_ratio" in s_df.columns else "roi_pct" if "roi_pct" in s_df.columns else None
        if sort_by:
            s_df = s_df.sort_values(sort_by, ascending=False)

        display_df = s_df[plot_cols].copy()

        # Format percentage columns
        for col in ["roi_pct", "win_rate_pct", "max_drawdown_pct"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):+.2f}%" if "roi" in col else f"{float(x):.1f}%")

        for col in ["total_profit"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"£{float(x):+,.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Side-by-side bar charts
        chart_cols = [c for c in ["sharpe_ratio", "roi_pct", "total_profit", "win_rate_pct", "max_drawdown_pct"]
                      if c in s_df.columns]
        if chart_cols:
            selected_chart = st.selectbox("Compare strategies by:", chart_cols,
                                           format_func=lambda x: x.replace("_", " ").title())
            if selected_chart:
                chart_data = s_df[[name_col, selected_chart]].dropna()
                fig = comparison_bar_chart(
                    chart_data, x_col=name_col, y_col=selected_chart,
                    title=f"Strategy Comparison: {selected_chart.replace('_', ' ').title()}",
                    horizontal=True, height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Scatter: Sharpe vs Drawdown
        if "sharpe_ratio" in s_df.columns and "max_drawdown_pct" in s_df.columns:
            st.markdown("### 🎯 Risk-Return Tradeoff")
            fig = px.scatter(
                s_df, x="max_drawdown_pct", y="sharpe_ratio",
                size="total_bets" if "total_bets" in s_df.columns else None,
                color="sharpe_ratio",
                hover_name=name_col,
                color_continuous_scale="RdYlGn",
                title="Risk-Adjusted Return: Top-left = best",
                labels={"max_drawdown_pct": "Max Drawdown (%)", "sharpe_ratio": "Sharpe Ratio"},
            )
            fig.add_hline(y=1, line_dash="dash", line_color=Colors.WARNING, line_width=1,
                          annotation_text="Sharpe=1")
            fig.add_vline(x=20, line_dash="dash", line_color=Colors.DANGER, line_width=1,
                          annotation_text="DD=20%")
            fig.update_layout(**{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                                 "font": {"color": Colors.TEXT_PRIMARY}})
            st.plotly_chart(fig, use_container_width=True)


# ── Market breakdown ──────────────────────────────────
bets_per_market = metrics.get("bets_per_market", data.get("bets_per_market", {}))
profit_per_market = metrics.get("profit_per_market", data.get("profit_per_market", {}))

if bets_per_market:
    section_header("📊 Market Breakdown", "📊")
    mkt_df = pd.DataFrame({
        "Market": list(bets_per_market.keys()),
        "Bets": list(bets_per_market.values()),
        "Profit": [profit_per_market.get(m, 0) for m in bets_per_market.keys()],
    })
    mkt_df["ROI"] = (mkt_df["Profit"] / mkt_df["Bets"] * 100).round(2)
    mkt_df["P&L"] = mkt_df["Profit"].apply(lambda x: f"£{x:+,.2f}")

    st.dataframe(mkt_df[["Market", "Bets", "P&L", "ROI"]], use_container_width=True, hide_index=True)

    fig = px.bar(
        mkt_df, x="Market", y=["Bets", "Profit"],
        title="Performance by Market",
        barmode="group",
        color_discrete_map={"Bets": Colors.PRIMARY, "Profit": Colors.SUCCESS},
    )
    fig.update_layout(**{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                         "font": {"color": Colors.TEXT_PRIMARY}})
    st.plotly_chart(fig, use_container_width=True)


# ── Model leaderboard (from summary) ──────────────────
summary = extract_metrics(data)
if "models" in data and isinstance(data["models"], list):
    section_header("🏆 Model Leaderboard", "🏆")
    models_data = data["models"]
    lb_df = pd.DataFrame(models_data)

    rank_cols = [c for c in ["rank", "model_name", "sharpe_ratio", "roi_pct", "win_rate_pct",
                               "total_profit", "brier", "total_bets", "max_drawdown_pct"]
                  if c in lb_df.columns]
    if rank_cols:
        display_lb = lb_df[rank_cols].copy()
        for col in ["roi_pct", "win_rate_pct", "max_drawdown_pct"]:
            if col in display_lb.columns:
                display_lb[col] = display_lb[col].apply(lambda x: f"{float(x):+.2f}%" if "roi" in col else f"{float(x):.1f}%")
        if "total_profit" in display_lb.columns:
            display_lb["total_profit"] = display_lb["total_profit"].apply(lambda x: f"£{float(x):+,.2f}")
        st.dataframe(display_lb, use_container_width=True, hide_index=True)

        # Best model highlight
        if "sharpe_ratio" in lb_df.columns:
            best_idx = lb_df["sharpe_ratio"].idxmax()
            best_row = lb_df.loc[best_idx]
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#141824 0%,#1a1f2e 100%);'
                f'border:1px solid #4fc3f7;border-radius:12px;padding:1rem 1.5rem;margin-top:0.5rem">'
                f'<div style="color:#4fc3f7;font-weight:600;font-size:0.85rem">🏆 BEST OVERALL</div>'
                f'<div style="color:#e0e0e0;font-size:1.1rem;font-weight:700;margin-top:0.2rem">'
                f'{best_row.get("model_name", "?")}</div>'
                f'<div style="color:#6b7280;font-size:0.8rem;margin-top:0.3rem">'
                f'Sharpe: {best_row.get("sharpe_ratio", 0):.2f} · '
                f'ROI: {best_row.get("roi_pct", 0):+.2f}% · '
                f'Win Rate: {best_row.get("win_rate_pct", 0):.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Download ───────────────────────────────────────────
st.markdown("---")
st.download_button(
    "⬇ Download Report JSON",
    data=json.dumps(data, indent=2, default=str).encode("utf-8"),
    file_name=f"betting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
)

render_footer()
