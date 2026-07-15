"""
Betting Results Dashboard — P&L charts, win rates, ROI analysis, streaks.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Betting Results", page_icon="💰", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-positive { color: #4caf50; }
    .metric-negative { color: #f44336; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.8rem; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 💰 Betting Results")
st.markdown("Analyse betting P&L, win rates, ROI, and streak performance.")


# ── Load backtest data ────────────────────────────────
@st.cache_data(ttl=60)
def load_backtest_data() -> list[dict]:
    """Load all backtest result JSON files."""
    results = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return results
    for pattern in ["backtest_*.json", "backtest_summary_*.json",
                     "bankroll_management_*.json", "bankroll_optimization_*.json",
                     "metrics_*.json", "betting_report*.json"]:
        for f in reports_dir.glob(pattern):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                results.append({"file": f.name, "data": data})
            except Exception:
                pass
    return results


backtests = load_backtest_data()

if not backtests:
    st.info("No backtest results found. Run backtest scripts first.")
    st.stop()


# ── Overview stats ────────────────────────────────────
total_files = len(backtests)
st.markdown(f"**{total_files} backtest report files** loaded.")


# ── Report selector ────────────────────────────────────
st.markdown("### Select Backtest Report")
file_names = [b["file"] for b in backtests]
selected_file = st.selectbox("Choose a report:", file_names)
report = next(b for b in backtests if b["file"] == selected_file)
data = report["data"]


# ── Extract metrics from various formats ───────────────
def extract_metrics(d: dict) -> dict:
    """Normalise metrics from various backtest report formats."""
    m = {}

    # BacktestEngine format
    if "metrics" in d:
        m = d["metrics"]
    # Bankroll management format (list of strategies)
    elif "stake_strategies" in d:
        strategies = d["stake_strategies"].get("results", [])
        if strategies:
            best = max(strategies, key=lambda s: s.get("sharpe_ratio", 0) if s.get("total_bets", 0) > 0 else -999)
            m = best
            # Rename keys
            m["model_name"] = best.get("strategy", best.get("name", "Unknown"))
    # Direct format
    elif "results" in d and isinstance(d["results"], list):
        results = d["results"]
        if results:
            m = results[0]
    else:
        m = d

    return m


metrics = extract_metrics(data)

# ── Metrics display ────────────────────────────────────
st.markdown(f"### Report: {report['file']}")

# Try to determine model/strategy name
model_name = metrics.get("model_name", metrics.get("strategy", metrics.get("name", "N/A")))

col1, col2, col3, col4 = st.columns(4)

# Total Profit
profit = metrics.get("total_profit", metrics.get("pnl", metrics.get("profit", 0)))
profit_color = "#4caf50" if profit >= 0 else "#f44336"
with col1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{profit_color}">'
        f'{"+" if profit >= 0 else ""}£{float(profit):,.2f}</div>'
        f'<div class="metric-label">Total P&L</div></div>',
        unsafe_allow_html=True,
    )

# ROI
roi = metrics.get("roi", metrics.get("roi_pct", metrics.get("roi_on_bankroll_pct", 0)))
roi_color = "#4caf50" if roi >= 0 else "#f44336"
with col2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{roi_color}">{float(roi):+.2f}%</div>'
        f'<div class="metric-label">ROI</div></div>',
        unsafe_allow_html=True,
    )

# Win Rate
win_rate = metrics.get("win_rate", metrics.get("win_rate_pct", 0))
with col3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{float(win_rate):.1f}%</div>'
        f'<div class="metric-label">Win Rate</div></div>',
        unsafe_allow_html=True,
    )

# Sharpe Ratio
sharpe = metrics.get("sharpe_ratio", 0)
sharpe_color = "#4caf50" if sharpe >= 1 else "#ffc107" if sharpe >= 0 else "#f44336"
with col4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{sharpe_color}">{float(sharpe):.2f}</div>'
        f'<div class="metric-label">Sharpe Ratio</div></div>',
        unsafe_allow_html=True,
    )

# More detail
col5, col6, col7, col8 = st.columns(4)
with col5:
    total_bets = metrics.get("total_bets", metrics.get("n_bets", 0))
    st.metric("Total Bets", int(total_bets))
with col6:
    drawdown = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0))
    st.metric("Max Drawdown", f"{float(drawdown):.1f}%")
with col7:
    profit_factor = metrics.get("profit_factor", 0)
    st.metric("Profit Factor", f"{float(profit_factor):.2f}")
with col8:
    avg_ev = metrics.get("avg_ev", metrics.get("average_ev", 0))
    st.metric("Avg EV", f"{float(avg_ev):+.2%}")


# ── Bankroll chart ─────────────────────────────────────
bankroll_history = metrics.get("bankroll_history", data.get("bankroll_history", []))
if bankroll_history:
    st.markdown("### 📈 Bankroll Growth")
    br_df = pd.DataFrame({
        "Bet": list(range(len(bankroll_history))),
        "Bankroll": bankroll_history,
    })
    fig = px.line(
        br_df, x="Bet", y="Bankroll",
        title="Bankroll Over Time",
        color_discrete_sequence=["#7c3aed"],
    )
    fig.add_hrect(
        y0=bankroll_history[0], y1=max(bankroll_history),
        fillcolor="green", opacity=0.03, layer="below", line_width=0,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Best/worst performers ──────────────────────────────
# If this is a strategy comparison report
strategies = []
if "stake_strategies" in data:
    strategies = data["stake_strategies"].get("results", [])
elif "risk_scenarios" in data:
    strategies = data["risk_scenarios"].get("results", [])
elif isinstance(data, list):
    strategies = data

if strategies:
    st.markdown("### Strategy Comparison")
    s_df = pd.DataFrame(strategies)
    # Select relevant columns
    plot_cols = [c for c in ["strategy", "name", "sharpe_ratio", "roi_pct", "roi",
                              "win_rate_pct", "win_rate", "max_drawdown_pct", "total_profit",
                              "total_bets"]
                 if c in s_df.columns]
    if plot_cols:
        name_col = "strategy" if "strategy" in s_df.columns else "name" if "name" in s_df.columns else plot_cols[0]
        # Sort by Sharpe
        sort_by = "sharpe_ratio" if "sharpe_ratio" in s_df.columns else "roi_pct" if "roi_pct" in s_df.columns else None
        if sort_by:
            s_df = s_df.sort_values(sort_by, ascending=False)

        st.dataframe(s_df[plot_cols], use_container_width=True, hide_index=True)


# ── Market breakdown ──────────────────────────────────
bets_per_market = metrics.get("bets_per_market", data.get("bets_per_market", {}))
profit_per_market = metrics.get("profit_per_market", data.get("profit_per_market", {}))

if bets_per_market:
    st.markdown("### Market Breakdown")
    mkt_df = pd.DataFrame({
        "Market": list(bets_per_market.keys()),
        "Bets": list(bets_per_market.values()),
        "Profit": [profit_per_market.get(m, 0) for m in bets_per_market.keys()],
    })
    fig = px.bar(
        mkt_df, x="Market", y=["Bets", "Profit"],
        title="Performance by Market",
        barmode="group",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Download ───────────────────────────────────────────
st.download_button(
    "⬇ Download Report JSON",
    data=json.dumps(data, indent=2, default=str).encode("utf-8"),
    file_name=f"betting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
)
