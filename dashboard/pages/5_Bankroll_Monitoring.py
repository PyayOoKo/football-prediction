"""
Bankroll Monitoring Dashboard — track bankroll growth, drawdown, risk metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Bankroll Monitoring", page_icon="🏦", layout="wide")

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
    .metric-neutral { color: #ffc107; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.8rem; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏦 Bankroll Monitoring")
st.markdown("Monitor bankroll growth, drawdown, risk metrics, and optimal stake sizing.")


# ── Load reports ──────────────────────────────────────
@st.cache_data(ttl=60)
def load_reports() -> list[dict]:
    """Load bankroll-related report files."""
    results = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return results
    for pattern in ["bankroll_management_*.json", "bankroll_optimization_*.json",
                     "bankroll_report*.json", "bankroll_*.json"]:
        for f in sorted(reports_dir.glob(pattern), reverse=True):
            try:
                with open(f) as fh:
                    results.append({"file": f.name, "data": json.load(fh)})
            except Exception:
                pass
    return results


reports = load_reports()

if not reports:
    st.info("No bankroll reports found. Run backtest/optimize scripts first.")
    st.stop()


# ── Main view ──────────────────────────────────────────
st.markdown(f"**{len(reports)} bankroll report files** loaded.")

# Try to extract best strategy from optimization report
best_strategy = None
best_sharpe = -999

for r in reports:
    data = r["data"]

    # Bankroll optimization format
    if "best_strategy" in data:
        bs = data["best_strategy"]
        sharpe = bs.get("sharpe_ratio", -999)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_strategy = bs
        break  # Usually only one optimization file

    # Bankroll management format
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

if best_strategy:
    st.markdown("### 🏆 Optimal Strategy")
    c1, c2, c3, c4 = st.columns(4)

    strategy_name = (
        best_strategy.get("strategy", best_strategy.get("name",
            best_strategy.get("model_name", "Optimal")))
    )

    c1.markdown(
        f'<div class="metric-card"><div class="metric-value" '
        f'style="font-size:1rem;color:#7c3aed">{strategy_name}</div>'
        f'<div class="metric-label">Strategy</div></div>',
        unsafe_allow_html=True,
    )

    for m, col in [("sharpe_ratio", c2), ("total_profit", c3), ("max_drawdown_pct", c4)]:
        val = best_strategy.get(m, 0)
        if m == "sharpe_ratio":
            col.markdown(
                f'<div class="metric-card"><div class="metric-value '
                f'{"metric-positive" if val>=1 else "metric-neutral"}">{float(val):.2f}</div>'
                f'<div class="metric-label">Sharpe Ratio</div></div>',
                unsafe_allow_html=True,
            )
        elif m == "total_profit":
            col.markdown(
                f'<div class="metric-card"><div class="metric-value '
                f'{"metric-positive" if val>=0 else "metric-negative"}">£{float(val):+,.2f}</div>'
                f'<div class="metric-label">Total P&L</div></div>',
                unsafe_allow_html=True,
            )
        elif m == "max_drawdown_pct":
            col.markdown(
                f'<div class="metric-card"><div class="metric-value '
                f'{"metric-positive" if val<10 else "metric-neutral"}">{float(val):.1f}%</div>'
                f'<div class="metric-label">Max Drawdown</div></div>',
                unsafe_allow_html=True,
            )

    # More metrics
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Win Rate", f"{float(best_strategy.get('win_rate', best_strategy.get('win_rate_pct', 0))):.1f}%")
    c6.metric("Total Bets", int(best_strategy.get("total_bets", 0)))
    c7.metric("ROI", f"{float(best_strategy.get('roi', best_strategy.get('roi_pct', 0))):+.2f}%")
    c8.metric("Profit Factor", f"{float(best_strategy.get('profit_factor', 0)):.2f}")


# ── Report selector ────────────────────────────────────
st.markdown("---")
st.markdown("### Select Report")
report_names = [r["file"] for r in reports]
sel_file = st.selectbox("Choose a report:", report_names)
sel = next(r for r in reports if r["file"] == sel_file)
data = sel["data"]


# ── Strategy comparison ────────────────────────────────
strategies = []
for section in ["stake_strategies", "risk_scenarios"]:
    if section in data:
        strategies = data[section].get("results", data.get(section, []))

if not strategies and isinstance(data, dict):
    # Check for other list structures
    for key in ["results", "strategies", "scenarios"]:
        items = data.get(key, [])
        if isinstance(items, list):
            strategies = items
            break

if strategies:
    s_df = pd.DataFrame(strategies)

    # Filter to only strategies with bets
    if "total_bets" in s_df.columns:
        s_df = s_df[s_df["total_bets"] > 0]

    if len(s_df) > 0:
        st.markdown(f"### Strategy Comparison ({len(s_df)} strategies)")

        # Sort by Sharpe
        sort_col = "sharpe_ratio" if "sharpe_ratio" in s_df.columns else None
        if sort_col:
            s_df = s_df.sort_values(sort_col, ascending=False)

        # Select columns to show
        cols = [c for c in ["strategy", "name", "category", "sharpe_ratio", "roi_pct", "roi",
                              "total_profit", "max_drawdown_pct", "win_rate_pct", "win_rate",
                              "total_bets", "profit_factor"]
                 if c in s_df.columns]
        display_df = s_df[cols] if cols else s_df

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Scatter plot: Sharpe vs Drawdown ─────────────
        if "sharpe_ratio" in s_df.columns and "max_drawdown_pct" in s_df.columns:
            st.markdown("### Risk-Adjusted Return (Sharpe vs Drawdown)")
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
            fig.add_hline(y=1, line_dash="dash", line_color="#ffc107", annotation_text="Sharpe=1")
            fig.add_vline(x=20, line_dash="dash", line_color="#f44336", annotation_text="DD=20%")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    # Show raw report content
    with st.expander("📄 Raw Report Data"):
        st.json(data)


# ── Bankroll sim (what-if) ─────────────────────────────
st.markdown("---")
st.markdown("### 🔬 What-If Bankroll Simulator")
st.markdown("Simulate a bankroll with custom starting balance and stake strategy.")

sim_col1, sim_col2, sim_col3 = st.columns(3)
initial_br = sim_col1.number_input("Initial Bankroll (£)", min_value=100, value=1000, step=100)
stake_pct = sim_col2.slider("Stake (% of bankroll)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
n_bets_sim = sim_col3.number_input("Number of Bets", min_value=10, max_value=10000, value=500, step=100)

win_rate_sim = st.slider("Assumed Win Rate (%)", min_value=30, max_value=70, value=52, step=1)
avg_odds_sim = st.slider("Average Odds", min_value=1.5, max_value=10.0, value=2.0, step=0.1)

if st.button("▶️ Simulate", type="primary"):
    import numpy as np
    rng = np.random.default_rng()  # Random seed each click for varied simulation
    bankroll = initial_br
    history = [bankroll]
    for _ in range(n_bets_sim):
        stake = bankroll * (stake_pct / 100)
        stake = min(stake, bankroll * 0.5)  # Safety cap
        won = rng.random() < (win_rate_sim / 100)
        if won:
            profit = stake * (avg_odds_sim - 1)
        else:
            profit = -stake
        bankroll += profit
        history.append(max(bankroll, 0))

    sim_df = pd.DataFrame({"Bet": range(len(history)), "Bankroll": history})

    # Metrics
    total_bets_sim = n_bets_sim
    wins_sim = int(n_bets_sim * (win_rate_sim / 100))
    final_br = history[-1]
    total_pnl = final_br - initial_br
    roi_sim = (total_pnl / initial_br) * 100

    peak = max(history)
    dd = max((peak - v) / peak * 100 for v in history) if peak > 0 else 0

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Final Bankroll", f"£{final_br:,.2f}",
               delta=f"£{total_pnl:+,.2f}" if total_pnl != 0 else None)
    sc2.metric("ROI", f"{roi_sim:+.1f}%")
    sc3.metric("Max Drawdown", f"{dd:.1f}%")
    sc4.metric("Est. Bets", f"{total_bets_sim:,}")

    fig = px.line(
        sim_df, x="Bet", y="Bankroll",
        title=f"Simulated Bankroll: {stake_pct}% stake, {win_rate_sim}% WR, {avg_odds_sim} odds",
        color_discrete_sequence=["#7c3aed"],
    )
    fig.add_hline(y=initial_br, line_dash="dash", line_color="#666",
                  annotation_text=f"Initial: £{initial_br:,.0f}")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
    )
    st.plotly_chart(fig, use_container_width=True)
