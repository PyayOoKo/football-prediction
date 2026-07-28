"""
Portfolio Tracker — Live bankroll monitoring across the 4-strategy portfolio.

Strategies
----------
1. **F1 Over 2.5** — France Ligue 1, Over 2.5 value bets (30% allocation)
2. **E0 Over 2.5** — England Premier League, Over 2.5 value bets (25%)
3. **I1 Under 2.5** — Italy Serie A, Under 2.5 value bets (25%)
4. **D1 Over 2.5** — Germany Bundesliga, Over 2.5 value bets (20%)

Shows combined bankroll, per-strategy P&L, drawdown, bet history, and
upcoming value bet recommendations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dashboard.components import (
    init_theme,
    sidebar_theme_radio,
    render_custom_css,
    render_hero,
    render_footer,
    section_header,
    metric_card,
    status_badge,
    gauge_chart,
    info_row,
    Colors,
    get_plotly_layout,
)
from dashboard.portfolio_data import (
    load_portfolio_state,
    run_full_portfolio_simulation,
    get_recent_value_bets,
    STRATEGIES,
    PortfolioState,
)

st.set_page_config(page_title="Portfolio Tracker", page_icon="💰", layout="wide")

# ── Theme initialisation ───────────────────────────────
init_theme()

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    sidebar_theme_radio()
    st.markdown("---")
    st.markdown("### 💰 Portfolio")
    st.markdown(
        "<div style='color:var(--text-secondary,#6b7280);font-size:0.8rem'>"
        "4-strategy value betting portfolio using per-league trained models "
        "with Kelly staking (25% fraction, min EV 5%).</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

render_custom_css()

# ── Session state ─────────────────────────────────────
if "portfolio_state" not in st.session_state:
    st.session_state.portfolio_state = None
if "portfolio_loaded" not in st.session_state:
    st.session_state.portfolio_loaded = False


# ── Data loading ──────────────────────────────────────
@st.cache_data(ttl=300)
def load_portfolio() -> PortfolioState:
    """Load portfolio state (cached for 5 min)."""
    return run_full_portfolio_simulation()


def get_state() -> PortfolioState:
    if st.session_state.portfolio_state is None:
        st.session_state.portfolio_state = load_portfolio()
        st.session_state.portfolio_loaded = True
    return st.session_state.portfolio_state


# ── Hero ───────────────────────────────────────────────
render_hero(
    title="💰 Portfolio Tracker",
    subtitle="Live multi-strategy bankroll monitoring with per-league model blends, "
             "Kelly staking, and real-time value bet recommendations. "
             "Simulation uses DC+Elo OU blend (approximation of full 5-model blend).",
    badges=[
        ("4 strategies", "📊"),
        ("Kelly 25%", "🎯"),
        ("Auto-refresh 5min", "🔄"),
    ],
)

state = get_state()

# ══════════════════════════════════════════════════════════
#  PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════

section_header("📊 Portfolio Overview", "📊")

total_br = state.total_bankroll
total_pnl = state.total_profit
total_ret = state.total_return_pct
total_bets = state.n_bets_total
total_pending = state.n_pending

col1, col2, col3, col4, col5 = st.columns(5)

metric_card(col1, f"£{total_br:,.2f}", "Total Bankroll",
            delta=f"£{total_pnl:+,.2f}", up=total_pnl > 0)
metric_card(col2, f"{total_ret:+.2f}%", "Total Return",
            delta="on £10,000 initial", up=total_ret > 0)
metric_card(col3, str(total_bets), "Total Bets Placed",
            delta=f"{total_pending} pending", up=True)
roi_val = state.portfolio_yield
metric_card(col4, f"{roi_val:+.2f}%", "Portfolio Yield",
            delta="profit / staked", up=roi_val > 0)
metric_card(col5, f"£{state.total_staked:,.2f}", "Total Staked",
            delta=f"across {len(state.strategies)} strategies", up=True)

# ── Strategy Allocation ───────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
strat_cols = st.columns(4)
for i, (name, ss) in enumerate(state.strategies.items()):
    cfg = STRATEGIES.get(name, {})
    color = cfg.get("color", Colors.PRIMARY)
    with strat_cols[i]:
        profit = ss.total_profit
        yield_val = ss.yield_pct
        st.markdown(
            f'<div style="background:linear-gradient(135deg,{color}10 0%,{color}05 100%);'
            f'border:1px solid {color}30;border-radius:14px;padding:1rem 1.2rem;'
            f'text-align:center;height:100%">'
            f'<div style="color:{color};font-weight:700;font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.05em">{name}</div>'
            f'<div style="font-size:1.3rem;font-weight:800;color:var(--text-primary,#e0e0e0);'
            f'margin-top:0.3rem">£{ss.bankroll:,.2f}</div>'
            f'<div style="font-size:0.75rem;color:var(--text-secondary,#6b7280);margin-top:0.2rem">'
            f'<span style="color:{"var(--success,#4caf50)" if profit >= 0 else "var(--danger,#f44336)"}">'
            f'{"+" if profit >= 0 else ""}£{profit:+,.2f}</span> · '
            f'{yield_val:+.1f}% yield</div>'
            f'<div style="font-size:0.7rem;color:var(--text-muted,#555);margin-top:0.1rem">'
            f'{ss.n_bets} bets · {ss.n_won}W/{ss.n_lost}L · '
            f'DD: {ss.max_drawdown_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════
#  BANKROLL GROWTH CHART
# ══════════════════════════════════════════════════════════

section_header("📈 Bankroll Growth", "📈")

# Build combined bankroll history from all strategies
all_bets = []
for name, ss in state.strategies.items():
    cfg = STRATEGIES.get(name, {})
    for b in ss.bets:
        all_bets.append({
            "date": b.date,
            "strategy": name,
            "profit": b.profit if b.won is not None else 0,
            "color": cfg.get("color", Colors.PRIMARY),
            "league": cfg.get("league", ""),
        })

all_bets.sort(key=lambda x: x["date"])

# Combined bankroll timeline
if all_bets:
    initial = state.initial_bankroll
    timeline = []
    br = initial
    timeline.append({"date": all_bets[0]["date"], "bankroll": br, "type": "Combined"})
    for b in all_bets:
        br += b["profit"]
        timeline.append({"date": b["date"], "bankroll": round(br, 2), "type": "Combined"})
    tl_df = pd.DataFrame(timeline)

    # Per-strategy timelines
    per_strat = []
    for name, ss in state.strategies.items():
        cfg = STRATEGIES.get(name, {})
        s_br = ss.initial_capital
        per_strat.append({"date": ss.bets[0].date if ss.bets else "2000-01-01",
                          "bankroll": s_br, "strategy": name})
        for b in ss.bets:
            if b.won is not None:
                s_br += b.profit
                per_strat.append({"date": b.date, "bankroll": round(s_br, 2),
                                  "strategy": name})
    ps_df = pd.DataFrame(per_strat)

    tab1, tab2 = st.tabs(["📊 Combined View", "📈 Per-Strategy View"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tl_df["date"], y=tl_df["bankroll"],
            mode="lines", name="Portfolio",
            line=dict(color=Colors.PRIMARY, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba(79, 195, 247, 0.08)",
            hovertemplate="%{x}<br>£%{y:,.2f}<extra></extra>",
        ))
        # Initial bankroll line
        fig.add_hline(y=initial, line_dash="dash", line_color=Colors.TEXT_MUTED,
                      line_width=1,
                      annotation_text=f"Initial: £{initial:,.0f}",
                      annotation_font=dict(color=Colors.TEXT_SECONDARY, size=10))
        # Profit/loss shading
        max_br = max(b["bankroll"] for b in timeline)
        min_br = min(b["bankroll"] for b in timeline)
        fig.add_hrect(y0=initial, y1=max_br, fillcolor=Colors.SUCCESS,
                      opacity=0.03, layer="below", line_width=0)
        fig.add_hrect(y0=min_br, y1=initial, fillcolor=Colors.DANGER,
                      opacity=0.03, layer="below", line_width=0)
        fig.update_layout(
            title=dict(text="Portfolio Bankroll Over Time",
                       font=dict(color=Colors.TEXT_PRIMARY, size=14)),
            xaxis=dict(title="Date", gridcolor=Colors.GRID_COLOR,
                       tickfont=dict(size=9)),
            yaxis=dict(title="Bankroll (£)", gridcolor=Colors.GRID_COLOR,
                       tickfont=dict(size=9)),
            height=400, hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=Colors.TEXT_PRIMARY),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown chart
        peak = np.maximum.accumulate(tl_df["bankroll"].values)
        dd = (tl_df["bankroll"].values - peak) / peak * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=tl_df["date"], y=dd, mode="lines", name="Drawdown",
            line=dict(color=Colors.DANGER, width=1.5),
            fill="tozeroy", fillcolor="rgba(244, 67, 54, 0.08)",
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        ))
        fig_dd.update_layout(
            title=dict(text="Drawdown", font=dict(color=Colors.TEXT_PRIMARY, size=12)),
            yaxis=dict(title="Drawdown (%)", gridcolor=Colors.GRID_COLOR,
                       tickfont=dict(size=9), zerolinecolor=Colors.ZERO_LINE),
            xaxis=dict(gridcolor=Colors.GRID_COLOR, tickfont=dict(size=9)),
            height=180, hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=Colors.TEXT_PRIMARY),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab2:
        if not ps_df.empty:
            fig2 = go.Figure()
            for name in state.strategies:
                s_df = ps_df[ps_df["strategy"] == name].copy()
                cfg = STRATEGIES.get(name, {})
                color = cfg.get("color", Colors.PRIMARY)
                if not s_df.empty:
                    fig2.add_trace(go.Scatter(
                        x=s_df["date"], y=s_df["bankroll"],
                        mode="lines", name=name,
                        line=dict(color=color, width=1.8),
                        hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
                    ))
            fig2.update_layout(
                title=dict(text="Per-Strategy Bankroll",
                           font=dict(color=Colors.TEXT_PRIMARY, size=14)),
                xaxis=dict(title="Date", gridcolor=Colors.GRID_COLOR,
                           tickfont=dict(size=9)),
                yaxis=dict(title="Bankroll (£)", gridcolor=Colors.GRID_COLOR,
                           tickfont=dict(size=9)),
                height=400, hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=Colors.TEXT_PRIMARY),
                legend=dict(font=dict(color=Colors.TEXT_PRIMARY), orientation="h",
                           y=-0.2),
            )
            st.plotly_chart(fig2, use_container_width=True)

else:
    info_row("No bet history yet. Run the portfolio simulation to populate data.")
    st.markdown(
        '<div style="background:#141824;border:1px solid #1e2235;'
        'border-radius:12px;padding:2rem;text-align:center">'
        '<div style="font-size:3rem;margin-bottom:0.5rem">📭</div>'
        '<div style="color:#9ca3af">No bets placed yet.</div>'
        '<div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">'
        'Click "Refresh Portfolio" below to run the full simulation.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════
#  PER-STRATEGY DETAIL
# ══════════════════════════════════════════════════════════

section_header("🔍 Per-Strategy Analysis", "🔍")

strategy_tabs = st.tabs(list(state.strategies.keys()))

for i, (name, ss) in enumerate(state.strategies.items()):
    cfg = STRATEGIES.get(name, {})
    color = cfg.get("color", Colors.PRIMARY)
    with strategy_tabs[i]:
        # Metrics row
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        profit = ss.total_profit
        metric_card(mc1, f"£{ss.bankroll:,.2f}", "Bankroll",
                    delta=f"£{profit:+,.2f}", up=profit >= 0)
        metric_card(mc2, f"{ss.yield_pct:+.2f}%", "Yield (ROI)",
                    up=ss.yield_pct > 0)
        metric_card(mc3, f"{ss.win_rate*100:.1f}%", "Win Rate",
                    delta=f"{ss.n_won}W/{ss.n_lost}L", up=ss.win_rate > 0.5)
        metric_card(mc4, f"{ss.max_drawdown_pct:.1f}%", "Max Drawdown",
                    up=ss.max_drawdown_pct < 20)
        metric_card(mc5, str(ss.n_bets), "Total Bets",
                    delta=f"£{ss.total_staked:,.2f} staked", up=True)

        # Gauges
        gcol1, gcol2, gcol3, gcol4 = st.columns(4)
        gauge_chart(gcol1, "Bankroll vs Initial", ss.bankroll,
                    target=ss.initial_capital, unit="", lower_better=False,
                    height=180)
        gauge_chart(gcol2, "Yield", ss.yield_pct,
                    target=10.0, unit="%", lower_better=False, height=180)
        gauge_chart(gcol3, "Max Drawdown", ss.max_drawdown_pct,
                    target=20.0, unit="%", lower_better=True, height=180)
        gauge_chart(gcol4, "Win Rate", ss.win_rate * 100,
                    target=50.0, unit="%", lower_better=False, height=180)

        # Bets table
        if ss.bets:
            st.markdown(f"### 📋 Bet History ({len(ss.bets)} bets)")
            bet_rows = []
            for b in reversed(ss.bets[-50:]):  # Last 50
                result_icon = "✅" if b.won else ("❌" if b.won is False else "⏳")
                bet_rows.append({
                    "Date": b.date,
                    "Home": b.home,
                    "Away": b.away,
                    "Market": b.market,
                    "Odds": b.odds,
                    "Model Prob": f"{b.model_prob:.1%}",
                    "EV": f"{b.ev:.1%}",
                    "Stake": f"£{b.stake:.2f}",
                    "Result": result_icon,
                    "P&L": f"{b.profit:+.2f}",
                })
            bdf = pd.DataFrame(bet_rows)

            # Color the P&L column
            def _color_pnl(val):
                if val.startswith("+"):
                    return f"color: var(--success, #4caf50); font-weight: 600"
                elif val.startswith("-"):
                    return f"color: var(--danger, #f44336); font-weight: 600"
                return ""

            styled = bdf.style.map(
                _color_pnl, subset=["P&L"]
            ).map(
                lambda v: "color: #4fc3f7; font-weight: 600",
                subset=["EV"]
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            info_row("No bets placed for this strategy yet.")

# ══════════════════════════════════════════════════════════
#  RISK METRICS
# ══════════════════════════════════════════════════════════

section_header("🛡️ Risk Dashboard", "🛡️")

rc1, rc2, rc3, rc4 = st.columns(4)

# Combined max drawdown
dds = [ss.max_drawdown_pct for ss in state.strategies.values()]
combined_dd = max(dds) if dds else 0
metric_card(rc1, f"{combined_dd:.1f}%", "Worst Strategy DD",
            delta="max individual drawdown", up=combined_dd < 20)

# Sharpe-like: return / dd
risk_adjusted = total_ret / max(combined_dd, 0.1) if combined_dd > 0 else 0
metric_card(rc2, f"{risk_adjusted:.2f}", "Return / Risk Ratio",
            delta="higher = better risk-adjusted", up=risk_adjusted > 0.5)

# Capital utilisation
util = (state.total_staked / state.initial_bankroll) * 100 if state.initial_bankroll > 0 else 0
metric_card(rc3, f"{util:.1f}%", "Capital Utilisation",
            delta="% of bankroll turned over", up=util > 50)

# Winning strategies
winning = sum(1 for ss in state.strategies.values() if ss.total_profit > 0)
metric_card(rc4, f"{winning}/{len(state.strategies)}", "Profitable Strategies",
            delta=f"{winning * 100 // max(len(state.strategies), 1)}% win rate", up=winning > 2)

# Risk overview table
risk_data = []
for name, ss in state.strategies.items():
    cfg = STRATEGIES.get(name, {})
    risk_data.append({
        "Strategy": name,
        "Capital": f"£{ss.initial_capital:,.2f}",
        "Current": f"£{ss.bankroll:,.2f}",
        "P&L": f"£{ss.total_profit:+,.2f}",
        "Yield": f"{ss.yield_pct:+.1f}%",
        "Bets": ss.n_bets,
        "Win Rate": f"{ss.win_rate*100:.0f}%",
        "Max DD": f"{ss.max_drawdown_pct:.1f}%",
    })
risk_df = pd.DataFrame(risk_data)

st.markdown("### Strategy Risk Comparison")
st.dataframe(risk_df, use_container_width=True, hide_index=True)

# ── Risk gauge ──
st.markdown("### Portfolio Health")
health_score = 100
# Deductions
if combined_dd > 30:
    health_score -= 30
elif combined_dd > 20:
    health_score -= 15
elif combined_dd > 10:
    health_score -= 5
if total_pnl < 0:
    health_score -= 20
if util > 100:
    health_score -= 10
if winning < len(state.strategies) / 2:
    health_score -= 15
health_score = max(health_score, 0)

health_color = Colors.SUCCESS if health_score >= 70 else (Colors.WARNING if health_score >= 40 else Colors.DANGER)
health_label = "Healthy" if health_score >= 70 else ("Moderate" if health_score >= 40 else "At Risk")

gcol1, gcol2 = st.columns([1, 3])
with gcol1:
    fig_h = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        number={"suffix": "/100", "font": {"color": health_color, "size": 24}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": Colors.TEXT_SECONDARY, "size": 9}},
            "bar": {"color": health_color, "thickness": 0.4},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 30], "color": "rgba(244, 67, 54, 0.12)"},
                {"range": [30, 60], "color": "rgba(255, 193, 7, 0.12)"},
                {"range": [60, 100], "color": "rgba(76, 175, 80, 0.12)"},
            ],
        }
    ))
    fig_h.update_layout(
        height=200, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=health_color, size=14),
    )
    st.plotly_chart(fig_h, use_container_width=True)

with gcol2:
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{health_color}15 0%,{health_color}08 100%);'
        f'border:1px solid {health_color}30;border-radius:14px;padding:1.2rem 1.5rem;'
        f'height:100%;display:flex;flex-direction:column;justify-content:center">'
        f'<div style="font-size:1.3rem;font-weight:700;color:{health_color}">'
        f'Portfolio Status: {health_label}</div>'
        f'<div style="font-size:0.85rem;color:var(--text-secondary,#6b7280);margin-top:0.3rem">'
        f'{"Portfolio is performing well with controlled drawdowns." if health_score >= 70 else '
        f"Monitor risk metrics closely and consider reducing position sizes." if health_score >= 40 else '
        f"High risk detected — consider pausing or restructuring the portfolio."}'
        f'</div>'
        f'<div style="font-size:0.75rem;color:var(--text-muted,#555);margin-top:0.5rem">'
        f'Combined DD: {combined_dd:.1f}% · '
        f'Profitable legs: {winning}/{len(state.strategies)} · '
        f'Total return: {total_ret:+.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
#  UPCOMING VALUE BETS
# ══════════════════════════════════════════════════════════

section_header("🔮 Upcoming Value Bets", "🔮")

st.markdown(
    '<p style="color:var(--text-secondary,#6b7280);font-size:0.9rem">'
    'Predicted probabilities for upcoming matches. Actual value bets require '
    'live odds (run <code>today_league_value_bets.py</code> for live odds).</p>',
    unsafe_allow_html=True,
)

# Check each league for upcoming matches
upcoming_found = False
for name, cfg in STRATEGIES.items():
    direction = cfg["direction"]
    bets_data = get_recent_value_bets(cfg["league"], direction, max_results=5)
    if bets_data:
        upcoming_found = True
        color = cfg.get("color", Colors.PRIMARY)
        st.markdown(
            f'<div style="border-left:3px solid {color};padding-left:1rem;margin-top:0.5rem">'
            f'<span style="font-weight:600;color:{color}">{name}</span></div>',
            unsafe_allow_html=True,
        )
        bdf = pd.DataFrame(bets_data)
        # Format probabilities as percentages
        bdf["model_prob"] = bdf["model_prob"].apply(lambda p: f"{p:.1%}")
        st.dataframe(
            bdf[["date", "home", "away", "market", "model_prob"]],
            use_container_width=True,
            hide_index=True,
        )

if not upcoming_found:
    st.info(
        '<div style="background:#141824;border:1px solid #1e2235;border-radius:12px;'
        'padding:1.5rem;text-align:center">'
        '<div style="font-size:2rem;margin-bottom:0.3rem">🔮</div>'
        '<div style="color:#9ca3af;font-size:1rem">No upcoming matches found.</div>'
        '<div style="color:#6b7280;font-size:0.85rem">Run data collection to fetch fixtures.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
#  REFRESH CONTROLS
# ══════════════════════════════════════════════════════════

st.markdown("---")
ref_col1, ref_col2 = st.columns([1, 3])

if ref_col1.button("🔄 Refresh Portfolio", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.session_state.portfolio_state = load_portfolio()
    st.rerun()

with ref_col2:
    last_updated = state.last_updated[:19] if state.last_updated else "Never"
    st.markdown(
        f'<div style="color:var(--text-muted,#555);font-size:0.75rem;padding-top:0.5rem">'
        f'Last updated: {last_updated} · Data cached for 5 minutes</div>',
        unsafe_allow_html=True,
    )

render_footer()
