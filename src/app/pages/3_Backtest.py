"""
Backtest Page — display historical performance metrics, ROI, drawdown, and
profit/loss charts from the backtesting engine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import config
from src.app.utils import (
    build_feature_matrix,
    get_available_odds_cols,
    load_clean_data,
    load_model,
    run_backtest_cached,
)
from src.backtesting import get_backtest_guide
from src.feature_engineering import train_val_test_split

st.set_page_config(page_title="Backtest Results", page_icon="📊", layout="wide")

# ── Custom CSS ──────────────────────────────────────────
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
    .metric-value { font-size: 1.75rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.8rem; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-positive { color: #4caf50; }
    .metric-negative { color: #f44336; }
    .metric-neutral { color: #ffc107; }
    .backtest-card {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .chart-container {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────
st.markdown("# 📊 Backtest Performance")
st.markdown(
    "Simulate value betting over historical test data and evaluate strategy "
    "performance across multiple metrics."
)

# ── Load data ──────────────────────────────────────────
model = st.session_state.get("model")
if model is None:
    model = load_model()
    if model is not None:
        st.session_state["model"] = model

data = st.session_state.get("data")
if data is None:
    data = load_clean_data()
    if data is not None:
        st.session_state["data"] = data

if model is None:
    st.error("⚠ No trained model found. Run `python train_xgboost.py` first.")
    st.stop()

if data is None:
    st.error("⚠ No preprocessed data found.")
    st.stop()


# ── Run backtest button ────────────────────────────────
if st.button("▶️ RUN FULL BACKTEST", type="primary", use_container_width=True):
    with st.spinner("Running backtest simulation ..."):

        # Build feature matrix
        result = build_feature_matrix(data)
        if result is None:
            st.error("Feature engineering failed.")
            st.stop()

        X, y = result

        # Split chronologically
        splits = train_val_test_split(X, y)

        # Align test features to only what the model was trained on
        # (build_features may produce columns the model hasn't seen)
        model_features = model.get_booster().feature_names
        X_test_aligned = splits["X_test"].copy()
        for col in model_features:
            if col not in X_test_aligned.columns:
                X_test_aligned[col] = 0.0
        X_test_aligned = X_test_aligned[model_features]
        splits["X_test"] = X_test_aligned

        # Use the saved model as-is — it was trained chronologically in
        # train_xgboost.py and never saw test data, so predictions are valid.

        # Extract test odds if available
        odds_df = None
        odds_cols = get_available_odds_cols(data)
        if odds_cols:
            # Sort data the same way as build_features
            data_sorted = data.copy()
            if "date" in data_sorted.columns:
                data_sorted["date"] = pd.to_datetime(data_sorted["date"])
                data_sorted.sort_values(["date", "home_team"], inplace=True)
                data_sorted.reset_index(drop=True, inplace=True)

            n_total = len(X)
            n_test = len(splits["X_test"])
            test_start = n_total - n_test
            odds_df = data_sorted.iloc[test_start:test_start + n_test][
                list(odds_cols) + ["home_team", "away_team"]
            ].copy()

        # Run backtest
        bt_result = run_backtest_cached(
            model, splits["X_test"], splits["y_test"],
            odds_df=odds_df, odds_cols=odds_cols or ("BbAvA", "BbAvD", "BbAvH"),
        )

        metrics = bt_result["metrics"]
        chart_paths = bt_result["chart_paths"]

        # ── Store in session state ──────────────────────
        st.session_state["backtest_metrics"] = metrics
        st.session_state["backtest_charts"] = chart_paths
        st.session_state["backtest_complete"] = True

if not st.session_state.get("backtest_complete"):
    st.info("👆 Click **Run Full Backtest** to see historical performance data.")
    st.markdown(get_backtest_guide())
else:
    metrics = st.session_state["backtest_metrics"]
    chart_paths = st.session_state["backtest_charts"]

    # ── Summary metrics row ─────────────────────────────
    st.markdown("## 📈 Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        roi_color = "metric-positive" if metrics.roi_pct > 0 else "metric-negative" if metrics.roi_pct < 0 else "metric-neutral"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">ROI</div>'
            f'<div class="metric-value {roi_color}">{metrics.roi_pct:+.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        yield_color = "metric-positive" if metrics.yield_pct > 0 else "metric-negative"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">YIELD</div>'
            f'<div class="metric-value {yield_color}">{metrics.yield_pct:+.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">WIN RATE</div>'
            f'<div class="metric-value">{metrics.win_rate_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col4:
        dd_color = "metric-positive" if metrics.max_drawdown_pct < 10 else "metric-negative" if metrics.max_drawdown_pct > 25 else "metric-neutral"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">MAX DRAWDOWN</div>'
            f'<div class="metric-value {dd_color}">{metrics.max_drawdown_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Detail metrics ──────────────────────────────────
    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
    with detail_col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Total Bets</div>'
            f'<div class="metric-value">{metrics.total_bets:,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with detail_col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Profit / Loss</div>'
            f'<div class="metric-value {"metric-positive" if metrics.total_profit >= 0 else "metric-negative"}">'
            f'£{metrics.total_profit:+,.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with detail_col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Profit Factor</div>'
            f'<div class="metric-value">{metrics.profit_factor:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with detail_col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Avg EV</div>'
            f'<div class="metric-value {"metric-positive" if metrics.avg_ev > 0 else "metric-negative"}">'
            f'{metrics.avg_ev:+.2%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Streaks
    streak_col1, streak_col2, streak_col3, streak_col4 = st.columns(4)
    with streak_col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Winning Bets</div>'
            f'<div class="metric-value metric-positive">{metrics.winning_bets}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with streak_col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Losing Bets</div>'
            f'<div class="metric-value metric-negative">{metrics.losing_bets}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with streak_col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Longest Win Streak</div>'
            f'<div class="metric-value metric-positive">{metrics.longest_win_streak}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with streak_col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Longest Lose Streak</div>'
            f'<div class="metric-value metric-negative">{metrics.longest_lose_streak}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Charts ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Performance Charts")

    if chart_paths:
        # Bankroll curve + drawdown side by side
        chart_row1 = st.columns(2)
        with chart_row1[0]:
            if "bankroll_curve" in chart_paths:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.image(chart_paths["bankroll_curve"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with chart_row1[1]:
            if "drawdown" in chart_paths:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.image(chart_paths["drawdown"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Cumulative profit + bet outcomes side by side
        chart_row2 = st.columns(2)
        with chart_row2[0]:
            if "cumulative_profit" in chart_paths:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.image(chart_paths["cumulative_profit"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with chart_row2[1]:
            if "bet_outcomes" in chart_paths:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.image(chart_paths["bet_outcomes"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Performance Assessment ──────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Performance Assessment")

    assessment_parts = []
    if metrics.roi_pct > 0:
        assessment_parts.append(f"✅ Profitable strategy with {metrics.roi_pct:+.1f}% ROI")
    else:
        assessment_parts.append(f"🔴 Loss-making strategy with {metrics.roi_pct:+.1f}% ROI")

    if metrics.max_drawdown_pct < 10:
        assessment_parts.append(f"✓ Low drawdown ({metrics.max_drawdown_pct:.1f}%) — good risk management")
    elif metrics.max_drawdown_pct < 25:
        assessment_parts.append(f"⚠ Moderate drawdown ({metrics.max_drawdown_pct:.1f}%) — acceptable")
    else:
        assessment_parts.append(f"🔴 High drawdown ({metrics.max_drawdown_pct:.1f}%) — high risk of ruin")

    if metrics.profit_factor >= 2.0:
        assessment_parts.append("✓ Profit factor ≥ 2.0 — strong risk/reward")
    elif metrics.profit_factor >= 1.0:
        assessment_parts.append("~ Profit factor ≥ 1.0 — marginal profitability")
    else:
        assessment_parts.append("🔴 Profit factor < 1.0 — losses exceed gains")

    for part in assessment_parts:
        st.markdown(f"- {part}")

    # ── Detailed stats table ───────────────────────────
    with st.expander("📋 Detailed Backtest Statistics"):
        detail_data = {
            "Metric": [
                "Total Bets", "Winning Bets", "Losing Bets",
                "Win Rate", "Total Staked", "Total Profit",
                "ROI", "Yield", "Max Drawdown",
                "Average Odds", "Average EV", "Profit Factor",
                "Longest Win Streak", "Longest Losing Streak",
                "Final Bankroll", "Starting Bankroll",
            ],
            "Value": [
                metrics.total_bets, metrics.winning_bets, metrics.losing_bets,
                f"{metrics.win_rate_pct:.1f}%",
                f"£{metrics.total_staked:,.2f}",
                f"£{metrics.total_profit:+,.2f}",
                f"{metrics.roi_pct:+.2f}%",
                f"{metrics.yield_pct:+.2f}%",
                f"{metrics.max_drawdown_pct:.1f}%",
                f"{metrics.avg_odds:.4f}",
                f"{metrics.avg_ev:+.2%}",
                f"{metrics.profit_factor:.2f}",
                f"{metrics.longest_win_streak} bets",
                f"{metrics.longest_lose_streak} bets",
                f"£{metrics.final_bankroll:,.2f}",
                f"£{metrics.initial_bankroll:,.0f}",
            ],
        }
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)


# ── Navigation ──────────────────────────────────────────
st.markdown("---")
st.page_link("dashboard.py", label="← Back to Dashboard", use_container_width=True)
