"""
Value Bets Page — compute and display value betting opportunities using model
probabilities vs bookmaker odds.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app.utils import (
    build_feature_matrix,
    get_available_odds_cols,
    get_available_teams,
    load_clean_data,
    load_model,
)
from src.feature_engineering import build_features
from src.value_betting import compute_value_bets, get_calculation_guide

st.set_page_config(page_title="Value Bets", page_icon="💰", layout="wide")

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .value-card {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
    }
    .value-positive {
        border-left: 4px solid #4caf50;
    }
    .value-negative {
        border-left: 4px solid #f44336;
    }
    .ev-positive { color: #4caf50; font-weight: 700; }
    .ev-negative { color: #f44336; font-weight: 700; }
    .calc-box {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #8b8fa3;
    }
    .calc-box strong { color: #fff; }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────
st.markdown("# 💰 Value Bet Finder")
st.markdown(
    "Enter bookmaker odds and see which outcomes the model believes offer positive "
    "expected value."
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


# ── Team selection ─────────────────────────────────────
teams = get_available_teams(data)

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox(
        "🏠 **Home Team**", teams,
        index=teams.index("Manchester United") if "Manchester United" in teams else 0,
        key="vb_home",
    )
with col2:
    away_team = st.selectbox(
        "✈️ **Away Team**", teams,
        index=teams.index("Liverpool") if "Liverpool" in teams else (
            teams.index("Chelsea") if "Chelsea" in teams else min(1, len(teams) - 1)
        ),
        key="vb_away",
    )


# ── Odds input ─────────────────────────────────────────
st.markdown("### 📊 Enter Bookmaker Odds")
odds_col1, odds_col2, odds_col3 = st.columns(3)
with odds_col1:
    home_odds = st.number_input(
        f"**{home_team}** (Home)", min_value=1.01, max_value=100.0,
        value=2.10, step=0.05, format="%.2f",
    )
with odds_col2:
    draw_odds = st.number_input(
        "**Draw**", min_value=1.01, max_value=100.0,
        value=3.40, step=0.05, format="%.2f",
    )
with odds_col3:
    away_odds = st.number_input(
        f"**{away_team}** (Away)", min_value=1.01, max_value=100.0,
        value=3.80, step=0.05, format="%.2f",
    )


# ── Settings ───────────────────────────────────────────
with st.expander("⚙️ Betting Settings"):
    bankroll = st.number_input(
        "Bankroll (£)", min_value=100.0, max_value=1_000_000.0,
        value=1000.0, step=100.0,
    )
    kelly_fraction = st.slider(
        "Kelly Fraction", min_value=0.0, max_value=1.0,
        value=0.25, step=0.05,
        help="Fraction of Kelly Criterion to use. 0.25 = 25% Kelly (conservative).",
    )
    min_ev = st.slider(
        "Minimum EV", min_value=0.0, max_value=0.5,
        value=0.0, step=0.01,
        help="Only flag bets with EV above this threshold.",
    )


# ── Predict button ─────────────────────────────────────
if st.button("💰 CALCULATE VALUE", type="primary", use_container_width=True):
    with st.spinner("Running model and computing value metrics ..."):

        # Build feature matrix and get model prediction for this matchup
        synthetic = {
            "date": pd.Timestamp.now(),
            "home_team": home_team,
            "away_team": away_team,
            "result": "H",
            "home_goals": 0,
            "away_goals": 0,
        }
        for col in data.columns:
            if col not in synthetic:
                synthetic[col] = data[col].iloc[-1] if len(data) > 0 else 0

        df_extended = pd.concat([data, pd.DataFrame([synthetic])], ignore_index=True)
        X_full, _ = build_features(df_extended, is_training=False)
        feature_row = X_full.iloc[-1:]
        probs = model.predict_proba(feature_row)[0]  # [away, draw, home]

        # ── Compute value bets ──────────────────────────
        odds_array = [[away_odds, draw_odds, home_odds]]
        probs_array = [probs.tolist()]
        teams_list = [(home_team, away_team)]

        value_df = compute_value_bets(
            odds=odds_array,
            model_probs=probs_array,
            team_matches=teams_list,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_ev=min_ev,
        )

        # ── Display results ─────────────────────────────
        st.markdown("## 📊 Results")

        # Bookmaker margin
        implied = sum(1.0 / o for o in [home_odds, draw_odds, away_odds])
        margin_pct = (implied - 1.0) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="value-card">'
                f'<div style="color:#8b8fa3;font-size:0.85rem">Bookmaker Margin</div>'
                f'<div style="color:#fff;font-size:1.5rem;font-weight:700">{margin_pct:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            n_pos = value_df["positive_ev"].sum()
            st.markdown(
                f'<div class="value-card">'
                f'<div style="color:#8b8fa3;font-size:0.85rem">Value Bet Opportunities</div>'
                f'<div style="color:#4caf50;font-size:1.5rem;font-weight:700">{int(n_pos)} / 3</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col3:
            avg_ev = value_df["ev"].mean()
            ev_color = "#4caf50" if avg_ev > 0 else "#f44336"
            st.markdown(
                f'<div class="value-card">'
                f'<div style="color:#8b8fa3;font-size:0.85rem">Average EV</div>'
                f'<div style="color:{ev_color};font-size:1.5rem;font-weight:700">{avg_ev:+.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Per-outcome breakdown
        st.markdown("### Per-Outcome Analysis")

        for _, row in value_df.iterrows():
            is_pos = row["positive_ev"]
            card_class = "value-positive" if is_pos else "value-negative"
            rec = row["recommendation"]

            st.markdown(
                f'<div class="value-card {card_class}">',
                unsafe_allow_html=True,
            )

            outcome_col1, outcome_col2 = st.columns([1, 2])
            with outcome_col1:
                st.markdown(f"**{row['outcome_label']}**")
                st.markdown(f"Odds: **{row['decimal_odds']:.2f}**")

                if is_pos:
                    stake_str = f"£{row['kelly_stake']:.2f} ({row['kelly_pct']:.1%} of bankroll)"
                    st.markdown(f"💰 Stake: {stake_str}")
                    st.markdown(f"**{rec}**")

            with outcome_col2:
                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Model Prob", f"{row['model_prob']:.1%}")
                m2.metric("Fair Prob", f"{row['fair_prob']:.1%}")
                m3.metric("Edge", f"{row['prob_edge']:+.1%}")
                m4.metric("EV", f"{row['ev']:+.1%}",
                          delta_color="off")

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Explanation ─────────────────────────────────
        with st.expander("📖 How these calculations work"):
            st.markdown(get_calculation_guide())

else:
    st.info("👆 Enter odds above and click **Calculate Value** to see analysis.")


# ── Navigation ──────────────────────────────────────────
st.markdown("---")
st.page_link("dashboard.py", label="← Back to Dashboard", use_container_width=True)
