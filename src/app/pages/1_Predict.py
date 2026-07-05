"""
Prediction Page — select two teams and get instant match outcome predictions.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app.utils import (
    build_feature_matrix,
    get_available_teams,
    get_matchup_stats,
    load_clean_data,
    load_model,
)
from src.feature_engineering import build_features

st.set_page_config(page_title="Predict a Match", page_icon="🔮", layout="wide")

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .predict-card {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .prob-bar {
        height: 28px;
        border-radius: 14px;
        margin: 4px 0;
        position: relative;
        transition: width 1s ease;
    }
    .team-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .team-home { background: #1b5e20; color: #a5d6a7; }
    .team-away { background: #b71c1c; color: #ef9a9a; }
    .outcome-home { color: #4caf50; font-weight: 600; }
    .outcome-draw { color: #ffc107; font-weight: 600; }
    .outcome-away { color: #f44336; font-weight: 600; }
    .stat-label { color: #8b8fa3; font-size: 0.85rem; }
    .stat-value { color: #ffffff; font-size: 1.1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────
st.markdown("# 🔮 Match Predictor")
st.markdown(
    "Select two teams and click **Predict** to see the model's match outcome probabilities."
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
    st.error("⚠ No preprocessed data found. Run preprocessing first.")
    st.stop()


# ── Team selector ──────────────────────────────────────
teams = get_available_teams(data)

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 **Home Team**", teams, index=teams.index("Manchester United") if "Manchester United" in teams else 0)
with col2:
    away_team = st.selectbox("✈️ **Away Team**", teams, index=teams.index("Liverpool") if "Liverpool" in teams else (teams.index("Chelsea") if "Chelsea" in teams else min(1, len(teams) - 1)))


# ── Matchup info ───────────────────────────────────────
h2h = get_matchup_stats(data, home_team, away_team)

info_col1, info_col2, info_col3, info_col4 = st.columns(4)
with info_col1:
    st.markdown(f'<div class="stat-label">Previous meetings</div><div class="stat-value">{h2h["matches"]}</div>', unsafe_allow_html=True)
with info_col2:
    st.markdown(f'<div class="stat-label">{home_team} wins</div><div class="stat-value">{h2h["home_wins"]}</div>', unsafe_allow_html=True)
with info_col3:
    st.markdown(f'<div class="stat-label">Draws</div><div class="stat-value">{h2h["draws"]}</div>', unsafe_allow_html=True)
with info_col4:
    st.markdown(f'<div class="stat-label">{away_team} wins</div><div class="stat-value">{h2h["away_wins"]}</div>', unsafe_allow_html=True)

if h2h.get("last_results"):
    result_map = {"H": f"✅ {home_team}", "A": f"✅ {away_team}", "D": "🤝 Draw"}
    labels = [result_map.get(r, r) for r in h2h["last_results"]]
    st.markdown(f'<div class="stat-label">Last {len(labels)} meetings: {"  •  ".join(labels)}</div>', unsafe_allow_html=True)


# ── Predict button ─────────────────────────────────────
st.markdown("---")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_clicked = st.button("🔮 PREDICT NOW", type="primary", use_container_width=True)


if predict_clicked:
    with st.spinner("Building features and running model ..."):

        # Build features on the full dataset to get the feature matrix
        result = build_feature_matrix(data)
        if result is None:
            st.error("Feature engineering failed.")
            st.stop()

        X, y = result

        # Find the most recent row that matches our selected teams
        # We need to reconstruct the feature vector for this matchup.
        # The simplest approach: append a synthetic row to the data.
        synthetic = {
            "date": pd.Timestamp.now(),
            "home_team": home_team,
            "away_team": away_team,
            "result": "H",  # placeholder — will be replaced
            "home_goals": 0,
            "away_goals": 0,
        }
        # Add any other columns from data
        for col in data.columns:
            if col not in synthetic:
                synthetic[col] = data[col].iloc[-1] if len(data) > 0 else 0

        df_extended = pd.concat([data, pd.DataFrame([synthetic])], ignore_index=True)
        X_full, _ = build_features(df_extended, is_training=False)

        # The last row is our synthetic match
        feature_row = X_full.iloc[-1:]

        # Predict
        probs = model.predict_proba(feature_row)[0]  # [away, draw, home]
        pred_class = int(model.predict(feature_row)[0])
        labels = ["Away Win", "Draw", "Home Win"]
        confidence = probs[pred_class]

        # ── Results display ──────────────────────────────
        st.markdown("## 📊 Prediction Results")

        result_col1, result_col2 = st.columns([1, 1.5])

        with result_col1:
            st.markdown('<div class="predict-card">', unsafe_allow_html=True)
            st.markdown(f"### {home_team} vs {away_team}")
            st.markdown("---")

            # Predicted outcome
            outcome_label = labels[pred_class]
            if pred_class == 2:
                outcome_html = f'<span class="outcome-home">🏠 {home_team} Win</span>'
            elif pred_class == 1:
                outcome_html = f'<span class="outcome-draw">🤝 Draw</span>'
            else:
                outcome_html = f'<span class="outcome-away">✈️ {away_team} Win</span>'

            st.markdown(f"**Predicted:** {outcome_html}")
            st.markdown(f"**Confidence:** {confidence:.1%}")
            st.markdown("</div>", unsafe_allow_html=True)

        with result_col2:
            st.markdown('<div class="predict-card">', unsafe_allow_html=True)
            st.markdown("### Outcome Probabilities")

            # Home win
            home_pct = probs[2] * 100
            st.markdown(f"🏠 **{home_team}**")
            home_color = "#4caf50"
            st.progress(home_pct / 100)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between">'
                f'<span style="color:#8b8fa3">Probability</span>'
                f'<span style="color:#fff;font-weight:600">{home_pct:.1f}%</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            # Draw
            st.markdown("")
            draw_pct = probs[1] * 100
            st.markdown(f"🤝 **Draw**")
            st.progress(draw_pct / 100)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between">'
                f'<span style="color:#8b8fa3">Probability</span>'
                f'<span style="color:#fff;font-weight:600">{draw_pct:.1f}%</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            # Away win
            st.markdown("")
            away_pct = probs[0] * 100
            st.markdown(f"✈️ **{away_team}**")
            away_color = "#f44336"
            st.progress(away_pct / 100)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between">'
                f'<span style="color:#8b8fa3">Probability</span>'
                f'<span style="color:#fff;font-weight:600">{away_pct:.1f}%</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Interpretation ──────────────────────────────
        st.markdown("### 💡 Interpretation")
        if pred_class == 2:
            interpretation = (
                f"The model predicts **{home_team}** will win at home with "
                f"{confidence:.1%} confidence. "
            )
        elif pred_class == 0:
            interpretation = (
                f"The model predicts **{away_team}** will win away with "
                f"{confidence:.1%} confidence. "
            )
        else:
            interpretation = (
                f"The model predicts a **Draw** with {confidence:.1%} confidence. "
            )

        # Add context based on probabilities
        if home_pct > 55:
            interpretation += (
                f"{home_team} have strong home advantage ({home_pct:.0f}%), "
                "consistent with typical home win rates in football."
            )
        elif home_pct < 30 and away_pct > 40:
            interpretation += (
                f"{away_team} are favoured despite playing away ({away_pct:.0f}%)."
            )

        st.info(interpretation)

        # ── Feature impact ──────────────────────────────
        if hasattr(model, "feature_importances_"):
            st.markdown("### 🔍 Top Influential Features")
            importances = model.feature_importances_
            feature_names = X_full.columns
            indices = importances.argsort()[::-1][:8]

            impact_data = []
            for idx in indices:
                val = feature_row.iloc[0, idx]
                impact_data.append({
                    "Feature": feature_names[idx],
                    "Importance": f"{importances[idx]:.4f}",
                    "Value": f"{val:.3f}" if isinstance(val, (int, float)) else str(val),
                })

            st.dataframe(
                pd.DataFrame(impact_data),
                use_container_width=True,
                hide_index=True,
            )

elif not predict_clicked:
    st.info("👆 Select two teams above and click **Predict Now** to see results.")


# ── Navigation ──────────────────────────────────────────
st.markdown("---")
st.page_link("dashboard.py", label="← Back to Dashboard", use_container_width=True)
