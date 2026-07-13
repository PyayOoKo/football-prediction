"""
Football Prediction Dashboard — Main Page.

Provides an overview of the model, recent match data, and navigation to
prediction, value betting, and backtest pages.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config import config

# Must be the first Streamlit command
st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.app.utils import (
    get_available_teams,
    get_latest_matches,
    load_clean_data,
    load_model,
    run_model_diagnostic,
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base theme overrides ── */
    .stApp { background: #0e1117; }
    .stApp header { background: #1a1d27; }
    
    /* ── Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b8fa3;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ── Hero section ── */
    .hero {
        background: linear-gradient(135deg, #1a1d27 0%, #16213e 50%, #1a1d27 100%);
        border: 1px solid #2a2d3a;
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #4fc3f7, #81c784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #8b8fa3;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: #1b5e20; color: #81c784; }
    .badge-red { background: #b71c1c; color: #ef9a9a; }
    .badge-blue { background: #0d47a1; color: #90caf9; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────
if "model" not in st.session_state:
    st.session_state.model = load_model()
if "diagnostic" not in st.session_state:
    st.session_state.diagnostic = None
if "data" not in st.session_state:
    st.session_state.data = load_clean_data()


# ═══════════════════════════════════════════════════════════
#  Hero section
# ═══════════════════════════════════════════════════════════

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<h1>⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    "<p>AI-powered match outcome prediction, value betting analysis, "
    "and backtested performance tracking.</p>",
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  Key metrics row
# ═══════════════════════════════════════════════════════════

col1, col2, col3, col4 = st.columns(4)

data = st.session_state.data
model = st.session_state.model

with col1:
    n_matches = len(data) if data is not None else 0
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{n_matches:,}</div>'
        f'<div class="metric-label">Historical Matches</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col2:
    n_teams = len(get_available_teams(data)) if data is not None else 0
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{n_teams}</div>'
        f'<div class="metric-label">Teams</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col3:
    if model is not None:
        model_type = config.train.model_type.upper()
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{model_type}</div>'
            f'<div class="metric-label">Active Model</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:#e74c3c;">⚠</div>'
            f'<div class="metric-label">No Model Loaded</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col4:
    if data is not None and "date" in data.columns:
        dates = data["date"].dropna()
        if len(dates) > 0:
            latest = pd.to_datetime(dates.max())
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{latest.strftime("%b %Y")}</div>'
                f'<div class="metric-label">Latest Match</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════
#  Two-column layout: Recent matches + Quick actions
# ═══════════════════════════════════════════════════════════

left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("### 📋 Recent Matches")

    if data is not None:
        latest = get_latest_matches(data, n=15)
        if len(latest) > 0:
            display = latest.copy()
            # Format date
            if "date" in display.columns:
                display["date"] = pd.to_datetime(display["date"]).dt.strftime("%d %b %Y")

            # Create readable result column
            if all(c in display.columns for c in ["home_goals", "away_goals", "home_team", "away_team"]):
                display["score"] = display.apply(
                    lambda r: f"{int(r['home_goals'])}–{int(r['away_goals'])}"
                    if pd.notna(r["home_goals"]) and pd.notna(r["away_goals"])
                    else "—",
                    axis=1,
                )
                display["match"] = display.apply(
                    lambda r: f"{r['home_team']} vs {r['away_team']}", axis=1
                )
                show_cols = [c for c in ["date", "match", "score", "result"] if c in display.columns]
                st.dataframe(
                    display[show_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "result": st.column_config.TextColumn("Result", width="small"),
                        "score": st.column_config.TextColumn("Score", width="small"),
                    },
                )
            else:
                st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("No match data available.")
    else:
        st.warning("No data found. Run preprocessing first.")

with right_col:
    st.markdown("### 🚀 Quick Actions")
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)

    model_ok = model is not None
    data_ok = data is not None

    if model_ok and data_ok:
        st.success("✅ Model loaded and ready")
        st.page_link("pages/1_Predict.py", label="🔮 Predict a Match", use_container_width=True)
        st.page_link("pages/2_Value_Bets.py", label="💰 Find Value Bets", use_container_width=True)
        st.page_link("pages/3_Backtest.py", label="📊 View Backtest", use_container_width=True)
        st.page_link("pages/4_WorldCup.py", label="🏆 World Cup 2026", use_container_width=True)
    else:
        if not model_ok:
            st.error("⚠ No trained model found.")
            st.info(
                "Run `python train_xgboost.py` from the terminal "
                "to train and save a model."
            )
        if not data_ok:
            st.error("⚠ No preprocessed data found.")
            st.info(
                "Run `python -c \"from src.preprocessing import "
                "run_preprocessing; run_preprocessing()\"` to prepare data."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Model info card
    if model is not None:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Model Configuration**")
        st.markdown(f"- Type: `{config.train.model_type}`")
        st.markdown(f"- Estimators: `{config.train.n_estimators}`")
        st.markdown(f"- Max depth: `{config.train.max_depth}`")
        st.markdown(f"- Learning rate: `{config.train.learning_rate}`")
        st.markdown(f"- Features: rolling stats, H2H, league position")
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  Model Performance & Balance Section
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📊 Model Performance & Balance")

if model is not None and data is not None:
    # Run diagnostic if not cached
    diag = st.session_state.diagnostic
    if diag is None:
        with st.spinner("Running model diagnostic on test data ..."):
            diag = run_model_diagnostic(model, data)
            st.session_state.diagnostic = diag

    if diag is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{diag["accuracy"]:.1%}</div>'
                f'<div class="metric-label">Test Accuracy</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col2:
            best_base = diag["best_baseline"]
            imp = diag["improvement"]
            imp_color = "#4caf50" if imp > 0 else "#f44336"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{imp_color}">{imp:+.1%}</div>'
                f'<div class="metric-label">vs Baseline ({best_base:.0%})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col3:
            ll = diag["log_loss"]
            ll_color = "#4caf50" if ll < 0.9 else "#ffc107" if ll < 1.1 else "#f44336"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{ll_color}">{ll:.4f}</div>'
                f'<div class="metric-label">Log-Loss</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{diag["n_test"]:,}</div>'
                f'<div class="metric-label">Test Matches</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Per-class metrics table ──────────────────────
        st.markdown("### Per-Class Performance (Balance Check)")

        class_data = []
        for cls in ["Home Win", "Draw", "Away Win"]:
            p = diag["precision"][cls]
            r = diag["recall"][cls]
            f = diag["f1"][cls]
            actual = diag["actual_dist"][cls]
            predicted = diag["prediction_dist"][cls]
            diff = predicted - actual
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            diff_color = "#4caf50" if diff == 0 else "#f44336" if abs(diff) > 5 else "#ffc107"

            class_data.append({
                "Class": cls,
                "Precision": p,
                "Recall": r,
                "F1-Score": f,
                "Actual": actual,
                "Predicted": predicted,
                "Δ": f'<span style="color:{diff_color}">{diff_str}</span>',
            })

        st.markdown(
            "<div style='color:#8b8fa3;font-size:0.85rem;margin-bottom:0.5rem'>"
            "A well-balanced model has similar Precision, Recall, and F1 across all three classes. "
            "Large disparities (especially \"Draw\" being much lower) indicate class imbalance issues."
            "</div>",
            unsafe_allow_html=True,
        )

        df_class = pd.DataFrame(class_data)
        st.dataframe(
            df_class,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Δ": st.column_config.TextColumn("Δ Pred vs Actual"),
                "Precision": st.column_config.NumberColumn(format=".3f"),
                "Recall": st.column_config.NumberColumn(format=".3f"),
                "F1-Score": st.column_config.NumberColumn(format=".3f"),
            },
        )

        # ── Balance assessment ───────────────────────────
        draw_f1 = diag["f1"]["Draw"]
        home_f1 = diag["f1"]["Home Win"]
        away_f1 = diag["f1"]["Away Win"]
        draw_recall = diag["recall"]["Draw"]

        issues = []
        if draw_f1 < 0.2:
            issues.append("🔴 **Draw blindness** — model rarely predicts draws (F1 < 0.20). This is the #1 balance issue.")
        elif draw_f1 < 0.35:
            issues.append("⚠️ **Draw prediction is weak** (F1 < 0.35). The model struggles with the minority class.")

        if home_f1 > away_f1 + 0.15:
            issues.append("⚖️ **Home win bias** — model significantly favors home teams over away teams.")

        pred_draw = diag["prediction_dist"]["Draw"]
        actual_draw = diag["actual_dist"]["Draw"]
        if actual_draw > 0 and pred_draw < actual_draw * 0.5:
            issues.append(f"📉 **Under-predicts draws** — predicted {pred_draw} vs actual {actual_draw} draws.")

        if not issues:
            st.success("✅ Model appears well-balanced across all three outcomes.")
        else:
            st.warning("⚠️ Balance issues detected:")
            for issue in issues:
                st.markdown(f"- {issue}")

        # ── Confusion Matrix ────────────────────────────
        with st.expander("🔍 View Confusion Matrix"):
            cm = diag["confusion_matrix"]
            labels = diag["class_labels"]

            # Build confusion matrix as a dataframe
            cm_data = []
            for i, actual_label in enumerate(labels):
                row_data = {"Actual \\ Predicted": actual_label}
                for j, pred_label in enumerate(labels):
                    row_data[pred_label] = cm[i][j]
                row_data["Correct"] = cm[i][i]
                row_data["Total"] = sum(cm[i])
                row_data["Recall"] = f"{cm[i][i]/sum(cm[i]):.0%}" if sum(cm[i]) > 0 else "—"
                cm_data.append(row_data)

            st.dataframe(
                pd.DataFrame(cm_data),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "<div style='color:#8b8fa3;font-size:0.85rem'>"
                "<strong>Reading the confusion matrix:</strong> Rows = actual outcome, Columns = predicted outcome. "
                "Diagonal cells (top-left to bottom-right) are correct predictions. "
                "High off-diagonal values show systematic confusion between two outcomes. "
                "For example, if many actual Draws are predicted as Home Wins, the model is draw-blind."
                "</div>",
                unsafe_allow_html=True,
            )

    else:
        st.info("Run the diagnostic to see model performance metrics.")
        if st.button("🔬 Run Model Diagnostic", use_container_width=True):
            with st.spinner("Running diagnostic on test data ..."):
                diag = run_model_diagnostic(model, data)
                st.session_state.diagnostic = diag
                if diag is None:
                    st.error("Diagnostic failed. Check that a model is trained and data is available.")
                else:
                    st.rerun()

else:
    st.info("Load a model and data to see performance metrics.")


# ── Footer ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem'>"
    "Football Match Predictor | Built with ❤️ using XGBoost & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
