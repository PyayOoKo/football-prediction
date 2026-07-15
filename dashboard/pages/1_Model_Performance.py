"""
Model Performance Dashboard — monitor accuracy, confusion matrix, feature importance,
and per-class balance over time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import config

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

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
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.8rem; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 Model Performance")
st.markdown("Monitor accuracy trends, confusion matrices, feature importance, and class balance.")


# ── Load report files ──────────────────────────────────
@st.cache_data(ttl=60)
def load_validation_reports() -> list[dict]:
    """Load all available validation/performance report JSON files."""
    reports = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return reports
    for pattern in ["*validation*.json", "*performance*.json", "calibration_results*.json",
                     "phase3_vs_phase4*.json", "phase4_leaderboard*.csv"]:
        for f in reports_dir.glob(pattern):
            try:
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                    reports.append({"file": f.name, "type": "csv", "data": df.to_dict(orient="records")})
                else:
                    with open(f) as fh:
                        reports.append({"file": f.name, "type": "json", "data": json.load(fh)})
            except Exception:
                pass
    return reports


reports = load_validation_reports()

if not reports:
    st.info("No validation/performance reports found. Run training scripts to generate them.")
    st.stop()

# ── Report selector ────────────────────────────────────
st.markdown("### Select Report")
report_names = [r["file"] for r in reports]
selected = st.selectbox("Choose a report to view:", report_names)
selected_report = next(r for r in reports if r["file"] == selected)

# ── Display report content ─────────────────────────────
if selected_report["type"] == "json":
    data = selected_report["data"]

    # Try to find metrics at various nesting levels
    metrics = {}

    # Phase 3/4 validation format
    if "metrics" in data:
        metrics = data["metrics"]
    elif "overall" in data:
        metrics = data["overall"]

    # Direct keys
    for key in ["accuracy", "log_loss", "brier_score", "roc_auc", "f1", "precision", "recall"]:
        if key in data:
            metrics[key] = data[key]

    if metrics:
        st.markdown("### 📊 Key Metrics")
        cols = st.columns(min(len(metrics), 4))
        for i, (key, value) in enumerate(list(metrics.items())[:12]):
            with cols[i % 4]:
                if isinstance(value, (int, float)):
                    formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{formatted}</div>'
                        f'<div class="metric-label">{key.replace("_", " ").title()}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Per-class metrics
    for cls_key in ["per_class", "class_metrics", "per_class_metrics", "classes"]:
        per_class = data.get(cls_key, data.get("metrics", {}).get(cls_key, {}))
        if per_class:
            st.markdown("### Per-Class Performance")
            class_df = pd.DataFrame(per_class).T if isinstance(per_class, dict) else pd.DataFrame(per_class)
            st.dataframe(class_df, use_container_width=True)
            break

    # Confusion matrix (if available)
    cm = data.get("confusion_matrix", data.get("metrics", {}).get("confusion_matrix", None))
    if cm:
        st.markdown("### Confusion Matrix")
        labels = data.get("class_labels", ["Away Win", "Draw", "Home Win"])
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=500, height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    for fi_key in ["feature_importance", "feature_importances", "importances"]:
        fi = data.get(fi_key, data.get("metrics", {}).get(fi_key, {}))
        if fi and isinstance(fi, dict):
            st.markdown("### 🔍 Feature Importance")
            fi_df = pd.DataFrame([
                {"Feature": k, "Importance": v}
                for k, v in sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
            ])
            fig = px.bar(
                fi_df.head(20), y="Feature", x="Importance",
                orientation="h", title="Top 20 Features by Importance",
                color="Importance", color_continuous_scale="Viridis",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
            break

    # Raw JSON viewer
    with st.expander("📋 Raw Report Data"):
        st.json(data)

elif selected_report["type"] == "csv":
    st.dataframe(pd.DataFrame(selected_report["data"]), use_container_width=True)


# ── Manual model diagnostic ────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Run Model Diagnostic")

try:
    from src.app.utils import load_model, load_clean_data, run_model_diagnostic

    model = load_model()
    data = load_clean_data()

    if model is not None and data is not None:
        if st.button("▶️ Run Diagnostic", type="primary", use_container_width=True):
            with st.spinner("Running model diagnostic on test data..."):
                diag = run_model_diagnostic(model, data)
                if diag:
                    st.session_state["diagnostic"] = diag
                    st.success("Diagnostic complete!")

        if "diagnostic" in st.session_state:
            diag = st.session_state["diagnostic"]
            if diag:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{diag['accuracy']:.1%}")
                c2.metric("vs Baseline", f"{diag['improvement']:+.1%}")
                c3.metric("Log-Loss", f"{diag['log_loss']:.4f}")
                c4.metric("Test Samples", diag["n_test"])

                if "confusion_matrix" in diag:
                    st.markdown("### Confusion Matrix")
                    labels = diag.get("class_labels", ["Away", "Draw", "Home"])
                    fig = go.Figure(data=go.Heatmap(
                        z=diag["confusion_matrix"], x=labels, y=labels,
                        text=[[str(v) for v in row] for row in diag["confusion_matrix"]],
                        texttemplate="%{text}", colorscale="Blues",
                    ))
                    fig.update_layout(width=450, height=450, paper_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#ccc"))
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load a model and data to run diagnostics.")
except Exception:
    st.info("Diagnostic module not available. Train a model and ensure dependencies are installed.")
