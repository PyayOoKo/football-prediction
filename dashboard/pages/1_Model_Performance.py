"""
Model Performance Dashboard — monitor accuracy, confusion matrix, feature importance,
and per-class balance over time with rich Plotly visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import config
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
    confusion_matrix_heatmap,
    feature_importance_chart,
    radar_chart,
    area_trend_chart,
    comparison_bar_chart,
    info_row,
    Colors,
)

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

# ── Theme initialisation ───────────────────────────────
init_theme()

# ── Sidebar theme toggle ───────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    sidebar_theme_radio()
    st.markdown("---")

render_custom_css()

# ── Hero ────────────────────────────────────────────────
render_hero(
    title="🤖 Model Performance",
    subtitle="Monitor accuracy trends, confusion matrices, feature importance, "
             "and class balance across all trained models.",
    badges=[("Auto-refresh: 60s", "🔄"), ("Deep dive analytics", "📊")],
)


# ── Load report files ──────────────────────────────────
@st.cache_data(ttl=60)
def load_validation_reports() -> list[dict]:
    """Load all available validation/performance report JSON files."""
    reports = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return reports
    for pattern in [
        "*validation*.json", "*performance*.json", "calibration_results*.json",
        "phase3_vs_phase4*.json", "phase4_leaderboard*.csv", "phase3_leaderboard*.csv",
        "final_validation*.json", "backtest_comparison*.json",
    ]:
        for f in reports_dir.glob(pattern):
            try:
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                    reports.append({"file": f.name, "type": "csv", "data": df.to_dict(orient="records"), "df": df})
                else:
                    with open(f) as fh:
                        data = json.load(fh)
                    reports.append({"file": f.name, "type": "json", "data": data})
            except Exception:
                pass
    return reports


reports = load_validation_reports()

if not reports:
    st.info(
        '<div style="background:#141824;border:1px solid #1e2235;border-radius:12px;'
        'padding:2rem;text-align:center">'
        '<div style="font-size:3rem;margin-bottom:0.5rem">📭</div>'
        '<div style="color:#9ca3af;font-size:1rem">No validation/performance reports found.</div>'
        '<div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">'
        'Run training scripts to generate reports.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Report selector ────────────────────────────────────
report_names = sorted(set(r["file"] for r in reports))
section_header_sm("Select Report")
selected = st.selectbox("Choose a report to view:", report_names, label_visibility="collapsed")
selected_report = next(r for r in reports if r["file"] == selected)

# Metric mapping from various report formats
METRIC_ALIASES: dict[str, list[str]] = {
    "accuracy": ["accuracy", "acc", "overall_accuracy", "val_accuracy"],
    "log_loss": ["log_loss", "logloss", "val_log_loss", "cross_entropy"],
    "brier_score": ["brier_score", "brier", "brier_score_total", "brier_score_avg"],
    "roc_auc": ["roc_auc", "auc", "roc_auc_ovr", "auc_score"],
    "f1": ["f1", "f1_score", "f1_macro", "f1_weighted"],
    "precision": ["precision", "precision_macro"],
    "recall": ["recall", "recall_macro"],
    "ece": ["ece", "expected_calibration_error", "calibration_error"],
}


def _find_metric(data: dict, metric_name: str) -> float | None:
    """Find a metric value from nested dict using known aliases."""
    for alias in METRIC_ALIASES.get(metric_name, [metric_name]):
        # Direct key
        if alias in data:
            v = data[alias]
            return float(v) if isinstance(v, (int, float)) else None
        # Nested in metrics dict
        if "metrics" in data and isinstance(data["metrics"], dict):
            if alias in data["metrics"]:
                return float(data["metrics"][alias])
            # Nested in metrics.per_class
            if "per_class" in data["metrics"] and isinstance(data["metrics"]["per_class"], dict):
                for cls_metrics in data["metrics"]["per_class"].values():
                    if isinstance(cls_metrics, dict) and alias in cls_metrics:
                        return float(cls_metrics[alias])
        # Nested in overall
        if "overall" in data and isinstance(data["overall"], dict):
            if alias in data["overall"]:
                return float(data["overall"][alias])
    return None


# ── Extract metrics ────────────────────────────────────
if selected_report["type"] == "json":
    data = selected_report["data"]

    # Key metrics overview
    section_header("📊 Key Metrics", "📊")
    metrics_found = {}
    for name in ["accuracy", "log_loss", "brier_score", "roc_auc", "f1", "precision", "recall", "ece"]:
        v = _find_metric(data, name)
        if v is not None:
            metrics_found[name] = v

    if metrics_found:
        cols = st.columns(min(len(metrics_found), 4))
        for i, (key, value) in enumerate(metrics_found.items()):
            with cols[i % 4]:
                if key == "accuracy":
                    metric_card(cols[i % 4], f"{value:.1%}", "Accuracy",
                                delta=f"{'Above' if value > 0.5 else 'Below'} baseline",
                                up=value > 0.5)
                elif key == "log_loss":
                    metric_card(cols[i % 4], f"{value:.4f}", "Log Loss",
                                delta="Lower is better", up=value < 1.0)
                elif key == "brier_score":
                    metric_card(cols[i % 4], f"{value:.4f}", "Brier Score",
                                delta="Lower is better", up=value < 0.25)
                elif key == "roc_auc":
                    metric_card(cols[i % 4], f"{value:.3f}", "ROC AUC",
                                delta=f"{'Good' if value > 0.7 else 'Needs improvement'}",
                                up=value > 0.7)
                elif key == "ece":
                    metric_card(cols[i % 4], f"{value:.3f}", "Calibration Error (ECE)",
                                delta="Lower is better", up=value < 0.1)
                else:
                    metric_card(cols[i % 4], f"{value:.4f}", key.replace("_", " ").title())
    else:
        info_row("No standard metrics found in this report format.")

    # ── Radar chart from multiple reports ────────────────
    section_header("🕸️ Multi-Model Comparison", "🕸️")
    # Collect all reports that have accuracy + brier + log_loss
    multi_models = {}
    for r in reports:
        if r["type"] == "json":
            d = r["data"]
            acc = _find_metric(d, "accuracy")
            brier = _find_metric(d, "brier_score")
            logloss = _find_metric(d, "log_loss")
            if acc is not None and brier is not None and logloss is not None:
                # Normalise for radar (higher is better)
                model_name = d.get("model_name", d.get("model", r["file"].replace(".json", "").replace("backtest_", "")))
                multi_models[model_name] = {
                    "Accuracy (%)": acc * 100,
                    "Brier (inv)": (1 - brier) * 100,  # Invert so higher = better
                    "LogLoss (inv)": (2 - logloss) * 50 if logloss < 2 else 0,
                }

    if len(multi_models) >= 2:
        categories = ["Accuracy (%)", "Brier (inv)", "LogLoss (inv)"]
        values_dict = {
            name: [v["Accuracy (%)"], v["Brier (inv)"], v["LogLoss (inv)"]]
            for name, v in multi_models.items()
        }
        fig = radar_chart(categories, values_dict, title="Model Comparison Radar", height=400)
        st.plotly_chart(fig, use_container_width=True)
        info_row("Radar shows Accuracy, inverted Brier (higher=better), and inverted LogLoss (higher=better).")
    else:
        info_row("Need at least 2 models with accuracy + brier + log_loss for radar comparison.")

    # ── Per-class metrics ────────────────────────────────
    for cls_key in ["per_class", "class_metrics", "per_class_metrics", "classes"]:
        per_class = data.get(cls_key, data.get("metrics", {}).get(cls_key, {}))
        if per_class:
            section_header("📊 Per-Class Performance", "📊")
            class_df = pd.DataFrame(per_class).T if isinstance(per_class, dict) else pd.DataFrame(per_class)
            # Format percentages
            for col in class_df.columns:
                if class_df[col].dtype in (float, int) and class_df[col].max() <= 1.0 and class_df[col].min() >= 0:
                    class_df[col] = class_df[col].apply(lambda x: f"{x:.1%}")
            st.dataframe(class_df, use_container_width=True)

            # Per-class bar chart
            numeric_cols = []
            for c in class_df.columns:
                try:
                    pd.to_numeric(class_df[c].str.rstrip("%"))
                    numeric_cols.append(c)
                except (ValueError, AttributeError):
                    pass
            if numeric_cols:
                fig = px.bar(
                    class_df.reset_index().melt(id_vars="index", value_vars=numeric_cols),
                    x="index", y="value", color="variable", barmode="group",
                    title="Per-Class Metrics Comparison",
                    color_discrete_sequence=[Colors.PRIMARY, Colors.SUCCESS, Colors.GRADIENT_GOLD],
                )
                fig.update_layout(**{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                                     "font": {"color": Colors.TEXT_PRIMARY}})
                st.plotly_chart(fig, use_container_width=True)
            break

    # ── Confusion Matrix ────────────────────────────────
    cm = data.get("confusion_matrix", data.get("metrics", {}).get("confusion_matrix", None))
    if cm:
        section_header("🎯 Confusion Matrix", "🎯")
        labels = data.get("class_labels", ["Away Win", "Draw", "Home Win"])
        fig = confusion_matrix_heatmap(cm, labels, title="Confusion Matrix", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Derive metrics from CM
        cm_arr = np.array(cm)
        if cm_arr.shape == (3, 3):
            section_header_sm("Derived Metrics")
            diag = np.diag(cm_arr)
            row_sums = cm_arr.sum(axis=1)
            col_sums = cm_arr.sum(axis=0)
            total = cm_arr.sum()

            precision_vals = diag / np.where(col_sums > 0, col_sums, 1)
            recall_vals = diag / np.where(row_sums > 0, row_sums, 1)
            f1_vals = 2 * precision_vals * recall_vals / np.where(precision_vals + recall_vals > 0, precision_vals + recall_vals, 1)

            derived = pd.DataFrame({
                "Class": labels,
                "Support": row_sums.astype(int),
                "Precision": [f"{v:.1%}" for v in precision_vals],
                "Recall": [f"{v:.1%}" for v in recall_vals],
                "F1-Score": [f"{v:.1%}" for v in f1_vals],
            })
            derived.loc["Avg / Total"] = [
                "Weighted Avg",
                int(total),
                f"{np.average(precision_vals, weights=row_sums):.1%}",
                f"{np.average(recall_vals, weights=row_sums):.1%}",
                f"{np.average(f1_vals, weights=row_sums):.1%}",
            ]
            st.dataframe(derived, use_container_width=True, hide_index=True)

    # ── Feature Importance ────────────────────────────────
    for fi_key in ["feature_importance", "feature_importances", "importances"]:
        fi = data.get(fi_key, data.get("metrics", {}).get(fi_key, {}))
        if fi and isinstance(fi, dict):
            section_header("🔍 Feature Importance", "🔍")
            fig = feature_importance_chart(fi, title="Top 20 Features by Importance", top_n=20, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            fi_values = list(fi.values())
            info_row(f"Total features: {len(fi)} · "
                     f"Top 5 account for {sum(sorted(fi_values, reverse=True)[:5]) / sum(fi_values):.1%} of importance"
                     if sum(fi_values) > 0 else "")
            break

    # ── Calibration data ────────────────────────────────
    calibration = data.get("calibration", data.get("calibration_data", None))
    if calibration:
        section_header("🎚️ Calibration Plot", "🎚️")
        if isinstance(calibration, dict):
            cal_df = pd.DataFrame(calibration)
        elif isinstance(calibration, list):
            cal_df = pd.DataFrame(calibration)
        else:
            cal_df = None

        if cal_df is not None and not cal_df.empty:
            fig = go.Figure()
            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color=Colors.TEXT_MUTED, width=1),
                name="Perfect Calibration",
            ))
            # Model calibration
            if "mean_predicted" in cal_df.columns and "fraction_positives" in cal_df.columns:
                fig.add_trace(go.Scatter(
                    x=cal_df["mean_predicted"], y=cal_df["fraction_positives"],
                    mode="lines+markers",
                    line=dict(color=Colors.PRIMARY, width=2),
                    marker=dict(size=8, color=Colors.PRIMARY, symbol="diamond"),
                    name="Model",
                    hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",
                ))
                fig.update_layout(
                    xaxis=dict(title="Mean Predicted Probability", range=[0, 1], gridcolor="#1e2235"),
                    yaxis=dict(title="Fraction of Positives", range=[0, 1], gridcolor="#1e2235"),
                    height=400,
                )
                fig.update_layout(**{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                                     "font": {"color": Colors.TEXT_PRIMARY}})
                st.plotly_chart(fig, use_container_width=True)

    # ── Raw JSON viewer ──────────────────────────────────
    with st.expander("📋 Raw Report Data"):
        st.json(data)

elif selected_report["type"] == "csv":
    df = selected_report["df"]
    section_header("📊 CSV Report", "📊")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Visualize numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        str_cols = df.select_dtypes(include=["object"]).columns.tolist()
        x_col = str_cols[0] if str_cols else numeric_cols[0]
        section_header_sm("Column Distribution")
        sel_col = st.selectbox("Select column to visualize:", numeric_cols)
        fig = px.histogram(df, x=sel_col, nbins=30,
                          color_discrete_sequence=[Colors.PRIMARY],
                          title=f"{sel_col} Distribution")
        fig.update_layout(**{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                             "font": {"color": Colors.TEXT_PRIMARY}})
        st.plotly_chart(fig, use_container_width=True)


# ── Manual model diagnostic ────────────────────────────
section_header("🔬 Run Model Diagnostic", "🔬")

try:
    from src.app.utils import load_model, load_clean_data, run_model_diagnostic as run_diag

    model = load_model()
    data = load_clean_data()

    if model is not None and data is not None:
        if st.button("▶️ Run Diagnostic", type="primary", use_container_width=True):
            with st.spinner("Running model diagnostic on test data..."):
                diag = run_diag(model, data)
                if diag:
                    st.session_state["diagnostic"] = diag
                    st.success("✅ Diagnostic complete!")

        if "diagnostic" in st.session_state:
            diag = st.session_state["diagnostic"]
            if diag:
                cols = st.columns(4)
                metric_card(cols[0], f"{diag.get('accuracy', 0):.1%}", "Accuracy")
                metric_card(cols[1], f"{diag.get('improvement', 0):+.1%}", "vs Baseline", up=diag.get('improvement', 0) > 0)
                metric_card(cols[2], f"{diag.get('log_loss', 0):.4f}", "Log-Loss")
                metric_card(cols[3], str(diag.get('n_test', 0)), "Test Samples")

                if "confusion_matrix" in diag:
                    st.markdown("### Confusion Matrix")
                    labels = diag.get("class_labels", ["Away", "Draw", "Home"])
                    fig = confusion_matrix_heatmap(diag["confusion_matrix"], labels, title="Diagnostic Confusion Matrix", height=400)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            '<div style="background:#141824;border:1px solid #1e2235;border-radius:12px;'
            'padding:1.5rem;text-align:center">'
            '<div style="color:#9ca3af">Load a model and data to run diagnostics.</div>'
            '<div style="font-size:0.8rem;color:#6b7280;margin-top:0.3rem">'
            'Ensure a trained model exists in the models directory.</div></div>',
            unsafe_allow_html=True,
        )
except Exception:
    st.info(
        '<div style="background:#141824;border:1px solid #1e2235;border-radius:12px;'
        'padding:1.5rem">'
        '<div style="color:#9ca3af">Diagnostic module not available.</div>'
        '<div style="font-size:0.8rem;color:#6b7280;margin-top:0.3rem">'
        'Train a model and ensure dependencies are installed.</div></div>',
        unsafe_allow_html=True,
    )


render_footer()
