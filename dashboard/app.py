"""
Football Prediction Monitoring Dashboard — Main Entry Point.

Provides real-time monitoring of model performance, prediction history,
betting results, CLV tracking, and bankroll management.

Usage:
    streamlit run dashboard/app.py
    python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
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
    metric_card,
    status_badge,
    status_dot,
    info_row,
    area_trend_chart,
    gauge_chart,
    Colors,
)

st.set_page_config(
    page_title="Football Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme initialisation ───────────────────────────────
init_theme()

# ── Sidebar theme toggle ───────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    sidebar_theme_radio()
    st.markdown("---")

# ── Inject custom CSS ──────────────────────────────────
render_custom_css()


# ── Session state ──────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "data" not in st.session_state:
    st.session_state.data = None
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "pipeline_cache" not in st.session_state:
    st.session_state.pipeline_cache = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True


# ── Data helpers (cached) ──────────────────────────────

@st.cache_data(ttl=120)
def detect_data_pipeline_status() -> dict:
    """Use the data pipeline to assess the current dataset."""
    result: dict = {
        "found": False,
        "rows": 0,
        "completed": 0,
        "upcoming": 0,
        "teams": 0,
        "date_min": None,
        "date_max": None,
        "n_columns": 0,
        "pipeline_applied": False,
    }
    try:
        from src.services import load_and_prepare

        df = load_and_prepare(add_temporal=True)
        if df is not None and len(df) > 0:
            result["found"] = True
            result["rows"] = len(df)
            result["n_columns"] = len(df.columns)
            result["completed"] = int(df["result"].notna().sum())
            result["upcoming"] = int((df["result"].isna() | (df["result"] == "")).sum())
            result["pipeline_applied"] = True

            teams: set[str] = set()
            for col in ("home_team", "away_team"):
                if col in df.columns:
                    teams.update(df[col].dropna().unique())
            result["teams"] = len(teams)

            if "date" in df.columns:
                dates = df["date"].dropna()
                if len(dates) > 0:
                    result["date_min"] = str(dates.min())[:10]
                    result["date_max"] = str(dates.max())[:10]
    except Exception:
        pass
    return result


@st.cache_data(ttl=60)
def detect_data_file_count() -> dict[str, int]:
    """Count data files in standard directories (fallback when pipeline unavailable)."""
    counts: dict[str, int] = {}
    for d in [config.paths.raw, config.paths.processed, config.paths.data / "external"]:
        path = Path(d)
        if path.exists():
            counts[d.name] = len(list(path.glob("*.csv")))
        else:
            counts[d.name] = 0
    return counts


@st.cache_data(ttl=120)
def load_model_info() -> dict:
    """Check which model files exist and their details."""
    models_dir = config.paths.models
    info = {"total": 0, "found": [], "latest": None}
    key_models = [
        "three_model_blend.joblib",       # ⭐ Primary: 3-model blend (Poisson + Elo + XGBoost)
        "ensemble_model.joblib",
        "xgboost_model.joblib",
        "lightgbm_model",
        "worldcup_lightgbm.joblib",
        "worldcup_xgboost.joblib",
        "calibrated_xgboost.joblib",
        "calibrated_random_forest.joblib",
        "calibrated_lightgbm.joblib",
    ]
    for name in key_models:
        p = models_dir / name
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            size_mb = p.stat().st_size / (1024 * 1024)
            info["found"].append({"name": name, "mtime": mtime, "size_mb": size_mb})
    info["total"] = len(info["found"])
    if info["found"]:
        info["latest"] = max(info["found"], key=lambda x: x["mtime"])
        # Add a human-readable display name
        _display_names = {
            "three_model_blend.joblib": "3-Model Blend (Poisson+Elo+XGB)",
            "ensemble_model.joblib": "Ensemble (XGB+LR+Poisson)",
            "xgboost_model.joblib": "XGBoost",
            "lightgbm_model": "LightGBM",
            "worldcup_lightgbm.joblib": "LightGBM (World Cup)",
            "worldcup_xgboost.joblib": "XGBoost (World Cup)",
            "calibrated_xgboost.joblib": "XGBoost (Calibrated)",
            "calibrated_random_forest.joblib": "Random Forest (Calibrated)",
            "calibrated_lightgbm.joblib": "LightGBM (Calibrated)",
        }
        info["latest"]["display_name"] = _display_names.get(
            info["latest"]["name"], info["latest"]["name"]
        )
    return info


@st.cache_data(ttl=120)
def load_backtest_summary() -> dict | None:
    """Load the most recent backtest summary."""
    reports_dir = Path("reports")
    summaries = sorted(reports_dir.glob("backtest_summary_*.json"), reverse=True)
    if summaries:
        try:
            import json
            with open(summaries[0]) as f:
                return json.load(f)
        except Exception:
            pass
    return None


@st.cache_data(ttl=120)
def load_pipeline_log() -> list[dict]:
    """Parse the pipeline log for recent runs."""
    log_path = Path("logs/pipeline.log")
    entries = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for line in f.readlines()[-200:]:
                    if "PIPELINE" in line or "STEP" in line or "FAIL" in line or "COMPLETE" in line:
                        entries.append({"line": line.strip(), "ts": line[:19] if len(line) > 19 else ""})
        except Exception:
            pass
    return entries[-30:]  # last 30 relevant lines


# ── Auto-refresh logic ──────────────────────────────────
if st.session_state.auto_refresh:
    elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if elapsed > 120:
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ══════════════════════════════════════════════════════════

# ── Load model info early (needed by hero badge) ──────
model_info = load_model_info()
_model_badge = model_info["latest"]["display_name"] if model_info["latest"] else "No Model"

# ── Hero Section ───────────────────────────────────────
render_hero(
    title="📊 Football Monitoring Dashboard",
    subtitle="Monitor model performance, track predictions, analyse betting results, "
             "and manage bankroll risk — all in one place.",
    badges=[
        ("Auto-refresh: 120s", "🔄"),
        (_model_badge, "🤖"),
        ("v0.1.0", "📊"),
        (datetime.now().strftime("%B %d, %Y"), "📅"),
    ],
)


# ── System Health Section ──────────────────────────────
section_header("🔋 System Health", "🔋")
pipeline = detect_data_pipeline_status()
reports_dir_path = Path("reports")
report_count = len(list(reports_dir_path.rglob("*.json"))) if reports_dir_path.exists() else 0
prediction_count = len(list(Path("reports/predictions").glob("*.csv"))) if Path("reports/predictions").exists() else 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if model_info["latest"]:
        display = model_info["latest"].get("display_name", model_info["latest"]["name"][:20])
        metric_card(col1, str(model_info["total"]), "Trained Models", delta=f"Active: {display}", up=True)
    else:
        metric_card(col1, "0", "Trained Models", delta="No model found", up=False)

with col2:
    if pipeline["found"]:
        metric_card(
            col2, f"{pipeline['rows']:,}",
            "Match Records",
            delta=f"{pipeline['teams']} teams · {pipeline['completed']} completed",
            up=True,
        )
    else:
        file_counts = detect_data_file_count()
        total_files = sum(file_counts.values())
        metric_card(col2, str(total_files), "Data Files", delta="Pipeline inactive" if total_files == 0 else "Raw data", up=total_files > 0)

with col3:
    metric_card(col3, str(report_count), "Report Files", delta=".json reports", up=report_count > 0)

with col4:
    metric_card(col4, str(prediction_count), "Prediction CSVs", delta="saved to reports/predictions", up=prediction_count > 0)

with col5:
    metric_card(col5, config.train.model_type.upper(), "Active Config", delta="Default model type", up=True)


# ── Status indicators ──────────────────────────────────
st.markdown("---")
status_cols = st.columns(6)

with status_cols[0]:
    st.markdown("#### Status")
    st.markdown("")

with status_cols[1]:
    if model_info["total"] > 0:
        status_dot("green")
        st.markdown(f"**Models:** {model_info['total']} loaded")
    else:
        status_dot("red")
        st.markdown("**Models:** None")

with status_cols[2]:
    if pipeline["found"]:
        status_dot("green")
        st.markdown(f"**Data:** {pipeline['rows']:,} rows")
    else:
        status_dot("yellow")
        st.markdown("**Data:** Limited")

with status_cols[3]:
    has_reports = report_count > 0
    status_dot("green" if has_reports else "yellow")
    st.markdown(f"**Reports:** {report_count} files")

with status_cols[4]:
    pipeline_active = pipeline["pipeline_applied"]
    status_dot("green" if pipeline_active else "yellow")
    st.markdown(f"**Pipeline:** {'Active' if pipeline_active else 'Inactive'}")

with status_cols[5]:
    status_dot("blue")
    st.markdown(f"**Auto-refresh:** {'On' if st.session_state.auto_refresh else 'Off'}")


# ── Data Pipeline Status ─────────────────────────────
section_header("🔄 Data Pipeline Status", "🔄")

if pipeline["found"]:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(c1, f"{pipeline['rows']:,}", "Total Matches",
                    delta=f"{pipeline['completed']} completed · {pipeline['upcoming']} upcoming")

    with c2:
        metric_card(c2, str(pipeline["teams"]), "Unique Teams (normalised)",
                    delta=f"across dataset", up=True)

    with c3:
        metric_card(c3, str(pipeline["n_columns"]), "Feature Columns",
                    delta=f"engineered features", up=True)

    with c4:
        pip_status = "✅ Active" if pipeline["pipeline_applied"] else "⚠️ Raw"
        metric_card(c4, pip_status, "Data Pipeline",
                    delta=f"v2 pipeline", up=pipeline["pipeline_applied"])

    if pipeline["date_min"] and pipeline["date_max"]:
        info_row(
            f"📅 Date range: {pipeline['date_min']} → {pipeline['date_max']}  ·  "
            f"{pipeline['completed']:,} completed, {pipeline['upcoming']} upcoming matches"
        )

    if pipeline["upcoming"] > 0:
        status_badge(f"🔮 {pipeline['upcoming']} upcoming matches ready for prediction", "blue")
        st.markdown("")

    # Pipeline flow diagram (visual)
    st.markdown("### Pipeline Flow")
    flow_cols = st.columns(4)
    flow_steps = [
        ("📥", "Collect", "Data ingestion"),
        ("🧹", "Process", "Clean & normalise"),
        ("🤖", "Train", "Model training"),
        ("🔮", "Predict", "Generate predictions"),
    ]
    for i, (icon, title, desc) in enumerate(flow_steps):
        with flow_cols[i]:
            st.markdown(
                f'<div style="text-align:center;padding:1rem;background:var(--bg-card-from,#141824);'
                f'border:1px solid var(--border,#1e2235);border-radius:12px;">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="font-weight:600;color:var(--text-primary,#e0e0e0);margin-top:0.3rem">{title}</div>'
                f'<div style="font-size:0.75rem;color:var(--text-secondary,#6b7280)">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if i < len(flow_steps) - 1:
            st.markdown(
                f'<div style="text-align:center;padding-top:2rem;color:var(--border,#1e2235);font-size:1.5rem">→</div>',
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        '<div style="background:var(--bg-card-from,#141824);border:1px solid var(--border,#1e2235);border-radius:12px;'
        'padding:2rem;text-align:center">'
        '<div style="font-size:3rem;margin-bottom:0.5rem">📭</div>'
        '<div style="color:var(--text-secondary,#9ca3af);font-size:1rem">No match data found.</div>'
        '<div style="color:var(--text-secondary,#6b7280);font-size:0.85rem;margin-top:0.3rem">'
        'Run data collection or place a CSV file in the data directory to activate the pipeline.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    file_counts = detect_data_file_count()
    if any(file_counts.values()):
        info_row(f"📁 Raw: {file_counts.get('raw', 0)} · "
                 f"Processed: {file_counts.get('processed', 0)} · "
                 f"External: {file_counts.get('external', 0)}")


# ── Latest Backtest Summary ────────────────────────────
section_header("📈 Latest Backtest Snapshot", "📈")

backtest_summary = load_backtest_summary()
if backtest_summary:
    models_data = backtest_summary.get("models", [])
    if models_data:
        # Extract key metrics
        sorted_models = sorted(models_data, key=lambda m: m.get("sharpe_ratio", -999), reverse=True)
        top_5 = sorted_models[:5]

        st.markdown("#### Top Models by Sharpe Ratio")

        # Gauge for best model
        best = top_5[0]
        gcol1, gcol2 = st.columns([1, 3])
        with gcol1:
            best_sharpe = best.get("sharpe_ratio", 0)
            gauge_chart(
                gcol1, best.get("model_name", "Best Model"),
                best_sharpe, target=1.0,
                lower_better=False, height=200,
            )

        with gcol2:
            # Table of top models
            model_rows = []
            for m in top_5:
                model_rows.append({
                    "Model": m.get("model_name", "?"),
                    "Sharpe": f"{m.get('sharpe_ratio', 0):.2f}",
                    "ROI": f"{m.get('roi_pct', 0):+.2f}%",
                    "Win Rate": f"{m.get('win_rate_pct', 0):.1f}%",
                    "Brier": f"{m.get('brier', 0):.4f}",
                    "Profit": f"£{m.get('total_profit', 0):+,.2f}",
                })
            st.dataframe(
                pd.DataFrame(model_rows),
                use_container_width=True,
                hide_index=True,
            )

        # Chart: Sharpe by model
        chart_df = pd.DataFrame([
            {"Model": m.get("model_name", "?"), "Sharpe Ratio": m.get("sharpe_ratio", 0)}
            for m in sorted_models
        ])
        if not chart_df.empty:
            fig = go.Figure()
            colors = [Colors.SUCCESS if v >= 0 else Colors.DANGER for v in chart_df["Sharpe Ratio"]]
            fig.add_trace(go.Bar(
                x=chart_df["Model"], y=chart_df["Sharpe Ratio"],
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            ))
            fig.add_hline(y=1, line_dash="dash", line_color=Colors.WARNING, line_width=1,
                          annotation_text="Sharpe=1", annotation_font=dict(color=Colors.TEXT_SECONDARY, size=9))
            fig.add_hline(y=0, line_color=Colors.TEXT_MUTED, line_width=0.5)
            fig.update_layout(
                xaxis_tickangle=-45,
                height=250,
                **{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                   "font": {"color": Colors.TEXT_PRIMARY, "size": 10}},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        total_bets = sum(m.get("total_bets", 0) for m in sorted_models)
        avg_roi = sum(m.get("roi_pct", 0) for m in sorted_models) / len(sorted_models) if sorted_models else 0
        st.markdown(
            f'<div style="display:flex;gap:2rem;padding:0.5rem 0;color:var(--text-secondary,#9ca3af);font-size:0.85rem">'
            f'<span>📊 <b>{len(sorted_models)} models</b> backtested</span>'
            f'<span>🎯 <b>{total_bets} total bets</b> placed</span>'
            f'<span>📈 <b>Avg ROI: {avg_roi:+.2f}%</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Quick Pipeline Log ────────────────────────────────
section_header("📋 Recent Pipeline Activity", "📋")

pipeline_log = load_pipeline_log()
if pipeline_log:
    log_text = "\n".join(e["line"] for e in pipeline_log[-10:])
    st.markdown(
        f'<div style="background:var(--bg-app,#0a0d14);border:1px solid var(--border,#1e2235);border-radius:10px;'
        f'padding:0.8rem 1rem;font-family:monospace;font-size:0.75rem;color:var(--text-secondary,#9ca3af);'
        f'max-height:180px;overflow-y:auto;white-space:pre-wrap">'
        f'{log_text}'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    info_row("No pipeline logs found. Run the pipeline to generate activity.")


# ── Dashboard Pages Overview ──────────────────────────
section_header("📋 Dashboard Pages", "📋")
st.markdown("Navigate to any page from the sidebar to explore detailed insights.")

page_info = [
    ("🤖", "Model Performance", "Accuracy trends, confusion matrix, feature importance, per-class balance"),
    ("🔮", "Prediction History", "Historical predictions, match search, model comparison"),
    ("💰", "Betting Results", "P&L charts, win rates, ROI analysis, streak tracking"),
    ("🎯", "CLV Tracking", "Closing Line Value over time, by model, by market"),
    ("🏦", "Bankroll Monitoring", "Bankroll growth, drawdown, risk metrics, stake analysis"),
]

for icon, title, desc in page_info:
    cols = st.columns([1, 11])
    with cols[0]:
        st.markdown(f"<div style='font-size:2rem;text-align:center'>{icon}</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            f'<div style="background:var(--bg-card-from,#141824);border:1px solid var(--border,#1e2235);border-radius:12px;'
            f'padding:0.8rem 1.2rem;margin-bottom:0.5rem">'
            f'<div style="font-weight:600;color:var(--text-primary,#e0e0e0)">{title}</div>'
            f'<div style="font-size:0.85rem;color:var(--text-secondary,#6b7280)">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Footer ─────────────────────────────────────────────
render_footer()


# ── Auto-refresh trigger ──────────────────────────────
if st.session_state.auto_refresh:
    elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
    remaining = max(0, 120 - int(elapsed))
    st.markdown(
        f'<div style="text-align:center;color:var(--text-muted,#444);font-size:0.7rem;margin-top:1rem">'
        f'Auto-refresh in {remaining}s · '
        f'<a href="#" style="color:var(--primary,#4fc3f7);text-decoration:none" '
        f'onclick="window.location.reload()">Refresh now</a>'
        f'</div>',
        unsafe_allow_html=True,
    )
