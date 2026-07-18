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
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st

from config import config

st.set_page_config(
    page_title="Football Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .stApp header { background: #1a1d27; }
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
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #fff; margin: 0; }
    .metric-label { font-size: 0.8rem; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.05em; }
    .hero {
        background: linear-gradient(135deg, #1a1d27 0%, #16213e 50%, #1a1d27 100%);
        border: 1px solid #2a2d3a;
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #4fc3f7, #81c784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { color: #8b8fa3; font-size: 1rem; margin: 0; }
    .badge-green { background: #1b5e20; color: #81c784; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .badge-red { background: #b71c1c; color: #ef9a9a; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .badge-blue { background: #0d47a1; color: #90caf9; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "data" not in st.session_state:
    st.session_state.data = None
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "pipeline_cache" not in st.session_state:
    st.session_state.pipeline_cache = None


# ── Data helpers (cached) ──────────────────────────────

@st.cache_data(ttl=120)
def detect_data_pipeline_status() -> dict[str, Any]:
    """Use the data pipeline to assess the current dataset.

    Returns a dict with: ``found``, ``rows``, ``completed``,
    ``upcoming``, ``teams``, ``date_min``, ``date_max``,
    ``n_columns``, ``pipeline_applied``.
    """
    result: dict[str, Any] = {
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

            # Team count
            teams: set[str] = set()
            for col in ("home_team", "away_team"):
                if col in df.columns:
                    teams.update(df[col].dropna().unique())
            result["teams"] = len(teams)

            # Date range
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


# ── Hero Section ───────────────────────────────────────
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("<h1>📊 Football Monitoring Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Monitor model performance, track predictions, analyse betting results, "
    "and manage bankroll risk — all in one place.</p>",
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# ── System Status ──────────────────────────────────────
st.markdown("## 🔌 System Status")
col1, col2, col3, col4 = st.columns(4)

# Model status
with col1:
    model_paths = [
        config.paths.models / "ensemble.pkl",
        config.paths.models / "xgboost_model.pkl",
        config.paths.models / "model.pkl",
    ]
    model_found = any(p.exists() for p in model_paths)
    status_icon = "✅" if model_found else "❌"
    status_color = "#4caf50" if model_found else "#f44336"
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{status_color}">{status_icon}</div>'
        f'<div class="metric-label">Trained Model</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Data status (via pipeline)
with col2:
    pipeline = detect_data_pipeline_status()
    if pipeline["found"]:
        ds_icon = "✅"
        ds_color = "#4caf50"
        ds_info = (
            f"{pipeline['rows']:,} rows · {pipeline['teams']} teams · "
            f"{pipeline['completed']} completed"
        )
        if pipeline["upcoming"] > 0:
            ds_info += f" · {pipeline['upcoming']} upcoming"
        if pipeline["pipeline_applied"]:
            ds_info += " · 🌀 Pipeline Active"
    else:
        # Fallback: count files
        file_counts = detect_data_file_count()
        total_files = sum(file_counts.values())
        ds_icon = "⚠️" if total_files > 0 else "❌"
        ds_color = "#ffc107" if total_files > 0 else "#f44336"
        ds_info = f"{total_files} CSV files (pipeline inactive)"
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{ds_color}">{ds_icon}</div>'
        f'<div class="metric-label">{ds_info}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Reports status
with col3:
    reports_dir = Path("reports")
    report_count = len(list(reports_dir.rglob("*.json"))) if reports_dir.exists() else 0
    rc_color = "#4caf50" if report_count > 0 else "#ffc107"
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{rc_color}">{report_count}</div>'
        f'<div class="metric-label">Report Files</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Config
with col4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{config.train.model_type.upper()}</div>'
        f'<div class="metric-label">Active Config</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Data Pipeline Status ─────────────────────────────
st.markdown("## 🔄 Data Pipeline Status")
pipeline = detect_data_pipeline_status()
if pipeline["found"]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{pipeline["rows"]:,}</div>'
            f'<div class="metric-label">Total Matches</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:#81c784">{pipeline["teams"]}</div>'
            f'<div class="metric-label">Unique Teams (normalised)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:#4fc3f7">{pipeline["n_columns"]}</div>'
            f'<div class="metric-label">Feature Columns</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c4:
        pip_label = "✅ Active" if pipeline["pipeline_applied"] else "⚠️ Raw"
        pip_color = "#4caf50" if pipeline["pipeline_applied"] else "#ffc107"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{pip_color}">{pip_label}</div>'
            f'<div class="metric-label">Data Pipeline</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if pipeline["date_min"] and pipeline["date_max"]:
        st.markdown(
            f"<p style='color:#8b8fa3;font-size:0.85rem;margin-top:-0.5rem'>"
            f"📅 Date range: {pipeline['date_min']} → {pipeline['date_max']}  ·  "
            f"{pipeline['completed']:,} completed, {pipeline['upcoming']} upcoming matches</p>",
            unsafe_allow_html=True,
        )

    if pipeline["upcoming"] > 0:
        st.markdown(
            f"<span class='badge-blue'>🔮 {pipeline['upcoming']} upcoming matches ready for prediction</span>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        "<p style='color:#8b8fa3;font-size:0.9rem'>⚠️ No match data found. "
        "Run data collection or place a CSV file in the data directory to activate the pipeline.</p>",
        unsafe_allow_html=True,
    )
    # Show fallback file counts
    file_counts = detect_data_file_count()
    if any(file_counts.values()):
        st.markdown(
            f"<p style='color:#8b8fa3;font-size:0.8rem'>"
            f"📁 Raw: {file_counts.get('raw', 0)} · "
            f"Processed: {file_counts.get('processed', 0)} · "
            f"External: {file_counts.get('external', 0)}</p>",
            unsafe_allow_html=True,
        )


# ── Quick overview cards ───────────────────────────────
st.markdown("## 📋 Dashboard Pages")
st.markdown("Navigate to any page from the sidebar to explore detailed insights.")

page_info = [
    ("🤖 Model Performance", "Accuracy trends, confusion matrix, feature importance, per-class balance"),
    ("🔮 Prediction History", "Historical predictions, match search, model comparison"),
    ("💰 Betting Results", "P&L charts, win rates, ROI analysis, streak tracking"),
    ("🎯 CLV Tracking", "Closing Line Value over time, by model, by market"),
    ("🏦 Bankroll Monitoring", "Bankroll growth, drawdown, risk metrics, stake analysis"),
]

for icon_title, desc in page_info:
    c = st.columns([1, 5])
    with c[0]:
        st.markdown(f"### {icon_title.split(' ')[0]}")
    with c[1]:
        st.markdown(f"**{icon_title.split(' ', 1)[1]}**")
        st.markdown(f"<span style='color:#8b8fa3;font-size:0.9rem'>{desc}</span>", unsafe_allow_html=True)
    st.markdown("---")


# ── Footer ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem'>"
    "Football Monitoring Dashboard | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
