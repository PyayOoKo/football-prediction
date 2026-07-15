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

# Data status
with col2:
    data_dirs = [config.paths.raw, config.paths.processed]
    data_files = []
    for d in data_dirs:
        if d.exists():
            data_files.extend(list(d.glob("*.csv")))
    data_found = len(data_files) > 0
    ds_icon = "✅" if data_found else "❌"
    ds_color = "#4caf50" if data_found else "#f44336"
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{ds_color}">{ds_icon}</div>'
        f'<div class="metric-label">Data Files ({len(data_files)})</div>'
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
