"""
Prediction History Dashboard — view historical predictions, search by team,
and compare model outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from config import config

st.set_page_config(page_title="Prediction History", page_icon="🔮", layout="wide")

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

st.markdown("# 🔮 Prediction History")
st.markdown("Browse historical predictions, search by team or date, and compare model outputs.")


# ── Load prediction files ──────────────────────────────
@st.cache_data(ttl=60)
def load_predictions() -> list[dict]:
    """Load all available prediction CSV/JSON files."""
    results = []
    pred_dirs = [
        Path(config.worldcup.predictions_dir),
        Path("reports"),
        Path(config.worldcup.data_path).parent,
    ]
    for d in pred_dirs:
        if not d.exists():
            continue
        for pattern in ["*prediction*.csv", "*prediction*.json", "worldcup_predictions*",
                         "fixtures*.csv", "results.csv"]:
            for f in d.glob(pattern):
                try:
                    if f.suffix == ".csv":
                        df = pd.read_csv(f)
                        results.append({
                            "file": f.name,
                            "size": len(df),
                            "columns": list(df.columns),
                            "data": df,
                            "type": "csv",
                        })
                    elif f.suffix == ".json":
                        with open(f) as fh:
                            data = json.load(fh)
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        elif isinstance(data, dict):
                            rows = data.get("predictions", data.get("results", data.get("data", [data])))
                            df = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame([data])
                        results.append({
                            "file": f.name,
                            "size": len(df),
                            "columns": list(df.columns),
                            "data": df,
                            "type": "json",
                        })
                except Exception:
                    pass
    return results


predictions = load_predictions()

if not predictions:
    st.info("No prediction files found. Run training/prediction scripts first.")
    st.stop()


# ── Overview metrics ───────────────────────────────────
total_preds = sum(p["size"] for p in predictions)
st.markdown(f"**{total_preds:,} total predictions** loaded from **{len(predictions)} files**.")

# ── File selector ──────────────────────────────────────
st.markdown("### Select Prediction File")
file_names = [p["file"] for p in predictions]
selected_file = st.selectbox("Choose a file:", file_names)
pred_data = next(p for p in predictions if p["file"] == selected_file)
df = pred_data["data"]

st.markdown(f"**{pred_data['file']}** — {len(df)} rows × {len(df.columns)} cols")


# ── Filters ────────────────────────────────────────────
st.markdown("### Filters")
fcol1, fcol2, fcol3 = st.columns(3)

# Column detection
team_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["team", "opponent", "opp"])]
date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["date", "time", "day"])]
prob_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["prob", "confidence", "prediction"])]

# Team filter
if team_cols:
    all_teams = sorted(set(
        v for col in team_cols
        for v in df[col].dropna().unique()
    ))
    with fcol1:
        selected_team = st.selectbox("Filter by team:", ["All"] + all_teams)

# Date filter
if date_cols:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    if pd.notna(min_date) and pd.notna(max_date):
        with fcol2:
            date_range = st.date_input(
                "Date range:",
                value=(min_date.date() if hasattr(min_date, "date") else min_date,
                       max_date.date() if hasattr(max_date, "date") else max_date),
            )

# Prediction filter
pred_col = None
for c in ["prediction", "predicted_outcome", "result"]:
    if c in df.columns:
        pred_col = c
        break
if pred_col:
    with fcol3:
        pred_filter = st.selectbox("Prediction:", ["All"] + sorted(df[pred_col].dropna().unique().tolist()))
else:
    pred_filter = "All"

# ── Apply filters ──────────────────────────────────────
filtered = df.copy()

if team_cols and selected_team != "All":
    mask = pd.Series(False, index=df.index)
    for col in team_cols:
        mask |= df[col].astype(str).str.contains(selected_team, case=False, na=False)
    filtered = filtered[mask]

if date_cols and pd.notna(min_date) and pd.notna(max_date):
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = filtered[
            (filtered[date_col].dt.date >= date_range[0]) &
            (filtered[date_col].dt.date <= date_range[1])
        ]

if pred_filter != "All" and pred_col:
    filtered = filtered[filtered[pred_col] == pred_filter]


# ── Display table ──────────────────────────────────────
st.markdown(f"### Results — {len(filtered)} rows")
st.dataframe(filtered, use_container_width=True, hide_index=True)


# ── Visualization ──────────────────────────────────────
if prob_cols:
    st.markdown("### Probability Distribution")

    probs_to_plot = [c for c in prob_cols if c in filtered.columns and filtered[c].dtype in ["float64", "int64"]]
    if len(probs_to_plot) >= 1:
        fig = px.histogram(
            filtered, x=probs_to_plot[0],
            title=f"{probs_to_plot[0]} Distribution",
            color_discrete_sequence=["#7c3aed"],
            nbins=30,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(probs_to_plot) >= 3:
        st.markdown("### Outcome Probability Comparison")
        scatter_df = filtered[probs_to_plot[:3]].copy()
        scatter_df.columns = ["Away", "Draw", "Home"][:len(probs_to_plot)]
        fig = px.scatter_ternary(
            scatter_df, a="Home" if "Home" in scatter_df.columns else scatter_df.columns[0],
            b="Draw" if "Draw" in scatter_df.columns else scatter_df.columns[1],
            c="Away" if "Away" in scatter_df.columns else scatter_df.columns[2],
            title="Ternary Probability Plot",
            opacity=0.6,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"))
        fig.update_traces(marker=dict(size=6, color="#7c3aed"))
        st.plotly_chart(fig, use_container_width=True)


# ── Download ───────────────────────────────────────────
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download Filtered CSV", data=csv,
    file_name=f"predictions_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)
