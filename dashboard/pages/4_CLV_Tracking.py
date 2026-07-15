"""
CLV Tracking Dashboard — monitor Closing Line Value over time, by model, and by market.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CLV Tracking", page_icon="🎯", layout="wide")

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

st.markdown("# 🎯 Closing Line Value (CLV) Tracking")
st.markdown("Monitor CLV trends across models, markets, and time periods.")


# ── Load CLV data ─────────────────────────────────────
@st.cache_data(ttl=60)
def load_clv_data() -> list[dict]:
    """Load all CLV-related JSON reports."""
    results = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return results
    for pattern in ["clv_*.json", "clv_summary_*.json", "clv_comparison_*.csv",
                     "clv_tracking_*.json", "clv_report*.json"]:
        for f in sorted(reports_dir.glob(pattern), reverse=True):
            try:
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                    results.append({"file": f.name, "type": "csv", "data": df.to_dict(orient="records"), "df": df})
                else:
                    with open(f) as fh:
                        data = json.load(fh)
                    results.append({"file": f.name, "type": "json", "data": data})
            except Exception:
                pass
    return results


clv_reports = load_clv_data()

if not clv_reports:
    st.info("No CLV reports found. Run CLV calculation scripts first.")
    st.stop()


# ── Aggregate CLV metrics across all reports ──────────
st.markdown("### 📊 Aggregate CLV Summary")

all_clvs = []
for r in clv_reports:
    data = r["data"]
    if r["type"] == "json":
        # Try different nesting patterns
        if isinstance(data, dict):
            # Per-model CLV results
            for key in ["clv_values", "clv_per_bet", "results", "models"]:
                items = data.get(key, [])
                if items:
                    for item in items:
                        if isinstance(item, dict):
                            all_clvs.append({
                                "file": r["file"],
                                "model": item.get("model", item.get("model_name", item.get("name", "Unknown"))),
                                "clv": item.get("clv", item.get("avg_clv", item.get("average_clv", 0))),
                                "positive_clv_pct": item.get("positive_clv_pct", item.get("clv_gt_0_pct", 0)),
                                "clv_gt_5_pct": item.get("clv_gt_5_pct", item.get("clv_gt_5_pct", 0)),
                                "bets": item.get("bets", item.get("n_bets", item.get("total_bets", 0))),
                            })
                    break
            # Single CLV result
            if not all_clvs:
                clv_val = data.get("avg_clv", data.get("clv", data.get("average_clv", None)))
                if clv_val is not None:
                    all_clvs.append({
                        "file": r["file"],
                        "model": data.get("model", data.get("model_name", "Aggregate")),
                        "clv": clv_val,
                        "positive_clv_pct": data.get("positive_clv_pct", data.get("clv_gt_0_pct", 0)),
                        "clv_gt_5_pct": data.get("clv_gt_5_pct", 0),
                        "bets": data.get("bets", data.get("n_bets", data.get("total_bets", 0))),
                    })
    elif r["type"] == "csv":
        df = r["df"]
        for _, row in df.iterrows():
            all_clvs.append({
                "file": r["file"],
                "model": row.get("model", row.get("model_name", row.get("name", "Unknown"))),
                "clv": row.get("avg_clv", row.get("clv", row.get("average_clv", 0))),
                "positive_clv_pct": row.get("positive_clv_pct", row.get("clv_gt_0_pct", 0)),
                "clv_gt_5_pct": row.get("clv_gt_5_pct", 0),
                "bets": row.get("bets", row.get("n_bets", row.get("total_bets", 0))),
            })

if all_clvs:
    clv_df = pd.DataFrame(all_clvs)

    # Summary metrics
    avg_clv = clv_df["clv"].mean()
    max_clv = clv_df["clv"].max()
    min_clv = clv_df["clv"].min()
    avg_pos_pct = clv_df["positive_clv_pct"].mean()
    n_models = clv_df["model"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:{"#4caf50" if avg_clv>0 else "#f44336"}">'
        f'{avg_clv:+.4%}</div><div class="metric-label">Avg CLV</div></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:{"#4caf50" if avg_pos_pct>50 else "#ffc107"}">'
        f'{avg_pos_pct:.1f}%</div><div class="metric-label">Avg CLV > 0%</div></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="metric-card"><div class="metric-value">{n_models}</div>'
        f'<div class="metric-label">Models Tracked</div></div>',
        unsafe_allow_html=True,
    )
    col4.markdown(
        f'<div class="metric-card"><div class="metric-value">{len(clv_reports)}</div>'
        f'<div class="metric-label">Report Files</div></div>',
        unsafe_allow_html=True,
    )

    # ── CLV by model bar chart ─────────────────────────
    st.markdown("### CLV by Model")
    fig = px.bar(
        clv_df.sort_values("clv", ascending=False),
        x="model", y="clv",
        color="clv",
        color_continuous_scale=["#f44336", "#ffc107", "#4caf50"],
        title="Average CLV per Model",
        labels={"model": "Model", "clv": "CLV"},
        hover_data=["positive_clv_pct", "bets"],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#666", line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── CLV > 0% comparison ───────────────────────────
    st.markdown("### CLV > 0% Rate by Model")
    fig2 = px.bar(
        clv_df.sort_values("positive_clv_pct", ascending=False),
        x="model", y="positive_clv_pct",
        color="positive_clv_pct",
        color_continuous_scale="Blues",
        title="% of Bets with Positive CLV",
        labels={"model": "Model", "positive_clv_pct": "CLV > 0%"},
    )
    fig2.add_hline(y=50, line_dash="dash", line_color="#ffc107", line_width=1,
                   annotation_text="50% Benchmark")
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── CLV vs bets scatter ────────────────────────────
    st.markdown("### CLV vs Number of Bets")
    fig3 = px.scatter(
        clv_df, x="bets", y="clv",
        size="bets", color="clv",
        hover_name="model",
        color_continuous_scale="RdYlGn",
        title="CLV vs Sample Size",
        labels={"bets": "Number of Bets", "clv": "Average CLV"},
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="#666")
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Detail table ───────────────────────────────────
    st.markdown("### Full CLV Table")
    st.dataframe(clv_df.style.format({
        "clv": "{:+.4%}",
        "positive_clv_pct": "{:.1f}%",
        "clv_gt_5_pct": "{:.1f}%",
    }), use_container_width=True, hide_index=True)

else:
    st.info("No CLV data extracted from reports.")


# ── Individual report viewer ───────────────────────────
st.markdown("---")
st.markdown("### 📋 Individual Report Viewer")
report_names = [r["file"] for r in clv_reports]
sel_report = st.selectbox("Select a report to inspect:", report_names)
sel = next(r for r in clv_reports if r["file"] == sel_report)
with st.expander(f"📄 {sel['file']}"):
    if sel["type"] == "json":
        st.json(sel["data"])
    else:
        st.dataframe(sel["df"])
