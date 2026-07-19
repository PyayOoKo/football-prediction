"""
CLV Tracking Dashboard — monitor Closing Line Value over time, by model, and by market
with rich Plotly visualizations and interactive filtering.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    info_row,
    area_trend_chart,
    comparison_bar_chart,
    Colors,
)

st.set_page_config(page_title="CLV Tracking", page_icon="🎯", layout="wide")

# ── Theme initialisation ───────────────────────────────
init_theme()

# ── Sidebar theme toggle ───────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    sidebar_theme_radio()
    st.markdown("---")

render_custom_css()

render_hero(
    title="🎯 Closing Line Value (CLV) Tracking",
    subtitle="Monitor CLV trends across models, markets, and time periods. "
             "CLV is a leading indicator of betting performance.",
    badges=[("Leading indicator", "📈"), ("Per-model analysis", "🎯")],
)


# ── Data helpers ──────────────────────────────────────
@st.cache_data(ttl=60)
def load_clv_data() -> list[dict]:
    """Load all CLV-related JSON reports."""
    results = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return results
    for pattern in [
        "clv_*.json", "clv_summary_*.json", "clv_comparison_*.csv",
        "clv_tracking_*.json", "clv_report*.json",
    ]:
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
    st.info(
        '<div style="background:#141824;border:1px solid #1e2235;border-radius:12px;'
        'padding:2rem;text-align:center">'
        '<div style="font-size:3rem;margin-bottom:0.5rem">📭</div>'
        '<div style="color:#9ca3af;font-size:1rem">No CLV reports found.</div>'
        '<div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">'
        'Run CLV calculation scripts first.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Extract all CLV records ──────────────────────────
all_clvs = []
for r in clv_reports:
    data = r["data"]
    if r["type"] == "json":
        if isinstance(data, dict):
            for key in ["clv_values", "clv_per_bet", "results", "models"]:
                items = data.get(key, [])
                if items:
                    for item in items:
                        if isinstance(item, dict):
                            all_clvs.append({
                                "file": r["file"],
                                "model": item.get("model", item.get("model_name", item.get("name", "Unknown"))),
                                "clv": float(item.get("clv", item.get("avg_clv", item.get("average_clv", 0)))),
                                "positive_clv_pct": float(item.get("positive_clv_pct", item.get("clv_gt_0_pct", 0))),
                                "clv_gt_5_pct": float(item.get("clv_gt_5_pct", 0)),
                                "bets": int(item.get("bets", item.get("n_bets", item.get("total_bets", 0)))),
                            })
                    break
            if not all_clvs:
                clv_val = data.get("avg_clv", data.get("clv", data.get("average_clv", None)))
                if clv_val is not None:
                    all_clvs.append({
                        "file": r["file"],
                        "model": data.get("model", data.get("model_name", "Aggregate")),
                        "clv": float(clv_val),
                        "positive_clv_pct": float(data.get("positive_clv_pct", data.get("clv_gt_0_pct", 0))),
                        "clv_gt_5_pct": float(data.get("clv_gt_5_pct", 0)),
                        "bets": int(data.get("bets", data.get("n_bets", data.get("total_bets", 0)))),
                    })
    elif r["type"] == "csv":
        df = r["df"]
        for _, row in df.iterrows():
            all_clvs.append({
                "file": r["file"],
                "model": str(row.get("model", row.get("model_name", row.get("name", "Unknown")))),
                "clv": float(row.get("avg_clv", row.get("clv", row.get("average_clv", 0)))),
                "positive_clv_pct": float(row.get("positive_clv_pct", row.get("clv_gt_0_pct", 0))),
                "clv_gt_5_pct": float(row.get("clv_gt_5_pct", 0)),
                "bets": int(row.get("bets", row.get("n_bets", row.get("total_bets", 0)))),
            })


# ── Aggregate CLV Summary ────────────────────────────
section_header("📊 Aggregate CLV Summary", "📊")

if all_clvs:
    clv_df = pd.DataFrame(all_clvs)

    avg_clv = clv_df["clv"].mean()
    max_clv = clv_df["clv"].max()
    min_clv = clv_df["clv"].min()
    avg_pos_pct = clv_df["positive_clv_pct"].mean()
    n_models = clv_df["model"].nunique()
    n_bets_total = clv_df["bets"].sum()

    col1, col2, col3, col4 = st.columns(4)
    metric_card(col1, f"{avg_clv:+.4%}", "Average CLV",
                delta=f"{'Positive' if avg_clv > 0 else 'Negative'} edge", up=avg_clv > 0)
    metric_card(col2, f"{avg_pos_pct:.1f}%", "Avg CLV > 0%",
                delta=f"{'Above' if avg_pos_pct > 50 else 'Below'} 50% benchmark", up=avg_pos_pct > 50)
    metric_card(col3, str(n_models), "Models Tracked",
                delta=f"Across {len(clv_reports)} report files", up=n_models > 0)
    metric_card(col4, f"{n_bets_total:,}", "Total Bets Tracked",
                delta="CLV sample size", up=n_bets_total > 100)

    # ── CLV by Model (Bar Chart) ─────────────────────────
    section_header("🎯 CLV by Model", "🎯")

    # Aggregate by model
    model_agg = clv_df.groupby("model").agg({
        "clv": "mean",
        "positive_clv_pct": "mean",
        "clv_gt_5_pct": "mean",
        "bets": "sum",
    }).reset_index().sort_values("clv", ascending=False)

    fig = go.Figure()
    colors = [Colors.SUCCESS if v >= 0 else Colors.DANGER for v in model_agg["clv"]]
    fig.add_trace(go.Bar(
        x=model_agg["model"], y=model_agg["clv"],
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{x}<br>CLV: %{y:+.4%}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=Colors.TEXT_MUTED, line_width=0.5)
    fig.update_layout(
        xaxis_tickangle=-45,
        height=350,
        **{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
           "font": {"color": Colors.TEXT_PRIMARY, "size": 10}},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── CLV > 0% Rate ──────────────────────────────────
    section_header("✅ Positive CLV Rate by Model", "✅")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=model_agg["model"], y=model_agg["positive_clv_pct"],
        marker=dict(
            color=model_agg["positive_clv_pct"],
            colorscale=[[0, Colors.DANGER], [0.5, Colors.WARNING], [1, Colors.SUCCESS]],
            line=dict(width=0),
        ),
        hovertemplate="%{x}<br>CLV > 0%%: %{y:.1f}%%<extra></extra>",
    ))
    fig2.add_hline(y=50, line_dash="dash", line_color=Colors.WARNING, line_width=1,
                   annotation_text="50% Benchmark")
    fig2.update_layout(
        xaxis_tickangle=-45, height=300,
        **{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
           "font": {"color": Colors.TEXT_PRIMARY, "size": 10}},
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── CLV vs Bets Scatter ────────────────────────────
    section_header("📊 CLV vs Sample Size", "📊")
    fig3 = px.scatter(
        model_agg, x="bets", y="clv",
        size="bets", color="clv",
        hover_name="model",
        color_continuous_scale=["#f44336", "#ffc107", "#4caf50"],
        title="CLV vs Number of Bets (larger = more reliable)",
        labels={"bets": "Number of Bets", "clv": "Average CLV"},
        size_max=50,
    )
    fig3.add_hline(y=0, line_dash="dash", line_color=Colors.TEXT_MUTED)
    fig3.update_layout(
        height=400,
        **{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
           "font": {"color": Colors.TEXT_PRIMARY}},
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── CLV Trend over Time (if tracking data available) ──
    tracking_reports = [r for r in clv_reports if "tracking" in r["file"]]
    if tracking_reports:
        section_header("📈 CLV Trend Over Time", "📈")
        # Parse timestamps from file names or data
        trend_data = []
        for r in tracking_reports:
            # Extract timestamp from filename like clv_tracking_YYYYMMDD_HHMMSS.json
            try:
                ts_str = r["file"].split("_")[-2] + "_" + r["file"].split("_")[-1].replace(".json", "")
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            except (ValueError, IndexError):
                ts = datetime.now()

            if r["type"] == "json":
                d = r["data"]
                if isinstance(d, dict):
                    clv_val = d.get("avg_clv", d.get("clv", None))
                    if clv_val is not None:
                        trend_data.append({"timestamp": ts, "clv": float(clv_val), "source": r["file"]})

        if trend_data:
            trend_df = pd.DataFrame(trend_data).sort_values("timestamp")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df["timestamp"], y=trend_df["clv"],
                mode="lines+markers",
                line=dict(color=Colors.PRIMARY, width=2),
                marker=dict(color=Colors.PRIMARY, size=6),
                fill="tozeroy",
                fillcolor="rgba(79, 195, 247, 0.08)",
                hovertemplate="%{x|%b %d, %H:%M}<br>CLV: %{y:+.4%}<extra></extra>",
                name="CLV Trend",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color=Colors.TEXT_MUTED, line_width=0.5)
            fig.update_layout(
                height=300,
                **{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                   "font": {"color": Colors.TEXT_PRIMARY}},
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Full CLV Table ──────────────────────────────────
    section_header("📋 Full CLV Table", "📋")
    st.dataframe(
        clv_df.style.format({
            "clv": "{:+.4%}",
            "positive_clv_pct": "{:.1f}%",
            "clv_gt_5_pct": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("No CLV data extracted from reports.")


# ── Individual Report Viewer ───────────────────────────
section_header("📋 Individual Report Viewer", "📋")
report_names = sorted(set(r["file"] for r in clv_reports))
sel_report = st.selectbox("Select a report to inspect:", report_names)
sel = next(r for r in clv_reports if r["file"] == sel_report)
with st.expander(f"📄 {sel['file']}"):
    if sel["type"] == "json":
        st.json(sel["data"])
    else:
        st.dataframe(sel["df"])


render_footer()
