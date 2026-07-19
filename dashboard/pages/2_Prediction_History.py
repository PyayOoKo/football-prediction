"""
Prediction History Dashboard — view historical predictions, search by team,
compare files, analyse confidence distributions, and download filtered data.
Fully revamped with the dashboard components library.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

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
    info_row,
    comparison_bar_chart,
    get_plotly_layout,
    Colors,
)

st.set_page_config(page_title="Prediction History", page_icon="🔮", layout="wide")

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
    title="🔮 Prediction History",
    subtitle="Browse historical predictions, search by team or date, compare prediction "
             "files, and analyse confidence distributions across models.",
    badges=[("Interactive filters", "🔍"), ("Download CSV", "⬇")],
)


# ── Load prediction files ──────────────────────────────
@st.cache_data(ttl=60)
def load_predictions() -> list[dict]:
    """Load all available prediction CSV/JSON files."""
    results = []
    pred_dirs = [
        Path(config.worldcup.predictions_dir),
        Path("reports/predictions"),
        Path("reports"),
        Path(config.worldcup.data_path).parent,
    ]
    for d in pred_dirs:
        if not d.exists():
            continue
        for pattern in [
            "*prediction*.csv", "*prediction*.json", "worldcup_predictions*",
            "fixtures*.csv", "results.csv", "predictions_*.csv",
        ]:
            for f in sorted(d.glob(pattern), reverse=True):
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
    st.info(
        '<div style="background:var(--bg-card-from,#141824);border:1px solid var(--border,#1e2235);'
        'border-radius:12px;padding:2rem;text-align:center">'
        '<div style="font-size:3rem;margin-bottom:0.5rem">📭</div>'
        '<div style="color:var(--text-secondary,#9ca3af);font-size:1rem">'
        'No prediction files found.</div>'
        '<div style="color:var(--text-muted,#555);font-size:0.85rem;margin-top:0.3rem">'
        'Run training/prediction scripts first.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Overview metrics ───────────────────────────────────
section_header("📊 Overview", "📊")

total_preds = sum(p["size"] for p in predictions)
total_files = len(predictions)
all_data_cols = set()
for p in predictions:
    all_data_cols.update(p["columns"])

col1, col2, col3, col4 = st.columns(4)
metric_card(col1, f"{total_preds:,}", "Total Predictions",
            delta=f"Across {total_files} files", up=total_preds > 0)
metric_card(col2, str(total_files), "Prediction Files",
            delta="CSV / JSON sources", up=total_files > 0)

# Count unique teams across all files
all_teams_set: set[str] = set()
date_min, date_max = None, None
for p in predictions:
    df = p["data"]
    for col in df.columns:
        if re.search(r"team|opponent|opp", col, re.IGNORECASE):
            all_teams_set.update(str(v) for v in df[col].dropna().unique())
        if re.search(r"date|time", col, re.IGNORECASE):
            try:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(dates) > 0:
                    dmin = dates.min()
                    dmax = dates.max()
                    date_min = dmin if date_min is None else min(date_min, dmin)
                    date_max = dmax if date_max is None else max(date_max, dmax)
            except Exception:
                pass

metric_card(col3, str(len(all_teams_set)), "Unique Teams",
            delta="Across all files", up=len(all_teams_set) > 0)

if date_min and date_max:
    date_range_str = f"{date_min.strftime('%b %d')} → {date_max.strftime('%b %d, %Y')}"
    metric_card(col4, date_range_str, "Date Range", delta="Coverage period", up=True)
else:
    metric_card(col4, "N/A", "Date Range", delta="No dates found")


# File size comparison
section_header("📁 Prediction Files Comparison", "📁")

file_summary = pd.DataFrame([
    {"File": p["file"], "Rows": p["size"], "Columns": len(p["columns"]), "Type": p["type"]}
    for p in predictions
]).sort_values("Rows", ascending=False)

st.dataframe(file_summary, use_container_width=True, hide_index=True)

if len(file_summary) > 1:
    fig = comparison_bar_chart(
        file_summary, x_col="File", y_col="Rows",
        title="Prediction Files by Row Count",
        horizontal=True, height=min(250, 30 * len(file_summary)),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── File selector ──────────────────────────────────────
section_header("🔍 Prediction Explorer", "🔍")
file_names = [p["file"] for p in predictions]
selected_file = st.selectbox("Choose a prediction file to explore:", file_names)
pred_data = next(p for p in predictions if p["file"] == selected_file)
df = pred_data["data"].copy()

info_row(f"📄 **{pred_data['file']}** — {len(df)} rows × {len(df.columns)} cols · CSV")

# ── Column detection ──────────────────────────────────
team_cols = [c for c in df.columns if re.search(r"team|opponent|opp", c, re.IGNORECASE)]
date_cols = [c for c in df.columns if re.search(r"date|time", c, re.IGNORECASE)]
prob_cols = [c for c in df.columns if re.search(r"prob|confidence|prediction", c, re.IGNORECASE)]
result_cols = [c for c in df.columns if re.search(r"^result$|^prediction$|predicted_outcome|outcome", c, re.IGNORECASE)]
score_cols = [c for c in df.columns if re.search(r"score|goal", c, re.IGNORECASE) and c not in result_cols]

# ── Filters ────────────────────────────────────────────
st.markdown("### Filters")
fcol1, fcol2, fcol3, fcol4 = st.columns(4)

# Team filter
selected_team = "All"
if team_cols:
    all_teams = sorted(set(
        str(v) for col in team_cols
        for v in df[col].dropna().unique()
    ))
    with fcol1:
        selected_team = st.selectbox("Filter by team:", ["All"] + all_teams)

# Date filter
date_range = None
min_date, max_date = None, None
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
pred_filter = "All"
pred_col = None
for c in ["prediction", "predicted_outcome", "result"]:
    if c in df.columns:
        pred_col = c
        break
if pred_col:
    with fcol3:
        pred_options = ["All"] + sorted(str(v) for v in df[pred_col].dropna().unique())
        pred_filter = st.selectbox("Prediction outcome:", pred_options)

# Model filter
model_col = next((c for c in df.columns if "model" in c.lower()), None)
model_filter = "All"
if model_col:
    with fcol4:
        model_options = ["All"] + sorted(str(v) for v in df[model_col].dropna().unique())
        model_filter = st.selectbox("Model:", model_options)

# ── Apply filters ──────────────────────────────────────
filtered = df.copy()

if team_cols and selected_team != "All":
    mask = pd.Series(False, index=df.index)
    for col in team_cols:
        mask |= df[col].astype(str).str.contains(selected_team, case=False, na=False)
    filtered = filtered[mask]

if date_cols and pd.notna(min_date) and pd.notna(max_date) and date_range:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = filtered[
            (filtered[date_col].dt.date >= date_range[0]) &
            (filtered[date_col].dt.date <= date_range[1])
        ]

if pred_filter != "All" and pred_col:
    filtered = filtered[filtered[pred_col].astype(str) == pred_filter]

if model_filter != "All" and model_col:
    filtered = filtered[filtered[model_col].astype(str) == model_filter]

# ── Display results ───────────────────────────────────
section_header_sm(f"Results — {len(filtered)} rows (filtered from {len(df)})")

if len(filtered) > 0:
    st.dataframe(filtered, use_container_width=True, hide_index=True)
else:
    info_row("No results match the selected filters.")

c = Colors
layout = get_plotly_layout()

# ── Visualizations ─────────────────────────────────────
if len(filtered) > 0:
    # Prediction outcome breakdown
    if pred_col:
        section_header("📊 Prediction Outcome Breakdown", "📊")
        outcome_counts = filtered[pred_col].value_counts()
        
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            fig = go.Figure(data=[go.Pie(
                labels=outcome_counts.index.tolist(),
                values=outcome_counts.values.tolist(),
                hole=0.4,
                marker=dict(
                    colors=[c.PRIMARY, c.SUCCESS, c.WARNING, c.DANGER, c.ACCENT, c.INFO],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                textinfo="label+percent",
                textfont=dict(color=c.TEXT_PRIMARY, size=11),
            )])
            fig.update_layout(
                title=dict(text="Outcome Distribution", font=dict(color=c.TEXT_PRIMARY, size=13)),
                height=350,
                showlegend=False,
                **layout,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_bar:
            fig = go.Figure()
            max_count = outcome_counts.max()
            bar_colors = [
                f"rgba(79, 195, 247, {0.3 + 0.7 * v / max_count})"
                for v in outcome_counts.values
            ]
            fig.add_trace(go.Bar(
                x=outcome_counts.index.tolist(),
                y=outcome_counts.values.tolist(),
                marker=dict(color=bar_colors, line=dict(width=0)),
                hovertemplate="%{x}: %{y}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text="Outcome Counts", font=dict(color=c.TEXT_PRIMARY, size=13)),
                height=350,
                **layout,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Probability distribution
    if prob_cols:
        section_header("📈 Probability Distribution", "📈")
        probs_to_plot = [
            c for c in prob_cols
            if c in filtered.columns and filtered[c].dtype in ["float64", "int64"]
        ]
        
        if probs_to_plot:
            selected_prob = st.selectbox("Select probability column:", probs_to_plot)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered[selected_prob].dropna(),
                nbinsx=30,
                marker=dict(
                    color=c.PRIMARY,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                hovertemplate="Prob range: %{x}<br>Count: %{y}<extra></extra>",
                name=selected_prob,
            ))
            fig.update_layout(
                title=dict(text=f"{selected_prob} Distribution", font=dict(color=c.TEXT_PRIMARY, size=13)),
                xaxis=dict(title="Probability", range=[0, 1], gridcolor=c.GRID_COLOR),
                yaxis=dict(title="Count", gridcolor=c.GRID_COLOR),
                height=350,
                bargap=0.05,
                **layout,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Confidence summary
            probs_data = filtered[selected_prob].dropna()
            if len(probs_data) > 0:
                high_conf = (probs_data >= 0.7).sum()
                med_conf = ((probs_data >= 0.4) & (probs_data < 0.7)).sum()
                low_conf = (probs_data < 0.4).sum()
                
                conf_cols = st.columns(3)
                metric_card(conf_cols[0], f"{high_conf}", "High Confidence (≥70%)",
                           delta=f"{high_conf/len(probs_data):.1%} of total", up=high_conf/len(probs_data) > 0.3)
                metric_card(conf_cols[1], f"{med_conf}", "Medium Confidence (40-70%)",
                           delta=f"{med_conf/len(probs_data):.1%} of total")
                metric_card(conf_cols[2], f"{low_conf}", "Low Confidence (<40%)",
                           delta=f"{low_conf/len(probs_data):.1%} of total", up=low_conf/len(probs_data) < 0.3)

    # Ternary probability plot (if 3+ probability columns)
    if len(probs_to_plot) >= 3:
        section_header("🎯 Outcome Probability Triangle", "🎯")
        ternary_cols = probs_to_plot[:3]
        scatter_df = filtered[ternary_cols].dropna().copy()
        
        # Rename to standard H/D/A
        labels = ["Home", "Draw", "Away"]
        # Try to match columns to labels
        mapped = {}
        remaining = list(ternary_cols)
        for label in labels:
            for col in remaining:
                if label.lower() in col.lower() or col.lower() in label.lower():
                    mapped[label] = col
                    remaining.remove(col)
                    break
        if len(mapped) < 3:
            # Just use first 3 columns
            mapped = {labels[i]: ternary_cols[i] for i in range(3)}
        
        if len(scatter_df) > 0:
            fig = px.scatter_ternary(
                scatter_df,
                a=mapped["Home"], b=mapped["Draw"], c=mapped["Away"],
                title="Ternary Probability Plot — Each dot = one prediction",
                opacity=0.5,
                color_discrete_sequence=[c.ACCENT],
            )
            fig.update_traces(marker=dict(size=5, color=c.ACCENT, line=dict(width=0.5, color="rgba(255,255,255,0.3)")))
            fig.update_layout(
                height=500,
                ternary=dict(
                    bgcolor="rgba(0,0,0,0)",
                    aaxis=dict(title=mapped["Home"], gridcolor=c.GRID_COLOR),
                    baxis=dict(title=mapped["Draw"], gridcolor=c.GRID_COLOR),
                    caxis=dict(title=mapped["Away"], gridcolor=c.GRID_COLOR),
                ),
                **layout,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Corner analysis
            home_corner = (scatter_df[mapped["Home"]] >= 0.7).sum()
            draw_corner = (scatter_df[mapped["Draw"]] >= 0.7).sum()
            away_corner = (scatter_df[mapped["Away"]] >= 0.7).sum()
            uncertain = ((scatter_df[mapped["Home"]] < 0.5) & 
                        (scatter_df[mapped["Draw"]] < 0.5) & 
                        (scatter_df[mapped["Away"]] < 0.5)).sum()
            
            cc = st.columns(4)
            metric_card(cc[0], f"{home_corner}", "Confident Home (≥70%)",
                       delta=f"{home_corner/len(scatter_df):.1%}", up=True)
            metric_card(cc[1], f"{draw_corner}", "Confident Draw (≥70%)",
                       delta=f"{draw_corner/len(scatter_df):.1%}")
            metric_card(cc[2], f"{away_corner}", "Confident Away (≥70%)",
                       delta=f"{away_corner/len(scatter_df):.1%}")
            metric_card(cc[3], f"{uncertain}", "Uncertain (<50% all)",
                       delta=f"{uncertain/len(scatter_df):.1%}")

    # Team performance breakdown
    if team_cols and pred_col:
        section_header("🏆 Team Prediction Breakdown", "🏆")
        # Find home/away team columns
        home_col = next((c for c in team_cols if "home" in c.lower()), None)
        away_col = next((c for c in team_cols if "away" in c.lower()), None)
        
        if home_col and away_col and home_col in filtered.columns and away_col in filtered.columns:
            # Count predictions per team (home + away appearances)
            home_counts = Counter(filtered[home_col].dropna())
            away_counts = Counter(filtered[away_col].dropna())
            team_total = Counter()
            for team, count in home_counts.items():
                team_total[team] += count
            for team, count in away_counts.items():
                team_total[team] += count
            
            top_teams = team_total.most_common(15)
            if top_teams:
                team_df = pd.DataFrame(top_teams, columns=["Team", "Appearances"])
                
                col_t1, col_t2 = st.columns([1, 1])
                with col_t1:
                    st.dataframe(team_df, use_container_width=True, hide_index=True)
                with col_t2:
                    fig = comparison_bar_chart(
                        team_df, x_col="Team", y_col="Appearances",
                        title="Top 15 Teams by Prediction Count",
                        horizontal=True, height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Model comparison (if model column exists)
    if model_col and pred_col:
        section_header("🤖 Model Comparison", "🤖")
        model_counts = filtered[model_col].value_counts()
        if len(model_counts) > 1:
            fig = go.Figure()
            colors_list = [c.PRIMARY, c.SUCCESS, c.GRADIENT_GOLD, c.GRADIENT_PURPLE, c.DANGER]
            for i, (model_name, count) in enumerate(model_counts.items()):
                model_data = filtered[filtered[model_col] == model_name]
                if pred_col and model_data[pred_col].dtype in ("float64", "int64"):
                    fig.add_trace(go.Box(
                        y=model_data[pred_col].dropna(),
                        name=str(model_name),
                        marker_color=colors_list[i % len(colors_list)],
                        boxmean=True,
                    ))
            
            if len(fig.data) > 0:
                fig.update_layout(
                    title=dict(text="Prediction Distribution by Model", font=dict(color=c.TEXT_PRIMARY, size=13)),
                    yaxis=dict(title="Probability", gridcolor=c.GRID_COLOR),
                    height=350,
                    **layout,
                )
                st.plotly_chart(fig, use_container_width=True)


# ── Download ───────────────────────────────────────────
st.markdown("---")
col_dl1, col_dl2 = st.columns([1, 1])

with col_dl1:
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Filtered CSV",
        data=csv,
        file_name=f"predictions_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    json_bytes = filtered.to_json(orient="records", indent=2).encode("utf-8")
    st.download_button(
        "⬇ Download Filtered JSON",
        data=json_bytes,
        file_name=f"predictions_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )


render_footer()
