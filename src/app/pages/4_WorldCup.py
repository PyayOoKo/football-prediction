"""
World Cup 2026 Dashboard — Bracket Tree, Probability Bars,
Poisson Scoreline Distributions & Confidence Trends.

Run with:
    streamlit run src/app/dashboard.py
    → then navigate to the "🏆 World Cup 2026" page
"""

from __future__ import annotations

from math import exp, factorial
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="World Cup 2026",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PREDICTIONS_CSV = PROJECT_ROOT / "reports" / "predictions_worldcup" / "worldcup_predictions.csv"
DATA_CSV = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"


# ═══════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .stApp header { background: #1a1d27; }

    /* ── Cards ── */
    .wc-card {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .wc-card h3 {
        margin: 0 0 0.5rem 0;
        color: #ffffff;
    }

    /* ── Bracket ── */
    .bracket-container {
        overflow-x: auto;
        padding: 1rem 0;
    }
    .bracket-match {
        background: linear-gradient(135deg, #1a1d27, #222639);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        min-width: 200px;
        font-size: 0.85rem;
    }
    .bracket-match .teams {
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 0.2rem;
    }
    .bracket-match .prob {
        font-size: 0.75rem;
        color: #8b8fa3;
    }
    .bracket-match .pred-home { border-left: 3px solid #4caf50; }
    .bracket-match .pred-away { border-left: 3px solid #f44336; }
    .bracket-match .pred-draw  { border-left: 3px solid #ffc107; }
    .bracket-match .pred-tbd   { border-left: 3px solid #555; opacity: 0.6; }
    .bracket-round-label {
        color: #8b8fa3;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .bracket-winner {
        color: #4fc3f7;
        font-weight: 700;
        font-size: 0.8rem;
    }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #1a1d27 0%, #16213e 50%, #1a1d27 100%);
        border: 1px solid #2a2d3a;
        border-radius: 16px;
        padding: 2rem 2.5rem;
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
    .hero p {
        color: #8b8fa3;
        font-size: 1rem;
        margin: 0;
    }

    /* ── Metric tiles ── */
    .metric-tile {
        background: linear-gradient(135deg, #1a1d27, #222639);
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-tile .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-tile .label {
        font-size: 0.75rem;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading predictions …")
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_CSV)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner="Loading World Cup data …")
def load_wc_data() -> pd.DataFrame:
    if not DATA_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_CSV, low_memory=False, parse_dates=["date"])
    return df


@st.cache_resource(show_spinner="Fitting Poisson model …")
def fit_poisson_model(df_completed: pd.DataFrame):
    from src.poisson_model import PoissonModel

    poisson = PoissonModel(min_matches=0, max_goals=8)
    poisson.add_poisson_features(df_completed.copy())
    return poisson


# ── Load ──
preds = load_predictions()
data = load_wc_data()

if preds.empty:
    st.error("⚠ No predictions found. Run `python train_worldcup.py` first.")
    st.stop()


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (exp(-lam) * (lam ** k)) / factorial(k)


def get_scoreline_probs(
    lam_h: float, lam_a: float, max_g: int = 5,
) -> list[dict]:
    """Return list of {score, prob} for most likely scorelines."""
    rows = []
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a)
            rows.append({"score": f"{i}-{j}", "home": i, "away": j, "prob": p})
    df = pd.DataFrame(rows)
    total = df["prob"].sum()
    if total > 0:
        df["prob"] /= total
    df.sort_values("prob", ascending=False, inplace=True)
    return df.head(8).to_dict("records")


# ═══════════════════════════════════════════════════════════
#  HERO SECTION
# ═══════════════════════════════════════════════════════════

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown(
    "<h1>🏆 World Cup 2026 — Knockout Stage Predictions</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p>AI-powered predictions using XGBoost + Poisson models, trained on "
    "658 completed international matches (7 World Cups + Euro, Copa America, AFCON). "
    "Probabilities are swap-averaged for neutral venue fairness.</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Filter toggle ────────────────────────────────────────
show_only_high_conf = st.checkbox(
    "🔍 Show only high confidence matches (≥45%)", value=False,
    help="Filter the visualizations below to only show matches where the model has at least 45% confidence.",
)

all_matches = preds.sort_values("date").reset_index(drop=True)
if show_only_high_conf:
    visible_matches = all_matches[all_matches["confidence"] >= 0.45].copy()
    if visible_matches.empty:
        st.info("No matches with ≥45% confidence. Showing all matches instead.")
        visible_matches = all_matches.copy()
else:
    visible_matches = all_matches.copy()

# ── Summary metrics row ─────────────────────────────────
avg_conf = preds["confidence"].mean()
n_home = (preds["prediction"] == "Home Win").sum()
n_away = (preds["prediction"] == "Away Win").sum()
n_draw = (preds["prediction"] == "Draw").sum()

cols = st.columns(5)
with cols[0]:
    st.markdown(f'<div class="metric-tile"><div class="value">{len(preds)}</div><div class="label">R16 Matches</div></div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown(f'<div class="metric-tile"><div class="value">{avg_conf:.0%}</div><div class="label">Avg Confidence</div></div>', unsafe_allow_html=True)
with cols[2]:
    st.markdown(f'<div class="metric-tile"><div class="value" style="color:#4caf50">{n_home}</div><div class="label">Home Wins</div></div>', unsafe_allow_html=True)
with cols[3]:
    st.markdown(f'<div class="metric-tile"><div class="value" style="color:#f44336">{n_away}</div><div class="label">Away Wins</div></div>', unsafe_allow_html=True)
with cols[4]:
    st.markdown(f'<div class="metric-tile"><div class="value" style="color:#ffc107">{n_draw}</div><div class="label">Draws</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SECTION 1: BRACKET TREE
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🏆 Knockout Bracket")

st.markdown(
    "<p style='color:#8b8fa3'>R16 matchups with predictions. "
    "QF/SF/Final slots auto-populate as matches are played and the pipeline refreshes.</p>",
    unsafe_allow_html=True,
)

# Color class per match
def match_css_class(row):
    pred = str(row.get("prediction", ""))
    if "Home" in pred or row.get("home_win_prob", 0) > row.get("away_win_prob", 0):
        return "pred-home"
    elif "Away" in pred:
        return "pred-away"
    return "pred-draw"

def match_html(row):
    css_class = match_css_class(row)
    home = row["home_team"]
    away = row["away_team"]
    hw = row.get("home_win_prob", 0) * 100
    aw = row.get("away_win_prob", 0) * 100
    dr = row.get("draw_prob", 0) * 100

    # Determine which team is predicted to advance
    if hw >= aw:
        winner = f"<div class='bracket-winner'>⬆ {home}</div>"
    else:
        winner = f"<div class='bracket-winner'>⬆ {away}</div>"

    return f"""
    <div class="bracket-match {css_class}">
        <div class="teams">{home} vs {away}</div>
        <div class="prob">{home}: {hw:.0f}%  |  Draw: {dr:.0f}%  |  {away}: {aw:.0f}%</div>
        {winner}
    </div>
    """

# Helper for QF/SF placeholder slots
def tbd_html(label="QF"):
    return f"""
    <div class="bracket-match pred-tbd">
        <div class="teams" style="color:#666">TBD vs TBD</div>
        <div class="prob" style="color:#555">Awaiting R16 results</div>
    </div>
    """

# Assign matches to bracket positions
all_matches_list = [visible_matches.iloc[i] for i in range(len(visible_matches))]
left_half = all_matches_list[:4] if len(all_matches_list) >= 4 else all_matches_list
right_half = all_matches_list[4:8] if len(all_matches_list) >= 8 else []

# Helper to build QF projected matchups from R16 predictions
def _qf_html(m1, m2):
    w1 = m1["home_team"] if m1.get("home_win_prob", 0) >= m1.get("away_win_prob", 0) else m1["away_team"]
    w2 = m2["home_team"] if m2.get("home_win_prob", 0) >= m2.get("away_win_prob", 0) else m2["away_team"]
    return f"""
    <div class="bracket-match pred-tbd">
        <div class="teams" style="color:#aaa">{w1} vs {w2}</div>
        <div class="prob" style="color:#777">Projected QF matchup</div>
    </div>
    """

# Build a single-table bracket

bracket_html = '<div class="bracket-container"><table style="width:100%;border-collapse:collapse;"><tr>'

# ── Left half ──
# Round of 16
bracket_html += '<td style="vertical-align:top;padding:0.5rem;width:15%;"><div class="bracket-round-label">Round of 16</div>'
for m in left_half:
    bracket_html += match_html(m)
bracket_html += '</td>'

bracket_html += '<td style="vertical-align:middle;width:3%;text-align:center;color:#333;font-size:1.3rem;">→</td>'

# Quarter-Finals
bracket_html += '<td style="vertical-align:top;padding:0.5rem;width:15%;"><div class="bracket-round-label">Quarter-Finals</div>'
for i in range(0, len(left_half), 2):
    if i + 1 < len(left_half):
        bracket_html += _qf_html(left_half[i], left_half[i+1])
    else:
        bracket_html += tbd_html('QF')
bracket_html += '</td>'

bracket_html += '<td style="vertical-align:middle;width:3%;text-align:center;color:#333;font-size:1.3rem;">→</td>'

# Semi-Finals
bracket_html += '<td style="vertical-align:top;padding:0.5rem;width:15%;"><div class="bracket-round-label">Semi-Finals</div>'
bracket_html += tbd_html('SF')
bracket_html += tbd_html('SF')
bracket_html += '</td>'

bracket_html += '<td style="vertical-align:middle;width:3%;text-align:center;color:#333;font-size:1.3rem;">→</td>'

# Final
bracket_html += '<td style="vertical-align:top;padding:0.5rem;width:15%;"><div class="bracket-round-label">Final</div>'
bracket_html += tbd_html('Final')
bracket_html += '</td>'

# ── Gap between left and right halves ──
bracket_html += '<td style="width:3%;"></td>'

# ── Right half ──
if right_half:
    bracket_html += '<td style="vertical-align:top;padding:0.5rem;width:15%;"><div class="bracket-round-label">Round of 16</div>'
    for m in right_half:
        bracket_html += match_html(m)
    bracket_html += '</td>'

    bracket_html += '<td style="vertical-align:middle;width:3%;text-align:center;color:#333;font-size:1.3rem;">→</td>'

    bracket_html += '<td style="vertical-align:top;padding:0.5rem;width:15%;"><div class="bracket-round-label">Quarter-Finals</div>'
    for i in range(0, len(right_half), 2):
        if i + 1 < len(right_half):
            bracket_html += _qf_html(right_half[i], right_half[i+1])
        else:
            bracket_html += tbd_html('QF')
    bracket_html += '</td>'

    bracket_html += '<td style="vertical-align:middle;width:3%;text-align:center;color:#333;font-size:1.3rem;">→</td>'

    bracket_html += '<td style="vertical-align:top;padding:0.5rem;width:15%;"><div class="bracket-round-label">Semi-Finals</div>'
    bracket_html += tbd_html('SF')
    bracket_html += tbd_html('SF')
    bracket_html += '</td>'

    bracket_html += '<td style="vertical-align:middle;width:3%;text-align:center;color:#333;font-size:1.3rem;">→</td>'

    bracket_html += '<td style="vertical-align:top;padding:0.5rem;width:15%;"><div class="bracket-round-label">Final</div>'
    bracket_html += tbd_html('Final')
    bracket_html += '</td>'

bracket_html += '</tr></table></div>'

st.markdown(bracket_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SECTION 2: MATCH-BY-MATCH PROBABILITY BARS
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📊 Match-by-Match Probabilities")

st.markdown(
    "<p style='color:#8b8fa3'>Horizontal stacked bars showing the probability distribution "
    "for each Round of 16 matchup. Swap-averaged to remove neutral venue bias.</p>",
    unsafe_allow_html=True,
)

for _, row in visible_matches.iterrows():
    date_str = str(row["date"])[:10] if pd.notna(row.get("date")) else ""
    match_label = f"{row['home_team']} vs {row['away_team']}"
    hw = row["home_win_prob"]
    dr = row["draw_prob"]
    aw = row["away_win_prob"]

    # Determine predicted outcome
    probs = {"🏠 " + row['home_team']: hw, "🤝 Draw": dr, "✈️ " + row['away_team']: aw}
    pred_outcome = max(probs, key=probs.get)

    st.markdown(f'<div class="wc-card">', unsafe_allow_html=True)

    cols = st.columns([1.5, 0.5, 5])
    with cols[0]:
        st.markdown(f"**{row['home_team']}**", unsafe_allow_html=True)
        st.markdown(
            f"<span style='color:#8b8fa3;font-size:0.8rem'>vs {row['away_team']}</span>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(f"<span style='font-size:0.75rem;color:#555'>{date_str}</span>", unsafe_allow_html=True)

    with cols[2]:
        # Three progress bars
        st.markdown(
            f"<div style='display:flex;align-items:center;margin-bottom:2px'>"
            f"<span style='color:#4caf50;width:8rem;font-size:0.8rem'>🏠 {row['home_team']}</span>"
            f"<div style='flex:1;height:20px;background:#1a1d27;border-radius:10px;margin:0 0.5rem'>"
            f"<div style='height:100%;width:{hw*100:.1f}%;background:linear-gradient(90deg,#2e7d32,#4caf50);"
            f"border-radius:10px;transition:width 1s'></div></div>"
            f"<span style='color:#fff;width:3.5rem;text-align:right;font-weight:600'>{hw:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex;align-items:center;margin-bottom:2px'>"
            f"<span style='color:#ffc107;width:8rem;font-size:0.8rem'>🤝 Draw</span>"
            f"<div style='flex:1;height:20px;background:#1a1d27;border-radius:10px;margin:0 0.5rem'>"
            f"<div style='height:100%;width:{dr*100:.1f}%;background:linear-gradient(90deg,#f57f17,#ffc107);"
            f"border-radius:10px;transition:width 1s'></div></div>"
            f"<span style='color:#fff;width:3.5rem;text-align:right;font-weight:600'>{dr:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex;align-items:center;margin-bottom:2px'>"
            f"<span style='color:#f44336;width:8rem;font-size:0.8rem'>✈️ {row['away_team']}</span>"
            f"<div style='flex:1;height:20px;background:#1a1d27;border-radius:10px;margin:0 0.5rem'>"
            f"<div style='height:100%;width:{aw*100:.1f}%;background:linear-gradient(90deg,#c62828,#f44336);"
            f"border-radius:10px;transition:width 1s'></div></div>"
            f"<span style='color:#fff;width:3.5rem;text-align:right;font-weight:600'>{aw:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Prediction badge
    st.markdown(
        f"<div style='text-align:right;color:#8b8fa3;font-size:0.8rem'>"
        f"Predicted: <strong style='color:#fff'>{pred_outcome}</strong> "
        f"(conf: {row['confidence']:.0%})</div>",
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SECTION 3: POISSON SCORELINE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## ⚽ Poisson Scoreline Distributions")

st.markdown(
    "<p style='color:#8b8fa3'>For each match, the Poisson model (fitted on 658 completed matches) "
    "generates the probability of every possible scoreline. Below are the top contenders per matchup.</p>",
    unsafe_allow_html=True,
)

# Fit Poisson model on completed data
if not data.empty:
    completed = data[data["result"].notna()].copy()
    try:
        poisson = fit_poisson_model(completed)

        # For each match, compute scoreline probabilities
        for _, row in visible_matches.iterrows():
            h = row["home_team"]
            a = row["away_team"]

            lam_h, lam_a = poisson.expected_goals(h, a)
            scorelines = get_scoreline_probs(lam_h, lam_a)

            st.markdown(f'<div class="wc-card">', unsafe_allow_html=True)
            st.markdown(
                f"<h3 style='margin:0 0 0.3rem 0'>{h} vs {a}</h3>"
                f"<span style='color:#8b8fa3;font-size:0.85rem'>"
                f"λ = {lam_h:.2f} (home) / {lam_a:.2f} (away) | "
                f"Most likely: <strong>{scorelines[0]['score']}</strong> ({scorelines[0]['prob']*100:.1f}%)"
                f"</span>",
                unsafe_allow_html=True,
            )

            # Build a horizontal bar chart of top scorelines
            scores = [s["score"] for s in scorelines]
            probs_scores = [s["prob"] * 100 for s in scorelines]
            colors = ["#4caf50" if s["home"] > s["away"] else "#f44336" if s["home"] < s["away"] else "#ffc107" for s in scorelines]

            fig = go.Figure(go.Bar(
                x=probs_scores,
                y=scores,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{p:.1f}%" for p in probs_scores],
                textposition="outside",
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(
                    title="Probability (%)",
                    showgrid=True,
                    gridcolor="#2a2d3a",
                    zeroline=False,
                ),
                yaxis=dict(title="", autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b8fa3"),
                bargap=0.3,
            )
            # Remove hover for cleaner look
            fig.update_traces(hovertemplate="%{y}: %{x:.1f}%")

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Could not compute Poisson scorelines: {e}")

else:
    st.info("World Cup data not available for Poisson calculations.")


# ═══════════════════════════════════════════════════════════
#  SECTION 4: CONFIDENCE TRENDS & INSIGHTS
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📈 Confidence Analysis & Trends")

st.markdown(
    "<p style='color:#8b8fa3'>Understanding where the model is most and least confident helps "
    "identify high-conviction bets and uncertain matches.</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Confidence by Match", "Edge Analysis", "Model Insights"])

with tab1:
    # Horizontal bar chart of confidence levels
    conf_plot_data = visible_matches.sort_values("confidence", ascending=True)
    match_labels_conf = [f"{r['home_team'][:12]} vs {r['away_team'][:12]}" for _, r in conf_plot_data.iterrows()]
    conf_values = conf_plot_data["confidence"].values * 100

    fig_conf = go.Figure(go.Bar(
        x=conf_values,
        y=match_labels_conf,
        orientation="h",
        marker=dict(
            color=conf_values,
            colorscale=[[0, "#f44336"], [0.35, "#ffc107"], [0.5, "#8bc34a"], [1, "#4caf50"]],
            cmin=30,
            cmax=55,
        ),
        text=[f"{c:.1f}%" for c in conf_values],
        textposition="outside",
    ))
    fig_conf.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            title="Confidence (%)",
            range=[20, 65],
            showgrid=True,
            gridcolor="#2a2d3a",
            zeroline=False,
        ),
        yaxis=dict(title="", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b8fa3"),
        bargap=0.4,
    )
    fig_conf.update_traces(hovertemplate="%{y}: %{x:.1f}% confidence")

    st.plotly_chart(fig_conf, use_container_width=True)

    # Interpretation
    high_conf = visible_matches[visible_matches["confidence"] >= 0.50]
    low_conf = visible_matches[visible_matches["confidence"] < 0.40]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="wc-card">', unsafe_allow_html=True)
        st.markdown("**🔒 High Confidence (≥50%)**")
        if len(high_conf) > 0:
            for _, r in high_conf.iterrows():
                st.markdown(
                    f"- {r['home_team']} vs {r['away_team']}: "
                    f"<span style='color:#4caf50'>{r['confidence']:.0%}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("No matches above 50% confidence threshold.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="wc-card">', unsafe_allow_html=True)
        st.markdown("**⚠️ Low Confidence (<40%)**")
        if len(low_conf) > 0:
            for _, r in low_conf.iterrows():
                st.markdown(
                    f"- {r['home_team']} vs {r['away_team']}: "
                    f"<span style='color:#f44336'>{r['confidence']:.0%}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("No matches below 40% confidence.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    # Edge analysis: show matches where home/away probability gap is largest
    edge_data = visible_matches.copy()
    edge_data["prob_gap"] = abs(
        edge_data["home_win_prob"] - edge_data["away_win_prob"]
    )
    edge_data["edge_type"] = edge_data.apply(
        lambda r: (
            f"{r['home_team']} edge" if r["home_win_prob"] >= r["away_win_prob"]
            else f"{r['away_team']} edge"
        ),
        axis=1,
    )
    matches_edge = edge_data.sort_values("prob_gap", ascending=False)

    fig_edge = go.Figure()

    fig_edge.add_trace(go.Bar(
        name="Home Win",
        x=matches_edge["home_win_prob"] * 100,
        y=[f"{r['home_team'][:12]} vs {r['away_team'][:12]}" for _, r in matches_edge.iterrows()],
        orientation="h",
        marker=dict(color="#4caf50"),
    ))
    fig_edge.add_trace(go.Bar(
        name="Draw",
        x=matches_edge["draw_prob"] * 100,
        y=[f"{r['home_team'][:12]} vs {r['away_team'][:12]}" for _, r in matches_edge.iterrows()],
        orientation="h",
        marker=dict(color="#ffc107"),
    ))
    fig_edge.add_trace(go.Bar(
        name="Away Win",
        x=matches_edge["away_win_prob"] * 100,
        y=[f"{r['home_team'][:12]} vs {r['away_team'][:12]}" for _, r in matches_edge.iterrows()],
        orientation="h",
        marker=dict(color="#f44336"),
    ))

    fig_edge.update_layout(
        barmode="stack",
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="Probability (%)", showgrid=True, gridcolor="#2a2d3a", zeroline=False),
        yaxis=dict(title="", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b8fa3"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    st.plotly_chart(fig_edge, use_container_width=True)

    # Edge highlights
    st.markdown('<div class="wc-card">', unsafe_allow_html=True)
    st.markdown("**🔍 Largest Probability Gaps**")
    for _, r in matches_edge.head(3).iterrows():
        gap = r["prob_gap"] * 100
        fav = r["home_team"] if r["home_win_prob"] >= r["away_win_prob"] else r["away_team"]
        underdog = r["away_team"] if r["home_win_prob"] >= r["away_win_prob"] else r["home_team"]
        st.markdown(f"- **{r['home_team']} vs {r['away_team']}**: {fav} favored by {gap:.1f}pp over {underdog}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    # Model performance metrics
    st.markdown('<div class="wc-card">', unsafe_allow_html=True)
    st.markdown("**🧠 Model Configuration**")
    st.markdown("""
    | Component | Detail |
    |---|---|
    | **Model** | XGBoost (multi:softprob) |
    | **Training data** | 658 completed matches (7 World Cups + 3 intl tournaments) |
    | **Features** | 80+ rolling stats, Elo ratings, xG metrics, Poisson λ |
    | **Top feature** | Expected_Goal_Difference |
    | **Venue handling** | Swap-and-average for neutral knockout matches |
    | **Test accuracy** | 60.6% (beats baseline by 12pp) |
    | **Poisson model** | Fitted on 658 matches, max 8 goals/team |
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Expected goals vs confidence scatter (computed from Poisson model)
    if not data.empty:
        try:
            completed = data[data["result"].notna()].copy()
            poisson_team = fit_poisson_model(completed)
            scatter_data = visible_matches.copy()
            eg_home = []
            eg_away = []
            for _, r in visible_matches.iterrows():
                lam_h, lam_a = poisson_team.expected_goals(r["home_team"], r["away_team"])
                eg_home.append(round(lam_h, 3))
                eg_away.append(round(lam_a, 3))
            scatter_data["exp_home_goals"] = eg_home
            scatter_data["exp_away_goals"] = eg_away
            
            fig_scatter = px.scatter(
                scatter_data,
                x="exp_home_goals",
                y="confidence",
                color="home_win_prob",
                hover_data=["home_team", "away_team"],
                labels={
                    "exp_home_goals": "Expected Home Goals (Poisson)",
                    "confidence": "Model Confidence",
                    "home_win_prob": "Home Win Prob",
                },
                color_continuous_scale="Viridis",
            )
            fig_scatter.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b8fa3"),
            )
            fig_scatter.update_traces(marker=dict(size=14, line=dict(width=1, color="#333")))
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception:
            pass

    # Team strength radar (top teams by appearance)
    completed = data[data["result"].notna()] if not data.empty else pd.DataFrame()
    if not completed.empty:
        try:
            poisson_team = fit_poisson_model(completed)
            strengths = poisson_team.team_strengths

            # Find the R16 teams
            r16_teams = set()
            for _, r in visible_matches.iterrows():
                r16_teams.add(r["home_team"])
                r16_teams.add(r["away_team"])

            # Show attack vs defense for R16 teams
            team_data = []
            for team in sorted(r16_teams):
                if team in strengths:
                    att, deff = strengths[team]
                    team_data.append({
                        "Team": team,
                        "Attack": round(att, 3),
                        "Defense": round(deff, 3),
                    })

            if team_data:
                df_teams = pd.DataFrame(team_data)
                df_teams["Net_Strength"] = df_teams["Attack"] - df_teams["Defense"]
                df_teams.sort_values("Net_Strength", ascending=False, inplace=True)

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Bar(
                    x=df_teams["Team"],
                    y=df_teams["Attack"],
                    name="Attack (α)",
                    marker=dict(color="#4caf50"),
                ))
                fig_radar.add_trace(go.Bar(
                    x=df_teams["Team"],
                    y=df_teams["Defense"],
                    name="Defense (β)",
                    marker=dict(color="#f44336"),
                ))
                fig_radar.update_layout(
                    barmode="group",
                    height=350,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(title="", tickangle=-45),
                    yaxis=dict(title="Strength (1.0 = avg)", showgrid=True, gridcolor="#2a2d3a", zeroline=True, zerolinecolor="#333"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b8fa3"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown(
                    "<p style='color:#8b8fa3;font-size:0.85rem'>"
                    "<strong>Attack (α)</strong>: >1.0 = stronger than average | "
                    "<strong>Defense (β)</strong>: <1.0 = concedes fewer than average "
                    "(lower is better for defense)</p>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem'>"
    "World Cup 2026 Predictions | XGBoost + Poisson Model | "
    f"Data: 658 completed matches | "
    f"Generated: {pd.Timestamp.now().strftime('%d %b %Y %H:%M')}"
    "</div>",
    unsafe_allow_html=True,
)


# ── Sidebar ─────────────────────────────────────────────
st.sidebar.markdown("## 🏆 World Cup 2026")
st.sidebar.markdown("---")

st.sidebar.markdown("### Prediction Summary")
total = len(visible_matches)
home_c = int((visible_matches["home_win_prob"] >= visible_matches["away_win_prob"]).sum())
away_c = total - home_c
st.sidebar.markdown(
    f"""
    - 🟢 Home favorites: **{home_c}** ({home_c/total*100:.0f}%)
    - 🔴 Away favorites: **{away_c}** ({away_c/total*100:.0f}%)
    - 📊 Avg confidence: **{visible_matches['confidence'].mean():.1%}**
    - 🎯 Max confidence: **{visible_matches['confidence'].max():.1%}**
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.page_link("dashboard.py", label="← Back to Dashboard", use_container_width=True)
