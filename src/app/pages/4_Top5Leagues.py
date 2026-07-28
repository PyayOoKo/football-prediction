"""
Top 5 European Leagues Dashboard — EPL, La Liga, Bundesliga, Serie A, Ligue 1.

Shows league standings, upcoming fixtures with predictions, recent results,
and performance analysis for the top 5 European football leagues.

Run with:
    streamlit run src/app/dashboard.py
    → then navigate to the "🏆 Top 5 European Leagues" page
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Top 5 Leagues",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "results_clean.csv"

# ── League config ───────────────────────────────────────
TOP5_LEAGUES = {
    "EPL": {"name": "Premier League", "country": "England", "icon": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "La_Liga": {"name": "La Liga", "country": "Spain", "icon": "🇪🇸"},
    "Bundesliga": {"name": "Bundesliga", "country": "Germany", "icon": "🇩🇪"},
    "Serie_A": {"name": "Serie A", "country": "Italy", "icon": "🇮🇹"},
    "Ligue_1": {"name": "Ligue 1", "country": "France", "icon": "🇫🇷"},
}
LEAGUE_ORDER = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
LEAGUE_COLORS = {
    "EPL": "#3b82f6",
    "La_Liga": "#f97316",
    "Bundesliga": "#22c55e",
    "Serie_A": "#6366f1",
    "Ligue_1": "#eab308",
}


# ═══════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════

st.markdown(
    """
<style>
    .stApp { background: #0e1117; }
    .stApp header { background: #1a1d27; }

    .league-card {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .league-card h3 {
        margin: 0 0 0.5rem 0;
        color: #ffffff;
    }

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

    .standing-row {
        display: flex;
        align-items: center;
        padding: 0.4rem 0.8rem;
        border-bottom: 1px solid #1e2235;
        font-size: 0.85rem;
    }
    .standing-row:hover {
        background: rgba(255,255,255,0.03);
    }
    .standing-row .pos { width: 2rem; font-weight: 700; color: #8b8fa3; }
    .standing-row .team { flex: 1; color: #e0e0e0; font-weight: 500; }
    .standing-row .stat { width: 3rem; text-align: center; color: #8b8fa3; }
    .standing-row .pts { width: 3rem; text-align: center; font-weight: 700; color: #fff; }

    .fixture-item {
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid #1e2235;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .fixture-item .teams { flex: 1; }
    .fixture-item .prob-bar { flex: 2; height: 6px; border-radius: 3px; background: #1a1d27; display: flex; }
    .fixture-item .prob-bar .seg { height: 100%; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Loading league data …")
def load_league_data() -> pd.DataFrame:
    if not DATA_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_CSV, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "league" in df.columns:
        df = df[df["league"].isin(TOP5_LEAGUES.keys())].copy()
    return df


@st.cache_data(show_spinner="Computing standings …")
def compute_standings(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """Compute league standings from match results."""
    league_df = df[df["league"] == league].copy()
    if league_df.empty or "result" not in league_df.columns:
        return pd.DataFrame()

    completed = league_df[league_df["result"].notna()].copy()

    teams: dict[str, dict[str, float]] = {}
    for _, row in completed.iterrows():
        home, away = row["home_team"], row["away_team"]
        res = row["result"]
        hg = row.get("home_goals", 0) or 0
        ag = row.get("away_goals", 0) or 0

        for team in [home, away]:
            if team not in teams:
                teams[team] = {
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "gf": 0,
                    "ga": 0,
                    "pts": 0,
                    "gd": 0,
                }

        teams[home]["played"] += 1
        teams[away]["played"] += 1
        teams[home]["gf"] += int(hg)
        teams[home]["ga"] += int(ag)
        teams[away]["gf"] += int(ag)
        teams[away]["ga"] += int(hg)

        if res == "H":
            teams[home]["wins"] += 1
            teams[home]["pts"] += 3
            teams[away]["losses"] += 1
        elif res == "A":
            teams[away]["wins"] += 1
            teams[away]["pts"] += 3
            teams[home]["losses"] += 1
        else:
            teams[home]["draws"] += 1
            teams[away]["draws"] += 1
            teams[home]["pts"] += 1
            teams[away]["pts"] += 1

    standings = []
    for team, stats in teams.items():
        stats["gd"] = stats["gf"] - stats["ga"]
        stats["team"] = team
        standings.append(stats)

    result_df = pd.DataFrame(standings)
    if not result_df.empty:
        result_df = result_df.sort_values(
            ["pts", "gd", "gf"], ascending=False
        ).reset_index(drop=True)
        result_df.index = result_df.index + 1
        result_df.index.name = "pos"
    return result_df


@st.cache_data(show_spinner="Loading predictions …")
def load_predictions() -> pd.DataFrame:
    """Load model predictions from the saved blend predictions file."""
    pred_path = PROJECT_ROOT / "reports" / "predictions" / "latest_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


# ── Load all data ──
df = load_league_data()

if df.empty:
    st.error("⚠ No league data found. Run the data pipeline first.")
    st.stop()

# ── Get recent season ──
latest_season = df["season"].max() if "season" in df.columns else None
df_season = df[df["season"] == latest_season].copy() if latest_season else df
df_completed = df_season[df_season["result"].notna()].copy()
df_upcoming = df_season[df_season["result"].isna()].copy()

has_upcoming = len(df_upcoming) > 0


# ═══════════════════════════════════════════════════════════
#  HERO SECTION
# ═══════════════════════════════════════════════════════════

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown(
    "<h1>🏆 Top 5 European Leagues</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p>📊 Live predictions and analysis for the Premier League, La Liga, Bundesliga, "
    "Serie A, and Ligue 1 — powered by the 5-Model Blend "
    "(Dixon-Coles + Elo + XGBoost + LightGBM + CatBoost).</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Global metrics row ──────────────────────────────────
total_matches = len(df_completed)
total_upcoming = len(df_upcoming)
total_teams = df_completed["home_team"].nunique() if not df_completed.empty else 0
avg_home_win = (df_completed["result"] == "H").mean() if not df_completed.empty else 0

cols = st.columns(5)
with cols[0]:
    st.markdown(
        f'<div class="metric-tile"><div class="value">{total_matches:,}</div><div class="label">⚽ Completed Matches</div></div>',
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        f'<div class="metric-tile"><div class="value">{total_upcoming:,}</div><div class="label">🔮 Upcoming Fixtures</div></div>',
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        f'<div class="metric-tile"><div class="value">{total_teams}</div><div class="label">🏃 Teams</div></div>',
        unsafe_allow_html=True,
    )
with cols[3]:
    st.markdown(
        f'<div class="metric-tile"><div class="value">{avg_home_win:.0%}</div><div class="label">🏠 Home Win Rate</div></div>',
        unsafe_allow_html=True,
    )
with cols[4]:
    st.markdown(
        '<div class="metric-tile"><div class="value">5</div><div class="label">🏆 Leagues</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
#  LEAGUE SELECTOR & STANDINGS
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📋 League Standings")

# League selector
selected_league = st.selectbox(
    "Select League",
    options=LEAGUE_ORDER,
    format_func=lambda x: (
        f"{TOP5_LEAGUES[x]['icon']} {TOP5_LEAGUES[x]['name']} ({TOP5_LEAGUES[x]['country']})"
    ),
    index=0,
)

# Compute and show standings
standings = compute_standings(df_season, selected_league)

if not standings.empty:
    league_info = TOP5_LEAGUES[selected_league]
    st.markdown(
        f'<div class="league-card">'
        f"<h3>{league_info['icon']} {league_info['name']} — Season {latest_season}</h3>",
        unsafe_allow_html=True,
    )

    # Top 6 highlighted
    top6 = standings.head(6)
    st.markdown("**Top 6 — Champions League spots**")

    # Mini table
    for pos, (_, row) in enumerate(top6.iterrows(), 1):
        color = "#4caf50" if pos <= 4 else "#ffc107" if pos <= 6 else "#8b8fa3"
        badge = "⭐" if pos <= 4 else "🔶" if pos <= 6 else ""
        st.markdown(
            f'<div class="standing-row">'
            f'<span class="pos" style="color:{color}">{pos}.</span>'
            f'<span class="team">{badge} {row["team"]}</span>'
            f'<span class="stat">{int(row["played"])}P</span>'
            f'<span class="stat">{int(row["wins"])}W</span>'
            f'<span class="stat">{int(row["draws"])}D</span>'
            f'<span class="stat">{int(row["losses"])}L</span>'
            f'<span class="stat">{int(row["gf"])}:{int(row["ga"])}</span>'
            f'<span class="pts">{int(row["pts"])}pts</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Full standings in a dataframe for scrolling
    with st.expander("📄 Full Standings", expanded=False):
        display_df = standings.copy()
        display_df = display_df.rename(
            columns={
                "team": "Team",
                "played": "P",
                "wins": "W",
                "draws": "D",
                "losses": "L",
                "gf": "GF",
                "ga": "GA",
                "gd": "GD",
                "pts": "Pts",
            }
        )
        st.dataframe(
            display_df[["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]],
            use_container_width=True,
            hide_index=False,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Points distribution chart ──
    st.markdown("### Points Distribution")
    fig = px.bar(
        standings.head(12),
        x="team",
        y="pts",
        color="pts",
        color_continuous_scale="Viridis",
        labels={"team": "Team", "pts": "Points"},
        text="pts",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=350,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#8b8fa3"},
        xaxis_tickangle=-45,
    )
    fig.update_xaxes(gridcolor="#2a2d3a")
    fig.update_yaxes(gridcolor="#2a2d3a")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        f"No completed matches found for {TOP5_LEAGUES[selected_league]['name']} in season {latest_season}."
    )


# ═══════════════════════════════════════════════════════════
#  UPCOMING FIXTURES
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🔮 Upcoming Fixtures & Predictions")

if has_upcoming:
    league_upcoming = df_upcoming[df_upcoming["league"] == selected_league]
    if not league_upcoming.empty:
        st.markdown(
            f'<div class="league-card">'
            f"<h3>📅 Next {min(len(league_upcoming), 15)} Fixtures — {TOP5_LEAGUES[selected_league]['name']}</h3>",
            unsafe_allow_html=True,
        )

        for _, row in league_upcoming.head(15).iterrows():
            date_str = str(row["date"])[:10] if pd.notna(row.get("date")) else "TBD"
            home = row["home_team"]
            away = row["away_team"]
            pred_h = row.get("home_win_prob", 0.4)
            pred_d = row.get("draw_prob", 0.25)
            pred_a = row.get("away_win_prob", 0.35)

            # If predictions aren't in the data, use equal defaults
            if "home_win_prob" not in row.index:
                pred_h = pred_d = pred_a = 1 / 3

            pred_h = float(pred_h) if pd.notna(pred_h) else 0.34
            pred_d = float(pred_d) if pd.notna(pred_d) else 0.33
            pred_a = float(pred_a) if pd.notna(pred_a) else 0.33
            total = pred_h + pred_d + pred_a
            if total > 0:
                pred_h /= total
                pred_d /= total
                pred_a /= total

            fav = max(pred_h, pred_d, pred_a)
            fav_label = (
                f"🏠 {home}"
                if fav == pred_h
                else f"✈️ {away}"
                if fav == pred_a
                else "🤝 Draw"
            )

            st.markdown(
                f'<div class="fixture-item">'
                f'<span style="color:#555;font-size:0.75rem;width:5rem">{date_str}</span>'
                f'<span class="teams"><strong>{home}</strong> vs <strong>{away}</strong></span>'
                f'<div class="prob-bar">'
                f'<div class="seg" style="width:{pred_h * 100:.0f}%;background:#4caf50;border-radius:3px 0 0 3px" title="Home {pred_h:.0%}"></div>'
                f'<div class="seg" style="width:{pred_d * 100:.0f}%;background:#ffc107" title="Draw {pred_d:.0%}"></div>'
                f'<div class="seg" style="width:{pred_a * 100:.0f}%;background:#f44336;border-radius:0 3px 3px 0" title="Away {pred_a:.0%}"></div>'
                f"</div>"
                f'<span style="color:#4fc3f7;font-size:0.75rem;width:6rem;text-align:right">{fav_label} ({fav:.0%})</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"No upcoming fixtures for {TOP5_LEAGUES[selected_league]['name']}.")
else:
    st.info("No upcoming fixtures found in the current dataset.")


# ═══════════════════════════════════════════════════════════
#  LEAGUE STATS & ANALYSIS
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📊 League Comparison & Analysis")

st.markdown(
    "<p style='color:#8b8fa3'>Compare key metrics across all top 5 leagues to identify "
    "trends, scoring patterns, and competitive balance.</p>",
    unsafe_allow_html=True,
)

# ── Per-league summary stats ──
league_stats = []
for league in LEAGUE_ORDER:
    ld = df_completed[df_completed["league"] == league]
    if ld.empty:
        continue
    info = TOP5_LEAGUES[league]
    n_matches = len(ld)
    home_win_pct = (ld["result"] == "H").mean()
    draw_pct = (ld["result"] == "D").mean()
    away_win_pct = (ld["result"] == "A").mean()
    avg_goals = (ld["home_goals"].astype(float) + ld["away_goals"].astype(float)).mean()
    avg_home_goals = ld["home_goals"].astype(float).mean()
    avg_away_goals = ld["away_goals"].astype(float).mean()
    btts_pct = (
        (ld["home_goals"].astype(float) > 0) & (ld["away_goals"].astype(float) > 0)
    ).mean()
    over25_pct = (
        (ld["home_goals"].astype(float) + ld["away_goals"].astype(float)) > 2.5
    ).mean()

    league_stats.append(
        {
            "League": f"{info['icon']} {info['name']}",
            "Matches": n_matches,
            "Home Win": f"{home_win_pct:.0%}",
            "Draw": f"{draw_pct:.0%}",
            "Away Win": f"{away_win_pct:.0%}",
            "Avg Goals": round(avg_goals, 2),
            "Home G": round(avg_home_goals, 2),
            "Away G": round(avg_away_goals, 2),
            "BTTS": f"{btts_pct:.0%}",
            "O2.5": f"{over25_pct:.0%}",
        }
    )

if league_stats:
    st.markdown('<div class="league-card">', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(league_stats),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Goals comparison chart ──
    st.markdown("### ⚽ Average Goals per Match")
    fig_goals = go.Figure()
    for ls in league_stats:
        league_name = ls["League"]
        fig_goals.add_trace(
            go.Bar(
                name=league_name,
                x=[league_name],
                y=[ls["Avg Goals"]],
                marker={
                    "color": LEAGUE_COLORS.get(
                        LEAGUE_ORDER[
                            [l["League"] for l in league_stats].index(league_name)
                        ],
                        "#888",
                    )
                    if league_name in [l["League"] for l in league_stats]
                    else "#888"
                },
                text=[f"{ls['Avg Goals']}"],
                textposition="outside",
            )
        )
    fig_goals.update_layout(
        height=300,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#8b8fa3"},
        yaxis={"title": "Goals per Match", "gridcolor": "#2a2d3a"},
        showlegend=False,
    )
    st.plotly_chart(fig_goals, use_container_width=True)

    # ── Result distribution chart ──
    st.markdown("### 📊 Result Distribution by League")
    fig_results = go.Figure()
    for ls in league_stats:
        league_name = ls["League"]
        hw_pct = float(ls["Home Win"].strip("%")) / 100
        dr_pct = float(ls["Draw"].strip("%")) / 100
        aw_pct = float(ls["Away Win"].strip("%")) / 100
        fig_results.add_trace(
            go.Bar(
                name=league_name,
                x=["Home Win", "Draw", "Away Win"],
                y=[hw_pct, dr_pct, aw_pct],
                marker={
                    "color": LEAGUE_COLORS.get(
                        LEAGUE_ORDER[
                            [l["League"] for l in league_stats].index(league_name)
                        ],
                        "#888",
                    )
                    if league_name in [l["League"] for l in league_stats]
                    else "#888"
                },
                opacity=0.8,
            )
        )
    fig_results.update_layout(
        barmode="group",
        height=350,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#8b8fa3"},
        yaxis={"title": "Proportion", "tickformat": ".0%", "gridcolor": "#2a2d3a"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    st.plotly_chart(fig_results, use_container_width=True)

    # ── BTTS & Over/Under chart ──
    st.markdown("### 🎯 BTTS & Over 2.5 Rates")
    fig_binary = go.Figure()
    for ls in league_stats:
        league_name = ls["League"]
        btts = float(ls["BTTS"].strip("%")) / 100
        ou25 = float(ls["O2.5"].strip("%")) / 100
        fig_binary.add_trace(
            go.Bar(
                name=league_name,
                x=["BTTS", "Over 2.5"],
                y=[btts, ou25],
                marker={
                    "color": LEAGUE_COLORS.get(
                        LEAGUE_ORDER[
                            [l["League"] for l in league_stats].index(league_name)
                        ],
                        "#888",
                    )
                    if league_name in [l["League"] for l in league_stats]
                    else "#888"
                },
                opacity=0.8,
            )
        )
    fig_binary.update_layout(
        barmode="group",
        height=300,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#8b8fa3"},
        yaxis={"title": "Rate", "tickformat": ".0%", "gridcolor": "#2a2d3a"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    st.plotly_chart(fig_binary, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  RECENT RESULTS
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📅 Recent Results")

league_completed = df_completed[df_completed["league"] == selected_league]
if not league_completed.empty:
    recent = league_completed.sort_values("date", ascending=False).head(20)

    st.markdown(
        f'<div class="league-card">'
        f"<h3>Last {len(recent)} Matches — {TOP5_LEAGUES[selected_league]['name']}</h3>",
        unsafe_allow_html=True,
    )

    for _, row in recent.iterrows():
        date_str = str(row["date"])[:10] if pd.notna(row.get("date")) else ""
        home = row["home_team"]
        away = row["away_team"]
        hg = int(row.get("home_goals", 0) or 0)
        ag = int(row.get("away_goals", 0) or 0)
        res = row.get("result", "")

        if res == "H":
            result_color, score_str = "#4caf50", f"<strong>{hg}</strong>-{ag}"
        elif res == "A":
            result_color, score_str = "#f44336", f"{hg}-<strong>{ag}</strong>"
        else:
            result_color, score_str = "#ffc107", f"{hg}-{ag}"

        st.markdown(
            f'<div class="fixture-item">'
            f'<span style="color:#555;font-size:0.75rem;width:5rem">{date_str}</span>'
            f'<span class="teams">{home} vs {away}</span>'
            f'<span style="color:{result_color};font-weight:700;width:3rem;text-align:center">{score_str}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info(f"No completed matches for {TOP5_LEAGUES[selected_league]['name']}.")


# ═══════════════════════════════════════════════════════════
#  FULL DATA TABLE
# ═══════════════════════════════════════════════════════════

st.markdown("---")
with st.expander("📄 Full Match Data", expanded=False):
    league_data = df_season[df_season["league"] == selected_league]
    if not league_data.empty:
        display_cols = [
            c
            for c in [
                "date",
                "home_team",
                "away_team",
                "result",
                "home_goals",
                "away_goals",
                "season",
            ]
            if c in league_data.columns
        ]
        st.dataframe(
            league_data[display_cols].sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem'>"
    "Top 5 European Leagues | Powered by 5-Model Blend "
    "(Dixon-Coles + Elo + XGBoost + LightGBM + CatBoost) | "
    f"Data: results_clean.csv | "
    f"Generated: {pd.Timestamp.now().strftime('%d %b %Y %H:%M')}"
    "</div>",
    unsafe_allow_html=True,
)


# ── Sidebar ─────────────────────────────────────────────
st.sidebar.markdown("## 🏆 Top 5 European Leagues")
st.sidebar.markdown("---")

st.sidebar.markdown("### Season Overview")
st.sidebar.markdown(f"📅 **Season:** {latest_season}")
st.sidebar.markdown(f"⚽ **Matches:** {total_matches:,}")
st.sidebar.markdown(f"🔮 **Upcoming:** {total_upcoming:,}")
st.sidebar.markdown(f"🏃 **Teams:** {total_teams}")

st.sidebar.markdown("---")
st.sidebar.markdown("### League Quick Stats")
for league in LEAGUE_ORDER:
    ld = df_completed[df_completed["league"] == league]
    if not ld.empty:
        info = TOP5_LEAGUES[league]
        n = len(ld)
        hw = (ld["result"] == "H").mean()
        st.sidebar.markdown(
            f"{info['icon']} **{info['name']}** — {n} matches, {hw:.0%} home wins"
        )

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.page_link(
    "dashboard.py", label="← Back to Dashboard", use_container_width=True
)
