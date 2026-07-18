"""
EDA — Exploratory Data Analysis for football match data.

Generates six publication-quality figures with matplotlib and saves them to
``reports/figures/``.  Each chart includes a text annotation explaining what
it reveals.

Charts generated
----------------
1. **Win distribution**       — bar chart of H / D / A proportions
2. **Goals distribution**     — histogram of home & away goals
3. **Home advantage**         — side-by-side win/draw/loss rates by venue
4. **Team statistics**        — top-N teams by goals scored & conceded
5. **Correlation matrix**     — heatmap of numeric feature correlations
6. **Missing values**         — heatmap of missing-value patterns

Typical usage::

    from src.eda import run_eda
    report = run_eda()            # loads from data/processed/results_clean.csv
    report = run_eda(df=my_df)    # or pass a DataFrame directly
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import seaborn as sns

from config import config as _global_config

# Use Agg backend only if no interactive backend is already active
if matplotlib.get_backend() in ("", None):
    matplotlib.use("Agg")  # non-interactive for headless environments

logger = logging.getLogger(__name__)

# ── Style ───────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Output directory
_FIGURE_DIR = _global_config.paths.data.parent / "reports" / "figures"


# ═══════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════


def run_eda(
    df: pd.DataFrame | None = None,
    data_path: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Run the full EDA pipeline and save all figures.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Preprocessed match data.  If ``None``, loads from *data_path*.
    data_path : str | Path, optional
        Path to a cleaned CSV.  Defaults to ``data/processed/results_clean.csv``.
    show : bool
        If ``True``, also display each figure via ``plt.show()`` (useful in
        notebooks).  Default ``False``.

    Returns
    -------
    dict[str, Any]
        Report with file paths, chart descriptions, and dataset stats.
    """
    if df is None:
        path = Path(data_path or _global_config.paths.processed / "results_clean.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"EDA input not found at {path}. "
                "Run ``src.preprocessing.run_preprocessing()`` first."
            )
        df = pd.read_csv(path, low_memory=False)
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, path)

    logger.info("Running EDA on %d rows × %d columns", *df.shape)
    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "dataset": {"rows": len(df), "columns": len(df.columns)},
        "charts": [],
    }

    # ── Generate each chart ───────────────────────
    charts = [
        ("01_win_distribution", _chart_win_distribution),
        ("02_goals_distribution", _chart_goals_distribution),
        ("03_home_advantage", _chart_home_advantage),
        ("04_team_statistics", _chart_team_statistics),
        ("05_correlation_matrix", _chart_correlation_matrix),
        ("06_missing_values", _chart_missing_values),
    ]

    for name, fn in charts:
        try:
            fig, explanation = fn(df)
            path = _FIGURE_DIR / f"{name}.png"
            fig.savefig(path)
            report["charts"].append({
                "file": str(path),
                "title": explanation.split("\n")[0],
                "explanation": explanation,
            })
            logger.info("  ✓ %s — %s", name, explanation.split("\n")[0])
            if show:
                plt.show()
            plt.close(fig)
        except Exception as exc:
            logger.warning("  ⚠ %s failed: %s", name, exc)

    logger.info("EDA complete — %d charts saved to %s", len(report["charts"]), _FIGURE_DIR)
    return report


# ═══════════════════════════════════════════════════════════
#  1.  Win distribution
# ═══════════════════════════════════════════════════════════


def _chart_win_distribution(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Bar chart showing the proportion of Home wins, Draws, and Away wins.

    **Why this matters:** Football is famously low-scoring, which makes draws
    far more common than in most sports.  A classifier that always predicts
    "Home win" would already beat random chance (~45% accuracy in the Premier
    League).  Understanding the baseline distribution tells us the absolute
    upper ceiling of any naive strategy.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    if "result" not in df.columns:
        ax.text(0.5, 0.5, "No 'result' column found", ha="center", va="center")
        return fig, "No result column — skipped"

    # Ensure fixed order: H → D → A (regardless of frequency)
    fixed_order = ["H", "D", "A"]
    counts = df["result"].value_counts().reindex(fixed_order).fillna(0).astype(int)
    total = counts.sum()
    percentages = counts / total * 100

    colors = {"H": "#2ecc71", "D": "#f1c40f", "A": "#e74c3c"}
    bar_colors = [colors.get(r, "#95a5a6") for r in counts.index]

    bars = ax.bar(range(len(counts)), counts.values, color=bar_colors, edgecolor="white",
                  width=0.55, alpha=0.9)

    # Annotate each bar with count and percentage
    for i, (label, count_val) in enumerate(counts.items()):
        pct = percentages.get(label, 0)
        ax.text(i, count_val + total * 0.02,
                f"{int(count_val)}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Match Result")
    ax.set_ylabel("Number of Matches")
    ax.set_title("Win Distribution — Home / Draw / Away", fontweight="bold", fontsize=14)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Home Win", "Draw", "Away Win"])

    # Add baseline annotation
    home_pct = percentages.get("H", 0)
    ax.text(0.5, -0.15,
            f"Home wins account for {home_pct:.1f}% of matches — "
            f"a naive 'always predict home' baseline would achieve {home_pct:.1f}% accuracy.",
            transform=ax.transAxes, ha="center", fontsize=10, style="italic",
            color="#555555")

    explanation = (
        f"Win Distribution — Home: {percentages.get('H', 0):.1f}%, "
        f"Draw: {percentages.get('D', 0):.1f}%, "
        f"Away: {percentages.get('A', 0):.1f}%"
    )
    return fig, explanation


# ═══════════════════════════════════════════════════════════
#  2.  Goals distribution
# ═══════════════════════════════════════════════════════════


def _chart_goals_distribution(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Overlaid histogram of home goals vs away goals.

    **Why this matters:** Goal distributions are heavily right-skewed (most
    matches have 0–2 goals per team, with a long tail of high-scoring games).
    This skew affects model choice — count-based models (Poisson) are common
    in football analytics.  The histogram also reveals whether home teams
    systematically score more, which is the foundation of home advantage.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        ax.text(0.5, 0.5, "Goal columns not found", ha="center", va="center")
        return fig, "Goal columns not found — skipped"

    home = df["home_goals"].dropna()
    away = df["away_goals"].dropna()

    max_goals = max(home.max(), away.max())
    bins = np.arange(-0.5, max_goals + 1.5, 1)

    ax.hist(home, bins=bins, alpha=0.65, label="Home Goals", color="#2ecc71",
            edgecolor="white", linewidth=0.8)
    ax.hist(away, bins=bins, alpha=0.65, label="Away Goals", color="#e74c3c",
            edgecolor="white", linewidth=0.8)

    # Add vertical lines for means
    home_mean = home.mean()
    away_mean = away.mean()
    ax.axvline(home_mean, color="#2ecc71", linestyle="--", linewidth=1.5,
               label=f"Home mean = {home_mean:.2f}")
    ax.axvline(away_mean, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Away mean = {away_mean:.2f}")

    ax.set_xlabel("Goals Scored")
    ax.set_ylabel("Number of Matches")
    ax.set_title("Goals Distribution — Home vs Away", fontweight="bold", fontsize=14)
    ax.legend(fontsize=10)

    # Stats box
    stats_text = (
        f"Home: mean={home_mean:.2f}, median={home.median():.0f}, "
        f"max={int(home.max())}\n"
        f"Away: mean={away_mean:.2f}, median={away.median():.0f}, "
        f"max={int(away.max())}\n"
        f"Total: mean={df['home_goals'].add(df['away_goals'], fill_value=0).mean():.2f} goals/match"
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    explanation = (
        f"Goals Distribution — Home mean={home_mean:.2f}, "
        f"Away mean={away_mean:.2f}, "
        f"Total mean={(home_mean + away_mean):.2f}"
    )
    return fig, explanation


# ═══════════════════════════════════════════════════════════
#  3.  Home advantage
# ═══════════════════════════════════════════════════════════


def _chart_home_advantage(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Grouped bar chart comparing win/draw/loss rates for home vs away teams.

    **Why this matters:** Home advantage is the single most reliable finding
    in football analytics.  Teams playing at home win ~45% of matches vs
    ~30% away.  Factors include crowd support, travel fatigue, referee bias,
    and familiar pitch dimensions.  A model that ignores venue will
    systematically underestimate home teams.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if "result" not in df.columns:
        ax.text(0.5, 0.5, "No 'result' column", ha="center", va="center")
        return fig, "No result column — skipped"

    n = len(df)
    home_wins = (df["result"] == "H").sum() / n * 100
    away_wins = (df["result"] == "A").sum() / n * 100
    draws = (df["result"] == "D").sum() / n * 100

    categories = ["Home Team", "Away Team"]
    win_pcts = [home_wins, away_wins]
    draw_pcts = [draws, draws]  # Draws are the same match, shown twice for comparison
    loss_pcts = [away_wins, home_wins]  # Loss mirror of win

    x = np.arange(len(categories))
    width = 0.25

    bars_win = ax.bar(x - width, win_pcts, width, label="Win", color="#2ecc71",
                      edgecolor="white")
    bars_draw = ax.bar(x, draw_pcts, width, label="Draw", color="#f1c40f",
                       edgecolor="white")
    bars_loss = ax.bar(x + width, loss_pcts, width, label="Loss", color="#e74c3c",
                       edgecolor="white")

    # Annotate
    for bar, pct in zip(bars_win, win_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")
    for bar, pct in zip(bars_draw, draw_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")
    for bar, pct in zip(bars_loss, loss_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Percentage of Matches")
    ax.set_title("Home Advantage — Win/Draw/Loss by Venue", fontweight="bold", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(win_pcts), max(draw_pcts), max(loss_pcts)) + 8)

    advantage = home_wins - away_wins
    ax.text(0.5, -0.18,
            f"Home teams win {home_wins:.1f}% of matches vs {away_wins:.1f}% for away teams — "
            f"a home advantage of {advantage:.1f} percentage points.",
            transform=ax.transAxes, ha="center", fontsize=10, style="italic",
            color="#555555")

    explanation = (
        f"Home Advantage — home win: {home_wins:.1f}%, "
        f"away win: {away_wins:.1f}%, "
        f"advantage: {advantage:.1f} pp"
    )
    return fig, explanation


# ═══════════════════════════════════════════════════════════
#  4.  Team statistics
# ═══════════════════════════════════════════════════════════


def _chart_team_statistics(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Horizontal bar chart of top-N teams by goals scored and conceded.

    **Why this matters:** Team-level aggregates reveal which teams are
    offensively dominant (high goals scored — usually title contenders) vs
    defensively weak (high goals conceded — relegation candidates).  This
    helps validate the data against domain knowledge (e.g. Manchester City
    should be near the top for scoring, not a newly-promoted side).
    """
    n_top = 10

    required = ["home_team", "away_team", "home_goals", "away_goals"]
    if not all(c in df.columns for c in required):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Team/goal columns not found", ha="center", va="center")
        return fig, "Team/goal columns not found — skipped"

    # Compute per-team goals scored & conceded
    home_scored = df.groupby("home_team")["home_goals"].sum()
    away_scored = df.groupby("away_team")["away_goals"].sum()
    goals_scored = home_scored.add(away_scored, fill_value=0).sort_values(ascending=False)

    home_conceded = df.groupby("home_team")["away_goals"].sum()
    away_conceded = df.groupby("away_team")["home_goals"].sum()
    goals_conceded = home_conceded.add(away_conceded, fill_value=0).sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # ── Left: Top-N scorers ──
    top_scorers = goals_scored.head(n_top).iloc[::-1]
    bars1 = ax1.barh(range(len(top_scorers)), top_scorers.values,
                     color="#2ecc71", edgecolor="white", alpha=0.85)
    ax1.set_yticks(range(len(top_scorers)))
    ax1.set_yticklabels(top_scorers.index, fontsize=9)
    ax1.set_xlabel("Total Goals Scored")
    ax1.set_title(f"Top {n_top} — Most Goals Scored", fontweight="bold", fontsize=13)
    for bar, val in zip(bars1, top_scorers.values):
        ax1.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                 f"{int(val)}", va="center", fontsize=9, fontweight="bold")

    # ── Right: Top-N conceded ──
    top_conceded = goals_conceded.head(n_top).iloc[::-1]
    bars2 = ax2.barh(range(len(top_conceded)), top_conceded.values,
                     color="#e74c3c", edgecolor="white", alpha=0.85)
    ax2.set_yticks(range(len(top_conceded)))
    ax2.set_yticklabels(top_conceded.index, fontsize=9)
    ax2.set_xlabel("Total Goals Conceded")
    ax2.set_title(f"Top {n_top} — Most Goals Conceded", fontweight="bold", fontsize=13)
    for bar, val in zip(bars2, top_conceded.values):
        ax2.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                 f"{int(val)}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()

    explanation = (
        f"Team Statistics — Top scorer: {goals_scored.index[0]} ({int(goals_scored.values[0])} goals), "
        f"Most conceded: {goals_conceded.index[0]} ({int(goals_conceded.values[0])} goals)"
    )
    return fig, explanation


# ═══════════════════════════════════════════════════════════
#  5.  Correlation matrix
# ═══════════════════════════════════════════════════════════


def _chart_correlation_matrix(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Heatmap of Pearson correlations between numeric columns.

    **Why this matters:** High correlations between features indicate
    redundancy (multicollinearity), which can destabilise linear models
    (logistic regression) and inflate feature importance in tree-based
    models.  The heatmap also reveals expected relationships — e.g. home
    goals correlate positively with home win — which serves as a
    sanity-check on the data.

    **What to look for:**
    - Dark red squares = strong positive correlation (move together).
    - Dark blue squares = strong negative correlation (move opposite).
    - Diagonal is always 1.0 (a feature correlates perfectly with itself).
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Need at least 2 numeric columns", ha="center", va="center")
        return fig, "Insufficient numeric columns — skipped"

    # Select a reasonable subset for readability
    priority_cols = [
        c for c in ["home_goals", "away_goals", "goal_diff", "total_goals",
                     "target", "year", "month", "day_of_week",
                     "is_midweek", "week_of_season"]
        if c in numeric.columns
    ]
    # Add any feature-engineered columns that exist
    fe_cols = [c for c in numeric.columns if c.startswith(("h_", "a_", "h2h_"))
               and c in numeric.columns]
    selected = list(dict.fromkeys(priority_cols + fe_cols))  # dedup preserving order
    selected = selected[:20]  # cap at 20 for readability

    if len(selected) < 2:
        selected = numeric.columns[:15].tolist()

    corr = numeric[selected].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(max(10, len(selected) * 0.6),
                                    max(8, len(selected) * 0.55)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, annot=len(selected) <= 15, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.75},
                ax=ax)

    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=14)
    ax.tick_params(axis="x", rotation=45)

    explanation = (
        f"Correlation Matrix — {len(selected)} features, "
        f"{((corr.abs() > 0.7).sum().sum() - len(corr)) // 2} pairs with |r| > 0.7"
    )
    return fig, explanation


# ═══════════════════════════════════════════════════════════
#  6.  Missing values
# ═══════════════════════════════════════════════════════════


def _chart_missing_values(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Heatmap and bar chart of missing-value patterns across columns.

    **Why this matters:** Missing data is pervasive in football datasets —
    older seasons lack detailed statistics (shots, corners, cards),
    postponed matches have missing results, and some leagues report fewer
    columns than others.  This chart tells us:
    - Which columns are safe to use (0% missing).
    - Which need imputation (< 50% missing).
    - Which are too sparse to use (> 50% missing) and will be dropped.

    **Interpreting the heatmap:**
    - White / light cells = no missing data.
    - Dark purple cells = missing data present.
    - Rows sort by completeness so patterns are visible.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(df.columns) * 0.25)))

    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(1)
    missing_df = pd.DataFrame({"Count": missing_count, "Percent": missing_pct})
    missing_df = missing_df[missing_df["Count"] > 0].sort_values("Count", ascending=False)

    # ── Left: Missing count bar chart ──
    if len(missing_df) > 0:
        top_missing = missing_df.head(min(20, len(missing_df)))
        colors = ["#e74c3c" if p > 50 else "#f39c12" if p > 20 else "#3498db"
                  for p in top_missing["Percent"]]
        bars = ax1.barh(range(len(top_missing)), top_missing["Count"],
                        color=colors, edgecolor="white", alpha=0.85)
        ax1.set_yticks(range(len(top_missing)))
        ax1.set_yticklabels(top_missing.index, fontsize=9)
        ax1.set_xlabel("Missing Value Count")
        ax1.set_title("Columns with Missing Values", fontweight="bold", fontsize=13)
        for bar, cnt, pct in zip(bars, top_missing["Count"], top_missing["Percent"]):
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                     f"{int(cnt)} ({pct:.1f}%)", va="center", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No missing values!", ha="center", va="center",
                 fontsize=14, fontweight="bold", color="#2ecc71")

    # ── Right: Missing value heatmap ──
    missing_cols = missing_df.index.tolist()
    if missing_cols:
        # Sample rows for visualisation (max 2000 rows for performance)
        sample = df[missing_cols].copy()
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        # Sort rows by completeness for pattern visibility
        sample["_missing_count"] = sample.isnull().sum(axis=1)
        sample = sample.sort_values("_missing_count").drop(columns="_missing_count")

        sns.heatmap(sample.isnull(), cbar=False, yticklabels=False,
                    cmap=["#2ecc71", "#4a235a"], ax=ax2)
        ax2.set_xlabel("")
        ax2.set_title("Missing Value Pattern (sample)", fontweight="bold", fontsize=13)
        # Color bar legend
        legend_elements = [
            Patch(facecolor="#4a235a", label="Missing"),
            Patch(facecolor="#2ecc71", label="Present"),
        ]
        ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No missing values!", ha="center", va="center",
                 fontsize=14, fontweight="bold", color="#2ecc71")

    plt.tight_layout()

    total_missing = int(df.isnull().sum().sum())
    missing_cells_pct = total_missing / (df.shape[0] * df.shape[1]) * 100
    explanation = (
        f"Missing Values — {total_missing} missing cells "
        f"({missing_cells_pct:.2f}% of all cells), "
        f"{len(missing_df)} columns affected"
    )
    return fig, explanation
