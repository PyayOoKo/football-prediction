"""
Shared utilities for the Streamlit dashboard.

Handles model loading, data loading, feature building, and prediction
— all with caching so the app stays fast.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config import config

logger = logging.getLogger(__name__)


# ── Cached model loading ────────────────────────────────

@st.cache_resource(show_spinner="Loading trained model ...")
def load_model(file_name: str = "xgboost_model.joblib") -> Any | None:
    """Load a trained model from the models directory."""
    from src.train import load_model as _load

    try:
        model = _load(file_name)
        return model
    except FileNotFoundError:
        return None
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


@st.cache_resource(show_spinner="Loading preprocessed data ...")
def load_clean_data() -> pd.DataFrame | None:
    """Load the preprocessed match data."""
    path = config.paths.processed / "results_clean.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df


@st.cache_data(show_spinner="Building feature matrix ...")
def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | None:
    """Build the full feature matrix from preprocessed data."""
    from src.feature_engineering import build_features

    try:
        X, y = build_features(df, is_training=True)
        return X, y
    except Exception as exc:
        st.error(f"Feature engineering failed: {exc}")
        return None


@st.cache_data(show_spinner="Running backtest ...")
def run_backtest_cached(
    _model: Any, X_test: pd.DataFrame, y_test: pd.Series,
    odds_df: pd.DataFrame | None = None,
    odds_cols: tuple[str, str, str] = ("BbAvA", "BbAvD", "BbAvH"),
) -> dict[str, Any]:
    """Run and cache a backtest simulation.

    Note: ``_model`` has a leading underscore so Streamlit's ``@st.cache_data``
    skips hashing it (XGBoost models aren't hashable).
    """
    from src.backtesting import run_backtest

    result = run_backtest(
        model=_model, X_test=X_test, y_test=y_test,
        odds_df=odds_df, odds_cols=odds_cols,
        team_cols=("home_team", "away_team"),
        initial_bankroll=config.value_betting.bankroll,
        kelly_fraction=config.value_betting.kelly_fraction,
        min_ev=config.value_betting.min_ev,
        output_dir=config.paths.data.parent / "reports" / "backtest",
        print_report=False, show_charts=False,
    )
    return result


# ── Helpers ─────────────────────────────────────────────

def get_available_teams(df: pd.DataFrame) -> list[str]:
    """Return sorted list of all team names from the dataset."""
    teams = set()
    if "home_team" in df.columns:
        teams.update(df["home_team"].dropna().unique())
    if "away_team" in df.columns:
        teams.update(df["away_team"].dropna().unique())
    return sorted(teams)


def get_available_odds_cols(df: pd.DataFrame) -> tuple[str, str, str] | None:
    """Detect odds columns available in the DataFrame."""
    candidates = [
        ("BbAvA", "BbAvD", "BbAvH"),
        ("B365A", "B365D", "B365H"),
        ("BWA", "BWD", "BWH"),
        ("IWA", "IWD", "IWH"),
        ("PSA", "PSD", "PSH"),
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            non_null = df[list(cols)].notna().all(axis=1).mean()
            if non_null > 0.5:
                return cols
    return None


def get_latest_matches(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the N most recent matches with team names and result."""
    if "date" not in df.columns:
        return df.head(n)
    cols = [c for c in ["date", "home_team", "away_team", "result",
                         "home_goals", "away_goals"] if c in df.columns]
    subset = df[cols].dropna(subset=["date"]).sort_values("date", ascending=False)
    return subset.head(n)


# ── Value bets cache loading ──────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading latest value bets...")
def load_latest_value_bets() -> pd.DataFrame | None:
    """Load the latest value bets report saved by today_value_bets_live.py."""
    path = config.paths.data.parent / "reports" / "value_bets" / "latest.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_value_bets_meta() -> pd.DataFrame | None:
    """Load prediction metadata from latest value bets run."""
    path = config.paths.data.parent / "reports" / "value_bets" / "latest_meta.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


def get_matchup_stats(
    df: pd.DataFrame, home_team: str, away_team: str,
) -> dict[str, Any]:
    """Return historical stats for a specific team matchup."""
    stats: dict[str, Any] = {"matches": 0, "home_wins": 0, "away_wins": 0, "draws": 0}

    mask = (df["home_team"] == home_team) & (df["away_team"] == away_team)
    matches = df[mask]
    stats["matches"] = len(matches)

    if stats["matches"] > 0 and "result" in matches.columns:
        stats["home_wins"] = int((matches["result"] == "H").sum())
        stats["away_wins"] = int((matches["result"] == "A").sum())
        stats["draws"] = int((matches["result"] == "D").sum())
        stats["last_results"] = matches["result"].tail(5).tolist()

    return stats
