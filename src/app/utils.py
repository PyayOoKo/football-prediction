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
def load_model(file_name: str | None = None) -> Any | None:
    """Load a trained model from the models directory.

    Tries ensemble model first (``ensemble_model.joblib``), then
    falls back to XGBoost (``xgboost_model.joblib``), then league model.
    """
    if file_name is not None:
        # Explicit file specified
        return _load_from_file(file_name)

    # Try ensemble model first (modern), then XGBoost, then league
    candidates = [
        ("ensemble_model.joblib", _try_load_ensemble),
        ("xgboost_model.joblib", _try_load_xgb),
        ("worldcup_xgboost.joblib", _try_load_xgb),
        ("league_xgboost.joblib", _try_load_xgb),
    ]

    for name, loader in candidates:
        model = loader(name)
        if model is not None:
            logger.info("Loaded model: %s", name)
            return model

    return None


def _load_from_file(file_name: str) -> Any | None:
    """Try loading a model from a specific file."""
    # Try as regular sklearn/xgboost model first
    try:
        from src.train import load_model as load_xgb
        return load_xgb(file_name)
    except Exception:
        pass

    # Try as ensemble model
    try:
        from src.ensemble import EnsembleModel
        model_path = config.paths.models / file_name
        if model_path.exists():
            return EnsembleModel.load(str(model_path))
    except Exception:
        pass

    return None


def _try_load_ensemble(name: str) -> Any | None:
    """Try loading an EnsembleModel from file."""
    try:
        from src.ensemble import EnsembleModel
        model_path = config.paths.models / name
        if model_path.exists():
            return EnsembleModel.load(str(model_path))
    except Exception:
        pass
    return None


def _try_load_xgb(name: str) -> Any | None:
    """Try loading a sklearn/XGBoost model from file."""
    try:
        from src.train import load_model as load_xgb
        return load_xgb(name)
    except Exception:
        pass
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


# ── Model diagnostic ───────────────────────────────────

@st.cache_data(show_spinner="Running model diagnostic ...")
def run_model_diagnostic(
    model: Any,
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Evaluate the model on test data and return per-class balance metrics.

    Returns a dict with:
    - ``accuracy``, ``baseline_home``, ``baseline_away``, ``baseline_draw``
    - ``log_loss``, ``n_test``
    - Per-class metrics: ``precision``, ``recall``, ``f1`` (each a dict of class->value)
    - ``confusion_matrix`` (2D list)
    - ``prediction_dist`` / ``actual_dist``
    - ``class_labels`` (list of 3 strings)
    """
    from src.feature_engineering import build_features, train_val_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        log_loss, confusion_matrix,
    )

    try:
        X, y = build_features(df, is_training=True)
        if X is None or len(X) < 10:
            return None

        splits = train_val_test_split(X, y)
        X_test = splits["X_test"]
        y_test = splits["y_test"]

        if len(y_test) < 5:
            return None

        # Align columns to model's expected features
        if hasattr(model, "get_booster"):
            try:
                model_features = model.get_booster().feature_names
                for col in model_features:
                    if col not in X_test.columns:
                        X_test[col] = 0.0
                X_test = X_test[model_features]
            except Exception:
                pass

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        ll = float(log_loss(y_test, y_proba))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

        # Per-class metrics
        precision = precision_score(y_test, y_pred, average=None, labels=[0, 1, 2])
        recall = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2])
        f1 = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])

        # Baselines
        baseline_home = float((y_test == 2).mean())
        baseline_away = float((y_test == 0).mean())
        baseline_draw = float((y_test == 1).mean())

        actual_dist = {
            "Home Win": int((y_test == 2).sum()),
            "Draw": int((y_test == 1).sum()),
            "Away Win": int((y_test == 0).sum()),
        }
        pred_dist = {
            "Home Win": int((y_pred == 2).sum()),
            "Draw": int((y_pred == 1).sum()),
            "Away Win": int((y_pred == 0).sum()),
        }

        class_labels = ["Away Win", "Draw", "Home Win"]

        return {
            "accuracy": acc,
            "log_loss": ll,
            "n_test": len(y_test),
            "baseline_home": baseline_home,
            "baseline_away": baseline_away,
            "baseline_draw": baseline_draw,
            "best_baseline": max(baseline_home, baseline_away, baseline_draw),
            "improvement": acc - max(baseline_home, baseline_away, baseline_draw),
            "precision": {"Away Win": precision[0], "Draw": precision[1], "Home Win": precision[2]},
            "recall": {"Away Win": recall[0], "Draw": recall[1], "Home Win": recall[2]},
            "f1": {"Away Win": f1[0], "Draw": f1[1], "Home Win": f1[2]},
            "confusion_matrix": cm,
            "class_labels": class_labels,
            "actual_dist": actual_dist,
            "prediction_dist": pred_dist,
        }
    except Exception as exc:
        logger.warning("Model diagnostic failed: %s", exc)
        return None


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
