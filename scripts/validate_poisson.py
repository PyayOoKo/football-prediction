#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Poisson Model Validation                                                 ║
║                                                                           ║
║  Validates the Poisson goal-distribution model against a logistic         ║
║  regression baseline using a chronological train/test split.              ║
║                                                                           ║
║  Metrics tracked:                                                         ║
║  - Brier Score (multi-class MSE of predicted vs true probabilities)       ║
║  - Log Loss (cross-entropy)                                              ║
║  - Accuracy (1X2 match outcome)                                          ║
║  - BTTS Accuracy & Brier                                                 ║
║  - Over/Under 2.5 Accuracy & Brier                                       ║
║  - Calibration (reliability diagram)                                     ║
║                                                                           ║
║  Outputs:                                                                 ║
║  - models/poisson_model.joblib                                           ║
║  - reports/poisson_validation_{timestamp}.json                           ║
║  - reports/poisson_vs_baseline_{timestamp}.json                          ║
║  - reports/poisson_calibration_{timestamp}.png (reliability curve)      ║
║                                                                           ║
║  Usage:                                                                   ║
║      python scripts/validate_poisson.py                                   ║
║      python scripts/validate_poisson.py --data data/raw/results.csv       ║
║      python scripts/validate_poisson.py --quiet                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from config import config
from src.poisson_model import PoissonModel

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

# Default chronological split date (train before, test from)
_DEFAULT_TRAIN_END = "2023-01-01"
_DEFAULT_TEST_END = "2026-12-31"

# Required columns in the input CSV
_REQUIRED_COLS = [
    "date", "home_team", "away_team", "result",
    "home_goals", "away_goals",
]

# Output paths
_MODEL_DIR = PROJECT_ROOT / "models"
_REPORT_DIR = PROJECT_ROOT / "reports"


# ═══════════════════════════════════════════════════════════
#  Data loading & splitting
# ═══════════════════════════════════════════════════════════


def load_data(
    path: str | Path,
    min_date: str | None = "2016-01-01",
) -> pd.DataFrame:
    """Load match data from CSV and validate required columns.

    Note on Feature Store
    ---------------------
    The Poisson model requires **raw match data** (team names, goals, results)
    to compute its expanding-window attack/defence strengths.  The Feature
    Store stores *computed* feature values (keyed by definition + entity),
    not raw match records.  Therefore raw match data is loaded from CSV
    rather than queried from the Feature Store.

    If a Feature Store backend were to be used, it would require a dedicated
    "match" entity table with goal and result columns — a schema extension
    beyond the current Feature Store model.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    min_date : str, optional
        Earliest match date to include (default "2016-01-01").
        Set to None to include all available data.

    Returns
    -------
    pd.DataFrame
        Loaded and sorted DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing_cols = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Expected at least: {_REQUIRED_COLS}"
        )

    # Save original dates before any parsing (for fallback)
    raw_dates = df["date"].copy()
    # Parse dates: try ISO format first (no dayfirst), then dayfirst=True
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    # If most dates are NaT, try with dayfirst=True on the saved original
    if df["date"].isna().sum() > len(df) * 0.5:
        logger.warning("ISO format parsing failed — retrying with dayfirst=True")
        df["date"] = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop rows with invalid dates
    before = len(df)
    df = df.dropna(subset=["date"])
    if len(df) < before:
        logger.warning("Dropped %d rows with invalid dates", before - len(df))

    # Filter to starting date (default: 2016-01-01 as per requirement)
    if min_date is not None:
        cutoff = pd.Timestamp(min_date)
        df = df[df["date"] >= cutoff].copy()

    # Filter to matches with known results (needed for evaluation)
    df = df[df["result"].notna() & df["result"].isin(["H", "D", "A"])].copy()

    logger.info(
        "Loaded %d matches from %s to %s (%s)%s",
        len(df),
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
        path.name,
        f" — filtered from {min_date}" if min_date else "",
    )
    return df


def chronological_split(
    df: pd.DataFrame,
    train_end: str = _DEFAULT_TRAIN_END,
    test_end: str = _DEFAULT_TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Match data sorted by date.
    train_end : str
        Cutoff date for training data (exclusive). Matches before this
        date go to training.
    test_end : str
        End date for test data (inclusive). Matches from train_end to
        test_end go to testing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_train, df_test)``.
    """
    train_cutoff = pd.Timestamp(train_end)
    test_cutoff = pd.Timestamp(test_end)

    df_train = df[df["date"] < train_cutoff].copy()
    df_test = df[
        (df["date"] >= train_cutoff) & (df["date"] <= test_cutoff)
    ].copy()

    logger.info(
        "Chronological split — train: %d matches (before %s), "
        "test: %d matches (%s to %s)",
        len(df_train), train_end,
        len(df_test), train_end, test_end,
    )
    return df_train, df_test


def get_targets(df: pd.DataFrame) -> np.ndarray:
    """Convert result column to numeric targets.

    Returns
    -------
    np.ndarray
        0 = Away win, 1 = Draw, 2 = Home win.
        Unmapped values are set to -1 (should not occur with clean data).
    """
    mapping = {"A": 0, "D": 1, "H": 2}
    result = df["result"].map(mapping).fillna(-1).values.astype(int)
    # Sanity check: warn if any -1 values
    n_bad = int((result == -1).sum())
    if n_bad > 0:
        logger.warning("get_targets: %d rows have unmapped result values", n_bad)
    return result


# ═══════════════════════════════════════════════════════════
#  Metrics computation
# ═══════════════════════════════════════════════════════════


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Compute all evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (0=Away, 1=Draw, 2=Home).
    probs : np.ndarray
        Predicted probabilities, shape (n, 3) [away, draw, home].
    model_name : str
        Label for the model (used in log output).

    Returns
    -------
    dict[str, Any]
        Metrics dictionary with: accuracy, log_loss, brier_score.
    """
    from sklearn.metrics import log_loss as sk_log_loss

    pred_labels = np.argmax(probs, axis=1)

    # Accuracy
    accuracy = float(np.mean(pred_labels == y_true))

    # Log loss
    ll = float(sk_log_loss(y_true, probs))

    # Multi-class Brier score (MSE of probability vs one-hot)
    y_onehot = np.zeros((len(y_true), 3))
    for i, v in enumerate(y_true):
        if 0 <= v <= 2:
            y_onehot[i, int(v)] = 1
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    logger.info(
        "%s — accuracy=%.4f  log_loss=%.4f  brier=%.4f",
        model_name, accuracy, ll, brier,
    )

    return {
        "accuracy": round(accuracy, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "n": int(len(y_true)),
    }


def compute_btts_metrics(
    df_test: pd.DataFrame,
    pred_btts_probs: np.ndarray,
) -> dict[str, Any]:
    """Compute BTTS evaluation metrics.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data with home_goals, away_goals columns.
    pred_btts_probs : np.ndarray
        Predicted BTTS probabilities.

    Returns
    -------
    dict[str, Any]
        btts_accuracy, btts_brier.
    """
    actual_btts = (
        (df_test["home_goals"].values.astype(float) > 0)
        & (df_test["away_goals"].values.astype(float) > 0)
    ).astype(float)

    pred_btts = (pred_btts_probs > 0.5).astype(float)

    accuracy = float(np.mean(pred_btts == actual_btts))
    brier = float(np.mean((pred_btts_probs - actual_btts) ** 2))

    return {
        "btts_accuracy": round(accuracy, 4),
        "btts_brier": round(brier, 4),
    }


def compute_ou_metrics(
    df_test: pd.DataFrame,
    pred_ou_probs: np.ndarray,
    threshold: float = 2.5,
) -> dict[str, Any]:
    """Compute Over/Under evaluation metrics.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data with home_goals, away_goals columns.
    pred_ou_probs : np.ndarray
        Predicted Over threshold probabilities.
    threshold : float
        Over/Under threshold (default 2.5).

    Returns
    -------
    dict[str, Any]
        over_under_accuracy, over_under_brier.
    """
    actual_total = (
        df_test["home_goals"].values.astype(float)
        + df_test["away_goals"].values.astype(float)
    )
    actual_ou = (actual_total > threshold).astype(float)

    pred_ou = (pred_ou_probs > 0.5).astype(float)

    accuracy = float(np.mean(pred_ou == actual_ou))
    brier = float(np.mean((pred_ou_probs - actual_ou) ** 2))

    return {
        f"over_under_{threshold:.1f}_accuracy".replace(".", "_"): round(accuracy, 4),
        f"over_under_{threshold:.1f}_brier".replace(".", "_"): round(brier, 4),
    }


# ═══════════════════════════════════════════════════════════
#  Calibration
# ═══════════════════════════════════════════════════════════


def calibration_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration (reliability) statistics.

    Uses a one-vs-rest approach for multi-class: for each class,
    bin the predicted probabilities and measure accuracy within
    each bin.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0, 1, 2).
    probs : np.ndarray
        Predicted probabilities, shape (n, 3).
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    dict[str, Any]
        Calibration statistics including ECE and per-class metrics.
    """
    n_classes = probs.shape[1]
    ece_total = 0.0
    per_class: list[dict[str, Any]] = []

    for c in range(n_classes):
        y_binary = (y_true == c).astype(float)
        p = probs[:, c]

        bins = np.linspace(0, 1, n_bins + 1)
        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            in_bin = (p >= bins[i]) & (p < bins[i + 1])
            count = in_bin.sum()
            bin_counts[i] = count
            if count > 0:
                bin_acc[i] = y_binary[in_bin].mean()
                bin_conf[i] = p[in_bin].mean()

        # Miscalibration per bin
        gaps = np.abs(bin_acc - bin_conf)
        total_count = bin_counts.sum()
        if total_count > 0:
            ece_class = float(np.sum(bin_counts * gaps) / total_count)
        else:
            ece_class = 0.0
        ece_total += ece_class * bin_counts.sum()

        class_labels = ["Away", "Draw", "Home"]
        per_class.append({
            "class": class_labels[c],
            "ece": round(ece_class, 4),
            "bins": [
                {
                    "bin_center": round(float((bins[i] + bins[i + 1]) / 2), 3),
                    "count": int(bin_counts[i]),
                    "accuracy": round(float(bin_acc[i]), 4) if bin_counts[i] > 0 else None,
                    "confidence": round(float(bin_conf[i]), 4) if bin_counts[i] > 0 else None,
                    "gap": round(float(gaps[i]), 4) if bin_counts[i] > 0 else None,
                }
                for i in range(n_bins)
            ],
        })

    total_count = len(y_true) * n_classes
    ece_overall = round(ece_total / total_count, 4) if total_count > 0 else 0.0

    return {
        "ece": ece_overall,
        "n_bins": n_bins,
        "per_class": per_class,
    }


def save_calibration_plot(
    y_true: np.ndarray,
    probs_poisson: np.ndarray,
    probs_baseline: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> str | None:
    """Generate and save a calibration (reliability) diagram.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    probs_poisson : np.ndarray
        Poisson predicted probabilities, shape (n, 3).
    probs_baseline : np.ndarray, optional
        Baseline predicted probabilities.
    save_path : str | Path, optional
        Where to save the plot.

    Returns
    -------
    str | None
        Path to saved image, or None if matplotlib not available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping calibration plot")
        return None

    if save_path is None:
        save_path = _REPORT_DIR / "poisson_calibration.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    class_labels = ["Away Win", "Draw", "Home Win"]
    n_classes = 3

    fig, axes = plt.subplots(1, n_classes, figsize=(15, 4))
    fig.suptitle("Calibration (Reliability) Diagrams", fontsize=14, y=1.02)

    for c in range(n_classes):
        ax = axes[c]
        y_binary = (y_true == c).astype(float)

        # Compute calibration curve for Poisson
        p_poisson = probs_poisson[:, c]
        bin_acc_poisson = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        for i in range(n_bins):
            in_bin = (p_poisson >= bins[i]) & (p_poisson < bins[i + 1])
            if in_bin.sum() > 0:
                bin_acc_poisson[i] = y_binary[in_bin].mean()
                bin_conf[i] = p_poisson[in_bin].mean()

        ax.plot(bin_conf, bin_acc_poisson, "o-", label="Poisson", color="#1f77b4", linewidth=2)
        ax.fill_between(bin_conf, bin_acc_poisson, bin_conf, alpha=0.1, color="#1f77b4")

        # Baseline if provided
        if probs_baseline is not None:
            p_baseline = probs_baseline[:, c]
            bin_acc_baseline = np.zeros(n_bins)
            for i in range(n_bins):
                in_bin = (p_baseline >= bins[i]) & (p_baseline < bins[i + 1])
                if in_bin.sum() > 0:
                    bin_acc_baseline[i] = y_binary[in_bin].mean()
            ax.plot(bin_conf, bin_acc_baseline, "s--", label="LR Baseline", color="#ff7f0e", linewidth=2)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(class_labels[c])
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration plot saved to %s", save_path)
    return str(save_path)


# ═══════════════════════════════════════════════════════════
#  Baseline model (Logistic Regression)
# ═══════════════════════════════════════════════════════════


def _compute_simple_rolling_features(
    df: pd.DataFrame,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
) -> pd.DataFrame:
    """Compute simple per-team rolling averages (leakage-free).

    For each match chronologically, computes each team's average goals
    scored and conceded using ONLY matches that occurred before it.
    This is an independent feature set — NOT derived from the Poisson model.

    Returns a DataFrame with the same index as *df* and columns:
    - ``home_goals_avg``, ``home_conceded_avg``
    - ``away_goals_avg``, ``away_conceded_avg``
    """
    df = df.copy()
    n = len(df)

    home_goals_avg = np.zeros(n)
    home_conceded_avg = np.zeros(n)
    away_goals_avg = np.zeros(n)
    away_conceded_avg = np.zeros(n)

    # Running team stats: {team: [total_scored, total_conceded, matches]}
    team_stats: dict[str, list[float]] = {}

    for idx in range(n):
        home = df[home_team_col].iloc[idx]
        away = df[away_team_col].iloc[idx]
        hg = float(df[home_goals_col].iloc[idx] or 0)
        ag = float(df[away_goals_col].iloc[idx] or 0)

        def _get_avg(team: str, stat_type: str = "scored") -> float:
            s = team_stats.get(team)
            if s is None or s[2] == 0:
                return 0.0
            total = s[0] if stat_type == "scored" else s[1]
            return total / s[2]

        home_goals_avg[idx] = _get_avg(home, "scored")
        home_conceded_avg[idx] = _get_avg(home, "conceded")
        away_goals_avg[idx] = _get_avg(away, "scored")
        away_conceded_avg[idx] = _get_avg(away, "conceded")

        # Update stats with current match's result (for future matches)
        for team, scored, conceded in (
            (home, hg, ag),
            (away, ag, hg),
        ):
            s = team_stats.get(team)
            if s is None:
                team_stats[team] = [scored, conceded, 1.0]
            else:
                s[0] += scored
                s[1] += conceded
                s[2] += 1.0

    # Compute global average for fallback when teams are unseen
    global_avg = None
    if team_stats:
        total_scored = sum(s[0] for s in team_stats.values())
        total_matches = sum(s[2] for s in team_stats.values())
        if total_matches > 0:
            global_avg = total_scored / total_matches

    result = pd.DataFrame({
        "home_goals_avg": home_goals_avg,
        "home_conceded_avg": home_conceded_avg,
        "away_goals_avg": away_goals_avg,
        "away_conceded_avg": away_conceded_avg,
    }, index=df.index)

    # For teams with zero appearances in team_stats, their features are 0.0.
    # Replace those with global average to avoid "team scores nothing" bias.
    # Only replace where ALL four features are zero (truly unseen team).
    if global_avg is not None and global_avg > 0:
        unseen_mask = (
            (result["home_goals_avg"] == 0.0)
            & (result["home_conceded_avg"] == 0.0)
            & (result["away_goals_avg"] == 0.0)
            & (result["away_conceded_avg"] == 0.0)
        )
        n_unseen = unseen_mask.sum()
        if n_unseen > 0:
            logger.info(
                "Replacing %d unseen-team rows with global avg (%.3f)",
                n_unseen, global_avg,
            )
            for col in result.columns:
                result.loc[unseen_mask, col] = global_avg

    return result


def train_baseline_lr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Train a logistic regression baseline using independent rolling features.

    Features are simple per-team rolling averages (goals scored/conceded)
    computed from an expanding window of **pre-match** data only.
    These are independent of the Poisson model, providing a fair baseline
    comparison.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training match data.
    df_test : pd.DataFrame
        Test match data.

    Returns
    -------
    tuple[Any, np.ndarray, dict[str, Any]]
        ``(lr_model, test_probs, metrics)``.
    """
    from sklearn.linear_model import LogisticRegression

    # Build independent rolling features for both train and test
    # We fit the featurizer on combined data to get rolling stats for test
    combined = pd.concat([df_train, df_test], ignore_index=True)
    feat_df = _compute_simple_rolling_features(combined)

    n_train = len(df_train)
    train_feats = feat_df.iloc[:n_train].fillna(0).values
    test_feats = feat_df.iloc[n_train:].fillna(0).values

    y_train = get_targets(df_train)
    y_test = get_targets(df_test)

    # Train LR
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
        C=1.0,
        class_weight="balanced",
    )
    lr.fit(train_feats, y_train)
    test_probs = lr.predict_proba(test_feats)

    # Evaluate
    metrics = compute_metrics(y_test, test_probs, "LR Baseline")

    logger.info(
        "LR baseline trained — accuracy=%.4f (features: home/away goals avg)",
        metrics["accuracy"],
    )
    return lr, test_probs, metrics


# ═══════════════════════════════════════════════════════════
#  Main validation pipeline
# ═══════════════════════════════════════════════════════════


def run_validation(
    data_path: str | Path,
    train_end: str = _DEFAULT_TRAIN_END,
    test_end: str = _DEFAULT_TEST_END,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the full Poisson model validation pipeline.

    Parameters
    ----------
    data_path : str | Path
        Path to the match data CSV.
    train_end : str
        Training cut-off date.
    test_end : str
        Test end date.
    quiet : bool
        Suppress logging output if True.

    Returns
    -------
    dict[str, Any]
        Full validation report with all metrics, comparison, and paths.
    """
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "data_path": str(data_path),
        "train_end": train_end,
        "test_end": test_end,
        "model_type": "PoissonModel",
        "baseline_type": "LogisticRegression",
    }

    print("\n" + "=" * 70)
    print("  POISSON MODEL VALIDATION")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────
    print(f"\n[1/7] Loading data from {data_path}...")
    df = load_data(data_path)
    report["total_matches"] = len(df)
    report["date_range"] = {
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
    }

    # ── 2. Split chronologically ──────────────────────
    print(f"\n[2/7] Splitting chronologically (train < {train_end})...")
    df_train, df_test = chronological_split(df, train_end, test_end)
    report["train_size"] = len(df_train)
    report["test_size"] = len(df_test)

    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError(
            f"Empty train ({len(df_train)}) or test ({len(df_test)}) set. "
            f"Check date range and split dates."
        )

    y_test = get_targets(df_test)

    # ── 3. Train Poisson model ────────────────────────
    print(f"\n[3/7] Training Poisson model on {len(df_train)} matches...")
    poisson = PoissonModel(min_matches=0, max_goals=8)

    # Fit using add_poisson_features() for proper expanding-window
    # computation (leakage-free)
    poisson.add_poisson_features(df_train)

    logger.info(
        "Poisson fitted — μ_home=%.3f, μ_away=%.3f, %d teams",
        poisson.league_avg_home, poisson.league_avg_away,
        len(poisson.team_strengths),
    )

    # ── 4. Predict on test set ────────────────────────
    print(f"\n[4/7] Predicting {len(df_test)} test matches...")
    test_preds = poisson.predict_matches(df_test)

    # Detailed metrics
    probs_poisson = np.column_stack([
        test_preds["away_win_prob"].values,
        test_preds["draw_prob"].values,
        test_preds["home_win_prob"].values,
    ])

    poisson_metrics = compute_metrics(y_test, probs_poisson, "Poisson")
    btts_metrics = compute_btts_metrics(df_test, test_preds["btts_prob"].values)
    ou_metrics = compute_ou_metrics(
        df_test,
        test_preds.get("over_2_5_prob", test_preds.get("over_2.5_prob", pd.Series([0.5] * len(df_test)))).values,
    )

    poisson_results = {**poisson_metrics, **btts_metrics, **ou_metrics}

    # ── 5. Train LR baseline ──────────────────────────
    print(f"\n[5/7] Training Logistic Regression baseline...")
    lr_model, lr_probs, lr_metrics = train_baseline_lr(df_train, df_test)

    report["poisson"] = poisson_results
    report["baseline"] = lr_metrics

    # ── 6. Comparison ─────────────────────────────────
    print(f"\n[6/7] Computing comparison...")
    comparison = {}
    for metric in ["accuracy", "log_loss", "brier_score"]:
        if metric in poisson_results and metric in lr_metrics:
            poisson_val = poisson_results[metric]
            baseline_val = lr_metrics[metric]
            diff = poisson_val - baseline_val
            better = "poisson" if (
                (metric in ["accuracy"] and diff > 0)
                or (metric in ["log_loss", "brier_score"] and diff < 0)
            ) else "baseline" if diff != 0 else "tie"
            comparison[metric] = {
                "poisson": poisson_val,
                "baseline": baseline_val,
                "difference": round(diff, 4),
                "better": better,
            }

    report["comparison"] = comparison
    report["acceptance"] = {
        "poisson_trains_without_errors": True,
        "brier_score_vs_baseline": poisson_results["brier_score"] < lr_metrics["brier_score"],
        "log_loss_vs_baseline": poisson_results["log_loss"] < lr_metrics["log_loss"],
        "accuracy_vs_baseline": poisson_results["accuracy"] > lr_metrics["accuracy"],
        "brier_score_target": poisson_results["brier_score"] < 0.45,
    }

    # ── 7. Calibration ────────────────────────────────
    print(f"\n[7/7] Computing calibration and saving outputs...")
    cal = calibration_report(y_test, probs_poisson)

    # Save calibration plot
    plot_path = save_calibration_plot(y_test, probs_poisson, lr_probs)
    if plot_path:
        report["calibration_plot"] = plot_path

    # Print summary
    duration = time.time() - start_time
    report["duration_seconds"] = round(duration, 2)

    print("\n" + "=" * 70)
    print("  RESULTS".center(68))
    print("=" * 70)
    print(f"\n  Duration:           {duration:.2f}s")
    print(f"  Training matches:   {len(df_train)}")
    print(f"  Test matches:       {len(df_test)}")
    print(f"  Teams in model:     {len(poisson.team_strengths)}")
    print(f"  League avg:         home={poisson.league_avg_home:.3f}, away={poisson.league_avg_away:.3f}")

    print(f"\n  {'Metric':<25s} {'Poisson':<12s} {'Baseline (LR)':<15s} {'Better':<10s}")
    print(f"  {'-'*60}")
    for metric in ["accuracy", "log_loss", "brier_score"]:
        if metric in comparison:
            c = comparison[metric]
            print(f"  {metric:<25s} {c['poisson']:<12.4f} {c['baseline']:<15.4f} {c['better']:<10s}")

    print(f"\n  {'Metric':<25s} {'Value':<12s}")
    print(f"  {'-'*37}")
    for key in ["btts_accuracy", "btts_brier", "over_under_2_5_accuracy", "over_under_2_5_brier"]:
        if key in poisson_results:
            print(f"  {key:<25s} {poisson_results[key]:<12.4f}")

    print(f"\n  {'Acceptance Criteria':<50s} {'Status'}")
    print(f"  {'-'*60}")
    for criterion, passed in report["acceptance"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {criterion:<50s} {status}")

    # ── Save outputs ─────────────────────────────────
    print(f"\n  Saving outputs...")

    # Save model
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "poisson_model.joblib"
    joblib.dump(poisson, model_path)
    report["model_path"] = str(model_path)
    print(f"    Model:  {model_path}")

    # Validate model loads
    loaded = joblib.load(model_path)
    assert loaded.fitted, "Loaded model must be fitted!"
    print(f"    [OK] Model loads and validates correctly")

    # Save validation report
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    val_report_path = _REPORT_DIR / f"poisson_validation_{timestamp}.json"
    with open(val_report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    Report: {val_report_path}")

    # Save comparison report
    comp_report_path = _REPORT_DIR / f"poisson_vs_baseline_{timestamp}.json"
    comparison_report = {
        "timestamp": timestamp,
        "model": "PoissonModel",
        "baseline": "LogisticRegression",
        "comparison": comparison,
        "acceptance": report["acceptance"],
        "poisson_metrics": poisson_results,
        "baseline_metrics": lr_metrics,
        "calibration": cal,
    }
    with open(comp_report_path, "w") as f:
        json.dump(comparison_report, f, indent=2, default=str)
    print(f"    Comparison: {comp_report_path}")

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE".center(68))
    print("=" * 70)

    return report


# ═══════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════


def main() -> int:
    """Parse arguments and run validation."""
    parser = argparse.ArgumentParser(
        description="Validate the Poisson prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_poisson.py
  python scripts/validate_poisson.py --data data/raw/results.csv
  python scripts/validate_poisson.py --train-end 2022-01-01 --test-end 2023-12-31
  python scripts/validate_poisson.py --quiet
        """,
    )
    parser.add_argument(
        "--data",
        default=str(config.paths.raw / config.data_collection.output_file),
        help=f"Path to match data CSV (default: {config.paths.raw / config.data_collection.output_file})",
    )
    parser.add_argument(
        "--train-end",
        default=_DEFAULT_TRAIN_END,
        help=f"Training data cut-off date (default: {_DEFAULT_TRAIN_END})",
    )
    parser.add_argument(
        "--test-end",
        default=_DEFAULT_TEST_END,
        help=f"Test data end date (default: {_DEFAULT_TEST_END})",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    try:
        run_validation(
            data_path=args.data,
            train_end=args.train_end,
            test_end=args.test_end,
            quiet=args.quiet,
        )
        return 0
    except Exception as e:
        logger.error("Validation failed: %s", e, exc_info=True)
        print(f"\n[FAIL] Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
