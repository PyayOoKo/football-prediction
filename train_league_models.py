"""
Split-League Model Trainer — trains DC + Elo + XGBoost + LightGBM per league.

Reads match data from football_data.db, splits each league chronologically,
fits models independently, evaluates them, and saves per-league model files.

Tree models (XGBoost, LightGBM) are trained with:
  - Full feature engineering (rolling form, Elo, H2H, attack/defence ratios)
  - Time-decay sample weights (2-year halflife by default)
  - Chronological train/val split (no leakage)

Usage
-----
    python train_league_models.py                           # Train all leagues
    python train_league_models.py --leagues SE1 NO          # Specific leagues
    python train_league_models.py --leagues SE1 --no-trees  # DC+Elo only
    python train_league_models.py --min-matches 300         # Skip small leagues
    python train_league_models.py --eval-only                # Evaluate without retraining
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Models ───────────────────────────────────────────────

from src.dixon_coles import DixonColesModel
from src.elo import EloSystem

# ── Reuse league names from football_data package ────────

from football_data.config import LEAGUE_NAMES

# ── Fix Windows stdout encoding ──────────────────────────

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_league_models")


# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════

DB_PATH = Path("data/football_data.db")
MODELS_DIR = Path("models/per_league")
REPORTS_DIR = Path("reports")

# Train/val/test split: chronological
TRAIN_FRAC = 0.60
VAL_FRAC = 0.25

# Minimum matches required to train a league model
MIN_MATCHES = 200

# Test fraction computed implicitly as 1 - train_frac - val_frac

# Dixon-Coles settings
DC_DECAY_HALFLIFE = 1460.0  # 4 years
DC_USE_IMPORTANCE = False    # No tournament importance for domestic leagues

# Elo settings (default — overridden per league by LEAGUE_ELO_CONFIG)
ELO_K = 32
ELO_HOME_ADV = 100
ELO_INITIAL = 1500

# Per-league Elo overrides: second-tier leagues need higher K (more rating
# volatility) and lower home advantage (smaller crowd effect).
LEAGUE_ELO_CONFIG: dict[str, dict[str, int]] = {
    "SE1": {"k": 48, "home_advantage": 70},
    "NO2": {"k": 48, "home_advantage": 70},
    "FI2": {"k": 48, "home_advantage": 70},
}

# Tree model settings
TREE_N_ESTIMATORS = 300
TREE_MAX_DEPTH = 6
TREE_LEARNING_RATE = 0.05

# Minimum matches to attempt tree model training (need sufficient data)
MIN_MATCHES_FOR_TREES = 500


# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════


def load_league_data(league: str) -> pd.DataFrame:
    """Load matches for a specific league from the database."""
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result,
               home_odds, draw_odds, away_odds, season,
               home_xg, away_xg,
               home_shots, away_shots, home_shots_target, away_shots_target,
               home_corners, away_corners, home_fouls, away_fouls,
               home_yellow, away_yellow, home_red, away_red
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    return df


def get_available_leagues() -> list[dict[str, Any]]:
    """Return leagues available in the database with match counts."""
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT league, COUNT(*) as cnt
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        GROUP BY league
        ORDER BY cnt DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict(orient="records")


# ═══════════════════════════════════════════════════════════
#  Chronological Split
# ═══════════════════════════════════════════════════════════


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically (no leakage)."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


# ═══════════════════════════════════════════════════════════
#  Feature preparation for tree models
# ═══════════════════════════════════════════════════════════


def _prepare_for_tree_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a league DataFrame for the ``build_features()`` pipeline.

    Adds any missing columns that the full feature pipeline expects:
    - ``season`` (inferred from date if missing)
    - ``target`` (from ``result`` column)
    - Placeholder odds columns (will be ignored, but prevents warnings)
    - Converts object columns with numeric-looking values to proper types
    """
    df = df.copy()

    # Ensure target column exists
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"A": 0, "D": 1, "H": 2})

    # Ensure season exists (infer from date year if missing)
    if "season" not in df.columns:
        df["season"] = pd.to_datetime(df["date"]).dt.year.astype(str)
    else:
        # Clean season column: replace empty/None with year from date
        mask = df["season"].isna() | (df["season"].astype(str).str.strip() == "")
        if mask.any():
            df.loc[mask, "season"] = pd.to_datetime(df.loc[mask, "date"]).dt.year.astype(str)

    # Add placeholder league column for encoding stability
    if "league" not in df.columns:
        df["league"] = "UNKNOWN"

    # Convert object columns that should be numeric
    numeric_candidates = [
        "home_goals", "away_goals", "home_odds", "draw_odds", "away_odds",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
        "home_yellow", "away_yellow", "home_red", "away_red",
    ]
    for col in numeric_candidates:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df


def _train_tree_model(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[Any, pd.DataFrame | None, pd.Series | None, pd.DataFrame | None, pd.Series | None]:
    """Train an XGBoost or LightGBM model on per-league data.

    Builds features once, trains the model, and returns the model along with
    the feature matrices so callers can re-use them for evaluation/blending.

    Args:
        model_type: ``"xgboost"`` or ``"lightgbm"``
        train_df: Training match data (raw, not feature-engineered)
        val_df: Validation match data

    Returns:
        ``(model, X_train, y_train, X_val, y_val)`` — any may be ``None`` if
        training fails.
    """
    import time
    from config import config as cfg

    t0 = time.time()
    logger.info("  Preparing features for %s...", model_type)

    train_prep = _prepare_for_tree_features(train_df)
    val_prep = _prepare_for_tree_features(val_df)
    combined = pd.concat([train_prep, val_prep], ignore_index=True)

    # Disable expensive optional modules for tree model feature engineering
    _orig_vals = {
        "weather.enabled": cfg.weather.enabled,
        "referee.enabled": cfg.referee.enabled,
        "extended_features.enabled": cfg.extended_features.enabled,
        "dixon_coles.enabled": cfg.dixon_coles.enabled if hasattr(cfg, "dixon_coles") else False,
    }
    cfg.weather.enabled = False
    cfg.referee.enabled = False
    cfg.extended_features.enabled = False
    if hasattr(cfg, "dixon_coles"):
        cfg.dixon_coles.enabled = False

    X_full = None
    y_full = None
    try:
        from src.feature_engineering import build_features
        X_full, y_full = build_features(combined, is_training=True, use_cache=False)
    except Exception as exc:
        logger.warning("  Feature engineering failed for %s: %s", model_type, exc)
        return None, None, None, None, None
    finally:
        for key, val in _orig_vals.items():
            parts = key.split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is not None:
                try:
                    setattr(obj, parts[-1], val)
                except Exception:
                    pass

    if X_full is None or len(X_full) == 0:
        logger.warning("  No features generated for %s", model_type)
        return None, None, None, None, None

    n_train = len(train_prep)
    X_train_f = X_full.iloc[:n_train].copy()
    y_train_f = y_full.iloc[:n_train].copy()
    X_val_f = X_full.iloc[n_train:].copy()
    y_val_f = y_full.iloc[n_train:].copy()

    logger.info("  Feature matrix: train=%d x %d, val=%d x %d",
                X_train_f.shape[0], X_train_f.shape[1],
                X_val_f.shape[0], X_val_f.shape[1])

    from src.train import compute_sample_weights
    sample_weights = compute_sample_weights(train_prep, halflife_days=730.0)
    if sample_weights is not None:
        logger.info("  Time-decay sample weights computed (halflife=730d, mean=%.3f)",
                     float(np.mean(sample_weights)))

    _orig_type = cfg.train.model_type
    cfg.train.model_type = model_type

    try:
        from src.train import train_model
        model, history = train_model(
            X_train_f, y_train_f,
            X_val_f, y_val_f,
            sample_weight=sample_weights,
        )
        elapsed = time.time() - t0
        val_acc = history.get("val_accuracy", [0])[0]
        val_loss = history.get("val_loss", [0])[0]
        logger.info("  %s: val_acc=%.1f%%, val_log_loss=%.4f (%.1fs)",
                     model_type, val_acc * 100, val_loss, elapsed)
        return model, X_train_f, y_train_f, X_val_f, y_val_f
    except Exception as exc:
        logger.warning("  %s training failed: %s", model_type, exc)
        return None, None, None, None, None
    finally:
        cfg.train.model_type = _orig_type


# ═══════════════════════════════════════════════════════════
#  Training Functions
# ═══════════════════════════════════════════════════════════


def train_league_models(
    league: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    skip_trees: bool = False,
) -> dict[str, Any]:
    """Train DC + Elo + optionally XGBoost + LightGBM models for a single league.

    Args:
        league: League code (e.g. "SE1", "E0")
        train_df: Training match data
        val_df: Validation match data
        skip_trees: If True, skip tree model training (DC+Elo only)
    """
    logger.info("  Training Dixon-Coles...")
    dc = DixonColesModel(
        decay_halflife_days=DC_DECAY_HALFLIFE,
        use_importance=DC_USE_IMPORTANCE,
    )
    dc.fit(train_df, verbose=False)

    # Use league-specific Elo params if configured
    elo_cfg = LEAGUE_ELO_CONFIG.get(league, {})
    elo_k = elo_cfg.get("k", ELO_K)
    elo_home_adv = elo_cfg.get("home_advantage", ELO_HOME_ADV)
    logger.info("  Training Elo (k=%d, home_advantage=%d)...", elo_k, elo_home_adv)
    elo = EloSystem(k=elo_k, home_advantage=elo_home_adv, initial_rating=ELO_INITIAL)
    elo.process_matches(train_df)

    # Tree models (only if enough data and not skipped)
    xgb_model = None
    lgb_model = None
    train_trees = (
        not skip_trees
        and val_df is not None and len(val_df) > 0
        and len(train_df) >= MIN_MATCHES_FOR_TREES
    )

    # Tree model results: store model + feature matrices for reuse
    xgb_model = None
    xgb_X_val = None
    xgb_y_val = None
    lgb_model = None
    lgb_X_val = None
    lgb_y_val = None
    rf_model = None
    lr_model = None

    if train_trees:
        # Apply league-specific Elo overrides so tree models are trained
        # on features computed with the correct K/home_advantage for this league.
        from config import config as _cfg
        _elo_override = _cfg.elo.per_league.get(league)
        if _elo_override:
            _saved_k = _cfg.elo.k
            _saved_home = _cfg.elo.home_advantage
            _cfg.elo.k = _elo_override["k"]
            _cfg.elo.home_advantage = _elo_override["home_advantage"]
            logger.info("  Overrode Elo feature params: k=%d, home_advantage=%d",
                         _cfg.elo.k, _cfg.elo.home_advantage)

        try:
            logger.info("  Training XGBoost...")
            xgb_model, _, _, xgb_X_val, xgb_y_val = _train_tree_model("xgboost", train_df, val_df)

            logger.info("  Training LightGBM...")
            lgb_model, _, _, lgb_X_val, lgb_y_val = _train_tree_model("lightgbm", train_df, val_df)

            logger.info("  Training Random Forest...")
            rf_model, _, _, _, _ = _train_tree_model("random_forest", train_df, val_df)

            logger.info("  Training Logistic Regression...")
            lr_model, _, _, _, _ = _train_tree_model("logistic_regression", train_df, val_df)
        finally:
            if _elo_override:
                _cfg.elo.k = _saved_k
                _cfg.elo.home_advantage = _saved_home
    else:
        if skip_trees:
            logger.info("  Tree models skipped (--no-trees flag)")
        elif val_df is None or len(val_df) == 0:
            logger.info("  Tree models skipped (no validation set)")
        else:
            logger.info("  Tree models skipped (only %d train matches, need %d)",
                         len(train_df), MIN_MATCHES_FOR_TREES)

    # ── Evaluation ───────────────────────────────────────
    metrics: dict[str, Any] = {}
    calibrator = None  # Will be set if Platt calibration is fitted
    if val_df is not None and not val_df.empty:
        logger.info("  Evaluating on validation set (%d matches)...", len(val_df))
        dc_metrics = dc.evaluate(val_df)
        elo_metrics = elo.evaluate(val_df)
        metrics = {
            "dc": dc_metrics,
            "elo": elo_metrics,
            "n_train": len(train_df),
            "n_val": len(val_df),
        }

        from sklearn.metrics import log_loss as sk_log_loss
        actual_result = val_df["result"].map({"A": 0, "D": 1, "H": 2}).values
        y_onehot = np.zeros((len(actual_result), 3))
        for i, v in enumerate(actual_result):
            if not np.isnan(v) and 0 <= v <= 2:
                y_onehot[i, int(v)] = 1

        # DC+Elo Blend
        blend_probs = None
        try:
            dc_probs = dc.predict_proba(val_df)
            elo_probs = elo.predict_proba(val_df)
            blend_probs = (dc_probs + elo_probs) / 2.0

            blend_accuracy = float(np.mean(np.argmax(blend_probs, axis=1) == actual_result))
            blend_log_loss = float(sk_log_loss(actual_result, blend_probs))
            blend_brier = float(np.mean(np.sum((blend_probs - y_onehot) ** 2, axis=1)))

            metrics["blend"] = {
                "accuracy": round(blend_accuracy, 4),
                "log_loss": round(blend_log_loss, 4),
                "brier_score": round(blend_brier, 4),
            }
        except Exception as exc:
            logger.warning("  DC+Elo blend evaluation skipped: %s", exc)

        # Evaluate tree models (reuse feature matrices from training)
        if xgb_model is not None and xgb_X_val is not None and xgb_y_val is not None:
            try:
                xgb_probs = xgb_model.predict_proba(xgb_X_val)
                xgb_preds = np.argmax(xgb_probs, axis=1)
                xgb_acc = float(np.mean(xgb_preds == xgb_y_val.values))
                xgb_ll = float(sk_log_loss(xgb_y_val, xgb_probs))
                xgb_yoh = np.zeros((len(xgb_y_val), 3))
                for i, v in enumerate(xgb_y_val.values):
                    if not np.isnan(v) and 0 <= v <= 2:
                        xgb_yoh[i, int(v)] = 1
                xgb_brier = float(np.mean(np.sum((xgb_probs - xgb_yoh) ** 2, axis=1)))
                metrics["xgb"] = {
                    "accuracy": round(xgb_acc, 4),
                    "log_loss": round(xgb_ll, 4),
                    "brier_score": round(xgb_brier, 4),
                }
            except Exception as exc:
                logger.warning("  XGBoost evaluation failed: %s", exc)

        if lgb_model is not None and lgb_X_val is not None and lgb_y_val is not None:
            try:
                lgb_probs = lgb_model.predict_proba(lgb_X_val)
                lgb_preds = np.argmax(lgb_probs, axis=1)
                lgb_acc = float(np.mean(lgb_preds == lgb_y_val.values))
                lgb_ll = float(sk_log_loss(lgb_y_val, lgb_probs))
                lgb_yoh = np.zeros((len(lgb_y_val), 3))
                for i, v in enumerate(lgb_y_val.values):
                    if not np.isnan(v) and 0 <= v <= 2:
                        lgb_yoh[i, int(v)] = 1
                lgb_brier = float(np.mean(np.sum((lgb_probs - lgb_yoh) ** 2, axis=1)))
                metrics["lgb"] = {
                    "accuracy": round(lgb_acc, 4),
                    "log_loss": round(lgb_ll, 4),
                    "brier_score": round(lgb_brier, 4),
                }
            except Exception as exc:
                logger.warning("  LightGBM evaluation failed: %s", exc)

        # Full blend: DC + Elo + all available tree models
        if blend_probs is not None:
            try:
                full_blend = blend_probs.copy()
                tree_count = 0
                for m, X_v in [(xgb_model, xgb_X_val), (lgb_model, lgb_X_val)]:
                    if m is not None and X_v is not None:
                        full_blend += m.predict_proba(X_v)
                        tree_count += 1
                if tree_count > 0:
                    full_blend /= (1.0 + tree_count)

                    full_acc = float(np.mean(np.argmax(full_blend, axis=1) == actual_result))
                    full_ll = float(sk_log_loss(actual_result, full_blend))
                    full_brier = float(np.mean(np.sum((full_blend - y_onehot) ** 2, axis=1)))

                    metrics["full_blend"] = {
                        "accuracy": round(full_acc, 4),
                        "log_loss": round(full_ll, 4),
                        "brier_score": round(full_brier, 4),
                    }
            except Exception as exc:
                logger.warning("  Full blend evaluation skipped: %s", exc)

        # ── Post-training: Fit Platt calibrator on blend probs ──
        if blend_probs is not None and actual_result is not None and len(blend_probs) >= 50:
            try:
                from src.calibration import PlattScalingCalibrator
                logger.info("  Fitting Platt calibration on blend predictions...")
                calibrator = PlattScalingCalibrator(n_classes=3, max_iter=2000)
                calibrator.fit(blend_probs, actual_result)
                # Evaluate calibration on the validation set
                cal_probs = calibrator.transform(blend_probs)
                cal_ll = float(sk_log_loss(actual_result, cal_probs))
                cal_brier = float(np.mean(np.sum((cal_probs - y_onehot) ** 2, axis=1)))
                cal_acc = float(np.mean(np.argmax(cal_probs, axis=1) == actual_result))
                logger.info("    Calibrated: acc=%.1f%%, ll=%.4f, brier=%.4f (raw: %.1f%%, %.4f, %.4f)",
                             cal_acc * 100, cal_ll, cal_brier,
                             blend_accuracy * 100, blend_log_loss, blend_brier)
            except Exception as exc:
                logger.warning("  Calibration fitting failed: %s", exc)
                calibrator = None

    return {
        "league": league,
        "dc_model": dc,
        "elo_model": elo,
        "xgb_model": xgb_model,
        "lgb_model": lgb_model,
        "calibrator": calibrator,
        "metrics": metrics,
    }


def save_league_models(league: str, result: dict[str, Any]) -> Path:
    """Save per-league models to disk."""
    import joblib

    league_dir = MODELS_DIR / league
    league_dir.mkdir(parents=True, exist_ok=True)

    dc_path = league_dir / "dixon_coles.joblib"
    joblib.dump(result["dc_model"], dc_path)
    logger.info("    Dixon-Coles saved to %s", dc_path)

    elo_path = league_dir / "elo.joblib"
    joblib.dump(result["elo_model"], elo_path)
    logger.info("    Elo saved to %s", elo_path)

    # Save tree models if present
    xgb_model = result.get("xgb_model")
    if xgb_model is not None:
        xgb_path = league_dir / "xgboost.joblib"
        joblib.dump(xgb_model, xgb_path)
        logger.info("    XGBoost saved to %s", xgb_path)

    lgb_model = result.get("lgb_model")
    if lgb_model is not None:
        lgb_path = league_dir / "lightgbm.joblib"
        joblib.dump(lgb_model, lgb_path)
        logger.info("    LightGBM saved to %s", lgb_path)

    rf_model = result.get("rf_model")
    if rf_model is not None:
        rf_path = league_dir / "random_forest.joblib"
        joblib.dump(rf_model, rf_path)
        logger.info("    Random Forest saved to %s", rf_path)

    lr_model = result.get("lr_model")
    if lr_model is not None:
        lr_path = league_dir / "logistic_regression.joblib"
        joblib.dump(lr_model, lr_path)
        logger.info("    Logistic Regression saved to %s", lr_path)

    # Save calibrator if present
    calibrator = result.get("calibrator")
    if calibrator is not None:
        cal_path = league_dir / "blend_calibrator.joblib"
        joblib.dump(calibrator, cal_path)
        logger.info("    Platt calibrator saved to %s", cal_path)

    elo_cfg = LEAGUE_ELO_CONFIG.get(league, {})
    elo_k = elo_cfg.get("k", ELO_K)
    elo_home_adv = elo_cfg.get("home_advantage", ELO_HOME_ADV)

    meta = result.get("metrics", {})
    meta_path = league_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "league": league,
                "league_name": LEAGUE_NAMES.get(league, league),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_train": meta.get("n_train", 0),
                "n_val": meta.get("n_val", 0),
                "trained_models": {
                    "dc": True,
                    "elo": True,
                    "xgb": xgb_model is not None,
                    "lgb": lgb_model is not None,
                },
                "dc_decay_halflife_days": DC_DECAY_HALFLIFE,
                "elo_k": elo_k,
                "elo_home_advantage": elo_home_adv,
            },
            f,
            indent=2,
        )
    logger.info("    Metadata saved to %s", meta_path)
    return league_dir


def load_league_models(league: str) -> dict[str, Any] | None:
    """Load previously saved per-league models."""
    import joblib

    league_dir = MODELS_DIR / league
    dc_path = league_dir / "dixon_coles.joblib"
    elo_path = league_dir / "elo.joblib"
    if not dc_path.exists() or not elo_path.exists():
        return None

    dc = joblib.load(dc_path)
    elo = joblib.load(elo_path)

    # Load tree models if they exist
    xgb_path = league_dir / "xgboost.joblib"
    xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None

    lgb_path = league_dir / "lightgbm.joblib"
    lgb_model = joblib.load(lgb_path) if lgb_path.exists() else None

    rf_path = league_dir / "random_forest.joblib"
    rf_model = joblib.load(rf_path) if rf_path.exists() else None

    lr_path = league_dir / "logistic_regression.joblib"
    lr_model = joblib.load(lr_path) if lr_path.exists() else None

    meta = {}
    meta_path = league_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return {
        "league": league,
        "dc_model": dc,
        "elo_model": elo,
        "xgb_model": xgb_model,
        "lgb_model": lgb_model,
        "rf_model": rf_model,
        "lr_model": lr_model,
        "metadata": meta,
    }


# ═══════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════


def print_league_report(league: str, result: dict[str, Any]):
    """Print a report for a league's training."""
    metrics = result.get("metrics", {})
    league_name = LEAGUE_NAMES.get(league, league)

    print()
    print("=" * 60)
    print(f"  {league} - {league_name}")
    print("=" * 60)

    if not metrics:
        print("  (No validation metrics - trained only)")
        return

    print(f"  Training set:   {metrics.get('n_train', 0):,} matches")
    print(f"  Validation set: {metrics.get('n_val', 0):,} matches")
    print()

    model_configs = []
    if result.get("dc_model"):
        model_configs.append(("dc", "Dixon-Coles"))
    if result.get("elo_model"):
        model_configs.append(("elo", "Elo"))
    if result.get("xgb_model"):
        model_configs.append(("xgb", "XGBoost"))
    if result.get("lgb_model"):
        model_configs.append(("lgb", "LightGBM"))
    model_configs.append(("blend", "Blend (DC+Elo)"))
    if result.get("xgb_model") or result.get("lgb_model"):
        model_configs.append(("full_blend", "Full Blend (all models)"))

    for model_key, model_label in model_configs:
        m = metrics.get(model_key)
        if m:
            print(f"  {model_label}:")
            print(f"    Accuracy:    {m.get('accuracy', 0)*100:.1f}%")
            print(f"    Log Loss:    {m.get('log_loss', 0):.4f}")
            print(f"    Brier Score: {m.get('brier_score', 0):.4f}")
            print()


def generate_comparison_report(results: list[dict[str, Any]]):
    """Generate a cross-league comparison table."""
    print()
    print("=" * 80)
    print("  CROSS-LEAGUE MODEL COMPARISON")
    print("=" * 80)
    header = f"  {'League':6s} {'Model':18s} {'Accuracy':>10s} {'Log Loss':>10s} {'Brier':>8s} {'n_train':>8s}"
    print(header)
    print(f"  {'-'*6} {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    rows = []
    for r in results:
        league = r["league"]
        metrics = r.get("metrics", {})

        model_order = [
            ("dc", "Dixon-Coles"),
            ("elo", "Elo"),
            ("blend", "DC+Elo Avg"),
            ("xgb", "XGBoost"),
            ("lgb", "LightGBM"),
            ("full_blend", "All-Model Blend"),
        ]
        for model_key, model_label in model_order:
            m = metrics.get(model_key)
            if m:
                acc = m.get("accuracy", 0) * 100
                ll = m.get("log_loss", 0)
                brier = m.get("brier_score", 0)
                n_train = metrics.get("n_train", 0)
                rows.append((league, model_label, acc, ll, brier, n_train))
                print(f"  {league:6s} {model_label:18s} {acc:>9.1f}% {ll:>10.4f} {brier:>8.4f} {n_train:>8,}")

    print()

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "per_league_comparison.json"
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leagues_trained": len(results),
        "results": [
            {"league": r["league"], "metrics": r.get("metrics", {})}
            for r in results
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Comparison report saved to %s", report_path)

    return rows


# ═══════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Train per-league models (DC + Elo + XGBoost + LightGBM)"
    )
    parser.add_argument(
        "--leagues", nargs="+",
        help="Leagues to train (default: all available)"
    )
    parser.add_argument(
        "--min-matches", type=int, default=MIN_MATCHES,
        help=f"Minimum matches required to train (default: {MIN_MATCHES})"
    )
    parser.add_argument(
        "--no-trees", action="store_true",
        help="Skip tree model training (DC+Elo only)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Load saved models and evaluate only (no retraining)"
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover leagues
    if args.leagues:
        league_codes = args.leagues
    else:
        available = get_available_leagues()
        league_codes = [
            l["league"] for l in available
            if l["cnt"] >= args.min_matches
        ]
        logger.info("Auto-detected leagues with >= %d matches: %s",
                     args.min_matches, league_codes)

    print()
    print("=" * 60)
    print("  SPLIT-LEAGUE MODEL TRAINING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    results: list[dict[str, Any]] = []

    for league in league_codes:
        league_name = LEAGUE_NAMES.get(league, league)
        print()
        print("-" * 60)
        print(f"  {league} - {league_name}")

        if args.eval_only:
            saved = load_league_models(league)
            if saved is None:
                logger.warning("  No saved models for %s - skipping", league)
                continue
            logger.info("  Loaded saved models")
            result = {
                "league": league,
                "dc_model": saved["dc_model"],
                "elo_model": saved["elo_model"],
                "xgb_model": saved.get("xgb_model"),
                "lgb_model": saved.get("lgb_model"),
                "metrics": {},
            }

        else:
            df = load_league_data(league)
            if len(df) < args.min_matches:
                logger.warning("  Only %d matches - need %d, skipping",
                               len(df), args.min_matches)
                continue

            logger.info("  Loaded %d matches", len(df))
            logger.info("  Date range: %s to %s",
                        df["date"].iloc[0][:10], df["date"].iloc[-1][:10])

            train_df, val_df, _ = chronological_split(df)
            logger.info("  Split: %d train / %d val",
                        len(train_df), len(val_df))

            result = train_league_models(league, train_df, val_df,
                                          skip_trees=args.no_trees)
            save_league_models(league, result)

        print_league_report(league, result)
        results.append(result)

    # Cross-league comparison
    if len(results) >= 2:
        generate_comparison_report(results)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
