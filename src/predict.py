"""
Predict — generate match outcome predictions from a trained model.

Supports saving predictions to CSV, JSON, or printing to the console.

Typical usage::

    from src.predict import predict_fixtures
    predictions = predict_fixtures(model, fixtures_df)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import config as _global_config
from src.confidence_scoring import ConfidenceScorer

logger = logging.getLogger(__name__)


def predict_fixtures(
    model: Any,
    fixtures_df: pd.DataFrame,
    output_path: str | Path | None = None,
    individual_probs: dict[str, np.ndarray] | None = None,
    calibration_brier: float | None = None,
    config: Any | None = None,
) -> pd.DataFrame:
    """Generate outcome predictions for a set of fixtures.

    Parameters
    ----------
    model : Any
        Trained model with a ``predict`` and/or ``predict_proba`` method.
    fixtures_df : pd.DataFrame
        Feature-engineered fixture data.
    output_path : str | Path, optional
        If given, save predictions to this file (format inferred from extension:
        ``.csv`` → CSV, ``.json`` → JSON, else console).
    individual_probs : dict[str, np.ndarray], optional
        Per-model probability arrays for ensemble agreement scoring.
        Each value must be shape ``(n, 3)``.
    calibration_brier : float, optional
        Brier score from a held-out validation set for calibration scoring.
    config : Config, optional
        Config instance for dependency injection. Defaults to the
        global singleton ``config``.

    Returns
    -------
    pd.DataFrame
        Original fixtures augmented with ``prediction``, ``probability``,
        and ``confidence`` columns.
    """
    cfg = config or _global_config
    logger.info("Generating predictions for %d fixtures", len(fixtures_df))
    df = fixtures_df.copy()

    # Strip out non-feature columns that should not go into predict()
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feature_cols].values

    # Class probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        df["home_win_prob"] = probs[:, 0]
        df["draw_prob"] = probs[:, 1]
        df["away_win_prob"] = probs[:, 2]

        # Confidence scoring
        scorer = ConfidenceScorer()
        result = scorer.score(
            probs,
            individual_probs=individual_probs,
            calibration_brier=calibration_brier,
        )
        df["prediction"] = result["prediction"]
        df["probability"] = result["probability"]
        df["confidence"] = result["confidence"]
    else:
        logger.warning("Model does not implement predict_proba; using hard labels.")

    # Hard predictions (fallback for models without predict_proba)
    if "prediction" not in df.columns:
        if hasattr(model, "predict"):
            df["prediction"] = model.predict(X)
        elif hasattr(model, "predict_classes"):
            df["prediction"] = model.predict_classes(X)
        else:
            raise AttributeError("Model has no predict or predict_classes method.")

    _output_predictions(df, output_path, config=cfg)
    return df


# ── Helpers ─────────────────────────────────────────────


def _output_predictions(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Write / print predictions according to ``config.predict.output_format``."""
    path = Path(output_path) if output_path else None
    fmt = cfg.predict.output_format

    if path:
        fmt = path.suffix.lstrip(".")  # infer format from extension

    if fmt == "csv":
        _write_csv(df, path)
    elif fmt == "json":
        _write_json(df, path)
    else:
        _print_console(df)


def _write_csv(df: pd.DataFrame, path: Path | None) -> None:
    _ensure_path(path)
    df.to_csv(path, index=False)
    logger.info("Predictions saved to %s", path)


def _write_json(df: pd.DataFrame, path: Path | None) -> None:
    _ensure_path(path)
    data = df.to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Predictions saved to %s", path)


def _print_console(df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))


def _ensure_path(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
