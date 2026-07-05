"""
Football Prediction — Source Package.

Exposes the pipeline stages so consumers can do::

    from src import data_loader, feature_engineering
"""

from __future__ import annotations

from src import (
    backtesting,
    confidence_scoring,
    data_collection,
    data_loader,
    elo,
    ensemble,
    evaluate,
    feature_engineering,
    hyperparameter_tuning,
    odds_processing,
    player_info,
    poisson_model,
    preprocessing,
    predict,
    time_series_cv,
    train,
    value_betting,
    xg_features,
)

__all__ = [
    "backtesting",
    "confidence_scoring",
    "data_collection",
    "data_loader",
    "elo",
    "ensemble",
    "evaluate",
    "feature_engineering",
    "hyperparameter_tuning",
    "odds_processing",
    "player_info",
    "poisson_model",
    "preprocessing",
    "predict",
    "time_series_cv",
    "train",
    "value_betting",
    "xg_features",
]
