"""Optimisation wrappers — GridSearchCV / RandomizedSearchCV for each model."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import config
from src.hyperparameter_tuning.models import (
    impute,
    needs_impute,
)
from src.hyperparameter_tuning.params import (
    lgbm_param_dist,
    lr_param_grid,
    rf_param_dist,
    xgb_param_dist,
)
from src.time_series_cv import create_time_series_folds

logger = logging.getLogger(__name__)


def evaluate(
    model: Any, X: pd.DataFrame, y: pd.Series, model_type: str
) -> tuple[float, float]:
    X_eval = impute(X) if needs_impute(model_type) else X
    probs = model.predict_proba(X_eval)
    preds = model.predict(X_eval)
    return float(log_loss(y, probs)), float(accuracy_score(y, preds))


def check_model_type(model_type: str) -> None:
    valid = {"logistic_regression", "random_forest", "xgboost", "lightgbm"}
    if model_type not in valid:
        raise ValueError(f"Unknown model_type '{model_type}'. Must be one of {valid}")


def optimise_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    verbose: bool,
) -> tuple[dict[str, Any], float]:
    logger.info("  GridSearchCV (Logistic Regression) — %d-fold time-series CV", cv)
    ts_cv = create_time_series_folds(n_splits=cv)
    base = LogisticRegression(random_state=config.train.seed, class_weight="balanced")
    searcher = GridSearchCV(
        base,
        lr_param_grid(),
        cv=ts_cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )
    searcher.fit(impute(X_train), y_train)
    return searcher.best_params_, -searcher.best_score_


def optimise_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    n_iter: int,
    verbose: bool,
) -> tuple[dict[str, Any], float]:
    logger.info(
        "  RandomizedSearchCV (Random Forest) — %d-fold time-series CV, %d iters",
        cv,
        n_iter,
    )
    ts_cv = create_time_series_folds(n_splits=cv)
    base = RandomForestClassifier(
        random_state=config.train.seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    searcher = RandomizedSearchCV(
        base,
        rf_param_dist(),
        n_iter=n_iter,
        cv=ts_cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        random_state=config.train.seed,
        verbose=1 if verbose else 0,
    )
    searcher.fit(impute(X_train), y_train)
    return searcher.best_params_, -searcher.best_score_


def optimise_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    n_iter: int,
    verbose: bool,
) -> tuple[dict[str, Any], float]:
    logger.info(
        "  RandomizedSearchCV (XGBoost) — %d-fold time-series CV, %d iters", cv, n_iter
    )
    ts_cv = create_time_series_folds(n_splits=cv)
    import xgboost as xgb

    base = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=config.train.seed,
        n_jobs=-1,
    )
    searcher = RandomizedSearchCV(
        base,
        xgb_param_dist(),
        n_iter=n_iter,
        cv=ts_cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        random_state=config.train.seed,
        verbose=1 if verbose else 0,
    )
    searcher.fit(X_train, y_train)
    return searcher.best_params_, -searcher.best_score_


def optimise_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    n_iter: int,
    verbose: bool,
) -> tuple[dict[str, Any], float]:
    logger.info(
        "  RandomizedSearchCV (LightGBM) — %d-fold time-series CV, %d iters", cv, n_iter
    )
    ts_cv = create_time_series_folds(n_splits=cv)
    import lightgbm as lgb

    base = lgb.LGBMClassifier(
        objective="multiclass",
        metric="multi_logloss",
        random_state=config.train.seed,
        n_jobs=-1,
        verbose=-1,
    )
    searcher = RandomizedSearchCV(
        base,
        lgbm_param_dist(),
        n_iter=n_iter,
        cv=ts_cv,
        scoring="neg_log_loss",
        n_jobs=-1,
        random_state=config.train.seed,
        verbose=1 if verbose else 0,
    )
    searcher.fit(X_train, y_train)
    return searcher.best_params_, -searcher.best_score_


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
    n_folds: int = 5,
    n_iter: int = 50,
    verbose: bool = False,
) -> dict[str, Any]:
    check_model_type(model_type)
    optim_fn = {
        "logistic_regression": lambda: optimise_lr(X_train, y_train, n_folds, verbose),
        "random_forest": lambda: optimise_rf(
            X_train, y_train, n_folds, n_iter, verbose
        ),
        "xgboost": lambda: optimise_xgb(X_train, y_train, n_folds, n_iter, verbose),
        "lightgbm": lambda: optimise_lgbm(X_train, y_train, n_folds, n_iter, verbose),
    }
    if verbose:
        logger.info("  Tuning %s...", model_type)
    best_params, cv_loss = optim_fn[model_type]()
    if verbose:
        logger.info("    Best CV log-loss: %.4f", cv_loss)
        logger.info("    Best params: %s", best_params)
    return {"best_params": best_params, "cv_log_loss": cv_loss}
