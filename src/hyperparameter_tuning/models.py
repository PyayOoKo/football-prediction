"""Model factories — build baseline and parameterised models."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import config


def build_baseline(model_type: str) -> Any:
    cfg = config.train
    if model_type == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=cfg.seed,
            class_weight="balanced",
            C=1.0,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            random_state=cfg.seed,
            n_jobs=-1,
        )
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="multiclass",
            metric="multi_logloss",
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            num_leaves=31,
            min_child_samples=cfg.min_samples_leaf,
            random_state=cfg.seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def build_with_params(model_type: str, params: dict[str, Any]) -> Any:
    cfg = config.train
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=params.get("max_iter", 2000),
            random_state=cfg.seed,
            class_weight="balanced",
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", cfg.n_estimators),
            max_depth=params.get("max_depth", cfg.max_depth),
            min_samples_leaf=params.get("min_samples_leaf", cfg.min_samples_leaf),
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=cfg.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=cfg.seed,
            n_jobs=-1,
            **params,
        )
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="multiclass",
            metric="multi_logloss",
            random_state=cfg.seed,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def impute(X: pd.DataFrame) -> pd.DataFrame:
    return X.fillna(X.mean().fillna(0))


def needs_impute(model_type: str) -> bool:
    return model_type not in ("xgboost", "lightgbm")


def get_params(model: Any) -> dict[str, Any]:
    try:
        return cast("dict[str, Any]", model.get_params())
    except Exception:
        return {}
