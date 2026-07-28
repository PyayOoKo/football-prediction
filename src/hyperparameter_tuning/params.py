"""Parameter grids/distributions for each model type."""

from __future__ import annotations

from typing import Any


def lr_param_grid() -> dict[str, list[Any]]:
    return {
        "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "solver": ["lbfgs", "liblinear", "newton-cg"],
        "max_iter": [1000, 2000, 5000],
    }


def rf_param_dist() -> dict[str, list[Any]]:
    return {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [4, 6, 8, 10, 15, 20, None],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }


def xgb_param_dist() -> dict[str, list[Any]]:
    return {
        "n_estimators": [100, 200, 300, 500, 800, 1000],
        "max_depth": [3, 4, 5, 6, 8, 10, 12],
        "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.5],
    }


def lgbm_param_dist() -> dict[str, list[Any]]:
    return {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 8, 10, -1],
        "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
        "num_leaves": [15, 31, 63, 127, 255],
        "min_child_samples": [5, 10, 20, 50, 100],
        "min_child_weight": [1e-3, 1e-2, 0.1, 1.0, 10.0],
    }
