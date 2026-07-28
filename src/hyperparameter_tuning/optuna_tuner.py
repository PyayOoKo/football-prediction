"""OptunaTuner — Bayesian hyper-parameter optimisation via Optuna."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.time_series_cv import create_time_series_folds
from src.hyperparameter_tuning.models import build_with_params, impute, needs_impute
from src.hyperparameter_tuning.optimisers import tune_hyperparameters

try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    optuna = None

logger = logging.getLogger(__name__)


class OptunaTuner:
    def __init__(
        self,
        n_trials: int = 100,
        timeout_seconds: int | None = None,
        cv_folds: int = 5,
        return_study: bool = False,
        seed: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.cv_folds = cv_folds
        self.return_study = return_study
        self.seed = seed
        if not _HAS_OPTUNA:
            logger.warning(
                "Optuna not installed. Install with: pip install optuna\n"
                "Falling back to RandomizedSearchCV."
            )

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "xgboost",
        verbose: bool = False,
    ) -> dict[str, Any]:
        if not _HAS_OPTUNA:
            logger.info("Optuna not available — falling back to RandomizedSearchCV")
            result = tune_hyperparameters(
                X_train, y_train,
                model_type=model_type,
                n_folds=self.cv_folds,
                n_iter=self.n_trials,
                verbose=verbose,
            )
            return {
                "best_params": result["best_params"],
                "best_value": result["cv_log_loss"],
                "n_trials": self.n_trials,
                "method": "random_search_fallback",
            }

        ts_cv = create_time_series_folds(n_splits=self.cv_folds)

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, model_type)
            model = build_with_params(model_type, params)
            losses: list[float] = []
            for train_idx, val_idx in ts_cv.split(X_train, y_train):
                X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
                try:
                    if needs_impute(model_type):
                        model.fit(impute(X_tr), y_tr)
                        probs = model.predict_proba(impute(X_vl))
                    else:
                        model.fit(X_tr, y_tr)
                        probs = model.predict_proba(X_vl)
                    losses.append(float(log_loss(y_vl, probs)))
                except Exception:
                    return 1.0
            return float(np.mean(losses)) if losses else 1.0

        logger.info("Optuna tuning %s — %d trials, %d-fold CV", model_type, self.n_trials, self.cv_folds)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=self.n_trials, reduction_factor=3,
        )
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner,
            study_name=f"tune_{model_type}",
        )
        study.optimize(
            objective, n_trials=self.n_trials,
            timeout=self.timeout_seconds, show_progress_bar=verbose,
        )

        if verbose:
            logger.info("  Optuna complete — best log-loss: %.4f", study.best_value)
            logger.info("  Best params: %s", study.best_params)

        optuna_result: dict[str, Any] = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "method": "optuna_tpe",
        }
        if self.return_study:
            optuna_result["study"] = study
        return optuna_result

    @staticmethod
    def _suggest_params(trial: optuna.Trial, model_type: str) -> dict[str, Any]:
        if model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            }
        elif model_type == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, step=8),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0, log=True),
            }
        elif model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        elif model_type == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear", "newton-cg"]),
                "max_iter": trial.suggest_int("max_iter", 1000, 5000, step=500),
            }
        elif model_type == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 100, 800, step=50),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.001, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
