"""Ensemble Training - full training pipeline for ensemble models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from config import EnsembleConfig, config
from src.models.protocol import ensure_predict_proba
from src.poisson_model import PoissonModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of multiple football prediction models.

    Combines XGBoost, Logistic Regression, and Poisson using optimised
    weighted averaging. Designed for speed - trains in ~20-40 seconds.
    Extra models (LightGBM, CatBoost) can be added via config.

    Parameters
    ----------
    config_override : EnsembleConfig, optional
        Override default configuration.  Falls back to ``config.ensemble``
        from the project's global config if not provided.
    """

    def __init__(
        self,
        config_override: EnsembleConfig | None = None,
    ) -> None:
        cfg = config_override or config.ensemble
        self.cfg = cfg

        # Trained sub-models (populated by ``fit()``)
        self.models: dict[str, Any] = {}
        self.weights: dict[str, float] = {}

        # Poisson model gets fitted separately (works on raw data)
        self._poisson_model: PoissonModel | None = None

        # Label binarizer for consistent predict_proba output shape
        self._lb: LabelBinarizer | None = None

        # Training metrics
        self._train_log_loss: float | None = None
        self._val_log_loss: float | None = None
        self._individual_log_losses: dict[str, float] = {}

        # Fast-training overrides to keep training time reasonable
        self._n_estimators = min(config.train.n_estimators, 80)
        self._max_depth = min(config.train.max_depth, 5)
        self._weight_step = max(self.cfg.weight_grid_step, 0.10)

    # -- Properties ------------------------------------------

    @property
    def trained(self) -> bool:
        """Whether the ensemble has been fitted."""
        return len(self.models) > 0

    @property
    def weight_summary(self) -> str:
        """Human-readable weight summary."""
        parts = [f"  {name}: {w:.3f}" for name, w in sorted(self.weights.items())]
        return "Weights:\n" + "\n".join(parts)

    # -- Fit ------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        df_train: pd.DataFrame | None = None,
        df_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Train the ensemble.

        Steps:
        1. Train each ML sub-model on ``(X_train, y_train)``.
        2. Fit the Poisson model on ``df_train`` (raw match data).
        3. Get individual model predictions on ``X_val`` and ``df_val``.
        4. Optimise ensemble weights by minimising validation log-loss.
        5. Store all trained models and weights.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target (0=Away, 1=Draw, 2=Home).
        X_val : pd.DataFrame
            Validation feature matrix.
        y_val : pd.Series
            Validation target.
        df_train : pd.DataFrame, optional
            Raw match data for Poisson model training (not feature-engineered).
            Required if Poisson is in the ensemble.
        df_val : pd.DataFrame, optional
            Raw match data for Poisson model validation.

        Returns
        -------
        dict[str, Any]
            Training report with keys: ``train_log_loss``, ``val_log_loss``,
            ``weights``, ``individual_log_losses``.
        """
        logger.info("Fitting ensemble with %d models", len(self.cfg.model_names))

        # -- 1. Train ML sub-models -------------------------
        self._train_ml_models(X_train, y_train, X_val, y_val)

        # -- 2. Train Poisson model -------------------------
        self._train_poisson_model(df_train)

        # -- 3. Get validation predictions ------------------
        val_preds = self._get_all_predictions(
            X_val, df_val, y_val,
            label="validation",
        )

        # -- 4. Optimise weights ----------------------------
        self.weights = self._optimise_weights(val_preds, y_val)

        # -- 4b. Apply weight constraints -------------------
        self._apply_weight_constraints()

        # -- 5. Evaluate ------------------------------------
        weighted_val = self._apply_weights(val_preds, self.weights)
        self._val_log_loss = float(log_loss(y_val, weighted_val))

        logger.info(
            "Ensemble fitted - val log-loss: %.4f, weights: %s",
            self._val_log_loss,
            {k: f"{v:.3f}" for k, v in sorted(self.weights.items())},
        )

        return {
            "train_log_loss": self._train_log_loss,
            "val_log_loss": self._val_log_loss,
            "weights": dict(self.weights),
            "individual_log_losses": dict(self._individual_log_losses),
        }

    # -- Internal: train sub-models -------------------------

    def _train_ml_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> None:
        """Train XGBoost, Logistic Regression (fast models).

        Each model is trained with lightweight settings for speed.
        Extra models like LightGBM / CatBoost can be added via config.
        """
        names = [n for n in self.cfg.model_names if n != "poisson"]
        col_means = X_train.mean().fillna(0)
        X_train_clean = X_train.fillna(col_means)
        X_val_clean = X_val.fillna(col_means) if X_val is not None else None

        for name in names:
            logger.info("Training sub-model: %s", name)

            if name == "xgboost":
                try:
                    import xgboost as xgb
                except ImportError:
                    logger.warning("xgboost not installed - skipping")
                    continue
                model = xgb.XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_estimators=self._n_estimators,
                    max_depth=self._max_depth,
                    learning_rate=config.train.learning_rate,
                    subsample=config.train.subsample,
                    colsample_bytree=config.train.colsample_bytree,
                    reg_lambda=config.train.reg_lambda,
                    reg_alpha=config.train.reg_alpha,
                    random_state=config.train.seed,
                    n_jobs=-1,
                )
                eval_set = [(X_train, y_train)]
                if X_val is not None and y_val is not None:
                    eval_set.append((X_val, y_val))
                model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

            elif name == "lightgbm":
                try:
                    import lightgbm as lgb
                except ImportError:
                    logger.warning("lightgbm not installed - skipping")
                    continue
                model = lgb.LGBMClassifier(
                    objective="multiclass",
                    metric="multi_logloss",
                    n_estimators=self._n_estimators,
                    max_depth=self._max_depth,
                    learning_rate=config.train.learning_rate,
                    subsample=config.train.subsample,
                    colsample_bytree=config.train.colsample_bytree,
                    reg_lambda=config.train.reg_lambda,
                    reg_alpha=config.train.reg_alpha,
                    num_leaves=31,
                    min_child_samples=config.train.min_samples_leaf,
                    random_state=config.train.seed,
                    n_jobs=-1,
                    verbose=-1,
                )
                eval_set = [(X_train, y_train)]
                if X_val is not None and y_val is not None:
                    eval_set.append((X_val, y_val))
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(10)],
                )

            elif name == "catboost":
                try:
                    from catboost import CatBoostClassifier
                except ImportError:
                    logger.warning("catboost not installed - skipping")
                    continue
                model = CatBoostClassifier(
                    iterations=self._n_estimators,
                    depth=min(self._max_depth, 10),
                    learning_rate=config.train.learning_rate,
                    l2_leaf_reg=config.train.reg_lambda,
                    random_seed=config.train.seed,
                    loss_function="MultiClass",
                    verbose=False,
                    allow_writing_files=False,
                    early_stopping_rounds=10,
                )
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val) if X_val is not None and y_val is not None else None,
                )

            elif name == "logistic_regression":
                model = LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=config.train.seed,
                    class_weight="balanced",
                    C=1.0,
                    n_jobs=-1,
                )
                model.fit(X_train_clean, y_train)

            elif name == "logistic_calibrated":
                try:
                    from sklearn.calibration import CalibratedClassifierCV
                except ImportError:
                    logger.warning("CalibratedClassifierCV not available - skipping")
                    continue
                base_lr = LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=config.train.seed,
                    class_weight="balanced",
                    C=1.0,
                )
                model = CalibratedClassifierCV(
                    base_lr,
                    method="sigmoid",
                    cv=2,
                )
                model.fit(X_train_clean, y_train)

            else:
                logger.warning("Unknown model '%s' - skipping", name)
                continue

            self.models[name] = model

        # Compute training and validation log-loss for each model
        for name, model in self.models.items():
            train_probs = self._ml_predict_proba(model, X_train)
            self._individual_log_losses[f"{name}_train"] = float(
                log_loss(y_train, train_probs)
            )
            if X_val is not None and y_val is not None:
                val_probs = self._ml_predict_proba(model, X_val)
                self._individual_log_losses[f"{name}_val"] = float(
                    log_loss(y_val, val_probs)
                )

    def _train_poisson_model(
        self, df_train: pd.DataFrame | None,
    ) -> None:
        """Fit the Poisson model if included and raw data available."""
        if "poisson" not in self.cfg.model_names:
            return

        if df_train is None or df_train.empty:
            logger.warning(
                "Poisson model selected but no raw match data provided. "
                "Poisson will be excluded from the ensemble."
            )
            return

        self._poisson_model = PoissonModel(
            min_matches=config.poisson.min_matches,
            max_goals=config.poisson.max_goals,
        )
        self._poisson_model.fit(df_train)
        self.models["poisson"] = self._poisson_model

    # -- Internal: get predictions from all models ----------

    def _get_all_predictions(
        self,
        X: pd.DataFrame,
        df_raw: pd.DataFrame | None,
        y_true: pd.Series | None = None,
        label: str = "data",
    ) -> dict[str, np.ndarray]:
        """Get probability predictions from every sub-model.

        Returns a dict ``{model_name: np.ndarray of shape (n, 3)}``.
        """
        preds: dict[str, np.ndarray] = {}

        for name, model in self.models.items():
            if name == "poisson":
                probs = self._poisson_predict_proba(df_raw)
            else:
                probs = self._ml_predict_proba(model, X)

            preds[name] = probs

            if y_true is not None:
                loss = log_loss(y_true, probs)
                self._individual_log_losses[f"{name}_{label}"] = loss
                logger.debug("  %s (%s): log-loss = %.4f", name, label, loss)

        return preds

    @staticmethod
    def _ml_predict_proba(
        model: Any, X: pd.DataFrame,
    ) -> np.ndarray:
        """Get predict_proba from a scikit-learn / XGBoost model with NaN handling."""
        col_means = X.mean().fillna(0)
        X_clean = X.fillna(col_means)
        return model.predict_proba(X_clean)

    def _poisson_predict_proba(
        self, df_raw: pd.DataFrame | None,
    ) -> np.ndarray:
        """Get match outcome probabilities from the Poisson model.

        Returns an array of shape ``(n, 3)`` where columns are
        ``[away_prob, draw_prob, home_prob]``.
        """
        if self._poisson_model is None or df_raw is None or df_raw.empty:
            # Return equal probabilities as fallback
            n = len(df_raw) if df_raw is not None else 0
            if n == 0:
                return np.zeros((0, 3))
            return np.full((n, 3), 1.0 / 3.0)

        preds_df = self._poisson_model.predict_matches(df_raw)
        n = len(preds_df)

        # Map from predict_matches columns to [away, draw, home]
        probs = np.zeros((n, 3))
        if "away_win_prob" in preds_df.columns:
            probs[:, 0] = preds_df["away_win_prob"].values
        if "draw_prob" in preds_df.columns:
            probs[:, 1] = preds_df["draw_prob"].values
        if "home_win_prob" in preds_df.columns:
            probs[:, 2] = preds_df["home_win_prob"].values

        # Renormalise any rows that don't sum to 1.0
        row_sums = probs.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        probs = probs / row_sums[:, np.newaxis]

        return probs

    # -- Weight optimisation --------------------------------

    def _optimise_weights(
        self,
        preds: dict[str, np.ndarray],
        y_val: pd.Series,
    ) -> dict[str, float]:
        """Find ensemble weights that minimise validation log-loss.

        Strategy:
        1. Coarse grid search over weight combinations (step 0.05).
        2. Return the combination with lowest log-loss.

        For ensembles of 3+ models, we enumerate all combinations where
        weights are multiples of ``_GRID_STEP`` and sum to 1.0.  For 4
        models with step 0.05, this is ~1,000 combinations - fast enough.
        """
        model_names = list(preds.keys())
        n_models = len(model_names)

        if n_models == 0:
            return {}

        if n_models == 1:
            return {model_names[0]: 1.0}

        step = self._weight_step
        best_loss = float("inf")
        best_weights: list[float] = []

        # Enumerate unique weight combinations via composition.
        # Generate raw integer vectors (w1..wn), normalise, deduplicate.
        n_steps = int(round(1.0 / step))
        seen: set[tuple[float, ...]] = set()

        for raw_weights in itertools.product(range(n_steps + 1), repeat=n_models):
            total = sum(raw_weights)
            if total == 0:
                continue
            norm = tuple(w / total for w in raw_weights)
            if norm in seen:
                continue
            seen.add(norm)

            weighted = self._apply_weights(preds, dict(zip(model_names, norm)))
            loss = float(log_loss(y_val, weighted))

            if loss < best_loss:
                best_loss = loss
                best_weights = list(norm)

        logger.info(
            "Weight optimisation complete - best val log-loss: %.4f",
            best_loss,
        )
        return dict(zip(model_names, best_weights))

    def _apply_weight_constraints(self) -> None:
        """Enforce min/max weight ranges for each model in the ensemble.

        After the grid-search optimiser finds the best weights for
        minimising log-loss, this method adjusts them so each model
        stays within its configured range (min, max).
        """
        ranges = self.cfg.model_weight_ranges
        if not ranges:
            return

        # Feasibility check
        total_min = sum(lo for lo, _ in ranges.values() if lo > 0)
        if total_min > 1.0:
            logger.warning(
                "Weight ranges min sum = %.2f > 1.0 - constraints impossible to satisfy",
                total_min,
            )

        max_iter = 30
        for _ in range(max_iter):
            adjusted = False

            # Check all models against their ranges
            under: list[tuple[str, float]] = []  # (name, deficit)
            over: list[tuple[str, float]] = []   # (name, excess)

            for name, (lo, hi) in ranges.items():
                if name not in self.weights:
                    continue
                w = self.weights[name]
                if w < lo:
                    under.append((name, lo - w))
                elif w > hi:
                    over.append((name, w - hi))

            if not under and not over:
                break  # All constraints satisfied

            # Fix underweight models
            for name, deficit in under:
                givers = {
                    k: v for k, v in self.weights.items()
                    if k != name and k in ranges
                    and v > ranges[k][0]
                    and k not in {u[0] for u in under}
                }
                if not givers:
                    givers = {k: v for k, v in self.weights.items() if k != name}

                giver_total = sum(givers.values())
                if giver_total > 0:
                    for k in givers:
                        share = givers[k] / giver_total
                        reduction = deficit * share
                        self.weights[k] = max(self.weights[k] - reduction, 0.0)
                    self.weights[name] += deficit
                    adjusted = True

            # Fix overweight models
            for name, excess in over:
                receivers = {
                    k: v for k, v in self.weights.items()
                    if k != name and k in ranges
                    and v < ranges[k][1]
                    and k not in {o[0] for o in over}
                }
                if not receivers:
                    receivers = {k: v for k, v in self.weights.items() if k != name}

                receiver_total = sum(receivers.values())
                if receiver_total > 0:
                    for k in receivers:
                        share = receivers[k] / receiver_total
                        self.weights[k] = self.weights[k] + excess * share
                    self.weights[name] -= excess
                    adjusted = True

            # Renormalise
            total = sum(self.weights.values())
            if total > 0:
                for name in self.weights:
                    self.weights[name] /= total

            if not adjusted:
                break

        w_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.weights.items()))
        logger.info("Weight constraints applied - final weights: %s", w_str)

    @staticmethod
    def _apply_weights(
        preds: dict[str, np.ndarray],
        weights: dict[str, float],
    ) -> np.ndarray:
        """Compute weighted average of model predictions.

        Parameters
        ----------
        preds : dict[str, np.ndarray]
            ``{model_name: probs_array}`` where each array is (n, 3).
        weights : dict[str, float]
            ``{model_name: weight}`` - must sum to 1.0.

        Returns
        -------
        np.ndarray
            Weighted average probabilities, shape ``(n, 3)``.
        """
        if not preds:
            return np.zeros((0, 3))

        n = len(next(iter(preds.values())))
        weighted = np.zeros((n, 3))

        for name, probs in preds.items():
            w = weights.get(name, 0.0)
            if w > 0:
                weighted += w * probs

        # Renormalise
        row_sums = weighted.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weighted = weighted / row_sums[:, np.newaxis]

        return weighted

    # -- Public prediction ----------------------------------

    def predict_proba(
        self,
        X: pd.DataFrame,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict match outcome probabilities using the trained ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (for ML models).
        df_raw : pd.DataFrame, optional
            Raw match data (for Poisson model).

        Returns
        -------
        np.ndarray
            Probability array of shape ``(n, 3)`` with columns
            ``[away_prob, draw_prob, home_prob]``.
        """
        if not self.trained:
            raise RuntimeError("Ensemble must be fitted before predicting.")

        preds = self._get_all_predictions(X, df_raw)
        return self._apply_weights(preds, self.weights)

    def predict(self, X: pd.DataFrame, df_raw: pd.DataFrame | None = None) -> np.ndarray:
        """Predict hard class labels (0=Away, 1=Draw, 2=Home)."""
        probs = self.predict_proba(X, df_raw)
        return np.argmax(probs, axis=1)

    # -- Evaluation -----------------------------------------

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_test: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Evaluate the ensemble on test data.

        Returns individual model log-losses AND ensemble log-loss so
        you can directly compare performance.
        """
        if not self.trained:
            raise RuntimeError("Ensemble must be fitted before evaluating.")

        # Get individual model predictions
        preds = self._get_all_predictions(X_test, df_test, y_test, label="test")

        # Ensemble prediction
        ensemble_probs = self._apply_weights(preds, self.weights)
        ensemble_loss = float(log_loss(y_test, ensemble_probs))

        # Accuracy
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        accuracy = float(np.mean(ensemble_preds == y_test.values))

        report: dict[str, Any] = {
            "ensemble_log_loss": ensemble_loss,
            "ensemble_accuracy": accuracy,
            "individual_log_losses": {},
        }

        for name, probs in preds.items():
            report["individual_log_losses"][name] = float(
                log_loss(y_test, probs)
            )

        # Comparison with best single model
        best_single = min(report["individual_log_losses"].values())
        improvement = best_single - ensemble_loss
        report["improvement_over_best_single"] = improvement
        report["best_single_model"] = min(
            report["individual_log_losses"],
            key=report["individual_log_losses"].get,
        )

        logger.info(
            "Ensemble test log-loss: %.4f (best single: %.4f, Delta=%.4f)",
            ensemble_loss, best_single, improvement,
        )

        return report

    # -- Save / Load ----------------------------------------

    def save(self, path: str | None = None) -> str:
        """Save the entire ensemble (sub-models + weights) via joblib.

        Parameters
        ----------
        path : str, optional
            Output path.  Default: ``models/ensemble_model.joblib``.

        Returns
        -------
        str
            Path to the saved file.
        """
        import joblib

        if path is None:
            path = str(config.paths.models / "ensemble_model.joblib")

        payload = {
            "models": self.models,
            "weights": self.weights,
            "poisson_model": self._poisson_model,
            "cfg": self.cfg,
        }
        joblib.dump(payload, path)
        logger.info("Ensemble saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> EnsembleModel:
        """Load a saved ensemble from disk.

        Parameters
        ----------
        path : str
            Path to the saved ensemble file.

        Returns
        -------
        EnsembleModel
            Loaded ensemble with all sub-models and weights restored.
        """
        import joblib

        payload = joblib.load(path)
        ensemble = cls(config_override=payload["cfg"])
        ensemble.models = payload["models"]
        ensemble.weights = payload["weights"]
        ensemble._poisson_model = payload["poisson_model"]
        logger.info("Ensemble loaded from %s", path)
        return ensemble


# ============================================================
#  Training script convenience function
# ============================================================


