"""Stacking Ensemble - meta-learner based ensemble."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from config import config
from src.models.protocol import ensure_predict_proba

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Stacking ensemble with a logistic regression meta-learner.

    Unlike ``WeightedEnsemble`` (which uses fixed weight averaging) and
    ``EnsembleModel`` (which uses grid-search weights), this class trains
    a **meta-learner** (LogisticRegression) on the base models' out-of-fold
    predictions to learn optimal combination weights.

    This is more powerful than weighted averaging because:
    - The meta-learner can assign **non-uniform** weights per class
    - It learns the **correlation** between model errors
    - It can handle **non-convex** trade-offs between models

    Parameters
    ----------
    model_names : tuple[str, ...]
        Names of base models to include (default: all available).
    meta_learner : Any, optional
        sklearn-compatible classifier for the meta-level.
        Default: ``LogisticRegression(multi_class='multinomial', C=1.0)``.
    use_cv_predictions : bool
        If True, use out-of-fold predictions from 3-fold CV to train the
        meta-learner (avoids overfitting). If False, use training set
        predictions directly (faster but may overfit). Default True.
    name : str
        Optional name for the ensemble.

    Example
    -------
    ::

        stacking = StackingEnsemble(
            model_names=("xgboost", "lightgbm", "logistic_regression")
        )
        result = stacking.fit(X_train, y_train, X_val, y_val)
        probs = stacking.predict_proba(X_test)
        print(stacking.weight_summary)
    """

    def __init__(
        self,
        model_names: tuple[str, ...] | None = None,
        meta_learner: Any = None,
        use_cv_predictions: bool = True,
        name: str = "StackingEnsemble",
    ) -> None:
        self.name = name
        self.model_names = model_names or (
            "xgboost", "lightgbm", "logistic_regression",
            "random_forest", "catboost",
        )
        self.use_cv_predictions = use_cv_predictions

        # Base models (populated by fit)
        self.base_models: dict[str, Any] = {}

        # Meta-learner (default: multinomial logistic regression)
        self._meta: Any = meta_learner or LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
            random_state=config.train.seed,
            class_weight="balanced",
        )

        # Training metrics
        self._fitted: bool = False
        self._train_log_loss: float | None = None
        self._val_log_loss: float | None = None

    # ── Properties ────────────────────────────────────────

    @property
    def trained(self) -> bool:
        return self._fitted

    @property
    def weight_summary(self) -> str:
        """Show meta-learner coefficients as weight-like interpretation.

        For a multinomial LR with 3 classes, returns the mean absolute
        coefficient per base model as a proxy for importance.
        """
        if not self._fitted or not hasattr(self._meta, "coef_"):
            return "Not fitted"

        coefs = self._meta.coef_  # shape (n_classes, n_base_models)
        model_names = list(self.base_models.keys())

        # Mean absolute coefficient as importance proxy
        importances = np.mean(np.abs(coefs), axis=0)
        total = importances.sum()
        if total > 0:
            importances = importances / total

        parts = [f"  {name}: {imp:.3f}" for name, imp in zip(model_names, importances)]
        return "Meta-learner importance:\n" + "\n".join(parts)

    # ── Fit ───────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Train the stacking ensemble.

        Steps:
        1. Train each base model on training data.
        2. Generate base model predictions (OOF or direct).
        3. Train the meta-learner on base model predictions.
        4. Evaluate on validation set if provided.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : optional validation data

        Returns
        -------
        dict with keys: ``base_log_losses``, ``meta_log_loss``, ``val_log_loss``
        """
        logger.info("Fitting %s with %d base models", self.name, len(self.model_names))
        col_means = X_train.mean().fillna(0)

        # 1. Train base models
        self._train_base_models(X_train, y_train)

        # 2. Generate meta-features (predictions from base models)
        if self.use_cv_predictions and len(X_train) >= 50:
            meta_train = self._oof_predictions(X_train, y_train)
        else:
            meta_train = self._direct_predictions(X_train, self.base_models)

        # 3. Train meta-learner
        logger.info("Training meta-learner on %d meta-features", meta_train.shape[1])
        X_meta = X_train.copy() if hasattr(X_train, "copy") else X_train
        # Get meta-features
        meta_probs = self._get_meta_features(
            X_train, self.base_models, meta_train,
        )
        self._meta.fit(meta_probs, y_train)

        # Training log-loss
        train_probs = self._meta.predict_proba(meta_probs)
        self._train_log_loss = float(log_loss(y_train, train_probs))

        # 4. Validation evaluation
        val_metrics: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            val_meta = self._get_meta_features(
                X_val, self.base_models, None,
            )
            val_probs = self._meta.predict_proba(val_meta)
            self._val_log_loss = float(log_loss(y_val, val_probs))
            val_accuracy = float(np.mean(np.argmax(val_probs, axis=1) == y_val.values))
            val_metrics = {"val_log_loss": self._val_log_loss, "val_accuracy": val_accuracy}

        self._fitted = True

        logger.info(
            "%s fitted — train log-loss: %.4f%s",
            self.name, self._train_log_loss,
            f", val log-loss: {self._val_log_loss:.4f}" if self._val_log_loss else "",
        )

        return {
            "base_log_losses": self._evaluate_base_models(X_train, y_train),
            "meta_log_loss": self._train_log_loss,
            **val_metrics,
        }

    def _train_base_models(
        self, X_train: pd.DataFrame, y_train: pd.Series,
    ) -> None:
        """Train all base models."""
        col_means = X_train.mean().fillna(0)
        X_clean = X_train.fillna(col_means)

        for name in self.model_names:
            if name == "xgboost":
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(
                        objective="multi:softprob", eval_metric="mlogloss",
                        n_estimators=100, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=config.train.seed, n_jobs=-1,
                    )
                    model.fit(X_train, y_train)
                except ImportError:
                    logger.warning("xgboost not installed — skipping")
                    continue

            elif name == "lightgbm":
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(
                        objective="multiclass", metric="multi_logloss",
                        n_estimators=100, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        num_leaves=31, random_state=config.train.seed,
                        n_jobs=-1, verbose=-1,
                    )
                    model.fit(X_train, y_train)
                except ImportError:
                    logger.warning("lightgbm not installed — skipping")
                    continue

            elif name == "catboost":
                try:
                    from catboost import CatBoostClassifier
                    model = CatBoostClassifier(
                        iterations=100, depth=5, learning_rate=0.05,
                        l2_leaf_reg=3.0, random_seed=config.train.seed,
                        loss_function="MultiClass", verbose=False,
                        allow_writing_files=False,
                    )
                    model.fit(X_train, y_train)
                except ImportError:
                    logger.warning("catboost not installed — skipping")
                    continue

            elif name == "logistic_regression":
                model = LogisticRegression(
                    solver="lbfgs", max_iter=2000,
                    random_state=config.train.seed,
                    class_weight="balanced", C=1.0, n_jobs=-1,
                )
                model.fit(X_clean, y_train)

            elif name == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=8,
                    min_samples_leaf=10, random_state=config.train.seed,
                    class_weight="balanced_subsample", n_jobs=-1,
                )
                model.fit(X_clean, y_train)

            else:
                logger.warning("Unknown model '%s' — skipping", name)
                continue

            self.base_models[name] = model
            logger.debug("  Trained base model: %s", name)

    def _oof_predictions(
        self, X: pd.DataFrame, y: pd.Series,
    ) -> dict[str, np.ndarray]:
        """Generate out-of-fold (OOF) predictions for meta-training.

        Uses 3-fold TimeSeriesSplit to avoid target leakage.
        """
        from src.time_series_cv import create_time_series_folds
        from sklearn.model_selection import cross_val_predict

        oof: dict[str, np.ndarray] = {}

        for name in list(self.base_models.keys()):
            model = self.base_models[name]
            col_means = X.mean().fillna(0)

            try:
                ts_cv = create_time_series_folds(n_splits=3)
                if name in ("xgboost", "lightgbm", "catboost"):
                    probs = cross_val_predict(
                        model.__class__(**model.get_params()),
                        X, y, cv=ts_cv, method="predict_proba",
                        n_jobs=1,  # 1 job to avoid pickling issues with custom classes
                    )
                else:
                    probs = cross_val_predict(
                        model.__class__(**model.get_params()),
                        X.fillna(col_means), y,
                        cv=ts_cv, method="predict_proba", n_jobs=1,
                    )
                oof[name] = probs
                logger.debug("  OOF predictions for %s: shape %s", name, probs.shape)
            except Exception as exc:
                logger.warning(
                    "OOF predictions failed for %s: %s — using direct", name, exc,
                )
                # Fallback to direct predictions
                if name in ("xgboost", "lightgbm", "catboost"):
                    oof[name] = model.predict_proba(X)
                else:
                    oof[name] = model.predict_proba(X.fillna(col_means))

        return oof

    def _direct_predictions(
        self, X: pd.DataFrame, models: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Generate direct (non-OOF) predictions."""
        col_means = X.mean().fillna(0)
        X_clean = X.fillna(col_means)

        preds: dict[str, np.ndarray] = {}
        for name, model in models.items():
            if name in ("xgboost", "lightgbm", "catboost"):
                preds[name] = model.predict_proba(X)
            else:
                preds[name] = model.predict_proba(X_clean)
        return preds

    def _get_meta_features(
        self,
        X: pd.DataFrame,
        models: dict[str, Any],
        oof_preds: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        """Stack base model probabilities as meta-features.

        Concatenates all base model probability arrays column-wise.
        For ``n`` models each outputting ``k=3`` class probs, this produces
        an ``(N, n*k)`` feature matrix for the meta-learner.
        """
        preds = oof_preds if oof_preds is not None else self._direct_predictions(X, models)
        if not preds:
            raise RuntimeError("No base model predictions available")

        meta_list = [preds[name] for name in sorted(preds.keys())]
        return np.column_stack(meta_list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the stacked ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability array of shape ``(n, 3)``.
        """
        if not self._fitted:
            raise RuntimeError("StackingEnsemble must be fitted before predicting.")

        meta = self._get_meta_features(X, self.base_models, None)
        return self._meta.predict_proba(meta)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _evaluate_base_models(
        self, X: pd.DataFrame, y: pd.Series,
    ) -> dict[str, float]:
        """Compute log-loss for each base model on training data."""
        losses: dict[str, float] = {}
        col_means = X.mean().fillna(0)
        X_clean = X.fillna(col_means)

        for name, model in self.base_models.items():
            try:
                if name in ("xgboost", "lightgbm", "catboost"):
                    probs = model.predict_proba(X)
                else:
                    probs = model.predict_proba(X_clean)
                losses[name] = float(log_loss(y, probs))
            except Exception:
                losses[name] = float("inf")

        return losses

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series,
    ) -> dict[str, Any]:
        """Evaluate the stacking ensemble on test data.

        Returns
        -------
        dict with keys: ``ensemble_log_loss``, ``ensemble_accuracy``,
        ``individual_log_losses``, ``improvement_over_best_single``.
        """
        if not self._fitted:
            raise RuntimeError("StackingEnsemble must be fitted before evaluating.")

        # Individual base model losses
        individual_losses: dict[str, float] = {}
        col_means = X_test.mean().fillna(0)
        X_clean = X_test.fillna(col_means)

        for name, model in self.base_models.items():
            try:
                if name in ("xgboost", "lightgbm", "catboost"):
                    probs = model.predict_proba(X_test)
                else:
                    probs = model.predict_proba(X_clean)
                individual_losses[name] = float(log_loss(y_test, probs))
            except Exception:
                individual_losses[name] = float("inf")

        # Ensemble prediction
        ensemble_probs = self.predict_proba(X_test)
        ensemble_loss = float(log_loss(y_test, ensemble_probs))
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        accuracy = float(np.mean(ensemble_preds == y_test.values))

        best_single = min(individual_losses, key=individual_losses.get)
        improvement = individual_losses[best_single] - ensemble_loss

        report = {
            "ensemble_log_loss": ensemble_loss,
            "ensemble_accuracy": accuracy,
            "individual_log_losses": individual_losses,
            "improvement_over_best_single": improvement,
            "best_single_model": best_single,
        }

        logger.info(
            "%s test log-loss: %.4f (best single: %.4f, Δ=%.4f)",
            self.name, ensemble_loss, individual_losses[best_single], improvement,
        )
        return report

    def save(self, path: str | None = None) -> str:
        """Save the stacking ensemble via joblib."""
        import joblib
        if path is None:
            path = str(config.paths.models / "stacking_ensemble.joblib")

        payload = {
            "base_models": self.base_models,
            "meta": self._meta,
            "model_names": self.model_names,
            "name": self.name,
            "fitted": self._fitted,
        }
        joblib.dump(payload, path)
        logger.info("%s saved to %s", self.name, path)
        return path

    @classmethod
    def load(cls, path: str) -> StackingEnsemble:
        """Load a saved stacking ensemble from disk."""
        import joblib
        payload = joblib.load(path)
        ensemble = cls(
            model_names=payload["model_names"],
            name=payload["name"],
        )
        ensemble.base_models = payload["base_models"]
        ensemble._meta = payload["meta"]
        ensemble._fitted = payload["fitted"]
        return ensemble


# ============================================================
#  Ensemble Model
# ============================================================


