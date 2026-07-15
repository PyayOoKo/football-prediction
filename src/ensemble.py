"""
Ensemble Model - combine XGBoost, Logistic Regression, and Poisson.

Why this ensemble is fast
-------------------------
Unlike 5-model ensembles (XGBoost + LightGBM + CatBoost + Poisson + LR)
that can take 2-5 minutes to train, this 3-model ensemble trains in
~20-40 seconds on typical hardware because:

1. **XGBoost** - fast tree-based model (80 trees, depth 5, parallelised)
2. **Logistic Regression** - the fastest possible baseline (seconds)
3. **Poisson** - goal-based generative model (already fast)
4. **Weight grid search** - with 3 models and step 0.10, only ~66
   combinations to evaluate (vs ~1001 for 5 models)

The module also provides a lightweight ``WeightedEnsemble`` class for
combining pre-trained (and optionally calibrated) models via fixed or
optimised weighted averaging.

Usage
-----
::

    from src.ensemble import EnsembleModel, WeightedEnsemble

    # Full training pipeline (trains sub-models internally)
    ensemble = EnsembleModel()
    result = ensemble.fit(X_train, y_train, X_val, y_val, df_train, df_val)
    probs = ensemble.predict_proba(X_test, df_test)

    # Lightweight ensemble of pre-trained models
    weighted = WeightedEnsemble([
        (xgb_model, 0.5),
        (lr_model, 0.3),
        (poisson_model, 0.2),
    ])
    probs = weighted.predict_proba(X_test, df_test)
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

from config import EnsembleConfig, config
from src.poisson_model import PoissonModel

logger = logging.getLogger(__name__)

# -- Default model names ---------------------------------
_MODEL_NAMES = ["xgboost", "logistic_regression", "poisson"]

# Default weight grid search step
_GRID_STEP = 0.10


# ═══════════════════════════════════════════════════════════
#  Weighted Ensemble — combine pre-trained models
# ═══════════════════════════════════════════════════════════


class WeightedEnsemble:
    """Lightweight ensemble combining pre-trained models via weighted averaging.

    Unlike ``EnsembleModel`` (which trains sub-models internally), this class
    accepts a list of already-trained ``(model, weight)`` tuples and averages
    their predictions.  It supports both sklearn-compatible models (Phase 4)
    and statistical models with ``predict_matches()`` (Phase 3).

    Parameters
    ----------
    models_and_weights : list[tuple[Any, float]] | None
        List of ``(model, weight)`` tuples.  Weights are normalised to sum
        to 1.0.  If ``None``, start empty and weights can be learned via
        ``fit()``.
    name : str
        Optional name for the ensemble (used in logging / reports).

    Examples
    --------
    ::

        # Pre-trained Phase 4 models
        ensemble = WeightedEnsemble([
            (xgb_model, 0.5),
            (lr_model, 0.3),
            (rf_model, 0.2),
        ])
        probs = ensemble.predict_proba(X_test)

        # Mixed Phase 3 + Phase 4 models
        ensemble = WeightedEnsemble([
            (xgb_model, 0.4),
            (poisson_model, 0.3),
            (elo_model, 0.3),
        ])
        probs = ensemble.predict_proba(X_feat, df_raw=df_raw)

        # Learn weights from validation data
        ensemble = WeightedEnsemble(name="optimised")
        ensemble.add_model(xgb_model)
        ensemble.add_model(lr_model, weight=0.0)  # weight learned by fit()
        ensemble.fit(X_val, y_val, df_val=df_val)
    """

    def __init__(
        self,
        models_and_weights: list[tuple[Any, float]] | None = None,
        name: str = "WeightedEnsemble",
    ) -> None:
        self.name = name
        # Internal storage: list of (model, weight, model_type)
        self._members: list[tuple[Any, float, str]] = []

        if models_and_weights is not None:
            # Add all models first, then normalise ONCE to preserve intended ratios
            for model, weight in models_and_weights:
                mtype = self._detect_model_type(model)
                if mtype == "unknown":
                    logger.warning(
                        "Model %s has neither predict_proba nor predict_matches.",
                        self._model_name(model),
                    )
                self._members.append((model, float(weight), mtype))
            self._normalise_weights()

        self._fitted: bool = len(self._members) > 0
        self._n_classes: int = 3

    # ── Properties ────────────────────────────────────────

    @property
    def members(self) -> list[tuple[Any, float, str]]:
        """Return a copy of ensemble members ``[(model, weight, type), ...]``."""
        return list(self._members)

    @property
    def weights(self) -> dict[str, float]:
        """Return ``{model_name: weight}`` for all members.

        Note: duplicate model types (e.g. two ``LogisticRegression`` instances)
        produce only the last entry in the dict.  The internal ``_members``
        list always stores separate weights per model.
        """
        return {
            self._model_name(m): w
            for m, w, _ in self._members
        }

    @property
    def trained(self) -> bool:
        """Whether the ensemble has at least one member."""
        return len(self._members) > 0

    @property
    def weight_summary(self) -> str:
        """Human-readable weight summary."""
        parts = []
        for model, weight, mtype in self._members:
            name = self._model_name(model)
            parts.append(f"  {name} ({mtype}): {weight:.3f}")
        return "Weights:\n" + "\n".join(parts)

    # ── Member management ────────────────────────────────

    @staticmethod
    def _model_name(model: Any) -> str:
        """Return a readable name for a model."""
        raw = type(model).__name__
        if raw == "XGBClassifier":
            return "XGBoost"
        if raw == "LGBMClassifier":
            return "LightGBM"
        if raw == "CalibratedTemperatureWrapper":
            return "CalibratedTemp"
        if raw == "CalibratedStatsModel":
            return "CalibratedStats"
        return raw

    @staticmethod
    def _detect_model_type(model: Any) -> str:
        """Detect whether a model is Phase 4 (sklearn) or Phase 3 (stats).

        Prioritises ``predict_matches`` over ``predict_proba`` because many
        Phase 3 models (e.g. ``PoissonModel``, ``DixonColesModel``) also have
        a ``predict_proba`` method that takes a **raw DataFrame** (not a feature
        matrix), which would break when passed an ML feature matrix.

        Returns
        -------
        str
            ``"phase4"`` if the model has ``predict_proba`` only,
            ``"phase3"`` if it has ``predict_matches``,
            ``"unknown"`` otherwise.
        """
        # Check predict_matches first — this is the more specific Phase 3 interface
        if hasattr(model, "predict_matches"):
            return "phase3"
        if hasattr(model, "predict_proba"):
            return "phase4"
        return "unknown"

    def add_model(
        self,
        model: Any,
        weight: float = 1.0,
    ) -> WeightedEnsemble:
        """Add a model to the ensemble.

        Parameters
        ----------
        model : Any
            Trained model (Phase 4 or Phase 3).
        weight : float
            Initial weight (will be normalised with all other weights
            on the next ``fit()`` or ``set_weights()`` call).
            Default 1.0.

        Returns
        -------
        WeightedEnsemble
            Self, for method chaining.
        """
        mtype = self._detect_model_type(model)
        if mtype == "unknown":
            logger.warning(
                "Model %s has neither predict_proba nor predict_matches.",
                self._model_name(model),
            )
        self._members.append((model, float(weight), mtype))
        self._normalise_weights()
        self._fitted = True
        return self

    def _normalise_weights(self) -> None:
        """Normalise all member weights to sum to 1.0."""
        total = sum(w for _, w, _ in self._members)
        if total <= 0:
            # Equal weights as fallback
            for i in range(len(self._members)):
                member = list(self._members[i])
                member[1] = 1.0 / max(len(self._members), 1)
                self._members[i] = tuple(member)
        else:
            for i in range(len(self._members)):
                member = list(self._members[i])
                member[1] /= total
                self._members[i] = tuple(member)

    def set_weights(self, weights: dict[str, float]) -> WeightedEnsemble:
        """Explicitly set weights by model name (type name).

        Parameters
        ----------
        weights : dict[str, float]
            Mapping from model type name (e.g. ``"XGBoost"``, ``"PoissonModel"``)
            to unnormalised weight.  Unspecified models get weight 0.

        Returns
        -------
        WeightedEnsemble
            Self, for method chaining.
        """
        for i in range(len(self._members)):
            model, _, mtype = self._members[i]
            name = self._model_name(model)
            member = list(self._members[i])
            member[1] = weights.get(name, 0.0)
            self._members[i] = tuple(member)
        self._normalise_weights()
        return self

    # ── Prediction ────────────────────────────────────────

    def _predict_single(
        self,
        model: Any,
        mtype: str,
        X: pd.DataFrame,
        df_raw: pd.DataFrame | None,
    ) -> np.ndarray:
        """Get (n, 3) probability array from a single model.

        Handles:
        - Phase 4: ``model.predict_proba(X)`` directly
        - Phase 3: ``model.predict_matches(df_raw)`` → extract probs
        - Unknown: zeros (fallback)
        """
        n = len(X) if hasattr(X, "__len__") else 0

        if mtype == "phase4":
            try:
                probs = model.predict_proba(X)
                return np.asarray(probs, dtype=np.float64)
            except Exception:
                try:
                    # Fallback: fill NaN and retry
                    col_means = X.mean().fillna(0) if hasattr(X, "mean") else 0
                    X_clean = X.fillna(col_means) if hasattr(X, "fillna") else X
                    probs = model.predict_proba(X_clean)
                    return np.asarray(probs, dtype=np.float64)
                except Exception as e:
                    logger.warning(
                        "predict_proba failed for %s: %s",
                        self._model_name(model), e,
                    )
                    return np.full((n, self._n_classes), 1.0 / self._n_classes)

        if mtype == "phase3":
            if df_raw is None or df_raw.empty:
                logger.warning(
                    "Model %s needs raw match data (df_raw) for predict_matches "
                    "but none provided — returning uniform probs.",
                    self._model_name(model),
                )
                return np.full((n, self._n_classes), 1.0 / self._n_classes)
            try:
                preds_df = model.predict_matches(df_raw)
                probs = np.column_stack([
                    preds_df["away_win_prob"].values,
                    preds_df["draw_prob"].values,
                    preds_df["home_win_prob"].values,
                ])
                return np.asarray(probs, dtype=np.float64)
            except Exception as e:
                logger.warning(
                    "predict_matches failed for %s: %s",
                    self._model_name(model), e,
                )
                return np.full((n, self._n_classes), 1.0 / self._n_classes)

        # Unknown type
        logger.warning(
            "Model %s has unsupported type '%s' — returning uniform probs.",
            self._model_name(model), mtype,
        )
        return np.full((n, self._n_classes), 1.0 / self._n_classes)

    def predict_proba(
        self,
        X: pd.DataFrame,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict match outcome probabilities via weighted averaging.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (for Phase 4 ML models).
        df_raw : pd.DataFrame, optional
            Raw match data (for Phase 3 statistical models).
            Required if any ensemble member is Phase 3.

        Returns
        -------
        np.ndarray
            Probability array of shape ``(n, 3)`` with columns
            ``[away_prob, draw_prob, home_prob]``.  Rows sum to 1.0.
        """
        if not self.trained:
            raise RuntimeError(
                "WeightedEnsemble must have at least one model before predicting."
            )

        n = len(X) if hasattr(X, "__len__") else 0
        if n == 0:
            return np.zeros((0, self._n_classes))

        weighted = np.zeros((n, self._n_classes))

        for model, weight, mtype in self._members:
            if weight <= 0:
                continue
            probs = self._predict_single(model, mtype, X, df_raw)
            if probs.shape != (n, self._n_classes):
                logger.warning(
                    "Model %s returned probs shape %s, expected (%d, %d) — skipping.",
                    self._model_name(model), probs.shape, n, self._n_classes,
                )
                continue
            weighted += weight * probs

        # Renormalise to sum to 1.0
        row_sums = weighted.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return weighted / row_sums

    def predict(
        self,
        X: pd.DataFrame,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict hard class labels (0=Away, 1=Draw, 2=Home)."""
        probs = self.predict_proba(X, df_raw=df_raw)
        return np.argmax(probs, axis=1)

    # ── Optional fitting: learn weights ───────────────────

    def fit(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        df_val: pd.DataFrame | None = None,
        weight_grid_step: float = 0.05,
        max_weight: float = 1.0,
    ) -> dict[str, Any]:
        """Optimise ensemble weights to minimise validation log-loss.

        Uses a coarse-to-fine grid search over weight combinations.
        At least one model must already be added to the ensemble.
        Existing weights are used as a starting point.

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation feature matrix.
        y_val : pd.Series
            Validation target labels.
        df_val : pd.DataFrame, optional
            Raw validation match data (for Phase 3 models).
        weight_grid_step : float
            Grid resolution for weight search (default 0.05).
            Lower = more granular but slower.
        max_weight : float
            Maximum weight for any single model (default 1.0).
            Set to < 1.0 to prevent a single model from dominating.

        Returns
        -------
        dict[str, Any]
            ``{"best_weights", "best_log_loss", "individual_log_losses"}``
        """
        if len(self._members) < 2:
            logger.warning(
                "Fit requires at least 2 models; got %d. Using equal weights.",
                len(self._members),
            )
            self._normalise_weights()
            return {"best_weights": self.weights, "best_log_loss": float("inf")}

        n_models = len(self._members)
        step = max(weight_grid_step, 0.02)  # Prevent excessively fine grids

        # Get individual predictions once
        logger.info(
            "Fitting WeightedEnsemble '%s' — %d models, grid step=%.2f",
            self.name, n_models, step,
        )
        preds_list: list[np.ndarray] = []
        individual_losses: dict[str, float] = {}

        for model, weight, mtype in self._members:
            probs = self._predict_single(model, mtype, X_val, df_val)
            name = self._model_name(model)
            preds_list.append(probs)
            try:
                loss = float(log_loss(y_val, probs))
                individual_losses[name] = loss
                logger.debug("  %s: log-loss = %.4f", name, loss)
            except Exception:
                individual_losses[name] = float("inf")

        if not preds_list:
            return {"best_weights": {}, "best_log_loss": float("inf")}

        # ── Grid search (coarse-to-fine) ──
        MAX_COMBINATIONS = 100_000
        n_fine = int(round(max_weight / step))
        n_bins = max(2, min(n_fine, int(MAX_COMBINATIONS ** (1.0 / max(n_models, 1))) - 1))
        if n_bins < n_fine:
            logger.warning(
                "Full grid would evaluate %d combos (%d models, step=%.3f) — "
                "capping to %d bins (≈%d combos)",
                (n_fine + 1) ** n_models, n_models, step,
                n_bins, (n_bins + 1) ** n_models,
            )

        best_loss = float("inf")
        best_weights: list[float] = [0.0] * n_models
        seen: set[tuple[float, ...]] = set()
        total_combinations = 0

        for raw in itertools.product(range(n_bins + 1), repeat=n_models):
            total = sum(raw)
            if total == 0:
                continue
            norm = tuple(r / total for r in raw)
            if norm in seen:
                continue
            seen.add(norm)
            total_combinations += 1

            # Weighted average
            weighted = np.zeros_like(preds_list[0])
            for i, probs in enumerate(preds_list):
                weighted += norm[i] * probs

            # Renormalise
            row_sums = weighted.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            weighted = weighted / row_sums

            try:
                loss = float(log_loss(y_val, weighted))
            except Exception:
                continue

            if loss < best_loss:
                best_loss = loss
                best_weights = list(norm)

        # ── Apply best weights ──
        for i in range(n_models):
            member = list(self._members[i])
            member[1] = best_weights[i]
            self._members[i] = tuple(member)

        self._fitted = True
        logger.info(
            "WeightedEnsemble '%s' fitted — val log-loss: %.4f "
            "(evaluated %d combinations)",
            self.name, best_loss, total_combinations,
        )

        return {
            "best_weights": self.weights,
            "best_log_loss": best_loss,
            "individual_log_losses": individual_losses,
            "combinations_evaluated": total_combinations,
        }

    # ── Evaluation ────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_test: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Evaluate the ensemble on test data.

        Returns
        -------
        dict[str, Any]
            ``{"ensemble_log_loss", "ensemble_accuracy", "individual_log_losses",
            "improvement_over_best_single", "best_single_model"}``
        """
        if not self.trained:
            raise RuntimeError("Ensemble must be fitted before evaluating.")

        # Individual predictions
        individual_losses: dict[str, float] = {}
        for model, weight, mtype in self._members:
            name = self._model_name(model)
            probs = self._predict_single(model, mtype, X_test, df_test)
            try:
                loss = float(log_loss(y_test, probs))
            except Exception:
                loss = float("inf")
            individual_losses[name] = loss

        # Ensemble prediction
        ensemble_probs = self.predict_proba(X_test, df_raw=df_test)
        ensemble_loss = float(log_loss(y_test, ensemble_probs))
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        accuracy = float(np.mean(ensemble_preds == y_test.values))

        # Comparison with best single model
        best_single_name = min(individual_losses, key=individual_losses.get)
        best_single_loss = individual_losses[best_single_name]
        improvement = best_single_loss - ensemble_loss

        report: dict[str, Any] = {
            "ensemble_log_loss": ensemble_loss,
            "ensemble_accuracy": accuracy,
            "individual_log_losses": individual_losses,
            "improvement_over_best_single": improvement,
            "best_single_model": best_single_name,
        }

        logger.info(
            "WeightedEnsemble '%s' test log-loss: %.4f "
            "(best single: %.4f, Delta=%.4f)",
            self.name, ensemble_loss, best_single_loss, improvement,
        )

        return report

    # ── Save / Load ───────────────────────────────────────

    def save(self, path: str | None = None) -> str:
        """Save the ensemble (members + weights) via joblib.

        Parameters
        ----------
        path : str, optional
            Output path.  Default: ``models/weighted_ensemble.joblib``.

        Returns
        -------
        str
            Path to the saved file.
        """
        import joblib

        if path is None:
            path = str(config.paths.models / "weighted_ensemble.joblib")

        payload = {
            "name": self.name,
            "members": self._members,
            "n_classes": self._n_classes,
        }
        joblib.dump(payload, path)
        logger.info("WeightedEnsemble '%s' saved to %s", self.name, path)
        return path

    @classmethod
    def load(cls, path: str) -> WeightedEnsemble:
        """Load a saved weighted ensemble from disk.

        Parameters
        ----------
        path : str
            Path to the saved ensemble file.

        Returns
        -------
        WeightedEnsemble
            Loaded ensemble with all models and weights restored.
        """
        import joblib

        payload = joblib.load(path)
        ensemble = cls(name=payload["name"])
        ensemble._members = payload["members"]
        ensemble._n_classes = payload.get("n_classes", 3)
        ensemble._fitted = True
        logger.info("WeightedEnsemble '%s' loaded from %s", ensemble.name, path)
        return ensemble


# ============================================================
#  Original Ensemble Model (trains sub-models internally)
# ============================================================


# ═══════════════════════════════════════════════════════════
#  Stacking Ensemble — meta-learner combines base models
# ═══════════════════════════════════════════════════════════


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
            multi_class="multinomial",
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


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_train: pd.DataFrame | None = None,
    df_val: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a complete ensemble end-to-end and return all results.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data (used for weight optimisation)
    X_test, y_test : test data (held-out evaluation)
    df_train, df_val, df_test : raw match DataFrames for Poisson model
    verbose : bool
        Print summary to console if True.

    Returns
    -------
    dict[str, Any]
        ``{ensemble, test_report, weights, ensemble_probs}``
    """
    ensemble = EnsembleModel()

    # Train & optimise weights
    fit_report = ensemble.fit(
        X_train, y_train, X_val, y_val,
        df_train=df_train, df_val=df_val,
    )

    # Evaluate on test
    test_report = ensemble.evaluate(X_test, y_test, df_test)

    # Predictions
    ensemble_probs = ensemble.predict_proba(X_test, df_test)

    if verbose:
        print("\n" + "=" * 90)
        print("  ENSEMBLE TRAINING RESULTS".center(88))
        print("=" * 90)

        print(f"\n  Validation log-loss: {fit_report['val_log_loss']:.4f}")
        print(f"  Test log-loss:       {test_report['ensemble_log_loss']:.4f}")
        print(f"  Test accuracy:       {test_report['ensemble_accuracy']:.2%}")
        print(f"\n  Best single model:   {test_report['best_single_model']} "
              f"({test_report['individual_log_losses'][test_report['best_single_model']]:.4f})")
        print(f"  Improvement:         Delta = {test_report['improvement_over_best_single']:+.4f}")
        print(f"\n  {ensemble.weight_summary}")
        print(f"\n  {'=' * 30}  LOG-LOSS BREAKDOWN {'=' * 30}")
        for name, loss in sorted(test_report["individual_log_losses"].items()):
            marker = " <- BEST" if abs(loss - min(test_report["individual_log_losses"].values())) < 1e-6 else ""
            print(f"    {name:<30s}  {loss:.4f}{marker}")
        print("=" * 90)
        print()

    return {
        "ensemble": ensemble,
        "test_report": test_report,
        "weights": fit_report["weights"],
        "ensemble_probs": ensemble_probs,
    }
