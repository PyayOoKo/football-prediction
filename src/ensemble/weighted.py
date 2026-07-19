"""Weighted Ensemble - combine pre-trained models via weighted averaging."""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import EnsembleConfig, config
from src.models.protocol import IModel, ensure_predict_proba

logger = logging.getLogger(__name__)

# Default weight grid search step
_GRID_STEP = 0.10


class WeightedEnsemble:
    """Lightweight ensemble combining pre-trained models via weighted averaging.

    Unlike ``EnsembleModel`` (which trains sub-models internally), this class
    accepts a list of already-trained ``(model, weight)`` tuples and averages
    their predictions. It uses a unified model interface via adapters that
    normalize both sklearn-compatible models and statistical models to a
    common ``predict_proba(X, df_raw)`` signature.

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

        # ML models (sklearn-style)
        ensemble = WeightedEnsemble([
            (xgb_model, 0.5),
            (lr_model, 0.3),
            (rf_model, 0.2),
        ])
        probs = ensemble.predict_proba(X_test)

        # Statistical models
        ensemble = WeightedEnsemble([
            (poisson_model, 0.4),
            (elo_model, 0.3),
            (dixon_coles_model, 0.3),
        ])
        probs = ensemble.predict_proba(df_raw=df_raw)

        # Mixed models
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
        # Internal storage: list of (wrapped_model, weight)
        # All models are wrapped with IModel protocol adapters
        self._members: list[tuple[IModel, float]] = []

        if models_and_weights is not None:
            # Add all models first, then normalise ONCE to preserve intended ratios
            for model, weight in models_and_weights:
                wrapped = ensure_predict_proba(model)
                self._members.append((wrapped, float(weight)))
            self._normalise_weights()

        self._fitted: bool = len(self._members) > 0
        self._n_classes: int = 3

    # ── Properties ────────────────────────────────────────

    @property
    def members(self) -> list[tuple[IModel, float]]:
        """Return a copy of ensemble members ``[(model, weight), ...]``."""
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
            for m, w in self._members
        }

    @property
    def trained(self) -> bool:
        """Whether the ensemble has at least one member."""
        return len(self._members) > 0

    @property
    def weight_summary(self) -> str:
        """Human-readable weight summary."""
        parts = []
        for model, weight in self._members:
            name = self._model_name(model)
            parts.append(f"  {name}: {weight:.3f}")
        return "Weights:\n" + "\n".join(parts)

    # ── Member management ────────────────────────────────

    @staticmethod
    def _model_name(model: IModel | Any) -> str:
        """Return a readable name for a model."""
        # Unwrap adapter to get actual model
        if hasattr(model, '_model'):
            raw = type(model._model).__name__
        else:
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

    def add_model(
        self,
        model: Any,
        weight: float = 1.0,
    ) -> WeightedEnsemble:
        """Add a model to the ensemble.

        Parameters
        ----------
        model : Any
            Trained model (any type with predict_proba or predict_matches).
        weight : float
            Initial weight (will be normalised with all other weights
            on the next ``fit()`` or ``set_weights()`` call).
            Default 1.0.

        Returns
        -------
        WeightedEnsemble
            Self, for method chaining.
        """
        wrapped = ensure_predict_proba(model)
        self._members.append((wrapped, float(weight)))
        self._normalise_weights()
        self._fitted = True
        return self

    def _normalise_weights(self) -> None:
        """Normalise all member weights to sum to 1.0."""
        total = sum(w for _, w in self._members)
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
            model, _ = self._members[i]
            name = self._model_name(model)
            member = list(self._members[i])
            member[1] = weights.get(name, 0.0)
            self._members[i] = tuple(member)
        self._normalise_weights()
        return self

    # ── Prediction ────────────────────────────────────────

    def _predict_single(
        self,
        model: IModel,
        X: pd.DataFrame,
        df_raw: pd.DataFrame | None,
    ) -> np.ndarray:
        """Get (n, 3) probability array from a single model.
        
        All models are wrapped with IModel protocol adapters that provide
        a unified predict_proba(X, df_raw) interface.
        """
        n = len(X) if hasattr(X, "__len__") else 0
        
        try:
            probs = model.predict_proba(X=X, df_raw=df_raw)
            return np.asarray(probs, dtype=np.float64)
        except Exception as e:
            logger.warning(
                "predict_proba failed for %s: %s",
                self._model_name(model), e,
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
            Feature matrix (for ML models).
        df_raw : pd.DataFrame, optional
            Raw match data (for statistical models).
            Required if any ensemble member needs raw data.

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

        for model, weight in self._members:
            if weight <= 0:
                continue
            probs = self._predict_single(model, X, df_raw)
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
            Raw validation match data (for statistical models).
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


