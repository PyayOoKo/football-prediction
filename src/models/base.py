"""
Abstract Base Model — universal interface for all football prediction models.

Every model type (Poisson, Elo, XGBoost, Neural Network, etc.) must
implement the 10 abstract methods defined here. The base class provides
automatic logging, timing, version tracking, input validation, and
metadata collection via a metaclass.

Usage
-----
::

    from src.models.base import BaseModel

    class MyModel(BaseModel):
        \"\"\"Concrete model example.\"\"\"

        def fit(self, X_train, y_train, **kwargs):
            self._register_fit()
            self._model = ...
            return self._report()

        def predict(self, X, **kwargs):
            self._require_fitted()
            return np.argmax(self.predict_proba(X, **kwargs), axis=1)

        def predict_proba(self, X, **kwargs):
            self._require_fitted()
            return self._model.predict_proba(X)

        # ... implement remaining 7 methods
"""

from __future__ import annotations

import abc
import functools
import inspect
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Model Meta — auto-logging & input validation
# ═══════════════════════════════════════════════════════════


class ModelMeta(abc.ABCMeta):
    """Metaclass that wraps every public method with auto-logging.

    For each method matching the public API pattern, a logging wrapper
    is applied that records:

    - Method entry/exit with parameters
    - Execution duration
    - Return type
    - Exceptions

    Subclasses inherit this behaviour automatically.
    """

    _MONITORED_METHODS: ClassVar[set[str]] = {
        "fit", "predict", "predict_proba",
        "cross_validate", "feature_importance",
        "calibrate", "evaluate",
        "save", "load",
    }

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any],
    ) -> ModelMeta:
        cls = super().__new__(mcs, name, bases, namespace)

        for method_name in mcs._MONITORED_METHODS:
            if method_name in namespace:
                original = namespace[method_name]
                if callable(original) and not getattr(original, "_is_abstract", False):
                    wrapped = mcs._wrap_method(method_name, original)
                    setattr(cls, method_name, wrapped)

        return cls

    @staticmethod
    def _wrap_method(name: str, fn: Any) -> Any:
        """Wrap a method with entry/exit logging and timing."""

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            _log_entry(self, name, args, kwargs)
            start = time.perf_counter()
            try:
                result = fn(self, *args, **kwargs)
                elapsed = time.perf_counter() - start
                _log_exit(self, name, elapsed, result)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error(
                    "%s.%s failed after %.2fs: %s",
                    type(self).__name__, name, elapsed, exc,
                )
                raise

        return wrapper


def _log_entry(self: Any, method: str, args: tuple, kwargs: dict) -> None:
    """Log method entry with key parameters."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    sig = _resolve_method_signature(type(self), method)
    bound = sig.bind(self, *args, **kwargs)
    bound.apply_defaults()
    params = {}
    for p_name, p_val in bound.arguments.items():
        if p_name == "self":
            continue
        if isinstance(p_val, (pd.DataFrame, np.ndarray)):
            params[p_name] = f"{type(p_val).__name__}({len(p_val)} rows)"
        elif isinstance(p_val, pd.Series):
            params[p_name] = f"Series({len(p_val)} rows)"
        else:
            params[p_name] = repr(p_val)[:80]
    logger.debug("%s.%s(%s)", type(self).__name__, method, params)


def _log_exit(self: Any, method: str, elapsed: float, result: Any) -> None:
    """Log method exit with timing."""
    result_size = ""
    if isinstance(result, pd.DataFrame):
        result_size = f" → DataFrame({len(result)}×{len(result.columns)})"
    elif isinstance(result, np.ndarray):
        result_size = f" → array{result.shape}"
    elif isinstance(result, dict):
        result_size = f" → dict({len(result)} keys)"
    logger.debug(
        "%s.%s completed in %.3fs%s",
        type(self).__name__, method, elapsed, result_size,
    )


def _resolve_method_signature(cls: type, method: str) -> Any:
    """Resolve inspect.Signature for *method* walking the MRO.

    ``type(self).__dict__[method]`` raises ``KeyError`` when the
    subclass does **not** override the method and inherits it from
    a parent.  This helper walks the MRO to find the first class
    that defines the method, then returns its signature.

    The ``@functools.wraps`` decorator on the metaclass wrapper
    copies ``__wrapped__`` so ``inspect.signature`` can follow it
    automatically via the ``follow_wrapped`` parameter (default
    ``True``).
    """
    for ancestor in cls.__mro__:
        if method in ancestor.__dict__:
            original = ancestor.__dict__[method]
            return inspect.signature(original)
    msg = f"{method!r} not found in {cls.__name__} MRO"
    raise AttributeError(msg)


# ═══════════════════════════════════════════════════════════
#  Abstract Base Model
# ═══════════════════════════════════════════════════════════


class BaseModel(metaclass=ModelMeta):
    """Universal abstract base class for all prediction models.

    Every model type must implement all abstract methods. The metaclass
    provides automatic logging, timing, and exception tracking for every
    public API method.

    Parameters
    ----------
    model_name : str, optional
        Human-readable name (auto-generated if omitted).
    model_version : str, optional
        Semantic version string (default ``0.1.0``).
    metadata : dict, optional
        Arbitrary key-value metadata (author, tags, notes, etc.).
    random_seed : int, optional
        Random seed for reproducibility.
    """

    # ── Model identity ─────────────────────────────────
    model_type: str = "base"  # Override in subclass

    def __init__(
        self,
        model_name: str | None = None,
        model_version: str = "0.1.0",
        metadata: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.model_id: str = str(uuid.uuid4())[:8]
        self.model_name: str = model_name or f"{self.model_type}_{self.model_id}"
        self.model_version: str = model_version
        self.metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": __import__("sys").version,
            **(metadata or {}),
        }
        self.random_seed: int | None = random_seed

        # ── Training state ──
        self._fitted: bool = False
        self._fit_started_at: datetime | None = None
        self._fit_completed_at: datetime | None = None
        self._fit_duration_seconds: float = 0.0
        self._n_features: int = 0
        self._n_classes: int = 0
        self._feature_names: list[str] = []
        self._training_metrics: dict[str, Any] = {}

        # ── Calibration state ──
        self._calibrated: bool = False

    # ── Properties ─────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted

    @property
    def calibrated(self) -> bool:
        """Whether the model has been calibrated."""
        return self._calibrated

    @property
    def fit_duration(self) -> float:
        """Training duration in seconds."""
        return self._fit_duration_seconds

    @property
    def model_summary(self) -> str:
        """Human-readable model summary."""
        status = "fitted" if self._fitted else "unfitted"
        cal = ", calibrated" if self._calibrated else ""
        return (
            f"<{self.model_type} '{self.model_name}' v{self.model_version} "
            f"[{status}{cal}] id={self.model_id}>"
        )

    # ── Abstract methods (must implement) ──────────────

    @abc.abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model on the provided data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target vector.
        X_val : pd.DataFrame, optional
            Validation feature matrix.
        y_val : pd.Series, optional
            Validation target vector.
        **kwargs
            Model-specific training parameters.

        Returns
        -------
        dict[str, Any]
            Training report with at minimum:
            ``train_loss``, ``val_loss``, ``duration_seconds``.
        """
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Predict hard class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        **kwargs
            Model-specific predict parameters.

        Returns
        -------
        np.ndarray
            Integer label array of shape ``(n,)``.
        """
        ...

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        **kwargs
            Model-specific predict parameters.

        Returns
        -------
        np.ndarray
            Probability array of shape ``(n, n_classes)``.
        """
        ...

    @abc.abstractmethod
    def save(self, path: str | None = None, **kwargs: Any) -> str:
        """Serialise the model to disk.

        Parameters
        ----------
        path : str, optional
            Output file path. If omitted, auto-generates one based
            on model name and version.
        **kwargs
            Serialisation options (format, compression, etc.).

        Returns
        -------
        str
            Path to the saved file.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str, **kwargs: Any) -> BaseModel:
        """Deserialise a model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file.
        **kwargs
            Deserialisation options.

        Returns
        -------
        BaseModel
            Loaded model instance.
        """
        ...

    @abc.abstractmethod
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run k-fold cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix.
        y : pd.Series
            Full target vector.
        n_folds : int
            Number of folds (default 5).
        **kwargs
            CV options (fold strategy, scoring, etc.).

        Returns
        -------
        dict[str, Any]
            CV results: per-fold scores, mean, std.
        """
        ...

    @abc.abstractmethod
    def feature_importance(
        self,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Return feature importance scores.

        Returns
        -------
        dict[str, float]
            Mapping of feature name to importance score.
            Returns empty dict if the model does not support
            feature importance.

        Raises
        ------
        ModelNotFittedError
            If called before ``fit()``.
        """
        ...

    @abc.abstractmethod
    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Calibrate model probabilities on a held-out set.

        Parameters
        ----------
        X_cal : pd.DataFrame
            Calibration feature matrix.
        y_cal : pd.Series
            Calibration target vector.
        **kwargs
            Calibration options (method, cv folds, etc.).

        Returns
        -------
        dict[str, Any]
            Calibration report with ECE, Brier score, etc.

        Raises
        ------
        ModelNotFittedError
            If called before ``fit()``.
        """
        ...

    @abc.abstractmethod
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate the model on test data.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix.
        y_test : pd.Series
            True labels.
        **kwargs
            Evaluation options (metrics to compute, plots, etc.).

        Returns
        -------
        dict[str, Any]
            Evaluation report with all computed metrics.

        Raises
        ------
        ModelNotFittedError
            If called before ``fit()``.
        """
        ...

    # ── Template method: end-to-end pipeline ────────────

    def train_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        X_cal: pd.DataFrame | None = None,
        y_cal: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the full training pipeline: fit -> calibrate -> evaluate.

        This is a template method that subclasses can override to
        customise the pipeline flow, but they should call
        ``super().train_pipeline()`` at the appropriate point.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : validation data (optional)
        X_test, y_test : test data (optional, for evaluation)
        X_cal, y_cal : calibration data (optional)
        **kwargs
            Passed to ``fit()``, ``calibrate()``, ``evaluate()``.

        Returns
        -------
        dict[str, Any]
            Complete training report with fit, calibration, evaluation
            results, and model metadata.
        """
        report: dict[str, Any] = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "model_id": self.model_id,
        }

        # Fit
        fit_result = self.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            **kwargs.get("fit_kwargs", {}),
        )
        report["fit"] = fit_result

        # Calibrate
        if X_cal is not None and y_cal is not None:
            cal_result = self.calibrate(
                X_cal, y_cal,
                **kwargs.get("calibrate_kwargs", {}),
            )
            report["calibrate"] = cal_result

        # Evaluate
        if X_test is not None and y_test is not None:
            eval_result = self.evaluate(
                X_test, y_test,
                **kwargs.get("evaluate_kwargs", {}),
            )
            report["evaluate"] = eval_result

        # Cross-validate (optional, uses full dataset)
        if kwargs.get("cross_validate", False):
            cv_result = self.cross_validate(
                pd.concat([X_train, X_val]) if X_val is not None else X_train,
                pd.concat([y_train, y_val]) if y_val is not None else y_train,
                **kwargs.get("cv_kwargs", {}),
            )
            report["cross_validate"] = cv_result

        return report

    # ── Protected helpers for subclasses ───────────────

    def _register_fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Call at the start of ``fit()`` to record training metadata."""
        self._fitted = False
        self._fit_started_at = datetime.now(timezone.utc)
        self._n_features = X_train.shape[1]
        self._n_classes = int(y_train.nunique())
        self._feature_names = list(X_train.columns)

    def _complete_fit(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Call at the end of ``fit()`` to finalise training metadata.

        Parameters
        ----------
        metrics : dict
            Training metrics from the subclass.

        Returns
        -------
        dict
            Enriched metrics with timing and metadata.
        """
        self._fit_completed_at = datetime.now(timezone.utc)
        if self._fit_started_at is not None:
            self._fit_duration_seconds = (
                self._fit_completed_at - self._fit_started_at
            ).total_seconds()
        self._fitted = True
        self._training_metrics = metrics

        metrics["duration_seconds"] = self._fit_duration_seconds
        metrics["n_features"] = self._n_features
        metrics["n_classes"] = self._n_classes
        metrics["fitted_at"] = self._fit_completed_at.isoformat()

        logger.info(
            "%s fitted: %d features, %d classes, %.2fs",
            self.model_summary,
            self._n_features,
            self._n_classes,
            self._fit_duration_seconds,
        )
        return metrics

    def _require_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._fitted:
            from src.utils.exceptions import ModelNotFittedError
            raise ModelNotFittedError(
                f"{self.model_summary} has not been fitted. "
                "Call .fit() first."
            )

    def _validate_input(self, X: pd.DataFrame, method: str = "predict") -> None:
        """Validate input data against fitted model expectations.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        method : str
            Calling method name (for error messages).

        Raises
        ------
        ValueError
            If feature count mismatch.
        """
        self._require_fitted()
        if self._n_features > 0 and X.shape[1] != self._n_features:
            raise ValueError(
                f"{self.model_summary}.{method}() expected "
                f"{self._n_features} features but got {X.shape[1]}. "
                f"Features seen during fit: {self._feature_names[:5]}..."
            )

    def __repr__(self) -> str:
        return self.model_summary


# ═══════════════════════════════════════════════════════════
#  Custom exceptions
# ═══════════════════════════════════════════════════════════


class ModelNotFittedError(RuntimeError):
    """Raised when a model operation requires a fitted model."""
    pass


class ModelNotFoundError(RuntimeError):
    """Raised when a model file cannot be loaded."""
    pass


class ModelValidationError(ValueError):
    """Raised when input validation fails."""
    pass
