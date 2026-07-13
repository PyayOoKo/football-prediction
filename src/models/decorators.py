"""
Decorators — auto-logging, auto-metrics, retry, and caching for model methods.

Designed as composable decorators that can be applied to any model method,
providing cross-cutting concerns without modifying the method body.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ═══════════════════════════════════════════════════════════
#  Auto-Metrics — compute standard metrics after predict
# ═══════════════════════════════════════════════════════════


def auto_metrics(
    y_true_arg: str = "y_test",
    y_pred_arg: str | None = None,
    y_proba_arg: str = "y_proba",
) -> Callable[[F], F]:
    """Decorator that automatically computes evaluation metrics.

    Wraps a method that returns predictions. If ``y_true`` is available
    (as a keyword argument), standard metrics are computed automatically
    and added to the return dict.

    Parameters
    ----------
    y_true_arg : str
        Name of the keyword argument containing true labels.
    y_pred_arg : str, optional
        Name of the keyword argument containing hard predictions.
        If None, computes from ``predict_proba`` output.
    y_proba_arg : str
        Name of the keyword argument containing probability predictions.

    Usage
    -----
    ::

        @auto_metrics(y_true_arg="y_true")
        def evaluate(self, X_test, y_true=None, y_proba=None, **kwargs):
            probs = self.predict_proba(X_test)
            preds = np.argmax(probs, axis=1)
            return {"predictions": preds, "probabilities": probs}
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Extract true labels and predicted probabilities
            y_true = kwargs.get(y_true_arg)
            if y_true is None:
                return result

            if isinstance(result, dict):
                y_pred = kwargs.get(y_pred_arg) if y_pred_arg else result.get("predictions")
                y_proba = kwargs.get(y_proba_arg) if y_proba_arg else result.get("probabilities")
            else:
                y_pred = None
                y_proba = None

            if y_proba is None and hasattr(result, "__getitem__"):
                # Try to extract from result tuple
                pass

            if y_proba is not None:
                metrics = _compute_metrics(y_true, y_pred, y_proba)
                if isinstance(result, dict):
                    result.update(metrics)
                return result

            return result

        return wrapper  # type: ignore

    return decorator


def _compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | None,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute a standard set of classification metrics."""
    if y_pred is None and y_proba is not None:
        y_pred = np.argmax(y_proba, axis=1)

    metrics: dict[str, float] = {}

    if y_pred is not None:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        try:
            metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        except Exception:
            pass

    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            pass
        try:
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

    return metrics


# ═══════════════════════════════════════════════════════════
#  Timing — measure and record execution duration
# ═══════════════════════════════════════════════════════════


def timed(func: F) -> F:
    """Decorator that measures and logs execution time.

    Adds ``duration_seconds`` to the return dict if it's a dict.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if isinstance(result, dict):
            result["duration_seconds"] = round(elapsed, 4)

        logger.debug("%s took %.3fs", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore


# ═══════════════════════════════════════════════════════════
#  Retry — retry on transient failures
# ═══════════════════════════════════════════════════════════


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a function on failure.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts (default 3).
    delay : float
        Initial delay in seconds (default 1.0).
    backoff : float
        Multiplier for exponential backoff (default 2.0).
    exceptions : tuple
        Exception types to catch and retry.

    Usage
    -----
    ::

        @retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        def download_data(self, url):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = delay * (backoff ** (attempt - 1))
                        logger.warning(
                            "%s attempt %d/%d failed: %s. Retrying in %.1fs...",
                            func.__qualname__,
                            attempt, max_attempts, exc, wait,
                        )
                        time.sleep(wait)
            raise last_exc  # type: ignore

        return wrapper  # type: ignore

    return decorator


# ═══════════════════════════════════════════════════════════
#  Validate Input — check feature columns match training
# ═══════════════════════════════════════════════════════════


def validate_input(func: F) -> F:
    """Decorator that validates input DataFrame columns.

    Checks that the first positional arg (``X``) has the same columns
    as ``self._feature_names`` if set.
    """

    @functools.wraps(func)
    def wrapper(self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self, "_feature_names") and self._feature_names:
            missing = set(self._feature_names) - set(X.columns)
            if missing:
                sorted_missing = sorted(missing)[:10]
                raise ValueError(
                    f"Input missing {len(missing)} features seen during fit: "
                    f"{sorted_missing}. Consider using feature alignment."
                )
        return func(self, X, *args, **kwargs)

    return wrapper  # type: ignore
