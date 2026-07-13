"""
Models Framework — universal prediction model interface.

Provides an abstract base class, plugin architecture, factory pattern,
model registry with automatic versioning, and automatic serialization.

Architecture
------------
::

    ┌──────────────────────────────────────────────────────────────┐
    │                        ModelFactory                          │
    │  Creates model instances via plugin resolution + fallback     │
    └────────┬─────────────────────────────────────────────────────┘
             │ creates
    ┌────────▼─────────────────────────────────────────────────────┐
    │                        BaseModel (ABC)                       │
    │  fit() | predict() | predict_proba() | save() | load()       │
    │  cross_validate() | feature_importance() | calibrate()       │
    │  evaluate()                                                  │
    │                                                              │
    │  Auto-logging via ModelMeta metaclass                        │
    │  Input validation via _validate_input()                      │
    └────────┬─────────────────────────────────────────────────────┘
             │ discovered by
    ┌────────▼────────────┐  ┌────────────────┐  ┌───────────────┐
    │   PluginRegistry    │  │ ModelRegistry   │  │ ModelSerializer│
    │  discovery + avail  │  │ versioning +    │  │ joblib/pickle │
    │  entry points + pkg │  │ promote + DB    │  │ ONNX/JSON     │
    └─────────────────────┘  └────────────────┘  └───────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │                    Decorators                                │
    │  @auto_metrics | @timed | @retry | @validate_input           │
    └──────────────────────────────────────────────────────────────┘

Quick Start
-----------
::

    from src.models import ModelFactory, BaseModel

    # Create a model via the factory
    factory = ModelFactory()
    model = factory.create("xgboost")

    # Train
    report = model.fit(X_train, y_train, X_val, y_val)

    # Predict
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    # Evaluate + calibrate
    eval_report = model.evaluate(X_test, y_test)
    cal_report = model.calibrate(X_cal, y_cal)

    # Save / Load
    path = model.save()
    loaded = model.__class__.load(path)

    # Cross-validate
    cv_report = model.cross_validate(X, y, n_folds=5)

    # Feature importance
    importance = model.feature_importance()
"""

from __future__ import annotations

from src.models.base import BaseModel, ModelMeta, ModelNotFittedError, ModelNotFoundError
from src.models.decorators import auto_metrics, timed, retry, validate_input
from src.models.factory import ModelFactory
from src.models.plugins import Plugin, PluginRegistry
from src.models.registry import ModelRegistry
from src.models.serialization import ModelSerializer, ONNXWrapper

__all__ = [
    # Core
    "BaseModel",
    "ModelMeta",
    # Errors
    "ModelNotFittedError",
    "ModelNotFoundError",
    # Decorators
    "auto_metrics",
    "timed",
    "retry",
    "validate_input",
    # Factory & Registry
    "ModelFactory",
    "Plugin",
    "PluginRegistry",
    "ModelRegistry",
    # Serialization
    "ModelSerializer",
    "ONNXWrapper",
]

# ── Convenience: create default factory ──────────────

_default_factory: ModelFactory | None = None


def get_default_factory() -> ModelFactory:
    """Get or create the default model factory (singleton)."""
    global _default_factory
    if _default_factory is None:
        _default_factory = ModelFactory()
    return _default_factory


def create_model(model_type: str, **kwargs: object) -> BaseModel:
    """Convenience function — create a model from the default factory.

    Usage
    -----
    ::

        model = create_model("xgboost", random_seed=42)
    """
    return get_default_factory().create(model_type, **kwargs)
