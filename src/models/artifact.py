"""
ModelArtifact — versioned inference artifact bundling model + metadata.

Ensures that training and inference use identical feature schemas.
The artifact is serialised alongside the raw estimator so the API can
validate column count, names, and ordering before inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = "2.0"
TARGET_MAPPING: dict[str, int] = {"A": 0, "D": 1, "H": 2}
INVERSE_TARGET_MAPPING: dict[int, str] = {v: k for k, v in TARGET_MAPPING.items()}


@dataclass
class ModelArtifact:
    """Complete inference artifact that bundles an estimator with metadata.

    When saved, the artifact contains everything needed to run inference
    without coupling to the original training configuration.

    Attributes
    ----------
    model : Any
        Trained sklearn-compatible estimator with ``predict`` and
        ``predict_proba``.
    feature_names : list[str]
        Ordered list of feature column names the model expects.
    selected_feature_names : list[str]
        Feature names after optional selection (subset of *feature_names*).
    preprocessing_config : dict
        Snapshot of the preprocessing configuration at training time.
    feature_engineering_version : str
        Version identifier for the feature pipeline.
    model_type : str
        Algorithm type (e.g. ``xgboost``, ``lightgbm``).
    trained_at : str
        ISO-8601 timestamp of training completion.
    target_mapping : dict[str, int]
        Mapping from outcome label (``H``/``D``/``A``) to encoded integer.
    calibration_metadata : dict | None
        Calibration parameters if the model was calibrated.
    artifact_version : str
        Schema version for forward/backward compatibility.
    """

    model: Any
    feature_names: list[str]
    selected_feature_names: list[str] | None = None
    preprocessing_config: dict = field(default_factory=dict)
    feature_engineering_version: str = "1.0"
    model_type: str = "unknown"
    trained_at: str = ""
    target_mapping: dict[str, int] = field(default_factory=lambda: dict(TARGET_MAPPING))
    calibration_metadata: dict | None = None
    target_encoder_state: dict | None = None
    artifact_version: str = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if self.selected_feature_names is None:
            self.selected_feature_names = list(self.feature_names)

    @property
    def n_features(self) -> int:
        """Number of features the model expects (after selection)."""
        return len(self.selected_feature_names or self.feature_names)

    # ── Column management ───────────────────────────────────

    def select_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reindex *X* to match the persisted feature schema.

        - Drops columns not in *selected_feature_names*.
        - Adds missing columns filled with NaN (with a logged warning).
        - Reorders columns to exactly match *selected_feature_names*.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix from the pipeline.

        Returns
        -------
        pd.DataFrame
            Column-aligned feature matrix.

        Raises
        ------
        ValueError
            If any required column is missing and cannot be safely added.
        """
        required = set(self.selected_feature_names or [])
        actual = set(X.columns)
        missing = required - actual
        extra = actual - required

        if missing:
            logger.warning(
                "Artifact expects %d features but %d are missing: %s. "
                "Adding NaN-filled columns.",
                len(required), len(missing), sorted(missing)[:10],
            )
            for col in missing:
                X[col] = np.nan

        if extra:
            logger.debug("Dropping %d unexpected columns.", len(extra))
            X = X.drop(columns=list(extra), errors="ignore")

        # Reorder to match training-time order
        return X[list(self.selected_feature_names)]

    @property
    def predict_proba(self) -> Any:
        """Delegate to the underlying model's predict_proba."""
        return self.model.predict_proba

    @property
    def predict(self) -> Any:
        """Delegate to the underlying model's predict."""
        return self.model.predict

    @property
    def classes_(self) -> Any:
        """Delegate to the underlying model's classes_."""
        return self.model.classes_

    def __getattr__(self, name: str) -> Any:
        """Fall through to the underlying model for attributes like ``best_iteration_``.

        Guarded against recursion during unpickling when ``model`` may not
        yet be set on the instance.
        """
        # Prevent recursion during unpickling / __init__
        if name.startswith("_") or name in ("model",):
            raise AttributeError(name)
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    # ── Serialisation ───────────────────────────────────────

    def save(self, path: str) -> str:
        """Serialize the artifact to *path* via joblib.

        Returns the absolute path string.
        """
        import os
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info(
            "Artifact v%s saved to %s (%d features, %s)",
            self.artifact_version, p, self.n_features, self.model_type,
        )
        return str(p.absolute())

    @staticmethod
    def load(path: str) -> ModelArtifact:
        """Load a saved artifact from *path*.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the loaded object is not a ``ModelArtifact``.
        """
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {p}")

        obj = joblib.load(p)
        if not isinstance(obj, ModelArtifact):
            raise ValueError(
                f"Loaded object is not a ModelArtifact: {type(obj).__name__}. "
                "This file may be a raw estimator — retrain with the current pipeline."
            )
        logger.info(
            "Artifact v%s loaded from %s (%d features, %s)",
            obj.artifact_version, p, obj.n_features, obj.model_type,
        )
        return obj
