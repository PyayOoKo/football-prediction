"""ModelLoader — model discovery, loading, and type detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelLoader:
    DEFAULT_SEARCH_PATHS: list[str] = [
        "models/three_model_blend.joblib",
        "models/ensemble.pkl",
        "models/ensemble_model.joblib",
        "models/weighted_ensemble.joblib",
        "models/xgboost_model.pkl",
        "models/model.pkl",
        "models/xgboost.pkl",
    ]

    @staticmethod
    def detect_model_type(model: Any) -> str:
        modname = type(model).__module__
        clsname = type(model).__name__

        if clsname == "EnsembleModel":
            return "ensemble_model"
        if clsname == "WeightedEnsemble":
            return "weighted_ensemble"
        if hasattr(model, "predict_matches"):
            return "phase3"
        if hasattr(model, "predict_proba"):
            return "phase4"
        return "unknown"

    @staticmethod
    def load(path: str | Path | None = None) -> tuple[Any, dict[str, Any]]:
        import joblib

        metadata: dict[str, Any] = {
            "path": "",
            "model_type": "none",
            "name": "none",
            "loaded": False,
        }

        candidates: list[Path] = []
        if path is not None:
            candidates = [Path(path)]
        else:
            candidates = [Path(p) for p in ModelLoader.DEFAULT_SEARCH_PATHS]

        for candidate in candidates:
            if not candidate.exists():
                logger.debug("Model not found at: %s", candidate)
                continue

            try:
                model = joblib.load(candidate)
                mtype = ModelLoader.detect_model_type(model)

                metadata = {
                    "path": str(candidate.resolve()),
                    "model_type": mtype,
                    "name": candidate.stem,
                    "loaded": True,
                }

                logger.info(
                    "Loaded model: %s (type=%s)", candidate.name, mtype,
                )
                return model, metadata

            except Exception as exc:
                logger.warning("Failed to load %s: %s", candidate, exc)
                continue

        logger.warning("No model found at any search path")
        return None, metadata

    @staticmethod
    def get_model_name(model: Any) -> str:
        if hasattr(model, "model_name"):
            return str(model.model_name)
        if hasattr(model, "name"):
            return str(model.name)
        return type(model).__name__
