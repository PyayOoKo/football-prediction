"""
Tests for the prediction service — PredictionService.

Uses temporary directories and a simple dummy model class to avoid
depending on real files or external services.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.services.prediction_service import PredictionService


# ── Dummy model that can be pickled by joblib ─────────────────
class _DummyModel:
    """Minimal model with predict/predict_proba for testing."""

    def __init__(self) -> None:
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.array([0] * n)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.full((n, 3), [0.5, 0.3, 0.2])


@pytest.fixture
def temp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def dummy_model() -> _DummyModel:
    return _DummyModel()


class TestPredictionService:
    def test_init_defaults(self, temp_models_dir: Path) -> None:
        with patch("src.services.prediction_service._global_config") as mock_cfg:
            mock_cfg.paths.models = temp_models_dir
            service = PredictionService()
            assert service._model_dir == temp_models_dir

    def test_init_with_model_dir(self) -> None:
        service = PredictionService(model_dir=Path("/tmp/models"))
        assert service._model_dir == Path("/tmp/models")

    def test_load_model_by_name(
        self, temp_models_dir: Path, dummy_model: _DummyModel
    ) -> None:
        """Loading a model by name should succeed."""
        import joblib

        model_path = temp_models_dir / "test_model.joblib"
        joblib.dump(dummy_model, model_path)

        with patch("src.services.prediction_service._global_config") as mock_cfg:
            mock_cfg.paths.models = temp_models_dir
            service = PredictionService(model_dir=temp_models_dir)
            loaded = service._load_model("test_model.joblib")
            assert loaded is not None

    def test_load_model_missing_raises(self, temp_models_dir: Path) -> None:
        """Loading a non-existent model should raise FileNotFoundError."""
        service = PredictionService(model_dir=temp_models_dir)
        with pytest.raises(FileNotFoundError):
            service._load_model("nonexistent.joblib")

    def test_load_model_no_models_raises(self, temp_models_dir: Path) -> None:
        """Loading with no models in directory should raise FileNotFoundError."""
        service = PredictionService(model_dir=temp_models_dir)
        with pytest.raises(FileNotFoundError):
            service._load_model()

    def test_predict_upcoming_no_model(self, temp_models_dir: Path) -> None:
        """predict_upcoming should raise FileNotFoundError if no model."""
        service = PredictionService(model_dir=temp_models_dir)
        with pytest.raises(FileNotFoundError):
            service.predict_upcoming()

    def test_predict_upcoming_data_not_found(
        self, temp_models_dir: Path, dummy_model: _DummyModel
    ) -> None:
        """predict_upcoming should raise FileNotFoundError if data missing."""
        import joblib

        joblib.dump(dummy_model, temp_models_dir / "model.joblib")

        service = PredictionService(model_dir=temp_models_dir)
        with pytest.raises(FileNotFoundError, match="Match data not found"):
            service.predict_upcoming(data_path="/nonexistent/path.csv")

    def test_predict_match_nonexistent(self, temp_models_dir: Path) -> None:
        """predict_match should raise FileNotFoundError when no model exists."""
        service = PredictionService(model_dir=temp_models_dir)
        with pytest.raises(FileNotFoundError):
            service.predict_match(match_id=99999)

    def test_predict_match_with_model_no_data(
        self, temp_models_dir: Path, dummy_model: _DummyModel
    ) -> None:
        """predict_match should return None if data doesn't exist."""
        import joblib

        joblib.dump(dummy_model, temp_models_dir / "model.joblib")

        service = PredictionService(model_dir=temp_models_dir)
        # Data path doesn't exist
        with patch.object(service, "_load_model", return_value=dummy_model):
            result = service.predict_match(match_id=99999)
            assert result is None

    def test_predict_with_odds_no_odds(
        self, temp_models_dir: Path, dummy_model: _DummyModel
    ) -> None:
        """predict_with_odds should return empty list if no predictions."""
        import joblib

        joblib.dump(dummy_model, temp_models_dir / "model.joblib")

        service = PredictionService(model_dir=temp_models_dir)
        with patch.object(service, "predict_upcoming", return_value=[]):
            results = service.predict_with_odds()
            assert results == []

    def test_backfill_empty(self, temp_models_dir: Path) -> None:
        """backfill_predictions should return empty list for no data."""
        service = PredictionService(model_dir=temp_models_dir)
        with patch.object(service, "_load_model", return_value=_DummyModel()):
            results = service.backfill_predictions(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
            )
            assert results == []

    def test_save_predictions_csv(self, temp_models_dir: Path) -> None:
        """_save_predictions should save CSV correctly."""
        service = PredictionService(model_dir=temp_models_dir)
        results = [{"match_id": 1, "prediction": "Home Win", "confidence": 0.8}]
        output_path = temp_models_dir / "out.csv"
        service._save_predictions(results, output_path)
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.iloc[0]["match_id"] == 1

    def test_save_predictions_json(self, temp_models_dir: Path) -> None:
        """_save_predictions should save JSON correctly."""
        service = PredictionService(model_dir=temp_models_dir)
        results = [{"match_id": 1, "prediction": "Home Win", "confidence": 0.8}]
        output_path = temp_models_dir / "out.json"
        service._save_predictions(results, output_path)
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["match_id"] == 1

    def test_service_logs_activity(self, temp_models_dir: Path) -> None:
        """Service methods should log their actions."""
        with patch("src.services.prediction_service.logger") as mock_logger:
            service = PredictionService(model_dir=temp_models_dir)
            try:
                service.predict_upcoming(data_path="/dev/null/missing.csv")
            except FileNotFoundError:
                pass
            mock_logger.info.assert_called()
