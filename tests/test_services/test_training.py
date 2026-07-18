"""
Tests for the training service — TrainingService.

Uses temporary directories and mocking to avoid depending on real data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.training_service import TrainingService


@pytest.fixture
def temp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


class TestTrainingService:
    def test_init_defaults(self, temp_models_dir: Path) -> None:
        with patch("src.services.training_service._global_config") as mock_cfg:
            mock_cfg.paths.models = temp_models_dir
            service = TrainingService()
            assert service._model_dir == temp_models_dir

    def test_init_with_model_dir(self) -> None:
        service = TrainingService(model_dir=Path("/tmp/models"))
        assert service._model_dir == Path("/tmp/models")

    def test_init_with_config(self) -> None:
        """Should accept explicit config object."""
        mock_cfg = MagicMock()
        mock_cfg.paths.models = Path("/custom/models")
        service = TrainingService(config=mock_cfg)
        assert service._config is mock_cfg
        assert service._model_dir == Path("/custom/models")

    def test_train_no_data_raises(self, temp_models_dir: Path) -> None:
        """Training without data should raise FileNotFoundError."""
        service = TrainingService(model_dir=temp_models_dir)
        with pytest.raises(FileNotFoundError, match="Training data not found"):
            service.train(data_path="/nonexistent/path.csv")

    def test_train_data_not_found(self, temp_models_dir: Path) -> None:
        """Training with missing data path should raise FileNotFoundError."""
        service = TrainingService(model_dir=temp_models_dir)
        with pytest.raises(FileNotFoundError):
            service.train(data_path="/dev/null/missing.csv")

    def test_evaluate_model(self, temp_models_dir: Path) -> None:
        """Evaluate should raise FileNotFoundError for missing model."""
        service = TrainingService(model_dir=temp_models_dir)
        with pytest.raises(FileNotFoundError):
            service.evaluate(model_path="/tmp/model.joblib")

    def test_list_models_empty(self, temp_models_dir: Path) -> None:
        """list_models should return empty list for empty directory."""
        service = TrainingService(model_dir=temp_models_dir)
        models = service.list_models()
        assert models == []

    def test_list_models_with_files(self, tmp_path: Path) -> None:
        """list_models should return metadata about .joblib files."""
        import joblib

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"dummy": True}, model_dir / "test_model.joblib")
        (model_dir / "other.txt").write_text("ignore")

        mock_cfg = MagicMock()
        mock_cfg.paths.models = model_dir

        service = TrainingService(model_dir=model_dir, config=mock_cfg)
        models = service.list_models()
        assert len(models) == 1
        assert models[0]["file_name"] == "test_model.joblib"

    def test_list_models_nonexistent_dir(self, temp_models_dir: Path) -> None:
        """list_models should return empty list if directory does not exist."""
        nonexistent = Path("/nonexistent/models")
        service = TrainingService(model_dir=nonexistent)
        models = service.list_models()
        assert models == []

    def test_service_logs_training(self, temp_models_dir: Path) -> None:
        """Training service should log its actions."""
        with patch("src.services.training_service.logger") as mock_logger:
            with patch("src.services.training_service._global_config") as mock_cfg:
                mock_cfg.paths.models = temp_models_dir
                mock_cfg.train.model_type = "logistic_regression"
                service = TrainingService()
                try:
                    service.train(data_path="/nonexistent/missing.csv")
                except FileNotFoundError:
                    pass
                mock_logger.info.assert_called()

    def test_apply_feature_selection_disabled(self, temp_models_dir: Path) -> None:
        """Feature selection should be a no-op when disabled."""
        import pandas as pd

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        y = pd.Series([0, 1, 0])

        from dataclasses import dataclass

        @dataclass
        class _MockFS:
            enabled: bool = False
            drop_redundant_first: bool = False
            correlation_threshold: float = 1.0
            method: str = "none"  # Not a real method — no selection occurs
            n_features: int = 30
            importance_threshold: float = 0.01

        mock_cfg = MagicMock()
        mock_cfg.feature_selection = _MockFS(enabled=False)

        service = TrainingService(model_dir=temp_models_dir, config=mock_cfg)
        result = service._apply_feature_selection(X, y)
        assert len(result.columns) == 3
        assert list(result.columns) == ["a", "b", "c"]

    def test_extract_importances_with_model(self, temp_models_dir: Path) -> None:
        """_extract_importances should return None for models without importances."""
        import pandas as pd

        model = MagicMock(spec=[])  # No feature_importances_
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = TrainingService._extract_importances(model, X)
        assert result is None
