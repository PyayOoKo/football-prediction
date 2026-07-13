"""
Tests for the training service — TrainingService.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.training_service import TrainingService


class TestTrainingService:
    def test_init_defaults(self) -> None:
        service = TrainingService()
        assert service._model_dir is None

    def test_init_with_model_dir(self) -> None:
        service = TrainingService(model_dir=Path("/tmp/models"))
        assert service._model_dir == Path("/tmp/models")

    def test_train_xgboost_default(self) -> None:
        service = TrainingService()
        result = service.train(model_type="xgboost")
        assert result["status"] == "not_implemented"
        assert result["model_type"] == "xgboost"

    def test_train_with_tuning(self) -> None:
        service = TrainingService()
        result = service.train(model_type="xgboost", tune_hyperparams=True)
        assert result["status"] == "not_implemented"

    def test_train_logistic_regression(self) -> None:
        service = TrainingService()
        result = service.train(model_type="logistic_regression")
        assert result["model_type"] == "logistic_regression"

    def test_evaluate_model(self) -> None:
        service = TrainingService()
        result = service.evaluate(model_path="/tmp/model.joblib")
        assert isinstance(result, dict)
        assert result == {}

    def test_list_models_default(self) -> None:
        service = TrainingService()
        models = service.list_models()
        assert models == []

    def test_service_logs_training(self) -> None:
        with patch("src.services.training_service.logger") as mock_logger:
            service = TrainingService()
            service.train(model_type="xgboost", tune_hyperparams=False)
            mock_logger.info.assert_called_once()

    def test_service_logs_evaluation(self) -> None:
        with patch("src.services.training_service.logger") as mock_logger:
            service = TrainingService()
            service.evaluate(model_path="/tmp/model.joblib")
            mock_logger.info.assert_called_once()

    def test_training_returns_report(self) -> None:
        """Training should return a dict with at minimum model_type and status."""
        service = TrainingService()
        report = service.train(model_type="random_forest")
        assert "model_type" in report
        assert "status" in report
