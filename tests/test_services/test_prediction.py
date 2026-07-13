"""
Tests for the prediction service — PredictionService.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.services.prediction_service import PredictionService


class TestPredictionService:
    def test_init_defaults(self) -> None:
        service = PredictionService()
        assert service._match_repo is None

    def test_init_with_repo(self) -> None:
        mock_repo = MagicMock()
        service = PredictionService(match_repo=mock_repo)
        assert service._match_repo is mock_repo

    def test_predict_upcoming_default(self) -> None:
        service = PredictionService()
        results = service.predict_upcoming()
        assert results == []  # Not implemented yet

    def test_predict_upcoming_with_limit(self) -> None:
        service = PredictionService()
        results = service.predict_upcoming(limit=5)
        assert results == []

    def test_predict_match_found(self) -> None:
        service = PredictionService()
        result = service.predict_match(match_id=1)
        assert result is None  # Not implemented yet

    def test_predict_match_nonexistent(self) -> None:
        service = PredictionService()
        result = service.predict_match(match_id=99999)
        assert result is None

    def test_backfill_predictions(self) -> None:
        service = PredictionService()
        results = service.backfill_predictions(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert results == []

    def test_backfill_same_date(self) -> None:
        service = PredictionService()
        d = date(2024, 6, 15)
        results = service.backfill_predictions(start_date=d, end_date=d)
        assert results == []

    def test_predict_upcoming_no_limit(self) -> None:
        """predict_upcoming should accept no args."""
        service = PredictionService()
        result = service.predict_upcoming()
        assert isinstance(result, list)

    def test_service_logs_activity(self) -> None:
        """Service methods log their actions."""
        with patch("src.services.prediction_service.logger") as mock_logger:
            service = PredictionService()
            service.predict_upcoming(limit=3)
            mock_logger.info.assert_called_once()
