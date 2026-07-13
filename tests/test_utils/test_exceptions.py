"""
Tests for custom exception hierarchy.
"""

from __future__ import annotations

import pytest

from src.utils.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    DatabaseError,
    FootballPredictionError,
    ModelNotFoundError,
    PredictionError,
    ScraperAuthError,
    ScraperError,
    ScraperRateLimitError,
    TrainingError,
    ValidationError,
)


class TestBaseException:
    def test_base_exception(self) -> None:
        err = FootballPredictionError("Base error")
        assert str(err) == "Base error"
        assert isinstance(err, Exception)


class TestDataExceptions:
    def test_data_not_found_default(self) -> None:
        err = DataNotFoundError()
        assert "Data not found" in str(err)
        assert err.resource == ""

    def test_data_not_found_with_resource(self) -> None:
        err = DataNotFoundError(resource="matches_2024.csv")
        assert "matches_2024.csv" in str(err)
        assert err.resource == "matches_2024.csv"

    def test_data_not_found_custom_message(self) -> None:
        err = DataNotFoundError(message="Custom message", resource="file.csv")
        assert "Custom message" in str(err)

    def test_database_error(self) -> None:
        err = DatabaseError("Connection refused")
        assert "Connection refused" in str(err)
        assert isinstance(err, FootballPredictionError)


class TestModelExceptions:
    def test_model_not_found_default(self) -> None:
        err = ModelNotFoundError()
        assert "Model not found" in str(err)
        assert err.model_name == ""

    def test_model_not_found_with_name(self) -> None:
        err = ModelNotFoundError(model_name="xgboost_v1.joblib")
        assert "xgboost_v1.joblib" in str(err)
        assert err.model_name == "xgboost_v1.joblib"

    def test_training_error(self) -> None:
        err = TrainingError("Convergence failed")
        assert "Convergence failed" in str(err)

    def test_prediction_error(self) -> None:
        err = PredictionError("Feature mismatch")
        assert "Feature mismatch" in str(err)


class TestScraperExceptions:
    def test_scraper_error(self) -> None:
        err = ScraperError("Generic scraper error")
        assert "Generic scraper error" in str(err)

    def test_rate_limit_default(self) -> None:
        err = ScraperRateLimitError()
        assert "Rate limited" in str(err)
        assert err.retry_after == 60

    def test_rate_limit_custom(self) -> None:
        err = ScraperRateLimitError(message="Too fast", retry_after=120)
        assert "Too fast" in str(err)
        assert err.retry_after == 120

    def test_auth_error(self) -> None:
        err = ScraperAuthError("Invalid API key")
        assert "Invalid API key" in str(err)


class TestConfigException:
    def test_configuration_error(self) -> None:
        err = ConfigurationError("Missing .env file")
        assert "Missing .env file" in str(err)

    def test_validation_error(self) -> None:
        err = ValidationError("Invalid team name")
        assert "Invalid team name" in str(err)


class TestInheritance:
    def test_all_inherit_from_base(self) -> None:
        """All custom exceptions should inherit from FootballPredictionError."""
        exceptions = [
            DataNotFoundError(),
            DatabaseError(),
            ModelNotFoundError(),
            TrainingError(),
            PredictionError(),
            ScraperError(),
            ScraperRateLimitError(),
            ScraperAuthError(),
            ConfigurationError(),
            ValidationError(),
        ]
        for exc in exceptions:
            assert isinstance(exc, FootballPredictionError), f"{type(exc).__name__} does not inherit from FootballPredictionError"
            assert isinstance(exc, Exception), f"{type(exc).__name__} does not inherit from Exception"

    def test_catch_base(self) -> None:
        """Catching FootballPredictionError catches all subclasses."""
        try:
            raise ScraperRateLimitError()
        except FootballPredictionError:
            pass  # Expected
        else:
            pytest.fail("ScraperRateLimitError not caught by FootballPredictionError")
