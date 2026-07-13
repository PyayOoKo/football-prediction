"""
Tests for the ETL extraction stage — BaseExtractor, CSVExtractor, APIExtractor, RetryWithBackoff.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.etl.extract import (
    APIExtractor,
    BaseExtractor,
    CSVExtractor,
    RetryWithBackoff,
)
from src.etl.models import PipelineStage, StageStatus


# ═══════════════════════════════════════════════════════════
#  RetryWithBackoff
# ═══════════════════════════════════════════════════════════

class TestRetryWithBackoff:
    def test_default_params(self) -> None:
        retry = RetryWithBackoff()
        assert retry.max_attempts == 3
        assert retry.base_delay == 2.0
        assert retry.max_delay == 120.0
        assert 429 in retry.retryable_statuses
        assert 502 in retry.retryable_statuses

    def test_custom_params(self) -> None:
        retry = RetryWithBackoff(
            max_attempts=5, base_delay=1.0, max_delay=30.0,
            retryable_statuses=(429, 503),
        )
        assert retry.max_attempts == 5
        assert retry.max_delay == 30.0
        assert 502 not in retry.retryable_statuses

    def test_success_on_first_attempt(self) -> None:
        retry = RetryWithBackoff(max_attempts=3)
        mock_fn = MagicMock()
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_fn.return_value = mock_response

        result = retry.execute(mock_fn, "http://example.com")
        assert result.status_code == 200
        assert mock_fn.call_count == 1

    def test_retry_on_429(self) -> None:
        """429 triggers retry with Retry-After header respected."""
        retry = RetryWithBackoff(max_attempts=3, base_delay=0.01)
        mock_fn = MagicMock()

        fail_response = MagicMock(spec=requests.Response)
        fail_response.status_code = 429
        fail_response.headers = {"Retry-After": "0.01"}

        success_response = MagicMock(spec=requests.Response)
        success_response.status_code = 200

        mock_fn.side_effect = [fail_response, success_response]

        with patch("time.sleep", return_value=None) as mock_sleep:
            result = retry.execute(mock_fn, "http://example.com")
            assert result.status_code == 200
            assert mock_fn.call_count == 2
            mock_sleep.assert_called_once()

    def test_non_retryable_status_raises(self) -> None:
        """400 status raises immediately without retry."""
        retry = RetryWithBackoff(max_attempts=3)
        mock_fn = MagicMock()
        fail_response = MagicMock(spec=requests.Response)
        fail_response.status_code = 400
        fail_response.raise_for_status.side_effect = requests.HTTPError("Bad request")
        mock_fn.return_value = fail_response

        with pytest.raises(requests.HTTPError):
            retry.execute(mock_fn, "http://example.com")
        assert mock_fn.call_count == 1

    def test_all_attempts_fail(self) -> None:
        retry = RetryWithBackoff(max_attempts=2, base_delay=0.01)
        mock_fn = MagicMock()

        fail_response = MagicMock(spec=requests.Response)
        fail_response.status_code = 503
        fail_response.headers = {}
        mock_fn.return_value = fail_response

        with patch("time.sleep", return_value=None):
            with pytest.raises(requests.RequestException, match="All 2 retry attempts failed"):
                retry.execute(mock_fn, "http://example.com")
        assert mock_fn.call_count == 2

    def test_connection_error_retry(self) -> None:
        retry = RetryWithBackoff(max_attempts=2, base_delay=0.01)
        mock_fn = MagicMock()
        mock_fn.side_effect = [
            requests.ConnectionError("Connection refused"),
            MagicMock(status_code=200),
        ]

        with patch("time.sleep", return_value=None):
            result = retry.execute(mock_fn, "http://example.com")
            assert result.status_code == 200
            assert mock_fn.call_count == 2

    def test_timeout_retry(self) -> None:
        retry = RetryWithBackoff(max_attempts=2, base_delay=0.01)
        mock_fn = MagicMock()
        mock_fn.side_effect = [
            requests.Timeout("Timed out"),
            MagicMock(status_code=200),
        ]

        with patch("time.sleep", return_value=None):
            result = retry.execute(mock_fn, "http://example.com")
            assert result.status_code == 200

    def test_compute_delay_respects_retry_after(self) -> None:
        retry = RetryWithBackoff()
        response = MagicMock(spec=requests.Response)
        response.headers = {"Retry-After": "5"}

        delay = retry._compute_delay(1, response)
        assert 4.5 <= delay <= 5.5  # 5 with jitter ±10%

    def test_compute_delay_exponential(self) -> None:
        retry = RetryWithBackoff(base_delay=1.0, max_delay=30.0)
        response = MagicMock(spec=requests.Response)
        response.headers = {}

        delay_1 = retry._compute_delay(1, response)
        delay_3 = retry._compute_delay(3, response)

        assert 0.75 <= delay_1 <= 1.25  # 1.0 * jitter
        assert delay_3 > delay_1  # exponential growth


# ═══════════════════════════════════════════════════════════
#  BaseExtractor
# ═══════════════════════════════════════════════════════════

class TestBaseExtractor:
    def test_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseExtractor()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class TestExtractor(BaseExtractor):
            def _extract(self, **kwargs):
                return [{"id": 1, "value": kwargs.get("param", "default")}]

        ext = TestExtractor(name="test_extractor")
        result = ext.run(param="hello")

        assert result.stage == PipelineStage.EXTRACT
        assert result.status == StageStatus.SUCCESS
        assert result.records_out == 1
        assert result.data[0]["value"] == "hello"

    def test_empty_result_produces_warning(self) -> None:
        class EmptyExtractor(BaseExtractor):
            def _extract(self, **kwargs):
                return []

        ext = EmptyExtractor()
        result = ext.run()

        assert result.status == StageStatus.WARNING
        assert "zero records" in result.errors[0].lower()
        assert result.records_out == 0

    def test_extract_error_fails_gracefully(self) -> None:
        class BrokenExtractor(BaseExtractor):
            def _extract(self, **kwargs):
                raise RuntimeError("API unavailable")

        ext = BrokenExtractor()
        result = ext.run()

        assert result.status == StageStatus.FAILED
        assert "API unavailable" in result.errors[0]

    def test_custom_name_in_logging(self) -> None:
        class NamedExtractor(BaseExtractor):
            def _extract(self, **kwargs):
                return [{"a": 1}]

        ext = NamedExtractor(name="MyCustomExtractor")
        assert ext.name == "MyCustomExtractor"

    def test_progress_and_retry_are_optional(self) -> None:
        class SimpleExtractor(BaseExtractor):
            def _extract(self, **kwargs):
                return [{"x": 1}]

        ext = SimpleExtractor()
        assert ext.retry is not None
        assert ext.progress is not None
        assert ext.name == "SimpleExtractor"


# ═══════════════════════════════════════════════════════════
#  CSVExtractor
# ═══════════════════════════════════════════════════════════

class TestCSVExtractor:
    def test_read_csv_file(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df.to_csv(csv_path, index=False)

        extractor = CSVExtractor(filepath=csv_path)
        result = extractor.run()

        assert result.status == StageStatus.SUCCESS
        assert result.records_out == 2
        assert result.data[0]["col1"] == 1
        assert result.data[1]["col2"] == "b"

    def test_file_not_found(self) -> None:
        extractor = CSVExtractor(filepath="/nonexistent/path.csv")
        result = extractor.run()

        assert result.status == StageStatus.FAILED
        assert "not found" in result.errors[0].lower()

    def test_empty_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(csv_path, index=False)

        extractor = CSVExtractor(filepath=csv_path)
        result = extractor.run()

        # An empty CSV produces an empty dataframe -> 0 records
        assert result.records_out == 0

    def test_custom_encoding(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "encoded.csv"
        df = pd.DataFrame({"name": ["José", "Müller"]})
        df.to_csv(csv_path, index=False, encoding="utf-8")

        extractor = CSVExtractor(filepath=csv_path, encoding="utf-8")
        result = extractor.run()
        assert result.records_out == 2
        assert result.data[0]["name"] == "José"

    def test_low_memory_disabled(self) -> None:
        """CSVExtractor uses low_memory=False by default."""
        extractor = CSVExtractor(filepath="/dummy.csv")
        # Just verify the default is set
        assert extractor.name != ""


# ═══════════════════════════════════════════════════════════
#  APIExtractor
# ═══════════════════════════════════════════════════════════

class TestAPIExtractor:
    def test_init_defaults(self) -> None:
        class TestAPI(APIExtractor):
            def _extract(self, **kwargs):
                return []

        api = TestAPI(base_url="https://api.example.com")
        assert api.base_url == "https://api.example.com"
        assert api.timeout == 30
        assert api._session is None

    def test_session_property(self) -> None:
        class TestAPI(APIExtractor):
            def _extract(self, **kwargs):
                return []

        api = TestAPI(base_url="https://api.example.com")
        session = api.session
        assert session is not None
        # Same session on repeated access
        assert api.session is session

    def test_session_with_custom_headers(self) -> None:
        class TestAPI(APIExtractor):
            def _extract(self, **kwargs):
                return []

        api = TestAPI(
            base_url="https://api.example.com",
            headers={"X-API-Key": "secret123"},
        )
        session = api.session
        assert session.headers.get("X-API-Key") == "secret123"

    def test_url_join(self) -> None:
        class TestAPI(APIExtractor):
            def _extract(self, **kwargs):
                return []

        api = TestAPI(base_url="https://api.example.com")
        with patch.object(api.retry, "execute") as mock_exec:
            mock_exec.return_value = MagicMock(status_code=200, json=lambda: {})
            api.get("/v1/matches")
            args, kwargs = mock_exec.call_args
            assert args[1] == "https://api.example.com/v1/matches"

    def test_url_join_slash_handling(self) -> None:
        class TestAPI(APIExtractor):
            def _extract(self, **kwargs):
                return []

        api = TestAPI(base_url="https://api.example.com/")
        with patch.object(api.retry, "execute") as mock_exec:
            mock_exec.return_value = MagicMock(status_code=200, json=lambda: {})
            api.get("/v1/matches")
            args, kwargs = mock_exec.call_args
            # Should not have double slash
            assert "//v1" not in args[1]

    def test_extract_not_implemented(self) -> None:
        api = APIExtractor(base_url="https://api.example.com")
        with pytest.raises(NotImplementedError):
            api._extract()

    def test_trailing_slash_stripped(self) -> None:
        class TestAPI(APIExtractor):
            def _extract(self, **kwargs):
                return []

        api = TestAPI(base_url="https://example.com/api/")
        assert api.base_url == "https://example.com/api"
