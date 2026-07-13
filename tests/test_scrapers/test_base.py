"""
Tests for the scraper base framework — BaseScraper, ScrapeResult, ScraperStatus.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.scrapers.base import BaseScraper, ScrapeResult, ScraperStatus


class TestScraperStatus:
    def test_values(self) -> None:
        assert ScraperStatus.SUCCESS.value == "success"
        assert ScraperStatus.PARTIAL.value == "partial"
        assert ScraperStatus.FAILED.value == "failed"
        assert ScraperStatus.SKIPPED.value == "skipped"


class TestScrapeResult:
    def test_default_failed(self) -> None:
        result = ScrapeResult()
        assert result.status == ScraperStatus.FAILED
        assert result.data == []
        assert result.records_fetched == 0

    def test_custom_values(self) -> None:
        result = ScrapeResult(
            status=ScraperStatus.SUCCESS,
            data=[{"id": 1}],
            source="test_source",
            records_fetched=1,
        )
        assert result.status == ScraperStatus.SUCCESS
        assert result.records_fetched == 1
        assert result.source == "test_source"

    def test_duration_with_timestamps(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 10)
        result = ScrapeResult(started_at=start, completed_at=end)
        assert result.duration_seconds == 10.0

    def test_duration_without_timestamps(self) -> None:
        result = ScrapeResult()
        assert result.duration_seconds == 0.0

    def test_partial_status(self) -> None:
        result = ScrapeResult(
            status=ScraperStatus.PARTIAL,
            records_fetched=50,
            errors=["Some matches failed"],
        )
        assert result.status == ScraperStatus.PARTIAL
        assert len(result.errors) == 1

    def test_errors_list(self) -> None:
        result = ScrapeResult(errors=["Error 1", "Error 2"])
        assert len(result.errors) == 2


class MockScraper(BaseScraper):
    """A concrete scraper for testing BaseScraper functionality."""

    def __init__(self, data=None, fail=False, name=""):
        super().__init__(name=name)
        self._data = data or [{"id": 1}]
        self._fail = fail

    def fetch(self, **kwargs):
        if self._fail:
            return ScrapeResult(
                status=ScraperStatus.FAILED,
                source=self.name,
                errors=["Fetch failed"],
            )
        return ScrapeResult(
            status=ScraperStatus.SUCCESS,
            data=list(self._data),
            source=self.name,
            records_fetched=len(self._data),
        )


class TestBaseScraper:
    def test_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseScraper()  # type: ignore[abstract]

    def test_name_defaults_to_classname(self) -> None:
        scraper = MockScraper()
        assert scraper.name == "MockScraper"

    def test_custom_name(self) -> None:
        scraper = MockScraper(name="MyCustomScraper")
        assert scraper.name == "MyCustomScraper"

    def test_fetch_returns_result(self) -> None:
        scraper = MockScraper(data=[{"a": 1}, {"b": 2}])
        result = scraper.fetch()
        assert result.status == ScraperStatus.SUCCESS
        assert result.records_fetched == 2

    def test_fetch_failure(self) -> None:
        scraper = MockScraper(fail=True)
        result = scraper.fetch()
        assert result.status == ScraperStatus.FAILED
        assert "Fetch failed" in result.errors

    def test_run_calls_fetch_clean_validate(self) -> None:
        """run() should call fetch then clean then validate."""
        scraper = MockScraper(data=[{"team": "Arsenal"}])

        with pytest.MonkeyPatch.context() as mp:
            # Track call order
            calls = []

            original_clean = scraper.clean
            def tracking_clean(result):
                calls.append("clean")
                return original_clean(result)

            scraper.clean = tracking_clean

            result = scraper.run()
            assert result.status == ScraperStatus.SUCCESS

    def test_clean_default_passthrough(self) -> None:
        scraper = MockScraper(data=[{"x": 1}])
        result = scraper.run()
        assert result.data == [{"x": 1}]

    def test_validate_default_passthrough(self) -> None:
        scraper = MockScraper(data=[{"x": 1}])
        result = scraper.run()
        assert result is not None

    def test_source_set_in_result(self) -> None:
        scraper = MockScraper(name="TestScraper")
        result = scraper.fetch()
        assert result.source == "TestScraper"

    def test_kwargs_passed_to_fetch(self) -> None:
        """fetch() receives arbitrary kwargs."""

        class KwargScraper(BaseScraper):
            def fetch(self, **kwargs):
                return ScrapeResult(
                    status=ScraperStatus.SUCCESS,
                    data=[{"season": kwargs.get("season")}],
                    records_fetched=1,
                )

        scraper = KwargScraper()
        result = scraper.fetch(season="2024")
        assert result.data[0]["season"] == "2024"

    def test_fetch_with_url_param(self) -> None:
        class URLScraper(BaseScraper):
            def fetch(self, **kwargs):
                url = kwargs.get("url", "")
                return ScrapeResult(
                    status=ScraperStatus.SUCCESS,
                    data=[{"url": url}],
                    records_fetched=1,
                )

        scraper = URLScraper()
        result = scraper.fetch(url="https://example.com")
        assert result.data[0]["url"] == "https://example.com"

    def test_validate_override(self) -> None:
        """Subclass can override validate to add checks."""

        class ValidatingScraper(BaseScraper):
            def fetch(self, **kwargs):
                return ScrapeResult(
                    status=ScraperStatus.SUCCESS,
                    data=[{"score": 100}],
                    records_fetched=1,
                )

            def validate(self, result):
                if result.data and result.data[0].get("score", 0) > 50:
                    result.errors.append("Score too high")
                    result.status = ScraperStatus.PARTIAL
                return result

        scraper = ValidatingScraper()
        result = scraper.run()
        assert result.status == ScraperStatus.PARTIAL
        assert "Score too high" in result.errors
