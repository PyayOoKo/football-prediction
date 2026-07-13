"""
Abstract base class for all scrapers.

Defines a consistent interface for data collection modules:
each scraper implements ``fetch()`` and returns a ``ScrapeResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ScraperStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  # some data fetched, some failed
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScrapeResult:
    """Standardised result from any scraper.

    Attributes
    ----------
    status : ScraperStatus
        Outcome of the scrape operation.
    data : list[dict]
        Collected data rows as a list of dicts.
    source : str
        Name of the data source.
    records_fetched : int
        Number of records successfully fetched.
    errors : list[str]
        Any errors encountered during scraping.
    started_at : datetime
        When the scrape started.
    completed_at : datetime
        When the scrape finished.
    """

    status: ScraperStatus = ScraperStatus.FAILED
    data: list[dict] = field(default_factory=list)
    source: str = ""
    records_fetched: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


class BaseScraper(ABC):
    """Abstract base scraper.

    Subclasses must implement ``fetch()`` and may optionally
    override ``validate()`` and ``clean()``.

    Parameters
    ----------
    name : str
        Human-readable name for this scraper.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def fetch(self, **kwargs: object) -> ScrapeResult:
        """Fetch data from the source.

        Parameters
        ----------
        **kwargs
            Scraper-specific parameters (URLs, seasons, leagues, etc.).

        Returns
        -------
        ScrapeResult
            Standardised result object with data and metadata.
        """
        ...

    def validate(self, result: ScrapeResult) -> ScrapeResult:
        """Post-fetch validation hook.

        Override in subclasses to add data quality checks.
        """
        return result

    def clean(self, result: ScrapeResult) -> ScrapeResult:
        """Post-fetch cleaning hook.

        Override in subclasses to normalise or filter data.
        """
        return result

    def run(self, **kwargs: object) -> ScrapeResult:
        """Convenience method: fetch → clean → validate."""
        result = self.fetch(**kwargs)
        result = self.clean(result)
        result = self.validate(result)
        return result
