"""
Extraction stage — fetches raw data from external sources.

Provides the ``BaseExtractor`` that all source-specific extractors
inherit from, plus ``RetryWithBackoff`` for resilient API calls.

Key features
------------
- Configurable retry with exponential backoff + jitter
- HTTP session management with connection pooling
- File-based extraction (CSV, Parquet, JSON)
- API extraction with headers auth token injection
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from src.etl.models import PipelineStage, StageResult, StageStatus
from src.etl.progress import ProgressReporter

logger = logging.getLogger(__name__)


class RetryWithBackoff:
    """Exponential backoff with jitter for resilient API calls.

    Usage::

        retry = RetryWithBackoff(max_attempts=5, base_delay=1.0)
        response = retry.execute(requests.get, url, headers=headers)

    Parameters
    ----------
    max_attempts : int
        Maximum retry attempts (default 3).
    base_delay : float
        Base delay in seconds (default 2.0). Actual delay doubles each attempt.
    max_delay : float
        Maximum delay cap in seconds (default 120.0).
    retryable_statuses : tuple[int, ...]
        HTTP status codes that trigger a retry (default 429, 502, 503, 504).
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 120.0,
        retryable_statuses: tuple[int, ...] = (429, 502, 503, 504),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_statuses = retryable_statuses

    def execute(
        self,
        http_method: Any,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Execute an HTTP request with retry logic.

        Parameters
        ----------
        http_method : callable
            ``requests.get``, ``requests.post``, etc.
        url : str
            Target URL.
        **kwargs
            Passed through to ``http_method``.

        Returns
        -------
        requests.Response
            The final response after successful retries.

        Raises
        ------
        requests.RequestException
            If all retry attempts are exhausted.

        Notes
        -----
        - 429 responses: extracts ``Retry-After`` header for delay.
        - 5xx responses: doubles delay exponentially.
        - All retries add jitter (±25%) to avoid thundering herd.
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = http_method(url, **kwargs)

                if response.status_code == 200:
                    return response

                if response.status_code not in self.retryable_statuses:
                    # Non-retryable error — raise immediately
                    response.raise_for_status()

                # Retryable status
                delay = self._compute_delay(attempt, response)

                logger.warning(
                    "HTTP %d on %s (attempt %d/%d). Retrying in %.1fs...",
                    response.status_code,
                    url,
                    attempt,
                    self.max_attempts,
                    delay,
                )
                time.sleep(delay)

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exception = exc
                delay = min(
                    self.base_delay * (2 ** (attempt - 1)) * random.uniform(0.75, 1.25),
                    self.max_delay,
                )
                logger.warning(
                    "Connection error on %s (attempt %d/%d): %s. Retrying in %.1fs...",
                    url,
                    attempt,
                    self.max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise requests.RequestException(
            f"All {self.max_attempts} retry attempts failed for {url}"
        ) from last_exception

    def _compute_delay(self, attempt: int, response: requests.Response) -> float:
        """Calculate delay with jitter, respecting Retry-After header."""
        # Honour Retry-After (common with 429)
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after) * random.uniform(0.9, 1.1)
            except ValueError:
                pass

        delay = self.base_delay * (2 ** (attempt - 1))
        jittered = delay * random.uniform(0.75, 1.25)
        return min(jittered, self.max_delay)


# ── Extractors ─────────────────────────────────────────

class BaseExtractor(ABC):
    """Abstract base extractor.

    Every source-specific extractor subclasses this and implements
    ``_extract()``. The public ``run()`` method handles retries,
    timing, progress reporting, and result packaging.

    Parameters
    ----------
    name : str
        Extractor name for logging.
    retry : RetryWithBackoff, optional
        Retry strategy. Created with defaults if not provided.
    progress : ProgressReporter, optional
        Progress reporting instance. Created if not provided.
    """

    def __init__(
        self,
        name: str = "",
        retry: RetryWithBackoff | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.retry = retry or RetryWithBackoff()
        self.progress = progress or ProgressReporter()

    @abstractmethod
    def _extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Source-specific extraction logic.

        Subclasses must implement this. It receives the same
        ``**kwargs`` passed to ``run()``.

        Returns
        -------
        list[dict]
            Raw data rows as a list of dicts.
        """
        ...

    def run(self, **kwargs: Any) -> StageResult:
        """Execute extraction with retries, timing, and progress.

        Parameters
        ----------
        **kwargs
            Passed to ``_extract()``.

        Returns
        -------
        StageResult
            Stage result with extracted data and metadata.
        """
        stage = PipelineStage.EXTRACT
        result = StageResult(stage=stage, status=StageStatus.RUNNING)
        start = time.perf_counter()

        try:
            logger.info("[%s] Starting extraction: %s", self.name, kwargs)
            raw_data = self._extract(**kwargs)

            result.data = raw_data
            result.records_in = len(raw_data)
            result.records_out = len(raw_data)
            result.status = StageStatus.SUCCESS if raw_data else StageStatus.WARNING

            if not raw_data:
                result.errors.append("Extraction returned zero records")

        except Exception as exc:
            logger.exception("[%s] Extraction failed: %s", self.name, exc)
            result.status = StageStatus.FAILED
            result.errors.append(str(exc))

        result.duration_seconds = time.perf_counter() - start
        logger.info(
            "[%s] Extraction done: %d records in %.1fs",
            self.name,
            result.records_out,
            result.duration_seconds,
        )
        return result


class CSVExtractor(BaseExtractor):
    """Extract data from local CSV files.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.
    encoding : str
        File encoding (default ``utf-8``).
    """

    def __init__(
        self,
        filepath: str | Path,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.filepath = Path(filepath)
        self.encoding = encoding

    def _extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Read CSV and return rows as dicts."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV not found: {self.filepath}")

        df = pd.read_csv(self.filepath, encoding=self.encoding, low_memory=False)
        return df.to_dict(orient="records")


class APIExtractor(BaseExtractor):
    """Extract data from a REST API.

    Parameters
    ----------
    base_url : str
        Base URL for the API.
    headers : dict[str, str], optional
        Default HTTP headers (e.g. API key).
    timeout : int
        Request timeout in seconds (default 30).
    """

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._session: requests.Session | None = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.headers)
        return self._session

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        """Perform a retryable GET request.

        Parameters
        ----------
        path : str
            URL path (joined with base_url).
        **kwargs
            Extra arguments for ``requests.get()``.

        Returns
        -------
        requests.Response
        """
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        return self.retry.execute(
            self.session.get,
            url,
            timeout=self.timeout,
            **kwargs,
        )

    def _extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Override in subclasses for per-API extraction logic."""
        raise NotImplementedError
