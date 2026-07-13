"""
Data loader — loads match data from CSV, database, or API sources.

Provides a unified ``load_dataframe()`` interface regardless of
the underlying source, so callers don't need to worry about
file formats or connection details.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from src.utils.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

SourceType = Literal["csv", "parquet", "database"]


class DataLoader:
    """Load match data from various sources.

    Parameters
    ----------
    data_dir : Path, optional
        Base directory for local data files.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir

    def load_dataframe(
        self,
        source: str | Path,
        source_type: SourceType = "csv",
        **kwargs: object,
    ) -> pd.DataFrame:
        """Load match data as a pandas DataFrame.

        Parameters
        ----------
        source : str | Path
            File path, table name, or URL.
        source_type : SourceType
            The type of source (``csv``, ``parquet``, ``database``).
        **kwargs
            Additional arguments passed to the underlying reader.

        Returns
        -------
        pd.DataFrame
            Loaded match data.

        Raises
        ------
        DataNotFoundError
            If the source file or table does not exist.
        """
        logger.info("Loading data from %s (type=%s)", source, source_type)
        # TODO: Implement actual loading logic
        raise NotImplementedError("DataLoader not yet implemented")

    def load_results(self, path: str | Path | None = None) -> pd.DataFrame:
        """Convenience method: load match results CSV."""
        return self.load_dataframe(
            path or self._data_dir / "results.csv",
            source_type="csv",
        )

    def load_fixtures(self, path: str | Path | None = None) -> pd.DataFrame:
        """Convenience method: load fixtures CSV."""
        return self.load_dataframe(
            path or self._data_dir / "fixtures.csv",
            source_type="csv",
        )
