"""
Data loader — loads match data from CSV, database, or API sources.

Provides a unified ``load_dataframe()`` interface regardless of
the underlying source, so callers don't need to worry about
file formats or connection details.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from config import config as _global_config
from src.utils.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

SourceType = Literal["csv", "parquet", "database"]


class DataLoader:
    """Load match data from various sources.

    Parameters
    ----------
    data_dir : Path, optional
        Base directory for local data files.  Defaults to
        ``config.paths.raw``.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        config: Any | None = None,
    ) -> None:
        self._cfg = config or _global_config
        self._data_dir = data_dir or self._cfg.paths.raw

    # ── Public API ─────────────────────────────────────────

    def load_dataframe(
        self,
        source: str | Path,
        source_type: SourceType = "csv",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load match data as a pandas DataFrame.

        Parameters
        ----------
        source : str | Path
            File path, table name, or URL.
        source_type : SourceType
            The type of source (``csv``, ``parquet``, ``database``).
        **kwargs
            Additional arguments passed to the underlying reader
            (e.g. ``encoding``, ``low_memory`` for CSV, ``columns``
            for database queries).

        Returns
        -------
        pd.DataFrame
            Loaded match data.

        Raises
        ------
        DataNotFoundError
            If the source file or table does not exist.
        ValueError
            If an unsupported source_type is provided.
        """
        logger.info("Loading data from %s (type=%s)", source, source_type)

        if source_type == "csv":
            return self._load_csv(source, **kwargs)
        elif source_type == "parquet":
            return self._load_parquet(source, **kwargs)
        elif source_type == "database":
            return self._load_database(str(source), **kwargs)
        else:
            raise ValueError(
                f"Unsupported source_type: {source_type!r}. "
                f"Expected one of: csv, parquet, database"
            )

    def load_results(self, path: str | Path | None = None) -> pd.DataFrame:
        """Convenience method: load match results CSV.

        Parameters
        ----------
        path : str | Path, optional
            Explicit path.  Falls back to ``config.paths.raw / \"results.csv\"``
            then ``config.worldcup.data_path``.
        """
        if path is None:
            default = self._data_dir / "results.csv"
            path = default if default.exists() else Path(self._cfg.worldcup.data_path)
        return self.load_dataframe(path, source_type="csv")

    def load_fixtures(self, path: str | Path | None = None) -> pd.DataFrame:
        """Convenience method: load fixtures CSV.

        Parameters
        ----------
        path : str | Path, optional
            Explicit path.  Falls back to ``config.paths.raw / \"fixtures.csv\"``.
        """
        if path is None:
            path = self._data_dir / "fixtures.csv"
        return self.load_dataframe(path, source_type="csv")

    # ── Internal: CSV loading ───────────────────────────

    @staticmethod
    def _load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load a CSV file with sensible defaults."""
        path = Path(path)
        if not path.exists():
            raise DataNotFoundError(
                f"CSV file not found: {path}. "
                "Run data collection first or provide a valid path.",
                resource=str(path),
            )

        # Default kwargs for CSV reading
        csv_kwargs: dict[str, Any] = {
            "low_memory": False,
        }
        csv_kwargs.update(kwargs)

        try:
            df = pd.read_csv(path, **csv_kwargs)
            logger.info(
                "Loaded CSV: %s — %d rows × %d cols",
                path.name,
                len(df),
                len(df.columns),
            )
            return df
        except Exception as exc:
            raise DataNotFoundError(
                f"Failed to read CSV: {path}. Error: {exc}",
                resource=str(path),
            ) from exc

    # ── Internal: Parquet loading ───────────────────────

    @staticmethod
    def _load_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load a Parquet file."""
        path = Path(path)
        if not path.exists():
            raise DataNotFoundError(
                f"Parquet file not found: {path}",
                resource=str(path),
            )

        try:
            df = pd.read_parquet(path, **kwargs)
            logger.info(
                "Loaded Parquet: %s — %d rows × %d cols",
                path.name,
                len(df),
                len(df.columns),
            )
            return df
        except Exception as exc:
            raise DataNotFoundError(
                f"Failed to read Parquet: {path}. Error: {exc}",
                resource=str(path),
            ) from exc

    # ── Internal: Database loading ──────────────────────

    _VALID_TABLE_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    def _load_database(
        self,
        table_name: str,
        connection_string: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from a database table.

        Parameters
        ----------
        table_name : str
            Name of the table (safe, validated) **or** a raw SQL query
            passed via the ``sql`` keyword argument (see below).
        connection_string : str, optional
            Database connection URI.  Falls back to config.
        **kwargs
            Passed to ``pd.read_sql``.  To pass a raw SQL query, use
            the keyword ``sql=...`` instead — this keeps table names
            and SQL queries in separate parameters.

        Returns
        -------
        pd.DataFrame
            Query results.

        Raises
        ------
        DataNotFoundError
            If the database is not configured or the query fails.
        ValueError
            If the table name contains unsafe characters or the raw SQL
            parameter is mixed with table names.
        """
        conn_str = connection_string or self._cfg.data.api_url
        if not conn_str:
            raise DataNotFoundError(
                "Database loading requires a connection string. "
                "Set config.data.api_url or pass connection_string."
            )

        # ── Is this a raw SQL query (explicitly passed via sql= kwarg)? ──
        raw_sql: str | None = kwargs.pop("sql", None)
        if raw_sql is not None:
            query = raw_sql
            logger.info("Using raw SQL query (length=%d)", len(raw_sql))
        else:
            # Validate table identifier — reject anything unsafe
            if not isinstance(table_name, str) or not table_name.strip():
                raise ValueError("Table name must be a non-empty string.")
            table_name = table_name.strip()
            if not self._VALID_TABLE_RE.match(table_name):
                raise ValueError(
                    f"Invalid table name {table_name!r}. Table names must match "
                    r"^[a-zA-Z_][a-zA-Z0-9_]*$ and must not contain SQL keywords, "
                    "semicolons, or comments."
                )
            query = f"SELECT * FROM {table_name}"
            logger.info("Loading from table %s", table_name)

        try:
            import sqlalchemy as sa

            engine = sa.create_engine(conn_str)
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, **kwargs)
                logger.info(
                    "Loaded from database: %d rows × %d cols",
                    len(df),
                    len(df.columns),
                )
                return df

        except ImportError:
            raise DataNotFoundError(
                "Database loading requires sqlalchemy. Install: pip install sqlalchemy"
            ) from None
        except Exception as exc:
            raise DataNotFoundError(
                f"Database query failed: {table_name!r}. Error: {exc}",
                resource=table_name,
            ) from exc
