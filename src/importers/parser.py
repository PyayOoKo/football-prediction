"""
CSVParser — parses football-data.co.uk CSV files into standardised dicts.

Handles two CSV formats:
- **Historical** — ``mmz4281/{season}/{league}.csv`` (completed seasons)
- **Current** — ``new/{league}.csv`` (in-progress season, fewer columns)

Column mapping is version-aware: older seasons (pre-2000) have fewer
columns than modern ones.

Pipeline
--------
1. Read raw CSV text → pandas DataFrame
2. Detect CSV version/format
3. Rename columns to snake_case standard
4. Validate required columns exist
5. Coerce types (date, int, float)
6. Validate row-level constraints
7. Yield clean dicts for downstream stages
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from io import StringIO
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ── Column mapping: football-data.co.uk → standardised names ──

_COLUMN_MAP: dict[str, str] = {
    # Identity
    "div": "league",
    "season": "season",
    "date": "match_date",
    # Teams
    "hometeam": "home_team",
    "awayteam": "away_team",
    # Full-time result
    "fthg": "home_goals",
    "ftag": "away_goals",
    "ftr": "result",  # H / D / A
    # Half-time result
    "hthg": "home_goals_ht",
    "htag": "away_goals_ht",
    "htr": "result_ht",
    # Referee & attendance
    "referee": "referee_name",
    "attendance": "attendance",
    # Match statistics
    "hs": "home_shots",
    "as": "away_shots",
    "hst": "home_shots_target",
    "ast": "away_shots_target",
    "hsw": "home_shots_woodwork",
    "asw": "away_shots_woodwork",
    "hc": "home_corners",
    "ac": "away_corners",
    "hf": "home_fouls",
    "af": "away_fouls",
    "hy": "home_yellow",
    "ay": "away_yellow",
    "hr": "home_red",
    "ar": "away_red",
    "hof": "home_offsides",
    "aof": "away_offsides",
    # Expected goals (modern seasons)
    "hxG": "home_xg",
    "axG": "away_xg",
    "hxg": "home_xg",
    "axg": "away_xg",
    "hxga": "home_xga",
    "axga": "away_xga",
    # Booking points (some UK leagues)
    "hbp": "home_booking_points",
    "abp": "away_booking_points",
    # Season-specific columns (not standardised — kept as-is prefixed)
}

# ── Required columns per row ──────────────────────────────

_REQUIRED_COLS = [
    "home_team",
    "away_team",
    "match_date",
]

# ── Type coercion map ─────────────────────────────────────

_INT_COLS = {
    "home_goals", "away_goals",
    "home_goals_ht", "away_goals_ht",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_shots_woodwork", "away_shots_woodwork",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
    "home_offsides", "away_offsides",
    "home_booking_points", "away_booking_points",
    "attendance",
}

_FLOAT_COLS = {
    "home_xg", "away_xg",
    "home_xga", "away_xga",
}

# ── Odds column prefixes (detected dynamically) ───────────

_ODDS_PREFIXES = [
    "B365", "BS", "BW", "GB", "IW", "PS", "SB", "SJ", "SY",
    "VC", "WH", "BbAv", "BbMx",
]

_ODDS_SUFFIXES = ["H", "D", "A"]

# ── Season naming ─────────────────────────────────────────


def _four_digit_to_season_name(code: str) -> str:
    """Convert 4-digit football-data.co.uk code to a readable season name.

    Examples
    --------
    ``\"2425\"`` → ``\"2024/2025\"``
    ``\"9394\"`` → ``\"1993/1994\"``
    """
    try:
        start = int(code[:2])
        end = int(code[2:])
        prefix = 1900 if start > 80 else 2000
        start_year = prefix + start
        end_year = prefix + end
        return f"{start_year}/{end_year}"
    except (ValueError, IndexError):
        return code


# ═══════════════════════════════════════════════════════════
#  ParsedRow
# ═══════════════════════════════════════════════════════════


class ParsedRow:
    """A single parsed match row with original + standardised data.

    Attributes
    ----------
    raw : dict[str, Any]
        Original column→value mapping from the CSV.
    standardised : dict[str, Any]
        Renamed and type-coerced columns.
    errors : list[str]
        Row-level validation errors.
    warnings : list[str]
        Row-level warnings (non-critical issues).
    row_number : int
        0-based row index in the original CSV.
    """

    def __init__(self, row_number: int) -> None:
        self.raw: dict[str, Any] = {}
        self.standardised: dict[str, Any] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.row_number = row_number

    @property
    def valid(self) -> bool:
        """True if the row passed all validation checks."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Merge standardised + error info into a single dict."""
        return {
            **self.standardised,
            "_row": self.row_number,
            "_errors": "; ".join(self.errors) if self.errors else "",
            "_warnings": "; ".join(self.warnings) if self.warnings else "",
        }


# ═══════════════════════════════════════════════════════════
#  CSVParser
# ═══════════════════════════════════════════════════════════


class CSVParser:
    """Parses football-data.co.uk CSV text into validated, standardised rows.

    Parameters
    ----------
    detect_odds_cols : bool
        Auto-detect and preserve odds columns (default True).
    strict : bool
        Fail hard on missing required columns (default True).
    season_override : str, optional
        Override the season for all rows (e.g. ``\"2024/2025\"``).
    league_override : str, optional
        Override the league code for all rows (e.g. ``\"E0\"``).
    """

    def __init__(
        self,
        detect_odds_cols: bool = True,
        strict: bool = True,
        season_override: str | None = None,
        league_override: str | None = None,
    ) -> None:
        self.detect_odds_cols = detect_odds_cols
        self.strict = strict
        self.season_override = season_override
        self.league_override = league_override

    # ── Public API ─────────────────────────────────────

    def parse_raw(
        self,
        raw_text: str,
        source_file: str = "",
    ) -> list[ParsedRow]:
        """Parse raw CSV text into a list of validated ParsedRows.

        Parameters
        ----------
        raw_text : str
            Raw CSV content as a string.
        source_file : str
            Source identifier for logging (filename or URL).

        Returns
        -------
        list[ParsedRow]
            All parsed rows with errors flagged per row.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty CSV content from %s", source_file)
            return []

        # Read CSV into DataFrame
        df = pd.read_csv(
            StringIO(raw_text),
            na_values=["", "NA", "N/A", "NULL", "-"],
            keep_default_na=True,
            dtype=str,
        )

        if df.empty:
            logger.warning("CSV %s has no data rows", source_file)
            return []

        # Standardise column names
        df = self._standardise_columns(df)

        # Detect odds columns
        odds_cols = self._detect_odds_columns(df) if self.detect_odds_cols else []

        # Validate required columns
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            msg = f"Missing required columns in {source_file}: {missing}"
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)

        # Parse each row
        parsed_rows: list[ParsedRow] = []
        for idx, row in df.iterrows():
            parsed = ParsedRow(row_number=idx)

            # Store raw values
            for col in df.columns:
                val = row.get(col)
                if pd.isna(val):
                    parsed.raw[col] = None
                else:
                    parsed.raw[col] = val

            # Build standardised row
            std: dict[str, Any] = {}

            # Copy standard columns with type coercion
            for col in df.columns:
                val = row.get(col)
                if pd.isna(val):
                    std[col] = None
                elif col in _INT_COLS:
                    try:
                        std[col] = int(float(val))
                    except (ValueError, TypeError):
                        std[col] = None
                        parsed.warnings.append(f"Could not coerce {col}={val!r} to int")
                elif col in _FLOAT_COLS:
                    try:
                        std[col] = float(val)
                    except (ValueError, TypeError):
                        std[col] = None
                        parsed.warnings.append(f"Could not coerce {col}={val!r} to float")
                else:
                    std[col] = val

            # Parse date
            if "match_date" in std and std["match_date"] is not None:
                parsed_date = self._parse_date(str(std["match_date"]))
                if parsed_date:
                    std["match_date"] = parsed_date
                else:
                    parsed.errors.append(
                        f"Invalid date: {std['match_date']!r}"
                    )
                    std["match_date"] = None

            # Validate team names
            for team_col in ("home_team", "away_team"):
                val = std.get(team_col)
                if not val or str(val).strip() in ("", "None"):
                    parsed.errors.append(f"Missing {team_col}")
                    std[team_col] = ""
                else:
                    std[team_col] = str(val).strip()

            # Validate result if present
            result = std.get("result")
            if result is not None and str(result).strip():
                r = str(result).strip().upper()
                if r in ("H", "D", "A"):
                    std["result"] = r
                else:
                    parsed.warnings.append(f"Unrecognised result value: {result!r}")
                    std["result"] = None

            # Validate goals (can't be negative)
            for gcol in ("home_goals", "away_goals"):
                if std.get(gcol) is not None:
                    try:
                        g = int(std[gcol])
                        if g < 0:
                            parsed.warnings.append(f"Negative {gcol}: {g}")
                            std[gcol] = None
                        else:
                            std[gcol] = g
                    except (ValueError, TypeError):
                        pass

            # Override season/league if configured
            if self.season_override and "season" not in std:
                std["season"] = self.season_override
            if self.league_override and "league" not in std:
                std["league"] = self.league_override

            # Preserve odds columns
            for col in odds_cols:
                if col in std and std[col] is not None:
                    try:
                        std[col] = float(std[col])
                    except (ValueError, TypeError):
                        std[col] = None

            # Add metadata
            std["_source_file"] = source_file
            std["_parsed_at"] = datetime.utcnow().isoformat()

            parsed.standardised = std
            parsed_rows.append(parsed)

        valid_count = sum(1 for p in parsed_rows if p.valid)
        logger.info(
            "Parsed %d rows from %s (%d valid, %d with errors)",
            len(parsed_rows),
            source_file,
            valid_count,
            len(parsed_rows) - valid_count,
        )
        return parsed_rows

    def parse_to_dicts(
        self,
        raw_text: str,
        source_file: str = "",
    ) -> list[dict[str, Any]]:
        """Parse raw CSV text into a list of clean dicts (only valid rows).

        Parameters
        ----------
        raw_text : str
            Raw CSV content.
        source_file : str
            Source identifier.

        Returns
        -------
        list[dict[str, Any]]
            Only rows that passed validation.
        """
        rows = self.parse_raw(raw_text, source_file)
        return [r.to_dict() for r in rows if r.valid]

    # ── Column standardisation ─────────────────────────

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to the standardised snake_case schema.

        Preserves unknown columns (e.g. odds columns) with their
        original names lowercased.
        """
        col_map: dict[str, str] = {}
        for raw_col in df.columns:
            key = raw_col.strip().lower()
            if key in _COLUMN_MAP:
                col_map[raw_col] = _COLUMN_MAP[key]
            else:
                # Keep unknown columns as-is (lowercased, no spaces)
                col_map[raw_col] = key.replace(" ", "_").replace("-", "_")

        df.rename(columns=col_map, inplace=True)
        return df

    def _detect_odds_columns(self, df: pd.DataFrame) -> list[str]:
        """Detect columns that contain bookmaker odds data."""
        odds_cols: list[str] = []
        for col in df.columns:
            upper = col.upper()
            # Match patterns like B365H, BbAvD, PS_A, etc.
            for prefix in _ODDS_PREFIXES:
                if upper.startswith(prefix.upper()):
                    odds_cols.append(col)
                    break
            # Also match odds-related column names
            if col.lower() in ("maxh", "maxd", "maxa", "avgh", "avgd", "avga"):
                odds_cols.append(col)
        return sorted(set(odds_cols))

    @staticmethod
    def _parse_date(value: str) -> date | None:
        """Parse a date string in one of several common formats.

        Supports:
        - DD/MM/YY (most common in football-data.co.uk)
        - DD/MM/YYYY
        - YYYY-MM-DD
        - YYYY-MM-DD HH:MM:SS
        """
        if not value or value.strip() in ("", "None"):
            return None

        value = value.strip()

        formats = [
            "%d/%m/%y",
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d-%m-%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue

        # Last resort: pandas flexible parser
        try:
            ts = pd.to_datetime(value, dayfirst=True, errors="coerce")
            if pd.notna(ts):
                return ts.date()
        except Exception:
            pass

        return None


# ═══════════════════════════════════════════════════════════
#  Convenience: parse a raw file path
# ═══════════════════════════════════════════════════════════


def parse_csv_file(filepath: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Convenience: parse a football-data.co.uk CSV file on disk.

    Parameters
    ----------
    filepath : str
        Path to a CSV file.
    **kwargs
        Passed through to ``CSVParser.parse_to_dicts``.

    Returns
    -------
    list[dict[str, Any]]
    """
    from pathlib import Path

    path = Path(filepath)
    raw_text = path.read_text(encoding="utf-8")
    parser = CSVParser(**{k: v for k, v in kwargs.items() if k != "source_file"})
    return parser.parse_to_dicts(raw_text, source_file=str(path))
