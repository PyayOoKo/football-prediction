"""
Normalization stage — standardises values across sources.

Operations:
- Team name normalisation (Arsenal FC → Arsenal)
- Date parsing and formatting (DD/MM/YY → YYYY-MM-DD)
- Categorical encoding (label encoding, one-hot preparation)
- Case normalisation (UPPER / title / lower)
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from typing import Any, Literal

from src.etl.models import PipelineStage, StageResult, StageStatus

logger = logging.getLogger(__name__)

# ── Date parsing formats (tried in order) ──────────────
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%y",
    "%Y%m%d",
]

# ── Team name normalisation rules ─────────────────────

_TEAM_SUFFIXES = re.compile(
    r"\s+(FC|AFC|CFC|United|City|Wanderers|Rovers|Athletic|Athletique|"
    r"Hotspur|Albion|County|Town|Villa|Rangers|Celtic|Lions|Stars|"
    r"Barcelona|Madrid|Munich|Juventus|Milan|Inter|Roma|Napoli|"
    r"PSG|Olympique|Saint-)|\(.*?\)",
    re.IGNORECASE,
)

_TEAM_REPLACEMENTS: dict[str, str] = {
    "manchester united": "Manchester United",
    "man utd": "Manchester United",
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "liverpool fc": "Liverpool",
    "chelsea fc": "Chelsea",
    "arsenal fc": "Arsenal",
    "tottenham hotspur": "Tottenham",
    "tottenham": "Tottenham",
    "spurs": "Tottenham",
    "west ham": "West Ham",
    "newcastle": "Newcastle",
    "newcastle utd": "Newcastle",
    "leicester": "Leicester",
    "leicester city": "Leicester",
    "everton fc": "Everton",
    "aston villa": "Aston Villa",
    "wolverhampton": "Wolves",
    "wolves": "Wolves",
    "brighton": "Brighton",
    "southampton fc": "Southampton",
    "norwich city": "Norwich",
    "watford fc": "Watford",
    "crystal palace": "Crystal Palace",
    "burnley fc": "Burnley",
    "fulham fc": "Fulham",
    "leeds united": "Leeds",
    "brentford fc": "Brentford",
    "nottingham forest": "Nott'm Forest",
    "sheffield united": "Sheffield Utd",
    "sheffield wednesday": "Sheffield Wed",
    "cardiff city": "Cardiff",
    "swansea city": "Swansea",
    "derby county": "Derby",
    "middlesbrough fc": "Middlesbrough",
    "hull city": "Hull",
    "stoke city": "Stoke",
    "west brom": "West Brom",
    "west bromwich": "West Brom",
    "birmingham city": "Birmingham",
    "blackburn rovers": "Blackburn",
    "bolton wanderers": "Bolton",
    "ipswich town": "Ipswich",
    "millwall fc": "Millwall",
    "preston north end": "Preston",
    "qpr": "QPR",
    "queens park rangers": "QPR",
    "reading fc": "Reading",
    "sunderland fc": "Sunderland",
    "wigan athletic": "Wigan",
}


def normalize_team_name(name: str | None) -> str | None:
    """Normalise a team name to a canonical form.

    Steps:
    1. Strip whitespace
    2. Check manual replacement map
    3. Remove common suffixes (FC, United, etc.)
    4. Title case
    """
    if name is None or not name.strip():
        return name

    cleaned = name.strip()

    # Direct replacement lookup (case-insensitive)
    key = cleaned.lower()
    if key in _TEAM_REPLACEMENTS:
        return _TEAM_REPLACEMENTS[key]

    # Remove parenthesised suffixes like "(1)" or "(U23)"
    cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()

    # Remove common football suffixes
    cleaned = _TEAM_SUFFIXES.sub("", cleaned).strip()

    # Title case
    cleaned = cleaned.title()

    return cleaned if cleaned else name.strip()


def parse_date_flexible(value: Any) -> date | None:
    """Parse a date from various common formats.

    Tries formats from ``_DATE_FORMATS`` in order and returns
    the first successful parse.

    Supports:
    - ISO: 2024-01-07, 2024/01/07
    - EU: 07/01/2024, 07-01-2024
    - US: 01/07/2024
    - Datetime objects
    """
    if value is None:
        return None

    if isinstance(value, date):
        return value

    if isinstance(value, datetime):
        return value.date()

    s = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue

    logger.warning("Could not parse date: %r", value)
    return None


# ── Normalizer ─────────────────────────────────────────


class DataNormalizer:
    """Standardises team names, dates, and categorical values.

    Parameters
    ----------
    team_name_columns : list[str]
        Columns containing team names to normalise (default ``[]``).
    date_columns : list[str]
        Columns containing date strings to parse (default ``[]``).
    case_columns : dict[str, Literal["lower", "upper", "title"]]
        Per-column case transformations.
    categorical_columns : list[str]
        Columns to mark as categorical (stored as strings, not encoded).
    """

    def __init__(
        self,
        team_name_columns: list[str] | None = None,
        date_columns: list[str] | None = None,
        case_columns: dict[str, Literal["lower", "upper", "title"]] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> None:
        self.team_name_columns = team_name_columns or []
        self.date_columns = date_columns or []
        self.case_columns = case_columns or {}
        self.categorical_columns = categorical_columns or []

    def run(self, data: list[dict[str, Any]]) -> StageResult:
        """Execute normalisation.

        Parameters
        ----------
        data : list[dict]
            Cleaned data.

        Returns
        -------
        StageResult
            Normalised data with metrics.
        """
        stage = PipelineStage.NORMALIZE
        result = StageResult(stage=stage, status=StageStatus.RUNNING)
        start = time.perf_counter()

        result.records_in = len(data)
        metrics: dict[str, float] = {
            "team_names_normalized": 0.0,
            "dates_parsed": 0.0,
        }

        try:
            for row in data:
                # Team names
                for col in self.team_name_columns:
                    original = row.get(col)
                    normalized = normalize_team_name(original)
                    if normalized != original:
                        row[col] = normalized
                        metrics["team_names_normalized"] += 1

                # Dates
                for col in self.date_columns:
                    parsed = parse_date_flexible(row.get(col))
                    if parsed is not None:
                        row[col] = parsed.isoformat()
                        metrics["dates_parsed"] += 1

                # Case
                for col, case in self.case_columns.items():
                    val = row.get(col)
                    if isinstance(val, str):
                        if case == "lower":
                            row[col] = val.lower()
                        elif case == "upper":
                            row[col] = val.upper()
                        elif case == "title":
                            row[col] = val.title()

                # Categorical -> ensure strings
                for col in self.categorical_columns:
                    val = row.get(col)
                    if val is not None and not isinstance(val, str):
                        row[col] = str(val)

            result.data = data
            result.records_out = len(data)
            result.metrics = metrics
            result.status = StageStatus.SUCCESS

        except Exception as exc:
            logger.exception("Normalisation failed: %s", exc)
            result.status = StageStatus.FAILED
            result.errors.append(str(exc))

        result.duration_seconds = time.perf_counter() - start
        logger.info(
            "Normalisation: %d team names, %d dates in %.1fs",
            metrics["team_names_normalized"],
            metrics["dates_parsed"],
            result.duration_seconds,
        )
        return result
