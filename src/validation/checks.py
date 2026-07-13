"""
Validation checks — 9 domain-specific rules for football data.

Each check is a standalone function that takes ``(data, **kwargs)``
and returns a ``CheckResult``. Checks are stateless and reusable.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from src.validation.models import CheckResult, Severity

# ── Known league names (can be extended via kwargs) ───
# Pass ``known_leagues=["My League", ...]`` to ``check_incorrect_leagues``
# or ``ValidationEngine.run()``.
_KNOWN_LEAGUES: set[str] = {
    # England
    "E0", "E1", "E2", "E3", "EC", "FA Cup", "EFL Cup",
    "Premier League", "Championship", "League One", "League Two",
    "National League",
    # Scotland
    "SC0", "SC1", "SC2", "SC3",
    "Scottish Premiership", "Scottish Championship",
    # Germany
    "D1", "D2",
    "Bundesliga", "2. Bundesliga", "3. Liga",
    # Spain
    "SP1", "SP2",
    "La Liga", "La Liga 2", "Segunda Division",
    # Italy
    "I1", "I2",
    "Serie A", "Serie B", "Serie C",
    # France
    "F1", "F2",
    "Ligue 1", "Ligue 2",
    # Netherlands
    "N1",
    "Eredivisie", "Eerste Divisie",
    # Portugal
    "P1",
    "Primeira Liga", "Liga Portugal",
    # Belgium
    "B1",
    "Pro League", "Jupiler League",
    # Turkey
    "T1",
    "Super Lig",
    # International
    "World Cup", "FIFA World Cup", "European Championship",
    "Euros", "Copa America", "Africa Cup of Nations",
    "Champions League", "UEFA Champions League",
    "Europa League", "UEFA Europa League",
    "Conference League", "UEFA Conference League",
}

# ── Common date format patterns for validation ────────
_DATE_PATTERNS = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y"),
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "%d-%m-%Y"),
    (re.compile(r"^\d{4}\d{2}\d{2}$"), "%Y%m%d"),
]

# ── Odds column keywords for auto-detection ───────────
_ODDS_KEYWORDS = ["odds", "bbav", "b365", "psh", "psd", "psa",
                  "home_odds", "draw_odds", "away_odds",
                  "maxh", "maxd", "maxa", "avh", "avd", "ava"]


def _parse_date_flexible(value: Any) -> date | None:
    """Try to parse a date from various formats."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()

    s = str(value).strip()
    for pattern, fmt in _DATE_PATTERNS:
        if pattern.match(s):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
    return None


def _column_matches_odds(col: str) -> bool:
    """Check if a column name looks like it contains odds data."""
    col_lower = col.lower().strip()
    return any(kw in col_lower for kw in _ODDS_KEYWORDS)


# ── Check 1: Duplicate matches ────────────────────────

def check_duplicate_matches(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Detect duplicate matches (same teams, same date)."""
    seen: dict[tuple[str, str, str], list[int]] = {}
    violations: list[dict[str, Any]] = []

    for i, row in enumerate(data):
        home = str(row.get("home_team", row.get("home", "")) or "")
        away = str(row.get("away_team", row.get("away", "")) or "")
        dt = str(row.get("date", row.get("match_date", "")) or "")

        key = (home.lower().strip(), away.lower().strip(), str(dt).strip())
        if key in seen:
            violations.append({
                "row_index": i,
                "field": "match",
                "value": f"{home} vs {away} on {dt}",
                "message": f"Duplicate match: first seen at row {seen[key][0]}",
            })
            seen[key].append(i)
        else:
            seen[key] = [i]

    return CheckResult(
        check_name="Duplicate Matches",
        description="Same teams playing on the same date listed more than once",
        severity=Severity.ERROR,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )


# ── Check 2: Invalid dates ────────────────────────────

def check_invalid_dates(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Check for missing, non-existent, or future dates."""
    date_col = kwargs.get("date_column", "date")
    violations: list[dict[str, Any]] = []
    today = date.today()

    for i, row in enumerate(data):
        raw = row.get(date_col)

        # Null date
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            violations.append({
                "row_index": i,
                "field": date_col,
                "value": str(raw),
                "message": "Date is null or empty",
            })
            continue

        # Try to parse
        parsed = _parse_date_flexible(raw)
        if parsed is None:
            violations.append({
                "row_index": i,
                "field": date_col,
                "value": str(raw),
                "message": f"Unrecognised date format: {raw!r}",
            })
            continue

        # Future date more than 3 years ahead
        if parsed > date(today.year + 3, 1, 1):
            violations.append({
                "row_index": i,
                "field": date_col,
                "value": str(raw),
                "message": f"Date is more than 3 years in the future: {parsed}",
            })

    return CheckResult(
        check_name="Invalid Dates",
        description="Missing, malformed, or suspicious dates",
        severity=Severity.ERROR,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )


# ── Check 3: Invalid odds ─────────────────────────────

def check_invalid_odds(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Check all odds-like columns for invalid values.

    Auto-detects columns with names containing odds keywords
    (``BbAvH``, ``B365D``, ``odds_home``, etc.). Validates
    that each value is numeric and > 1.0.

    Parameters
    ----------
    data : list[dict]
        Match data.
    **kwargs
        ``odds_columns`` — explicit list of odds column names to check
        (auto-detected if not provided).
    """
    explicit_columns: list[str] | None = kwargs.get("odds_columns")
    violations: list[dict[str, Any]] = []

    # Pre-compute which columns are odds columns (once, not per row)
    odds_cols: list[str] = []
    if data:
        if explicit_columns:
            odds_cols = [c for c in explicit_columns if c in data[0]]
        else:
            odds_cols = [c for c in data[0] if _column_matches_odds(c)]

    # Check each odds value
    for i, row in enumerate(data):
        for col in odds_cols:
            val = row.get(col)
            if val is None:
                continue
            try:
                float_val = float(val)
                if float_val <= 1.0:
                    violations.append({
                        "row_index": i,
                        "field": col,
                        "value": str(val),
                        "message": f"Odds must be > 1.0, got {float_val}",
                    })
            except (ValueError, TypeError):
                violations.append({
                    "row_index": i,
                    "field": col,
                    "value": str(val),
                    "message": f"Non-numeric odds value: {val!r}",
                })

    return CheckResult(
        check_name="Invalid Odds",
        description="Odds ≤ 1.0, non-numeric odds, or missing odds for finished matches",
        severity=Severity.WARNING,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )


# ── Check 4: Missing goals ───────────────────────────

def check_missing_goals(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Detect finished matches without goal data."""
    home_goal_col = kwargs.get("home_goal_column", "home_goals")
    away_goal_col = kwargs.get("away_goal_column", "away_goals")
    status_col = kwargs.get("status_column", "status")
    result_col = kwargs.get("result_column", "result")

    violations: list[dict[str, Any]] = []

    for i, row in enumerate(data):
        # Determine if match is finished
        status = str(row.get(status_col, "unknown")).lower()
        result = str(row.get(result_col, "") or "")
        is_finished = (
            status in ("finished", "completed", "played")
            or result in ("H", "D", "A")
        )

        if not is_finished:
            continue

        home = row.get(home_goal_col)
        away = row.get(away_goal_col)

        home_missing = home is None or (isinstance(home, str) and home.strip() == "")
        away_missing = away is None or (isinstance(away, str) and away.strip() == "")

        if home_missing or away_missing:
            violations.append({
                "row_index": i,
                "field": f"{home_goal_col}/{away_goal_col}",
                "value": f"home={home!r}, away={away!r}",
                "message": f"Finished match has missing goals (status={status}, result={result})",
            })

    return CheckResult(
        check_name="Missing Goals",
        description="Finished matches without goal data",
        severity=Severity.ERROR,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )


# ── Check 5: Missing teams ───────────────────────────

def check_missing_teams(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Detect null or empty team names."""
    home_col = kwargs.get("home_team_column", "home_team")
    away_col = kwargs.get("away_team_column", "away_team")

    violations: list[dict[str, Any]] = []

    for i, row in enumerate(data):
        home = row.get(home_col)
        away = row.get(away_col)

        if home is None or (isinstance(home, str) and home.strip() == ""):
            violations.append({
                "row_index": i,
                "field": home_col,
                "value": str(home),
                "message": "Home team name is null or empty",
            })
        if away is None or (isinstance(away, str) and away.strip() == ""):
            violations.append({
                "row_index": i,
                "field": away_col,
                "value": str(away),
                "message": "Away team name is null or empty",
            })
        if (home is not None and away is not None
                and str(home).strip().lower() == str(away).strip().lower()):
            violations.append({
                "row_index": i,
                "field": f"{home_col}/{away_col}",
                "value": f"home={home}, away={away}",
                "message": "Home and away team names are identical",
            })

    return CheckResult(
        check_name="Missing Teams",
        description="Null, empty, or identical team names",
        severity=Severity.ERROR,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )


# ── Check 6: Incorrect league names ───────────────────

def check_incorrect_leagues(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Flag unrecognised competition names.

    Extend the known-league list by passing
    ``known_leagues=["My League", ...]`` as a keyword argument.
    """
    league_col = kwargs.get("league_column", "league")
    known_leagues: set[str] = set(kwargs.get("known_leagues", _KNOWN_LEAGUES))

    violations: list[dict[str, Any]] = []

    for i, row in enumerate(data):
        league = row.get(league_col)
        if league is None:
            violations.append({
                "row_index": i,
                "field": league_col,
                "value": "None",
                "message": "League column is null",
            })
            continue

        league_str = str(league).strip()
        if not league_str:
            violations.append({
                "row_index": i,
                "field": league_col,
                "value": "",
                "message": "League name is empty",
            })
            continue

        # Check against known list (case-insensitive)
        if league_str not in known_leagues and league_str.upper() not in known_leagues:
            violations.append({
                "row_index": i,
                "field": league_col,
                "value": league_str,
                "message": f"Unrecognised league name: {league_str!r}",
            })

    return CheckResult(
        check_name="Incorrect Leagues",
        description="Unrecognised or invalid competition names",
        severity=Severity.WARNING,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )


# ── Check 7: Invalid statistics ──────────────────────

def check_invalid_statistics(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Detect impossible or suspicious match statistics.

    Checks:
    - Possession must be 0-100
    - Shots, corners, cards must be non-negative
    - Attendance must be non-negative and reasonable
    """
    violations: list[dict[str, Any]] = []

    field_rules: list[tuple[str, str, Any, Any, str]] = [
        ("possession", "Possession", 0, 100, "Possession must be 0-100"),
        ("shots", "Shots", 0, None, "Shots cannot be negative"),
        ("shots_on", "Shots on target", 0, None, "Shots on target cannot be negative"),
        ("corners", "Corners", 0, None, "Corners cannot be negative"),
        ("fouls", "Fouls", 0, None, "Fouls cannot be negative"),
        ("yellow", "Yellow cards", 0, None, "Yellow cards cannot be negative"),
        ("red", "Red cards", 0, None, "Red cards cannot be negative"),
        ("offsides", "Offsides", 0, None, "Offsides cannot be negative"),
        ("attendance", "Attendance", 0, 200000, "Attendance seems unrealistic"),
    ]

    for i, row in enumerate(data):
        for col in list(row.keys()):
            col_lower = col.lower()
            val = row[col]

            for pattern, display, lo, hi, message in field_rules:
                if pattern not in col_lower:
                    continue
                if val is None:
                    continue

                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    continue

                if num_val < lo:
                    violations.append({
                        "row_index": i,
                        "field": col,
                        "value": str(val),
                        "message": f"{display} is negative: {num_val}",
                    })
                elif hi is not None and num_val > hi:
                    violations.append({
                        "row_index": i,
                        "field": col,
                        "value": str(val),
                        "message": f"{message}: {num_val}",
                    })

    return CheckResult(
        check_name="Invalid Statistics",
        description="Impossible or suspicious match statistics",
        severity=Severity.WARNING,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )


# ── Check 8: Duplicate IDs ───────────────────────────

def check_duplicate_ids(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Detect non-unique row/match identifiers."""
    id_col = kwargs.get("id_column", "id")
    violations: list[dict[str, Any]] = []

    seen: dict[str, list[int]] = {}

    for i, row in enumerate(data):
        id_val = row.get(id_col)
        if id_val is None:
            continue

        key = str(id_val).strip()
        if key in seen:
            violations.append({
                "row_index": i,
                "field": id_col,
                "value": key,
                "message": f"Duplicate ID: first seen at row {seen[key][0]}",
            })
            seen[key].append(i)
        else:
            seen[key] = [i]

    return CheckResult(
        check_name="Duplicate IDs",
        description="Non-unique row or match identifiers",
        severity=Severity.ERROR,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )


# ── Check 9: Impossible scores ────────────────────────

def check_impossible_scores(
    data: list[dict[str, Any]],
    **kwargs: Any,
) -> CheckResult:
    """Detect negative or excessively high scores.

    Also checks that result (H/D/A) is consistent with the score.
    """
    home_goal_col = kwargs.get("home_goal_column", "home_goals")
    away_goal_col = kwargs.get("away_goal_column", "away_goals")
    result_col = kwargs.get("result_column", "result")
    max_score = kwargs.get("max_score", 20)

    violations: list[dict[str, Any]] = []

    for i, row in enumerate(data):
        home = row.get(home_goal_col)
        away = row.get(away_goal_col)
        result = row.get(result_col)

        if home is None and away is None:
            continue  # Upcoming match

        # Check negative goals
        if home is not None:
            try:
                h = int(home)
                if h < 0:
                    violations.append({
                        "row_index": i,
                        "field": home_goal_col,
                        "value": str(home),
                        "message": f"Negative home goals: {h}",
                    })
                elif h > max_score:
                    violations.append({
                        "row_index": i,
                        "field": home_goal_col,
                        "value": str(home),
                        "message": f"Home goals exceeds max ({max_score}): {h}",
                    })
            except (ValueError, TypeError):
                violations.append({
                    "row_index": i,
                    "field": home_goal_col,
                    "value": str(home),
                    "message": f"Non-integer home goals: {home!r}",
                })

        if away is not None:
            try:
                a = int(away)
                if a < 0:
                    violations.append({
                        "row_index": i,
                        "field": away_goal_col,
                        "value": str(away),
                        "message": f"Negative away goals: {a}",
                    })
                elif a > max_score:
                    violations.append({
                        "row_index": i,
                        "field": away_goal_col,
                        "value": str(away),
                        "message": f"Away goals exceeds max ({max_score}): {a}",
                    })
            except (ValueError, TypeError):
                violations.append({
                    "row_index": i,
                    "field": away_goal_col,
                    "value": str(away),
                    "message": f"Non-integer away goals: {away!r}",
                })

        # Check result consistency
        if home is not None and away is not None:
            try:
                h_val = int(home)
                a_val = int(away)
                expected_result = "H" if h_val > a_val else "A" if a_val > h_val else "D"
                if result and result not in ("", None) and str(result).upper() != expected_result:
                    violations.append({
                        "row_index": i,
                        "field": result_col,
                        "value": f"score={home}-{away}, result={result}",
                        "message": (
                            f"Score {home}-{away} implies '{expected_result}' "
                            f"but result column says '{result}'"
                        ),
                    })
            except (ValueError, TypeError):
                pass  # Already caught non-integer goals above

    return CheckResult(
        check_name="Impossible Scores",
        description="Negative, excessively high, or result-inconsistent scores",
        severity=Severity.ERROR,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )
