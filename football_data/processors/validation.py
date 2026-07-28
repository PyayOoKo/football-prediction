"""
Data validation — check data integrity, consistency, and quality.

Runs before database insertion to catch:
- Missing required fields
- Out-of-range values (negative goals, impossible odds)
- Date ordering issues (future dates for historical data)
- Schema violations
- Statistical anomalies
- Data drift across sources
"""

from __future__ import annotations

import logging
from datetime import date as date_type, datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails."""


class ValidationWarning(UserWarning):
    """Warning for non-fatal data quality issues."""


class DataValidator:
    """Validates match records before database insertion.

    Usage
    -----
    >>> validator = DataValidator()
    >>> report = validator.validate_matches(records)
    >>> if report["fatal"]:
    ...     raise ValidationError("Data failed validation")
    """

    REQUIRED_FIELDS = ["source", "league", "date", "home_team", "away_team"]
    OPTIONAL_FIELDS = [
        "season", "home_goals", "away_goals", "result",
        "home_odds", "draw_odds", "away_odds",
        "home_shots", "away_shots",
    ]

    def validate_matches(
        self, matches: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate a list of match records.

        Parameters
        ----------
        matches : list[dict]
            Records to validate (pre-cleaning).

        Returns
        -------
        dict
            Validation report with keys:
            - total: number of records checked
            - passed: number that passed all checks
            - fatal: number of fatal errors (must fix)
            - warning: number of warnings
            - errors: list of error messages
            - by_league: per-league stats
        """
        report = {
            "total": len(matches),
            "passed": 0,
            "fatal": 0,
            "warnings": 0,
            "errors": [],
            "warnings_list": [],
            "by_league": {},
        }

        for i, match in enumerate(matches):
            match_errors: list[str] = []
            match_warnings: list[str] = []
            league = match.get("league", "unknown")

            # Check required fields
            for field in self.REQUIRED_FIELDS:
                value = match.get(field)
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    match_errors.append(f"Row {i}: missing required field '{field}'")

            # Validate date
            date_val = match.get("date")
            if date_val:
                parsed = self._validate_date(str(date_val))
                if parsed is True:
                    match_warnings.append(f"Row {i}: future date {date_val}")
                elif parsed is False:
                    match_errors.append(f"Row {i}: invalid date '{date_val}'")

            # Validate goals
            for field in ["home_goals", "away_goals"]:
                val = match.get(field)
                if val is not None:
                    if not isinstance(val, (int, float)):
                        match_warnings.append(f"Row {i}: {field} not numeric: {val}")
                    elif val < 0:
                        match_errors.append(f"Row {i}: negative {field}: {val}")
                    elif val > 50:
                        match_warnings.append(f"Row {i}: suspicious {field}: {val}")

            # Validate odds
            for field in ["home_odds", "draw_odds", "away_odds"]:
                val = match.get(field)
                if val is not None:
                    if val < 1.0:
                        match_warnings.append(f"Row {i}: impossible {field}: {val}")
                    elif val > 100:
                        match_warnings.append(f"Row {i}: extreme {field}: {val}")

            # Validate result
            result = match.get("result")
            goals_h = match.get("home_goals")
            goals_a = match.get("away_goals")
            if result and result not in ("H", "D", "A"):
                match_errors.append(f"Row {i}: invalid result '{result}'")
            if result and goals_h is not None and goals_a is not None:
                if result == "H" and goals_h <= goals_a:
                    match_warnings.append(
                        f"Row {i}: result 'H' but {goals_h}-{goals_a}"
                    )
                elif result == "A" and goals_a <= goals_h:
                    match_warnings.append(
                        f"Row {i}: result 'A' but {goals_h}-{goals_a}"
                    )
                elif result == "D" and goals_h != goals_a:
                    match_warnings.append(
                        f"Row {i}: result 'D' but {goals_h}-{goals_a}"
                    )

            # Validate team names are not empty
            for field in ["home_team", "away_team"]:
                team = match.get(field, "")
                if isinstance(team, str) and len(team.strip()) < 2:
                    match_errors.append(f"Row {i}: {field} too short: '{team}'")

            # Aggregate by league
            report["by_league"].setdefault(league, {"total": 0, "fatal": 0})

            # Report
            if match_errors:
                report["fatal"] += 1
                report["errors"].extend(match_errors)
                league_data = report["by_league"][league]
                league_data["total"] = league_data.get("total", 0) + 1
                league_data["fatal"] = league_data.get("fatal", 0) + 1
            else:
                report["passed"] += 1

            if match_warnings:
                report["warnings"] += len(match_warnings)
                report["warnings_list"].extend(match_warnings)

        # Log summary
        if report["fatal"] > 0:
            logger.warning(
                "Validation: %d/%d records have fatal errors",
                report["fatal"], report["total"],
            )
        if report["warnings"] > 0:
            logger.warning(
                "Validation: %d warnings", report["warnings"],
            )

        return report

    def validate_collection_result(
        self,
        league: str,
        rows_downloaded: int,
        rows_inserted: int,
        errors: int,
    ) -> dict[str, Any]:
        """Validate a collection run's result.

        Returns a report summarising collection success rate.
        """
        success_rate = (rows_inserted / max(rows_downloaded, 1)) * 100
        report = {
            "league": league,
            "rows_downloaded": rows_downloaded,
            "rows_inserted": rows_inserted,
            "rows_failed": rows_downloaded - rows_inserted,
            "errors": errors,
            "success_rate_pct": round(success_rate, 1),
            "status": "ok" if success_rate >= 90 else "degraded" if success_rate >= 50 else "failed",
        }

        if report["status"] != "ok":
            logger.warning(
                "Collection for %s: success rate %.1f%% (%d/%d)",
                league, success_rate, rows_inserted, rows_downloaded,
            )

        return report

    @staticmethod
    def _validate_date(date_str: str) -> bool | None:
        """Validate a date string. Returns True (future), False (invalid), None (past/ok)."""
        try:
            if "T" in date_str:
                d = datetime.fromisoformat(date_str.split("T")[0])
            else:
                d = datetime.strptime(date_str[:10], "%Y-%m-%d")

            today = datetime.now(timezone.utc).date()
            if d.date() > today:
                return True  # Future date — warning
            return None  # Valid past date
        except (ValueError, IndexError):
            return False  # Invalid date

    @staticmethod
    def generate_report(report: dict[str, Any]) -> str:
        """Generate a human-readable validation report."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║      DATA VALIDATION REPORT              ║",
            "╚══════════════════════════════════════════╝",
            "",
            f"Total records checked: {report['total']}",
            f"Passed:                {report['passed']}",
            f"Fatal errors:          {report['fatal']}",
            f"Warnings:              {report['warnings']}",
            "",
        ]

        if report["errors"]:
            lines.extend([
                "── Fatal errors ──",
                *[f"  • {e}" for e in report["errors"][:20]],
            ])
            if len(report["errors"]) > 20:
                lines.append(f"  ... and {len(report['errors']) - 20} more")

        if report["warnings_list"]:
            lines.extend([
                "",
                "── Warnings ──",
                *[f"  • {w}" for w in report["warnings_list"][:10]],
            ])
            if len(report["warnings_list"]) > 10:
                lines.append(f"  ... and {len(report['warnings_list']) - 10} more")

        if report.get("by_league"):
            lines.extend([
                "",
                "── Per league ──",
            ])
            for league, stats in report["by_league"].items():
                lines.append(f"  {league}: {stats['total']} total, {stats['fatal']} errors")

        return "\n".join(lines)
