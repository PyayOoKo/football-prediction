"""
ValidationEngine — orchestrates all validation checks.

Collects results from each check and produces a ``ValidationResult``.
Supports running specific checks or all 9 default checks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from src.validation.checks import (
    check_duplicate_ids,
    check_duplicate_matches,
    check_impossible_scores,
    check_incorrect_leagues,
    check_invalid_dates,
    check_invalid_odds,
    check_invalid_statistics,
    check_missing_goals,
    check_missing_teams,
)
from src.validation.models import CheckResult, Severity, ValidationResult

logger = logging.getLogger(__name__)

# Type: a check function
CheckFn = Callable[[list[dict[str, Any]]], Any]

# All 9 default checks with their kwargs
_DEFAULT_CHECKS: list[tuple[str, CheckFn, dict[str, Any]]] = [
    ("Duplicate Matches", check_duplicate_matches, {}),
    ("Invalid Dates", check_invalid_dates, {}),
    ("Invalid Odds", check_invalid_odds, {}),
    ("Missing Goals", check_missing_goals, {}),
    ("Missing Teams", check_missing_teams, {}),
    ("Incorrect Leagues", check_incorrect_leagues, {}),
    ("Invalid Statistics", check_invalid_statistics, {}),
    ("Duplicate IDs", check_duplicate_ids, {}),
    ("Impossible Scores", check_impossible_scores, {}),
]


class ValidationEngine:
    """Orchestrates validation checks and aggregates results.

    Parameters
    ----------
    checks : list[tuple[str, CheckFn, dict]], optional
        Custom check list. Each entry is ``(name, function, kwargs)``.
        Defaults to all 9 built-in checks.
    verbose : bool
        Log progress for each check (default True).
    """

    def __init__(
        self,
        checks: list[tuple[str, CheckFn, dict[str, Any]]] | None = None,
        verbose: bool = True,
    ) -> None:
        self.checks = checks or _DEFAULT_CHECKS
        self.verbose = verbose

    def run(
        self,
        data: list[dict[str, Any]],
        source_name: str = "unknown",
    ) -> ValidationResult:
        """Execute all validation checks on the dataset.

        Parameters
        ----------
        data : list[dict]
            The dataset to validate (list of row dicts).
        source_name : str
            Source name for the report (default ``unknown``).

        Returns
        -------
        ValidationResult
            Aggregated results from all checks.
        """
        result = ValidationResult(
            source_name=source_name,
            total_rows=len(data),
        )

        if not data:
            logger.warning("Empty dataset — no checks run")
            return result

        total_start = time.perf_counter()

        for name, check_fn, kwargs in self.checks:
            start = time.perf_counter()

            try:
                check_result = check_fn(data, **kwargs)
            except Exception as exc:
                logger.exception("Check '%s' raised an exception: %s", name, exc)
                check_result = CheckResult(
                    check_name=name,
                    description=f"Check raised an exception: {exc}",
                    severity=Severity.ERROR,
                    passed=False,
                    total_rows=len(data),
                    violation_count=1,
                    violations=[{
                        "row_index": -1,
                        "field": "engine",
                        "value": str(exc),
                        "message": f"Check failed with exception: {exc}",
                    }],
                )

            result.checks.append(check_result)

            if self.verbose:
                elapsed = time.perf_counter() - start
                status = "PASS" if check_result.passed else f"FAIL ({check_result.violation_count} violations)"
                logger.info(
                    "  [%s] %s — %s in %.2fs",
                    status,
                    check_result.check_name,
                    check_result.description,
                    elapsed,
                )

        total_elapsed = time.perf_counter() - total_start
        logger.info(
            "Validation complete: %d/%d checks passed, %d violations in %.1fs",
            result.passed_checks,
            result.total_checks,
            result.total_violations,
            total_elapsed,
        )

        return result

    def run_selected(
        self,
        data: list[dict[str, Any]],
        check_names: list[str],
        source_name: str = "unknown",
    ) -> ValidationResult:
        """Run only specific checks by name.

        Parameters
        ----------
        data : list[dict]
            The dataset to validate.
        check_names : list[str]
            Names of checks to run (e.g. ``[\"Duplicate Matches\", \"Invalid Dates\"]``).
        source_name : str
            Source name for the report.

        Returns
        -------
        ValidationResult
        """
        selected = [
            (n, fn, kw) for n, fn, kw in self.checks
            if n in check_names
        ]
        engine = ValidationEngine(checks=selected, verbose=self.verbose)
        return engine.run(data, source_name=source_name)
