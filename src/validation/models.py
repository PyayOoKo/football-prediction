"""
Data models for the validation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Severity level for a validation check result."""

    ERROR = "error"       # Data must be fixed
    WARNING = "warning"   # Suspicious but not necessarily wrong
    INFO = "info"         # Statistical observation


@dataclass
class CheckResult:
    """Result of a single validation check.

    Attributes
    ----------
    check_name : str
        Human-readable check name (e.g. "Duplicate Matches").
    description : str
        What the check validates.
    severity : Severity
        How serious violations are.
    passed : bool
        True if no violations found.
    total_rows : int
        Number of rows examined.
    violation_count : int
        Number of rows that failed the check.
    violations : list[dict]
        Specific row details for each violation.
    """

    check_name: str
    description: str
    severity: Severity
    passed: bool
    total_rows: int = 0
    violation_count: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "description": self.description,
            "severity": self.severity.value,
            "passed": self.passed,
            "total_rows": self.total_rows,
            "violation_count": self.violation_count,
            "sample_violations": self.violations[:10],
        }


@dataclass
class ValidationResult:
    """Aggregated results from all validation checks.

    Supports exporting to HTML, CSV, and JSON.
    """

    source_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_rows: int = 0
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_checks(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def total_violations(self) -> int:
        return sum(c.violation_count for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "timestamp": self.timestamp,
            "total_rows": self.total_rows,
            "passed": self.passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "total_violations": self.total_violations,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, filepath: str | None = None) -> str:
        """Export results as a JSON string or write to a file.

        Parameters
        ----------
        filepath : str, optional
            If provided, writes to file. Otherwise returns JSON string.

        Returns
        -------
        str
            JSON string (empty if written to file).
        """
        import json

        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            return ""
        return json_str

    def to_csv(self, filepath: str) -> None:
        """Export violation details to a CSV file.

        Each row in the CSV represents one violation from one check.

        Parameters
        ----------
        filepath : str
            Output file path.
        """
        import csv

        with open(filepath, "w", newline="") as f:
            fieldnames = [
                "check_name", "severity", "row_index",
                "field", "value", "message",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for check in self.checks:
                for violation in check.violations:
                    writer.writerow({
                        "check_name": check.check_name,
                        "severity": check.severity.value,
                        **violation,
                    })

    def to_html(self, filepath: str) -> None:
        """Export results as a professional HTML report.

        Parameters
        ----------
        filepath : str
            Output HTML file path.
        """
        from src.validation.reporter import HTMLReporter

        html = HTMLReporter.render(self)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
