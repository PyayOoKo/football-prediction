"""
Data models for data quality reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CoverageMetrics:
    """Coverage statistics for key data dimensions.

    Attributes
    ----------
    odds_coverage_pct : float
        Percentage of matches with non-null odds data (any source).
    xg_coverage_pct : float
        Percentage of matches with xG (expected goals) data.
    league_coverage : dict[str, int]
        Count of matches per league code.
    league_coverage_pct : float
        Percentage of matches mapped to known leagues.
    season_coverage : dict[str, int]
        Count of matches per season.
    season_count : int
        Number of distinct seasons.
    schema_version : str
        Current schema version identifier.
    n_columns_expected : int
        Number of expected columns in the schema.
    n_columns_actual : int
        Number of actual columns in the dataset.
    columns_missing : list[str]
        Expected columns that are absent.
    columns_added : list[str]
        Columns present but not in expected schema.
    """

    odds_coverage_pct: float = 0.0
    xg_coverage_pct: float = 0.0
    league_coverage: dict[str, int] = field(default_factory=dict)
    league_coverage_pct: float = 0.0
    season_coverage: dict[str, int] = field(default_factory=dict)
    season_count: int = 0
    schema_version: str = ""
    n_columns_expected: int = 0
    n_columns_actual: int = 0
    columns_missing: list[str] = field(default_factory=list)
    columns_added: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "odds_coverage_pct": round(self.odds_coverage_pct, 2),
            "xg_coverage_pct": round(self.xg_coverage_pct, 2),
            "league_coverage_pct": round(self.league_coverage_pct, 2),
            "league_coverage": self.league_coverage,
            "season_coverage": self.season_coverage,
            "season_count": self.season_count,
            "schema_version": self.schema_version,
            "n_columns_expected": self.n_columns_expected,
            "n_columns_actual": self.n_columns_actual,
            "columns_missing": self.columns_missing[:10],
            "columns_added": self.columns_added[:10],
        }


@dataclass
class DataQualitySnapshot:
    """Complete snapshot of every data quality dimension.

    Aggregates data from the monitoring store, profiling engine,
    validation engine, and coverage analyzer into one structure.
    """

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_name: str = ""
    n_rows: int = 0
    n_columns: int = 0

    # ── Missing Values ──
    missing_cells: int = 0
    missing_pct: float = 0.0
    columns_with_missing: int = 0

    # ── Duplicate Matches ──
    duplicate_count: int = 0
    duplicate_pct: float = 0.0

    # ── Coverage ──
    coverage: CoverageMetrics = field(default_factory=CoverageMetrics)

    # ── Data Drift ──
    drift_metrics_count: int = 0
    drift_warnings: int = 0
    drift_passed: bool = True

    # ── Schema Changes ──
    schema_ok: bool = True

    # ── Import / Pipeline ──
    import_success_rate: float = 1.0
    pipeline_runtime_avg: float = 0.0
    pipeline_runs: int = 0

    # ── Validation ──
    validation_passed: int = 0
    validation_total: int = 0
    validation_errors: int = 0

    # ── Database ──
    db_size_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source_name": self.source_name,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "missing_cells": self.missing_cells,
            "missing_pct": round(self.missing_pct, 2),
            "columns_with_missing": self.columns_with_missing,
            "duplicate_count": self.duplicate_count,
            "duplicate_pct": round(self.duplicate_pct, 2),
            "coverage": self.coverage.to_dict(),
            "drift_metrics_count": self.drift_metrics_count,
            "drift_warnings": self.drift_warnings,
            "drift_passed": self.drift_passed,
            "schema_ok": self.schema_ok,
            "import_success_rate": round(self.import_success_rate, 4),
            "pipeline_runtime_avg": round(self.pipeline_runtime_avg, 2),
            "pipeline_runs": self.pipeline_runs,
            "validation_passed": self.validation_passed,
            "validation_total": self.validation_total,
            "validation_errors": self.validation_errors,
            "db_size_mb": round(self.db_size_mb, 2),
        }


@dataclass
class DataQualitySummary:
    """Human-readable text summary of data quality."""

    lines: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.lines.append(line)

    def __str__(self) -> str:
        return "\n".join(self.lines)

    def to_dict(self) -> dict[str, str]:
        return {"summary": str(self)}
