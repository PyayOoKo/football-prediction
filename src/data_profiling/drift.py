"""
Data Drift Detection — compare current profiling results against previous imports.

Detects statistically significant changes in distributions, schema, null rates,
and key metrics between consecutive dataset versions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data_profiling.profiler import ProfileSection, ProfilingReport

logger = logging.getLogger(__name__)


@dataclass
class DriftMetric:
    """A single drift signal detected between two profiles."""

    name: str
    previous_value: Any
    current_value: Any
    delta: float
    severity: str = "info"  # info, warning, critical
    description: str = ""


@dataclass
class DriftReport:
    """Complete drift analysis comparing a current vs previous profile."""

    current_source: str
    previous_source: str
    metrics: list[DriftMetric] = field(default_factory=list)

    @property
    def n_warnings(self) -> int:
        return sum(1 for m in self.metrics if m.severity in ("warning", "critical"))

    @property
    def passed(self) -> bool:
        return self.n_warnings == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_source": self.current_source,
            "previous_source": self.previous_source,
            "passed": self.passed,
            "n_warnings": self.n_warnings,
            "n_metrics": len(self.metrics),
            "metrics": [
                {"name": m.name, "previous": str(m.previous_value),
                 "current": str(m.current_value), "delta": round(m.delta, 3),
                 "severity": m.severity, "description": m.description}
                for m in self.metrics
            ],
        }

    def summary_text(self) -> str:
        lines = [f"📊 Drift: {self.previous_source} → {self.current_source}"]
        if self.passed:
            lines.append("   ✅ No significant drift detected")
        else:
            lines.append(f"   ⚠ {self.n_warnings} drift signal(s)")
        for m in self.metrics:
            icon = {"info": "ℹ", "warning": "⚠", "critical": "❌"}.get(m.severity, "ℹ")
            lines.append(f"   {icon} {m.name}: {m.previous_value} → {m.current_value} ({m.delta:+.1%})")
        return "\n".join(lines)


class DataDriftDetector:
    """Detects drift between two profiling reports.

    Parameters
    ----------
    row_count_threshold : float
        Minimum relative change in row count to flag (default 0.05 = 5%).
    null_pct_threshold : float
        Minimum absolute change in null % to flag (default 5pp).
    metric_threshold : float
        Minimum relative change in key metrics to flag (default 0.1 = 10%).
    """

    def __init__(
        self,
        row_count_threshold: float = 0.05,
        null_pct_threshold: float = 5.0,
        metric_threshold: float = 0.10,
    ) -> None:
        self.row_count_threshold = row_count_threshold
        self.null_pct_threshold = null_pct_threshold
        self.metric_threshold = metric_threshold

    def detect(
        self,
        current: ProfilingReport,
        previous: ProfilingReport,
    ) -> DriftReport:
        """Compare two profiling reports and return drift signals.

        Parameters
        ----------
        current : ProfilingReport
            The newer (current) profiling report.
        previous : ProfilingReport
            The older (previous) profiling report.

        Returns
        -------
        DriftReport
        """
        report = DriftReport(
            current_source=current.source_name,
            previous_source=previous.source_name,
        )

        # 1. Row count drift
        self._check_row_count(current, previous, report)

        # 2. Column count drift
        self._check_column_count(current, previous, report)

        # 3. Null percentage drift per column
        self._check_null_drift(current, previous, report)

        # 4. Result distribution drift
        self._check_result_distribution_drift(current, previous, report)

        # 5. Home advantage drift
        self._check_home_advantage_drift(current, previous, report)

        # 6. Schema drift (columns added/removed)
        self._check_schema_drift(current, previous, report)

        # 7. Duplicate rate drift
        self._check_duplicate_drift(current, previous, report)

        logger.info(
            "Drift detection: %d metrics, %d warnings",
            len(report.metrics), report.n_warnings,
        )
        return report

    def _check_row_count(self, current: ProfilingReport, previous: ProfilingReport, report: DriftReport) -> None:
        if previous.n_rows == 0:
            return
        delta = (current.n_rows - previous.n_rows) / previous.n_rows
        if abs(delta) < 1e-9:
            return  # No change
        severity = "info" if abs(delta) < self.row_count_threshold else "warning" if abs(delta) < self.row_count_threshold * 3 else "critical"
        report.metrics.append(DriftMetric(
            name="Row Count",
            previous_value=previous.n_rows,
            current_value=current.n_rows,
            delta=delta,
            severity=severity,
            description=f"Row count changed by {delta:+.1%}",
        ))

    def _check_column_count(self, current: ProfilingReport, previous: ProfilingReport, report: DriftReport) -> None:
        if previous.n_columns == 0:
            return
        delta = (current.n_columns - previous.n_columns) / previous.n_columns
        if abs(delta) < 1e-9:
            return  # No change
        severity = "info" if abs(delta) < 0.2 else "warning"
        severity = "critical" if abs(delta) >= 0.5 else severity
        report.metrics.append(DriftMetric(
            name="Column Count",
            previous_value=previous.n_columns,
            current_value=current.n_columns,
            delta=delta,
            severity=severity,
            description=f"Column count changed: {previous.n_columns} → {current.n_columns}",
        ))

    def _check_null_drift(self, current: ProfilingReport, previous: ProfilingReport, report: DriftReport) -> None:
        prev_mv = previous.missing_values.data
        curr_mv = current.missing_values.data
        if not isinstance(prev_mv, dict) or not isinstance(curr_mv, dict):
            return

        for col in set(list(prev_mv.get("columns", {}).keys()) + list(curr_mv.get("columns", {}).keys())):
            prev_pct = float(prev_mv.get("columns", {}).get(col, 0))
            curr_pct = float(curr_mv.get("columns", {}).get(col, 0))
            diff = abs(curr_pct - prev_pct)
            if diff > self.null_pct_threshold:
                severity = "warning" if diff < self.null_pct_threshold * 2 else "critical"
                report.metrics.append(DriftMetric(
                    name=f"Null % - {col}",
                    previous_value=f"{prev_pct:.1f}%",
                    current_value=f"{curr_pct:.1f}%",
                    delta=diff / 100,
                    severity=severity,
                    description=f"Null % in {col} changed by {diff:.1f}pp",
                ))

    def _check_result_distribution_drift(self, current: ProfilingReport, previous: ProfilingReport, report: DriftReport) -> None:
        prev_data = previous.result_distribution.data
        curr_data = current.result_distribution.data
        if not isinstance(prev_data, dict) or not isinstance(curr_data, dict):
            return

        for outcome in ["H", "D", "A"]:
            prev_pct = float(prev_data.get("percentages", {}).get(outcome, 0))
            curr_pct = float(curr_data.get("percentages", {}).get(outcome, 0))
            diff = abs(curr_pct - prev_pct)
            if diff > self.metric_threshold * 100:
                report.metrics.append(DriftMetric(
                    name=f"Result {outcome}",
                    previous_value=f"{prev_pct:.1f}%",
                    current_value=f"{curr_pct:.1f}%",
                    delta=diff / 100,
                    severity="warning" if diff < 15 else "critical",
                    description=f"'{outcome}' distribution shifted by {diff:.1f}pp",
                ))

    def _check_home_advantage_drift(self, current: ProfilingReport, previous: ProfilingReport, report: DriftReport) -> None:
        prev_ha = previous.home_advantage.data
        curr_ha = current.home_advantage.data
        if not isinstance(prev_ha, dict) or not isinstance(curr_ha, dict):
            return
        diff = abs(curr_ha.get("home_win_pct", 0) - prev_ha.get("home_win_pct", 0))
        if diff > self.metric_threshold * 100:
            report.metrics.append(DriftMetric(
                name="Home Win %",
                previous_value=f"{prev_ha.get('home_win_pct', 0):.1f}%",
                current_value=f"{curr_ha.get('home_win_pct', 0):.1f}%",
                delta=diff / 100,
                severity="warning" if diff < 15 else "critical",
            ))

    def _check_schema_drift(self, current: ProfilingReport, previous: ProfilingReport, report: DriftReport) -> None:
        prev_cols = previous.column_summary.data
        curr_cols = current.column_summary.data

        def _get_column_names(data: Any) -> set[str]:
            if isinstance(data, pd.DataFrame):
                if "column" in data.columns:
                    return set(data["column"].dropna())
                return set()
            if isinstance(data, list):
                return {r.get("column") for r in data if isinstance(r, dict)}
            if isinstance(data, dict):
                cols = data.get("columns", {})
                if isinstance(cols, dict):
                    return set(cols.keys())
            return set()

        prev_names = _get_column_names(prev_cols)
        curr_names = _get_column_names(curr_cols)

        if not prev_names and not curr_names:
            return

        added = curr_names - prev_names
        removed = prev_names - curr_names

        if added:
            report.metrics.append(DriftMetric(
                name="Columns Added",
                previous_value=0, current_value=len(added),
                delta=len(added), severity="warning",
                description=f"New columns: {', '.join(sorted(added)[:10])}",
            ))
        if removed:
            report.metrics.append(DriftMetric(
                name="Columns Removed",
                previous_value=len(removed), current_value=0,
                delta=len(removed), severity="critical",
                description=f"Removed columns: {', '.join(sorted(removed)[:10])}",
            ))

    def _check_duplicate_drift(self, current: ProfilingReport, previous: ProfilingReport, report: DriftReport) -> None:
        prev_dups = previous.duplicate_records.data
        curr_dups = current.duplicate_records.data
        if not isinstance(prev_dups, dict) or not isinstance(curr_dups, dict):
            return
        prev_pct = float(prev_dups.get("pct", 0))
        curr_pct = float(curr_dups.get("pct", 0))
        diff = abs(curr_pct - prev_pct)
        if diff > 1.0:  # 1pp change in duplicate rate
            report.metrics.append(DriftMetric(
                name="Duplicate Rate",
                previous_value=f"{prev_pct:.2f}%",
                current_value=f"{curr_pct:.2f}%",
                delta=diff / 100,
                severity="warning" if diff < 5 else "critical",
                description=f"Duplicate rate changed by {diff:.2f}pp",
            ))
