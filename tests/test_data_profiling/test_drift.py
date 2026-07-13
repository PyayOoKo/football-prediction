"""
Tests for ``src.data_profiling.drift``.

Covers drift detection across row count, column count, nulls,
result distributions, home advantage, schema, and duplicates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_profiling.drift import DataDriftDetector, DriftMetric, DriftReport
from src.data_profiling.profiler import DataProfiler, ProfilingReport


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def profiler() -> DataProfiler:
    return DataProfiler()


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "home_team": ["A", "C", "E"],
        "away_team": ["B", "D", "F"],
        "result": ["H", "D", "A"],
        "home_goals": [2, 1, 0],
        "away_goals": [0, 1, 3],
        "league": ["E0", "E0", "E0"],
        "season": ["2425", "2425", "2425"],
    })


@pytest.fixture
def larger_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "home_team": ["A", "C", "E", "G"],
        "away_team": ["B", "D", "F", "H"],
        "result": ["H", "D", "A", "H"],
        "home_goals": [2, 1, 0, 3],
        "away_goals": [0, 1, 3, 1],
        "league": ["E0", "E0", "E0", "E1"],
        "season": ["2425", "2425", "2425", "2425"],
    })


# ── Test DriftReport ────────────────────────────────────

class TestDriftReport:
    def test_empty_report(self) -> None:
        report = DriftReport(current_source="curr", previous_source="prev")
        assert report.passed
        assert report.n_warnings == 0
        assert len(report.metrics) == 0

    def test_with_warnings(self) -> None:
        report = DriftReport(current_source="curr", previous_source="prev")
        report.metrics.append(DriftMetric(
            name="Row Count", previous_value=100, current_value=200,
            delta=1.0, severity="warning",
        ))
        assert not report.passed
        assert report.n_warnings == 1

    def test_to_dict(self) -> None:
        report = DriftReport(current_source="curr", previous_source="prev")
        report.metrics.append(DriftMetric(
            name="Row Count", previous_value=100, current_value=90,
            delta=-0.1, severity="info",
        ))
        d = report.to_dict()
        assert d["current_source"] == "curr"
        assert d["previous_source"] == "prev"
        assert len(d["metrics"]) == 1
        assert d["metrics"][0]["name"] == "Row Count"

    def test_summary_text_passed(self) -> None:
        report = DriftReport(current_source="curr", previous_source="prev")
        text = report.summary_text()
        assert "No significant drift" in text

    def test_summary_text_warnings(self) -> None:
        report = DriftReport(current_source="curr", previous_source="prev")
        report.metrics.append(DriftMetric(
            name="Row Count", previous_value=100, current_value=200,
            delta=1.0, severity="critical",
        ))
        text = report.summary_text()
        assert "drift signal" in text
        assert "Row Count" in text


# ── Test DriftMetric ────────────────────────────────────

class TestDriftMetric:
    def test_default_severity(self) -> None:
        m = DriftMetric(name="test", previous_value=1, current_value=2, delta=1.0)
        assert m.severity == "info"

    def test_description(self) -> None:
        m = DriftMetric(name="test", previous_value=1, current_value=2,
                        delta=1.0, description="Something changed")
        assert m.description == "Something changed"


# ── Test DataDriftDetector ──────────────────────────────

class TestDataDriftDetector:
    def test_detect_no_drift(self, profiler: DataProfiler, small_df: pd.DataFrame) -> None:
        """Profiling the same data twice should produce no drift."""
        prev = profiler.profile(small_df, source_name="prev")
        curr = profiler.profile(small_df.copy(), source_name="curr")
        detector = DataDriftDetector()
        drift = detector.detect(curr, prev)
        assert drift.passed, f"Expected no drift but got: {[m.name for m in drift.metrics]}"

    def test_detect_row_count_drift(self, profiler: DataProfiler,
                                     small_df: pd.DataFrame, larger_df: pd.DataFrame) -> None:
        prev = profiler.profile(small_df, source_name="prev")
        curr = profiler.profile(larger_df, source_name="curr")
        detector = DataDriftDetector(row_count_threshold=0.01)  # Very sensitive
        drift = detector.detect(curr, prev)
        # Row count changed from 3 to 4 (~33%) — should be detected
        row_metrics = [m for m in drift.metrics if "Row Count" in m.name]
        assert len(row_metrics) >= 1
        assert row_metrics[0].severity in ("warning", "critical")

    def test_detect_column_count_drift(self, profiler: DataProfiler, small_df: pd.DataFrame) -> None:
        prev = profiler.profile(small_df, source_name="prev")
        # Remove a column
        modified = small_df.drop(columns=["league"])
        curr = profiler.profile(modified, source_name="curr")
        detector = DataDriftDetector()
        drift = detector.detect(curr, prev)
        col_metrics = [m for m in drift.metrics if "Column" in m.name]
        assert len(col_metrics) >= 1

    def test_detect_null_drift(self) -> None:
        """Create profiles with clearly different null percentages."""
        profiler = DataProfiler()
        # prev: col_b has 20% null (1/5), curr: col_b has 60% null (3/5)
        prev_df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [None, "x", "y", "z", "w"]})
        curr_df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [None, None, None, "x", "y"]})
        prev = profiler.profile(prev_df, source_name="prev")
        curr = profiler.profile(curr_df, source_name="curr")
        detector = DataDriftDetector(null_pct_threshold=5.0)
        drift = detector.detect(curr, prev)
        null_metrics = [m for m in drift.metrics if "Null %" in m.name]
        # 20% → 60%: 40pp difference, well above 5pp threshold
        assert len(null_metrics) >= 1, f"Expected null drift but got: {[m.name for m in drift.metrics]}"

    def test_detect_result_distribution_drift(self) -> None:
        profiler = DataProfiler()
        prev_df = pd.DataFrame({"result": ["H"] * 90 + ["D"] * 5 + ["A"] * 5})  # strong home bias
        curr_df = pd.DataFrame({"result": ["H"] * 33 + ["D"] * 33 + ["A"] * 34})  # balanced
        prev = profiler.profile(prev_df, source_name="prev")
        curr = profiler.profile(curr_df, source_name="curr")
        detector = DataDriftDetector()
        drift = detector.detect(curr, prev)
        result_metrics = [m for m in drift.metrics if "Result" in m.name]
        assert len(result_metrics) >= 1

    def test_detect_home_advantage_drift(self) -> None:
        profiler = DataProfiler()
        prev_df = pd.DataFrame({"result": ["H"] * 80 + ["D"] * 10 + ["A"] * 10})
        curr_df = pd.DataFrame({"result": ["H"] * 30 + ["D"] * 35 + ["A"] * 35})
        prev = profiler.profile(prev_df, source_name="prev")
        curr = profiler.profile(curr_df, source_name="curr")
        detector = DataDriftDetector()
        drift = detector.detect(curr, prev)
        ha_metrics = [m for m in drift.metrics if "Home Win" in m.name]
        assert len(ha_metrics) >= 1

    def test_detect_schema_drift_add_remove_column(self) -> None:
        profiler = DataProfiler()
        prev_df = pd.DataFrame({"a": [1], "b": [2]})
        curr_df = pd.DataFrame({"a": [1], "c": [3]})  # b removed, c added
        prev = profiler.profile(prev_df, source_name="prev")
        curr = profiler.profile(curr_df, source_name="curr")
        detector = DataDriftDetector()
        drift = detector.detect(curr, prev)
        schema_metrics = [m for m in drift.metrics if "Column" in m.name]
        assert len(schema_metrics) >= 1

    def test_detect_duplicate_drift(self) -> None:
        profiler = DataProfiler()
        prev_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})  # 0 duplicates
        curr_df = pd.DataFrame({"a": [1, 1, 1, 2, 3]})  # duplicates
        prev = profiler.profile(prev_df, source_name="prev")
        curr = profiler.profile(curr_df, source_name="curr")
        detector = DataDriftDetector()
        drift = detector.detect(curr, prev)
        dup_metrics = [m for m in drift.metrics if "Duplicate" in m.name]
        # 0% → 40% duplicates should trigger
        assert len(dup_metrics) >= 1, f"Expected duplicate drift but got: {drift.metrics}"

    def test_identical_profiles(self, profiler: DataProfiler, small_df: pd.DataFrame) -> None:
        """Two profiles of the exact same data should produce zero drift."""
        prev = profiler.profile(small_df, source_name="test")
        curr = profiler.profile(small_df.copy(), source_name="test")
        detector = DataDriftDetector()
        drift = detector.detect(curr, prev)
        assert drift.passed, f"Expected no drift warnings but got: {[m.name for m in drift.metrics]}"
        assert len(drift.metrics) == 0, f"Expected 0 metrics but got: {drift.metrics}"

    def test_custom_thresholds(self, profiler: DataProfiler,
                                small_df: pd.DataFrame, larger_df: pd.DataFrame) -> None:
        prev = profiler.profile(small_df, source_name="prev")
        curr = profiler.profile(larger_df, source_name="curr")
        # Very relaxed thresholds → no warnings or critical, only info
        detector = DataDriftDetector(
            row_count_threshold=1.0,  # 100% change allowed without warning
            null_pct_threshold=100.0,
            metric_threshold=1.0,
        )
        drift = detector.detect(curr, prev)
        # Row count changed 33% which is under 100% → info severity only
        row_metrics = [m for m in drift.metrics if "Row Count" in m.name]
        if row_metrics:
            assert row_metrics[0].severity == "info"
        assert drift.passed  # No warnings or critical
