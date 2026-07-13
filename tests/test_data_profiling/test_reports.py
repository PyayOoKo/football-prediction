"""
Tests for ``src.data_profiling.reports``.

Covers the ReportGenerator class and HTML dashboard generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_profiling.profiler import DataProfiler, ProfilingReport
from src.data_profiling.reports import ReportGenerator


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2024-01-07", "2024-01-08", "2024-01-09"],
        "home_team": ["Team A", "Team C", "Team E"],
        "away_team": ["Team B", "Team D", "Team F"],
        "home_goals": [2, 1, 0],
        "away_goals": [1, 1, 3],
        "result": ["H", "D", "A"],
        "league": ["E0", "E0", "E1"],
        "season": ["2425", "2425", "2425"],
    })


@pytest.fixture
def report(sample_df: pd.DataFrame) -> ProfilingReport:
    profiler = DataProfiler()
    return profiler.profile(sample_df, source_name="test_source")


@pytest.fixture
def empty_report() -> ProfilingReport:
    profiler = DataProfiler()
    df = pd.DataFrame()
    return profiler.profile(df, source_name="empty")


# ── Test ReportGenerator ────────────────────────────────

class TestReportGenerator:
    def test_init(self, report: ProfilingReport) -> None:
        gen = ReportGenerator(report)
        assert gen.report is report

    def test_html_output_string(self, report: ProfilingReport) -> None:
        gen = ReportGenerator(report)
        html = gen._generate_html()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Dataset Profile" in html
        assert "test_source" in html

    def test_html_contains_plotly(self, report: ProfilingReport) -> None:
        gen = ReportGenerator(report)
        html = gen._generate_html()
        # The HTML should reference plotly CDN
        assert "cdn.plot.ly" in html or "plotly" in html

    def test_html_contains_sections(self, report: ProfilingReport) -> None:
        gen = ReportGenerator(report)
        html = gen._generate_html()
        section_names = [
            "Missing Values", "Duplicate Records", "Column Summary",
            "Result Distribution", "Goal Distribution", "Home Advantage",
            "League Distribution", "Team Distribution",
        ]
        for name in section_names:
            assert name in html, f"Missing section in HTML: {name}"

    def test_html_summary_cards(self, report: ProfilingReport) -> None:
        gen = ReportGenerator(report)
        html = gen._generate_html()
        assert "summary-card" in html
        assert str(report.n_rows) in html
        assert str(report.n_columns) in html

    def test_html_meta_info(self, report: ProfilingReport) -> None:
        gen = ReportGenerator(report)
        html = gen._generate_html()
        assert "Rows" in html
        assert "Columns" in html
        assert "Duration" in html

    def test_save_to_file(self, report: ProfilingReport) -> None:
        gen = ReportGenerator(report)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            gen.to_html(f.name)
            with open(f.name, "r", encoding="utf-8") as f2:
                html = f2.read()
                assert "<!DOCTYPE html>" in html
                assert len(html) > 500  # Substantial HTML
        Path(f.name).unlink(missing_ok=True)

    def test_empty_report_html(self, empty_report: ProfilingReport) -> None:
        """HTML generation should not crash on empty/edge-case reports."""
        gen = ReportGenerator(empty_report)
        html = gen._generate_html()
        assert "<!DOCTYPE html>" in html
        assert "empty" in html

    def test_html_escapes_columns(self) -> None:
        """Column names with special chars should be escaped."""
        df = pd.DataFrame({
            "safe_col": [1, 2],
            "<script>alert('xss')</script>": [3, 4],
            "team's stats": [5, 6],
        })
        profiler = DataProfiler()
        report = profiler.profile(df, source_name="xss_test")
        gen = ReportGenerator(report)
        html = gen._generate_html()
        # Should have escaped HTML-sensitive characters
        assert "&lt;script&gt;" in html or "&#39;" in html or "&amp;" in html

    def test_missing_values_section_html(self, report: ProfilingReport) -> None:
        """The missing values section should render as HTML."""
        gen = ReportGenerator(report)
        html = gen._generate_html()
        assert "Missing Values" in html

    def test_no_results_html(self) -> None:
        """Report with no result column should still render."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        profiler = DataProfiler()
        report = profiler.profile(df, source_name="no_results")
        gen = ReportGenerator(report)
        html = gen._generate_html()
        assert "no_results" in html
        assert "<!DOCTYPE html>" in html


# ── Test profiling lifecycle ────────────────────────────

class TestProfilingLifecycle:
    def test_profile_then_report(self, sample_df: pd.DataFrame) -> None:
        """Full cycle: profile → generate HTML → verify key metrics."""
        profiler = DataProfiler()
        report = profiler.profile(sample_df, source_name="lifecycle")

        assert report.n_rows == 3
        assert report.n_columns == len(sample_df.columns)

        gen = ReportGenerator(report)
        html = gen._generate_html()

        # Key data points should appear in HTML
        assert "3" in html  # 3 rows
        assert "H" in html or "Home Win" in html
        assert "E0" in html or "E1" in html
        assert "Team A" in html or "Team C" in html

    def test_to_json_then_to_html(self, sample_df: pd.DataFrame) -> None:
        """Profile → JSON → load → HTML should work."""
        profiler = DataProfiler()
        report = profiler.profile(sample_df, source_name="json_test")

        # Export to JSON
        json_str = report.to_json()
        assert len(json_str) > 0

        # Export to HTML
        gen = ReportGenerator(report)
        html = gen._generate_html()
        assert "json_test" in html

    def test_all_sections_render(self, sample_df: pd.DataFrame) -> None:
        """Every section renderer should produce non-empty output."""
        profiler = DataProfiler()
        report = profiler.profile(sample_df, source_name="all_sections")
        gen = ReportGenerator(report)
        html = gen._generate_html()

        # All major section headings should be present
        sections = [
            "Missing Values", "Duplicate Records", "Column Summary",
            "Result Distribution", "Goal Distribution", "Home Advantage",
            "Odds Distribution", "League Distribution", "Season Distribution",
            "Team Distribution", "Outliers", "Schema Validation",
            "Type Validation",
        ]
        for name in sections:
            assert name in html, f"Section '{name}' missing from HTML"
