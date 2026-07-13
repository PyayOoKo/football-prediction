"""
Tests for ``src.data_profiling.profiler``.

Covers the DataProfiler class and ProfilingReport data model
across all analysis dimensions: missing values, duplicates,
distributions, outliers, and schema/type validation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_profiling.profiler import DataProfiler, ProfileSection, ProfilingReport


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small synthetic football dataset with a variety of column types."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "home_team": np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd"], n),
        "away_team": np.random.choice(["Aston Villa", "Everton", "Spurs", "West Ham", "Wolves"], n),
        "result": np.random.choice(["H", "D", "A"], n, p=[0.45, 0.25, 0.30]),
        "home_goals": np.random.poisson(1.5, n),
        "away_goals": np.random.poisson(1.1, n),
        "season": np.random.choice(["2023/2024", "2024/2025"], n),
        "league": np.random.choice(["E0", "E1", "E2"], n),
        "BbAvH": np.round(np.random.uniform(1.5, 6.0, n), 2),
        "BbAvD": np.round(np.random.uniform(3.0, 5.0, n), 2),
        "BbAvA": np.round(np.random.uniform(1.5, 6.0, n), 2),
    })


@pytest.fixture
def df_with_nulls() -> pd.DataFrame:
    """DataFrame with known null percentages."""
    df = pd.DataFrame({
        "col_a": [1, 2, None, 4, 5],
        "col_b": [None, "x", "y", None, "z"],
        "col_c": [1.0, 2.0, 3.0, 4.0, 5.0],
        "col_d": [None, None, None, None, 1],  # 80% null
    })
    return df


@pytest.fixture
def profiler() -> DataProfiler:
    return DataProfiler()


# ── Test DataProfiler ───────────────────────────────────

class TestDataProfiler:
    def test_empty_df(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame()
        report = profiler.profile(df, source_name="empty")
        assert report.n_rows == 0
        assert report.n_columns == 0
        assert report.duration_seconds >= 0

    def test_single_column(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = profiler.profile(df, source_name="single")
        assert report.n_rows == 3
        assert report.n_columns == 1

    def test_source_name_persisted(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="test_league")
        assert report.source_name == "test_league"

    def test_missing_values_detected(self, profiler: DataProfiler, df_with_nulls: pd.DataFrame) -> None:
        report = profiler.profile(df_with_nulls, source_name="nulls")
        mv = report.missing_values.data
        assert isinstance(mv, dict)
        assert mv["total_missing"] > 0
        assert mv["columns_with_missing"] >= 2  # col_a, col_b, col_d
        # col_d has 4/5 null = 80%
        assert mv["columns"].get("col_d", 0) >= 80

    def test_no_missing_values(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        report = profiler.profile(df, source_name="clean")
        mv = report.missing_values.data
        assert isinstance(mv, dict)
        assert mv["total_missing"] == 0

    def test_duplicates_detected(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"id": [1, 2, 2, 3, 3, 3], "val": [10, 20, 20, 30, 30, 30]})
        report = profiler.profile(df, source_name="dups")
        dups = report.duplicate_records.data
        assert isinstance(dups, dict)
        assert dups["count"] > 0

    def test_no_duplicates(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        report = profiler.profile(df, source_name="clean")
        dups = report.duplicate_records.data
        assert isinstance(dups, dict)
        assert dups["count"] == 0

    def test_column_summary_has_expected_keys(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="cols")
        col_data = report.column_summary.data
        assert isinstance(col_data, pd.DataFrame)
        assert not col_data.empty
        expected = {"column", "dtype", "non_null", "null_pct", "unique"}
        assert expected.issubset(set(col_data.columns))

    def test_result_distribution(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="results")
        data = report.result_distribution.data
        assert isinstance(data, dict)
        assert "counts" in data
        for k in ["H", "D", "A"]:
            assert data["counts"].get(k, 0) >= 0

    def test_result_distribution_missing_column(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = profiler.profile(df, source_name="no_result")
        assert "error" in report.result_distribution.data

    def test_goal_distribution(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="goals")
        data = report.goal_distribution.data
        assert isinstance(data, dict)
        assert "home_mean" in data
        assert "away_mean" in data
        assert "max_goals" in data

    def test_goal_distribution_missing_columns(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = profiler.profile(df, source_name="no_goals")
        assert "error" in report.goal_distribution.data

    def test_odds_distribution_detected(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="odds")
        data = report.odds_distribution.data
        assert isinstance(data, dict)
        # Our sample has BbAvH, BbAvD, BbAvA
        assert data["n_odds_columns"] >= 3

    def test_odds_distribution_no_odds(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = profiler.profile(df, source_name="no_odds")
        assert "error" in report.odds_distribution.data

    def test_league_distribution(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="leagues")
        data = report.league_distribution.data
        assert isinstance(data, dict)
        assert data["n_unique"] >= 2

    def test_season_distribution(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="seasons")
        data = report.season_distribution.data
        assert isinstance(data, dict)
        assert data["n_unique"] >= 1

    def test_team_distribution(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="teams")
        data = report.team_distribution.data
        assert isinstance(data, dict)
        assert data["n_unique_teams"] >= 8  # 5 home + 5 away teams
        assert data["n_matches"] == len(sample_df)

    def test_team_distribution_missing_columns(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = profiler.profile(df, source_name="no_teams")
        assert "error" in report.team_distribution.data

    def test_home_advantage(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="home_adv")
        data = report.home_advantage.data
        assert isinstance(data, dict)
        assert "home_win_pct" in data
        assert "away_win_pct" in data
        assert "draw_pct" in data
        # Sum should be ~100%
        total = data["home_win_pct"] + data["draw_pct"] + data["away_win_pct"]
        assert abs(total - 100) < 1.0

    def test_home_advantage_no_result(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = profiler.profile(df, source_name="no_result")
        assert "error" in report.home_advantage.data

    def test_outliers(self, profiler: DataProfiler) -> None:
        # Create data with a clear outlier
        np.random.seed(42)
        values = np.random.normal(0, 1, 100).tolist() + [100, -100]  # extreme outliers
        df = pd.DataFrame({"normal": values, "uniform": np.random.uniform(0, 1, 102)})
        report = profiler.profile(df, source_name="outliers")
        data = report.outliers.data
        assert isinstance(data, dict)
        assert data["n_columns_with_outliers"] >= 1

    def test_no_outliers(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [1, 1, 1, 1, 1]})
        report = profiler.profile(df, source_name="no_outliers")
        data = report.outliers.data
        assert isinstance(data, dict)
        assert data["n_columns_with_outliers"] == 0

    def test_schema_validation(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="schema")
        data = report.schema_validation.data
        assert isinstance(data, dict)
        # Our sample has: date, home_team, away_team, result, home_goals, away_goals, league, season
        assert data["n_columns_present"] >= 6

    def test_type_validation(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="types")
        data = report.type_validation.data
        assert isinstance(data, dict)
        assert "columns" in data
        assert data["n_columns_with_issues"] >= 0

    def test_custom_odds_patterns(self) -> None:
        profiler = DataProfiler(odds_column_patterns=["my_odds"])
        df = pd.DataFrame({"my_odds_home": [1.5, 2.0], "my_odds_away": [3.0, 4.0], "other": [1, 2]})
        report = profiler.profile(df, source_name="custom_odds")
        assert report.odds_distribution.data["n_odds_columns"] == 2

    def test_custom_outlier_threshold(self) -> None:
        profiler = DataProfiler(outlier_std_threshold=1.0)  # Very sensitive
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
        report = profiler.profile(df, source_name="sensitive")
        assert report.outliers.data["n_columns_with_outliers"] >= 1

    def test_binary_result_column(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({
            "result": ["H", "A", "H", "H", "A"],
            "home_goals": [2, 0, 3, 1, 0],
            "away_goals": [0, 2, 1, 0, 3],
        })
        report = profiler.profile(df, source_name="binary")
        data = report.result_distribution.data
        assert data["counts"]["H"] == 3
        assert data["counts"]["A"] == 2

    def test_high_cardinality_team_distribution(self, profiler: DataProfiler) -> None:
        # Create data with many teams (more than max_unique=50 default)
        n_teams = 60
        teams = [f"Team_{i}" for i in range(n_teams)]
        df = pd.DataFrame({
            "home_team": np.random.choice(teams, 200),
            "away_team": np.random.choice(teams, 200),
        })
        report = profiler.profile(df, source_name="many_teams")
        data = report.team_distribution.data
        assert data["n_unique_teams"] <= n_teams * 2  # at most all unique


# ── Test ProfilingReport ────────────────────────────────

class TestProfilingReport:
    def test_to_dict_serializable(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="test")
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["source_name"] == "test"
        assert d["n_rows"] == len(sample_df)
        assert d["n_columns"] == len(sample_df.columns)
        assert "sections" in d

    def test_to_json_string(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="test")
        json_str = report.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["source_name"] == "test"

    def test_to_json_file(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="test")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report.to_json(f.name)
            f.flush()
            with open(f.name, "r") as f2:
                parsed = json.load(f2)
                assert parsed["source_name"] == "test"
        Path(f.name).unlink(missing_ok=True)

    def test_to_csv_file(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="test")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            report.to_csv(f.name)
            df_read = pd.read_csv(f.name)
            assert len(df_read) > 0
        Path(f.name).unlink(missing_ok=True)

    def test_to_html_file(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="test")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            report.to_html(f.name)
            with open(f.name, "r", encoding="utf-8") as f2:
                html = f2.read()
                assert "<!DOCTYPE html>" in html
                assert "plotly" in html.lower() or "Plotly" in html
        Path(f.name).unlink(missing_ok=True)

    def test_summary_text(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="test")
        summary = report.summary_text()
        assert "Profile:" in summary
        assert "Rows:" in summary
        assert "Columns:" in summary
        assert "Duration:" in summary

    def test_empty_report_summary(self) -> None:
        report = ProfilingReport(source_name="empty")
        summary = report.summary_text()
        assert "Rows: 0" in summary

    def test_all_sections_exist(self, profiler: DataProfiler, sample_df: pd.DataFrame) -> None:
        report = profiler.profile(sample_df, source_name="test")
        expected_sections = [
            "missing_values", "duplicate_records", "column_summary",
            "result_distribution", "goal_distribution", "odds_distribution",
            "league_distribution", "season_distribution", "team_distribution",
            "home_advantage", "outliers", "schema_validation", "type_validation",
        ]
        for name in expected_sections:
            section = getattr(report, name, None)
            assert section is not None, f"Missing section: {name}"
            assert isinstance(section, ProfileSection)

    def test_report_all_nulls(self, profiler: DataProfiler) -> None:
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        report = profiler.profile(df, source_name="all_nulls")
        mv = report.missing_values.data
        assert mv["missing_pct"] == 100.0


# ── Test ProfileSection ─────────────────────────────────

class TestProfileSection:
    def test_minimal_creation(self) -> None:
        section = ProfileSection("test", {"key": "value"})
        assert section.name == "test"
        assert section.data == {"key": "value"}
        assert section.chart_type == "table"

    def test_with_chart_type(self) -> None:
        section = ProfileSection("test", {"key": "value"}, chart_type="pie")
        assert section.chart_type == "pie"
