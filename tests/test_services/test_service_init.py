"""Regression tests for top-level service functions in src/services/__init__.py.

Covers:
  - load_and_prepare() with temporary CSV (default + injected config)
  - resolve_data_path() with existing/missing paths
  - add_target_col() mapping, idempotency, NaN handling
  - Clear error messages for missing files and malformed data
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════
#  Fixtures & helpers
# ═══════════════════════════════════════════════════════════


@dataclass
class _MockPaths:
    """Minimal paths object matching the config.paths interface."""

    raw: Path
    processed: Path = Path("/tmp/processed")
    models: Path = Path("/tmp/models")
    data: Path = Path("/tmp/data")


@dataclass
class _MockDataConfig:
    """Minimal data config matching cfg.data interface."""

    results_file: str = "results_clean.csv"


@dataclass
class _MockPreprocessingConfig:
    """Minimal preprocessing config."""

    normalise_teams: bool = True


@dataclass
class _MockWorldCupConfig:
    """Minimal worldcup config — points to non-existent file to avoid discovery."""

    data_path: str = "/tmp/nonexistent_worldcup.csv"


@dataclass
class _MockTrainConfig:
    """Minimal train config."""

    model_type: str = "lightgbm"
    cv_folds: int = 5
    tune_hyperparams: bool = False


@dataclass
class _MockFeatureSelectionConfig:
    """Minimal feature selection (disabled by default)."""

    enabled: bool = False
    method: str = "mutual_info"
    n_features: int = 30
    importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    drop_redundant_first: bool = True


@dataclass
class _MockConfig:
    """Minimal config for dependency injection testing.

    Uses temporary-dir-safe defaults so tests don't discover real project files.
    """

    paths: _MockPaths = field(default_factory=lambda: _MockPaths(raw=Path("/tmp/raw")))
    data: _MockDataConfig = field(default_factory=_MockDataConfig)
    preprocessing: _MockPreprocessingConfig = field(
        default_factory=_MockPreprocessingConfig
    )
    worldcup: _MockWorldCupConfig = field(default_factory=_MockWorldCupConfig)
    train: _MockTrainConfig = field(default_factory=_MockTrainConfig)
    feature_selection: _MockFeatureSelectionConfig = field(
        default_factory=_MockFeatureSelectionConfig
    )
    verbose: bool = False


def _write_minimal_csv(path: Path, rows: int = 5) -> None:
    """Write a small but valid match-results CSV to *path*.

    Uses a column set that avoids triggering the football-data.co.uk
    specific cleaner (which has fussy type checks on our random data).
    The generic pipeline path (lowercase + strip) is exercised instead.
    """
    rng = np.random.default_rng(42)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "home_team",
                "away_team",
                "result",
                "home_goals",
                "away_goals",
                "season",
            ]
        )
        for i in range(rows):
            writer.writerow(
                [
                    f"2024-0{i+1}-15",
                    f"Team_{rng.integers(1, 20)}",
                    f"Team_{rng.integers(1, 20)}",
                    rng.choice(["H", "D", "A"]),
                    int(rng.integers(0, 5)),
                    int(rng.integers(0, 5)),
                    "2024",
                ]
            )


# ═══════════════════════════════════════════════════════════
#  load_and_prepare() regression tests
# ═══════════════════════════════════════════════════════════


class TestLoadAndPrepare:
    """load_and_prepare() should never raise AttributeError after the config fix."""

    def test_loads_temp_csv_with_default_config(self) -> None:
        """load_and_prepare with a temporary CSV using default global config."""
        from src.services import load_and_prepare

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            _write_minimal_csv(csv_path)

            try:
                df = load_and_prepare(data_path=csv_path, add_temporal=True)
            except AttributeError as exc:
                pytest.fail(
                    f"load_and_prepare raised AttributeError: {exc}. "
                    "This is a regression of the config path bug."
                )

            assert df is not None
            assert len(df) > 0
            assert "target" in df.columns, "add_target_col() should have added 'target'"
            assert df["target"].dtype.name == "int8"

    def test_loads_temp_csv_with_injected_config(self) -> None:
        """load_and_prepare with a temporary CSV using an injected config object."""
        from src.services import load_and_prepare

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            _write_minimal_csv(csv_path)

            mock_cfg = _MockConfig(
                paths=_MockPaths(raw=Path(tmpdir)),
                preprocessing=_MockPreprocessingConfig(normalise_teams=False),
            )

            try:
                df = load_and_prepare(
                    data_path=csv_path,
                    add_temporal=False,
                    config=mock_cfg,
                )
            except AttributeError as exc:
                pytest.fail(
                    f"load_and_prepare with injected config raised AttributeError: {exc}."
                )

            assert df is not None
            assert len(df) > 0
            assert "target" in df.columns

    def test_injected_config_normalise_teams_true(self) -> None:
        """load_and_prepare with normalise_teams=True should not crash."""
        from src.services import load_and_prepare

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            _write_minimal_csv(csv_path)

            mock_cfg = _MockConfig(
                paths=_MockPaths(raw=Path(tmpdir)),
                preprocessing=_MockPreprocessingConfig(normalise_teams=True),
            )

            try:
                df = load_and_prepare(
                    data_path=csv_path,
                    add_temporal=False,
                    config=mock_cfg,
                )
            except AttributeError as exc:
                pytest.fail(
                    f"Injected config with normalise_teams=True raised AttributeError: {exc}"
                )

            assert df is not None
            assert "target" in df.columns

    def test_missing_file_raises_not_found_error(self) -> None:
        """load_and_prepare with a non-existent path should raise an error."""
        from src.services import load_and_prepare
        from src.utils.exceptions import DataNotFoundError

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does_not_exist.csv"

            with pytest.raises((FileNotFoundError, DataNotFoundError)):
                load_and_prepare(data_path=nonexistent)

    def test_empty_csv_returns_empty_dataframe(self) -> None:
        """load_and_prepare with an empty CSV (headers only) returns an empty DataFrame.

        The pipeline should not crash on empty data. The critical regression
        to guard against is AttributeError from the config path bug.
        """
        from src.services import load_and_prepare

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "date",
                        "home_team",
                        "away_team",
                        "result",
                        "home_goals",
                        "away_goals",
                    ]
                )

            try:
                df = load_and_prepare(data_path=csv_path, add_temporal=False)
            except AttributeError as exc:
                pytest.fail(f"Empty CSV raised AttributeError (regression): {exc}")

            # Empty data should result in an empty DataFrame
            if df is not None:
                assert "target" in df.columns
                assert len(df) == 0

    def test_malformed_csv_missing_home_away_columns(self) -> None:
        """CSV missing home_team/away_team columns should not crash with AttributeError."""
        from src.services import load_and_prepare

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "result", "odds"])
                writer.writerow(["2024-01-01", "H", "1.5"])

            try:
                df = load_and_prepare(data_path=csv_path)
            except AttributeError as exc:
                pytest.fail(f"Malformed CSV raised AttributeError (regression): {exc}")
            # Let other exceptions propagate naturally — they may
            # indicate real issues, not the config regression

            # If it succeeds, it should have at least a target column
            if df is not None and len(df) > 0:
                assert "target" in df.columns


# ═══════════════════════════════════════════════════════════
#  resolve_data_path() tests
# ═══════════════════════════════════════════════════════════


class TestResolveDataPath:
    """resolve_data_path should return a Path and handle missing files gracefully."""

    def test_returns_path_for_explicit_hint(self) -> None:
        """Explicit path hint should be returned as-is."""
        from src.services import resolve_data_path

        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "explicit.csv"
            existing.touch()

            result = resolve_data_path(hint=existing)
            assert result == existing

    def test_missing_hint_returns_first_candidate(self) -> None:
        """When no files exist, return the first candidate path without crashing."""
        from src.services import resolve_data_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock config with all paths inside the tmpdir and a non-existent
            # worldcup path so none of the candidates exist
            mock_cfg = _MockConfig(
                paths=_MockPaths(raw=Path(tmpdir)),
            )

            result = resolve_data_path(hint=None, config=mock_cfg)
            # Should return the first candidate path (results_file), not crash
            assert result is not None
            assert isinstance(result, Path)
            assert "results_clean" in str(result)

    def test_discovers_existing_file_in_candidates(self) -> None:
        """resolve_data_path should discover an existing file among candidates."""
        from src.services import resolve_data_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the file that cfg.data.results_file points to
            target = Path(tmpdir) / "results_clean.csv"
            target.touch()

            mock_cfg = _MockConfig(
                paths=_MockPaths(raw=Path(tmpdir)),
            )

            result = resolve_data_path(hint=None, config=mock_cfg)
            assert result == target

    def test_with_default_config_does_not_crash(self) -> None:
        """Calling resolve_data_path with no hint and no config uses global default."""
        from src.services import resolve_data_path

        try:
            result = resolve_data_path()
            assert isinstance(result, Path)
        except AttributeError as exc:
            pytest.fail(
                f"resolve_data_path with default config raised AttributeError: {exc}"
            )

    def test_custom_config_uses_worldcup_path(self) -> None:
        """resolve_data_path should use injected config, not global defaults."""
        from src.services import resolve_data_path

        with tempfile.TemporaryDirectory() as tmpdir:
            wc_path = Path(tmpdir) / "worldcup_all.csv"
            wc_path.touch()

            mock_cfg = _MockConfig(
                paths=_MockPaths(raw=Path(tmpdir)),
                worldcup=_MockWorldCupConfig(data_path=str(wc_path)),
            )

            result = resolve_data_path(hint=None, config=mock_cfg)
            assert result == wc_path


# ═══════════════════════════════════════════════════════════
#  add_target_col() tests
# ═══════════════════════════════════════════════════════════


class TestAddTargetCol:
    """add_target_col should correctly map result codes to numeric targets."""

    def test_maps_home_win(self) -> None:
        """'H' should map to 2."""
        from src.services import add_target_col

        df = pd.DataFrame({"result": ["H"]})
        result = add_target_col(df)
        assert result["target"].iloc[0] == 2

    def test_maps_draw(self) -> None:
        """'D' should map to 1."""
        from src.services import add_target_col

        df = pd.DataFrame({"result": ["D"]})
        result = add_target_col(df)
        assert result["target"].iloc[0] == 1

    def test_maps_away_win(self) -> None:
        """'A' should map to 0."""
        from src.services import add_target_col

        df = pd.DataFrame({"result": ["A"]})
        result = add_target_col(df)
        assert result["target"].iloc[0] == 0

    def test_maps_nan_to_neg_one(self) -> None:
        """Missing/NaN result should map to -1."""
        from src.services import add_target_col

        df = pd.DataFrame({"result": [None, float("nan"), ""]})
        df["result"] = df["result"].astype(object)
        df.loc[1, "result"] = float("nan")
        df.loc[2, "result"] = ""

        result = add_target_col(df)
        assert result["target"].iloc[0] == -1
        assert result["target"].iloc[1] == -1
        assert result["target"].iloc[2] == -1

    def test_is_idempotent(self) -> None:
        """Calling add_target_col twice should not change the result."""
        from src.services import add_target_col

        df = pd.DataFrame({"result": ["H", "D", "A"]})
        once = add_target_col(df)
        twice = add_target_col(once)

        pd.testing.assert_series_equal(once["target"], twice["target"])

    def test_target_column_dtype_is_int8(self) -> None:
        """Target should be int8 for memory efficiency."""
        from src.services import add_target_col

        df = pd.DataFrame({"result": ["H", "D", "A", None]})
        result = add_target_col(df)
        assert result["target"].dtype.name == "int8"

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame should still get the target column."""
        from src.services import add_target_col

        df = pd.DataFrame({"result": pd.Series(dtype="object")})
        result = add_target_col(df)
        assert "target" in result.columns
        assert len(result) == 0
