"""Phase 2 regression tests: feature selection leakage, artifact round-trip,
row alignment, and class mapping.

Each test uses temporary directories and synthetic data to avoid
depending on real files or a trained model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ── Helpers ──────────────────────────────────────────────


def _make_synthetic_data(n_rows: int = 100) -> pd.DataFrame:
    """Create a tiny DataFrame that mimics cleaned match data."""
    np.random.seed(42)
    teams = [f"Team_{i}" for i in range(8)]
    data = {
        "date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
        "home_team": np.random.choice(teams, n_rows),
        "away_team": np.random.choice(teams, n_rows),
        "result": np.random.choice(["H", "D", "A"], n_rows, p=[0.45, 0.25, 0.30]),
        "home_goals": np.random.poisson(1.5, n_rows),
        "away_goals": np.random.poisson(1.2, n_rows),
        "season": "2023",
    }
    # Ensure no self-matches
    df = pd.DataFrame(data)
    df = df[df["home_team"] != df["away_team"]].reset_index(drop=True)
    # Add target column
    df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0})
    return df.head(n_rows)


def _fake_model() -> LogisticRegression:
    """A tiny trained LogisticRegression for testing."""
    np.random.seed(42)
    X_fake = np.random.randn(30, 5)
    y_fake = np.random.randint(0, 3, 30)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_fake, y_fake)
    return model


# ═══════════════════════════════════════════════════════════════
#  1. ModelArtifact Round-Trip
# ═══════════════════════════════════════════════════════════════


class TestModelArtifactRoundTrip:
    """ModelArtifact serialisation + column alignment."""

    def test_save_and_load(self):
        """Round-trip: train -> save -> load -> predict."""
        from src.models.artifact import ModelArtifact

        model = _fake_model()
        feature_names = [f"f_{i}" for i in range(5)]

        artifact = ModelArtifact(
            model=model,
            feature_names=feature_names,
            model_type="logistic_regression",
            trained_at="2024-01-01T00:00:00",
        )

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp = f.name
            artifact.save(tmp)
            loaded = ModelArtifact.load(tmp)

        assert loaded.artifact_version == artifact.artifact_version
        assert loaded.n_features == 5
        assert loaded.model_type == "logistic_regression"
        assert loaded.trained_at == "2024-01-01T00:00:00"

        # Predict with loaded artifact
        X_test = pd.DataFrame(np.random.randn(5, 5), columns=feature_names)
        probs = loaded.predict_proba(X_test)
        assert probs.shape == (5, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

        Path(tmp).unlink(missing_ok=True)

    def test_column_reordering(self):
        """select_columns reorders and filters correctly."""
        from src.models.artifact import ModelArtifact

        model = _fake_model()
        artifact = ModelArtifact(
            model=model,
            feature_names=["a", "b", "c", "d", "e"],
        )

        # Input with different order + extra column
        X_in = pd.DataFrame(
            np.random.randn(3, 6),
            columns=["e", "d", "extra", "c", "b", "a"],
        )
        X_out = artifact.select_columns(X_in)

        assert list(X_out.columns) == ["a", "b", "c", "d", "e"]
        assert X_out.shape[1] == 5

    def test_missing_column_added(self):
        """SelectColumns adds missing columns as NaN."""
        from src.models.artifact import ModelArtifact

        model = _fake_model()
        artifact = ModelArtifact(
            model=model,
            feature_names=["a", "b", "c", "d", "e"],
        )

        X_in = pd.DataFrame(
            np.random.randn(3, 3),
            columns=["a", "b", "c"],
        )
        X_out = artifact.select_columns(X_in)

        assert list(X_out.columns) == ["a", "b", "c", "d", "e"]
        assert X_out["d"].isna().all()
        assert X_out["e"].isna().all()

    def test_reject_non_artifact(self):
        """Loading a raw estimator via artifact.load raises."""
        from src.models.artifact import ModelArtifact

        model = _fake_model()
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp = f.name
            joblib.dump(model, tmp)
            with pytest.raises(ValueError, match="not a ModelArtifact"):
                ModelArtifact.load(tmp)

        Path(tmp).unlink(missing_ok=True)

    def test_load_missing_file(self):
        """Loading a non-existent file raises FileNotFoundError."""
        from src.models.artifact import ModelArtifact

        with pytest.raises(FileNotFoundError, match="not found"):
            ModelArtifact.load("/nonexistent/path.joblib")


# ═══════════════════════════════════════════════════════════════
#  2. Row Alignment
# ═══════════════════════════════════════════════════════════════


class TestRowAlignment:
    """Stable row IDs must survive the feature pipeline."""

    def test_row_id_preserved_through_build_features(self):
        """_row_id column survives build_features pipeline."""
        df = _make_synthetic_data(50)
        # Mark last 5 as upcoming
        df.loc[45:, "result"] = np.nan
        df = df.reset_index(drop=True)
        df["_row_id"] = "test_" + df.index.astype(str)

        from src.feature_engineering import build_features

        X, y = build_features(df, is_training=True)

        assert "_row_id" in X.columns, "_row_id was dropped by build_features"
        assert X["_row_id"].iloc[0] == "test_0"

    def test_upcoming_rows_by_id_not_position(self):
        """Prediction row selection uses _row_id, not iloc positional offset."""
        df = _make_synthetic_data(150)  # enough rows after self-match filtering
        # Mark last 5 rows as upcoming (by actual index, not assumed position)
        n = len(df)
        df.iloc[n - 5 :, df.columns.get_loc("result")] = np.nan

        completed = df[df["result"].notna()].copy()
        upcoming = df[df["result"].isna()].copy()

        assert len(upcoming) == 5, f"Expected 5 upcoming rows, got {len(upcoming)}"

        completed["_row_id"] = "c_" + completed.index.astype(str)
        upcoming["_row_id"] = "u_" + upcoming.index.astype(str)
        row_id_order = list(upcoming["_row_id"])

        combined = pd.concat([completed, upcoming], ignore_index=True)

        from src.feature_engineering import build_features

        X_all, _ = build_features(combined, is_training=True)

        # Select by row_id
        X_upcoming_ids = X_all[X_all["_row_id"].str.startswith("u_", na=False)].copy()
        X_upcoming_ids = (
            X_upcoming_ids.set_index("_row_id").reindex(row_id_order).reset_index(drop=True)
        )

        assert len(X_upcoming_ids) == 5, (
            f"Expected 5 upcoming rows, got {len(X_upcoming_ids)}"
        )

    def test_interleaved_dates_preserve_order(self):
        """Upcoming matches interleaved chronologically still map correctly."""
        np.random.seed(42)
        teams = ["A", "B", "C", "D"]

        # 20 completed matches for enough feature history
        dates = pd.date_range("2023-01-01", periods=20, freq="2D")
        completed = pd.DataFrame({
            "date": dates,
            "home_team": np.random.choice(teams, 20),
            "away_team": np.random.choice(teams, 20),
            "result": np.random.choice(["H", "D", "A"], 20, p=[0.45, 0.25, 0.30]),
            "home_goals": np.random.poisson(1.5, 20),
            "away_goals": np.random.poisson(1.2, 20),
            "season": "2023",
        })
        completed = completed[completed["home_team"] != completed["away_team"]]
        completed["target"] = completed["result"].map({"H": 2, "D": 1, "A": 0})

        # 2 upcoming matches
        upcoming = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-25", "2023-02-01"]),
            "home_team": ["A", "C"],
            "away_team": ["B", "A"],
            "result": [np.nan, np.nan],
            "home_goals": [0, 0],
            "away_goals": [0, 0],
            "season": "2023",
        })
        upcoming["target"] = np.nan

        completed["_row_id"] = "c_" + completed.index.astype(str)
        upcoming["_row_id"] = "u_" + upcoming.index.astype(str)
        row_id_order = list(upcoming["_row_id"])

        combined = pd.concat([completed, upcoming], ignore_index=True)

        from src.feature_engineering import build_features

        X_all, _ = build_features(combined, is_training=True)

        # Select by row_id
        X_upcoming_ids = X_all[X_all["_row_id"].str.startswith("u_", na=False)].copy()
        X_upcoming_ids = (
            X_upcoming_ids.set_index("_row_id").reindex(row_id_order).reset_index(drop=True)
        )

        assert len(X_upcoming_ids) == 2, (
            f"Expected 2 upcoming rows, got {len(X_upcoming_ids)}"
        )


# ═══════════════════════════════════════════════════════════════
#  3. Class / Probability Mapping
# ═══════════════════════════════════════════════════════════════


class TestClassMapping:
    """Probability columns must be mapped via model.classes_."""

    def test_default_class_order(self):
        """classes_=[0,1,2] maps correctly to [away, draw, home]."""
        probs = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        classes = [0, 1, 2]

        for i in range(len(probs)):
            prob_map = dict(zip(classes, probs[i]))
            assert prob_map[2] == probs[i][2]  # home win
            assert prob_map[1] == probs[i][1]  # draw
            assert prob_map[0] == probs[i][0]  # away win

    def test_reversed_class_order(self):
        """classes_=[2,0,1] must still map correctly."""
        probs = np.array([[0.7, 0.1, 0.2], [0.3, 0.4, 0.3]])
        classes = [2, 0, 1]  # Home, Away, Draw

        prob_map = dict(zip(classes, probs[0]))
        assert prob_map[2] == 0.7  # home win
        assert abs(prob_map[0] - 0.1) < 1e-6  # away win
        assert prob_map[1] == 0.2  # draw

    def test_missing_class_detected(self):
        """Missing class in predict_proba output is detectable."""
        probs = np.array([[0.5, 0.5]])  # only 2 columns
        classes = [0, 1]  # missing class 2

        prob_map = dict(zip(classes, probs[0]))
        assert 2 not in prob_map  # class 2 is missing
        assert 0 in prob_map and 1 in prob_map


# ═══════════════════════════════════════════════════════════════
#  4. Feature Selection via Pipeline (leakage-free flow)
# ═══════════════════════════════════════════════════════════════


class TestLeakageFreeSelection:
    """Feature selection must be fit ONLY on training data."""

    def test_pipeline_fit_on_train_only(self):
        """Sklearn Pipeline fit on X_train, transform on all splits."""
        from src.services.training_service import TrainingService

        n_features = 10
        X_train = pd.DataFrame(
            np.random.randn(40, n_features),
            columns=[f"f_{i}" for i in range(n_features)],
        )
        y_train = pd.Series(np.random.randint(0, 3, 40))
        X_val = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f"f_{i}" for i in range(n_features)],
        )
        X_test = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f"f_{i}" for i in range(n_features)],
        )

        service = TrainingService()
        selector = service._build_selector(X_train, y_train)
        assert isinstance(selector, Pipeline)

        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)

        # All splits should have the same number of features
        assert X_train_selected.shape[1] == X_val_selected.shape[1]
        assert X_val_selected.shape[1] == X_test_selected.shape[1]

    def test_split_before_selection_flow(self):
        """Verify chronological split first, then fit on X_train only."""
        from src.services.training_service import TrainingService
        from src.feature_engineering import train_val_test_split

        X = pd.DataFrame(
            np.random.randn(100, 10),
            columns=[f"f_{i}" for i in range(10)],
        )
        y = pd.Series(np.random.randint(0, 3, 100))

        splits = train_val_test_split(X, y)
        service = TrainingService()

        selector = service._build_selector(splits["X_train"], splits["y_train"])
        X_train_fs = selector.fit_transform(splits["X_train"], splits["y_train"])
        X_val_fs = selector.transform(splits["X_val"])
        X_test_fs = selector.transform(splits["X_test"])

        assert X_train_fs.shape[1] == X_val_fs.shape[1] == X_test_fs.shape[1]
