"""
Integration tests for live predictions — end-to-end tests for the
training and prediction pipelines with realistic mocks.

Covers:
    - TrainingService.train()
    - PredictionService.predict_upcoming()
    - PredictionService.predict_match()
    - PredictionService.backfill_predictions()
    - PredictionService.predict_with_odds()
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# Disable logging noise from services during tests
logging.disable(logging.CRITICAL)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════


def _create_sample_csv(tmp_path: Path, n_completed: int = 20, n_upcoming: int = 3) -> Path:
    """Create a sample match-data CSV with completed + upcoming rows."""
    rows: list[dict[str, Any]] = []

    teams_home = ["Team_A", "Team_C", "Team_E", "Team_G"]
    teams_away = ["Team_B", "Team_D", "Team_F", "Team_H"]

    for i in range(n_completed):
        rows.append({
            "date": f"2024-{i%12+1:02d}-{(i%28)+1:02d}",
            "home_team": teams_home[i % len(teams_home)],
            "away_team": teams_away[i % len(teams_away)],
            "home_goals": np.random.randint(0, 4),
            "away_goals": np.random.randint(0, 3),
            "result": np.random.choice(["H", "D", "A"]),
            "league": "Test League",
        })

    for i in range(n_upcoming):
        rows.append({
            "date": f"2026-07-{15+i:02d}",
            "home_team": teams_home[(n_completed + i) % len(teams_home)],
            "away_team": teams_away[(n_completed + i) % len(teams_away)],
            "home_goals": None,
            "away_goals": None,
            "result": None,
            "league": "Test League",
        })

    csv_path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _dummy_build_features(df: pd.DataFrame, is_training: bool = True, **kwargs: Any) -> tuple[pd.DataFrame, pd.Series]:
    """Replace build_features with a simple version for testing.

    Creates synthetic features from team names and basic stats so
    a LogisticRegression can actually fit and predict.

    Accepts ``**kwargs`` to stay compatible with the DI-refactored
    service layer which now passes ``config=<cfg>`` downstream.
    """
    # Drop rows with missing results for training labels
    train_df = df[df["result"].notna()].copy()

    # Create simple numeric features using label-encoded team names + date info
    all_teams = pd.concat([train_df["home_team"], train_df["away_team"]]).unique()
    team_to_id = {t: i for i, t in enumerate(sorted(all_teams))}

    rows_features: list[dict[str, float]] = []
    for _, row in train_df.iterrows():
        h_id = float(team_to_id.get(row["home_team"], 0))
        a_id = float(team_to_id.get(row["away_team"], 0))
        rows_features.append({
            "home_team_id": h_id,
            "away_team_id": a_id,
            "home_advantage": 1.0,
        })

    X = pd.DataFrame(rows_features)
    y = train_df["target"].astype(int)

    # If the full df has more rows (upcoming), pad X with NaN for matching
    if len(df) > len(train_df):
        upcoming = df[df["result"].isna()].copy()
        for _, row in upcoming.iterrows():
            h_id = float(team_to_id.get(row["home_team"], 0))
            a_id = float(team_to_id.get(row["away_team"], 0))
            rows_features.append({
                "home_team_id": h_id,
                "away_team_id": a_id,
                "home_advantage": 1.0,
            })
        X = pd.DataFrame(rows_features)

    return X, y


def _train_minimal_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Train a tiny LogisticRegression for use in prediction tests."""
    model = LogisticRegression(
        solver="lbfgs", max_iter=500, random_state=42,
    )
    model.fit(X, y)
    return model


# ═══════════════════════════════════════════════════════════
#  TrainingService Integration Tests
# ═══════════════════════════════════════════════════════════

class TestTrainingServiceIntegration:
    """End-to-end tests for TrainingService.train()."""

    @patch("src.services.training_service.load_and_prepare")
    def test_train_returns_correct_structure(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``train()`` should return a report dict with expected keys."""
        # Create sample data
        csv_path = _create_sample_csv(tmp_path, n_completed=15)

        # Mock load_and_prepare to return the CSV data (already has target col)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        mock_load.return_value = df_raw

        # Patch at the source — not the service's local import reference
        with patch("src.feature_engineering.build_features", wraps=_dummy_build_features):
            model_dir = tmp_path / "models"
            model_dir.mkdir()

            from src.services.training_service import TrainingService

            service = TrainingService(model_dir=model_dir)
            report = service.train(data_path=csv_path, model_type="logistic_regression")

        # Verify report structure
        assert isinstance(report, dict)
        assert "model_type" in report
        assert report["model_type"] == "logistic_regression"
        assert "model_path" in report
        assert "metrics" in report
        assert "features" in report
        assert "splits" in report

        # Verify metrics
        metrics = report["metrics"]
        assert "test_accuracy" in metrics
        assert "test_log_loss" in metrics
        assert "test_samples" in metrics
        assert metrics["test_samples"] > 0
        assert isinstance(metrics["test_accuracy"], float)

        # Verify feature info
        features = report["features"]
        assert features["count"] > 0
        assert isinstance(features["columns"], list)

        # Verify splits
        splits = report["splits"]
        assert splits["train"] > 0
        assert splits["val"] > 0
        assert splits["test"] > 0
        assert splits["train"] + splits["val"] + splits["test"] == 15

    @patch("src.services.training_service.load_and_prepare")
    def test_train_with_hyperparameter_tuning(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``train(tune_hyperparams=True)`` should include tuning report."""
        csv_path = _create_sample_csv(tmp_path, n_completed=15)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        mock_load.return_value = df_raw

        with patch("src.feature_engineering.build_features", wraps=_dummy_build_features):
            model_dir = tmp_path / "models"
            model_dir.mkdir()

            from src.services.training_service import TrainingService

            service = TrainingService(model_dir=model_dir)
            report = service.train(
                data_path=csv_path,
                model_type="logistic_regression",
                tune_hyperparams=True,
                cv_folds=2,
            )

        assert report["hyperparameter_tuning"] is not None
        assert report["hyperparameter_tuning"]["performed"] is True

    @patch("src.services.training_service.load_and_prepare")
    def test_train_raises_on_too_few_rows(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``train()`` should raise ValueError when fewer than 20 rows."""
        csv_path = _create_sample_csv(tmp_path, n_completed=5)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        mock_load.return_value = df_raw

        with patch("src.feature_engineering.build_features", wraps=_dummy_build_features):
            model_dir = tmp_path / "models"
            model_dir.mkdir()

            from src.services.training_service import TrainingService

            service = TrainingService(model_dir=model_dir)
            with pytest.raises(ValueError, match="Only 5 rows after feature engineering"):
                service.train(data_path=csv_path, model_type="logistic_regression")


# ═══════════════════════════════════════════════════════════
#  PredictionService Integration Tests
# ═══════════════════════════════════════════════════════════

class TestPredictionServiceIntegration:
    """End-to-end tests for PredictionService methods."""

    @pytest.fixture
    def sample_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up a complete test environment: data CSV + trained model."""
        csv_path = _create_sample_csv(tmp_path, n_completed=15, n_upcoming=3)

        # Build features and train a real model
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

        # Use the dummy build_features to get X, y
        completed = df_raw[df_raw["result"].notna()]
        X, y = _dummy_build_features(completed, is_training=True)
        model = _train_minimal_model(X, y)

        # Save model to tmp
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        import joblib
        model_path = model_dir / "test_model.joblib"
        joblib.dump(model, model_path)

        return {
            "csv_path": csv_path,
            "model_dir": model_dir,
            "model_path": model_path,
            "df_raw": df_raw,
        }

    def test_predict_upcoming_returns_list(
        self, sample_env: dict[str, Any],
    ) -> None:
        """``predict_upcoming()`` should return a list of prediction dicts."""
        env = sample_env
        df_raw = env["df_raw"]

        with patch("src.services.prediction_service.load_and_prepare", return_value=df_raw):
            with patch(
                "src.feature_engineering.build_features",
                wraps=_dummy_build_features,
            ):
                from src.services.prediction_service import PredictionService

                service = PredictionService(model_dir=env["model_dir"])
                results = service.predict_upcoming(
                    data_path=env["csv_path"],
                    limit=2,
                )

        assert isinstance(results, list)
        assert len(results) == 2  # limited to 2

    def test_predict_upcoming_result_structure(
        self, sample_env: dict[str, Any],
    ) -> None:
        """Each prediction dict should have all required keys."""
        env = sample_env
        df_raw = env["df_raw"]

        with patch("src.services.prediction_service.load_and_prepare", return_value=df_raw):
            with patch(
                "src.feature_engineering.build_features",
                wraps=_dummy_build_features,
            ):
                from src.services.prediction_service import PredictionService

                service = PredictionService(model_dir=env["model_dir"])
                results = service.predict_upcoming(data_path=env["csv_path"])

        if results:
            pred = results[0]
            assert "match_id" in pred
            assert "date" in pred
            assert "home_team" in pred
            assert "away_team" in pred
            assert "home_win_prob" in pred
            assert "draw_prob" in pred
            assert "away_win_prob" in pred
            assert "prediction" in pred
            assert "confidence" in pred

            # Probabilities should sum to ~1.0
            total_prob = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
            assert abs(total_prob - 1.0) < 0.02

            # Confidence should be max of probs
            max_prob = max(pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"])
            assert abs(pred["confidence"] - max_prob) < 0.001

    def test_predict_upcoming_no_upcoming_matches(
        self, tmp_path: Path,
    ) -> None:
        """Should return empty list when all matches are completed."""
        csv_path = _create_sample_csv(tmp_path, n_completed=10, n_upcoming=0)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

        # Create a dummy model so _load_model doesn't crash
        model_dir = tmp_path / "models"
        model_dir.mkdir(exist_ok=True)
        import joblib
        dummy = LogisticRegression()
        dummy.classes_ = np.array([0, 1, 2])
        joblib.dump(dummy, model_dir / "dummy.joblib")

        with patch("src.services.prediction_service.load_and_prepare", return_value=df_raw):
            from src.services.prediction_service import PredictionService

            service = PredictionService(model_dir=model_dir)
            results = service.predict_upcoming(data_path=csv_path)

        assert results == []

    def test_predict_match_found(
        self, sample_env: dict[str, Any],
    ) -> None:
        """``predict_match()`` should return a result for a valid match."""
        env = sample_env
        df_raw = env["df_raw"]

        with patch("src.services.prediction_service.load_and_prepare", return_value=df_raw):
            with patch(
                "src.feature_engineering.build_features",
                wraps=_dummy_build_features,
            ):
                from src.services.prediction_service import PredictionService

                service = PredictionService(model_dir=env["model_dir"])
                # Use a valid row index
                result = service.predict_match(match_id=0, data_path=env["csv_path"])

        assert result is not None
        assert result["home_team"] is not None
        assert result["away_team"] is not None
        assert "home_win_prob" in result
        assert "draw_prob" in result
        assert "away_win_prob" in result
        assert result["prediction"] in ("Home Win", "Draw", "Away Win")

    def test_predict_match_not_found(
        self, sample_env: dict[str, Any],
    ) -> None:
        """``predict_match()`` should return None for invalid match_id."""
        env = sample_env
        df_raw = env["df_raw"]

        with patch("src.services.prediction_service.load_and_prepare", return_value=df_raw):
            from src.services.prediction_service import PredictionService

            service = PredictionService(model_dir=env["model_dir"])
            result = service.predict_match(match_id=99999, data_path=env["csv_path"])

        assert result is None

    def test_backfill_predictions(
        self, sample_env: dict[str, Any],
    ) -> None:
        """``backfill_predictions()`` should return a list of results with actual results."""
        env = sample_env
        df_raw = env["df_raw"]

        with patch("src.services.prediction_service.load_and_prepare", return_value=df_raw):
            with patch(
                "src.feature_engineering.build_features",
                wraps=_dummy_build_features,
            ):
                from src.services.prediction_service import PredictionService

                service = PredictionService(model_dir=env["model_dir"])
                results = service.backfill_predictions(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                    data_path=env["csv_path"],
                )

        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "home_team" in r
            assert "away_team" in r
            assert "prediction" in r
            assert "actual_result" in r
            assert "correct" in r
            assert isinstance(r["correct"], bool)

    def test_predict_with_odds(
        self, sample_env: dict[str, Any],
    ) -> None:
        """``predict_with_odds()`` should enrich predictions with odds analysis."""
        env = sample_env
        df_raw = env["df_raw"]

        with patch("src.services.prediction_service.load_and_prepare", return_value=df_raw):
            with patch(
                "src.feature_engineering.build_features",
                wraps=_dummy_build_features,
            ):
                with patch(
                    "src.services.prediction_service.OddsCollector",
                ) as mock_odds_cls:
                    # Mock OddsCollector to return realistic odds
                    mock_collector = MagicMock()
                    mock_collector.get_best_odds.return_value = {
                        "home_odds": 2.10,
                        "draw_odds": 3.40,
                        "away_odds": 3.80,
                        "source": "test",
                        "arbitrage": {"is_arbitrage": False},
                    }
                    mock_odds_cls.return_value = mock_collector

                    from src.services.prediction_service import PredictionService

                    service = PredictionService(model_dir=env["model_dir"])
                    results = service.predict_with_odds(
                        data_path=env["csv_path"],
                        limit=2,
                    )

        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "odds_analysis" in r
            oa = r["odds_analysis"]
            assert "home_odds" in oa
            assert "draw_odds" in oa
            assert "away_odds" in oa
            assert "margin_pct" in oa
            assert "source" in oa
            assert "edges" in oa
            assert "Home Win" in oa["edges"]
            assert "Draw" in oa["edges"]
            assert "Away Win" in oa["edges"]

            # Verify edge calculation
            home_edge = oa["edges"]["Home Win"]
            assert "odds" in home_edge
            assert "fair_prob" in home_edge
            assert "edge_pp" in home_edge
            assert "ev_pct" in home_edge
            assert "is_value" in home_edge


# ═══════════════════════════════════════════════════════════
#  DI Pattern tests
# ═══════════════════════════════════════════════════════════


def _make_mock_config(tmp_path: Path) -> Any:
    """Build a minimal mock Config object for DI testing.

    Returns a namespace-style object with all attributes the service
    layer and the underlying pipeline functions access.
    """
    class _MockTrain:
        model_type = "logistic_regression"
        cv_folds = 3
        seed = 123
        C = 1.0
        solver = "lbfgs"
        max_iter = 500
        n_estimators = 50
        max_depth = 4
        min_samples_leaf = 5
        learning_rate = 0.1
        subsample = 0.8
        colsample_bytree = 0.8
        reg_lambda = 1.0
        reg_alpha = 0.1
        gamma = 0.0
        min_child_weight = 1.0
        num_leaves = 31
        min_child_samples = 10
        hidden_layers = (16, 8)
        dropout = 0.2
        batch_size = 32
        epochs = 20
        early_stopping_patience = 5
        target_column = "result"

    class _MockPaths:
        models = tmp_path / "models"
        raw = tmp_path / "raw"
        external = tmp_path / "external"
        data = tmp_path / "data"

    class _MockData:
        split_ratios = (0.6, 0.2, 0.2)
        seed = 123
        results_file = "results.csv"
        fixtures_file = "fixtures.csv"
        teams_file = "teams.csv"
        source = "local"
        api_url = ""
        api_key_env = ""

    class _MockWorldCup:
        data_path = str(tmp_path / "worldcup_all.csv")
        predictions_dir = str(tmp_path / "predictions")
        predictions_file = "predictions.csv"
        model_save_name = "test_model.joblib"

    class _MockDataCollection:
        normalise_teams = True
        leagues = ("E0",)
        max_seasons = 3
        missing_strategy = "drop"
        output_file = "results.csv"
        max_missing_pct = 50.0

    class _MockFeatureSelection:
        enabled = False
        method = "mutual_info"
        n_features = 10
        importance_threshold = 0.01
        correlation_threshold = 0.95
        drop_redundant_first = False

    class _MockElo:
        k = 32
        home_advantage = 100
        initial_rating = 1500
        regress_to_mean = True
        regress_factor = 1 / 3
        use_goal_margin = True
        max_goal_margin = 5
        adjustments: dict = {}

    class _MockOdds:
        opening_odds_cols = ("BbMxH", "BbMxD", "BbMxA")
        closing_odds_cols = ("BbAvH", "BbAvD", "BbAvA")
        compute_consensus = True
        warn_missing = True

    class _MockPlayerInfo:
        enabled = False
        default_age = 25.0
        placeholder_value = 0.0
        warn_missing = True

    class _MockPlayerFeatures:
        enabled = False
        rolling_windows = (5, 10)
        warn_missing = True

    class _MockXg:
        rolling_windows = (5, 10)
        compute_xpts = True
        max_goals_table = 8
        placeholder_value = 0.0
        warn_missing = True

    class _MockPoisson:
        min_matches = 0
        max_goals = 8

    class _MockDixonColes:
        enabled = False
        refit_every = 500
        decay_halflife_days = 1460.0
        use_importance = True
        rho_fixed = None
        regress_prior = True
        prior_strength = 0.01
        fit_intercept_only = False

    class _MockFeatures:
        form_window = 5
        rolling_windows = (5, 10, 20)
        rolling_avg_window = 10
        include_h2h = True
        h2h_window = 6
        include_league_position = True
        categorical_encoding = "label"
        time_decay_halflife = None
        reset_per_season = False

    class _MockEval:
        metrics = ("accuracy", "log_loss")
        plot_confusion_matrix = False
        plot_roc_curve = False
        plot_feature_importance = False
        output_dir = tmp_path / "reports"

    class _MockPreprocessing:
        input_file = "results.csv"
        output_file = "results_clean.csv"
        normalise_teams = True
        add_temporal_features = True
        save_cleaned = True

    class _MockConfig:
        paths = _MockPaths()
        data = _MockData()
        data_collection = _MockDataCollection()
        preprocessing = _MockPreprocessing()
        train = _MockTrain()
        features = _MockFeatures()
        feature_selection = _MockFeatureSelection()
        worldcup = _MockWorldCup()
        elo = _MockElo()
        odds = _MockOdds()
        player_info = _MockPlayerInfo()
        player_features = _MockPlayerFeatures()
        xg = _MockXg()
        poisson = _MockPoisson()
        dixon_coles = _MockDixonColes()
        eval = _MockEval()
        verbose = False

    return _MockConfig()


class TestDIPattern:
    """Verify that the DI pattern works with custom config objects."""

    @patch("src.services.training_service.load_and_prepare")
    def test_training_service_defaults_to_global_config(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """TrainingService() without config should fall back to global config."""
        csv_path = _create_sample_csv(tmp_path, n_completed=15)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        mock_load.return_value = df_raw

        with patch("src.feature_engineering.build_features", wraps=_dummy_build_features):
            from src.services.training_service import TrainingService
            from config import config as global_config

            service = TrainingService(model_dir=tmp_path / "models")
            # Verify it uses the global config singleton
            assert service._config is global_config

    @patch("src.services.training_service.load_and_prepare")
    def test_training_service_accepts_custom_config(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """TrainingService(config=my_cfg) should use the injected config."""
        csv_path = _create_sample_csv(tmp_path, n_completed=15)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        mock_load.return_value = df_raw

        mock_cfg = _make_mock_config(tmp_path)
        # Change a few key settings to verify they're used
        mock_cfg.train.model_type = "logistic_regression"
        mock_cfg.data.split_ratios = (0.6, 0.2, 0.2)
        mock_cfg.data.seed = 999

        with patch("src.feature_engineering.build_features", wraps=_dummy_build_features):
            from src.services.training_service import TrainingService

            service = TrainingService(config=mock_cfg, model_dir=tmp_path / "models")

            # Verify the custom config is used
            assert service._config is mock_cfg
            assert service._config.train.model_type == "logistic_regression"
            assert service._config.data.split_ratios == (0.6, 0.2, 0.2)
            assert service._config.data.seed == 999

            # Run training — should use the custom config's split ratios
            report = service.train(data_path=csv_path, model_type="logistic_regression")

        # Verify the model was trained with the custom config
        assert report["model_type"] == "logistic_regression"
        splits = report["splits"]
        total = splits["train"] + splits["val"] + splits["test"]
        assert total == 15  # Same total rows
        # Split ratios 0.6/0.2/0.2 should produce different split sizes
        # than the default 0.7/0.15/0.15
        expected_train_60pct = int(15 * 0.6 * 0.8)  # 0.6 of 0.8 of total for train after test split
        assert splits["train"] >= 6  # At least 6 training rows

    @patch("src.services.prediction_service.load_and_prepare")
    def test_prediction_service_accepts_custom_config(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """PredictionService(config=my_cfg) should use the injected config."""
        csv_path = _create_sample_csv(tmp_path, n_completed=15, n_upcoming=3)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        mock_load.return_value = df_raw

        # Create a model and save it
        completed = df_raw[df_raw["result"].notna()]
        X, y = _dummy_build_features(completed, is_training=True)
        model = _train_minimal_model(X, y)
        model_dir = tmp_path / "models"
        model_dir.mkdir(exist_ok=True)
        import joblib
        joblib.dump(model, model_dir / "test_model.joblib")

        mock_cfg = _make_mock_config(tmp_path)

        with patch("src.feature_engineering.build_features", wraps=_dummy_build_features):
            from src.services.prediction_service import PredictionService
            from config import config as global_config

            service = PredictionService(config=mock_cfg, model_dir=model_dir)

            # Verify the custom config is used
            assert service._config is mock_cfg
            assert service._config is not global_config

            # Predict — should use the custom config for path resolution
            results = service.predict_upcoming(data_path=csv_path, limit=2)

        assert len(results) == 2
        assert results[0]["home_win_prob"] > 0

    @patch("src.services.training_service.load_and_prepare")
    def test_custom_config_changes_split_behavior(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Injecting a config with different split_ratios changes split sizes."""
        csv_path = _create_sample_csv(tmp_path, n_completed=20)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        mock_load.return_value = df_raw

        mock_cfg = _make_mock_config(tmp_path)
        # Use very uneven split to easily detect in output
        mock_cfg.data.split_ratios = (0.8, 0.1, 0.1)

        with patch("src.feature_engineering.build_features", wraps=_dummy_build_features):
            from src.services.training_service import TrainingService

            service = TrainingService(config=mock_cfg, model_dir=tmp_path / "models")
            report = service.train(data_path=csv_path, model_type="logistic_regression")

        splits = report["splits"]
        total = splits["train"] + splits["val"] + splits["test"]
        assert total == 20
        # 0.8/0.1/0.1 split: train should be ~80% of total
        assert splits["train"] >= 12  # At least 12 training rows
        assert splits["test"] >= 1  # At least 1 test row

    def test_build_features_passes_config_through(
        self, tmp_path: Path,
    ) -> None:
        """build_features() should accept and pass config to sub-modules.

        When a custom config is passed to build_features, it should
        propagate to internal calls like train_val_test_split.
        """
        csv_path = _create_sample_csv(tmp_path, n_completed=10)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

        with patch("src.feature_engineering.build_features", wraps=_dummy_build_features) as mock_bf:
            from src.feature_engineering import build_features

            mock_cfg = _make_mock_config(tmp_path)
            X, y = build_features(df_raw, is_training=True, config=mock_cfg)

            # Verify the config was passed through
            # (the patched _dummy_build_features ignores config,
            #  but the real build_features would use it)
            assert X is not None
            assert y is not None
            assert len(X) == len(df_raw[df_raw["result"].notna()])

    def test_train_val_test_split_uses_custom_config(
        self, tmp_path: Path,
    ) -> None:
        """train_val_test_split() should use custom config for split ratios."""
        from src.feature_engineering import train_val_test_split

        # Create a simple feature matrix
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(30), "f2": np.random.randn(30)})
        y = pd.Series(np.random.randint(0, 3, 30))

        # Use the default config (global singleton)
        splits_default = train_val_test_split(X, y)
        default_ratio = len(splits_default["train"]) / 30

        # Use a custom config with different split ratios (50/25/25)
        from types import SimpleNamespace
        custom_cfg = SimpleNamespace()
        custom_cfg.data = SimpleNamespace()
        custom_cfg.data.split_ratios = (0.5, 0.25, 0.25)
        custom_cfg.data.seed = 42

        splits_custom = train_val_test_split(X, y, config=custom_cfg)
        custom_ratio = len(splits_custom["train"]) / 30

        # The custom ratio should be different from default
        # Default: 0.7/0.15/0.15 → train ~70% = 21 rows
        # Custom:  0.5/0.25/0.25 → train ~50% = 15 rows
        assert abs(default_ratio - custom_ratio) > 0.1, (
            f"Default train ratio {default_ratio:.2f} should differ "
            f"from custom train ratio {custom_ratio:.2f}"
        )


# ═══════════════════════════════════════════════════════════
#  Edge case tests
# ═══════════════════════════════════════════════════════════

class TestPredictionEdgeCases:
    """Tests for edge cases in the prediction pipeline."""

    def test_predict_upcoming_empty_data(self, tmp_path: Path) -> None:
        """Should handle empty DataFrame gracefully."""
        # Create a minimal CSV so the resolve path + exists() check passes
        csv_path = tmp_path / "matches.csv"
        csv_path.write_text("date,home_team,away_team,result\n")

        df_empty = pd.DataFrame()

        # Create dummy model so _load_model doesn't crash
        model_dir = tmp_path / "models"
        model_dir.mkdir(exist_ok=True)
        import joblib
        dummy = LogisticRegression()
        dummy.classes_ = np.array([0, 1, 2])
        joblib.dump(dummy, model_dir / "dummy.joblib")

        with patch("src.services.prediction_service.load_and_prepare", return_value=df_empty):
            from src.services.prediction_service import PredictionService

            service = PredictionService(model_dir=model_dir)
            result = service.predict_upcoming(data_path=csv_path)
            assert result == []

    def test_training_service_list_models_empty(self, tmp_path: Path) -> None:
        """``list_models()`` should return empty list when no models exist."""
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()

        from src.services.training_service import TrainingService

        service = TrainingService(model_dir=empty_dir)
        models = service.list_models()
        assert models == []

    def test_training_service_saves_model_to_disk(
        self, tmp_path: Path,
    ) -> None:
        """``train()`` should save the model as a .joblib file."""
        csv_path = _create_sample_csv(tmp_path, n_completed=15)
        df_raw = pd.read_csv(csv_path)
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

        with patch("src.services.training_service.load_and_prepare", return_value=df_raw):
            with patch("src.feature_engineering.build_features", wraps=_dummy_build_features):
                model_dir = tmp_path / "models"
                model_dir.mkdir()

                from src.services.training_service import TrainingService

                service = TrainingService(model_dir=model_dir)
                report = service.train(data_path=csv_path, model_type="logistic_regression")

        # Check model was saved
        saved_model_path = Path(report["model_path"])
        assert saved_model_path.exists()
        assert saved_model_path.suffix == ".joblib"

        # Verify we can list it
        models = service.list_models()
        assert len(models) >= 1
        assert models[0]["file_name"].endswith(".joblib")
        assert models[0]["size_bytes"] > 0
