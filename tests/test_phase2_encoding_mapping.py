"""Phase 2 tests: target encoding leakage and class mapping.

Each test uses synthetic data to verify deterministic behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
#  1. SafeTargetEncoder Tests
# ═══════════════════════════════════════════════════════════════


class TestSafeTargetEncoder:
    """SafeTargetEncoder must respect training-only priors."""

    def test_fit_stores_category_means(self):
        """Encoder stores per-category means from fit data."""
        from src.features.encoding import SafeTargetEncoder

        X = pd.DataFrame({
            "home_team": ["A", "A", "B", "B", "C"],
            "away_team": ["B", "C", "A", "C", "A"],
        })
        y = pd.Series([2, 2, 1, 0, 2])  # target values

        encoder = SafeTargetEncoder(cols=["home_team", "away_team"])
        encoder.fit(X, y)

        assert encoder._fitted is True
        assert "home_team" in encoder._category_means
        assert "away_team" in encoder._category_means
        # Team A home mean = (2 + 2) / 2 = 2.0
        assert abs(encoder._category_means["home_team"]["A"] - 2.0) < 0.01
        # Team B home mean = (1 + 0) / 2 = 0.5
        assert abs(encoder._category_means["home_team"]["B"] - 0.5) < 0.01

    def test_unseen_category_uses_prior(self):
        """Unseen categories get the training-only global prior."""
        from src.features.encoding import SafeTargetEncoder

        X_train = pd.DataFrame({
            "home_team": ["A", "A", "B"],
            "away_team": ["B", "C", "A"],
        })
        y_train = pd.Series([2, 1, 0])

        encoder = SafeTargetEncoder(cols=["home_team"])
        encoder.fit(X_train, y_train)
        # prior = (2 + 1 + 0) / 3 = 1.0

        X_test = pd.DataFrame({"home_team": ["D"]})  # D unseen
        X_encoded = encoder.transform(X_test)

        assert "home_team_encoded" in X_encoded.columns
        assert abs(X_encoded["home_team_encoded"].iloc[0] - 1.0) < 0.01, (
            f"Expected prior=1.0, got {X_encoded['home_team_encoded'].iloc[0]}"
        )

    def test_prior_not_leaked_from_future_labels(self):
        """Prior is computed ONLY from fit() data, not transform() data."""
        from src.features.encoding import SafeTargetEncoder

        # Training data: teams A, B, C
        X_train = pd.DataFrame({"team": ["A", "A", "B", "B", "C"]})
        y_train = pd.Series([2, 2, 1, 0, 2])
        # Training prior = (2+2+1+0+2)/5 = 1.4

        encoder = SafeTargetEncoder(cols=["team"])
        encoder.fit(X_train, y_train)
        prior = encoder.prior
        assert abs(prior - 1.4) < 0.01

        # Test data has completely different distribution
        X_test = pd.DataFrame({"team": ["D", "E"]})
        y_test = pd.Series([0, 0])  # mean would be 0.0 but this should NOT leak

        X_encoded = encoder.transform(X_test)
        # D and E should get prior=1.4 NOT the test mean of 0.0
        assert abs(X_encoded["team_encoded"].iloc[0] - 1.4) < 0.01

    def test_transform_without_fit_raises(self):
        """Calling transform before fit raises RuntimeError."""
        from src.features.encoding import SafeTargetEncoder

        encoder = SafeTargetEncoder()
        X = pd.DataFrame({"home_team": ["A"]})
        with pytest.raises(RuntimeError, match="before fit"):
            encoder.transform(X)

    def test_fit_transform_e2e(self):
        """fit_transform returns correctly encoded DataFrame."""
        from src.features.encoding import SafeTargetEncoder

        X = pd.DataFrame({
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "C"],
        })
        y = pd.Series([2, 0, 1])

        encoder = SafeTargetEncoder(cols=["home_team", "away_team"])
        X_enc = encoder.fit_transform(X, y)

        # Original columns dropped, encoded columns added
        assert "home_team" not in X_enc.columns
        assert "away_team" not in X_enc.columns
        assert "home_team_encoded" in X_enc.columns
        assert "away_team_encoded" in X_enc.columns

    def test_get_state_and_from_state_round_trip(self):
        """Encoder state can be serialized and restored."""
        from src.features.encoding import SafeTargetEncoder

        X = pd.DataFrame({"home_team": ["A", "B", "A"]})
        y = pd.Series([2, 0, 1])

        encoder = SafeTargetEncoder(cols=["home_team"])
        encoder.fit(X, y)

        state = encoder.get_state()
        restored = SafeTargetEncoder.from_state(state)

        assert restored._fitted is True
        assert abs(restored.prior - 1.0) < 0.01
        assert "A" in restored._category_means["home_team"]

    def test_encode_categoricals_accepts_encoder(self):
        """_encode_categoricals uses pre-fitted encoder when provided."""
        from src.features.encoding import SafeTargetEncoder, _encode_categoricals

        # Fit encoder on training data
        X_train = pd.DataFrame({"home_team": ["A", "B"]})
        y_train = pd.Series([2, 0])
        encoder = SafeTargetEncoder(cols=["home_team"]).fit(X_train, y_train)

        # Inference data
        X_test = pd.DataFrame({"home_team": ["C"]})  # C unseen
        X_enc = _encode_categoricals(X_test, encoder=encoder)

        assert "home_team_encoded" in X_enc.columns
        # C gets prior = (2+0)/2 = 1.0
        assert abs(X_enc["home_team_encoded"].iloc[0] - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════
#  2. Class Mapping Tests
# ═══════════════════════════════════════════════════════════════


class TestClassMapping:
    """Probability columns must be mapped via model.classes_."""

    def test_default_classes(self):
        """classes_=[0,1,2] maps correctly."""
        from src.services.prediction_service import _resolve_class_mapping, _probs_by_class

        class FakeModel:
            classes_ = [0, 1, 2]

        classes, label_map = _resolve_class_mapping(FakeModel())
        assert classes == [0, 1, 2]
        assert label_map[0] == "Away Win"
        assert label_map[2] == "Home Win"

        probs = np.array([0.1, 0.2, 0.7])
        prob_map = _probs_by_class(FakeModel(), probs)
        assert abs(prob_map[2] - 0.7) < 0.01
        assert abs(prob_map[0] - 0.1) < 0.01

    def test_reversed_classes(self):
        """classes_=[2,0,1] maps correctly via dict(zip(classes, probs))."""
        from src.services.prediction_service import _resolve_class_mapping, _probs_by_class

        class FakeModel:
            classes_ = [2, 0, 1]  # Home, Away, Draw

        classes, _ = _resolve_class_mapping(FakeModel())
        assert classes == [2, 0, 1]

        probs = np.array([0.7, 0.1, 0.2])
        prob_map = _probs_by_class(FakeModel(), probs)
        assert abs(prob_map[2] - 0.7) < 0.01  # Home win
        assert abs(prob_map[0] - 0.1) < 0.01  # Away win
        assert abs(prob_map[1] - 0.2) < 0.01  # Draw

    def test_no_classes_fallback(self):
        """No classes_ attribute falls back to [0,1,2]."""
        from src.services.prediction_service import _resolve_class_mapping, _probs_by_class

        class FakeModel:
            pass

        classes, label_map = _resolve_class_mapping(FakeModel())
        assert classes == [0, 1, 2]

        probs = np.array([0.1, 0.2, 0.7])
        prob_map = _probs_by_class(FakeModel(), probs)
        assert abs(prob_map[2] - 0.7) < 0.01

    def test_ensemble_wrapper_classes(self):
        """Custom wrapper with classes_ works correctly."""
        from src.services.prediction_service import _resolve_class_mapping, _probs_by_class

        class EnsembleWrapper:
            classes_ = [0, 1, 2]

        classes, _ = _resolve_class_mapping(EnsembleWrapper())
        assert classes == [0, 1, 2]
