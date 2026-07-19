"""Unit tests for CalibratedTemperatureWrapper and CalibratedStatsModel.

Tests cover:
- CalibratedTemperatureWrapper: predict, predict_proba, temperature scaling application
- CalibratedStatsModel: predict_matches, probability column calibration, __getattr__ forwarding
- Edge cases: single class, extreme probabilities, missing attributes
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    CalibratedStatsModel,
    CalibratedTemperatureWrapper,
    TemperatureScalingCalibrator,
)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════


def _make_dummy_phase4_model(n_classes: int = 3) -> MagicMock:
    """Create a mock Phase 4 ML model with predict and predict_proba."""
    model = MagicMock()

    def fake_predict_proba(X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        # Return confident but slightly miscalibrated probabilities
        base = np.zeros((n, n_classes))
        base[:, 0] = 0.7
        base[:, 1] = 0.2
        base[:, 2] = 0.1
        return base

    model.predict_proba = fake_predict_proba
    model.predict = lambda X: np.argmax(fake_predict_proba(X), axis=1)
    return model


def _make_dummy_phase3_model(n_matches: int = 5) -> MagicMock:
    """Create a mock Phase 3 model with predict_matches returning a DataFrame."""
    model = MagicMock()

    def fake_predict_matches(df: Any) -> pd.DataFrame:
        n = len(df) if hasattr(df, "__len__") else n_matches
        return pd.DataFrame({
            "home_team": [f"Home{i}" for i in range(n)],
            "away_team": [f"Away{i}" for i in range(n)],
            "home_win_prob": np.full(n, 0.5),
            "draw_prob": np.full(n, 0.3),
            "away_win_prob": np.full(n, 0.2),
            "predicted_result": ["H"] * n,
        })

    model.predict_matches = fake_predict_matches
    model.add_features = MagicMock(return_value=None)
    model.fit = MagicMock(return_value=None)
    return model


@pytest.fixture
def temperature_calibrator() -> TemperatureScalingCalibrator:
    """Return a fitted TemperatureScalingCalibrator."""
    rng = np.random.RandomState(42)
    cal = TemperatureScalingCalibrator(n_classes=3, max_iter=50)
    logits = rng.randn(100, 3)
    y = rng.randint(0, 3, size=100)
    cal.fit(logits, y)
    return cal


@pytest.fixture
def dummy_platt_calibrator() -> MagicMock:
    """Return a mock Platt-scaling type calibrator.

    The transform method tiles a fixed probability vector to match
    the batch size of the input, ensuring shape compatibility.
    """
    cal = MagicMock()
    cal.transform.side_effect = (
        lambda x: np.tile(np.array([[0.6, 0.25, 0.15]]), (x.shape[0], 1))
    )
    return cal


# ═══════════════════════════════════════════════════════════
#  CalibratedTemperatureWrapper tests
# ═══════════════════════════════════════════════════════════


class TestCalibratedTemperatureWrapper:
    """Tests for CalibratedTemperatureWrapper."""

    def test_predict_proba_shape(self, temperature_calibrator: TemperatureScalingCalibrator) -> None:
        """predict_proba should return probabilities summing to 1.0 per row."""
        base = _make_dummy_phase4_model()
        wrapper = CalibratedTemperatureWrapper(base, temperature_calibrator)

        X = np.zeros((10, 5))
        probs = wrapper.predict_proba(X)

        assert probs.shape == (10, 3), f"Expected (10, 3), got {probs.shape}"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1.0"

    def test_predict_argmax(self, temperature_calibrator: TemperatureScalingCalibrator) -> None:
        """predict() should return argmax of calibrated probabilities."""
        base = _make_dummy_phase4_model()
        wrapper = CalibratedTemperatureWrapper(base, temperature_calibrator)

        X = np.zeros((10, 5))
        preds = wrapper.predict(X)
        probs = wrapper.predict_proba(X)

        assert preds.shape == (10,), f"Expected (10,), got {preds.shape}"
        assert np.array_equal(preds, np.argmax(probs, axis=1)), (
            "predict() must match argmax of predict_proba()"
        )

    def test_temperature_scaling_applied(self) -> None:
        """Predictions should differ from raw model after temperature scaling."""
        base = _make_dummy_phase4_model()

        # Fit a temperature calibrator that will modify probs
        rng = np.random.RandomState(42)
        cal = TemperatureScalingCalibrator(n_classes=3, max_iter=50)
        logits = rng.randn(100, 3)
        y = rng.randint(0, 3, size=100)
        cal.fit(logits, y)

        wrapper = CalibratedTemperatureWrapper(base, cal)
        X = np.zeros((5, 5))

        raw_probs = base.predict_proba(X)
        cal_probs = wrapper.predict_proba(X)

        # If temperature != 1.0, calibrated probs should differ
        if abs(cal.temperature - 1.0) > 0.01:
            assert not np.allclose(raw_probs, cal_probs, atol=0.05), (
                "Calibrated probs should differ from raw when temperature != 1.0"
            )

    def test_single_sample(self, temperature_calibrator: TemperatureScalingCalibrator) -> None:
        """Should handle a single sample (1D-ish input)."""
        base = _make_dummy_phase4_model()
        wrapper = CalibratedTemperatureWrapper(base, temperature_calibrator)

        X = np.zeros((1, 5))
        probs = wrapper.predict_proba(X)

        assert probs.shape == (1, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_extreme_probabilities(self) -> None:
        """Should handle extreme probabilities (all 0s, all 1s) without NaN."""
        base = _make_dummy_phase4_model()

        # Fit a calibrator
        rng = np.random.RandomState(42)
        cal = TemperatureScalingCalibrator(n_classes=3, max_iter=50)
        logits = rng.randn(100, 3)
        y = rng.randint(0, 3, size=100)
        cal.fit(logits, y)

        wrapper = CalibratedTemperatureWrapper(base, cal)

        # Probs with extreme values
        with patch.object(base, "predict_proba") as mock_proba:
            mock_proba.return_value = np.array([[0.0, 0.0, 1.0]])
            probs = wrapper.predict_proba(np.zeros((1, 5)))

            assert probs.shape == (1, 3), f"Expected (1, 3), got {probs.shape}"
            assert np.all(np.isfinite(probs)), "NaN or Inf in probs"
            assert np.allclose(probs.sum(axis=1), 1.0), "Must sum to 1.0"

    def test_base_model_passthrough(self, temperature_calibrator: TemperatureScalingCalibrator) -> None:
        """Wrapper should expose the base model."""
        base = _make_dummy_phase4_model()
        wrapper = CalibratedTemperatureWrapper(base, temperature_calibrator)
        assert wrapper.base_model is base


# ═══════════════════════════════════════════════════════════
#  CalibratedStatsModel tests
# ═══════════════════════════════════════════════════════════


class TestCalibratedStatsModel:
    """Tests for CalibratedStatsModel."""

    def test_predict_matches_probs_sum_to_one(self, dummy_platt_calibrator: MagicMock) -> None:
        """predict_matches should return probabilities summing to 1.0 per row."""
        base = _make_dummy_phase3_model(n_matches=3)
        wrapper = CalibratedStatsModel(base, dummy_platt_calibrator, method="platt")

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        result = wrapper.predict_matches(df)

        prob_cols = ["home_win_prob", "draw_prob", "away_win_prob"]
        probs = result[prob_cols].values
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), (
            f"Calibrated probs do not sum to 1: {probs.sum(axis=1)}"
        )

    def test_predict_matches_calls_calibrator(self, dummy_platt_calibrator: MagicMock) -> None:
        """Calibrator.transform should be called with correct raw probs."""
        base = _make_dummy_phase3_model(n_matches=2)
        wrapper = CalibratedStatsModel(base, dummy_platt_calibrator, method="platt")

        df = pd.DataFrame({"dummy": [1, 2]})
        wrapper.predict_matches(df)

        # transform should have been called
        assert dummy_platt_calibrator.transform.called, "calibrator.transform was not called"
        call_args = dummy_platt_calibrator.transform.call_args[0][0]
        assert call_args.shape == (2, 3), f"Expected (2, 3) input, got {call_args.shape}"

    def test_predict_matches_preserves_columns(self, dummy_platt_calibrator: MagicMock) -> None:
        """predict_matches should preserve non-probability columns."""
        base = _make_dummy_phase3_model(n_matches=2)
        wrapper = CalibratedStatsModel(base, dummy_platt_calibrator, method="platt")

        df = pd.DataFrame({"dummy": [1, 2]})
        result = wrapper.predict_matches(df)

        assert "home_team" in result.columns
        assert "away_team" in result.columns
        assert "predicted_result" in result.columns
        assert "home_win_prob" in result.columns
        assert "draw_prob" in result.columns
        assert "away_win_prob" in result.columns

    def test_getattr_forwards_whitelisted(self) -> None:
        """Whitelisted methods should be forwarded to base model."""
        base = _make_dummy_phase3_model()
        cal = MagicMock()
        wrapper = CalibratedStatsModel(base, cal, method="platt")

        # add_features and fit are in _forwarded_methods
        wrapper.add_features(pd.DataFrame({"x": [1]}))
        wrapper.fit(pd.DataFrame({"x": [1]}))

        assert base.add_features.called, "add_features was not forwarded"
        assert base.fit.called, "fit was not forwarded"

    def test_getattr_raises_for_non_whitelisted(self) -> None:
        """Non-whitelisted attributes should raise AttributeError."""
        base = _make_dummy_phase3_model()
        cal = MagicMock()
        wrapper = CalibratedStatsModel(base, cal, method="platt")

        with pytest.raises(AttributeError):
            _ = wrapper.non_existent_method

        with pytest.raises(AttributeError):
            _ = wrapper.some_other_attr

    def test_temperature_calibrator_accepts_logits(self) -> None:
        """When calibrator is TemperatureScalingCalibrator, probs are converted to logits first."""
        rng = np.random.RandomState(42)
        temp_cal = TemperatureScalingCalibrator(n_classes=3, max_iter=50)
        temp_cal.fit(rng.randn(100, 3), rng.randint(0, 3, size=100))

        base = _make_dummy_phase3_model(n_matches=2)
        wrapper = CalibratedStatsModel(base, temp_cal, method="temperature")

        df = pd.DataFrame({"dummy": [1, 2]})
        result = wrapper.predict_matches(df)

        prob_cols = ["home_win_prob", "draw_prob", "away_win_prob"]
        probs = result[prob_cols].values
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), (
            "Temperature-calibrated probs must sum to 1.0"
        )
        assert np.all(np.isfinite(probs)), "NaN in temperature-calibrated probs"

    def test_single_match(self, dummy_platt_calibrator: MagicMock) -> None:
        """Should handle a single match DataFrame."""
        base = _make_dummy_phase3_model(n_matches=1)
        wrapper = CalibratedStatsModel(base, dummy_platt_calibrator, method="platt")

        df = pd.DataFrame({"dummy": [1]})
        result = wrapper.predict_matches(df)

        assert len(result) == 1
        prob_cols = ["home_win_prob", "draw_prob", "away_win_prob"]
        assert np.allclose(result[prob_cols].sum(axis=1), 1.0)


# ═══════════════════════════════════════════════════════════
#  Integration: wrapper with real calibrators
# ═══════════════════════════════════════════════════════════


class TestCalibratedWrappersIntegration:
    """Integration tests for wrappers with real (non-mock) calibrators."""

    def test_temperature_wrapper_end_to_end(self) -> None:
        """End-to-end: fit temperature calibrator, wrap model, predict."""
        rng = np.random.RandomState(42)

        # Fit temperature calibrator
        cal = TemperatureScalingCalibrator(n_classes=3, max_iter=50)
        logits = rng.randn(200, 3)
        y = rng.randint(0, 3, size=200)
        cal.fit(logits, y)

        # Wrap model
        base = _make_dummy_phase4_model()
        wrapper = CalibratedTemperatureWrapper(base, cal)

        # Predict
        X = rng.randn(50, 5)
        probs = wrapper.predict_proba(X)
        preds = wrapper.predict(X)

        assert probs.shape == (50, 3), f"Expected (50, 3), got {probs.shape}"
        assert preds.shape == (50,), f"Expected (50,), got {preds.shape}"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probs must sum to 1.0"
        assert np.array_equal(preds, np.argmax(probs, axis=1)), (
            "predict() must match argmax of predict_proba()"
        )

    def test_stats_model_with_real_temp_calibrator(self) -> None:
        """End-to-end: fit temperature calibrator, wrap stats model, predict."""
        rng = np.random.RandomState(42)

        # Fit temperature calibrator
        cal = TemperatureScalingCalibrator(n_classes=3, max_iter=50)
        logits = rng.randn(200, 3)
        y = rng.randint(0, 3, size=200)
        cal.fit(logits, y)

        base = _make_dummy_phase3_model(n_matches=10)
        wrapper = CalibratedStatsModel(base, cal, method="temperature")

        df = pd.DataFrame({"dummy": range(10)})
        result = wrapper.predict_matches(df)

        prob_cols = ["home_win_prob", "draw_prob", "away_win_prob"]
        probs = result[prob_cols].values

        assert len(result) == 10
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), (
            f"Probs sum to {probs.sum(axis=1)}"
        )
        assert np.all(np.isfinite(probs)), "NaN in probabilities"

    def test_temp_calibrator_high_temperature(self) -> None:
        """Very high temperature should flatten probabilities toward uniform."""
        rng = np.random.RandomState(42)

        # Create a calibrator with T >> 1.0
        cal = TemperatureScalingCalibrator(n_classes=3, max_iter=100)
        # Force temperature very high
        logits = rng.randn(200, 3) * 10.0  # large spread logits
        y = rng.randint(0, 3, size=200)
        cal.fit(logits, y)

        base = _make_dummy_phase4_model()
        wrapper = CalibratedTemperatureWrapper(base, cal)

        X = rng.randn(5, 5)
        probs = wrapper.predict_proba(X)

        # With high T, probs should be closer to uniform
        assert np.all(probs > 0.2), f"High T should flatten probs, got {probs}"
        assert np.all(probs < 0.5), f"High T should flatten probs, got {probs}"
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_temp_calibrator_low_temperature(self) -> None:
        """Very low temperature should sharpen probabilities toward one-hot."""
        rng = np.random.RandomState(42)

        cal = TemperatureScalingCalibrator(n_classes=3, max_iter=100, init_temp=0.1)
        logits = rng.randn(200, 3)  # standard spread
        y = rng.randint(0, 3, size=200)
        cal.fit(logits, y)
        # Force temperature very low to verify sharpening behavior
        cal.temperature_ = 0.15

        base = _make_dummy_phase4_model()
        wrapper = CalibratedTemperatureWrapper(base, cal)

        X = rng.randn(5, 5)
        probs = wrapper.predict_proba(X)

        assert np.allclose(probs.sum(axis=1), 1.0), "Must sum to 1.0"
        # Very low T -> logits are divided by T, making confident classes sharper
        raw = base.predict_proba(X)
        assert np.any(probs > 0.9), (
            f"Low T (={cal.temperature_:.3f}) should sharpen, got max {probs.max():.3f}"
        )
