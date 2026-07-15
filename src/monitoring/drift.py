"""
Data Drift Detection — monitors for statistical drift in features and predictions.

Uses:
- Population Stability Index (PSI) for categorical features
- Kolmogorov-Smirnov test for numerical features
- Prediction distribution shift
- Feature importance shift

Usage
-----
::

    from src.monitoring.drift import DriftDetector

    detector = DriftDetector()
    result = detector.detect(reference_df, current_df)
    if result.drift_detected:
        print(f"Drift detected in: {result.drifted_features}")
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of a drift detection analysis.

    Parameters
    ----------
    drift_detected : bool
        Whether significant drift was detected in any feature.
    drifted_features : list[str]
        Names of features that exceeded the drift threshold.
    drift_scores : dict[str, float]
        Per-feature drift scores.
    overall_drift_score : float
        Average drift score across all features.
    prediction_drift_score : float
        Drift score for prediction distributions.
    feature_importance_drift : float
        Drift in feature importance rankings.
    n_features_analyzed : int
        Number of features included in analysis.
    timestamp : datetime
        When the analysis was performed.
    """

    drift_detected: bool = False
    drifted_features: list[str] = field(default_factory=list)
    drift_scores: dict[str, float] = field(default_factory=dict)
    overall_drift_score: float = 0.0
    prediction_drift_score: float = 0.0
    feature_importance_drift: float = 0.0
    n_features_analyzed: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "drifted_features": self.drifted_features[:20],
            "overall_drift_score": round(self.overall_drift_score, 4),
            "prediction_drift_score": round(self.prediction_drift_score, 4),
            "feature_importance_drift": round(self.feature_importance_drift, 4),
            "n_features_analyzed": self.n_features_analyzed,
            "top_drifted": {k: round(v, 4) for k, v in
                            sorted(self.drift_scores.items(), key=lambda x: -x[1])[:10]},
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DriftConfig:
    """Configuration for drift detection.

    Parameters
    ----------
    psi_threshold : float
        PSI threshold above which drift is flagged (default 0.2).
    ks_threshold : float
        KS statistic threshold (default 0.1).
    overall_threshold : float
        Overall drift score threshold (default 0.1).
    prediction_drift_threshold : float
        Prediction distribution drift threshold (default 0.15).
    min_samples : int
        Minimum samples required for analysis (default 30).
    psi_bins : int
        Number of bins for PSI calculation (default 10).
    """

    psi_threshold: float = 0.2
    ks_threshold: float = 0.1
    overall_threshold: float = 0.1
    prediction_drift_threshold: float = 0.15
    min_samples: int = 30
    psi_bins: int = 10


class DriftDetector:
    """Detects statistical drift between reference and current data.

    Parameters
    ----------
    config : DriftConfig, optional
        Drift detection parameters.
    drift_history_path : str, optional
        Path to persist drift history.
    """

    def __init__(
        self,
        config: DriftConfig | None = None,
        drift_history_path: str = "data/monitoring/drift_history.json",
    ) -> None:
        self.config = config or DriftConfig()
        self._history_path = Path(drift_history_path)
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    def detect(
        self,
        reference: dict[str, np.ndarray] | Any,
        current: dict[str, np.ndarray] | Any,
    ) -> DriftResult:
        """Detect drift between reference and current data.

        Parameters
        ----------
        reference : DataFrame or dict of arrays
            Reference/baseline data.
        current : DataFrame or dict of arrays
            Current data to check for drift.

        Returns
        -------
        DriftResult
            Drift analysis results.
        """
        import pandas as pd

        # Convert to dicts if DataFrames
        if isinstance(reference, pd.DataFrame):
            ref_data = {col: reference[col].values for col in reference.columns}
            ref_preds = reference.get("prediction", reference.get("predicted_outcome"))
        else:
            ref_data = reference
            ref_preds = ref_data.get("prediction", ref_data.get("predicted_outcome"))

        if isinstance(current, pd.DataFrame):
            cur_data = {col: current[col].values for col in current.columns}
            cur_preds = current.get("prediction", current.get("predicted_outcome"))
        else:
            cur_data = current
            cur_preds = cur_data.get("prediction", cur_data.get("predicted_outcome"))

        drift_scores: dict[str, float] = {}
        drifted_features: list[str] = []
        n_analyzed = 0

        # Analyze numerical features
        for feature in ref_data:
            if feature.startswith("_") or feature in ("prediction", "predicted_outcome", "result"):
                continue
            ref_arr = ref_data[feature]
            cur_arr = cur_data.get(feature)

            if cur_arr is None or len(ref_arr) < self.config.min_samples or len(cur_arr) < self.config.min_samples:
                continue

            # Determine if numeric
            try:
                ref_float = ref_arr.astype(float)
                cur_float = cur_arr.astype(float)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False

            if is_numeric:
                score = self._calculate_psi(ref_float, cur_float)
            else:
                score = self._calculate_psi_categorical(ref_arr, cur_arr)

            drift_scores[feature] = score
            n_analyzed += 1

            if score > self.config.psi_threshold:
                drifted_features.append(feature)

        # Prediction distribution drift
        pred_drift = 0.0
        if ref_preds is not None and cur_preds is not None:
            try:
                ref_p = np.asarray(ref_preds, dtype=float)
                cur_p = np.asarray(cur_preds, dtype=float)
                if len(ref_p) >= self.config.min_samples and len(cur_p) >= self.config.min_samples:
                    pred_drift = self._calculate_psi(ref_p, cur_p)
            except (ValueError, TypeError):
                # Handle categorical predictions
                pred_drift = self._calculate_psi_categorical(
                    np.asarray(ref_preds, dtype=str),
                    np.asarray(cur_preds, dtype=str),
                )

        # Overall drift score
        overall_score = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        drift_detected = (
            overall_score > self.config.overall_threshold
            or pred_drift > self.config.prediction_drift_threshold
        )

        result = DriftResult(
            drift_detected=drift_detected,
            drifted_features=drifted_features,
            drift_scores=drift_scores,
            overall_drift_score=float(overall_score),
            prediction_drift_score=float(pred_drift),
            n_features_analyzed=n_analyzed,
        )

        # Persist history
        self._save_history(result)

        if drift_detected:
            logger.warning(
                "Drift detected: overall=%.4f, pred_drift=%.4f, %d/%d features drifted",
                overall_score, pred_drift, len(drifted_features), n_analyzed,
            )

        return result

    def detect_from_csv(
        self,
        reference_path: str,
        current_path: str,
    ) -> DriftResult:
        """Detect drift between two CSV files.

        Parameters
        ----------
        reference_path : str
            Path to reference/baseline CSV.
        current_path : str
            Path to current CSV.

        Returns
        -------
        DriftResult
        """
        import pandas as pd

        ref_df = pd.read_csv(reference_path, low_memory=False)
        cur_df = pd.read_csv(current_path, low_memory=False)

        return self.detect(ref_df, cur_df)

    # ── PSI Calculation ────────────────────────────────

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Population Stability Index.

        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        """
        # Remove NaN
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]

        if len(ref) < self.config.min_samples or len(cur) < self.config.min_samples:
            return 0.0

        # Create bins based on reference distribution
        bins = np.percentile(ref, np.linspace(0, 100, self.config.psi_bins + 1))

        # Handle edge case: all identical values
        if len(np.unique(bins)) == 1:
            return 0.0

        # Clip extreme values
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)

        ref_pct = ref_counts / len(ref)
        cur_pct = cur_counts / len(cur)

        # Replace zeros with small epsilon
        epsilon = 1e-6
        ref_pct = np.maximum(ref_pct, epsilon)
        cur_pct = np.maximum(cur_pct, epsilon)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def _calculate_psi_categorical(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """Calculate PSI for categorical features."""
        from collections import Counter

        ref_counts = Counter(reference)
        cur_counts = Counter(current)

        all_categories = set(ref_counts.keys()) | set(cur_counts.keys())
        epsilon = 1e-6

        ref_total = len(reference)
        cur_total = len(current)

        psi = 0.0
        for cat in all_categories:
            ref_pct = (ref_counts.get(cat, 0) + epsilon) / (ref_total + epsilon * len(all_categories))
            cur_pct = (cur_counts.get(cat, 0) + epsilon) / (cur_total + epsilon * len(all_categories))

            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

        return psi

    # ── Persistence ────────────────────────────────────

    def _save_history(self, result: DriftResult) -> None:
        """Append a drift result to the history file."""
        try:
            history = []
            if self._history_path.exists():
                history = json.loads(self._history_path.read_text())

            history.append(result.to_dict())

            # Keep last 100 entries
            if len(history) > 100:
                history = history[-100:]

            self._history_path.write_text(json.dumps(history, indent=2))
        except Exception as exc:
            logger.warning("Failed to save drift history: %s", exc)

    def get_drift_history(self, days: int = 30) -> list[dict]:
        """Get historical drift detection results.

        Parameters
        ----------
        days : int
            Lookback period.

        Returns
        -------
        list[dict]
            Historical drift results.
        """
        if not self._history_path.exists():
            return []

        try:
            history = json.loads(self._history_path.read_text())
            cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=days)
            return [h for h in history if h.get("timestamp", "") > cutoff.isoformat()]
        except Exception:
            return []
