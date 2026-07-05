"""
Confidence Scoring — quantify prediction reliability on a 0–100 scale.

The confidence score is a **composite** of three independent signals:

.. code-block::

    Confidence = w₁ × SpreadScore + w₂ × AgreementScore + w₃ × CalibrationScore

    where w₁ + w₂ + w₃ = 1.0

Each component is explained below.

1. Probability Spread Score (w₁ = 0.40)
--------------------------------------
Measures how **peaked** the probability distribution is.  A distribution
where one outcome has 95 % probability is more confident than a uniform
distribution where all three outcomes have ~33 %.

**Formula:**

    H(p) = −Σ pᵢ × log₂(pᵢ)          (entropy in bits)
    H_max = log₂(3) ≈ 1.585           (entropy of uniform 3-class dist)

    SpreadScore = (1 − H(p) / H_max) × 100

    • Perfectly certain (p = [0, 0, 1]) → SpreadScore = 100
    • Uniform (p = [⅓, ⅓, ⅓])          → SpreadScore = 0

2. Model Agreement Score (w₂ = 0.35)
--------------------------------------
For **ensemble** predictions, this measures how much the individual
sub-models disagree.  When all sub-models predict nearly the same
probabilities, agreement is high.  When they disagree, confidence drops.

**Formula:**

    For each match, let Pₘ be the probability predicted by model m:
      σ(p) = std(Pₘ across models)   per outcome class
      σ̄ = mean(σ(p_home), σ(p_draw), σ(p_away))
      σ_max = 0.5    (max possible std for 3 models on 3 classes)

    AgreementScore = max(0, 1 − σ̄ / σ_max) × 100

    • All models identical    → AgreementScore = 100
    • Maximum disagreement    → AgreementScore = 0

When only a single model is available, AgreementScore defaults to 50
(moderate — no signal, no penalty).

3. Historical Calibration Score (w₃ = 0.25)
--------------------------------------
Measures how well the model's probability estimates match reality.
Uses the **Brier score** on a held-out calibration set:

    Brier = mean(Σ (pᵢ − oᵢ)²)          where oᵢ ∈ {0, 1} (one-hot actual)
    Brier_max = 2.0                      (worst possible for 3-class)
    Brier_min = 0.0                      (perfect)

    CalibrationScore = max(0, 1 − Brier / Brier_max) × 100

    • Perfect calibration → CalibrationScore = 100
    • Always wrong        → CalibrationScore ≈ 0

If no calibration data is provided, the score defaults to 50.

Usage
-----
::

    from src.confidence_scoring import ConfidenceScorer

    scorer = ConfidenceScorer()

    # Single model
    probs = model.predict_proba(X)          # shape (n, 3)
    result = scorer.score(probs)

    # Ensemble model with per-model predictions
    result = scorer.score(
        probs,
        individual_probs={
            "lr": lr_probs,
            "rf": rf_probs,
            "xgb": xgb_probs,
        },
        calibration_brier=0.15,             # from validation set
    )

    print(result["prediction"])    # 0, 1, or 2
    print(result["probability"])   # e.g. 0.72
    print(result["confidence"])    # e.g. 78 (integer 0–100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Default weights ─────────────────────────────────────
_DEFAULT_W_SPREAD = 0.40
_DEFAULT_W_AGREEMENT = 0.35
_DEFAULT_W_CALIBRATION = 0.25


# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════


@dataclass
class ConfidenceConfig:
    """Settings for the confidence scoring system.

    Attributes
    ----------
    weight_spread : float
        Weight for the probability spread component (default 0.40).
    weight_agreement : float
        Weight for the model agreement component (default 0.35).
    weight_calibration : float
        Weight for the historical calibration component (default 0.25).
    default_agreement : float
        Default agreement score when no ensemble predictions are provided
        (default 50 — neutral).
    default_calibration : float
        Default calibration score when no calibration data is available
        (default 50 — neutral).
    calibration_brier_default : float
        Fallback Brier score if none is provided (default 0.25 — moderate).
    """
    weight_spread: float = _DEFAULT_W_SPREAD
    weight_agreement: float = _DEFAULT_W_AGREEMENT
    weight_calibration: float = _DEFAULT_W_CALIBRATION
    default_agreement: float = 50.0
    default_calibration: float = 50.0
    calibration_brier_default: float = 0.25


# ═══════════════════════════════════════════════════════════
#  Confidence Scorer
# ═══════════════════════════════════════════════════════════


class ConfidenceScorer:
    """Compute composite confidence scores for match outcome predictions.

    Parameters
    ----------
    config_override : ConfidenceConfig, optional
        Override default configuration.
    """

    def __init__(
        self,
        config_override: ConfidenceConfig | None = None,
    ) -> None:
        self.cfg = config_override or ConfidenceConfig()
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure weights sum to 1.0 (normalise if not, guard against zero)."""
        total = self.cfg.weight_spread + self.cfg.weight_agreement + self.cfg.weight_calibration
        if total < 1e-12:
            # All zero — reset to defaults
            self.cfg.weight_spread = _DEFAULT_W_SPREAD
            self.cfg.weight_agreement = _DEFAULT_W_AGREEMENT
            self.cfg.weight_calibration = _DEFAULT_W_CALIBRATION
            total = 1.0
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "Confidence weights sum to %.2f — normalising to 1.0",
                total,
            )
            self.cfg.weight_spread /= total
            self.cfg.weight_agreement /= total
            self.cfg.weight_calibration /= total

    # ── Main scoring method ──────────────────────────────

    def score(
        self,
        probs: np.ndarray,
        individual_probs: dict[str, np.ndarray] | None = None,
        calibration_brier: float | None = None,
    ) -> dict[str, Any]:
        """Compute confidence scores for a batch of predictions.

        Parameters
        ----------
        probs : np.ndarray of shape (n, 3)
            Final predicted probabilities ``[away, draw, home]``.
        individual_probs : dict[str, np.ndarray], optional
            Per-model probabilities for ensemble agreement computation.
            Each value must have shape ``(n, 3)``.  If omitted, a default
            agreement score of 50 is used.
        calibration_brier : float, optional
            Brier score from a held-out validation set.  If omitted, a
            default calibration score of 50 is used.

        Returns
        -------
        dict[str, Any]
            Keys:
            - ``prediction`` (np.ndarray, shape=(n,)) — argmax class
            - ``probability`` (np.ndarray, shape=(n,)) — max probability
            - ``confidence`` (np.ndarray, shape=(n,)) — 0–100 integer scores
            - ``spread_score`` (np.ndarray, shape=(n,)) — component scores
            - ``agreement_score`` (float or np.ndarray) — component scores
            - ``calibration_score`` (float) — component score
            - ``formula`` (str) — plain-text formula explanation
        """
        n = probs.shape[0]

        # ── 1. Probability Spread Score ──────────────────
        spread = self._spread_score(probs)

        # ── 2. Model Agreement Score ─────────────────────
        if individual_probs is not None and len(individual_probs) > 1:
            agreement = self._agreement_score(individual_probs)
        else:
            agreement = np.full(n, self.cfg.default_agreement)

        # ── 3. Historical Calibration Score ──────────────
        calibration = self._calibration_score(calibration_brier)

        # ── 4. Composite ─────────────────────────────────
        confidence = (
            self.cfg.weight_spread * spread
            + self.cfg.weight_agreement * agreement
            + self.cfg.weight_calibration * calibration
        )
        confidence = np.clip(np.round(confidence), 0, 100).astype(int)

        # ── 5. Prediction & max probability ──────────────
        prediction = np.argmax(probs, axis=1)
        probability = np.max(probs, axis=1)

        return {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "spread_score": spread,
            "agreement_score": agreement,
            "calibration_score": np.full(n, calibration) if isinstance(calibration, float) else calibration,
            "formula": self._formula(),
        }

    # ── Score a single match ─────────────────────────────

    def score_one(
        self,
        probs: np.ndarray | list[float],
        individual_probs: dict[str, np.ndarray | list[float]] | None = None,
        calibration_brier: float | None = None,
    ) -> dict[str, Any]:
        """Compute confidence for a single match.

        Parameters
        ----------
        probs : array-like of length 3
            ``[away_prob, draw_prob, home_prob]``.
        individual_probs : dict[str, array-like], optional
            Per-model probabilities for agreement computation.
        calibration_brier : float, optional
            Brier score from validation set.

        Returns
        -------
        dict[str, Any]
            Single-match result with ``prediction`` (int), ``probability`` (float),
            and ``confidence`` (int 0–100).
        """
        probs_arr = np.atleast_2d(np.asarray(probs, dtype=float))
        indiv = None
        if individual_probs is not None:
            indiv = {
                k: np.atleast_2d(np.asarray(v, dtype=float))
                for k, v in individual_probs.items()
            }
        result = self.score(probs_arr, indiv, calibration_brier)
        return {
            "prediction": int(result["prediction"][0]),
            "probability": float(result["probability"][0]),
            "confidence": int(result["confidence"][0]),
            "spread_score": float(result["spread_score"][0]),
            "agreement_score": float(result["agreement_score"][0]),
            "calibration_score": float(result["calibration_score"][0]),
            "formula": result["formula"],
        }

    # ══════════════════════════════════════════════════════
    #  Component 1 — Probability Spread
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _spread_score(probs: np.ndarray) -> np.ndarray:
        """Compute spread component using normalised entropy.

        Parameters
        ----------
        probs : np.ndarray of shape (n, 3)

        Returns
        -------
        np.ndarray of shape (n,)
            Spread score 0–100.  High = peaked distribution.
        """
        # Replace NaN with uniform distribution (conservative — spread = 0)
        p = np.nan_to_num(probs, nan=1.0 / 3.0)
        # Clamp to avoid log(0)
        p = np.clip(p, 1e-15, 1.0)
        # Renormalise any rows that don't sum to 1.0 (floating point drift)
        row_sums = p.sum(axis=1, keepdims=True)
        p = p / row_sums
        # Entropy in bits: H = -sum(p * log2(p))
        entropy = -np.sum(p * np.log2(p), axis=1)
        # Maximum entropy for 3 classes
        h_max = np.log2(3)
        # Normalised: 0 = uniform, 1 = certain
        normalised = 1.0 - entropy / h_max
        return normalised * 100.0

    # ══════════════════════════════════════════════════════
    #  Component 2 — Model Agreement
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _agreement_score(
        individual_probs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute agreement component from per-model predictions.

        Parameters
        ----------
        individual_probs : dict[str, np.ndarray]
            ``{model_name: probs_array}``, each of shape ``(n, 3)``.

        Returns
        -------
        np.ndarray of shape (n,)
            Agreement score 0–100.  High = models agree.
        """
        # Stack: (n_models, n, 3)
        stacked = np.stack(list(individual_probs.values()), axis=0)
        # Standard deviation across models per class: (n, 3)
        std_per_class = np.std(stacked, axis=0, ddof=1)
        # Mean std across the three outcome classes: (n,)
        mean_std = np.mean(std_per_class, axis=1)
        # Max possible std for probabilities (0.5 for 3-class with 2+ models)
        # This is an approximate upper bound
        sigma_max = 0.5
        score = np.clip((1.0 - mean_std / sigma_max) * 100.0, 0, 100)
        return score

    # ══════════════════════════════════════════════════════
    #  Component 3 — Historical Calibration
    # ══════════════════════════════════════════════════════

    def _calibration_score(
        self,
        brier: float | None = None,
    ) -> float:
        """Compute calibration component from Brier score.

        Parameters
        ----------
        brier : float, optional
            Brier score from validation set (0 = perfect, 2 = worst).

        Returns
        -------
        float
            Calibration score 0–100.  High = well-calibrated.
        """
        if brier is None:
            brier = self.cfg.calibration_brier_default
        brier_max = 2.0  # worst possible Brier for 3-class
        return float(np.clip((1.0 - brier / brier_max) * 100.0, 0, 100))

    # ══════════════════════════════════════════════════════
    #  Brier score utility (for computing calibration offline)
    # ══════════════════════════════════════════════════════

    @staticmethod
    def compute_brier_score(
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Compute the multi-class Brier score.

        Parameters
        ----------
        probs : np.ndarray of shape (n, n_classes)
            Predicted probabilities.
        y_true : np.ndarray of shape (n,)
            True class labels (0, 1, 2).

        Returns
        -------
        float
            Brier score (0 = perfect, 2 = worst for 3-class).
        """
        n_classes = probs.shape[1]
        # One-hot encode y_true
        y_onehot = np.eye(n_classes)[y_true]
        return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    @staticmethod
    def compute_calibration_curve(
        probs: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Compute a calibration (reliability) curve.

        Groups predictions into probability bins and compares the mean
        predicted probability to the actual fraction of positive outcomes.

        Parameters
        ----------
        probs : np.ndarray of shape (n, n_classes)
            Predicted probabilities.
        y_true : np.ndarray of shape (n,)
            True class labels.
        n_bins : int
            Number of probability bins (default 10).

        Returns
        -------
        dict[str, Any]
            ``bin_centers``, ``accuracies``, ``confidences``, ``counts``.
        """
        # Use probability of the predicted class
        pred_class = np.argmax(probs, axis=1)
        pred_conf = np.max(probs, axis=1)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = np.zeros(n_bins)
        confidences = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for i in range(n_bins):
            in_bin = (pred_conf >= bins[i]) & (pred_conf < bins[i + 1])
            count = in_bin.sum()
            counts[i] = count
            if count > 0:
                accuracies[i] = (pred_class[in_bin] == y_true[in_bin]).mean()
                confidences[i] = pred_conf[in_bin].mean()

        return {
            "bin_centers": bin_centers,
            "accuracies": accuracies,
            "confidences": confidences,
            "counts": counts,
        }

    # ══════════════════════════════════════════════════════
    #  Formula guide
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _formula() -> str:
        """Return the formula explanation as a string."""
        return f"""\
CONFIDENCE SCORING — FORMULA
━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Confidence = w_Spread × SpreadScore + w_Agreement × AgreementScore
               + w_Calibration × CalibrationScore

  where:
    w_Spread      = {_DEFAULT_W_SPREAD}   (peakedness of probability distribution)
    w_Agreement   = {_DEFAULT_W_AGREEMENT}  (consensus across ensemble models)
    w_Calibration = {_DEFAULT_W_CALIBRATION} (historical Brier score on validation data)

  1. SPREAD SCORE  (0–100)
     ─────────────────────
     Measures how peaked the probability distribution is.

       entropy = -Σ pᵢ × log₂(pᵢ)
       SpreadScore = (1 − entropy / log₂(3)) × 100

     • [0.95, 0.03, 0.02] → entropy ≈ 0.33 → SpreadScore = 79
     • [0.34, 0.33, 0.33] → entropy ≈ 1.58 → SpreadScore = 0

  2. AGREEMENT SCORE  (0–100)
     ────────────────────────
     Measures how much the ensemble sub-models agree.

       σ̄ = mean(std(p_model_home), std(p_model_draw), std(p_model_away))
       AgreementScore = max(0, 1 − σ̄ / 0.5) × 100

     • All models predict same probs → σ̄ = 0 → AgreementScore = 100
     • Maximum disagreement          → σ̄ = 0.5 → AgreementScore = 0

  3. CALIBRATION SCORE  (0–100)
     ──────────────────────────
     Measures how well probabilities match reality.

       Brier = mean(Σ (pᵢ − oᵢ)²)   where oᵢ is one-hot actual outcome
       CalibrationScore = max(0, 1 − Brier / 2.0) × 100

     • Perfect calibration  → Brier = 0.00  → CalibrationScore = 100
     • Moderate calibration → Brier = 0.25  → CalibrationScore = 88
     • No better than chance → Brier = 0.67 → CalibrationScore = 67
"""

    def get_guide(self) -> str:
        """Return the formula explanation with current weights."""
        return f"""\
CONFIDENCE SCORING — WEIGHTS (active)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  w_Spread      = {self.cfg.weight_spread:.2f}
  w_Agreement   = {self.cfg.weight_agreement:.2f}
  w_Calibration = {self.cfg.weight_calibration:.2f}

{self._formula()}
"""
