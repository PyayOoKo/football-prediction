---
tags:
  - python-module
  - confidence
  - scoring
---

# `confidence_scoring.py` — Prediction Confidence

**Path:** `src/confidence_scoring.py`

Computes a composite confidence score for each prediction using three components:
- **Calibration quality** — how well probability estimates match observed frequencies
- **Model agreement** — spread across ensemble sub-models
- **Feature coverage** — how many features were available vs. missing

**Key function:** `compute_confidence_score(probs, model_probs_list, ...)` — weighted geometric mean of the three components.

See also: [[ensemble.py]], [[calibration.py]], [[value_betting.py]], [[Value Betting & Backtesting]]
