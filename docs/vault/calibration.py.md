---
tags:
  - python-module
  - calibration
  - probabilities
---

# `calibration.py` — Probability Calibration

**Path:** `src/calibration.py`

Calibrates model probabilities using Platt scaling or isotonic regression to ensure well-calibrated output (e.g., 70% predicted = 70% observed).

**Key functions:** `calibrate_model(model, X_cal, y_cal)` — wraps CalibratedClassifierCV from sklearn. `plot_calibration_curve(y_true, y_proba)` — reliability diagram.

**Enabled by default** in ensemble (`config.ensemble.calibrate = True`).

**Note:** Calibration uses a held-out validation set to avoid data leakage.

See also: [[ensemble.py]], [[evaluate.py]], [[config.py]], [[Auxiliary Modules]]
