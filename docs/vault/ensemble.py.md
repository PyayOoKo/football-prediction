---
tags:
  - python-module
  - ensemble
  - model
  - ml
---

# `ensemble.py` — Ensemble Model

**Path:** `src/ensemble.py`

Default prediction model. Combines XGBoost + Logistic Regression + Poisson using optimised weighted averaging.

**Key class:** `EnsembleModel` — fit, predict_proba, predict, save, load, evaluate

**Weight optimisation:** Grid search over weight combos (step=0.10, ~66 combos for 3 models) minimising validation log-loss.

See also: [[config.py]], [[poisson_model.py]], [[train.py]], [[Ensemble Model]], [[Runtime Sequence Diagrams]]
