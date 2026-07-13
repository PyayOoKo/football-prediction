---
tags:
  - python-module
  - tuning
  - hyperparameters
---

# `hyperparameter_tuning.py` — Hyper-parameter Tuning

**Path:** `src/hyperparameter_tuning.py`

Grid-search and random-search wrappers for base models. Uses `RandomizedSearchCV` with time-series aware cross-validation.

**Key function:** `tune_model(X, y, model_type, ...)` — searches over config-defined param grids, returns best estimator and CV scores.

**Default:** disabled for ensemble (set `config.ensemble.tune_base_models = True` to enable).

See also: [[train.py]], [[config.py]], [[time_series_cv.py]], [[Auxiliary Modules]]
