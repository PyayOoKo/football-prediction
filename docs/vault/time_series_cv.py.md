---
tags:
  - python-module
  - cv
  - time-series
---

# `time_series_cv.py` — Time-Series Cross Validation

**Path:** `src/time_series_cv.py`

Custom cross-validation that respects chronological order — no future data leaks into training.

**Key function:** `create_time_series_folds(X, y, n_folds, ...)` — expanding window or sliding window fold generation. Used by `hyperparameter_tuning.py` and `ensemble.py` weight optimisation.

**Strategy:** Each fold's training set ends before the validation set begins. Gap parameter can add a buffer between train/val to prevent temporal autocorrelation.

See also: [[hyperparameter_tuning.py]], [[ensemble.py]], [[train.py]], [[Auxiliary Modules]]
