---
tags:
  - python-module
  - training
  - ml
---

# `train.py` — Model Training

**Path:** `src/train.py`

Trains single ML models with validation and hyper-parameter tuning.

**Key functions:** `train_model()` — trains XGBoost/LR/RF/LightGBM/NeuralNet; `tune_hyperparameters()` — RandomizedSearchCV with time-series CV; `save_model()` / `load_model()` — joblib persistence.

Supports: logistic_regression, random_forest, xgboost, lightgbm, neural_network.

See also: [[config.py]], [[ensemble.py]], [[hyperparameter_tuning.py]], [[time_series_cv.py]], [[Auxiliary Modules]]
