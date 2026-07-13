---
tags:
  - football-prediction
  - modules
  - auxiliary
  - training
created: 2026-07-12
---

# 🧩 Auxiliary Modules

> Supporting modules for training, evaluation, calibration, prediction, and hyperparameter tuning.

See also: [[Ensemble Model]], [[Feature Engineering Pipeline]], [[Config System]]

---

## Training Module

**File:** [[train.py]]

Supports multiple model types via factory pattern:

| Function | Purpose |
|----------|---------|
| `train_model(X_train, y_train, X_val, y_val)` | Train a single model with validation |
| `tune_hyperparameters(X_train, y_train)` | Randomised CV search for best params |
| `save_model(model, file_name)` | Serialise via joblib to `models/` |
| `load_model(file_name)` | Deserialise from `models/` |

### Model Types

All configurable via `config.train.model_type`:

- `logistic_regression` — L-BFGS solver, balanced class weights
- `random_forest` — Bagged trees, balanced subsample
- `xgboost` — `multi:softprob`, early stopping (default)
- `lightgbm` — Leaf-wise growth, early stopping
- `neural_network` — PyTorch MLP (128→64→32→3, ReLU, dropout)

---

## Hyperparameter Tuning

**File:** [[hyperparameter_tuning.py]]

Provides `HyperTuner` class for systematic multi-model comparison:

```python
from src.hyperparameter_tuning import HyperTuner

tuner = HyperTuner()
results = tuner.run(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,   # optional
)

print(results["report_text"])    # formatted comparison
print(results["summary_df"])     # pandas table
```

### What It Does

| Step | Description |
|------|-------------|
| 1 | Train baseline with default `config.train.*` params |
| 2 | Run GridSearchCV (LR) or RandomizedSearchCV (RF, XGB, LGBM) |
| 3 | Train optimised version with best params |
| 4 | Compare log-loss and accuracy before/after |
| 5 | Save both models (`{type}_baseline.joblib`, `{type}_tuned.joblib`) |
| 6 | Generate formatted report + summary DataFrame |

### Time-Series CV

Uses `create_time_series_folds()` from [[time_series_cv.py]] — expanding window folds that respect chronological order (no future leakage).

---

## Evaluation

**File:** [[evaluate.py]]

```python
from src.evaluate import evaluate_model

report = evaluate_model(model, X_test, y_test)
```

Computes: accuracy, precision, recall, F1, ROC-AUC, log-loss

Generates plots (PNG → `reports/`):
- Confusion matrix heatmap
- ROC curve (one-vs-rest for multi-class)
- Feature importance bar chart (top 20)

---

## Calibration

**File:** [[calibration.py]]

Corrects miscalibrated probabilities (especially tree-based models):

| Method | Description | Best For |
|--------|-------------|----------|
| **Platt scaling** | Logistic regression on raw logits | Small validation sets |
| **Isotonic regression** | Non-parametric monotonic mapping | Large validation sets |

```python
from src.calibration import CalibratedModel

calibrated = CalibratedModel(base_model=xgb_model, method="platt")
calibrated.fit(X_train, y_train, X_val, y_val)

# Get calibrated probabilities
probs = calibrated.predict_proba(X_test)

# Compare raw vs calibrated
metrics = calibrated.evaluate_calibration(X_test, y_test)
```

---

## Predictions

**File:** [[predict.py]]

```python
from src.predict import predict_fixtures

predictions = predict_fixtures(
    model,
    fixtures_df,
    output_path="reports/predictions/output.csv",
    individual_probs=ensemble_probs,  # for confidence scoring
    calibration_brier=0.15,
)
```

Output formats: CSV, JSON, or console (configurable via `config.predict.output_format`)

---

## EDA (Exploratory Data Analysis)

**File:** [[eda.py]]

Generates 6 publication-quality charts → `reports/figures/`:

| Chart | Description |
|-------|-------------|
| Win distribution | H/D/A proportions with baseline annotation |
| Goals distribution | Overlaid histogram of home vs away goals |
| Home advantage | Grouped bar chart by venue |
| Team statistics | Top-N by goals scored and conceded |
| Correlation matrix | Heatmap of feature correlations |
| Missing values | Column missingness + pattern heatmap |

```python
from src.eda import run_eda
report = run_eda()
```

---

## Time Series Cross-Validation

**File:** [[time_series_cv.py]]

```python
from src.time_series_cv import create_time_series_folds

cv = create_time_series_folds(n_splits=5)
# Used by: train.py, hyperparameter_tuning.py
```

Creates expanding-window folds — each fold uses all data *before* the split point for training, data *after* for validation. No shuffling. No future leakage.
