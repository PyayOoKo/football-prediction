---
tags:
  - python-module
  - evaluation
  - metrics
---

# `evaluate.py` — Model Evaluation

**Path:** `src/evaluate.py`

Evaluation metrics for model comparison and reporting.

**Key functions:** `evaluate_model(y_true, y_pred, y_proba)` — computes classification report, confusion matrix, log-loss, Brier score, ROC-AUC. `plot_results(y_true, y_proba)` — produces calibration curve, ROC curve, precision-recall curves.

**Metrics:** Accuracy, Precision, Recall, F1, Log Loss, Brier Score, AUC-ROC, AUC-PR.

See also: [[calibration.py]], [[train.py]], [[ensemble.py]], [[Auxiliary Modules]]
