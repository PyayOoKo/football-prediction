---
tags:
  - python-module
  - validation
  - features
  - data-quality
---

# `validation/__init__.py` — Feature Validation

**Path:** `src/feature_framework/validation/`

- `__init__.py` — `FeatureValidator` orchestrator, PSI computation
- `checks.py` — 10 standalone check functions + `compute_psi()`
- `report.py` — 5 report dataclass types (`ValidationReport`, `CorrelationReport`, `MissingValueReport`, `DriftReport`, `FeatureImportancePlaceholder`)

**Key class:** `FeatureValidator` — validate(), correlation_matrix(), missing_value_report(), drift_report(), feature_importance_placeholder()

**77 tests** in `tests/test_feature_framework/test_feature_validation.py`

See also: [[Feature Orchestrator]], [[Feature Validation Framework]], [[orchestrator.py]]
