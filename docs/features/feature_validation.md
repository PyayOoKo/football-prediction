# Feature Validation Framework

Production-grade validation for computed features. Automatically detects 10 categories of data quality issues and generates 5 report types.

## Quick Start

```python
from src.feature_framework.validation import FeatureValidator

validator = FeatureValidator()

# Validate all feature columns
report = validator.validate(features_df)
print(report.summary())

# Generate specialised reports
correlation = validator.correlation_matrix(features_df)
missing = validator.missing_value_report(features_df)
drift = validator.drift_report(current_df, reference_df)
importance = validator.feature_importance_placeholder(
    feature_names=features_df.columns.tolist(),
    model_type="xgboost",
)
```

## Detection Checks

| # | Check | Description | Default Threshold |
|---|-------|-------------|-------------------|
| 1 | **Data Leakage** | Detects unsorted dates, out-of-order rows | — |
| 2 | **Constant Features** | Zero-variance columns (single unique value) | `min_unique_ratio=0.01` |
| 3 | **Highly Correlated** | Feature pairs with \|r\| > threshold | `correlation_threshold=0.95` |
| 4 | **Missing Values** | NaN/null in any column | — |
| 5 | **Invalid Ranges** | Values outside expected bounds | Configurable per-column |
| 6 | **Infinite Values** | Inf / -Inf in numeric columns | — |
| 7 | **NaN Values** | NaN in numeric columns specifically | — |
| 8 | **Duplicate Features** | Identical or near-identical (r > 0.999) columns | — |
| 9 | **Low Variance** | Variance below threshold | `variance_threshold=0.01` |
| 10 | **Feature Drift** | Distribution change vs reference (PSI) | `drift_threshold=0.1` |

## Reports

### Validation Report

Aggregated results from all checks:

```python
report = validator.validate(df)
print(report.summary())
# FEATURE VALIDATION REPORT
# ============================================================
#   Data:         100 rows × 10 columns
#   Checks:       10/10 passed
#   Result:       ✅ PASS

report.passed          # True if all checks pass
report.failed_checks   # Count of failed checks
report.total_violations  # Total violations across all checks
report.violations_dataframe  # Flattened DataFrame of all violations
report.to_dict()       # Serialisable dict
```

### Correlation Matrix

```python
report = validator.correlation_matrix(df)
report.n_high_pairs    # Pairs with |r| > threshold
report.high_correlation_pairs  # List of {feature_1, feature_2, correlation}
```

### Missing Value Report

```python
report = validator.missing_value_report(df)
report.n_missing_cells   # Total NaN cells
report.missing_rate      # Proportion of all cells missing
report.to_dataframe()    # Per-column details as DataFrame
```

### Drift Report

Uses Population Stability Index (PSI). PSI > 0.1 typically indicates meaningful drift:

```python
report = validator.drift_report(current_df, reference_df)
report.passed           # True if no features drifted
report.n_drifted         # Count of drifted features
```

### Feature Importance Placeholder

```python
placeholder = validator.feature_importance_placeholder(
    feature_names=df.columns.tolist(),
    model_type="xgboost",
)
# Actual importance requires a trained model
```

## Pipeline Integration

The `FeatureValidator` is automatically integrated into `FeaturePipeline`:

```python
from src.feature_framework import FeaturePipeline

pipeline = FeaturePipeline(...)
report = pipeline.run(entity_type="dataframe", df=matches_df)
# report.validation contains validation results
```

## Configuration

```python
validator = FeatureValidator(
    checks=["constant_features", "missing_values", "nan_values"],
    correlation_threshold=0.90,
    variance_threshold=0.001,
    drift_threshold=0.2,
    min_unique_ratio=0.005,
    range_bounds={
        "odds_*": (1.0, 100.0),
        "fair_prob_*": (0.0, 1.0),
        "clv_*": (-0.5, 0.5),
    },
    date_column="date",
    verbose=True,
)
```

## Population Stability Index (PSI)

PSI quantifies how much a feature's distribution has changed:

```
PSI = Σ((actual% − expected%) × ln(actual% / expected%))
```

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.1 | No significant drift |
| 0.1 – 0.25 | Moderate drift — investigate |
| > 0.25 | Significant drift — retrain recommended |

## Test Coverage

| Class | Tests | Key Coverage |
|-------|:-----:|--------------|
| `TestFeatureValidatorOrchestration` | 8 | Validate clean/dirty/empty, all checks present, pipeline integration |
| `TestCheckDataLeakage` | 4 | Sorted passes, unsorted fails, no date, empty |
| `TestCheckConstantFeatures` | 3 | Clean passes, constant detected, empty |
| `TestCheckHighlyCorrelated` | 4 | Clean passes, perfect correlation, single column, empty |
| `TestCheckMissingValues` | 3 | Clean passes, missing detected, empty |
| `TestCheckInvalidRanges` | 4 | Clean passes, out-of-range, wildcard, empty |
| `TestCheckInfiniteValues` | 3 | Clean passes, inf detected, empty |
| `TestCheckNanValues` | 4 | Clean passes, nan detected, empty, string ignored |
| `TestCheckDuplicateFeatures` | 4 | Clean passes, identical, empty, single |
| `TestCheckLowVariance` | 4 | Clean passes, low var detected, empty, string ignored |
| `TestCheckFeatureDrift` | 4 | No reference, similar, drifted, empty |
| `TestCorrelationMatrix` | 4 | Features, single column, summary, to_dict |
| `TestMissingValueReport` | 4 | With missing, clean, summary, to_dataframe |
| `TestDriftReport` | 4 | No reference, similar, drifted, summary |
| `TestFeatureImportancePlaceholder` | 3 | Creation, summary, to_dict |
| `TestValidationReport` | 5 | Properties, summary, to_dict, violations_df |
| `TestPSIComputation` | 3 | Identical, different, small samples |
| `TestCustomConfiguration` | 3 | Selected checks, custom thresholds, custom bounds |
| `TestEdgeCases` | 4 | All NaN, all constant, mixed types, large |

**Total: 77 tests**
