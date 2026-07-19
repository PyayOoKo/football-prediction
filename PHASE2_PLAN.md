# Phase 2: Code Quality & Modularity Plan

## Task 4: Split Oversized Modules (>1000 lines)

### Priority Files to Split:
1. **ensemble.py** (1,832 lines) - Split into ensemble/ package
2. **dixon_coles.py** (1,240 lines) - Separate optimization logic
3. **calibration.py** (1,146 lines) - Extract calibration strategies
4. **hyperparameter_tuning.py** (1,134 lines) - Modularize tuning strategies
5. **live_predictions.py** (1,121 lines) - Separate streaming logic
6. **prediction_engine.py** (999 lines) - Almost at threshold

### Split Strategy for ensemble.py:
```
src/ensemble/
├── __init__.py           # Public API exports
├── weighted.py           # WeightedEnsemble class (~600 lines)
├── training.py           # EnsembleModel training logic (~700 lines)
├── optimization.py       # Weight grid search & optimization (~300 lines)
└── adapters.py           # Model adapters (move from protocol.py)
```

## Task 5: Standardize Error Handling & Logging

### Steps:
1. Create exception hierarchy in `src/utils/exceptions.py`
2. Replace all `print()` statements with logging
3. Add structured logging context
4. Implement global exception handler

## Task 6: Enforce Type Safety

### Steps:
1. Run mypy on entire codebase
2. Replace dict returns with TypedDict/dataclasses
3. Add type hints to all public methods
4. Create stub files for complex types

---

## Execution Order:
Week 2 Day 1-2: Split ensemble.py
Week 2 Day 3: Split dixon_coles.py and calibration.py  
Week 2 Day 4: Split hyperparameter_tuning.py and live_predictions.py
Week 2 Day 5: Standardize error handling
Week 3 Day 1-2: Type safety enforcement
