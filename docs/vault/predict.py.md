---
tags:
  - python-module
  - prediction
  - inference
---

# `predict.py` — Match Prediction

**Path:** `src/predict.py`

High-level prediction functions for upcoming matches. Wraps trained model with feature building to provide end-to-end predictions.

**Key functions:** `predict_match(model, df, home_team, away_team)` — predicts single match outcome probabilities. `predict_matches(model, df)` — batch prediction on DataFrame.

Returns dict/Series with home_win, draw, away_win probabilities.

See also: [[ensemble.py]], [[feature_engineering.py]], [[train.py]], [[Auxiliary Modules]]
