---
tags:
  - python-module
  - dixon-coles
  - mle
  - model
---

# `dixon_coles.py` — Dixon-Coles MLE Model

**Path:** `src/dixon_coles.py`

Advanced MLE model extending Poisson with tau (ρ) correction for low-scoring results, recency weighting (halflife ~4 years), and tournament importance (WC 2.5×).

**Key class:** `DixonColesModel` — fit, predict, predict_matches, add_features

**Disabled by default** (`config.dixon_coles.enabled = False`) — MLE is slow on large datasets.

See also: [[poisson_model.py]], [[elo.py]], [[feature_engineering.py]], [[config.py]], [[Poisson & Elo Models]]
