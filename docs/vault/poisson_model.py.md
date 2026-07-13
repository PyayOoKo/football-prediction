---
tags:
  - python-module
  - poisson
  - model
  - goals
---

# `poisson_model.py` — Poisson Goals Model

**Path:** `src/poisson_model.py`

Predicts match scores using independent Poisson distributions. Core assumption: goals follow Pois(λ) where λ is expected goals.

**Key class:** `PoissonModel` — fit, predict, predict_matches, expected_goals, scoreline_table, add_poisson_features

**Formulas:** λ_home = μ_home × α_home × β_away, P(i,j) = Pois(i, λ_home) × Pois(j, λ_away)

See also: [[elo.py]], [[dixon_coles.py]], [[ensemble.py]], [[feature_engineering.py]], [[Poisson & Elo Models]]
