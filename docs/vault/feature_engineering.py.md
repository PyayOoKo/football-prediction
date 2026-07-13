---
tags:
  - python-module
  - features
  - engineering
  - core
---

# `feature_engineering.py` — Feature Creation Hub

**Path:** `src/feature_engineering.py`

Orchestrates all feature creation in strict order. Builds feature matrix (X) and target (y) from raw match data.

**Key function:** `build_features(df, is_training=True)` — calls add_elo_features → add_odds_features → add_player_features → add_xg_features → add_poisson_features → DC → rolling → H2H → league position → encode → ratios

**Leakage prevention:** Every rolling feature uses `.shift(1)` to exclude the current match.

See also: [[elo.py]], [[poisson_model.py]], [[dixon_coles.py]], [[xg_features.py]], [[odds_processing.py]], [[player_info.py]], [[config.py]], [[Feature Engineering Pipeline]], [[Runtime Sequence Diagrams]]
