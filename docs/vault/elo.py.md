---
tags:
  - python-module
  - elo
  - ratings
  - core
---

# `elo.py` — Elo Rating System

**Path:** `src/elo.py`

Dynamic team strength ratings for football. Adds `Home_Elo`, `Away_Elo`, `Elo_Difference` features.

**Key class:** `EloSystem` — expected_score, update_ratings, process_matches, regress_ratings

**Key features:** Goal-margin K-factor scaling, xG-margin support, season regression (1/3), host-nation bonus (+50), manual adjustments via `config.elo.adjustments`.

See also: [[config.py]], [[feature_engineering.py]], [[Poisson & Elo Models]]
