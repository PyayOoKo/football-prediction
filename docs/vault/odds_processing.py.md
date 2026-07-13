---
tags:
  - python-module
  - odds
  - processing
---

# `odds_processing.py` — Odds Processing

**Path:** `src/odds_processing.py`

Cleans and normalises bookmaker odds from raw data. Detects odds format (decimal, fractional, US), converts to decimal, fills missing odds via implied probability averaging across bookmakers.

**Key function:** `add_odds_features()` — adds columns for best_odds_home/draw/away, implied probabilities, and odds-derived features.

See also: [[value_betting.py]], [[feature_engineering.py]], [[config.py]], [[Feature Engineering Pipeline]]
