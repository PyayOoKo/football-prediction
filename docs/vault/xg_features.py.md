---
tags:
  - python-module
  - xg
  - expected-goals
---

# `xg_features.py` — Expected Goals Features

**Path:** `src/xg_features.py`

Generates rolling xG (expected goals) features from FBref/UnderStat shot data. Computes team attack/defence strength from recent xG, xGA, xG difference.

**Key function:** `add_xg_features()` — requires xG columns in data; rolls 5/10/38-game windows with `.shift(1)` leakage protection.

Auto-detects xG column naming: `xG`, `xg`, `xG_home`, `away_xG`, etc.

See also: [[feature_engineering.py]], [[Data Collection Sources]], [[Feature Engineering Pipeline]]
