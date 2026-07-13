---
tags:
  - python-module
  - player
  - transfermarkt
---

# `player_info.py` — Player Information

**Path:** `src/player_info.py`

Integrates Transfermarkt squad value and player rating data as predictive features. Adds team-level aggregate features like total squad value, average age, foreign player ratio.

**Key function:** `add_player_features()` — fetches Transfermarkt data if enabled, computes team aggregates, attaches to match DataFrame. **Disabled by default** (`config.player_info.enabled = False`) due to slow scraping.

See also: [[feature_engineering.py]], [[config.py]], [[Data Collection Sources]], [[Feature Engineering Pipeline]]
