---
tags:
  - python-module
  - preprocessing
  - cleaning
---

# `preprocessing.py` — Data Preprocessing

**Path:** `src/preprocessing.py`

Initial data preparation before feature engineering. Handles: date parsing, league/season extraction, basic cleaning, column renaming for consistency across data sources.

**Key function:** `preprocess(df)` — standardises column names, converts data types, drops duplicate/empty rows, adds derived columns (season, day_of_week, etc.).

See also: [[data/cleaners.py]], [[data/loader.py]], [[feature_engineering.py]], [[Config System]]
