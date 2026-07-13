---
tags:
  - python-module
  - data
  - loader
---

# `data_loader.py` — Data Loading

**Path:** `src/data_loader.py`

Legacy data loader — loads cleaned CSV data from disk into DataFrames. Provides unified access to match results, odds, and team data.

**Key function:** `load_data(path=None)` — loads from config.default.data_path or provided path, returns DataFrame with standard columns.

See also: [[data/loader.py]], [[data/cleaners.py]], [[preprocessing.py]], [[Config System]]
