---
tags:
  - python-module
  - config
  - core
---

# `config.py` — Global Configuration Singleton

**Path:** `config.py`

Central configuration hub. All sub-modules import `config` from here. Auto-loads `.env` at import time.

**Pattern:** Single `Config` dataclass with 18 nested sub-config dataclasses.

**Key sub-configs:** `config.paths`, `config.data`, `config.train`, `config.elo`, `config.ensemble`, `config.features`, etc.

See also: [[Ensemble Model]], [[Feature Engineering Pipeline]], [[Config System]]
