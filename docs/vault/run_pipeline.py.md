---
tags:
  - python-script
  - pipeline
  - orchestration
---

# `run_pipeline.py` — Automated Pipeline Orchestrator

**Path:** `run_pipeline.py`

The main orchestration script. Runs the full ML pipeline in 5 steps:
1. **Download** — collect latest match data
2. **Preprocess** — clean and validate
3. **Retrain** — rebuild ensemble model if stale
4. **Predict** — generate predictions for recent matches
5. **Report** — save and print summary

**CLI flags:** `--skip-download`, `--skip-train`, `--lightweight`, `--force-retrain`, `--version`

See also: [[ensemble.py]], [[feature_engineering.py]], [[config.py]], [[Quick Start Guide]], [[Runtime Sequence Diagrams]]
