---
tags:
  - python-module
  - cli
  - orchestrator
  - pipeline
---

# `orchestrator_cli.py` — Orchestrator CLI

**Path:** `src/feature_framework/orchestrator_cli.py`

5 CLI commands wrapping `FeatureOrchestrator`:

| Command | Description |
|---------|-------------|
| `build-features` | Full pipeline execution with input → output |
| `validate-features` | FeatureValidator integration with report output |
| `recompute-feature` | Single feature recompute by name |
| `list-features` | Enumerate with type/category filtering |
| `feature-status` | Detailed status with dependencies, columns |

**Entry point:** `main(argv)` — argparse-based, supports `--config`, `--quiet`, `--verbose`, `--force`, `--no-parallel`, `--cache-dir`, `--checkpoint-dir`, `--max-workers`, `--max-retries`

See also: [[orchestrator.py]], [[Feature Orchestrator]]
