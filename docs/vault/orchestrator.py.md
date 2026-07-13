---
tags:
  - python-module
  - orchestrator
  - pipeline
  - features
---

# `orchestrator.py` — Feature Pipeline Orchestrator

**Path:** `src/feature_framework/orchestrator.py`

Production-grade pipeline execution with discovery, DAG resolution, caching, retry, resume, parallelism, progress tracking, logging, metrics, and incremental updates.

**Key class:** `FeatureOrchestrator` — run(), resume(), recompute_feature(), list_features(), feature_status()

**Report type:** `OrchestratorReport` — summary(), to_dict(), per-feature execution records

**Enums:** `OrchestratorStage` (discover, resolve, compute, validate, store), `FeatureStatus` (pending, running, completed, skipped, failed, cached)

**72 tests** in `tests/test_feature_framework/test_orchestrator.py`

See also: [[orchestrator_cli.py]], [[Feature Orchestrator]], [[Feature Validation Framework]], [[Betting Market Features]]
