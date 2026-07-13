# Feature Pipeline Orchestrator

**Production-grade pipeline execution** with automatic discovery, dependency resolution, caching, retry, resume, parallelism, progress tracking, logging, metrics, and incremental updates.

## Responsibilities

| Responsibility | Implementation |
|----------------|----------------|
| **Discover** | Auto-find feature transformers via `FeaturePluginRegistry` |
| **Resolve DAG** | Build dependency graph; topological sort with cycle detection |
| **Execute order** | Compute features in DAG order (parallel where possible) |
| **Cache** | Store intermediate results with content-based checksums |
| **Retry** | Exponential backoff on transient failures (configurable attempts) |
| **Resume** | Checkpoint/restart after interruption |
| **Parallel** | Thread/process pool execution (configurable workers) |
| **Progress** | `tqdm` progress bars + per-feature timing |
| **Logging** | Structured JSON logs per pipeline run |
| **Metrics** | Timing, counts, success rates, per-feature stats |
| **Incremental** | Skip features that haven't changed since last run |

## Quick Start

### Programmatic Use

```python
from src.feature_framework import FeatureOrchestrator
import pandas as pd

# Create orchestrator
orchestrator = FeatureOrchestrator(
    config_path="features.yaml",
    show_progress=True,
    max_retries=2,
    parallel=True,
)

# Load data
df = pd.read_csv("matches.csv")

# Run pipeline
report = orchestrator.run(
    entity_type="dataframe",
    df=df,
    trigger="manual",
)

print(report.summary())
```

### CLI Usage

```bash
# Build features from CSV data
python -m src.feature_framework.orchestrator_cli build-features \
    --input matches.csv --output features.csv

# Validate computed features
python -m src.feature_framework.orchestrator_cli validate-features \
    --input features.csv

# Recompute a single feature
python -m src.feature_framework.orchestrator_cli recompute-feature \
    elo_rating --input matches.csv

# List all configured features
python -m src.feature_framework.orchestrator_cli list-features

# Get status of a feature
python -m src.feature_framework.orchestrator_cli feature-status elo_rating
```

## Architecture

```
FeatureOrchestrator
│
├── run()                              # Main entry point
│   ├── Stage 1: DISCOVER              # Auto-find features
│   ├── Stage 2: RESOLVE               # Build DAG, topo-sort
│   ├── Stage 3: COMPUTE               # Execute (DataFrame or Entity mode)
│   │   ├── Cache check                # Skip unchanged features
│   │   ├── Retry loop                 # Exponential backoff
│   │   └── Progress tracking          # tqdm bars
│   ├── Stage 4: VALIDATE              # FeatureValidator checks
│   └── Checkpoint save                # For resume support
│
├── resume()                           # Resume from checkpoint
├── recompute_feature()                # Single feature recompute
├── list_features()                    # Enumerate features
├── feature_status()                   # Feature details
├── clear_cache()                      # Cache management
└── list_checkpoints()                 # Checkpoint listing
```

## Modes of Operation

### DataFrame Mode

Processes features by passing the entire DataFrame through each transformer sequentially (in DAG order). Best for batch processing where all data fits in memory.

```python
report = orchestrator.run(
    entity_type="dataframe",
    df=my_dataframe,
    trigger="scheduled",
)
```

### Entity Mode

Processes features one entity at a time via the `FeatureStore` and `FeatureComputationEngine`. Best for production serving where entities are processed incrementally.

```python
report = orchestrator.run(
    entity_type="match",
    entity_ids=[101, 102, 103],
    trigger="scheduled",
)
```

## Configuration

Features can be defined in a YAML/JSON config file:

```yaml
version: "1.0"
pipeline:
  default_entity_type: match
  show_progress: true
  max_retries: 2
  parallel: true

features:
  - name: elo_rating
    version: 1
    type: elo
    category: elo_rating
    data_type: float
    output_columns: [h_elo, a_elo]
    dependencies: []
    params:
      k: 32
      home_advantage: 100

  - name: team_form
    version: 1
    type: team_form
    category: form
    data_type: float
    output_columns: []
    dependencies: []
    params:
      windows: [3, 5, 10]
      contexts: [overall, home, away]
```

Or inline via dict:

```python
orchestrator = FeatureOrchestrator(
    config_dict={
        "features": [
            {"name": "elo", "type": "elo", "category": "rating"},
        ],
    },
)
```

## Caching

The orchestrator caches intermediate results using content-based checksums:

- **Cache key**: Generated from feature name
- **Cache validation**: Row count comparison
- **Auto-invalidation**: Row count changes invalidate cache
- **Manual clear**: `orchestrator.clear_cache("feature_name")` or `orchestrator.clear_cache()`

```python
# Disable caching for a run
report = orchestrator.run(
    df=df,
    force_recompute=True,  # Skip all caches
)

# Disable incremental mode entirely
orchestrator.incremental = False
```

## Retry Logic

Transient failures are retried with exponential backoff:

| Retry | Delay |
|-------|-------|
| 1st retry | `retry_delay * 2^0` |
| 2nd retry | `retry_delay * 2^1` |
| Nth retry | `retry_delay * 2^(n-1)` |

Configure via constructor:

```python
orchestrator = FeatureOrchestrator(
    max_retries=3,
    retry_delay=1.0,  # seconds
)
```

## Checkpoint & Resume

When a pipeline run has failures, a checkpoint is automatically saved:

```python
# Initial run (some features fail)
report = orchestrator.run(df=df)

# Resume from checkpoint
report2 = orchestrator.resume(report.checkpoint_path, df=df)
```

## Report Format

The `OrchestratorReport` provides a complete record of the run:

```python
report = orchestrator.run(df=df)

print(report.summary())
#   FEATURE PIPELINE ORCHESTRATOR REPORT
#   =========================================
#   Run ID:       abc12345...
#   Trigger:      manual
#   Duration:     12.34s
#   Result:       ✅ PASS
#   Features:     5 configured, 4 computed, 1 cached, 1 failed

# Export to dict
data = report.to_dict()

# Per-feature details
for name, record in report.features.items():
    print(f"{name}: {record.status} in {record.duration:.3f}s")
```

## Metrics

Run-level metrics are automatically collected:

| Metric | Description |
|--------|-------------|
| `total_duration` | Total wall-clock time |
| `avg_feature_time` | Average time per completed feature |
| `success_rate` | `n_computed / n_features` |
| `features_per_second` | Throughput rate |
| `entities_per_second` | Entity throughput |

## Integration with Feature Validation

The orchestrator automatically validates computed features using `FeatureValidator`:

```python
# Validation runs after computation (Stage 4)
report = orchestrator.run(df=df)
if report.validation:
    print(f"Violations: {report.validation['total_violations']}")
    if not report.validation["passed"]:
        print("Some features failed validation!")
```

## CLI Reference

### build-features

```bash
python -m src.feature_framework.orchestrator_cli build-features \
    --input PATH \
    --output PATH \
    [--config PATH] \
    [--entity-type TYPE] \
    [--trigger TRIGGER] \
    [--force] \
    [--no-parallel] \
    [--quiet] \
    [--verbose] \
    [--max-workers N] \
    [--max-retries N] \
    [--cache-dir PATH] \
    [--checkpoint-dir PATH]
```

### validate-features

```bash
python -m src.feature_framework.orchestrator_cli validate-features \
    --input PATH \
    [--output PATH] \
    [--checks COMMA,SEPARATED] \
    [--quiet] \
    [--verbose]
```

### recompute-feature

```bash
python -m src.feature_framework.orchestrator_cli recompute-feature \
    NAME \
    [--input PATH] \
    [--entity-type TYPE] \
    [--quiet]
```

### list-features

```bash
python -m src.feature_framework.orchestrator_cli list-features \
    [--type TYPE] \
    [--category CATEGORY] \
    [--enabled-only] \
    [--verbose]
```

### feature-status

```bash
python -m src.feature_framework.orchestrator_cli feature-status \
    NAME \
    [--verbose]
```

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Feature definition YAML/JSON | — |
| `--quiet` | Suppress progress bars | `False` |
| `--verbose` | Detailed debug output | `False` |
| `--force` | Skip all caches | `False` |
| `--no-parallel` | Single-threaded execution | `False` |
| `--max-workers` | Worker pool size | CPU count |
| `--max-retries` | Retry attempts | `2` |
| `--cache-dir` | Cache location | `.cache/features/` |
| `--checkpoint-dir` | Checkpoints | `.checkpoints/` |

## Error Handling

- **FeatureDependencyCycleError**: Circular dependencies detected during DAG resolution
- **FeatureNotFoundError**: Feature not found in configuration or registry
- **FeatureComputationError**: Runtime computation failure (caught and retried)
- **IO errors**: Missing input files, permission issues (caught and reported)

All errors are captured in the report's `errors` list and do not crash the orchestrator.
