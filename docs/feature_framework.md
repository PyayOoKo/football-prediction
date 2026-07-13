# Feature Engineering Framework

> Production-grade infrastructure for creating, versioning, computing, and serving features for every ML model in the football prediction platform.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     FeaturePipeline (orchestrator)                │
│  Resolves feature DAG → parallel computation → validate → store  │
└────┬──────────┬────────────┬────────────┬───────────┬────────────┘
     │          │            │            │           │
┌────▼────┐┌───▼─────┐┌─────▼──────┐┌───▼──────┐┌───▼──────────┐
│Feature  ││Parallel ││Feature     ││Feature   ││FeatureStore  │
│Transform││Computer ││Config      ││Plugin    ││(from         │
│(ABC)    ││(pool)   ││(YAML)     ││Registry  ││ src.feature_ │
└─────────┘└─────────┘└────────────┘└──────────┘│  store)      │
                                                 └──────────────┘
```

### Layers

| Layer | Package | Purpose |
|-------|---------|---------|
| **Framework** | `src/feature_framework/` | Pipeline orchestration, plugin discovery, config-driven setup |
| **Storage** | `src/feature_store/` | Feature values DB, definitions registry, caching, lineage |
| **Database** | `src/database/` | SQLAlchemy engine, session management, migrations |

---

## Quick Start

### 1. Define Features in YAML

```yaml
# features.yaml
version: "1.0"
pipeline:
  default_entity_type: match
  show_progress: true
  parallel: true
  max_workers: 4

features:
  - name: elo_rating
    version: 1
    description: "Elo ratings for home and away teams"
    type: elo
    category: elo_rating
    data_type: float
    computation_time: medium
    output_columns: [h_elo, a_elo, elo_difference]
    params:
      k: 32
      home_advantage: 100
    validation:
      min: 1000
      max: 2500
    dependencies: []
    tags: [rating, historical]

  - name: home_attack_strength
    version: 2
    description: "Rolling average home goals scored"
    type: rolling_stat
    category: attack_strength
    data_type: float
    computation_time: fast
    output_columns: [h_goals_scored_avg5]
    params:
      window: 5
      stat: goals_scored
      role: home
    validation:
      min: 0
      max: 5
    dependencies: [elo_rating]
    tags: [form, attack]
```

### 2. Run the Pipeline

```python
from src.feature_framework import FeaturePipeline

# From YAML config
pipeline = FeaturePipeline(config_path="features.yaml")

# Or from dict
pipeline = FeaturePipeline(config_dict={
    "features": [
        {
            "name": "elo_rating",
            "type": "elo",
            "category": "elo_rating",
            "params": {"k": 32},
        }
    ]
})

# DataFrame mode
import pandas as pd
df = pd.read_csv("data/processed/matches.csv")
report = pipeline.run(entity_type="dataframe", df=df, trigger="manual")
print(report.to_dict())

# Entity mode (requires database)
report = pipeline.run(
    entity_type="match",
    entity_ids=[1, 2, 3, 4, 5],
    trigger="scheduled",
)
report.print_summary()
```

### 3. Resume an Interrupted Batch

```python
report = pipeline.resume(batch_id="some-batch-id")
```

---

## Core Concepts

### FeatureTransformer (ABC)

Every feature extends `FeatureTransformer`. Subclasses must implement `transform()`.

```python
from src.feature_framework import FeatureTransformer
from src.feature_framework.models import TransformContext
import pandas as pd

class MyFeatureTransformer(FeatureTransformer):
    # Class-level metadata
    name = "my_feature"
    version = 1
    description = "My custom feature"
    dependencies = ["dependency_feature"]
    output_columns = ["my_output_col"]
    data_type = "float"
    computation_time = "fast"
    category = "my_category"
    tags = ["custom"]

    def transform(self, df: pd.DataFrame, context: TransformContext | None = None) -> pd.DataFrame:
        df["my_output_col"] = df["some_input"] * 2
        return df
```

#### Lifecycle

1. **`__init__`** — store parameters from config
2. **`init(context)`** — optional setup (load reference data, etc.)
3. **`validate_input(df)`** — optional input validation (returns error list)
4. **`transform(df, context)`** — core computation (must implement)
5. **`validate_output(df)`** — optional output validation (returns error list)

### Pipeline Modes

| Mode | Input | Use Case |
|------|-------|----------|
| **DataFrame** | `pd.DataFrame` | Batch processing, backtesting, eval |
| **Entity** | `entity_ids` list | Production, incremental, scheduled |

---

## Configuration Reference

### Pipeline Config (`pipeline:` section)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_entity_type` | string | `match` | Default entity type |
| `show_progress` | bool | `true` | Enable progress bars (tqdm) |
| `max_retries` | int | `0` | Max retries per feature |
| `parallel` | bool | `true` | Enable parallel computation |
| `max_workers` | int | `4` | Max parallel workers |

### Feature Definition (`features[]` items)

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Unique feature name |
| `type` | Yes | string | — | Matches `FeatureTransformer.name` |
| `category` | Yes | string | — | Thematic category |
| `version` | No | int | `1` | Feature version |
| `description` | No | string | `""` | Human-readable description |
| `data_type` | No | string | `float` | `float`, `int`, `str`, `bool`, `categorical`, `datetime` |
| `computation_time` | No | string | `fast` | `fast`, `medium`, `slow` |
| `output_columns` | No | list[str] | `[]` | Column names produced |
| `params` | No | dict | `{}` | Parameters for the transformer |
| `validation` | No | dict | `{}` | Validation rules (min, max, nullable, etc.) |
| `dependencies` | No | list[str] | `[]` | Feature dependencies |
| `tags` | No | list[str] | `[]` | Searchable tags |
| `author` | No | string | `system` | Who created this |
| `source` | No | string | `""` | Source data identifier |
| `enabled` | No | bool | `true` | Enable/disable feature |

---

## Plugin System

Features auto-discover via `FeaturePluginRegistry` through three mechanisms:

### 1. Entry Points (pyproject.toml)

```toml
[project.entry-points."feature_transformers"]
my_feature = "src.feature_framework.transformers.my_feature"
```

### 2. Package Convention

Place transformer files in `src/feature_framework/transformers/`:

```
src/feature_framework/transformers/
├── __init__.py
├── elo_transformer.py
├── rolling_stats.py
└── h2h_stats.py
```

Each module is scanned for `FeatureTransformer` subclasses.

### 3. Explicit Registration

```python
pipeline = FeaturePipeline()
pipeline.register_transformer_class(MyFeatureTransformer)

# Or via the plugin registry
from src.feature_framework import FeaturePluginRegistry
registry = FeaturePluginRegistry()
registry.register(MyFeatureTransformer)
```

---

## Development Guide

### Creating a New Feature

1. **Create the transformer class** — subclass `FeatureTransformer`
2. **Set class-level metadata** — name, version, output_columns, etc.
3. **Implement `transform()`** — the computation logic
4. **Add to config** — in `features.yaml`
5. **Register plugin** — via entry points or package convention

### Best Practices

- **Leakage prevention**: Never use future data to compute current features
- **Deterministic**: Same input → same output (reproducible)
- **Idempotent**: Re-computing a feature should produce the same result
- **Column names**: Use prefixes (`h_`, `a_`, `h2h_`) for home/away variants
- **Validation**: Set min/max bounds in `validation_rules` 
- **Dependencies**: Declare all upstream feature dependencies explicitly

### Running Tests

```bash
# Unit tests
python -m pytest tests/test_feature_framework/ -v

# Integration tests
python -m pytest tests/test_feature_framework/ -v -m integration

# All tests
python -m pytest tests/ -v
```

---

## Custom Exceptions

| Exception | Raised When |
|-----------|-------------|
| `FeatureEngineError` | Base for all feature errors |
| `FeatureComputationError` | Feature computation fails for an entity |
| `FeatureNotFoundError` | Feature definition/value not found |
| `FeatureValidationError` | Computed value fails validation |
| `FeatureDependencyCycleError` | Circular dependency detected |
| `FeatureConfigError` | Invalid YAML/JSON config |

---

## Package Reference

### `src.feature_framework`

| Module | Key Classes/Functions |
|--------|----------------------|
| `__init__` | Package exports, version `0.1.0` |
| `base.py` | `FeatureTransformer`, `FeaturePipelineABC` |
| `models.py` | `ComputationResult`, `PipelineReport`, `FeatureMetadata`, `TransformContext`, `FeatureSet` |
| `config.py` | `FeatureConfig`, `FeatureDefinitionSchema`, `load_feature_config()` |
| `plugins.py` | `FeaturePluginRegistry` |
| `pipeline.py` | `FeaturePipeline` (main orchestrator) |
| `parallel.py` | `ParallelComputer`, `make_thread_pool()`, `make_process_pool()` |
| `decorators.py` | `@timeit`, `@log_call`, `@retry` |
| `exceptions.py` | Custom exceptions |

### `src.feature_store` (backend)

| Module | Purpose |
|--------|---------|
| `registry.py` | Feature definition registry with versioning |
| `store.py` | CRUD operations for feature values |
| `computers.py` | `FeatureComputer` ABC, `ComputerRegistry` |
| `computation.py` | Batch orchestration, resume, lazy loading |
| `validation.py` | Validation rules engine |
| `cache.py` | Look-aside caching |
| `models.py` | SQLAlchemy ORM models |
