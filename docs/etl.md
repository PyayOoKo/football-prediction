# ETL Pipeline

> Composable Extract → Validate → Clean → Normalize → Transform → Store pipeline for football data ingestion.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ETLPipeline                              │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  ┌──────────┐  ┌───────┐ │
│  │Extract  │→ │Validate  │→ │Clean  │→ │Normalize │→ │Transform│ │
│  │Stage 1  │  │Stage 2   │  │Stage 3│  │Stage 4   │  │Stage 5  │ │
│  └─────────┘  └──────────┘  └───────┘  └──────────┘  └───────┘ │
│                                                    ┌──────────┐ │
│                                                    │ Store    │ │
│                                                    │Stage 6   │ │
│                                                    └──────────┘ │
│  ┌──────────────┐  ┌────────────────────────────────────────┐   │
│  │ JobTracker   │  │ ProgressReporter                        │   │
│  │ (checkpoint) │  │ (tqdm progress bars)                    │   │
│  └──────────────┘  └────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

Each stage is **pluggable** and **testable** — wire in custom implementations for different data sources.

## Quick Start

```python
from src.etl import ETLPipeline
from src.etl.extract import CSVExtractor
from src.etl.clean import DataCleaner
from src.etl.normalize import DataNormalizer
from src.etl.store import DatabaseStore
from src.database.models import Match

pipeline = ETLPipeline(
    name="import_matches",
    source="football-data-co-uk",
    extractor=CSVExtractor("data/raw/results.csv"),
    cleaner=DataCleaner(fill_strategy="drop"),
    normalizer=DataNormalizer(
        team_name_columns=["home_team", "away_team"],
        date_columns=["date"],
    ),
    store=DatabaseStore(Match, unique_columns=["id"]),
    checkpoint=True,
)

result = pipeline.run()
print(result.status)  # StageStatus.SUCCESS
```

## Pipeline Stages

### 1. Extract (`BaseExtractor`)
| Implementation | Description |
|---|---|
| `CSVExtractor` | Reads data from CSV files |
| `APIExtractor` | Fetches data from REST endpoints |
| `ParquetExtractor` | Reads partitioned Parquet datasets |

### 2. Validate (`DataValidator`)
Runs configurable validation rules on the extracted data:
- Duplicate detection
- Schema validation (expected columns, types)
- Null/missing value checks
- Custom predicate rules

### 3. Clean (`DataCleaner`)
- Drop/flag duplicate rows
- Fill or drop missing values
- Type coercion (dates → datetime, goals → int)
- Outlier detection for odds and scores

### 4. Normalize (`DataNormalizer`)
- Team name normalisation (via `TeamNormalizer`)
- Date format standardisation (→ ISO 8601)
- League/competition code resolution
- Case folding and whitespace stripping

### 5. Transform (`DataTransformer`)
- Feature engineering (rolling stats, ELO, form)
- Column renaming and selection
- Aggregation and pivot operations
- Custom transform functions

### 6. Store (`DataStore`)
| Backend | Description | Best For |
|---|---|---|
| `DatabaseStore` | PostgreSQL upsert via SQLAlchemy | Production ingestion |
| `FileStore` | CSV or Parquet output | Data export, debugging |

## Storage Backend Details

### DatabaseStore
```python
store = DatabaseStore(
    model_class=Match,
    unique_columns=["id"],     # For upsert (ON CONFLICT DO NOTHING)
    batch_size=1000,           # Rows per transaction
)
```
Uses **parameterised SQL** via SQLAlchemy core — never raw string formatting.

### FileStore
```python
store = FileStore(
    output_dir="data/processed",
    format="parquet",          # "csv" or "parquet"
    filename="results_clean.parquet",
)
```

## Checkpoint & Resume

The pipeline supports checkpoint-based resume for long-running ingests:

```python
pipeline = ETLPipeline(
    ...,
    checkpoint=True,
)

# First run — saves checkpoint after each stage
result = pipeline.run()

# On failure — resume from last completed stage
result = pipeline.run(job_id=result.checkpoint_id)
```

## Progress Tracking

```
E0/2025:   0%|▍                                           | 1/6 [00:02<00:10,  2.0s/stage]
  ├─ Extract:    ✅ 15,234 rows in 0.52s
  ├─ Validate:   ✅ 15,234 rows in 1.23s (0 violations)
  ├─ Clean:      ✅ 15,210 rows in 0.89s
  ├─ Normalize:  ✅ 15,210 rows in 1.45s
  ├─ Transform:  🔄 Running ...
  └─ Store:      ⏳ Pending
```

## ETLResult Schema

```python
@dataclass
class ETLResult:
    pipeline_name: str
    source: str
    stages: dict[PipelineStage, StageResult]
    overall_status: StageStatus
    total_records: int
    total_errors: int
    total_duration_seconds: float
    started_at: datetime | None
    completed_at: datetime | None
    checkpoint_id: str | None

    @property
    def success(self) -> bool:
        return self.overall_status in (SUCCESS, WARNING)
```

## Configuration

```yaml
# scheduler_config.yaml
pipeline:
  name: import_matches
  source: football-data-co-uk
  batch_size: 1000
  parallel: false
  checkpoint: true

stages:
  extract:
    type: csv
    path: data/raw/results.csv
  clean:
    fill_strategy: drop
    deduplicate: true
  store:
    type: database
    table: matches
    unique_columns: ["id"]
```

## Performance

| Dataset Size | Pipeline | Time | Bottleneck |
|---|---|---|---|
| 10K rows (1 season) | Full ETL | ~3s | Normalize (team names) |
| 500K rows (50 seasons) | Full ETL | ~45s | Store (DB upsert) |
| 1M rows | Full ETL | ~90s | Store + Normalize |
| 10M rows | Extract only | ~5s | I/O (disk) |

## Testing ET Pipeline

```bash
# Run all ETL tests
python -m pytest tests/test_etl/ -v

# Run a single stage test
python -m pytest tests/test_etl/test_store.py -v

# Run pipeline integration test
python -m pytest tests/test_etl/test_pipeline.py -v
```
