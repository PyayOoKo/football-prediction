# Data Versioning System

Track, compare, and rollback dataset imports for the football prediction platform.

## Overview

Every data import automatically creates an immutable **version snapshot**. The
system maintains a full version history, enables diffing between any two
versions, and supports rollback to any previous state — all optimised for
datasets exceeding 10 million rows.

```
data/versions/
├── v001/
│   ├── snapshot.parquet    # Full dataset (Parquet, zstd compressed)
│   ├── metadata.json       # Version metadata
│   └── delta.parquet       # Changes vs previous version
├── v002/
│   ├── snapshot.parquet
│   ├── metadata.json
│   └── delta.parquet
└── current -> v002/        # Symlink to active version
```

## Quick Start

### From Python

```python
from src.data_versioning import VersionManager

mgr = VersionManager()

# Create a version from a DataFrame
info = mgr.create_version(
    df=my_dataframe,
    source="football-data-co-uk",
    league="E0",
    season="2425",
    user="pipeline",
)
print(f"Created {info.version_id} ({info.n_rows:,} rows)")

# List all versions
for v in mgr.list_versions():
    print(v)

# Compare two versions
diff = mgr.compare("v001", "v002")
print(f"Inserted: {diff.n_inserted}, Deleted: {diff.n_deleted}")

# Rollback to a previous version
mgr.rollback("v001")

# Load the current dataset
df = mgr.load_current_data()

# Verify data integrity
results = mgr.verify()
```

### From the CLI

```bash
# Create a version from a CSV
python -m src.data_versioning.cli create-version \
    --file data/raw/results.csv \
    --source football-data \
    --league E0 \
    --season 2425 \
    --user pipeline \
    --notes "Weekly import"

# List all versions
python -m src.data_versioning.cli list-versions

# Compare two versions
python -m src.data_versioning.cli compare v001 v002

# Rollback
python -m src.data_versioning.cli rollback v001

# Verify integrity
python -m src.data_versioning.cli verify

# Show version info
python -m src.data_versioning.cli info v001
```

### Auto-Versioning (Integration)

Enable automatic versioning for all existing collector and importer functions:

```python
from src.data_versioning.integration import patch_all
patch_all()

# Now every call to collect_all(), collect_worldcup(), etc.
# automatically creates a versioned snapshot.
```

Or use the context manager:

```python
from src.data_versioning.integration import versioning_context

with versioning_context(source="football-data-co-uk"):
    from src.data_collection import collector
    report = collector.collect_all()  # auto-versioned
```

## Version Metadata

Each version stores the following metadata in `metadata.json`:

| Field             | Description                                    |
|-------------------|------------------------------------------------|
| `version_id`      | Unique identifier (e.g. `v001`)                |
| `created_at`      | ISO-8601 timestamp (UTC)                       |
| `source`          | Data source (e.g. `football-data-co-uk`)       |
| `league`          | League code (e.g. `E0`, `WC`)                 |
| `season`          | Season identifier (e.g. `2425`, `2026`)       |
| `n_rows`          | Number of rows in the snapshot                |
| `n_columns`       | Number of columns                             |
| `hash`            | SHA256 hash of the dataset content            |
| `data_path`       | Path to the Parquet snapshot file             |
| `delta_path`      | Path to the delta file (optional)             |
| `previous_version`| Previous version ID (optional, first is None) |
| `import_duration` | Time in seconds to import                    |
| `user`            | User or process that created this version     |
| `notes`           | Free-text notes                               |
| `tags`            | Arbitrary key-value tags                      |

## Change Detection (Diff)

The system detects three types of changes between versions:

- **Inserted** — rows present in the new version but not the old
- **Deleted** — rows present in the old version but not the new
- **Updated** — rows with the same key but different values

Rows are identified using a **fingerprint** — a SHA256 hash of the key columns
(e.g. `date`, `home_team`, `away_team`). The diff output includes counts,
changed columns, and optional sample rows.

```python
diff = mgr.compare("v001", "v002")
print(diff)
# DIFF: v001 → v002
#   Unchanged:    4,500
#   Inserted:        120
#   Updated:          45
#   Deleted:          30
#   Changed cols: ['home_goals', 'away_goals']
```

## Rollback

Rollback restores a previous version as the "current" dataset. By default, a
backup of the current state is created first:

```python
# Rollback to v001 (auto-creates backup_v001 first)
mgr.rollback("v001")

# Rollback without backup
mgr.rollback("v001", create_backup=False)
```

From CLI:

```bash
python -m src.data_versioning.cli rollback v001
```

## Large Dataset Optimisation

The system is designed for datasets exceeding 10 million rows:

| Technique | Description |
|-----------|-------------|
| **Parquet format** | Columnar storage with zstd compression — 5-10x smaller than CSV |
| **Chunked writing** | Writes in 500k-row chunks to limit memory |
| **Chunked hashing** | Streaming SHA256 — never materialises full CSV string |
| **Column pruning** | Load only the columns you need |
| **Row-group filtering** | Parquet row-group statistics for fast scans |
| **Vectorised fingerprinting** | Uses `pd.util.hash_pandas_object` instead of row-wise apply |
| **Delta storage** | Only the diff is stored between versions (optional) |

## Integrity Verification

Every version stores a SHA256 hash of its content. Verify that stored data
hasn't been tampered with:

```python
mgr.verify("v001")         # single version
mgr.verify()               # all versions
```

```bash
python -m src.data_versioning.cli verify --version v001
python -m src.data_versioning.cli verify
```

## CLI Reference

```
Commands:
  create-version  Create a new dataset version from a file
  list-versions   List all dataset versions
  compare         Compare two versions
  rollback        Rollback to a previous version
  verify          Verify data integrity
  info            Show detailed version information

Global options:
  --data-dir PATH  Version storage directory (default: data/versions)

create-version:
  --file, -f PATH     Input data file (CSV or Parquet)
  --source, -s NAME   Data source identifier
  --league, -l CODE   League code
  --season CODE       Season identifier
  --user, -u NAME     User creating the version
  --notes, -n TEXT    Optional notes
  --tag KEY=VALUE     Key-value tags (repeatable)

compare:
  BASE     Base (older) version ID
  TARGET   Target (newer) version ID
  --no-samples  Don't include sample row data

rollback:
  VERSION_ID    Version to restore
  --no-backup   Don't create a backup
  --user NAME   User performing the rollback
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  VersionManager                      │
│  Orchestrates version lifecycle                      │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ storage  │  │  differ  │  │ integration       │   │
│  │ (Parquet │  │ (delta   │  │ (collector hooks) │   │
│  │  + hash) │  │  detect) │  │                   │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────┤
│                     CLI (argparse)                    │
│  create-version  list-versions  compare  rollback    │
└─────────────────────────────────────────────────────┘
```

## Requirements

- **Python** 3.10+
- **pandas** 2.0+
- **pyarrow** (for Parquet support)
- **numpy**

All dependencies are in `requirements.txt`.

## Tests

```bash
# Run all versioning tests
pytest tests/test_data_versioning/ -v

# Run with coverage
pytest tests/test_data_versioning/ --cov=src.data_versioning -v
```
