---
tags:
  - football-prediction
  - scheduler
  - dashboard
  - etl
created: 2026-07-12
---

# ⏰ Scheduler, Dashboard & ETL

> Automation, visualisation, and data pipeline infrastructure.

See also: [[Architecture Overview]], [[Config System]], [[Scripts Reference]]

---

## Scheduler System

**Files:** `src/scheduler/engine.py`, `cron_scheduler.py`, `windows_scheduler.py`

```mermaid
flowchart LR
    subgraph "Scheduler Config"
        YAML["scheduler_config.yaml"] --> SCHED["Scheduler Setup"]
    end
    
    subgraph "Task Engine"
        SCHED --> ENGINE["TaskEngine<br/>src/scheduler/engine.py"]
        ENGINE --> TASKS["Registered Tasks"]
        TASKS --> T1["download_fixtures"]
        TASKS --> T2["validate_data"]
        TASKS --> T3["clean_data"]
        TASKS --> T4["update_database"]
        TASKS --> T5["backup_database"]
        TASKS --> T6["generate_logs"]
    end
    
    subgraph "Platform Schedulers"
        SCHED --> CRON["cron_scheduler.py"]
        SCHED --> WIN["windows_scheduler.py"]
    end
```

### Task Engine Features

| Feature | Description |
|---------|-------------|
| **Dependency resolution** | Topological sort of tasks |
| **Retry logic** | Configurable retry count with linear backoff |
| **Abort-on-failure** | Can stop the chain on first error |
| **Reporting** | `RunReport` with per-task status |

### Built-in Tasks

| Task | Function | Description |
|------|----------|-------------|
| `download_fixtures` | `download_fixtures()` | Fetch latest match data |
| `validate_data` | `validate_data()` | Run validation checks |
| `clean_data` | `clean_data()` | Clean and standardise |
| `update_database` | `update_database()` | Persist to PostgreSQL |
| `backup_database` | `backup_database()` | Create backup |
| `generate_logs` | `generate_logs()` | Generate summary logs |

---

## ETL Pipeline

**File:** `src/etl/pipeline.py`

```mermaid
flowchart LR
    EXTRACT["Extract<br/>CSV/API/Database"] --> VALIDATE["Validate<br/>Schema + Data checks"]
    VALIDATE --> CLEAN["Clean<br/>Dedup, missing values"]
    CLEAN --> NORMALIZE["Normalize<br/>Types, encodings"]
    NORMALIZE --> TRANSFORM["Transform<br/>Business logic"]
    TRANSFORM --> STORE["Store<br/>Database/CSV"]
    
    EXTRACT -.->|checkpoint| TRACKER["JobTracker<br/>Checkpoint/Resume"]
    STORE -.-> TRACKER
```

### Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| Extract | `BaseExtractor` | Read from CSV, API, or database |
| Validate | `DataValidator` | Schema + data integrity checks |
| Clean | `DataCleaner` | Dedup, handle missing values |
| Normalize | `DataNormalizer` | Type coercion, encoding |
| Transform | `DataTransformer` | Business logic transforms |
| Store | `DataStore` | Write to database or file |

### Features

- **Checkpoint/Resume** — can restart from a failed stage
- **Progress reporting** — per-stage status via `ProgressReporter`
- **Job tracking** — `JobTracker` persists state across runs

---

## Dashboard (Streamlit)

**File:** `src/app/dashboard.py`

```mermaid
graph TD
    DASH["Streamlit Dashboard<br/>src/app/dashboard.py"] --> PG1["1_Predict.py<br/>Match Predictions"]
    DASH --> PG2["2_Value_Bets.py<br/>Value Bet Opportunities"]
    DASH --> PG3["3_Backtest.py<br/>Historical Simulation"]
    DASH --> PG4["4_WorldCup.py<br/>World Cup Predictions"]
    DASH --> UTILS["utils.py<br/>Shared helpers"]
```

### Pages

| Page | File | What It Shows |
|------|------|---------------|
| **Predict** | `1_Predict.py` | Match outcome probabilities for upcoming fixtures |
| **Value Bets** | `2_Value_Bets.py` | Positive EV betting opportunities |
| **Backtest** | `3_Backtest.py` | Historical simulation results and charts |
| **World Cup** | `4_WorldCup.py` | World Cup-specific predictions and bracket |

### Launch

```bash
python run_dashboard.py
```
