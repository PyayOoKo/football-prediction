# Scheduler

> Automated task orchestration for the football prediction pipeline — supports cron, Windows Task Scheduler, and manual CLI invocations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ScheduleConfig                            │
│  (pipeline_name, tasks[], parallel, abort_on_failure, ...)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                      TaskEngine                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Dependency resolution (topological sort)            │    │
│  │  Retry logic (exponential backoff)                   │    │
│  │  Task execution with timeout                         │    │
│  │  Structured RunReport output                         │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────┬──────────────────────────────────────────────────┘
           │
┌──────────▼──────────┐  ┌──────────▼──────────┐
│   Built-in Tasks     │  │   Custom Tasks       │
│                      │  │                      │
│ • download_fixtures  │  │  (register via       │
│ • validate_data      │  │   engine.register()) │
│ • clean_data         │  │                      │
│ • update_database    │  │                      │
│ • backup_database    │  │                      │
│ • generate_logs      │  │                      │
└─────────────────────┘  └──────────────────────┘
```

## Quick Start

```bash
# Run all scheduled tasks
python -m src.scheduler.cli run

# Run specific tasks
python -m src.scheduler.cli run --tasks download_fixtures,validate_data

# Run with custom config
python -m src.scheduler.cli run --config my_config.yaml

# List available tasks
python -m src.scheduler.cli list

# Show last run report
python -m src.scheduler.cli status

# Install Windows scheduled task
python -m src.scheduler.cli install-windows
```

## Task Definitions

### 1. `download_fixtures`
Downloads match data from configured sources (football-data.co.uk).
- **Timeout:** 300s
- **Retries:** 2
- **Dependencies:** None
- **Output:** Raw CSV files in `data/raw/`

### 2. `validate_data`
Runs 9 validation checks on downloaded data.
- **Timeout:** 120s
- **Dependencies:** `download_fixtures`
- **Output:** HTML/CSV/JSON validation reports in `reports/scheduler/`

### 3. `clean_data`
Deduplicates, normalises, and archives raw data.
- **Timeout:** 180s
- **Dependencies:** `validate_data`
- **Output:** Cleaned CSV in `data/processed/`

### 4. `update_database`
Ingests cleaned data into PostgreSQL + retrains the ensemble model.
- **Timeout:** 600s
- **Retries:** 2
- **Dependencies:** `clean_data`
- **Output:** Database rows, `models/ensemble_model.joblib`

### 5. `backup_database`
Creates a compressed database backup with retention policy.
- **Timeout:** 300s
- **Dependencies:** None (parallel group: `maintenance`)
- **Output:** `data/backups/football_db_*.sql.gz`

### 6. `generate_logs`
Rotates logs, archives old reports, writes run summary.
- **Timeout:** 60s
- **Dependencies:** `backup_database`
- **Output:** Compressed logs, `logs/run_summary_*.json`

## Engine API

```python
from src.scheduler import TaskEngine, ScheduleConfig

# Default configuration (all 6 tasks)
engine = TaskEngine()

# Custom configuration
from src.scheduler import Task

config = ScheduleConfig(
    pipeline_name="my_pipeline",
    tasks=[
        Task(name="download_fixtures", description="Scrape fixtures"),
        Task(name="validate", description="Check data quality",
             dependencies=["download_fixtures"]),
    ],
    abort_on_failure=True,
)
engine = TaskEngine(config)

# Register custom tasks
def my_custom_task(cfg: ScheduleConfig) -> TaskResult:
    # ... your logic ...
    return TaskResult(task_name="my_task", status=TaskStatus.SUCCESS)

engine.register("my_custom_task", my_custom_task)

# Run
report = engine.run_all()
print(f"{report.succeeded}/{report.total_tasks} tasks succeeded")
```

## Run Report

```python
@dataclass
class RunReport:
    pipeline_name: str
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float
    task_results: dict[str, TaskResult]
    total_tasks: int
    succeeded: int
    failed: int
    skipped: int
    errors: list[str]

    @property
    def success(self) -> bool:
        return self.failed == 0
```

## Task DAG

```
download_fixtures
        │
        ▼
  validate_data
        │
        ▼
   clean_data
        │
        ▼
  update_database
        │
   ┌────┴────┐
   │         │
   ▼         ▼
  backup   generate
  (parallel)  (parallel)
        │
        ▼
  generate_logs
```

## Windows Task Scheduler Integration

```bash
# Install as daily scheduled task (runs at 06:00)
python -m src.scheduler.cli install-windows

# This creates a Windows scheduled task named
# "FootballPredictionPipeline" in Task Scheduler
```

Python implementation:
```python
from src.scheduler import WindowsTaskManager

manager = WindowsTaskManager()
manager.create_scheduled_task(
    name="FootballPredictionPipeline",
    script="run_pipeline.py",
    time="06:00",
    days_of_week="Mon,Tue,Wed,Thu,Fri,Sat,Sun",
)
```

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `pipeline_name` | `football_pipeline` | Name for this pipeline instance |
| `parallel_groups` | `False` | Run same-group tasks concurrently |
| `abort_on_failure` | `True` | Stop pipeline if a task fails |
| `log_dir` | `logs/scheduler` | Log directory |
| `report_dir` | `reports/scheduler` | Report output directory |
| `backup_dir` | `data/backups` | Database backup directory |
| `backup_retention_days` | `7` | Days to keep old backups |
| `max_log_age_days` | `30` | Days before log rotation |

## Monitoring

```bash
# Production monitoring with systemd timer (Linux)
[Unit]
Description=Football Prediction Pipeline

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target

# Or use cron
0 6 * * * cd /opt/football-prediction && .venv/bin/python run_pipeline.py
```
