# Automation System — Football Prediction

> **Version:** 2.0.0  
> **Last Updated:** July 2026  
> **Status:** ✅ 100% Automated

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Pipeline Flow](#2-pipeline-flow)
3. [Scheduling Details](#3-scheduling-details)
4. [Task Reference](#4-task-reference)
5. [Notification System](#5-notification-system)
6. [Monitoring Setup](#6-monitoring-setup)
7. [Alerting Rules](#7-alerting-rules)
8. [Data Drift Detection](#8-data-drift-detection)
9. [Pipeline Health Dashboard](#9-pipeline-health-dashboard)
10. [GitHub Actions Automation](#10-github-actions-automation)
11. [Troubleshooting Guide](#11-troubleshooting-guide)
12. [Manual Override](#12-manual-override)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     SCHEDULER LAYER                        │
│  ┌────────────────────────────────────────────────────┐   │
│  │  TaskEngine (src/scheduler/engine.py)              │   │
│  │  • Task registry & dependency resolution           │   │
│  │  • Sequential/parallel execution                   │   │
│  │  • Retry with linear backoff                       │   │
│  │  • Abort-on-failure policy                         │   │
│  └────────────────────────────────────────────────────┘   │
│                           │                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Task Implementations (10 tasks)                    │   │
│  │  ┌─────────────┐   ┌───────────────────────┐      │   │
│  │  │ Core (6)    │   │ Daily Pipeline (4)     │      │   │
│  │  │ • download  │   │ • daily_data_pipeline  │      │   │
│  │  │ • validate  │   │ • daily_feature_comp   │      │   │
│  │  │ • clean     │   │ • daily_model_retrain  │      │   │
│  │  │ • update_db │   │ • daily_predictions    │      │   │
│  │  │ • backup    │   └───────────────────────┘      │   │
│  │  │ • logs      │                                  │   │
│  │  └─────────────┘                                  │   │
│  └────────────────────────────────────────────────────┘   │
│                           │                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Notification System (src/scheduler/notifications.py)│   │
│  │  • Console (always-on) • Email (SMTP)              │   │
│  │  • Slack (webhook)     • File log                   │   │
│  └────────────────────────────────────────────────────┘   │
│                           │                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Execution Platforms                                │   │
│  │  • Windows Task Scheduler  • cron (Unix)            │   │
│  │  • GitHub Actions (daily) • Manual (CLI)            │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                        │
│  ┌────────────────────┐  ┌────────────────────────┐      │
│  │  AlertEngine        │  │  DriftDetector         │      │
│  │  • 15 built-in rules│  │  • PSI-based drift     │      │
│  │  • Threshold checks │  │  • Feature drift       │      │
│  │  • Cooldown support │  │  • Prediction drift    │      │
│  │  • Slack/email      │  │  • History persistence │      │
│  └────────────────────┘  └────────────────────────┘      │
│  ┌────────────────────────────────────────────────────┐   │
│  │  PipelineHealth                                    │   │
│  │  • Composite health score (0.0-1.0)                │   │
│  │  • Task success rate tracking                      │   │
│  │  • System resource monitoring                      │   │
│  │  • Data quality metrics                            │   │
│  │  • Recent alerts count                             │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  MonitoringStore (SQLite)                          │   │
│  │  • ETL metrics  • System metrics                  │   │
│  │  • Data quality  • Cache metrics                  │   │
│  │  • Retention policy (90 days)                      │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline Flow

### 2.1 Full Daily Pipeline

```
Daily at 06:00 UTC
       │
       ▼
┌──────────────────┐
│ 1. DATA PIPELINE  │  ← Fetches from all sources
│    (20 tasks)     │    football-data.co.uk, openfootball,
│                   │    Transfermarkt, Understat
└────────┬─────────┘
         │ (depends on)
         ▼
┌──────────────────────┐
│ 2. FEATURE COMPUTE   │  ← 140+ features computed
│    (15 min)          │    rolling stats, Elo, xG, H2H, form
└────────┬────────────┘
         │ (depends on)
         ▼
┌──────────────────────┐
│ 3. MODEL RETRAINING  │  ← Checks threshold, retrains if needed
│    (30 min)          │    XGBoost, RF, LR → validate → save best
└────────┬────────────┘
         │ (depends on)
         ▼
┌──────────────────────┐
│ 4. PREDICTIONS       │  ← Loads fixtures, generates predictions
│    (10 min)          │    Saves CSV + JSON with timestamps
└────────┬────────────┘
         │
         ▼
┌──────────────────────┐
│ 5. MONITORING        │  ← Health check, drift detection
│    (5 min)           │    Alerts if pipeline degraded
└──────────────────────┘
```

### 2.2 Dependency Graph

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
  update_database ──── backup_database ──── generate_logs

  daily_data_pipeline (parallel to above, runs independently)
       │
       ▼
  daily_feature_computation
       │
       ▼
  daily_model_retraining
       │
       ▼
  daily_predictions
```

### 2.3 Hourly Auto-Commit

```
Every hour:
    git add -A
    git commit -m "Auto-commit: {timestamp} [{+added ~modified -deleted}]"
    git push
```

---

## 3. Scheduling Details

### 3.1 Schedule Table

| Task | Frequency | Time | Platform | Max Duration | Retries |
|------|-----------|------|----------|-------------|---------|
| **Full pipeline** | Daily | 06:00 UTC | GitHub Actions | 90 min | — |
| **Data fetch** | Daily | 06:00 UTC | GitHub Actions | 20 min | 2 |
| **Feature compute** | Daily | After data | GitHub Actions | 15 min | 1 |
| **Model retrain** | Daily | After features | GitHub Actions | 30 min | 1 |
| **Predictions** | Daily | After retrain | GitHub Actions | 10 min | 1 |
| **Auto-commit** | Hourly | :00 | Win Scheduler | 1 min | — |
| **DB backup** | Weekly | Sun 08:00 | Win/cron | 5 min | 2 |
| **Log rotation** | Weekly | Sun 08:05 | Win/cron | 1 min | — |

### 3.2 Windows Task Scheduler

Install via CLI:
```bash
python -m src.scheduler.cli install --platform windows
```

Or generate a .bat script:
```bash
python -m src.scheduler.cli generate --platform windows --output setup_automation.bat
```

### 3.3 Cron (Unix)

Install via CLI:
```bash
python -m src.scheduler.cli install --platform cron
```

Or generate crontab:
```bash
python -m src.scheduler.cli generate --platform cron --output crontab.txt
```

### 3.4 GitHub Actions

The workflow `.github/workflows/automation.yml` runs daily at 06:00 UTC with:
- 5 sequential jobs (Data → Features → Retrain → Predict → Monitor)
- Artifact passing between jobs
- Conditional notification on failure
- Manual dispatch with override options

---

## 4. Task Reference

### 4.1 Core Tasks (6)

| Task | Description | Timeout | Retries | Depends On |
|------|-------------|---------|---------|------------|
| `download_fixtures` | Fetch new match data from sources | 300s | 2 | — |
| `validate_data` | Validate integrity/schema | 120s | 1 | download_fixtures |
| `clean_data` | Deduplicate, archive, normalize | 180s | 1 | validate_data |
| `update_database` | Ingest into PostgreSQL + retrain ensemble | 600s | 2 | clean_data |
| `backup_database` | pg_dump or SQLite backup | 300s | 1 | — |
| `generate_logs` | Rotate logs, archive reports | 60s | 1 | backup_database |

### 4.2 Daily Pipeline Tasks (4)

| Task | Description | Timeout | Retries | Depends On |
|------|-------------|---------|---------|------------|
| `daily_data_pipeline` | Fetch all sources, clean & merge | 600s | 2 | — |
| `daily_feature_computation` | Compute 140+ features via Feature Store | 600s | 1 | daily_data_pipeline |
| `daily_model_retraining` | Retrain models, validate, save best | 1200s | 1 | daily_feature_computation |
| `daily_predictions` | Load fixtures, predict, save with timestamp | 300s | 1 | daily_model_retraining |

### 4.3 Custom Scripts

| Script | What It Does | Called By |
|--------|-------------|-----------|
| `scripts/daily_data_pipeline.py` | Fetches worldcup + leagues + player + xG data | scheduler, GitHub Actions |
| `scripts/daily_feature_computation.py` | Builds 140+ features, computes Elo | scheduler, GitHub Actions |
| `scripts/daily_model_retraining.py` | Trains XGBoost/RF/LR, picks best, saves | scheduler, GitHub Actions |
| `scripts/daily_predictions.py` | Loads best model + fixtures, generates predictions | scheduler, GitHub Actions |

---

## 5. Notification System

### 5.1 Channels

| Channel | Default | Configuration |
|---------|---------|---------------|
| **Console** | ✅ Always | Log level: `logging.INFO` |
| **Email** | ❌ Off | `NOTIFY_EMAIL_*` env vars |
| **Slack** | ❌ Off | `NOTIFY_SLACK_WEBHOOK` env var |
| **File** | ✅ On | `logs/notifications.log` |

### 5.2 Configuration

Environment variables:

```bash
# Email
export NOTIFY_EMAIL_ENABLED=true
export NOTIFY_SMTP_HOST="smtp.gmail.com"
export NOTIFY_SMTP_PORT=587
export NOTIFY_SMTP_USER="your@email.com"
export NOTIFY_SMTP_PASSWORD="app-password"
export NOTIFY_EMAIL_TO="admin@example.com"

# Slack
export NOTIFY_SLACK_ENABLED=true
export NOTIFY_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export NOTIFY_SLACK_CHANNEL="#alerts"

# Minimum level (info, warning, error)
export NOTIFY_MIN_LEVEL="warning"
```

### 5.3 Programmatic Usage

```python
from src.scheduler.notifications import Notifier

notifier = Notifier()

# Send simple notification
notifier.send("Pipeline Complete", "All tasks succeeded", level="info")

# Send pipeline report
report = engine.run_all()
notifier.send_pipeline_report(report)

# Send failure
notifier.send_failure("daily_data_pipeline", "Connection timeout")
```

---

## 6. Monitoring Setup

### 6.1 Monitor Configuration

```python
from src.monitoring import Monitor

monitor = Monitor(
    db_path="data/monitoring/monitor.db",
    output_dir="reports/monitoring",
    data_dir="data",
    retention_days=90,
)
```

### 6.2 What Gets Monitored

| Category | Metrics | Collection |
|----------|---------|------------|
| **ETL** | Duration, rows imported, duplicates, missing values, validation failures | Per pipeline run |
| **System** | CPU %, Memory %, Disk %, DB size | Every 60s (background) |
| **Data Quality** | Null %, Duplicate %, Validation errors | Per dataset |
| **Cache** | Hit rate, entries, size | Per cache operation |

### 6.3 Reports Generated

| Report | Format | Location | Frequency |
|--------|--------|----------|-----------|
| HTML Dashboard | `.html` | `reports/monitoring/` | Daily |
| JSON Metrics | `.json` | `reports/monitoring/` | Daily |
| CSV Export | `.csv` | `reports/monitoring/` | Daily |
| Daily Summary | `.txt` | `reports/monitoring/` | Daily |

### 6.4 Pipeline Health

```python
from src.monitoring.pipeline_health import PipelineHealth

health = PipelineHealth()
status = health.generate_health_report()

print(f"Health Score: {status.health_score:.3f}")  # 0.0–1.0
print(f"Status: {status.status}")                   # healthy/degraded/critical
print(f"Task Success Rate: {status.task_success_rate:.2%}")
print(f"Alert Count: {status.alert_count}")
```

---

## 7. Alerting Rules

### 7.1 Built-in Rules (15)

| Rule | Metric | Condition | Threshold | Severity |
|------|--------|-----------|-----------|----------|
| `high_cpu` | system.cpu_percent | > | 90% | warning |
| `critical_cpu` | system.cpu_percent | > | 95% | critical |
| `high_memory` | system.memory_percent | > | 85% | warning |
| `critical_memory` | system.memory_percent | > | 95% | critical |
| `disk_space` | system.disk_usage_pct | > | 90% | warning |
| `critical_disk` | system.disk_usage_pct | > | 95% | critical |
| `etl_duration` | etl.duration_seconds | > | 300 | warning |
| `etl_failures` | etl.validation_failures | > | 10 | warning |
| `etl_low_rows` | etl.rows_imported | < | 100 | warning |
| `etl_high_duplicates` | etl.duplicate_pct | > | 15% | warning |
| `etl_high_missing` | etl.missing_values_pct | > | 10% | warning |
| `dq_null_rate` | data_quality.null_pct | > | 20% | warning |
| `dq_duplicate_rate` | data_quality.duplicate_pct | > | 10% | warning |
| `cache_hit_rate_low` | cache.hit_rate | < | 0.5 | warning |
| `feature_drift` | drift.feature_drift_score | > | 0.3 | warning |

### 7.2 Cooldown

Each rule has a 300-second (5 minute) cooldown to prevent alert storms.

### 7.3 Custom Rules

```python
from src.monitoring.alerting import AlertEngine, AlertRule

engine = AlertEngine(rules=[
    AlertRule(
        name="my_custom_alert",
        metric="etl.rows_imported",
        condition="<",
        threshold=50,
        severity="critical",
        description="Very few rows imported",
        cooldown_seconds=600,
    ),
])
```

---

## 8. Data Drift Detection

### 8.1 What Gets Checked

| Drift Type | Method | Threshold | Description |
|-----------|--------|-----------|-------------|
| **Feature drift** | PSI (Population Stability Index) | 0.2 | Per-feature distribution shift |
| **Prediction drift** | PSI | 0.15 | Prediction distribution shift |
| **Overall drift** | Average PSI | 0.1 | Composite drift score |

### 8.2 Usage

```python
from src.monitoring.drift import DriftDetector

detector = DriftDetector()

# Compare two DataFrames
result = detector.detect(reference_df=old_data, current_df=new_data)

if result.drift_detected:
    print(f"Drift in: {result.drifted_features}")
    print(f"Score: {result.overall_drift_score:.4f}")

# Compare two CSV files
result = detector.detect_from_csv("data/training_data.csv", "data/new_data.csv")
```

### 8.3 Drift History

Drift results are persisted to `data/monitoring/drift_history.json` (last 100 entries).

---

## 9. Pipeline Health Dashboard

### 9.1 Web Dashboard

The monitoring dashboard (`dashboard/app.py`) provides visualizations for:

| Page | What It Shows |
|------|---------------|
| **🤖 Model Performance** | Accuracy trends, confusion matrix, feature importance |
| **🔮 Prediction History** | Historical predictions, filters, ternary probability plot |
| **💰 Betting Results** | P&L, ROI, win rates, strategy comparison |
| **🎯 CLV Tracking** | CLV by model/time, distribution, aggregate stats |
| **🏦 Bankroll Monitoring** | Bankroll growth, drawdown, what-if simulator |

Launch with:
```bash
streamlit run dashboard/app.py
```

### 9.2 CLI Monitoring

```bash
# View last pipeline run status
python -m src.scheduler.cli status

# List all tasks
python -m src.scheduler.cli list

# Generate monitoring reports
python -m src.monitoring.cli report
python -m src.monitoring.cli summary
```

### 9.3 Health Score Interpretation

| Score Range | Status | Meaning |
|------------|--------|---------|
| 0.80 – 1.00 | **healthy** | All systems operational |
| 0.50 – 0.79 | **degraded** | Minor issues (e.g., recent warning) |
| 0.00 – 0.49 | **critical** | Significant failures need attention |

---

## 10. GitHub Actions Automation

### 10.1 Daily Workflow

`.github/workflows/automation.yml` runs a full pipeline daily:

```yaml
on:
  schedule:
    - cron: "0 6 * * *"   # 06:00 UTC daily
  workflow_dispatch:        # Manual trigger
```

### 10.2 Manual Trigger

Go to GitHub → Actions → "Daily Automation" → "Run workflow"

Options:
- `skip_download` — Skip data download (use cached data)
- `skip_features` — Skip feature computation
- `skip_training` — Skip model retraining
- `force_retrain` — Force retrain even if threshold not met

### 10.3 Artifacts Produced

| Job | Artifacts | Retention |
|-----|-----------|-----------|
| Data Pipeline | Raw data, processed data, reports | 7 days |
| Feature Pipeline | Feature matrices, computed Elo | 7 days |
| Model Retraining | Trained models, retraining reports | 30 days |
| Predictions | CSV + JSON predictions | 90 days |
| Monitoring | Pipeline health report | 90 days |

---

## 11. Troubleshooting Guide

### 11.1 Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| "No handler registered for task" | Task not registered in engine | Register in `src/scheduler/engine.py` `__init__` |
| Pipeline stuck on one task | Task timeout too short | Increase `timeout_seconds` in config |
| All tasks show "SKIPPED" | First dependency failed | Check dependency chain in `ScheduleConfig` |
| Email notifications not sending | SMTP config missing | Set `NOTIFY_*` env vars |
| Slack notifications failing | Webhook invalid | Check webhook URL in Slack admin |
| Drift detected unexpectedly | Data format changed | Check feature columns match reference |
| Health score dropping | Multiple recent failures | Check last 7 days of run reports |
| GitHub Actions failing | Missing secrets/API keys | Set `THE_ODDS_API_KEY` in repo secrets |

### 11.2 Log Locations

| Log | Path | Content |
|-----|------|---------|
| Scheduler runs | `logs/scheduler/` | Per-task logs |
| Run reports | `reports/scheduler/` | JSON reports per run |
| Notifications | `logs/notifications.log` | All notification events |
| Monitoring | `logs/monitoring/` | Monitoring system logs |
| Drift history | `data/monitoring/drift_history.json` | Drift detection results |
| Alert history | `data/monitoring/alert_history.json` | Triggered alerts |
| GitHub Actions | GitHub UI → Actions tab | Full pipeline logs |

### 11.3 Manual Commands

```bash
# Run specific tasks
python -m src.scheduler.cli run --tasks daily_data_pipeline,daily_predictions

# Run full pipeline
python -m src.scheduler.cli run

# Check status
python -m src.scheduler.cli status

# Generate health report
python -c "
from src.monitoring.pipeline_health import PipelineHealth
import json
health = PipelineHealth()
status = health.generate_health_report()
print(json.dumps(status.to_dict(), indent=2))
"

# Detect drift manually
python -c "
from src.monitoring.drift import DriftDetector
result = DriftDetector().detect_from_csv('data/training_data.csv', 'data/new_data.csv')
print(f'Drift detected: {result.drift_detected}')
"
```

### 11.4 Recovery Steps

1. **Check the latest run report**: `python -m src.scheduler.cli status`
2. **Check task list**: `python -m src.scheduler.cli list`
3. **Run the failed task manually**: `python -m src.scheduler.cli run --tasks <failed_task>`
4. **Force retrain if needed**: `python scripts/daily_model_retraining.py --force`
5. **Regenerate predictions**: `python scripts/daily_predictions.py`
6. **Check monitoring**: `python -m src.monitoring.cli summary`

---

## 12. Manual Override

### 12.1 Skipping Dependencies

When running manually, you can skip dependency checks:

```bash
# Run just predictions (even if training hasn't run)
python scripts/daily_predictions.py
python -m src.scheduler.cli run --tasks daily_predictions
```

### 12.2 Force Operations

```bash
# Force retrain regardless of data threshold
python scripts/daily_model_retraining.py --force

# Force data re-download
python scripts/daily_data_pipeline.py
```

### 12.3 Emergency Stop

```bash
# Stop a running pipeline (Ctrl+C)
# Or kill the process:
# Windows: taskkill /F /IM python.exe
# Unix:    kill <pid>
```

---

*For more information, see [src/scheduler/](../src/scheduler/), [src/monitoring/](../src/monitoring/), and [.github/workflows/automation.yml](../.github/workflows/automation.yml).*
