"""
Pipeline Health — comprehensive health tracking for the automation pipeline.

Tracks:
- Task success/failure rates over time
- Execution times
- System resource usage
- Data quality trends
- Alert history
- Overall pipeline health score

Usage
-----
::

    from src.monitoring.pipeline_health import PipelineHealth

    health = PipelineHealth()
    score = health.compute_health_score()
    report = health.generate_health_report()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Overall pipeline health status at a point in time.

    Parameters
    ----------
    health_score : float
        Composite health score (0.0 = critical, 1.0 = perfect).
    status : str
        Overall status: "healthy", "degraded", "critical".
    task_success_rate : float
        Task success rate over the lookback period.
    avg_execution_time : float
        Average task execution time in seconds.
    system_health : str
        System resource health status.
    data_quality_health : str
        Data quality health status.
    alert_count : int
        Number of active/triggered alerts.
    last_successful_run : str or None
        Timestamp of last fully successful pipeline run.
    details : dict
        Detailed breakdown of each health dimension.
    timestamp : datetime
        When the health check was performed.
    """

    health_score: float = 1.0
    status: str = "healthy"
    task_success_rate: float = 1.0
    avg_execution_time: float = 0.0
    system_health: str = "healthy"
    data_quality_health: str = "healthy"
    alert_count: int = 0
    last_successful_run: str | None = None
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "health_score": round(self.health_score, 3),
            "status": self.status,
            "task_success_rate": round(self.task_success_rate, 4),
            "avg_execution_time": round(self.avg_execution_time, 2),
            "system_health": self.system_health,
            "data_quality_health": self.data_quality_health,
            "alert_count": self.alert_count,
            "last_successful_run": self.last_successful_run,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class PipelineHealth:
    """Tracks and reports on pipeline health.

    Parameters
    ----------
    scheduler_report_dir : str
        Directory containing scheduler run reports.
    monitoring_db_path : str
        Path to monitoring SQLite database.
    max_lookback_days : int
        Maximum days of history to consider.
    """

    def __init__(
        self,
        scheduler_report_dir: str = "reports/scheduler",
        monitoring_db_path: str = "data/monitoring/monitor.db",
        max_lookback_days: int = 30,
    ) -> None:
        self._report_dir = Path(scheduler_report_dir)
        self._db_path = Path(monitoring_db_path)
        self._max_lookback = max_lookback_days

    def compute_health_score(self) -> float:
        """Compute a composite health score (0.0–1.0).

        Factors:
        - Task success rate (40% weight)
        - Recent failure count (20% weight)
        - System resources (20% weight)
        - Data pipeline freshness (20% weight)
        """
        score = 1.0

        # Task success rate (40%)
        try:
            reports = self._load_recent_reports()
            if reports:
                total_succeeded = sum(r.get("succeeded", 0) for r in reports)
                total_failed = sum(r.get("failed", 0) for r in reports)
                total = total_succeeded + total_failed
                if total > 0:
                    success_rate = total_succeeded / total
                    score *= (0.4 * success_rate + 0.6)
            else:
                # No reports — assume healthy (new system)
                pass
        except Exception as exc:
            logger.warning("Failed to compute task success rate: %s", exc)

        # Recent failure penalty (20%)
        try:
            recent_reports = self._load_recent_reports(days=7)
            if recent_reports:
                recent_failures = sum(r.get("failed", 0) for r in recent_reports)
                failure_penalty = min(1.0, recent_failures * 0.05)
                score *= (0.2 * (1.0 - failure_penalty) + 0.8)
        except Exception:
            pass

        # System resources (20%)
        try:
            from src.monitoring.store import MonitoringStore
            store = MonitoringStore(db_path=str(self._db_path))
            latest_system = store.get_latest_system()
            if latest_system:
                sys_score = 1.0
                if latest_system.cpu_percent > 90:
                    sys_score -= 0.3
                elif latest_system.cpu_percent > 75:
                    sys_score -= 0.1
                if latest_system.memory_percent > 90:
                    sys_score -= 0.3
                elif latest_system.memory_percent > 75:
                    sys_score -= 0.1
                if latest_system.disk_usage_pct > 90:
                    sys_score -= 0.3
                elif latest_system.disk_usage_pct > 75:
                    sys_score -= 0.1
                score *= (0.2 * sys_score + 0.8)
        except Exception:
            pass

        # Freshness (20%) — check when pipeline last ran
        try:
            reports = self._load_recent_reports()
            if reports:
                latest = reports[0]
                last_run = latest.get("started_at") or latest.get("completed_at")
                if last_run:
                    from datetime import datetime
                    try:
                        last_dt = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
                        hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
                        if hours_since > 48:
                            score -= 0.2
                        elif hours_since > 24:
                            score -= 0.1
                    except Exception:
                        pass
        except Exception:
            pass

        return max(0.0, min(1.0, score))

    def get_status(self, health_score: float | None = None) -> str:
        """Get human-readable status based on health score."""
        if health_score is None:
            health_score = self.compute_health_score()

        if health_score >= 0.8:
            return "healthy"
        elif health_score >= 0.5:
            return "degraded"
        else:
            return "critical"

    def get_task_success_rate(self, days: int = 30) -> float:
        """Compute task success rate over the lookback period."""
        reports = self._load_recent_reports(days=days)
        if not reports:
            return 1.0

        total_succeeded = sum(r.get("succeeded", 0) for r in reports)
        total_failed = sum(r.get("failed", 0) for r in reports)
        total = total_succeeded + total_failed
        return total_succeeded / total if total > 0 else 1.0

    def get_last_successful_run(self) -> str | None:
        """Get the timestamp of the last fully successful pipeline run."""
        reports = self._load_recent_reports()
        for r in reports:
            if r.get("failed", 0) == 0:
                return r.get("completed_at") or r.get("started_at")
        return None

    def generate_health_report(self) -> HealthStatus:
        """Generate a comprehensive health status report."""
        score = self.compute_health_score()
        status = self.get_status(score)

        # Gather detailed metrics
        details = {}

        # Task success rate
        try:
            task_success_rate = self.get_task_success_rate()
            details["task_success_rate"] = round(task_success_rate, 4)
        except Exception as exc:
            details["task_success_rate_error"] = str(exc)
            task_success_rate = 1.0

        # Average execution time
        try:
            reports = self._load_recent_reports(days=7)
            if reports:
                times = [r.get("duration_seconds", 0) for r in reports if r.get("duration_seconds")]
                avg_time = sum(times) / len(times) if times else 0.0
            else:
                avg_time = 0.0
            details["avg_execution_time"] = round(avg_time, 2)
        except Exception:
            avg_time = 0.0

        # System health
        system_health = "unknown"
        try:
            from src.monitoring.store import MonitoringStore
            store = MonitoringStore(db_path=str(self._db_path))
            latest_system = store.get_latest_system()
            if latest_system:
                issues = []
                if latest_system.cpu_percent > 90:
                    issues.append(f"CPU at {latest_system.cpu_percent}%")
                if latest_system.memory_percent > 90:
                    issues.append(f"Memory at {latest_system.memory_percent}%")
                if latest_system.disk_usage_pct > 90:
                    issues.append(f"Disk at {latest_system.disk_usage_pct}%")
                system_health = ", ".join(issues) if issues else "healthy"
                details["system"] = {
                    "cpu": round(latest_system.cpu_percent, 1),
                    "memory": round(latest_system.memory_percent, 1),
                    "disk": round(latest_system.disk_usage_pct, 1),
                }
        except Exception as exc:
            details["system_error"] = str(exc)

        # Data quality health
        dq_health = "unknown"
        try:
            from src.monitoring.store import MonitoringStore
            store = MonitoringStore(db_path=str(self._db_path))
            latest_dq = store.get_latest_data_quality()
            if latest_dq:
                issues = []
                if latest_dq.null_pct > 20:
                    issues.append(f"Null rate at {latest_dq.null_pct}%")
                if latest_dq.duplicate_pct > 10:
                    issues.append(f"Duplicate rate at {latest_dq.duplicate_pct}%")
                if not latest_dq.validation_passed:
                    issues.append(f"{latest_dq.validation_errors} validation errors")
                dq_health = ", ".join(issues) if issues else "healthy"
                details["data_quality"] = {
                    "null_pct": round(latest_dq.null_pct, 2),
                    "duplicate_pct": round(latest_dq.duplicate_pct, 2),
                    "validation_errors": latest_dq.validation_errors,
                }
        except Exception as exc:
            details["data_quality_error"] = str(exc)

        # Alert count
        alert_count = 0
        try:
            from src.monitoring.alerting import AlertEngine
            engine = AlertEngine()
            history = engine.get_alert_history(days=7)
            alert_count = len(history)
            details["recent_alerts"] = alert_count
        except Exception:
            pass

        # Recent runs summary
        try:
            recent_runs = self._load_recent_reports(days=7)
            if recent_runs:
                details["recent_runs"] = []
                for r in recent_runs[:10]:
                    details["recent_runs"].append({
                        "started": r.get("started_at", ""),
                        "duration": r.get("duration_seconds", 0),
                        "succeeded": r.get("succeeded", 0),
                        "failed": r.get("failed", 0),
                    })
        except Exception:
            pass

        last_run = self.get_last_successful_run()

        return HealthStatus(
            health_score=score,
            status=status,
            task_success_rate=task_success_rate,
            avg_execution_time=avg_time,
            system_health=system_health,
            data_quality_health=dq_health,
            alert_count=alert_count,
            last_successful_run=last_run,
            details=details,
        )

    # ── Internal ───────────────────────────────────────

    def _load_recent_reports(self, days: int | None = None) -> list[dict]:
        """Load scheduler run reports from disk."""
        if not self._report_dir.exists():
            return []

        if days is None:
            days = self._max_lookback

        cutoff = datetime.now() - timedelta(days=days)
        reports = []

        for f in sorted(self._report_dir.glob("run_*.json"), reverse=True):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    break
                data = json.loads(f.read_text())
                reports.append(data)
            except Exception:
                continue

        return reports

    def get_history(self, days: int = 30) -> list[dict]:
        """Get health score history for charting.

        Parameters
        ----------
        days : int
            Lookback period.

        Returns
        -------
        list[dict]
            List of {timestamp, health_score, status} entries.
        """
        history = []
        for f in sorted(self._report_dir.glob("run_*.json"), reverse=True):
            if len(history) >= 100:
                break
            try:
                data = json.loads(f.read_text())
                health = {
                    "timestamp": data.get("started_at", data.get("completed_at", "")),
                    "succeeded": data.get("succeeded", 0),
                    "failed": data.get("failed", 0),
                    "duration": data.get("duration_seconds", 0),
                }
                history.append(health)
            except Exception:
                continue

        return history
