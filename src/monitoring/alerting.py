"""
Alerting System — threshold-based alerts for monitoring metrics.

Evaluates metric values against configurable thresholds and triggers
notifications when breached.

Usage
-----
::

    from src.monitoring.alerting import AlertEngine, AlertRule

    # Define rules
    rules = [
        AlertRule(
            name="high_cpu",
            metric="cpu_percent",
            condition=">",
            threshold=90.0,
            severity="warning",
            description="CPU usage above 90%",
        ),
    ]

    engine = AlertEngine(rules)
    triggered = engine.evaluate(metric_snapshot)
    engine.notify(triggered)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """A single alert rule with threshold and severity.

    Parameters
    ----------
    name : str
        Unique rule name.
    metric : str
        Metric key to evaluate (dot-separated for nested, e.g. "system.cpu_percent").
    condition : str
        Comparison operator: ">", "<", ">=", "<=", "==", "!=".
    threshold : float
        Threshold value.
    severity : str
        Alert severity: "info", "warning", "critical".
    description : str
        Human-readable description.
    enabled : bool
        Whether this rule is active (default True).
    cooldown_seconds : int
        Minimum seconds between consecutive alerts for this rule (default 300).
    """

    name: str = ""
    metric: str = ""
    condition: str = ">"
    threshold: float = 0.0
    severity: str = "warning"
    description: str = ""
    enabled: bool = True
    cooldown_seconds: int = 300


@dataclass
class AlertEvent:
    """A single triggered alert event.

    Parameters
    ----------
    rule_name : str
        Name of the rule that triggered.
    metric : str
        Metric that was evaluated.
    actual_value : float
        The actual metric value.
    threshold : float
        The threshold that was crossed.
    severity : str
        Alert severity.
    message : str
        Formatted alert message.
    timestamp : datetime
        When the alert was triggered.
    """

    rule_name: str = ""
    metric: str = ""
    actual_value: float = 0.0
    threshold: float = 0.0
    severity: str = "warning"
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "metric": self.metric,
            "actual_value": round(self.actual_value, 4),
            "threshold": self.threshold,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


DEFAULT_RULES = [
    # System
    AlertRule("high_cpu", "system.cpu_percent", ">", 90.0, "warning",
              "CPU usage above 90%"),
    AlertRule("critical_cpu", "system.cpu_percent", ">", 95.0, "critical",
              "CPU usage above 95% — immediate attention"),
    AlertRule("high_memory", "system.memory_percent", ">", 85.0, "warning",
              "Memory usage above 85%"),
    AlertRule("critical_memory", "system.memory_percent", ">", 95.0, "critical",
              "Memory usage above 95%"),
    AlertRule("disk_space", "system.disk_usage_pct", ">", 90.0, "warning",
              "Disk usage above 90%"),
    AlertRule("critical_disk", "system.disk_usage_pct", ">", 95.0, "critical",
              "Disk usage above 95% — running out of space"),

    # ETL
    AlertRule("etl_duration", "etl.duration_seconds", ">", 300.0, "warning",
              "ETL pipeline took more than 5 minutes"),
    AlertRule("etl_failures", "etl.validation_failures", ">", 10, "warning",
              "More than 10 validation failures in ETL run"),
    AlertRule("etl_low_rows", "etl.rows_imported", "<", 100, "warning",
              "ETL imported fewer than 100 rows (possible source issue)"),
    AlertRule("etl_high_duplicates", "etl.duplicate_pct", ">", 15.0, "warning",
              "Duplicate percentage above 15% in ETL data"),
    AlertRule("etl_high_missing", "etl.missing_values_pct", ">", 10.0, "warning",
              "Missing values percentage above 10%"),

    # Data quality
    AlertRule("dq_null_rate", "data_quality.null_pct", ">", 20.0, "warning",
              "Null percentage above 20% in quality check"),
    AlertRule("dq_duplicate_rate", "data_quality.duplicate_pct", ">", 10.0, "warning",
              "Duplicate percentage above 10% in quality check"),

    # Cache
    AlertRule("cache_hit_rate_low", "cache.hit_rate", "<", 0.5, "warning",
              "Cache hit rate below 50%"),

    # Drift
    AlertRule("feature_drift", "drift.feature_drift_score", ">", 0.3, "warning",
              "Feature drift score above 0.3 (significant drift detected)"),
    AlertRule("prediction_drift", "drift.prediction_drift_score", ">", 0.2, "warning",
              "Prediction distribution drift above 0.2"),

    # ── Model Performance ──────────────────────────────────
    AlertRule("accuracy_drop", "performance.accuracy", "<", 0.50, "warning",
              "Model accuracy below 50% — significant underperformance"),
    AlertRule("accuracy_critical", "performance.accuracy", "<", 0.40, "critical",
              "Model accuracy below 40% — immediate investigation required"),
    AlertRule("brier_spike", "performance.brier_score", ">", 0.30, "warning",
              "Brier score above 0.30 — calibration degrading"),
    AlertRule("brier_critical", "performance.brier_score", ">", 0.40, "critical",
              "Brier score above 0.40 — severe calibration failure"),
    AlertRule("log_loss_spike", "performance.log_loss", ">", 1.2, "warning",
              "Log loss above 1.2 — prediction confidence is misaligned"),
    AlertRule("log_loss_critical", "performance.log_loss", ">", 1.5, "critical",
              "Log loss above 1.5 — model is no better than random"),
    AlertRule("roi_negative", "performance.roi_pct", "<", -5.0, "warning",
              "ROI below -5% — betting strategy losing money"),
    AlertRule("roi_critical", "performance.roi_pct", "<", -15.0, "critical",
              "ROI below -15% — severe losses, halt betting"),
    AlertRule("clv_negative_persistent", "performance.avg_clv", "<", -0.01, "warning",
              "Average CLV negative — consistently getting worse odds than closing line"),
    AlertRule("clv_critical", "performance.avg_clv", "<", -0.03, "critical",
              "Average CLV below -3% — severe adverse line movement"),
    AlertRule("win_rate_low", "performance.win_rate_pct", "<", 40.0, "warning",
              "Win rate below 40% — model predictions are losing more than winning"),
    AlertRule("win_rate_critical", "performance.win_rate_pct", "<", 30.0, "critical",
              "Win rate below 30% — model is significantly worse than random"),
    AlertRule("sharpe_below_one", "performance.sharpe_ratio", "<", 1.0, "warning",
              "Sharpe ratio below 1.0 — risk-adjusted returns suboptimal"),
    AlertRule("sharpe_negative", "performance.sharpe_ratio", "<", 0.0, "critical",
              "Sharpe ratio negative — strategy destroying value"),
    AlertRule("drawdown_excessive", "performance.max_drawdown_pct", ">", 20.0, "warning",
              "Max drawdown above 20% — excessive risk exposure"),
    AlertRule("drawdown_critical", "performance.max_drawdown_pct", ">", 35.0, "critical",
              "Max drawdown above 35% — near-ruin level drawdown"),
    AlertRule("bankroll_decline", "performance.bankroll_change_pct", "<", -10.0, "warning",
              "Bankroll declined more than 10% from peak"),
    AlertRule("bankroll_critical", "performance.bankroll_change_pct", "<", -25.0, "critical",
              "Bankroll declined more than 25% from peak — risk of ruin"),
    AlertRule("bet_frequency_drop", "performance.bets_per_day", "<", 1.0, "info",
              "Less than 1 bet per day on average — possible data pipeline issue"),
    AlertRule("bet_frequency_surge", "performance.bets_per_day", ">", 50.0, "warning",
              "More than 50 bets per day — possible over-betting or filter failure"),
    AlertRule("avg_ev_negative", "performance.avg_ev", "<", -0.02, "warning",
              "Average expected value negative — odds consistently unfavourable"),
    AlertRule("confidence_drop", "performance.avg_confidence", "<", 30.0, "warning",
              "Average model confidence below 30 — predictions too uncertain"),
]


class AlertEngine:
    """Evaluates metrics against alert rules and tracks triggered alerts.

    Parameters
    ----------
    rules : list[AlertRule]
        Alert rules to evaluate. Uses defaults if not provided.
    alert_history_path : str, optional
        Path to persist alert history.
    """

    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        alert_history_path: str = "data/monitoring/alert_history.json",
    ) -> None:
        self.rules = rules or DEFAULT_RULES
        self._history_path = Path(alert_history_path)
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_alert: dict[str, float] = {}  # rule_name -> timestamp
        self._load_history()

    def _load_history(self) -> None:
        """Load suppression timestamps from disk."""
        if self._history_path.exists():
            try:
                data = json.loads(self._history_path.read_text())
                self._last_alert = data.get("last_alert", {})
            except Exception:
                self._last_alert = {}

    def _save_history(self) -> None:
        """Persist suppression timestamps."""
        try:
            self._history_path.write_text(json.dumps({
                "last_alert": self._last_alert,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }))
        except Exception as exc:
            logger.warning("Failed to save alert history: %s", exc)

    def _get_nested_value(self, data: dict, path: str) -> Any:
        """Get a value from a dict using dot-separated path.

        First checks if the full path exists as a flat key (e.g. ``{"system.cpu": 95}``),
        then falls back to nested access (e.g. ``{"system": {"cpu": 95}}``).
        """
        # Try flat key first
        if path in data:
            return data[path]
        # Fallback: nested access
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, None)
            else:
                return None
        return current

    def evaluate(self, snapshot: dict[str, Any]) -> list[AlertEvent]:
        """Evaluate all rules against a metric snapshot.

        Parameters
        ----------
        snapshot : dict
            Metric snapshot with values keyed by dot-path (e.g. {"system.cpu_percent": 85.0}).

        Returns
        -------
        list[AlertEvent]
            List of triggered alerts (cooldown-aware).
        """
        triggered: list[AlertEvent] = []
        now = time.time()

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Check cooldown
            last_time = self._last_alert.get(rule.name, 0)
            if now - last_time < rule.cooldown_seconds:
                continue

            # Get metric value
            value = self._get_nested_value(snapshot, rule.metric)
            if value is None:
                continue

            # Convert to float
            try:
                value_f = float(value)
            except (ValueError, TypeError):
                continue

            # Evaluate condition
            breached = False
            if rule.condition == ">":
                breached = value_f > rule.threshold
            elif rule.condition == "<":
                breached = value_f < rule.threshold
            elif rule.condition == ">=":
                breached = value_f >= rule.threshold
            elif rule.condition == "<=":
                breached = value_f <= rule.threshold
            elif rule.condition == "==":
                breached = value_f == rule.threshold
            elif rule.condition == "!=":
                breached = value_f != rule.threshold

            if breached:
                event = AlertEvent(
                    rule_name=rule.name,
                    metric=rule.metric,
                    actual_value=value_f,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=f"ALERT [{rule.severity.upper()}] {rule.description}: "
                            f"{rule.metric}={value_f:.2f} (threshold={rule.threshold})",
                )
                triggered.append(event)
                self._last_alert[rule.name] = now

        if triggered:
            self._save_history()

        return triggered

    def notify(self, events: list[AlertEvent]) -> list[dict[str, bool]]:
        """Send notifications for triggered alerts.

        Uses the scheduler Notifier to dispatch alerts to configured channels.

        Parameters
        ----------
        events : list[AlertEvent]
            Alert events to notify about.

        Returns
        -------
        list[dict[str, bool]]
            Per-event delivery results.
        """
        if not events:
            return []

        results = []
        try:
            from src.scheduler.notifications import Notifier
            notifier = Notifier()
        except Exception as exc:
            logger.warning("Notifier not available: %s. Logging alerts to console.", exc)
            notifier = None

        for event in events:
            metadata = {
                "rule": event.rule_name,
                "metric": event.metric,
                "actual_value": event.actual_value,
                "threshold": event.threshold,
            }

            if notifier:
                result = notifier.send(
                    title=f"🚨 Alert: {event.rule_name}",
                    message=event.message,
                    level=event.severity,
                    metadata=metadata,
                )
                results.append(result)
            else:
                # Fallback: log to console
                logger.warning(event.message)
                results.append({"console": True})

        return results

    def evaluate_and_notify(self, snapshot: dict[str, Any]) -> list[AlertEvent]:
        """Convenience: evaluate rules and send notifications in one call.

        Parameters
        ----------
        snapshot : dict
            Metric snapshot.

        Returns
        -------
        list[AlertEvent]
            Triggered alerts.
        """
        events = self.evaluate(snapshot)
        if events:
            self.notify(events)
        return events

    def get_alert_history(self, days: int = 7) -> list[dict]:
        """Retrieve recent alert history from persisted file.

        Parameters
        ----------
        days : int
            Lookback period.

        Returns
        -------
        list[dict]
            Recent alert events.
        """
        if not self._history_path.exists():
            return []

        try:
            data = json.loads(self._history_path.read_text())
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            history = data.get("alerts", [])
            return [h for h in history if h.get("timestamp", "") > cutoff.isoformat()]
        except Exception:
            return []

    @property
    def active_rules(self) -> list[AlertRule]:
        """Get all enabled rules."""
        return [r for r in self.rules if r.enabled]
