"""
Notification System — sends alerts on pipeline task failures and successes.

Supports multiple channels:
- Console (default, always-on)
- Email (via SMTP)
- Slack (via webhook)
- File log

Usage
-----
::

    from src.scheduler.notifications import Notifier

    notifier = Notifier()
    notifier.send(
        title="Pipeline Failure",
        message="daily_data_pipeline failed: connection timeout",
        level="error",
    )
"""

from __future__ import annotations

import json
import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for the notification system.

    Parameters
    ----------
    console_enabled : bool
        Log notifications to console (default True).
    email_enabled : bool
        Send email notifications (default False).
    slack_enabled : bool
        Send Slack notifications (default False).
    file_enabled : bool
        Write notifications to a log file (default True).
    smtp_host : str
        SMTP server hostname.
    smtp_port : int
        SMTP server port (default 587).
    smtp_user : str
        SMTP username.
    smtp_password : str
        SMTP password.
    email_from : str
        From address for email notifications.
    email_to : str
        Recipient address for email notifications.
    slack_webhook_url : str
        Slack incoming webhook URL.
    slack_channel : str
        Slack channel (default "#alerts").
    notification_file : str
        Path to notification log file (default "logs/notifications.log").
    min_level : str
        Minimum level to send: "info", "warning", "error" (default "warning").
    """

    console_enabled: bool = True
    email_enabled: bool = False
    slack_enabled: bool = False
    file_enabled: bool = True

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = "football-prediction@localhost"
    email_to: str = "admin@localhost"

    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"

    notification_file: str = "logs/notifications.log"
    min_level: str = "warning"

    LEVELS = {"info": 10, "warning": 20, "error": 30}

    @classmethod
    def from_env(cls) -> NotificationConfig:
        """Load configuration from environment variables."""
        import os

        cfg = cls()
        cfg.console_enabled = True
        cfg.email_enabled = bool(os.environ.get("NOTIFY_EMAIL_ENABLED", ""))
        cfg.slack_enabled = bool(os.environ.get("NOTIFY_SLACK_ENABLED", ""))
        cfg.smtp_host = os.environ.get("NOTIFY_SMTP_HOST", "")
        cfg.smtp_port = int(os.environ.get("NOTIFY_SMTP_PORT", "587"))
        cfg.smtp_user = os.environ.get("NOTIFY_SMTP_USER", "")
        cfg.smtp_password = os.environ.get("NOTIFY_SMTP_PASSWORD", "")
        cfg.email_from = os.environ.get("NOTIFY_EMAIL_FROM", "football-prediction@localhost")
        cfg.email_to = os.environ.get("NOTIFY_EMAIL_TO", "")
        cfg.slack_webhook_url = os.environ.get("NOTIFY_SLACK_WEBHOOK", "")
        cfg.slack_channel = os.environ.get("NOTIFY_SLACK_CHANNEL", "#alerts")
        cfg.min_level = os.environ.get("NOTIFY_MIN_LEVEL", "warning")
        return cfg


class Notifier:
    """Multi-channel notification dispatcher.

    Parameters
    ----------
    config : NotificationConfig, optional
        Notification configuration. Loads from env if not provided.
    """

    def __init__(self, config: NotificationConfig | None = None) -> None:
        self.config = config or NotificationConfig.from_env()

        # Ensure log directory exists
        if self.config.file_enabled:
            log_path = Path(self.config.notification_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        title: str,
        message: str,
        level: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """Send a notification through all enabled channels.

        Parameters
        ----------
        title : str
            Short notification title.
        message : str
            Detailed notification message.
        level : str
            Severity: "info", "warning", or "error".
        metadata : dict, optional
            Additional structured data to include.

        Returns
        -------
        dict[str, bool]
            Per-channel delivery status: {channel: success}.
        """
        results: dict[str, bool] = {}

        # Check minimum level
        min_val = self.config.LEVELS.get(self.config.min_level, 20)
        level_val = self.config.LEVELS.get(level, 10)
        if level_val < min_val:
            return results

        payload = {
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        # Console
        if self.config.console_enabled:
            results["console"] = self._send_console(payload)

        # Email
        if self.config.email_enabled and self.config.email_to:
            results["email"] = self._send_email(payload)

        # Slack
        if self.config.slack_enabled and self.config.slack_webhook_url:
            results["slack"] = self._send_slack(payload)

        # File
        if self.config.file_enabled:
            results["file"] = self._send_file(payload)

        return results

    def send_pipeline_report(self, report: dict[str, Any], level: str = "info") -> dict[str, bool]:
        """Send a pipeline run report as a notification.

        Parameters
        ----------
        report : dict
            Pipeline run report (from TaskEngine.run_all()).
        level : str
            Severity level.
        """
        name = report.get("pipeline_name", report.get("pipeline", "unknown"))
        succeeded = report.get("succeeded", 0)
        failed = report.get("failed", 0)
        duration = report.get("duration_seconds", 0)
        errors = report.get("errors", [])

        title = f"Pipeline: {name}"
        lines = [
            f"Pipeline: {name}",
            f"Duration: {duration:.1f}s",
            f"Succeeded: {succeeded}",
            f"Failed: {failed}",
        ]
        if errors:
            lines.append(f"Errors ({len(errors)}):")
            for e in errors[:5]:
                lines.append(f"  - {e[:200]}")

        return self.send(title=title, message="\n".join(lines), level=level, metadata=report)

    # ── Channel implementations ────────────────────────

    def _send_console(self, payload: dict) -> bool:
        """Send notification to console log."""
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🚨"}
        e = emoji.get(payload["level"], "ℹ️")
        msg = f"{e} [{payload['level'].upper()}] {payload['title']}: {payload['message']}"
        if payload["level"] == "error":
            logger.error(msg)
        elif payload["level"] == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)
        return True

    def _send_email(self, payload: dict) -> bool:
        """Send notification via SMTP email."""
        if not self.config.smtp_host or not self.config.email_to:
            return False

        try:
            body = (
                f"Title: {payload['title']}\n"
                f"Level: {payload['level'].upper()}\n"
                f"Time:  {payload['timestamp']}\n"
                f"\n{payload['message']}\n"
            )
            if payload.get("metadata"):
                body += f"\nMetadata:\n{json.dumps(payload['metadata'], indent=2, default=str)}"

            msg = MIMEText(body)
            msg["Subject"] = f"[Football Prediction] {payload['level'].upper()}: {payload['title']}"
            msg["From"] = self.config.email_from
            msg["To"] = self.config.email_to

            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls(context=context)
                if self.config.smtp_user:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.sendmail(self.config.email_from, [self.config.email_to], msg.as_string())

            logger.info("Email notification sent to %s", self.config.email_to)
            return True

        except Exception as exc:
            logger.error("Failed to send email notification: %s", exc)
            return False

    def _send_slack(self, payload: dict) -> bool:
        """Send notification to Slack via webhook."""
        try:
            import requests

            color_map = {"info": "#36a64f", "warning": "#ffcc00", "error": "#ff0000"}

            blocks = [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": payload["title"]},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": payload["message"][:3000]},
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"*Level:* {payload['level']} | *Time:* {payload['timestamp']}"},
                    ],
                },
            ]

            if payload.get("metadata"):
                fields = []
                for k, v in list(payload["metadata"].items())[:6]:
                    if isinstance(v, (int, float)):
                        fields.append({"type": "mrkdwn", "text": f"*{k}:* {v}"})
                if fields:
                    blocks.append({"type": "section", "fields": fields})

            data = {
                "channel": self.config.slack_channel,
                "attachments": [{
                    "color": color_map.get(payload["level"], "#36a64f"),
                    "blocks": blocks,
                }],
            }

            response = requests.post(
                self.config.slack_webhook_url,
                json=data,
                timeout=10,
            )
            success = response.status_code == 200
            if not success:
                logger.warning("Slack notification failed: HTTP %d", response.status_code)
            return success

        except ImportError:
            logger.debug("requests not installed — cannot send Slack notification")
            return False
        except Exception as exc:
            logger.error("Failed to send Slack notification: %s", exc)
            return False

    def _send_file(self, payload: dict) -> bool:
        """Write notification to a log file."""
        try:
            log_path = Path(self.config.notification_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
            return True
        except Exception as exc:
            logger.error("Failed to write notification log: %s", exc)
            return False

    def send_failure(self, task_name: str, error: str) -> dict[str, bool]:
        """Convenience method for sending a failure notification."""
        return self.send(
            title=f"Task Failed: {task_name}",
            message=error[:1000],
            level="error",
            metadata={"task": task_name, "error": error},
        )

    def send_success(self, task_name: str, message: str = "") -> dict[str, bool]:
        """Convenience method for sending a success notification."""
        return self.send(
            title=f"Task Succeeded: {task_name}",
            message=message or "Completed successfully",
            level="info",
            metadata={"task": task_name},
        )

    def send_summary(self, report: dict[str, Any]) -> dict[str, bool]:
        """Send a summary of a pipeline run.

        Determines appropriate level based on failures.
        """
        level = "error" if report.get("failed", 0) > 0 else "info"
        return self.send_pipeline_report(report, level=level)
