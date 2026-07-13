#!/usr/bin/env python3
"""
CI/CD Notification Script — sends alerts on pipeline failures.

Supports multiple notification channels:
- Slack (webhook)
- Email (SMTP)
- GitHub Checks API

Configuration is via environment variables (set in CI secrets).

Usage
-----
::

    # Send a failure notification
    python scripts/notify.py \\
        --status failure \\
        --workflow "CI/CD" \\
        --run-id 123456789 \\
        --repo "owner/repo" \\
        --branch "main" \\
        --commit "${{ github.sha }}" \\
        --author "${{ github.actor }}" \\
        --url "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def send_slack(webhook_url: str, payload: dict[str, Any]) -> bool:
    """Send a message to Slack via webhook.

    Parameters
    ----------
    webhook_url : str
        Slack incoming webhook URL.
    payload : dict
        Slack message payload (blocks format).

    Returns
    -------
    bool
        True if successful.
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(webhook_url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.error("Slack notification failed: %s", exc)
        return False


def send_email(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    to: str,
    subject: str,
    body: str,
) -> bool:
    """Send an email notification via SMTP.

    Parameters
    ----------
    smtp_server : str
        SMTP server address.
    smtp_port : int
        SMTP port (587 for TLS).
    username : str
        SMTP username.
    password : str
        SMTP password.
    to : str
        Recipient email.
    subject : str
        Email subject.
    body : str
        Email body (plain text).

    Returns
    -------
    bool
        True if sent successfully.
    """
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = username
        msg["To"] = to

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.starttls(context=context)
            server.login(username, password)
            server.send_message(msg)
        return True
    except Exception as exc:
        logger.error("Email notification failed: %s", exc)
        return False


def build_slack_payload(
    status: str,
    workflow: str,
    repo: str,
    branch: str,
    commit: str,
    author: str,
    url: str,
    details: str = "",
) -> dict[str, Any]:
    """Build a Slack message block payload.

    Parameters
    ----------
    status : str
        Pipeline status: ``failure``, ``success``, ``cancelled``.
    workflow : str
        Workflow name.
    repo : str
        Repository name (owner/repo).
    branch : str
        Branch name.
    commit : str
        Commit SHA (short).
    author : str
        Commit author.
    url : str
        Link to the workflow run.
    details : str
        Additional details.

    Returns
    -------
    dict
        Slack message payload.
    """
    color = {
        "failure": "#e74c3c",
        "success": "#2ecc71",
        "cancelled": "#f39c12",
    }.get(status, "#95a5a6")

    emoji = {"failure": "🚨", "success": "✅", "cancelled": "⚠️"}.get(status, "ℹ️")

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} CI Pipeline {status.upper()}: {workflow}",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Repository:*\n{repo}"},
                {"type": "mrkdwn", "text": f"*Branch:*\n{branch}"},
                {"type": "mrkdwn", "text": f"*Commit:*\n`{commit[:7]}`"},
                {"type": "mrkdwn", "text": f"*Author:*\n{author}"},
            ],
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🔍 View Run"},
                    "url": url,
                    "style": "danger" if status == "failure" else "primary",
                },
            ],
        },
    ]

    if details:
        blocks.insert(
            2,
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": details},
            },
        )

    return {
        "text": f"{emoji} CI {status}: {workflow} ({repo}/{branch})",
        "blocks": blocks,
        "attachments": [{"color": color, "blocks": blocks}],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send CI/CD pipeline notifications",
    )
    parser.add_argument("--status", required=True, choices=["success", "failure", "cancelled"])
    parser.add_argument("--workflow", default="Unknown")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--author", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--details", default="")
    args = parser.parse_args()

    sent_any = False

    # ── Slack notification ──────────────────────────────
    slack_webhook = os.environ.get("SLACK_WEBHOOK", "")
    if slack_webhook:
        payload = build_slack_payload(
            status=args.status,
            workflow=args.workflow,
            repo=args.repo,
            branch=args.branch,
            commit=args.commit,
            author=args.author,
            url=args.url,
            details=args.details,
        )
        if send_slack(slack_webhook, payload):
            logger.info("✅ Slack notification sent.")
            sent_any = True
        else:
            logger.warning("❌ Slack notification failed.")
    else:
        logger.info("SLACK_WEBHOOK not set — skipping Slack notification.")

    # ── Email notification ──────────────────────────────
    email_to = os.environ.get("CI_EMAIL", "")
    smtp_server = os.environ.get("EMAIL_SERVER", "")
    smtp_port = int(os.environ.get("EMAIL_PORT", "587"))
    email_user = os.environ.get("EMAIL_USER", "")
    email_pass = os.environ.get("EMAIL_PASSWORD", "")

    if all([email_to, smtp_server, email_user, email_pass]):
        subject = f"🚨 CI {args.status.upper()}: {args.workflow} ({args.repo})"
        body = f"""
CI Pipeline Status: {args.status.upper()}
Workflow: {args.workflow}
Repository: {args.repo}
Branch: {args.branch}
Commit: {args.commit}
Author: {args.author}
URL: {args.url}

Details:
{args.details}
"""
        if send_email(smtp_server, smtp_port, email_user, email_pass, email_to, subject, body):
            logger.info("✅ Email notification sent.")
            sent_any = True
        else:
            logger.warning("❌ Email notification failed.")
    else:
        logger.info("Email config not set — skipping email notification.")

    if not sent_any:
        logger.warning("No notification channels configured. Set SLACK_WEBHOOK or CI_EMAIL env vars.")
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
