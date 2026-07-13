"""
Report generators — HTML dashboard, JSON, CSV, and daily summaries.

Produces four output formats:

* **HTML** — Standalone interactive dashboard using Plotly charts
* **JSON** — Structured machine-readable metrics export
* **CSV** — Time-series CSV export per metric table
* **Daily summary** — Human-readable text summary of today's activity
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.monitoring.store import MonitoringStore

logger = logging.getLogger(__name__)

# ── Brand colours ────────────────────────────────────────
COLORS = {
    "primary": "#1f77b4",
    "success": "#2ca02c",
    "warning": "#ff7f0e",
    "danger": "#d62728",
    "purple": "#9467bd",
    "teal": "#17becf",
    "grid": "#e8e8e8",
    "bg": "#fafafa",
}


class ReportGenerator:
    """Base report generator.

    Parameters
    ----------
    store : MonitoringStore
        The metrics store to read from.
    output_dir : str | Path
        Directory to write reports into.
    """

    def __init__(
        self,
        store: MonitoringStore,
        output_dir: str | Path = "reports/monitoring",
    ) -> None:
        self.store = store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class HTMLReport(ReportGenerator):
    """Generate an interactive HTML dashboard with Plotly charts.

    Produces a single self-contained ``dashboard.html`` with:
    - Summary KPI cards
    - ETL performance over time (duration, rows, speed)
    - System resource usage (CPU, memory, disk, DB size)
    - Data quality trends
    - Cache hit rate and entries
    - Validation failure counts

    The HTML file embeds Plotly as JSON so it can be opened
    in any browser without a server.
    """

    CHART_LAYOUT: dict[str, Any] = {
        "template": "plotly_white",
        "margin": dict(l=50, r=20, t=40, b=40),
        "font": dict(family="Segoe UI, Arial, sans-serif", size=12),
        "paper_bgcolor": COLORS["bg"],
        "plot_bgcolor": COLORS["bg"],
        "hovermode": "x unified",
        "xaxis": dict(gridcolor=COLORS["grid"], zeroline=False),
        "yaxis": dict(gridcolor=COLORS["grid"], zeroline=False),
    }

    def generate(self, days: int = 30) -> Path:
        """Generate the HTML dashboard and return the file path."""
        import plotly.graph_objects as go
        import plotly.io as pio

        etl_data = self.store.get_etl_history(days=days)
        sys_data = self.store.get_system_history(days=days)
        dq_data = self.store.get_data_quality_history(days=days)
        cache_data = self.store.get_cache_history(days=days)
        latest = self.store.get_latest()
        trends = self.store.get_trends(days=days)

        figures: list[str] = []
        kpi_values: list[tuple[str, str, str, str]] = []

        # ── KPI helpers ──────────────────────────────────
        def kpi(title: str, value: str, subtitle: str, color: str = COLORS["primary"]) -> None:
            kpi_values.append((title, value, subtitle, color))

        # Build KPIs from latest data
        if latest.etl:
            kpi("⏱ Pipeline Duration",
                f"{latest.etl.duration_seconds:.1f}s",
                f"Latest: {latest.etl.pipeline}", COLORS["primary"])
            kpi("📥 Rows Imported",
                f"{latest.etl.rows_imported:,}",
                f"{latest.etl.duplicate_pct:.1f}% dupes", COLORS["success"])
            kpi("📤 Processing Speed",
                f"{latest.etl.processing_speed_rows_s:.0f} rows/s",
                f"{latest.etl.download_speed_mbps:.1f} Mbps", COLORS["purple"])
        if latest.system:
            kpi("💻 CPU",
                f"{latest.system.cpu_percent:.1f}%",
                f"Mem: {latest.system.memory_percent:.1f}%", COLORS["warning"])
            kpi("💾 DB Size",
                f"{latest.system.db_size_mb:.1f} MB",
                f"Disk: {latest.system.disk_usage_pct:.1f}%", COLORS["danger"])
        if latest.cache:
            kpi("🎯 Cache Hit Rate",
                f"{latest.cache.hit_rate:.1%}",
                f"{latest.cache.entries:,} entries", COLORS["teal"])

        # ── Chart: ETL Duration ──────────────────────────
        if len(etl_data) >= 2:
            etl_sorted = sorted(etl_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in etl_sorted]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts,
                y=[r["duration_seconds"] for r in etl_sorted],
                mode="lines+markers",
                name="Duration (s)",
                line=dict(color=COLORS["primary"], width=2),
                marker=dict(size=5),
            ))
            fig.add_trace(go.Bar(
                x=ts,
                y=[r["rows_imported"] for r in etl_sorted],
                name="Rows Imported",
                yaxis="y2",
                marker_color=COLORS["success"],
                opacity=0.5,
            ))
            layout = {**self.CHART_LAYOUT}
            layout["yaxis2"] = dict(
                overlaying="y", side="right",
                gridcolor=COLORS["grid"], zeroline=False,
            )
            layout["title"] = "📊 ETL Pipeline Performance"
            fig.update_layout(**layout)
            figures.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

        # ── Chart: Download & Processing Speed ───────────
        if len(etl_data) >= 2:
            etl_sorted = sorted(etl_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in etl_sorted]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts,
                y=[r["download_speed_mbps"] for r in etl_sorted],
                mode="lines+markers",
                name="Download Speed (Mbps)",
                line=dict(color=COLORS["purple"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=ts,
                y=[r["processing_speed_rows_s"] for r in etl_sorted],
                mode="lines+markers",
                name="Processing Speed (rows/s)",
                line=dict(color=COLORS["teal"], width=2),
            ))
            layout = {**self.CHART_LAYOUT}
            layout["title"] = "📈 Throughput Speed"
            fig.update_layout(**layout)
            figures.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

        # ── Chart: System Resources ──────────────────────
        if len(sys_data) >= 2:
            sys_sorted = sorted(sys_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in sys_sorted]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts, y=[r["cpu_percent"] for r in sys_sorted],
                mode="lines+markers", name="CPU %",
                line=dict(color=COLORS["warning"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=[r["memory_percent"] for r in sys_sorted],
                mode="lines+markers", name="Memory %",
                line=dict(color=COLORS["danger"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=[r["disk_usage_pct"] for r in sys_sorted],
                mode="lines+markers", name="Disk %",
                line=dict(color=COLORS["primary"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=[r["db_size_mb"] for r in sys_sorted],
                mode="lines+markers", name="DB Size (MB)",
                line=dict(color=COLORS["purple"], width=2),
                yaxis="y2",
            ))
            layout = {**self.CHART_LAYOUT}
            layout["yaxis2"] = dict(
                overlaying="y", side="right",
                gridcolor=COLORS["grid"], zeroline=False,
            )
            layout["title"] = "🖥 System Resources"
            fig.update_layout(**layout)
            figures.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

        # ── Chart: Data Quality ──────────────────────────
        if len(dq_data) >= 2:
            dq_sorted = sorted(dq_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in dq_sorted]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts, y=[r["null_pct"] for r in dq_sorted],
                mode="lines+markers", name="Null %",
                line=dict(color=COLORS["danger"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=[r["duplicate_pct"] for r in dq_sorted],
                mode="lines+markers", name="Duplicate %",
                line=dict(color=COLORS["warning"], width=2),
            ))
            fig.add_trace(go.Bar(
                x=ts, y=[r["n_rows"] for r in dq_sorted],
                name="Row Count", yaxis="y2",
                marker_color=COLORS["primary"], opacity=0.4,
            ))
            layout = {**self.CHART_LAYOUT}
            layout["yaxis2"] = dict(
                overlaying="y", side="right",
                gridcolor=COLORS["grid"], zeroline=False,
            )
            layout["title"] = "✅ Data Quality"
            fig.update_layout(**layout)
            figures.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

        # ── Chart: Cache Performance ─────────────────────
        if len(cache_data) >= 2:
            cache_sorted = sorted(cache_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in cache_sorted]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts, y=[r["hit_rate"] for r in cache_sorted],
                mode="lines+markers", name="Hit Rate",
                line=dict(color=COLORS["success"], width=2),
                fill="tozeroy",
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=[r["entries"] for r in cache_sorted],
                mode="lines+markers", name="Entries",
                line=dict(color=COLORS["teal"], width=2),
                yaxis="y2",
            ))
            layout = {**self.CHART_LAYOUT}
            layout["yaxis2"] = dict(
                overlaying="y", side="right",
                gridcolor=COLORS["grid"], zeroline=False,
            )
            layout["title"] = "🎯 Cache Performance"
            fig.update_layout(**layout)
            figures.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

        # ── Trend indicators ─────────────────────────────
        trend_html = ""
        for trend in trends:
            icon = {"up": "📈", "down": "📉", "stable": "➡️"}.get(trend.direction, "➡️")
            color = {"up": "green", "down": "red", "stable": "gray"}.get(trend.direction, "gray")
            trend_html += (
                f'<div class="trend-card">'
                f'  <span class="trend-icon">{icon}</span>'
                f'  <div class="trend-info">'
                f'    <strong>{trend.metric_name}</strong>'
                f'    <span style="color:{color}">{trend.change_pct:+.1f}%</span>'
                f'    <small>{trend.values[-1][1]:.2f} → {trend.values[0][1]:.2f}</small>'
                f'  </div>'
                f'</div>'
            )

        # ── Build KPI cards ──────────────────────────────
        kpi_cards = ""
        for title, value, subtitle, color in kpi_values:
            kpi_cards += f"""
            <div class="kpi-card" style="border-left: 4px solid {color};">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-subtitle">{subtitle}</div>
            </div>
            """

        # ── Assemble HTML ────────────────────────────────
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ETL Monitoring Dashboard</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', -apple-system, Arial, sans-serif;
    background: #f5f7fa; color: #333; padding: 20px;
}}
.header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px; padding-bottom: 16px;
    border-bottom: 2px solid #e0e0e0;
}}
.header h1 {{ font-size: 28px; color: #1a1a2e; }}
.header .meta {{ color: #888; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.kpi-card {{
    background: white; border-radius: 10px; padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.15s, box-shadow 0.15s;
}}
.kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.1); }}
.kpi-title {{ font-size: 12px; text-transform: uppercase; color: #888; margin-bottom: 4px; }}
.kpi-value {{ font-size: 26px; font-weight: 700; color: #1a1a2e; }}
.kpi-subtitle {{ font-size: 12px; color: #aaa; margin-top: 2px; }}
.chart {{ background: white; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.trends {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin: 16px 0; }}
.trend-card {{
    display: flex; align-items: center; gap: 12px;
    background: white; border-radius: 8px; padding: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}}
.trend-icon {{ font-size: 24px; }}
.trend-info {{ display: flex; flex-direction: column; gap: 2px; }}
.trend-info strong {{ font-size: 13px; }}
.trend-info small {{ color: #999; font-size: 11px; }}
.footer {{ text-align: center; color: #aaa; font-size: 12px; margin-top: 32px; padding-top: 16px; border-top: 1px solid #e0e0e0; }}
</style>
</head>
<body>
<div class="header">
    <div>
        <h1>📊 ETL Monitoring Dashboard</h1>
        <div class="meta">Last updated: {now} &middot; {days}-day view</div>
    </div>
</div>

<div class="grid">
    {kpi_cards}
</div>

<h3>📈 Trends ({days}d)</h3>
<div class="trends">
    {trend_html}
</div>

<div class="chart">
    {"".join(figures) if figures else "<p style='color:#888;text-align:center;padding:40px;'>Not enough data to render charts — collect metrics first.</p>"}
</div>

<div class="footer">
    Generated by Football Prediction Monitoring &middot;
    <a href="metrics.json">metrics.json</a> &middot;
    <a href="metrics.csv">metrics.csv</a>
</div>
</body>
</html>"""

        out_path = self.output_dir / "dashboard.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("HTML dashboard written to %s", out_path)
        return out_path


class JSONReport(ReportGenerator):
    """Export all metrics as structured JSON."""

    def generate(self, days: int = 30) -> Path:
        """Generate a JSON metrics export.

        Structure::

            {
                "generated_at": "...",
                "days": 30,
                "etl_metrics": [...],
                "system_metrics": [...],
                "data_quality_metrics": [...],
                "cache_metrics": [...],
                "latest": {...},
                "trends": [...]
            }
        """
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "days": days,
            "etl_metrics": self.store.get_etl_history(days=days),
            "system_metrics": self.store.get_system_history(days=days),
            "data_quality_metrics": self.store.get_data_quality_history(days=days),
            "cache_metrics": self.store.get_cache_history(days=days),
            "latest": self.store.get_latest().to_dict(),
            "trends": [t.to_dict() for t in self.store.get_trends(days=days)],
            "storage_stats": self.store.get_stats(),
        }

        out_path = self.output_dir / "metrics.json"
        out_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("JSON metrics written to %s", out_path)
        return out_path


class CSVReport(ReportGenerator):
    """Export time-series metrics as CSV files.

    Generates one CSV per metric table:
    - ``etl_metrics.csv``
    - ``system_metrics.csv``
    - ``data_quality_metrics.csv``
    - ``cache_metrics.csv``
    """

    def generate(self, days: int = 30) -> list[Path]:
        """Generate CSV exports for all metric tables.

        Returns
        -------
        list[Path]
            Paths to the generated CSV files.
        """
        exports: list[tuple[str, str, list[dict[str, Any]]]] = [
            ("etl_metrics", "etl_metrics.csv", self.store.get_etl_history(days=days)),
            ("system_metrics", "system_metrics.csv", self.store.get_system_history(days=days)),
            ("data_quality_metrics", "data_quality_metrics.csv", self.store.get_data_quality_history(days=days)),
            ("cache_metrics", "cache_metrics.csv", self.store.get_cache_history(days=days)),
        ]

        paths: list[Path] = []
        for _name, filename, rows in exports:
            if not rows:
                continue
            out_path = self.output_dir / filename
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            paths.append(out_path)
            logger.info("CSV written: %s (%d rows)", out_path, len(rows))

        if not paths:
            # Write an empty placeholder
            empty_path = self.output_dir / "etl_metrics.csv"
            empty_path.write_text("no_data\n", encoding="utf-8")
            paths.append(empty_path)

        return paths


class DailySummaryReport(ReportGenerator):
    """Generate a human-readable daily summary.

    The summary is plain text designed for console output or
    email notifications.
    """

    def generate(
        self,
        label: str | None = None,
        latest: dict[str, Any] | None = None,
    ) -> str:
        """Generate a daily summary string.

        Parameters
        ----------
        label : str, optional
            Custom label (e.g. "Daily Summary — 2026-07-13").
        latest : dict, optional
            Pre-fetched latest metrics dict. If None, fetched from store.

        Returns
        -------
        str
            Formatted plain text summary.
        """
        snap = latest or self.store.get_latest().to_dict()
        date_label = label or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        lines: list[str] = [
            f"╔══════════════════════════════════════════╗",
            f"║  {date_label:<35s} ║",
            f"╚══════════════════════════════════════════╝",
            "",
        ]

        # ETL section
        etl = snap.get("etl")
        if etl and etl.get("pipeline"):
            lines.append("📥 ETL Pipeline")
            lines.append(f"  Pipeline:       {etl['pipeline']}")
            lines.append(f"  Duration:       {etl['duration_seconds']:.1f}s")
            lines.append(f"  Rows imported:  {etl['rows_imported']:,}")
            lines.append(f"  Rows skipped:   {etl['rows_skipped']:,}")
            lines.append(f"  Download speed: {etl['download_speed_mbps']:.1f} Mbps")
            lines.append(f"  Processing:     {etl['processing_speed_rows_s']:.0f} rows/s")
            lines.append(f"  Retries:        {etl['retry_count']}")
            lines.append(f"  Dup rate:       {etl['duplicate_pct']:.2f}%")
            lines.append(f"  Missing val %:  {etl['missing_values_pct']:.2f}%")
            lines.append(f"  Validation err: {etl['validation_failures']}")
            status = "✅ Success" if etl.get("success") else "❌ Failed"
            lines.append(f"  Status:         {status}")
            if etl.get("error_message"):
                lines.append(f"  Error:          {etl['error_message']}")
            lines.append("")

        # System section
        sys = snap.get("system")
        if sys:
            lines.append("💻 System")
            lines.append(f"  CPU:         {sys['cpu_percent']:.1f}%")
            lines.append(f"  Memory:      {sys['memory_percent']:.1f}% ({sys['memory_used_mb']:.0f} MB)")
            lines.append(f"  Disk:        {sys['disk_usage_pct']:.1f}%")
            lines.append(f"  DB Size:     {sys['db_size_mb']:.2f} MB")
            lines.append("")

        # Cache section
        cache = snap.get("cache")
        if cache:
            lines.append("🎯 Cache")
            lines.append(f"  Hit rate:   {cache['hit_rate']:.1%}")
            lines.append(f"  Hits:       {cache['hits']:,}")
            lines.append(f"  Misses:     {cache['misses']:,}")
            lines.append(f"  Entries:    {cache['entries']:,}")
            lines.append(f"  Size:       {cache.get('size_mb', 0):.2f} MB" if 'size_mb' in cache else "")
            lines.append("")

        # Data quality section
        dq = snap.get("data_quality")
        if dq:
            lines.append("✅ Data Quality")
            lines.append(f"  Source:         {dq['source']}")
            lines.append(f"  Rows:           {dq['n_rows']:,}")
            lines.append(f"  Columns:        {dq['n_columns']}")
            lines.append(f"  Null rate:      {dq['null_pct']:.2f}%")
            lines.append(f"  Dup rate:       {dq['duplicate_pct']:.2f}%")
            lines.append(f"  Validation err: {dq['validation_errors']}")
            lines.append("")

        # Trends section
        try:
            trends = self.store.get_trends(days=7)
            if trends:
                lines.append("📈 7-Day Trends")
                for t in trends:
                    icon = {"up": "📈", "down": "📉", "stable": "➡️"}.get(t.direction, "➡️")
                    lines.append(f"  {icon} {t.metric_name}: {t.change_pct:+.1f}%")
                lines.append("")
        except Exception:
            pass

        lines.append(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

        return "\n".join(lines) + "\n"
