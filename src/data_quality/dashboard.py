"""
DataQualityDashboard — professional interactive HTML dashboard.

Generates a self-contained HTML page showing all 12 data quality
dimensions with KPIs, Plotly trend charts, severity badges, and
drill-down tables. Also produces JSON and CSV exports.

Designed to be run:
1. Standalone: ``python -m src.data_quality.cli generate``
2. After every ETL run: ``dq.generate()`` in the pipeline
3. From the Monitor: ``dq = DataQualityDashboard.from_monitor(monitor)``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_quality.coverage import CoverageAnalyzer
from src.data_quality.models import DataQualitySnapshot, DataQualitySummary
from src.monitoring.store import MonitoringStore
from src.validation.engine import ValidationEngine

logger = logging.getLogger(__name__)

# ── Brand palette (dark theme) ──────────────────────────
C = {
    "bg": "#0e1117",
    "card": "#1a1d27",
    "border": "#2a2d3a",
    "text": "#e0e0e0",
    "muted": "#8b8fa3",
    "accent": "#4fc3f7",
    "green": "#4caf50",
    "amber": "#ffc107",
    "red": "#ef4444",
    "blue": "#3b82f6",
    "purple": "#a855f7",
    "teal": "#14b8a6",
}


class DataQualityDashboard:
    """Generates a comprehensive data quality dashboard.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The match dataset to analyze. If provided, coverage and basic
        profiling are computed automatically.
    source_name : str
        Identifier for the data source (e.g. ``worldcup-2026``).
    output_dir : str | Path
        Directory to write generated reports.
    monitor_store : MonitoringStore, optional
        If provided, historical monitoring data is used for trend charts.
    validation_engine : ValidationEngine, optional
        If provided, runs validation checks and includes results.
    df_previous : pd.DataFrame, optional
        Previous dataset version for data drift detection.
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        source_name: str = "unknown",
        output_dir: str | Path = "reports/data_quality",
        monitor_store: MonitoringStore | None = None,
        validation_engine: ValidationEngine | None = None,
        df_previous: pd.DataFrame | None = None,
    ) -> None:
        self.df = df
        self.source_name = source_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor_store
        self.validator = validation_engine or ValidationEngine()
        self.df_previous = df_previous

        self.snapshot: DataQualitySnapshot | None = None

    # ── Build the snapshot ──────────────────────────────

    def build_snapshot(self) -> DataQualitySnapshot:
        """Assemble a complete DataQualitySnapshot from all sources."""
        snap = DataQualitySnapshot(source_name=self.source_name)

        if self.df is not None and not self.df.empty:
            df = self.df
            snap.n_rows = len(df)
            snap.n_columns = len(df.columns)

            # ── 1. Missing Values ──
            total_cells = df.size
            missing = int(df.isna().sum().sum())
            snap.missing_cells = missing
            snap.missing_pct = missing / total_cells * 100 if total_cells > 0 else 0.0
            snap.columns_with_missing = int((df.isna().sum() > 0).sum())

            # ── 2. Duplicate Matches ──
            key_cols = [c for c in ["date", "home_team", "away_team"]
                        if c in df.columns]
            if key_cols:
                dup_count = int(df.duplicated(subset=key_cols, keep="first").sum())
            else:
                dup_count = int(df.duplicated().sum())
            snap.duplicate_count = dup_count
            snap.duplicate_pct = dup_count / len(df) * 100 if len(df) > 0 else 0.0

            # ── 3-6. Coverage ──
            analyzer = CoverageAnalyzer()
            snap.coverage = analyzer.analyze(df)

            # ── 7-8. Data Drift & Schema ──
            if self.df_previous is not None and not self.df_previous.empty:
                from src.data_profiling import DataProfiler, DataDriftDetector

                profiler = DataProfiler()
                curr_report = profiler.profile(df, source_name=self.source_name)
                prev_report = profiler.profile(
                    self.df_previous, source_name="previous"
                )
                drift = DataDriftDetector().detect(curr_report, prev_report)
                snap.drift_metrics_count = len(drift.metrics)
                snap.drift_warnings = drift.n_warnings
                snap.drift_passed = drift.passed

            # Schema
            snap.schema_ok = len(snap.coverage.columns_missing) == 0

            # ── 9. Validation ──
            from src.validation.models import ValidationResult

            try:
                data = df.to_dict(orient="records")
                vresult = self.validator.run(data, source_name=self.source_name)
                snap.validation_passed = vresult.passed_checks
                snap.validation_total = vresult.total_checks
                snap.validation_errors = vresult.total_violations
            except Exception as exc:
                logger.warning("Validation failed in dashboard: %s", exc)
                snap.validation_errors = -1  # Signal error

        # ── 10-11. Import / Pipeline from monitoring ──
        if self.monitor is not None:
            etl_hist = self.monitor.get_etl_history(days=90)
            if etl_hist:
                snap.pipeline_runs = len(etl_hist)
                snap.pipeline_runtime_avg = float(
                    sum(r["duration_seconds"] for r in etl_hist) / len(etl_hist)
                )
                successes = sum(1 for r in etl_hist if r.get("success"))
                snap.import_success_rate = successes / len(etl_hist) if etl_hist else 1.0

            # ── 12. Database Growth ──
            sys_hist = self.monitor.get_system_history(days=90)
            if sys_hist:
                snap.db_size_mb = float(
                    max(r.get("db_size_mb", 0) for r in sys_hist)
                )

        self.snapshot = snap
        return snap

    # ── Generate all outputs ────────────────────────────

    def generate(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Generate the HTML dashboard, JSON export, and CSV export.

        Parameters
        ----------
        days : int
            Lookback period for trend charts (default 30).

        Returns
        -------
        dict[str, Any]
            Paths to generated files: ``html``, ``json``, ``csv``.
        """
        snap = self.build_snapshot() if self.snapshot is None else self.snapshot

        results: dict[str, Any] = {}

        try:
            path = self._generate_html(snap, days=days)
            results["html"] = str(path)
        except Exception as exc:
            logger.error("HTML dashboard failed: %s", exc, exc_info=True)
            results["html_error"] = str(exc)

        try:
            path = self._generate_json(snap)
            results["json"] = str(path)
        except Exception as exc:
            logger.error("JSON export failed: %s", exc)
            results["json_error"] = str(exc)

        try:
            path = self._generate_csv(snap)
            results["csv"] = str(path)
        except Exception as exc:
            logger.error("CSV export failed: %s", exc)
            results["csv_error"] = str(exc)

        try:
            path = self._generate_summary(snap)
            results["summary"] = str(path)
        except Exception as exc:
            logger.error("Summary failed: %s", exc)
            results["summary_error"] = str(exc)

        logger.info(
            "Data Quality Dashboard generated: HTML=%s JSON=%s CSV=%s",
            results.get("html"), results.get("json"), results.get("csv"),
        )
        return results

    # ── HTML Generator ──────────────────────────────────

    def _generate_html(self, snap: DataQualitySnapshot, days: int = 30) -> Path:
        """Generate the self-contained HTML dashboard with Plotly."""
        cove = snap.coverage
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── KPI Cards ─────────────────────────────────
        kpis = self._kpi_cards(snap)

        # ── Trend Charts (from monitor) ─────────────
        charts_html = self._build_charts(days=days)

        # ── Data Tables ──────────────────────────────
        tables_html = self._build_tables(snap)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>⚽ Data Quality Dashboard — {self._e(self.source_name)}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: {C["bg"]}; color: {C["text"]}; padding: 24px;
  }}
  .container {{ max-width: 1440px; margin: 0 auto; }}

  /* Header */
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #16213e 50%, #1a1d27 100%);
    border: 1px solid {C["border"]}; border-radius: 16px;
    padding: 2rem; margin-bottom: 2rem;
  }}
  .header h1 {{
    font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem 0;
    background: linear-gradient(90deg, {C["accent"]}, {C["green"]});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .header .meta {{ color: {C["muted"]}; font-size: 0.9rem; }}
  .header .meta span {{ margin-right: 1.5rem; }}

  /* KPI Grid */
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }}
  .kpi-card {{
    background: linear-gradient(135deg, {C["card"]}, #222639);
    border: 1px solid {C["border"]}; border-radius: 12px;
    padding: 14px 16px; position: relative; overflow: hidden;
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }}
  .kpi-card .accent {{ position:absolute; top:0; left:0; width:4px; height:100%; }}
  .kpi-card .title {{ font-size:11px; text-transform:uppercase; color:{C["muted"]};
                     letter-spacing:0.05em; margin-bottom:4px; }}
  .kpi-card .value {{ font-size:24px; font-weight:700; color:#fff; line-height:1.2; }}
  .kpi-card .sub {{ font-size:11px; color:{C["muted"]}; margin-top:2px; }}
  .kpi-card .icon {{ position:absolute; top:12px; right:12px; font-size:20px; opacity:0.3; }}

  /* Sections */
  .section {{
    background: linear-gradient(135deg, {C["card"]} 0%, #222639 100%);
    border: 1px solid {C["border"]}; border-radius: 12px;
    padding: 1.5rem; margin-bottom: 1.5rem;
  }}
  .section h2 {{ font-size:1.1rem; font-weight:600; color:#fff; margin-bottom:0.75rem;
                display:flex; align-items:center; gap:8px; }}
  .section .subtitle {{ color:{C["muted"]}; font-size:0.85rem; margin-bottom:1rem; }}

  /* Badges */
  .badge {{ display:inline-flex; align-items:center; gap:4px; padding:2px 10px;
            border-radius:20px; font-size:12px; font-weight:600; }}
  .badge-good {{ background:#1b5e20; color:#81c784; }}
  .badge-warn {{ background:#e65100; color:#ffcc80; }}
  .badge-bad  {{ background:#b71c1c; color:#ef9a9a; }}
  .badge-info {{ background:#0d47a1; color:#90caf9; }}

  /* Tables */
  table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  th {{ text-align:left; padding:8px 10px; background:#2a2d3a; color:{C["muted"]};
       font-weight:600; font-size:0.7rem; text-transform:uppercase;
       letter-spacing:0.05em; border-bottom:2px solid #333; }}
  td {{ padding:6px 10px; border-bottom:1px solid {C["border"]};
       font-family:'SFMono-Regular',Consolas,monospace; font-size:0.78rem; }}
  tr:hover td {{ background:#2a2d3a; }}

  /* Charts */
  .chart-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;
  }}
  @media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  .chart-box {{
    background: linear-gradient(135deg, {C["card"]} 0%, #222639 100%);
    border: 1px solid {C["border"]}; border-radius: 10px; padding: 12px;
  }}

  /* Footer */
  .footer {{ text-align:center; color:#555; font-size:0.8rem; margin-top:2rem;
             padding-top:1rem; border-top:1px solid {C["border"]}; }}

  /* Severity indicators */
  .ok {{ color:{C["green"]}; }} .warn {{ color:{C["amber"]}; }} .err {{ color:{C["red"]}; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>⚽ Data Quality Dashboard</h1>
    <div class="meta">
      <span>📁 Source: <strong>{self._e(self.source_name)}</strong></span>
      <span>📅 {now}</span>
      <span>📊 {snap.n_rows:,} rows × {snap.n_columns} cols</span>
      <span>⏱ {days}-day trends</span>
    </div>
  </div>

  <!-- KPI Grid -->
  <div class="kpi-grid">
    {kpis}
  </div>

  <!-- Trend Charts -->
  <div class="section">
    <h2>📈 Historical Trends</h2>
    <div class="subtitle">Time-series from monitoring store (last {days} days)</div>
    <div class="chart-row">
      {charts_html}
    </div>
  </div>

  <!-- Detail Tables -->
  <div class="section">
    <h2>📋 Detailed Metrics</h2>
    {tables_html}
  </div>

  <!-- Data Exports -->
  <div class="section">
    <h2>📦 Data Exports</h2>
    <div class="subtitle">Machine-readable exports for external analysis</div>
    <p style="color:{C["muted"]};font-size:0.85rem;">
      <a href="data_quality.json" style="color:{C["accent"]};">📄 JSON Export</a> &middot;
      <a href="data_quality.csv" style="color:{C["accent"]};">📊 CSV Export</a> &middot;
      <a href="daily_summary.txt" style="color:{C["accent"]};">📝 Daily Summary</a>
    </p>
  </div>

  <div class="footer">
    Generated by Football Data Quality Dashboard &middot; {now}
  </div>
</div>
</body>
</html>"""

        out = self.output_dir / "data_quality.html"
        out.write_text(html, encoding="utf-8")
        logger.info("HTML dashboard written to %s", out)
        return out

    # ── KPI Card Builder ──────────────────────────────

    def _kpi_cards(self, snap: DataQualitySnapshot) -> str:
        c = self._kpi_card
        cards = ""

        # 1. Missing Values
        mv_sev = "good" if snap.missing_pct < 1 else "warn" if snap.missing_pct < 5 else "bad"
        cards += c("Missing Values", f"{snap.missing_pct:.1f}%", f"{snap.missing_cells:,} cells / {snap.columns_with_missing} cols", "#f44336", mv_sev)

        # 2. Duplicate Matches
        dup_sev = "good" if snap.duplicate_count == 0 else "warn" if snap.duplicate_pct < 2 else "bad"
        cards += c("Duplicate Matches", f"{snap.duplicate_count:,}", f"{snap.duplicate_pct:.2f}% of rows", "#ff9800", dup_sev)

        # 3. Odds Coverage
        oc = snap.coverage.odds_coverage_pct
        oc_sev = "good" if oc >= 90 else "warn" if oc >= 50 else "bad"
        cards += c("Odds Coverage", f"{oc:.1f}%", f"of matches have odds data", "#3b82f6", oc_sev)

        # 4. xG Coverage
        xg = snap.coverage.xg_coverage_pct
        xg_sev = "good" if xg >= 80 else "warn" if xg >= 30 else "bad"
        cards += c("xG Coverage", f"{xg:.1f}%", f"of matches have xG data", "#a855f7", xg_sev)

        # 5. League Coverage
        lc = snap.coverage.league_coverage_pct
        lc_sev = "good" if lc >= 95 else "warn" if lc >= 80 else "bad"
        n_leagues = len(snap.coverage.league_coverage)
        cards += c("League Coverage", f"{lc:.1f}%", f"{n_leagues} leagues mapped", "#14b8a6", lc_sev)

        # 6. Season Coverage
        sc = snap.coverage.season_count
        cards += c("Season Coverage", f"{sc}", f"distinct seasons", "#8bc34a", "good")

        # 7. Data Drift
        if snap.drift_passed:
            ds = "✅ Passed"
            dsev = "good"
        else:
            ds = f"⚠ {snap.drift_warnings} warnings"
            dsev = "warn" if snap.drift_warnings < 5 else "bad"
        cards += c("Data Drift", ds, f"{snap.drift_metrics_count} metrics checked", "#ffc107", dsev)

        # 8. Schema Changes
        if snap.schema_ok and snap.coverage.n_columns_actual >= snap.coverage.n_columns_expected:
            ss = "✅ Stable"
            ssev = "good"
        else:
            ss = f"⚠ {len(snap.coverage.columns_missing)} missing / {len(snap.coverage.columns_added)} added"
            ssev = "warn"
        cards += c("Schema", ss, f"{snap.coverage.n_columns_actual}/{snap.coverage.n_columns_expected} cols", "#607d8b", ssev)

        # 9. Import Success Rate
        isr = snap.import_success_rate * 100
        isev = "good" if isr >= 98 else "warn" if isr >= 90 else "bad"
        cards += c("Import Success", f"{isr:.1f}%", f"{snap.pipeline_runs} runs tracked", "#4caf50", isev)

        # 10. Pipeline Runtime
        avg = snap.pipeline_runtime_avg
        psev = "good" if avg < 60 else "warn" if avg < 300 else "bad"
        cards += c("Avg Runtime", f"{avg:.0f}s", f"over {snap.pipeline_runs} runs", "#2196f3", psev)

        # 11. Validation Errors
        ve = snap.validation_errors
        vsev = "good" if ve == 0 else "warn" if ve < 20 else "bad"
        cards += c("Validation Errors", f"{ve:,}", f"{snap.validation_passed}/{snap.validation_total} checks passed", "#e91e63", vsev)

        # 12. Database Growth
        dbs = snap.db_size_mb
        dbsev = "good" if dbs < 100 else "warn" if dbs < 1000 else "bad"
        cards += c("DB Size", f"{dbs:.1f} MB", f"database file size", "#00bcd4", dbsev)

        return cards

    def _kpi_card(self, title: str, value: str, sub: str, color: str,
                  severity: str = "good") -> str:
        badge = f'<span class="badge badge-{severity}">{severity.upper()}</span>'
        return f"""
<div class="kpi-card">
  <div class="accent" style="background:{color};"></div>
  <div class="title">{title} {badge}</div>
  <div class="value">{value}</div>
  <div class="sub">{sub}</div>
</div>"""

    # ── Plotly Charts Builder ─────────────────────────

    def _build_charts(self, days: int = 30) -> str:
        """Render Plotly charts from monitoring store data."""
        if self.monitor is None:
            return '<p style="color:#888;padding:32px;text-align:center;">No monitoring store connected — historical trends unavailable.</p>'

        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except ImportError:
            return '<p style="color:#888;padding:32px;text-align:center;">Plotly not installed — install with: pip install plotly</p>'

        figures = []
        GRID = dict(gridcolor="#2a2d3a", zeroline=False)
        YAXIS2 = dict(overlaying="y", side="right", **GRID)
        layout = dict(
            template="plotly_dark",
            margin=dict(l=40, r=16, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#8b8fa3"),
            hovermode="x unified",
            xaxis=GRID,
            yaxis=GRID,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )

        # Chart 1: Pipeline Runtime + Rows Imported
        etl_data = self.monitor.get_etl_history(days=days)
        if len(etl_data) >= 2:
            etl_sorted = sorted(etl_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in etl_sorted]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts, y=[r["duration_seconds"] for r in etl_sorted],
                mode="lines+markers", name="Runtime (s)",
                line=dict(color=C["blue"], width=2), marker=dict(size=5),
            ))
            fig.add_trace(go.Bar(
                x=ts, y=[r["rows_imported"] for r in etl_sorted],
                name="Rows Imported", yaxis="y2",
                marker_color=C["green"], opacity=0.4,
            ))
            fig.update_layout(
                **layout,
                title="⏱ Pipeline Runtime & Volume",
                yaxis2=YAXIS2,
            )
            figures.append(f'<div class="chart-box">{pio.to_html(fig, include_plotlyjs=False, full_html=False, default_width="100%", default_height="300px")}</div>')

        # Chart 2: Data Quality Trends
        dq_data = self.monitor.get_data_quality_history(days=days)
        if len(dq_data) >= 2:
            dq_sorted = sorted(dq_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in dq_sorted]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts, y=[r.get("null_pct", 0) for r in dq_sorted],
                mode="lines+markers", name="Null %",
                line=dict(color=C["red"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=[r.get("duplicate_pct", 0) for r in dq_sorted],
                mode="lines+markers", name="Duplicate %",
                line=dict(color=C["amber"], width=2),
            ))
            fig.add_trace(go.Bar(
                x=ts, y=[r.get("n_rows", 0) for r in dq_sorted],
                name="Row Count", yaxis="y2",
                marker_color=C["blue"], opacity=0.3,
            ))
            fig.update_layout(
                **layout,
                title="✅ Data Quality Trends",
                yaxis2=YAXIS2,
            )
            figures.append(f'<div class="chart-box">{pio.to_html(fig, include_plotlyjs=False, full_html=False, default_width="100%", default_height="300px")}</div>')

        # Chart 3: DB Size Growth
        sys_data = self.monitor.get_system_history(days=days)
        if len(sys_data) >= 2:
            sys_sorted = sorted(sys_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in sys_sorted]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts, y=[r.get("db_size_mb", 0) for r in sys_sorted],
                mode="lines+markers", name="DB Size (MB)",
                line=dict(color=C["teal"], width=2), fill="tozeroy",
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=[r.get("cpu_percent", 0) for r in sys_sorted],
                mode="lines+markers", name="CPU %",
                line=dict(color=C["amber"], width=2), yaxis="y2",
            ))
            fig.update_layout(
                **layout,
                title="💾 Database Growth & System Load",
                yaxis2=YAXIS2,
            )
            figures.append(f'<div class="chart-box">{pio.to_html(fig, include_plotlyjs=False, full_html=False, default_width="100%", default_height="300px")}</div>')

        # Chart 4: Import Success Rate
        if len(etl_data) >= 2:
            etl_sorted = sorted(etl_data, key=lambda r: r["recorded_at"])
            ts = [r["recorded_at"] for r in etl_sorted]
            fig = go.Figure()
            successes = [1 if r.get("success") else 0 for r in etl_sorted]
            fig.add_trace(go.Bar(
                x=ts, y=successes,
                name="Success", marker_color=C["green"], opacity=0.7,
            ))
            fig.add_trace(go.Scatter(
                x=ts, y=[r.get("retry_count", 0) for r in etl_sorted],
                mode="lines+markers", name="Retries",
                line=dict(color=C["red"], width=2), yaxis="y2",
            ))
            # Add validation errors
            fig.add_trace(go.Scatter(
                x=ts, y=[r.get("validation_failures", 0) for r in etl_sorted],
                mode="lines+markers", name="Validation Failures",
                line=dict(color=C["amber"], width=2, dash="dot"), yaxis="y2",
            ))
            fig.update_layout(
                **layout,
                title="📊 Import Success & Retries",
                yaxis=dict(range=[-0.1, 1.3], gridcolor="#2a2d3a", zeroline=False,
                           tickvals=[0, 1], ticktext=["Fail", "Success"]),
                yaxis2=YAXIS2,
            )
            figures.append(f'<div class="chart-box">{pio.to_html(fig, include_plotlyjs=False, full_html=False, default_width="100%", default_height="300px")}</div>')

        if not figures:
            return '<p style="color:#888;padding:32px;text-align:center;">Not enough historical data — run the pipeline a few times to populate trends.</p>'

        # Pair charts into rows
        rows = []
        for i in range(0, len(figures), 2):
            pair = figures[i:i+2]
            rows.append(f'<div class="chart-row">{"".join(pair)}</div>')
        return "".join(rows)

    # ── Detail Tables ─────────────────────────────────

    def _build_tables(self, snap: DataQualitySnapshot) -> str:
        """Build HTML tables for detailed metric breakdowns."""
        cove = snap.coverage

        # Top leagues table
        league_rows = ""
        top_leagues = dict(sorted(cove.league_coverage.items(),
                                   key=lambda x: x[1], reverse=True)[:15])
        for league, count in top_leagues.items():
            pct = count / snap.n_rows * 100 if snap.n_rows > 0 else 0
            league_rows += f"<tr><td>{self._e(league)}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>"

        # Top seasons table
        season_rows = ""
        for season, count in sorted(cove.season_coverage.items()):
            pct = count / snap.n_rows * 100 if snap.n_rows > 0 else 0
            season_rows += f"<tr><td>{self._e(season)}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>"

        # Schema changes
        schema_rows = ""
        for col in cove.columns_missing[:10]:
            schema_rows += f'<tr><td style="color:{C["red"]}">✗ {self._e(col)}</td><td>Missing</td></tr>'
        if not cove.columns_missing:
            schema_rows += f'<tr><td style="color:{C["green"]}">✅ All {cove.n_columns_expected} expected columns present</td><td></td></tr>'

        return f"""
<table>
  <thead><tr><th style="width:50%;">League Distribution (Top 15)</th><th>Matches</th><th>%</th></tr></thead>
  <tbody>{league_rows}</tbody>
</table>
<br>
<table>
  <thead><tr><th style="width:50%;">Season Coverage</th><th>Matches</th><th>%</th></tr></thead>
  <tbody>{season_rows}</tbody>
</table>
<br>
<table>
  <thead><tr><th>Schema Validation</th><th></th></tr></thead>
  <tbody>{schema_rows}</tbody>
</table>"""

    # ── JSON Export ─────────────────────────────────────

    def _generate_json(self, snap: DataQualitySnapshot) -> Path:
        """Export the full snapshot as JSON."""
        data = snap.to_dict()
        # Add trend data from monitor
        if self.monitor is not None:
            data["etl_history"] = self.monitor.get_etl_history(days=90)
            data["dq_history"] = self.monitor.get_data_quality_history(days=90)
            data["system_history"] = self.monitor.get_system_history(days=90)

        out = self.output_dir / "data_quality.json"
        out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return out

    # ── CSV Export ──────────────────────────────────────

    def _generate_csv(self, snap: DataQualitySnapshot) -> Path:
        """Export key metrics as a flat CSV."""
        import csv
        out = self.output_dir / "data_quality.csv"

        rows = [
            {"metric": "n_rows", "value": snap.n_rows},
            {"metric": "n_columns", "value": snap.n_columns},
            {"metric": "missing_cells", "value": snap.missing_cells},
            {"metric": "missing_pct", "value": round(snap.missing_pct, 2)},
            {"metric": "columns_with_missing", "value": snap.columns_with_missing},
            {"metric": "duplicate_count", "value": snap.duplicate_count},
            {"metric": "duplicate_pct", "value": round(snap.duplicate_pct, 2)},
            {"metric": "odds_coverage_pct", "value": round(snap.coverage.odds_coverage_pct, 2)},
            {"metric": "xg_coverage_pct", "value": round(snap.coverage.xg_coverage_pct, 2)},
            {"metric": "league_coverage_pct", "value": round(snap.coverage.league_coverage_pct, 2)},
            {"metric": "season_count", "value": snap.coverage.season_count},
            {"metric": "drift_passed", "value": int(snap.drift_passed)},
            {"metric": "drift_warnings", "value": snap.drift_warnings},
            {"metric": "schema_ok", "value": int(snap.schema_ok)},
            {"metric": "import_success_rate", "value": round(snap.import_success_rate, 4)},
            {"metric": "pipeline_runtime_avg", "value": round(snap.pipeline_runtime_avg, 2)},
            {"metric": "pipeline_runs", "value": snap.pipeline_runs},
            {"metric": "validation_passed", "value": snap.validation_passed},
            {"metric": "validation_total", "value": snap.validation_total},
            {"metric": "validation_errors", "value": snap.validation_errors},
            {"metric": "db_size_mb", "value": round(snap.db_size_mb, 2)},
        ]

        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "value"])
            w.writeheader()
            w.writerows(rows)
        return out

    # ── Summary Text ────────────────────────────────────

    def _generate_summary(self, snap: DataQualitySnapshot) -> Path:
        """Generate a human-readable text summary."""
        summary = DataQualitySummary()
        summary.add(f"╔══════════════════════════════════════════════╗")
        summary.add(f"║  Data Quality Report — {snap.source_name:<20s} ║")
        summary.add(f"╚══════════════════════════════════════════════╝")
        summary.add("")
        summary.add(f"📊 Dataset: {snap.n_rows:,} rows × {snap.n_columns} columns")
        summary.add("")
        summary.add(f"🟢 Missing Values:     {snap.missing_pct:.2f}% ({snap.missing_cells:,} cells, {snap.columns_with_missing} cols)")
        summary.add(f"🟢 Duplicate Matches:  {snap.duplicate_count:,} ({snap.duplicate_pct:.2f}%)")
        summary.add(f"🔵 Odds Coverage:      {snap.coverage.odds_coverage_pct:.1f}%")
        summary.add(f"🟣 xG Coverage:        {snap.coverage.xg_coverage_pct:.1f}%")
        summary.add(f"🟢 League Coverage:    {snap.coverage.league_coverage_pct:.1f}% ({len(snap.coverage.league_coverage)} leagues)")
        summary.add(f"🟢 Season Coverage:    {snap.coverage.season_count} seasons")
        summary.add("")
        drift_icon = "✅" if snap.drift_passed else "⚠️"
        schema_icon = "✅" if snap.schema_ok else "⚠️"
        summary.add(f"{drift_icon} Data Drift:          {snap.drift_warnings} warnings ({snap.drift_metrics_count} metrics)")
        summary.add(f"{schema_icon} Schema:              {snap.coverage.n_columns_actual}/{snap.coverage.n_columns_expected} cols")
        if snap.coverage.columns_missing:
            summary.add(f"                        Missing: {', '.join(snap.coverage.columns_missing[:5])}")
        summary.add("")
        summary.add(f"🟢 Import Success:     {snap.import_success_rate:.1%} ({snap.pipeline_runs} runs)")
        summary.add(f"🔵 Avg Runtime:        {snap.pipeline_runtime_avg:.0f}s")
        summary.add(f"🔴 Validation Errors:  {snap.validation_errors:,}")
        summary.add(f"🟣 DB Size:            {snap.db_size_mb:.1f} MB")
        summary.add("")
        summary.add(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

        text = str(summary)
        out = self.output_dir / "daily_summary.txt"
        out.write_text(text, encoding="utf-8")
        return out

    # ── Factory: from Monitor ──────────────────────────

    @classmethod
    def from_monitor(
        cls,
        monitor: Any,
        df: pd.DataFrame | None = None,
        source_name: str = "monitoring",
        output_dir: str | Path = "reports/data_quality",
        df_previous: pd.DataFrame | None = None,
    ) -> DataQualityDashboard:
        """Create a dashboard from an existing Monitor instance.

        Parameters
        ----------
        monitor : Monitor
            An active ``src.monitoring.Monitor`` instance.
        df : pd.DataFrame, optional
            Current dataset for coverage analysis.
        source_name : str
            Source identifier.
        output_dir : str | Path
            Output directory.
        df_previous : pd.DataFrame, optional
            Previous dataset for drift detection.

        Returns
        -------
        DataQualityDashboard
        """
        return cls(
            df=df,
            source_name=source_name,
            output_dir=output_dir,
            monitor_store=monitor.store,
            df_previous=df_previous,
        )

    @classmethod
    def from_monitoring_store(
        cls,
        store: MonitoringStore,
        df: pd.DataFrame | None = None,
        source_name: str = "monitoring",
        output_dir: str | Path = "reports/data_quality",
    ) -> DataQualityDashboard:
        """Create a dashboard from a MonitoringStore directly."""
        return cls(
            df=df,
            source_name=source_name,
            output_dir=output_dir,
            monitor_store=store,
        )

    # ── Utility ─────────────────────────────────────────

    @staticmethod
    def _e(text: str) -> str:
        """Escape HTML special characters."""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
