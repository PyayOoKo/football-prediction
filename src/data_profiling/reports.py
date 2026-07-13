"""
Report Generator — produces HTML dashboards with Plotly interactive charts.

Generates a self-contained HTML report covering all profiling dimensions.
Each metric section includes an interactive Plotly visualization.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_profiling.profiler import ProfileSection, ProfilingReport

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates an interactive HTML dashboard from a ProfilingReport.

    Parameters
    ----------
    report : ProfilingReport
        The profiling report to visualize.
    """

    def __init__(self, report: ProfilingReport) -> None:
        self.report = report

    def to_html(self, filepath: str) -> None:
        """Generate and save the HTML dashboard."""
        html = self._generate_html()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML dashboard saved to %s", filepath)

    def _generate_html(self) -> str:
        report = self.report
        figures_html = self._render_all_sections()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📊 Data Profile — {report.source_name}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0e1117; color: #e0e0e0; padding: 24px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}

  .header {{ background: linear-gradient(135deg, #1a1d27 0%, #16213e 50%, #1a1d27 100%);
             border: 1px solid #2a2d3a; border-radius: 16px; padding: 2rem;
             margin-bottom: 2rem; }}
  .header h1 {{ font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem 0;
               background: linear-gradient(90deg, #4fc3f7, #81c784);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .header .meta {{ color: #8b8fa3; font-size: 0.9rem; }}
  .header .meta span {{ margin-right: 1.5rem; }}

  .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 1rem; margin-bottom: 2rem; }}
  .summary-card {{ background: linear-gradient(135deg, #1a1d27, #222639);
                   border: 1px solid #2a2d3a; border-radius: 12px;
                   padding: 1.25rem; text-align: center; }}
  .summary-card .value {{ font-size: 1.8rem; font-weight: 700; color: #fff; }}
  .summary-card .label {{ font-size: 0.8rem; color: #8b8fa3; text-transform: uppercase;
                          letter-spacing: 0.05em; margin-top: 0.25rem; }}

  .section {{ background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
              border: 1px solid #2a2d3a; border-radius: 12px;
              padding: 1.5rem; margin-bottom: 1.5rem; }}
  .section h2 {{ font-size: 1.2rem; font-weight: 600; color: #fff; margin-bottom: 0.5rem; }}
  .section .desc {{ color: #8b8fa3; font-size: 0.85rem; margin-bottom: 1rem; }}

  .chart-container {{ width: 100%; height: auto; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 0.6rem 0.8rem; background: #2a2d3a;
        color: #8b8fa3; font-weight: 600; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 0.05em;
        border-bottom: 2px solid #333; }}
  td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #2a2d3a;
        font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.8rem; }}
  tr:hover td {{ background: #2a2d3a; }}

  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
            font-size: 0.7rem; font-weight: 600; }}
  .badge-ok {{ background: #1b5e20; color: #81c784; }}
  .badge-warn {{ background: #e65100; color: #ffcc80; }}
  .badge-err {{ background: #b71c1c; color: #ef9a9a; }}

  .footer {{ text-align: center; color: #555; font-size: 0.8rem; margin-top: 2rem;
             padding-top: 1rem; border-top: 1px solid #2a2d3a; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📊 Dataset Profile — {self._esc(report.source_name)}</h1>
    <div class="meta">
      <span>📅 {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}</span>
      <span>📊 {report.n_rows:,} rows × {report.n_columns} columns</span>
      <span>⏱ {report.duration_seconds:.2f}s</span>
    </div>
  </div>

  <div class="summary-cards">
    <div class="summary-card"><div class="value">{report.n_rows:,}</div><div class="label">Rows</div></div>
    <div class="summary-card"><div class="value">{report.n_columns}</div><div class="label">Columns</div></div>
    {self._summary_card("Missing", self._get_missing_pct(), "%", "warn" if self._get_missing_pct() > 0 else "ok")}
    {self._summary_card("Duplicates", self._get_dup_count(), "", "warn" if self._get_dup_count() > 0 else "ok")}
    {self._summary_card("Duration", f"{report.duration_seconds:.1f}", "s", "ok")}
  </div>

  {figures_html}

  <div class="footer">
    Generated by Football Data Profiling System &mdash; {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
  </div>
</div>
</body>
</html>"""

    def _summary_card(self, label: str, value: Any, unit: str = "",
                      badge: str = "ok") -> str:
        return f'<div class="summary-card"><div class="value">{value}{unit}</div><div class="label"><span class="badge badge-{badge}">{label}</span></div></div>'

    def _get_missing_pct(self) -> float:
        mv = self.report.missing_values.data
        if isinstance(mv, dict):
            return float(mv.get('missing_pct', 0))
        return 0.0

    def _get_dup_count(self) -> int:
        dups = self.report.duplicate_records.data
        if isinstance(dups, dict):
            return int(dups.get("count", 0))
        return 0

    # ── Render all sections ───────────────────────────

    def _render_all_sections(self) -> str:
        renderers = [
            ("Missing Values", self._render_missing_values),
            ("Duplicate Records", self._render_duplicates),
            ("Column Summary", self._render_column_summary),
            ("Result Distribution", self._render_result_distribution),
            ("Goal Distribution", self._render_goal_distribution),
            ("Home Advantage", self._render_home_advantage),
            ("Odds Distribution", self._render_odds),
            ("League Distribution", self._render_categorical, "league_distribution"),
            ("Season Distribution", self._render_categorical, "season_distribution"),
            ("Team Distribution", self._render_team_distribution),
            ("Outliers", self._render_outliers),
            ("Schema Validation", self._render_schema),
            ("Type Validation", self._render_types),
        ]

        sections = []
        for renderer in renderers:
            try:
                if len(renderer) == 2:
                    name, fn = renderer
                    html = fn()
                else:
                    name, fn, attr = renderer
                    section = getattr(self.report, attr)
                    html = fn(section)

                if html:
                    sections.append(f'<div class="section"><h2>{name}</h2>{html}</div>')
            except Exception as exc:
                logger.warning("Failed to render section '%s': %s", name, exc)

        return "\n".join(sections)

    # ── Section renderers ────────────────────────────

    def _render_missing_values(self) -> str:
        mv = self.report.missing_values.data
        if not isinstance(mv, dict) or mv.get("total_cells", 0) == 0:
            return "<p>No data</p>"

        cols = mv.get("columns", {})
        if not cols:
            return '<p style="color:#81c784">✅ No missing values found</p>'

        # Bar chart
        top = dict(sorted(cols.items(), key=lambda x: x[1], reverse=True)[:20])
        fig = go.Figure(go.Bar(
            x=list(top.values()),
            y=list(top.keys()),
            orientation="h",
            marker=dict(color=["#ef4444" if v > 50 else "#f59e0b" if v > 20 else "#3b82f6" for v in top.values()]),
            text=[f"{v:.1f}%" for v in top.values()],
            textposition="outside",
        ))
        fig.update_layout(
            height=250, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title="Null %", showgrid=True, gridcolor="#2a2d3a"),
            yaxis=dict(title="", autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"),
        )
        return f"""<div class="desc">{mv.get('total_missing', 0):,} missing cells ({mv.get('missing_pct', 0):.1f}%) across {mv.get('columns_with_missing', 0)} columns</div>
<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False, default_width='100%', default_height='100%')}</div>"""

    def _render_duplicates(self) -> str:
        dups = self.report.duplicate_records.data
        if not isinstance(dups, dict):
            return "<p>No data</p>"
        count = dups.get("count", 0)
        if count == 0:
            return '<p style="color:#81c784">✅ No duplicate rows found</p>'
        return f'<p style="color:#ef9a9a">⚠ {count:,} duplicate rows ({dups.get("pct", 0):.1f}%)</p>'

    def _render_column_summary(self) -> str:
        df_cols = self.report.column_summary.data
        if not isinstance(df_cols, pd.DataFrame) or df_cols.empty:
            return "<p>No column data</p>"

        # Limit to first 50 columns for display
        display = df_cols.head(50)
        rows_html = ""
        for _, r in display.iterrows():
            badge = "ok" if r["null_pct"] == 0 else "warn" if r["null_pct"] < 20 else "err"
            rows_html += f"<tr><td>{self._esc(r['column'])}</td><td>{r['dtype']}</td><td><span class='badge badge-{badge}'>{r['null_pct']:.1f}%</span></td><td>{r['unique']}</td><td>{self._fmt(r['min'])}</td><td>{self._fmt(r['max'])}</td><td>{self._fmt(r['mean'])}</td></tr>"

        return f"""<table><thead><tr><th>Column</th><th>Type</th><th>Null %</th><th>Unique</th><th>Min</th><th>Max</th><th>Mean</th></tr></thead><tbody>{rows_html}</tbody></table>"""

    def _render_result_distribution(self) -> str:
        data = self.report.result_distribution.data
        if not isinstance(data, dict) or "counts" not in data:
            return "<p>No result data</p>"

        counts = data.get("counts", {})
        labels = {"H": "🏠 Home Win", "D": "🤝 Draw", "A": "✈️ Away Win"}
        colors = {"H": "#4caf50", "D": "#ffc107", "A": "#f44336"}

        fig = go.Figure()
        for k in ["H", "D", "A"]:
            val = counts.get(k, 0)
            if val > 0:
                fig.add_trace(go.Bar(
                    name=labels.get(k, k),
                    x=[labels.get(k, k)],
                    y=[val],
                    marker=dict(color=colors.get(k, "#888")),
                    text=[f"{val}<br>({data.get('percentages', {}).get(k, 0):.1f}%)"],
                    textposition="inside",
                ))

        fig.update_layout(
            height=250, margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(title="Count", showgrid=True, gridcolor="#2a2d3a"),
            xaxis=dict(title=""),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"), showlegend=False,
        )
        return f'<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%")}</div>'

    def _render_goal_distribution(self) -> str:
        data = self.report.goal_distribution.data
        if not isinstance(data, dict) or "histogram" not in data:
            return "<p>No goal data</p>"

        hh = data.get("home_hist", [])
        ah = data.get("away_hist", [])
        bins = list(range(len(max(hh, ah, key=len) if len(hh) < len(ah) else hh)))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bins, y=hh[:len(bins)], name="🏠 Home Goals",
                             marker=dict(color="#4caf50"), opacity=0.7))
        fig.add_trace(go.Bar(x=bins, y=ah[:len(bins)], name="✈️ Away Goals",
                             marker=dict(color="#f44336"), opacity=0.7))

        fig.update_layout(
            barmode="overlay", height=250, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title="Goals", dtick=1, showgrid=True, gridcolor="#2a2d3a"),
            yaxis=dict(title="Matches", showgrid=True, gridcolor="#2a2d3a"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        return f"""<div class="desc">Mean: {data.get('home_mean', '?'):.2f} (home) / {data.get('away_mean', '?'):.2f} (away) — Max: {data.get('max_goals', '?')}</div>
<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%")}</div>"""

    def _render_home_advantage(self) -> str:
        data = self.report.home_advantage.data
        if not isinstance(data, dict) or "home_win_pct" not in data:
            return "<p>No data</p>"

        fig = go.Figure(go.Bar(
            x=["🏠 Home Win", "🤝 Draw", "✈️ Away Win"],
            y=[data["home_win_pct"], data["draw_pct"], data["away_win_pct"]],
            marker=dict(color=["#4caf50", "#ffc107", "#f44336"]),
            text=[f"{data['home_win_pct']:.1f}%", f"{data['draw_pct']:.1f}%", f"{data['away_win_pct']:.1f}%"],
            textposition="outside",
        ))
        fig.update_layout(
            height=200, margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(range=[0, 60], title="%", showgrid=True, gridcolor="#2a2d3a"),
            xaxis=dict(title=""),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"), showlegend=False,
        )
        ha = data.get("home_advantage_pp", 0)
        return f"""<div class="desc">Home advantage: <strong>{ha:+.1f}pp</strong> (home {data['home_win_pct']:.1f}% vs away {data['away_win_pct']:.1f}%)</div>
<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%")}</div>"""

    def _render_odds(self) -> str:
        data = self.report.odds_distribution.data
        if not isinstance(data, dict) or "columns" not in data:
            return "<p>No odds data</p>"

        cols = data.get("columns", {})
        rows_html = ""
        for col, stats in list(cols.items())[:15]:
            rows_html += f"<tr><td>{self._esc(col)}</td><td>{stats['mean']:.2f}</td><td>{stats['median']:.2f}</td><td>{stats['min']:.2f}</td><td>{stats['max']:.2f}</td><td>{stats['n_valid']}</td><td>{stats['n_null']}</td></tr>"

        hist = data.get("histogram")
        hist_html = ""
        if hist:
            hb = hist["bins"]
            hc = hist["counts"]
            fig = go.Figure(go.Bar(
                x=[f"{hb[i]:.1f}-{hb[i+1]:.1f}" for i in range(len(hb)-1)],
                y=hc,
                marker=dict(color="#3b82f6"),
            ))
            fig.update_layout(
                height=200, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(title="Odds", showgrid=True, gridcolor="#2a2d3a"),
                yaxis=dict(title="Count", showgrid=True, gridcolor="#2a2d3a"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b8fa3"), showlegend=False,
            )
            hist_html = f'<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%")}</div>'

        return f"""<div class="desc">{data.get('n_odds_columns', 0)} odds columns detected</div>
<table><thead><tr><th>Column</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Valid</th><th>Null</th></tr></thead><tbody>{rows_html}</tbody></table>{hist_html}"""

    def _render_categorical(self, section: ProfileSection) -> str:
        data = section.data
        if not isinstance(data, dict) or "counts" not in data:
            return "<p>No data</p>"

        counts = data.get("counts", {})
        # Pie/bar chart
        keys = list(counts.keys())
        vals = list(counts.values())

        # Determine chart type based on cardinality
        if len(keys) <= 10:
            fig = go.Figure(go.Pie(
                labels=keys, values=vals,
                textinfo="label+percent",
                marker=dict(colors=px.colors.qualitative.Plotly),
            ))
        else:
            # Bar chart for high cardinality
            fig = go.Figure(go.Bar(
                x=vals[:30], y=keys[:30],
                orientation="h",
                marker=dict(color="#3b82f6"),
                text=[f"{v:,}" for v in vals[:30]],
            ))
            fig.update_layout(yaxis=dict(autorange="reversed"))

        fig.update_layout(
            height=250, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"),
        )
        return f"""<div class="desc">{data.get('n_unique', 0)} unique values</div>
<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%")}</div>"""

    def _render_team_distribution(self) -> str:
        data = self.report.team_distribution.data
        if not isinstance(data, dict) or "counts" not in data:
            return "<p>No team data</p>"

        counts = data.get("counts", {})
        # Top 20 teams as horizontal bar
        top = dict(list(counts.items())[:30])
        fig = go.Figure(go.Bar(
            x=list(top.values()),
            y=list(top.keys()),
            orientation="h",
            marker=dict(color="#8bc34a"),
            text=[f"{v:,}" for v in top.values()],
        ))
        fig.update_layout(
            height=400, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title="Matches", showgrid=True, gridcolor="#2a2d3a"),
            yaxis=dict(title="", autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b8fa3"), showlegend=False,
        )
        return f"""<div class="desc">{data.get('n_unique_teams', 0)} teams, {data.get('n_matches', 0)} matches</div>
<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%")}</div>"""

    def _render_outliers(self) -> str:
        data = self.report.outliers.data
        if not isinstance(data, dict) or "columns" not in data:
            return "<p>No outlier analysis</p>"

        cols = data.get("columns", {})
        if not cols:
            return f'<p style="color:#81c784">✅ No outliers detected (threshold: {data.get("threshold_std", 3)}σ)</p>'

        rows_html = ""
        for col, info in sorted(cols.items(), key=lambda x: x[1]["n_outliers"], reverse=True)[:20]:
            rows_html += f"<tr><td>{self._esc(col)}</td><td>{info['n_outliers']}</td><td>{info['pct']:.1f}%</td><td>{info['mean']:.2f}</td><td>{info['min_outlier']:.2f}</td><td>{info['max_outlier']:.2f}</td></tr>"

        return f"""<div class="desc">{data.get('n_columns_with_outliers', 0)} columns have outliers (>{data.get('threshold_std', 3)}σ)</div>
<table><thead><tr><th>Column</th><th>Outliers</th><th>%</th><th>Mean</th><th>Min Outlier</th><th>Max Outlier</th></tr></thead><tbody>{rows_html}</tbody></table>"""

    def _render_schema(self) -> str:
        data = self.report.schema_validation.data
        if not isinstance(data, dict):
            return "<p>No schema data</p>"

        missing = data.get("missing_columns", [])
        present = data.get("present_columns", [])

        parts = []
        if not missing:
            parts.append('<p style="color:#81c784">✅ All expected columns present</p>')
        else:
            parts.append(f'<p style="color:#ef9a9a">⚠ {len(missing)} expected columns missing: {", ".join(missing)}</p>')

        unexpected = data.get("unexpected_columns", [])
        if unexpected:
            parts.append(f'<p style="color:#8b8fa3">ℹ {data.get("n_unexpected", 0)} unexpected columns (e.g. {", ".join(unexpected[:8])})</p>')

        return "<br>".join(parts) + f"""<table><thead><tr><th>Expected</th><th>Status</th></tr></thead><tbody>
<tr><td>Columns present</td><td><span class="badge badge-ok">{len(present)}</span></td></tr>
<tr><td>Columns missing</td><td><span class="badge badge-{'ok' if not missing else 'err'}">{len(missing)}</span></td></tr>
<tr><td>Unexpected columns</td><td><span class="badge badge-warn">{data.get('n_unexpected', 0)}</span></td></tr>
</tbody></table>"""

    def _render_types(self) -> str:
        data = self.report.type_validation.data
        if not isinstance(data, dict) or "columns" not in data:
            return "<p>No type data</p>"

        cols = data.get("columns", {})
        issues = data.get("n_columns_with_issues", 0)

        rows_html = ""
        for col, info in list(cols.items())[:40]:
            issue_badge = ""
            if info["issues"]:
                issue_badge = f'<span class="badge badge-err">⚠</span>'
            rows_html += f"<tr><td>{self._esc(col)}</td><td>{info['dtype']}</td><td>{info['n_unique']}</td><td>{info['null_pct']:.1f}%</td><td>{issue_badge}</td></tr>"

        return f"""<div class="desc">{issues} columns with type issues</div>
<table><thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Null %</th><th></th></tr></thead><tbody>{rows_html}</tbody></table>"""

    @staticmethod
    def _esc(text: str) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

    @staticmethod
    def _fmt(val: Any) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)
