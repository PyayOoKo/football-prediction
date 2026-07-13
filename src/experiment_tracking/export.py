"""
Export module — save experiment tracking data as JSON, CSV, or HTML.

JSON
    Full structured export of experiments, runs, best models, and comparisons.
CSV
    Flat tables suitable for spreadsheet analysis (runs table, metrics table).
HTML
    Self-contained report with comparison tables, leaderboards,
    metric summary cards, and interactive Plotly charts.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.experiment_tracking.models import (
    BestModel,
    Experiment,
    ModelArtifact,
    Run,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  JSON Export
# ═══════════════════════════════════════════════════════════


def export_json(
    session: Session,
    *,
    experiment_id: str | None = None,
    output_path: str | Path | None = None,
    pretty: bool = True,
) -> str:
    """Export experiments and runs as JSON.

    Parameters
    ----------
    session : Session
    experiment_id : str, optional
        If specified, only export this experiment.
    output_path : str | Path, optional
        Write to file. If not provided, returns the JSON string.
    pretty : bool
        Pretty-print the JSON.

    Returns
    -------
    str
        JSON string (also written to ``output_path`` if provided).
    """
    data = _build_export_dict(session, experiment_id=experiment_id)
    json_str = json.dumps(data, indent=2, default=str) if pretty else json.dumps(data, default=str)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")
        logger.info("Exported experiments JSON to %s", output_path)

    return json_str


def _build_export_dict(
    session: Session,
    *,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Build the full export dictionary with nested runs and best models."""
    stmt = select(Experiment).order_by(Experiment.created_at.desc())
    if experiment_id is not None:
        stmt = stmt.where(Experiment.id == experiment_id)

    experiments = list(session.execute(stmt).scalars().all())

    exp_dicts = []
    for exp in experiments:
        exp_data = exp.to_dict()
        # Include runs eagerly so HTML/JSON exports have full data
        exp_data["runs"] = [r.to_dict() for r in (exp.runs or [])]
        exp_data["best_models_list"] = [
            b.to_dict() for b in (exp.best_models or [])
        ]
        exp_dicts.append(exp_data)

    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "experiment_count": len(experiments),
        "experiments": exp_dicts,
    }


# ═══════════════════════════════════════════════════════════
#  CSV Export
# ═══════════════════════════════════════════════════════════


def export_csv(
    session: Session,
    *,
    experiment_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Export experiment data as CSV files.

    Produces two CSV tables:
    - ``experiments.csv`` — summary of all experiments
    - ``runs.csv`` — all runs with metrics and hyperparameters flattened

    Parameters
    ----------
    session : Session
    experiment_id : str, optional
    output_dir : str | Path, optional
        Directory to write CSV files. If not provided, returns
        dict of ``{filename: csv_string}``.

    Returns
    -------
    dict[str, str]
        ``{"experiments.csv": "...", "runs.csv": "..."}``
    """
    # Build filter
    stmt_exp = select(Experiment).order_by(Experiment.created_at.desc())
    if experiment_id is not None:
        stmt_exp = stmt_exp.where(Experiment.id == experiment_id)
    experiments = list(session.execute(stmt_exp).scalars().all())

    results: dict[str, str] = {}

    # ── Experiments CSV ─────────────────────────────────
    exp_buf = io.StringIO()
    exp_writer = csv.writer(exp_buf)
    exp_writer.writerow([
        "id", "name", "description", "dataset_version", "feature_version",
        "model_version", "git_commit", "run_count", "created_at", "updated_at",
    ])
    for exp in experiments:
        exp_writer.writerow([
            exp.id, exp.name, exp.description or "", exp.dataset_version or "",
            exp.feature_version or "", exp.model_version or "",
            exp.git_commit or "", len(exp.runs) if exp.runs else 0,
            exp.created_at.isoformat() if exp.created_at else "",
            exp.updated_at.isoformat() if exp.updated_at else "",
        ])
    results["experiments.csv"] = exp_buf.getvalue()

    # ── Runs CSV ────────────────────────────────────────
    run_buf = io.StringIO()
    run_writer = csv.writer(run_buf)

    # Collect all metric names across runs
    all_runs: list[Run] = []
    for exp in experiments:
        if exp.runs:
            all_runs.extend(exp.runs)
    all_metric_names: set[str] = set()
    for run in all_runs:
        if run.metrics:
            all_metric_names.update(run.metrics.keys())

    # Headers
    base_headers = [
        "id", "experiment_id", "experiment_name", "run_name", "model_type",
        "model_version", "status", "random_seed",
        "training_duration_seconds", "git_commit", "error_message",
        "started_at", "finished_at",
    ]
    run_writer.writerow(base_headers + sorted(all_metric_names))

    for run in all_runs:
        exp_name = ""
        if run.experiment:
            exp_name = run.experiment.name
        metrics = run.metrics or {}
        run_writer.writerow([
            run.id, run.experiment_id, exp_name, run.run_name or "",
            run.model_type, run.model_version or "", run.status,
            run.random_seed or "", run.training_duration_seconds or "",
            run.git_commit or "", run.error_message or "",
            run.started_at.isoformat() if run.started_at else "",
            run.finished_at.isoformat() if run.finished_at else "",
        ] + [metrics.get(m, "") for m in sorted(all_metric_names)])

    results["runs.csv"] = run_buf.getvalue()

    # ── Write to disk ────────────────────────────────────
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for filename, content in results.items():
            (output_path / filename).write_text(content, encoding="utf-8")
        logger.info("Exported experiment CSVs to %s", output_path)

    return results


# ═══════════════════════════════════════════════════════════
#  HTML Report
# ═══════════════════════════════════════════════════════════


def export_html(
    session: Session,
    *,
    experiment_id: str | None = None,
    output_path: str | Path | None = None,
    title: str = "ML Experiment Report",
) -> str:
    """Generate a self-contained HTML report with charts and tables.

    The report includes:
    - Experiment summary cards
    - Per-experiment run tables
    - Metric comparison charts (Plotly.js via CDN)
    - Best-per-metric highlights
    - Leaderboards

    Parameters
    ----------
    session : Session
    experiment_id : str, optional
    output_path : str | Path, optional
    title : str

    Returns
    -------
    str
        HTML string.
    """
    data = _build_export_dict(session, experiment_id=experiment_id)
    html = _render_html_report(data, title=title)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Exported experiment HTML report to %s", output_path)

    return html


def _render_html_report(
    data: dict[str, Any],
    title: str = "ML Experiment Report",
) -> str:
    """Render the HTML report from export data."""
    experiments = data.get("experiments", [])

    # Build experiment cards and chart data
    exp_cards_html = ""
    chart_data_jsons: list[str] = []

    for i, exp in enumerate(experiments):
        runs = exp.get("runs", [])
        run_count = exp.get("run_count", 0)

        # Gather metrics across this experiment's runs
        all_metrics: dict[str, list[tuple[str, float]]] = {}
        for run in runs:
            metrics = run.get("metrics", {})
            for m_key, m_val in metrics.items():
                if isinstance(m_val, (int, float)):
                    all_metrics.setdefault(m_key, []).append(
                        (run.get("run_name") or run.get("model_type") or "?", m_val),
                    )

        # Build chart JSON for this experiment
        chart_json = _build_metric_chart_json(exp.get("name", "?"), all_metrics)
        chart_data_jsons.append(chart_json)

        # Build runs table HTML
        runs_table_html = _build_runs_table_html(runs)

        exp_cards_html += f"""
        <div class="exp-card">
            <h2>{html_escape(exp.get("name", "?"))}</h2>
            <div class="exp-meta">
                <span class="tag">ID: {exp.get("id", "?")[:8]}</span>
                <span class="tag">Runs: {run_count}</span>
                <span class="tag">Dataset: {html_escape(exp.get("dataset_version") or "—")}</span>
                <span class="tag">Features: {html_escape(exp.get("feature_version") or "—")}</span>
            </div>
            <div id="chart-{i}" class="chart-container"></div>
            {runs_table_html}
            <div class="exp-notes">{html_escape(exp.get("notes") or "")}</div>
        </div>
        """

    # Build leaderboard
    leaderboard_html = _build_leaderboard_html(experiments)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_escape(title)}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg: #0f1117;
            --bg-card: #1a1d2e;
            --bg-hover: #242840;
            --text: #e4e6f0;
            --text-dim: #8b8fa3;
            --accent: #6c5ce7;
            --accent-green: #00b894;
            --accent-red: #e17055;
            --accent-yellow: #fdcb6e;
            --border: #2d3052;
            --radius: 12px;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 40px 20px;
            max-width: 1200px;
            margin: 0 auto;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, var(--bg-card), #2d1b69);
            border-radius: var(--radius);
            border: 1px solid var(--border);
        }}
        .header h1 {{ font-size: 2rem; margin-bottom: 8px; }}
        .header .subtitle {{ color: var(--text-dim); font-size: 0.95rem; }}
        .exp-card {{
            background: var(--bg-card);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--border);
        }}
        .exp-card h2 {{ font-size: 1.3rem; margin-bottom: 12px; }}
        .exp-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
        .tag {{
            background: var(--bg-hover);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            color: var(--text-dim);
        }}
        .chart-container {{ height: 300px; margin: 16px 0; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-dim);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        tr:hover td {{ background: var(--bg-hover); }}
        .best {{ color: var(--accent-green); font-weight: 600; }}
        .exp-notes {{ margin-top: 12px; color: var(--text-dim); font-size: 0.85rem; font-style: italic; }}
        .leaderboard {{
            background: var(--bg-card);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--border);
        }}
        .leaderboard h2 {{ font-size: 1.3rem; margin-bottom: 16px; }}
        .rank-1 {{ color: #ffd700; font-weight: 700; }}
        .rank-2 {{ color: #c0c0c0; font-weight: 600; }}
        .rank-3 {{ color: #cd7f32; font-weight: 600; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
            margin: 16px 0;
        }}
        .metric-card {{
            background: var(--bg-hover);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .label {{ font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; }}
        .metric-card .value {{ font-size: 1.4rem; font-weight: 700; margin-top: 4px; }}
        .footer {{
            text-align: center;
            color: var(--text-dim);
            font-size: 0.8rem;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{html_escape(title)}</h1>
        <div class="subtitle">
            Exported {len(experiments)} experiment(s)
            &middot; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
        </div>
    </div>

    {leaderboard_html}

    {exp_cards_html}

    <div class="footer">
        Generated by Experiment Tracking System
    </div>

    <script>
    {_chart_js_code(chart_data_jsons)}
    </script>
</body>
</html>"""
    return html


def _build_runs_table_html(runs: list[dict[str, Any]]) -> str:
    """Build an HTML table for runs."""
    if not runs:
        return '<p style="color: var(--text-dim);">No runs recorded.</p>'

    # Collect all metric columns
    all_metric_keys: set[str] = set()
    for run in runs:
        if run.get("metrics"):
            all_metric_keys.update(run.get("metrics", {}).keys())

    sorted_metrics = sorted(all_metric_keys)

    # Determine best value per metric
    best_values: dict[str, float] = {}
    best_run_ids: dict[str, str] = {}
    for m in sorted_metrics:
        lower_is_better = any(n in m.lower() for n in ["loss", "error", "brier", "mse", "mae", "rmse"])
        best_val = None
        best_rid = None
        for run in runs:
            val = run.get("metrics", {}).get(m)
            if val is None:
                continue
            if best_val is None:
                best_val = val
                best_rid = run.get("id", "")
            elif lower_is_better and val < best_val:
                best_val = val
                best_rid = run.get("id", "")
            elif not lower_is_better and val > best_val:
                best_val = val
                best_rid = run.get("id", "")

        if best_val is not None:
            best_values[m] = best_val
            best_run_ids[m] = best_rid or ""

    # Table header
    headers = ["Run ID", "Model", "Status", "Duration (s)", "Seed"] + sorted_metrics
    rows_html = ""
    for run in runs:
        rid = run.get("id", "")[:8]
        model = html_escape(run.get("model_type", "?"))
        status = run.get("status", "?")
        dur = run.get("training_duration_seconds")
        dur_str = f"{dur:.2f}" if dur is not None else "—"
        seed = run.get("random_seed") or "—"

        cells = [rid, model, status, dur_str, str(seed)]
        for m in sorted_metrics:
            val = run.get("metrics", {}).get(m)
            if val is not None:
                val_str = f"{val:.4f}"
                # Highlight best value
                if m in best_values and rid == best_run_ids.get(m, "")[:8]:
                    val_str = f'<span class="best">{val_str} &#9733;</span>'
            else:
                val_str = "—"
            cells.append(val_str)

        row_cells = "".join(f"<td>{c}</td>" for c in cells)
        rows_html += f"<tr>{row_cells}</tr>\n"

    header_html = "".join(f"<th>{h}</th>" for h in headers)
    return f"""<table>
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>"""


def _build_metric_chart_json(experiment_name: str, all_metrics: dict[str, list[tuple[str, float]]]) -> str:
    """Build a JSON array of Plotly trace objects for each metric."""
    traces = []
    for metric_name, values in all_metrics.items():
        if len(values) < 2:
            continue  # Skip metrics with fewer than 2 data points
        names = [v[0] for v in values]
        vals = [v[1] for v in values]
        traces.append({
            "type": "bar",
            "name": metric_name,
            "x": names,
            "y": vals,
            "text": [f"{v:.4f}" for v in vals],
            "textposition": "auto",
            "marker": {"opacity": 0.85},
        })

    if not traces:
        return json.dumps({})

    return json.dumps({
        "data": traces,
        "layout": {
            "title": f"Metrics — {experiment_name}",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#e4e6f0"},
            "barmode": "group",
            "xaxis": {"title": "Run", "gridcolor": "#2d3052"},
            "yaxis": {"title": "Value", "gridcolor": "#2d3052"},
            "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    })


def _build_leaderboard_html(experiments: list[dict[str, Any]]) -> str:
    """Build the leaderboard section showing top models by each metric."""
    # Collect all (metric_name, value, model_type, run_id, exp_name) across experiments
    all_entries: dict[str, list[dict[str, Any]]] = {}

    for exp in experiments:
        exp_name = exp.get("name", "?")
        for run in exp.get("runs", []):
            metrics = run.get("metrics", {})
            for m_key, m_val in metrics.items():
                if isinstance(m_val, (int, float)):
                    all_entries.setdefault(m_key, []).append({
                        "value": m_val,
                        "model_type": run.get("model_type", "?"),
                        "run_name": run.get("run_name") or run.get("model_type", "?")[:12],
                        "experiment": exp_name,
                    })

    if not all_entries:
        return ""

    sections_html = ""
    for metric_name in sorted(all_entries.keys()):
        entries = all_entries[metric_name]
        lower_is_better = any(n in metric_name.lower() for n in ["loss", "error", "brier", "mse", "mae", "rmse"])
        entries.sort(key=lambda e: e["value"], reverse=not lower_is_better)

        # Top 5
        top5 = entries[:5]
        rows = []
        for rank, e in enumerate(top5, 1):
            rank_class = f"rank-{rank}" if rank <= 3 else ""
            rows.append(
                f'<tr><td class="{rank_class}">#{rank}</td>'
                f'<td>{html_escape(e["model_type"])}</td>'
                f'<td>{html_escape(e["experiment"])}</td>'
                f'<td>{e["value"]:.4f}</td></tr>'
            )

        sections_html += f"""
        <div style="margin-bottom: 20px;">
            <h3 style="margin-bottom: 8px;">{html_escape(metric_name)}</h3>
            <table>
                <thead><tr><th>Rank</th><th>Model</th><th>Experiment</th><th>Value</th></tr></thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        </div>
        """

    return f"""<div class="leaderboard">
        <h2>&#127942; Leaderboard</h2>
        <div class="metrics-grid" id="leaderboard-grid">
            {sections_html}
        </div>
    </div>"""


def _chart_js_code(chart_data_jsons: list[str]) -> str:
    """Generate JavaScript to render all charts."""
    js = ""
    for i, chart_json_str in enumerate(chart_data_jsons):
        js += f"""
        (function() {{
            var data = {chart_json_str or "null"};
            if (data && data.data && data.data.length > 0) {{
                Plotly.newPlot('chart-{i}', data.data, data.layout, {{
                    responsive: true,
                    displayModeBar: false,
                }});
            }}
        }})();
        """
    return js


def html_escape(text: str | None) -> str:
    """HTML-escape a string."""
    if text is None:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
