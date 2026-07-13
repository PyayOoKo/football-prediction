"""
Profiler — core profiling engine for football match datasets.

Analyzes a DataFrame and produces a comprehensive ``ProfilingReport``
with metrics across all requested categories.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProfileSection:
    """One section of a profiling report (e.g. missing values, distributions)."""

    name: str
    data: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame
    chart_type: str = "table"  # table, bar, histogram, heatmap, pie
    description: str = ""


@dataclass
class ProfilingReport:
    """Complete profiling report for a single dataset.

    Supports export to HTML (Plotly dashboard), JSON, and CSV.
    """

    source_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    n_rows: int = 0
    n_columns: int = 0
    duration_seconds: float = 0.0

    # Core metrics
    missing_values: ProfileSection = field(default_factory=lambda: ProfileSection("Missing Values", {}))
    duplicate_records: ProfileSection = field(default_factory=lambda: ProfileSection("Duplicate Records", {}))
    column_summary: ProfileSection = field(default_factory=lambda: ProfileSection("Column Summary", {}))
    result_distribution: ProfileSection = field(default_factory=lambda: ProfileSection("Result Distribution", {}))
    goal_distribution: ProfileSection = field(default_factory=lambda: ProfileSection("Goal Distribution", {}))
    odds_distribution: ProfileSection = field(default_factory=lambda: ProfileSection("Odds Distribution", {}))
    league_distribution: ProfileSection = field(default_factory=lambda: ProfileSection("League Distribution", {}))
    season_distribution: ProfileSection = field(default_factory=lambda: ProfileSection("Season Distribution", {}))
    team_distribution: ProfileSection = field(default_factory=lambda: ProfileSection("Team Distribution", {}))
    home_advantage: ProfileSection = field(default_factory=lambda: ProfileSection("Home Advantage", {}))
    outliers: ProfileSection = field(default_factory=lambda: ProfileSection("Outliers", {}))
    schema_validation: ProfileSection = field(default_factory=lambda: ProfileSection("Schema Validation", {}))
    type_validation: ProfileSection = field(default_factory=lambda: ProfileSection("Type Validation", {}))

    # Optional: trends vs previous report
    data_drift: ProfileSection | None = None

    def to_dict(self) -> dict[str, Any]:
        """Export the full report as a dict (JSON-serializable)."""
        def _section_to_dict(s: ProfileSection) -> dict[str, Any]:
            d = s.data
            if isinstance(d, pd.DataFrame):
                d = d.to_dict(orient="records") if len(d) < 1000 else {"_truncated": True, "n_rows": len(d)}
            return {"name": s.name, "data": d, "chart_type": s.chart_type}

        result: dict[str, Any] = {
            "source_name": self.source_name,
            "timestamp": self.timestamp.isoformat(),
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duration_seconds": round(self.duration_seconds, 3),
            "sections": {},
        }

        for field_name in [
            "missing_values", "duplicate_records", "column_summary",
            "result_distribution", "goal_distribution", "odds_distribution",
            "league_distribution", "season_distribution", "team_distribution",
            "home_advantage", "outliers", "schema_validation", "type_validation",
        ]:
            section = getattr(self, field_name)
            result["sections"][field_name] = _section_to_dict(section)

        if self.data_drift:
            result["sections"]["data_drift"] = _section_to_dict(self.data_drift)

        return result

    def to_json(self, filepath: str | None = None) -> str:
        """Export to JSON string or write to file."""
        import json
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
            return ""
        return json_str

    def to_csv(self, filepath: str) -> None:
        """Export key metrics to CSV for spreadsheet analysis."""
        rows: list[dict[str, Any]] = []

        # Column summary
        col_data = self.column_summary.data
        if isinstance(col_data, pd.DataFrame):
            for _, r in col_data.iterrows():
                rows.append(r.to_dict())

        # Missing values
        mv = self.missing_values.data
        if isinstance(mv, dict):
            for col, pct in mv.get("columns", {}).items():
                rows.append({"metric": "null_pct", "column": col, "value": pct})

        # Distributions
        for section_name in ["result_distribution", "league_distribution",
                              "season_distribution", "team_distribution"]:
            section = getattr(self, section_name)
            data = section.data
            if isinstance(data, dict):
                for key, val in data.get("counts", {}).items():
                    rows.append({"metric": f"{section_name}_count", "key": str(key), "value": val})

        df_out = pd.DataFrame(rows)
        df_out.to_csv(filepath, index=False)

    def to_html(self, filepath: str) -> None:
        """Generate an interactive HTML dashboard with Plotly charts."""
        from src.data_profiling.reports import ReportGenerator
        ReportGenerator(self).to_html(filepath)

    def summary_text(self) -> str:
        """Return a human-readable text summary."""
        lines = [
            f"📊 Profile: {self.source_name}",
            f"   Rows: {self.n_rows:,}  |  Columns: {self.n_columns}",
            f"   Duration: {self.duration_seconds:.2f}s",
            f"   Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]

        # Add key findings
        mv = self.missing_values.data
        if isinstance(mv, dict):
            null_cols = [k for k, v in mv.get("columns", {}).items() if v > 0]
            if null_cols:
                lines.append(f"   ⚠ {len(null_cols)} columns have missing values")
            else:
                lines.append("   ✅ No missing values")

        dups = self.duplicate_records.data
        if isinstance(dups, dict) and dups.get("count", 0) > 0:
            lines.append(f"   ⚠ {dups['count']} duplicate rows found")
        else:
            lines.append("   ✅ No duplicate rows")

        return "\n".join(lines)


class DataProfiler:
    """Profiles a football match DataFrame across all analysis dimensions.

    Parameters
    ----------
    odds_column_patterns : list[str], optional
        Column name substrings to identify odds columns (auto-detected).
    outlier_std_threshold : float
        Number of standard deviations to flag outliers (default 3.0).
    max_unique_for_distribution : int
        Max unique values for bar-chart distributions (default 50).
    """

    def __init__(
        self,
        odds_column_patterns: list[str] | None = None,
        outlier_std_threshold: float = 3.0,
        max_unique_for_distribution: int = 50,
    ) -> None:
        self.odds_column_patterns = odds_column_patterns or [
            "bbav", "b365", "bwin", "psh", "psd", "psa",
            "maxh", "maxd", "maxa", "avh", "avd", "ava",
            "home_odds", "draw_odds", "away_odds",
            "odds_home", "odds_draw", "odds_away",
        ]
        self.outlier_std_threshold = outlier_std_threshold
        self.max_unique = max_unique_for_distribution

    def profile(self, df: pd.DataFrame, source_name: str = "unknown") -> ProfilingReport:
        """Run all profiling analyses on the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to profile.
        source_name : str
            Identifier for the source (e.g. ``worldcup-2026``).

        Returns
        -------
        ProfilingReport
        """
        start = time.perf_counter()
        logger.info("Profiling dataset: %s (%d rows, %d cols)", source_name, len(df), len(df.columns))

        report = ProfilingReport(
            source_name=source_name,
            n_rows=len(df),
            n_columns=len(df.columns),
        )

        # Run each analysis
        report.missing_values = self._profile_missing_values(df)
        report.duplicate_records = self._profile_duplicates(df)
        report.column_summary = self._profile_columns(df)
        report.result_distribution = self._profile_result_distribution(df)
        report.goal_distribution = self._profile_goal_distribution(df)
        report.odds_distribution = self._profile_odds_distribution(df)
        report.league_distribution = self._profile_categorical(df, "league", "League Distribution")
        report.season_distribution = self._profile_categorical(df, "season", "Season Distribution")
        report.team_distribution = self._profile_teams(df)
        report.home_advantage = self._profile_home_advantage(df)
        report.outliers = self._profile_outliers(df)
        report.schema_validation = self._profile_schema(df)
        report.type_validation = self._profile_types(df)

        report.duration_seconds = time.perf_counter() - start
        logger.info("Profiling complete in %.2fs", report.duration_seconds)
        return report

    # ── 1. Missing Values ──────────────────────────────

    def _profile_missing_values(self, df: pd.DataFrame) -> ProfileSection:
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df) * 100).round(2)
        nonzero = null_pct[null_pct > 0]

        data = {
            "total_cells": int(df.size),
            "total_missing": int(df.isnull().sum().sum()),
            "missing_pct": round(float(df.isnull().sum().sum() / df.size * 100), 2),
            "columns_with_missing": len(nonzero),
            "columns": {k: float(v) for k, v in null_pct.items() if v > 0},
            "top_missing": nonzero.sort_values(ascending=False).head(20).to_dict(),
        }

        chart_data = pd.DataFrame({
            "column": nonzero.index,
            "null_pct": nonzero.values,
        }).sort_values("null_pct", ascending=False).head(20)

        return ProfileSection("Missing Values", data, chart_type="bar",
                              description="Columns with null values (count and percentage)")

    # ── 2. Duplicates ─────────────────────────────────

    def _profile_duplicates(self, df: pd.DataFrame) -> ProfileSection:
        dup_count = df.duplicated().sum()
        data = {
            "count": int(dup_count),
            "pct": round(float(dup_count / len(df) * 100) if len(df) > 0 else 0, 2),
        }
        if dup_count > 0:
            data["sample"] = df[df.duplicated(keep="first")].head(10).to_dict(orient="records")

        return ProfileSection("Duplicate Records", data, chart_type="table",
                              description=f"Duplicate rows found: {dup_count} ({data['pct']:.1f}%)")

    # ── 3. Column Summary ──────────────────────────────

    def _profile_columns(self, df: pd.DataFrame) -> ProfileSection:
        rows = []
        for col in df.columns:
            non_null = df[col].notna().sum()
            null_c = int(df[col].isna().sum())
            null_pct = round(null_c / len(df) * 100, 2) if len(df) > 0 else 0
            dtype = str(df[col].dtype)
            unique = int(df[col].nunique())

            if pd.api.types.is_numeric_dtype(df[col]):
                col_min = float(df[col].min()) if non_null > 0 else None
                col_max = float(df[col].max()) if non_null > 0 else None
                col_mean = float(df[col].mean()) if non_null > 0 else None
                col_std = float(df[col].std()) if non_null > 0 else None
            else:
                col_min = col_max = col_mean = col_std = None

            rows.append({
                "column": col,
                "dtype": dtype,
                "non_null": non_null,
                "null_pct": null_pct,
                "unique": unique,
                "min": col_min,
                "max": col_max,
                "mean": col_mean,
                "std": col_std,
            })

        return ProfileSection("Column Summary", pd.DataFrame(rows), chart_type="table",
                              description="Per-column statistics: type, nulls, uniqueness, range")

    # ── 4. Result Distribution ─────────────────────────

    def _profile_result_distribution(self, df: pd.DataFrame) -> ProfileSection:
        result_col = None
        for candidate in ["result", "FTR", "full_time_result"]:
            if candidate in df.columns:
                result_col = candidate
                break

        if result_col is None:
            return ProfileSection("Result Distribution", {"error": "No result column found"}, chart_type="pie",
                                  description="Match outcome distribution (H/D/A)")

        counts = df[result_col].value_counts()
        pcts = (counts / len(df) * 100).round(1)
        known = ["H", "D", "A"]
        for k in known:
            if k not in counts.index:
                counts[k] = 0
                pcts[k] = 0.0

        data = {
            "column": result_col,
            "counts": {str(k): int(v) for k, v in counts.items() if str(k) in known},
            "percentages": {str(k): float(v) for k, v in pcts.items() if str(k) in known},
        }

        return ProfileSection("Result Distribution", data, chart_type="pie",
                              description="Home win / Draw / Away win distribution")

    # ── 5. Goal Distribution ──────────────────────────

    def _profile_goal_distribution(self, df: pd.DataFrame) -> ProfileSection:
        hg = None
        ag = None
        for hc in ["home_goals", "home_goal", "FTHG"]:
            if hc in df.columns:
                hg = df[hc]
                break
        for ac in ["away_goals", "away_goal", "FTAG"]:
            if ac in df.columns:
                ag = df[ac]
                break

        if hg is None or ag is None:
            return ProfileSection("Goal Distribution", {"error": "Goal columns not found"}, chart_type="histogram",
                                  description="Goal scoring distribution")

        total_goals = hg.fillna(0) + ag.fillna(0)

        # Build histogram bins
        max_g = int(total_goals.max()) if len(total_goals) > 0 else 10
        max_g = min(max_g, 15)  # Cap at 15 for readability
        bins = list(range(0, max_g + 2))
        hist, edges = np.histogram(total_goals, bins=bins)

        data = {
            "home_mean": round(float(hg.mean()), 3),
            "away_mean": round(float(ag.mean()), 3),
            "total_mean": round(float(total_goals.mean()), 3),
            "max_goals": int(total_goals.max()),
            "histogram": {
                "bins": [f"{int(e)}-{int(bins[i+1])}" for i, e in enumerate(bins[:-1])],
                "counts": hist.tolist(),
            },
            "home_hist": np.histogram(hg.fillna(0), bins=list(range(0, 10)))[0].tolist(),
            "away_hist": np.histogram(ag.fillna(0), bins=list(range(0, 10)))[0].tolist(),
        }

        return ProfileSection("Goal Distribution", data, chart_type="histogram",
                              description="Home and away goal distributions")

    # ── 6. Odds Distribution ─────────────────────────

    def _profile_odds_distribution(self, df: pd.DataFrame) -> ProfileSection:
        # Auto-detect odds columns
        odds_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in self.odds_column_patterns):
                if pd.api.types.is_numeric_dtype(df[col]):
                    odds_cols.append(col)

        if not odds_cols:
            # Try known odds column patterns
            known_sets = [
                ("BbAvH", "BbAvD", "BbAvA"),
                ("B365H", "B365D", "B365A"),
                ("BWH", "BWD", "BWA"),
            ]
            for ks in known_sets:
                found = [c for c in ks if c in df.columns]
                if len(found) >= 2:
                    odds_cols = found
                    break

        if not odds_cols:
            return ProfileSection("Odds Distribution", {"error": "No odds columns detected"}, chart_type="histogram",
                                  description="Bookmaker odds distributions")

        stats = {}
        for col in odds_cols[:10]:  # Cap at 10 columns
            vals = df[col].dropna()
            if len(vals) > 0:
                stats[col] = {
                    "mean": round(float(vals.mean()), 3),
                    "median": round(float(vals.median()), 3),
                    "min": round(float(vals.min()), 3),
                    "max": round(float(vals.max()), 3),
                    "std": round(float(vals.std()), 3),
                    "n_valid": int(len(vals)),
                    "n_null": int(df[col].isna().sum()),
                }

        # Build histogram for first home odds column
        histogram = None
        home_odds_col = next((c for c in odds_cols if c.lower().endswith("h")), odds_cols[0])
        vals = df[home_odds_col].dropna()
        if len(vals) > 0:
            hist_vals = np.clip(vals, 1.0, 20.0)
            h, b = np.histogram(hist_vals, bins=20)
            histogram = {
                "column": home_odds_col,
                "bins": [round(float(x), 2) for x in b],
                "counts": h.tolist(),
            }

        data = {"columns": stats, "n_odds_columns": len(odds_cols)}
        if histogram:
            data["histogram"] = histogram

        return ProfileSection("Odds Distribution", data, chart_type="histogram",
                              description=f"Distribution of {len(odds_cols)} odds columns")

    # ── 7. Categorical Distribution (league, season) ──

    def _profile_categorical(self, df: pd.DataFrame, col: str, name: str) -> ProfileSection:
        if col not in df.columns:
            return ProfileSection(name, {"error": f"Column '{col}' not found"}, chart_type="pie")

        counts = df[col].value_counts()
        if len(counts) > self.max_unique:
            # Top-N for readability
            other_count = int(counts.iloc[self.max_unique:].sum())
            counts = counts.head(self.max_unique)
            if other_count > 0:
                counts["OTHER"] = other_count

        data = {
            "column": col,
            "n_unique": int(df[col].nunique()),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): round(float(v / len(df) * 100), 1) for k, v in counts.items()},
        }

        return ProfileSection(name, data, chart_type="pie",
                              description=f"Distribution of {df[col].nunique()} unique values")

    # ── 8. Team Distribution ──────────────────────────

    def _profile_teams(self, df: pd.DataFrame) -> ProfileSection:
        home_ok = "home_team" in df.columns
        away_ok = "away_team" in df.columns

        if not home_ok and not away_ok:
            return ProfileSection("Team Distribution", {"error": "Team columns not found"}, chart_type="table")

        all_teams = pd.concat([
            df["home_team"] if home_ok else pd.Series(dtype="object"),
            df["away_team"] if away_ok else pd.Series(dtype="object"),
        ]).dropna()

        counts = all_teams.value_counts()
        if len(counts) > self.max_unique:
            top = counts.head(self.max_unique)
            other = int(counts.iloc[self.max_unique:].sum())
            counts = top
        else:
            other = 0

        data = {
            "n_unique_teams": int(all_teams.nunique()),
            "n_matches": int(len(all_teams) / 2),
            "counts": {str(k): int(v) for k, v in counts.items()},
        }
        if other > 0:
            data["counts"]["OTHER"] = other

        return ProfileSection("Team Distribution", data, chart_type="bar",
                              description=f"{all_teams.nunique()} unique teams")

    # ── 9. Home Advantage ─────────────────────────────

    def _profile_home_advantage(self, df: pd.DataFrame) -> ProfileSection:
        result_col = None
        for candidate in ["result", "FTR"]:
            if candidate in df.columns:
                result_col = candidate
                break

        if result_col is None:
            return ProfileSection("Home Advantage", {"error": "No result column"}, chart_type="bar")

        n = len(df)
        home_wins = float((df[result_col] == "H").sum() / n * 100) if n > 0 else 0
        away_wins = float((df[result_col] == "A").sum() / n * 100) if n > 0 else 0
        draws = float((df[result_col] == "D").sum() / n * 100) if n > 0 else 0

        data = {
            "home_win_pct": round(home_wins, 1),
            "draw_pct": round(draws, 1),
            "away_win_pct": round(away_wins, 1),
            "home_advantage_pp": round(home_wins - away_wins, 1),
            "n_matches": n,
        }

        return ProfileSection("Home Advantage", data, chart_type="bar",
                              description=f"Home win rate: {home_wins:.1f}% vs Away: {away_wins:.1f}%")

    # ── 10. Outliers ──────────────────────────────────

    def _profile_outliers(self, df: pd.DataFrame) -> ProfileSection:
        numeric = df.select_dtypes(include=[np.number])
        outlier_info: dict[str, dict[str, Any]] = {}

        for col in numeric.columns:
            vals = numeric[col].dropna()
            if len(vals) < 5:
                continue
            mean, std = vals.mean(), vals.std()
            if std == 0:
                continue
            z_scores = np.abs((vals - mean) / std)
            n_outliers = int((z_scores > self.outlier_std_threshold).sum())
            if n_outliers > 0:
                outlier_info[col] = {
                    "n_outliers": n_outliers,
                    "pct": round(float(n_outliers / len(vals) * 100), 2),
                    "mean": round(float(mean), 3),
                    "std": round(float(std), 3),
                    "threshold": round(float(self.outlier_std_threshold * std), 3),
                    "min_outlier": round(float(vals[z_scores > self.outlier_std_threshold].min()), 3),
                    "max_outlier": round(float(vals[z_scores > self.outlier_std_threshold].max()), 3),
                }

        data = {
            "n_columns_with_outliers": len(outlier_info),
            "threshold_std": self.outlier_std_threshold,
            "columns": outlier_info,
        }

        return ProfileSection("Outliers", data, chart_type="table",
                              description=f"Z-score outliers above {self.outlier_std_threshold}σ")

    # ── 11. Schema Validation ─────────────────────────

    def _profile_schema(self, df: pd.DataFrame) -> ProfileSection:
        expected_columns = [
            "date", "home_team", "away_team", "result",
            "home_goals", "away_goals", "league", "season",
        ]
        present = [c for c in expected_columns if c in df.columns]
        missing = [c for c in expected_columns if c not in df.columns]

        data = {
            "n_columns_present": len(present),
            "n_columns_missing": len(missing),
            "present_columns": present,
            "missing_columns": missing,
            "unexpected_columns": [c for c in df.columns if c not in expected_columns and c not in ["home_goals_ht", "away_goals_ht", "match_id"]][:30],
            "n_unexpected": len(df.columns) - len(present),
        }

        return ProfileSection("Schema Validation", data, chart_type="table",
                              description=f"{len(present)}/{len(expected_columns)} expected columns present")

    # ── 12. Type Validation ──────────────────────────

    def _profile_types(self, df: pd.DataFrame) -> ProfileSection:
        type_info: dict[str, dict[str, Any]] = {}

        for col in df.columns:
            raw_dtype = df[col].dtype
            inferred = str(raw_dtype)
            n_unique = int(df[col].nunique())

            # Check for type issues
            issues = []
            if pd.api.types.is_object_dtype(df[col]) and n_unique > 1 and n_unique < 20:
                # Might be categorical rather than string
                pass
            if col in ["home_goals", "away_goals", "home_goals_ht", "away_goals_ht"]:
                if not pd.api.types.is_integer_dtype(df[col]):
                    issues.append(f"Expected integer, got {raw_dtype}")

            type_info[col] = {
                "dtype": inferred,
                "n_unique": n_unique,
                "null_pct": round(float(df[col].isna().mean() * 100), 2),
                "issues": issues,
            }

        issues_found = sum(1 for v in type_info.values() if v["issues"])

        return ProfileSection("Type Validation", {
            "columns": type_info,
            "n_columns_with_issues": issues_found,
        }, chart_type="table", description=f"{issues_found} columns with type issues")
