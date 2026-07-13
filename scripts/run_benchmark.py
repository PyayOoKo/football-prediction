"""
PostgreSQL Query Benchmark — EXPLAIN ANALYZE for common football analytics queries.

Generates before-and-after performance reports showing query plans, timing,
buffer usage, and index utilization across the optimized schema.

Usage
-----
::

    # Benchmark against production database
    python scripts/run_benchmark.py

    # Output to JSON file for diffing
    python scripts/run_benchmark.py --output reports/benchmark_results.json

    # Run specific benchmark group only
    python scripts/run_benchmark.py --group matches

Requirements
------------
- PostgreSQL connection configured in .env (DATABASE_URL)
- psycopg2 or psycopg (async) installed
- EXPLAIN ANALYZE privileges on the target database
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Benchmark definitions ───────────────────────────────

QUERY_GROUPS: dict[str, list[dict[str, Any]]] = {
    "matches": [
        {
            "name": "upcoming_matches",
            "description": "Fetch upcoming 20 matches with team names",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT m.id, m.match_date,
                       ht.name AS home_team, at.name AS away_team
                FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE m.match_date >= CURRENT_DATE
                  AND m.home_goals IS NULL
                  AND m.status NOT IN ('cancelled', 'abandoned')
                ORDER BY m.match_date
                LIMIT 20
            """,
        },
        {
            "name": "team_recent_history",
            "description": "Fetch last 50 matches for a specific team",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT m.match_date, m.home_team_id, m.away_team_id,
                       m.home_goals, m.away_goals, m.result,
                       ht.name AS home_name, at.name AS away_name
                FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE (m.home_team_id = 1 OR m.away_team_id = 1)
                  AND m.result IS NOT NULL
                ORDER BY m.match_date DESC
                LIMIT 50
            """,
        },
        {
            "name": "season_league_aggregation",
            "description": "Aggregate goals by competition and season for a year",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT c.name AS competition, s.name AS season,
                       COUNT(*) AS total_matches,
                       ROUND(AVG(m.home_goals)::numeric, 3) AS avg_home_goals,
                       ROUND(AVG(m.away_goals)::numeric, 3) AS avg_away_goals,
                       ROUND(AVG(m.home_goals + m.away_goals)::numeric, 3) AS avg_total_goals
                FROM matches m
                JOIN competitions c ON c.id = m.competition_id
                JOIN seasons s ON s.id = m.season_id
                WHERE m.match_date BETWEEN '2024-01-01' AND '2024-12-31'
                  AND m.result IS NOT NULL
                GROUP BY c.name, s.name
            """,
        },
        {
            "name": "h2h_matchup",
            "description": "Head-to-head history between two teams",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT m.match_date, c.name AS competition,
                       m.home_goals, m.away_goals, m.result
                FROM matches m
                JOIN competitions c ON c.id = m.competition_id
                WHERE m.home_team_id = 1 AND m.away_team_id = 2
                  AND m.result IS NOT NULL
                ORDER BY m.match_date DESC
                LIMIT 20
            """,
        },
        {
            "name": "date_range_scan",
            "description": "Scan all matches in a 6-month date range",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT m.match_date, c.name AS competition,
                       ht.name AS home, at.name AS away,
                       m.result, m.home_goals, m.away_goals
                FROM matches m
                JOIN competitions c ON c.id = m.competition_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE m.match_date BETWEEN '2024-01-01' AND '2024-06-30'
                  AND m.result IS NOT NULL
                ORDER BY m.match_date
            """,
        },
    ],
    "odds": [
        {
            "name": "odds_by_match_source",
            "description": "All odds for a match grouped by bookmaker",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT o.source, o.timestamp,
                       o.odds_home, o.odds_draw, o.odds_away,
                       o.implied_prob_home, o.implied_prob_draw, o.implied_prob_away
                FROM odds o
                WHERE o.match_id = 1000000
                ORDER BY o.timestamp DESC
            """,
        },
        {
            "name": "odds_league_aggregation",
            "description": "Average odds per bookmaker for a competition",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT o.source,
                       COUNT(*) AS n_odds,
                       ROUND(AVG(o.odds_home)::numeric, 4) AS avg_home,
                       ROUND(AVG(o.odds_draw)::numeric, 4) AS avg_draw,
                       ROUND(AVG(o.odds_away)::numeric, 4) AS avg_away
                FROM odds o
                JOIN matches m ON m.id = o.match_id
                WHERE m.competition_id = 12
                  AND m.match_date BETWEEN '2023-01-01' AND '2023-12-31'
                GROUP BY o.source
            """,
        },
    ],
    "player_stats": [
        {
            "name": "player_history",
            "description": "Match history for a specific player with stats",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT m.match_date, pms.minutes_played, pms.goals,
                       pms.assists, pms.rating, pms.xg, pms.xa,
                       ht.name AS home_team, at.name AS away_team
                FROM player_match_stats pms
                JOIN matches m ON m.id = pms.match_id
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE pms.player_id = 5000
                ORDER BY m.match_date DESC
                LIMIT 50
            """,
        },
        {
            "name": "match_lineup",
            "description": "All player stats for a single match",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT p.full_name, pms.team_id, pms.minutes_played,
                       pms.goals, pms.assists, pms.rating, pms.position
                FROM player_match_stats pms
                JOIN players p ON p.id = pms.player_id
                WHERE pms.match_id = 1000000
                ORDER BY pms.team_id, pms.is_starter DESC, pms.position
            """,
        },
    ],
    "predictions": [
        {
            "name": "match_predictions",
            "description": "Latest predictions for a match across models",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT model_name, model_version, prob_home, prob_draw,
                       prob_away, confidence, expected_value
                FROM predictions
                WHERE match_id = 1000000
                ORDER BY created_at DESC
            """,
        },
    ],
    "betting": [
        {
            "name": "strategy_performance",
            "description": "Win rate and P&L by betting strategy",
            "query": """
                EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
                SELECT strategy,
                       COUNT(*) AS n_bets,
                       SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins,
                       ROUND(AVG(CASE WHEN won THEN profit ELSE NULL END)::numeric, 2) AS avg_won,
                       ROUND(AVG(profit)::numeric, 2) AS avg_profit,
                       ROUND(SUM(profit)::numeric, 2) AS total_profit,
                       ROUND(AVG(roi_pct)::numeric, 2) AS avg_roi
                FROM betting_results
                WHERE created_at BETWEEN '2024-01-01' AND '2024-06-30'
                GROUP BY strategy
                ORDER BY total_profit DESC
            """,
        },
    ],
}


@dataclass
class BenchmarkResult:
    """Result of a single query benchmark execution."""

    name: str
    description: str
    group: str
    success: bool
    plan_lines: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    planning_time_ms: float = 0.0
    buffers_hit: int = 0
    buffers_read: int = 0
    buffers_dirtied: int = 0
    rows_estimated: int = 0
    rows_actual: int = 0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "success": self.success,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "planning_time_ms": round(self.planning_time_ms, 2),
            "total_time_ms": round(self.execution_time_ms + self.planning_time_ms, 2),
            "buffers_hit": self.buffers_hit,
            "buffers_read": self.buffers_read,
            "rows_estimated": self.rows_estimated,
            "rows_actual": self.rows_actual,
            "error": self.error,
            "plan": "\n".join(self.plan_lines[:50]),
            "timestamp": self.timestamp,
        }


def parse_explain_analyze(
    raw_lines: list[str],
) -> tuple[float, float, int, int, int, int]:
    """Parse EXPLAIN ANALYZE output to extract key metrics.

    Returns
    -------
    tuple of (execution_time_ms, planning_time_ms, buffers_hit,
              buffers_read, rows_estimated, rows_actual)
    """
    exec_time = 0.0
    plan_time = 0.0
    bh = br = bd = 0
    rows_est = 0
    rows_act = 0

    for line in raw_lines:
        line_lower = line.lower()

        # Execution time: "Execution Time: 123.456 ms"
        if "execution time" in line_lower:
            import re
            match = re.search(r"([\d.]+)\s*ms", line)
            if match:
                exec_time = float(match.group(1))

        # Planning time
        if "planning time" in line_lower:
            import re
            match = re.search(r"([\d.]+)\s*ms", line)
            if match:
                plan_time = float(match.group(1))

        # Buffers
        if "buffers:" in line_lower or "shared hit" in line_lower:
            import re
            hit = re.search(r"shared hit=(\d+)", line_lower)
            read = re.search(r"shared read=(\d+)", line_lower)
            dirtied = re.search(r"shared dirtied=(\d+)", line_lower)
            if hit:
                bh += int(hit.group(1))
            if read:
                br += int(read.group(1))
            if dirtied:
                bd += int(dirtied.group(1))

        # Row estimates from the top-level plan node
        if "rows=" in line_lower and rows_est == 0:
            import re
            rows_match = re.search(r"rows=(\d+)", line_lower)
            actual_match = re.search(r"actual.*?rows=(\d+)", line_lower)
            if actual_match:
                rows_act = int(actual_match.group(1))
            elif rows_match:
                rows_est = int(rows_match.group(1))

    return exec_time, plan_time, bh, br, rows_est, rows_act


def run_benchmark(
    conn: Any,
    query_def: dict[str, Any],
    group_name: str,
) -> BenchmarkResult:
    """Execute a single query benchmark and return the result."""
    result = BenchmarkResult(
        name=query_def["name"],
        description=query_def.get("description", ""),
        group=group_name,
    )

    try:
        with conn.cursor() as cur:
            start = time.perf_counter()
            cur.execute(query_def["query"])
            elapsed = time.perf_counter() - start

            raw_lines: list[str] = []
            for row in cur.fetchall():
                raw_lines.append(row[0] if row else "")

            result.plan_lines = raw_lines

            # Parse metrics
            (
                exec_time, plan_time, bh, br, rows_est, rows_act
            ) = parse_explain_analyze(raw_lines)

            result.execution_time_ms = exec_time
            result.planning_time_ms = plan_time
            result.buffers_hit = bh
            result.buffers_read = br
            result.rows_estimated = rows_est
            result.rows_actual = rows_act
            result.success = True

    except Exception as exc:
        result.success = False
        result.error = str(exc)

    return result


def run_all_benchmarks(
    conn: Any,
    groups: list[str] | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """Run all benchmark queries and group results."""
    results: dict[str, list[BenchmarkResult]] = {}

    for group_name, queries in QUERY_GROUPS.items():
        if groups and group_name not in groups:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  BENCHMARK GROUP: {group_name}")
        logger.info(f"{'='*60}")

        group_results: list[BenchmarkResult] = []
        for query_def in queries:
            logger.info(f"  Running: {query_def['name']}...")
            result = run_benchmark(conn, query_def, group_name)

            status = "✅" if result.success else "❌"
            if result.success:
                logger.info(
                    f"    {status} {result.execution_time_ms:>8.2f}ms exec "
                    f"(plan: {result.planning_time_ms:.2f}ms) | "
                    f"buffers: {result.buffers_hit} hit / {result.buffers_read} read"
                )
            else:
                logger.info(f"    {status} ERROR: {result.error}")

            group_results.append(result)

        results[group_name] = group_results

    return results


def generate_report(results: dict[str, list[BenchmarkResult]]) -> str:
    """Generate a human-readable benchmark report."""
    lines = [
        "=" * 70,
        "  POSTGRESQL QUERY BENCHMARK REPORT",
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "=" * 70,
    ]

    total_queries = 0
    total_success = 0
    total_time = 0.0

    for group, group_results in sorted(results.items()):
        lines.append(f"\n{'─'*70}")
        lines.append(f"  Group: {group}")
        lines.append(f"{'─'*70}")
        lines.append(
            f"  {'Query':<30s} {'Status':<8s} {'Exec (ms)':<12s} "
            f"{'Plan (ms)':<12s} {'Buffers':<20s}"
        )
        lines.append(f"  {'─'*82}")

        for r in group_results:
            total_queries += 1
            status = "OK" if r.success else "FAIL"
            if r.success:
                total_success += 1
                total_time += r.execution_time_ms

            buffer_str = f"{r.buffers_hit} hit / {r.buffers_read} read" if r.success else "-"
            lines.append(
                f"  {r.name:<30s} {status:<8s} "
                f"{r.execution_time_ms:<12.2f} "
                f"{r.planning_time_ms:<12.2f} "
                f"{buffer_str:<20s}"
            )
            if r.error:
                lines.append(f"  {'':>30s} Error: {r.error}")

    lines.append(f"\n{'═'*70}")
    lines.append(f"  SUMMARY")
    lines.append(f"{'═'*70}")
    lines.append(f"  Total queries:    {total_queries}")
    lines.append(f"  Successful:       {total_success}")
    lines.append(f"  Failed:           {total_queries - total_success}")
    lines.append(f"  Total exec time:  {total_time:.2f}ms")
    lines.append(f"  Avg exec time:    {total_time / max(total_success, 1):.2f}ms")
    lines.append(f"{'═'*70}")

    return "\n".join(lines)


def get_db_connection(dsn: str) -> Any:
    """Create a database connection for benchmarking."""
    conn = psycopg2.connect(dsn, options="-c statement_timeout=600000")  # 10 min timeout
    conn.set_session(autocommit=True)  # Don't wrap EXPLAIN ANALYZE in transactions
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PostgreSQL query benchmark for football analytics",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for JSON results (default: print to console)",
    )
    parser.add_argument(
        "--group", "-g",
        action="append",
        choices=list(QUERY_GROUPS.keys()) + ["all"],
        default=["all"],
        help="Benchmark group(s) to run (default: all)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database DSN (default: reads from .env DATABASE_URL)",
    )
    args = parser.parse_args()

    # Get connection string
    dsn = args.dsn
    if not dsn:
        try:
            from src.config.settings import config
            dsn = config.db.sa_url
        except ImportError:
            import os
            dsn = os.environ.get("DATABASE_URL")
            if not dsn:
                logger.error(
                    "No DATABASE_URL found. Set it in .env or pass --dsn."
                )
                return 1

    groups = None if "all" in args.group else args.group

    try:
        conn = get_db_connection(dsn)
        logger.info("Connected to database. Running benchmarks...\n")

        results = run_all_benchmarks(conn, groups)
        report = generate_report(results)

        print(report)

        if args.output:
            serializable: dict[str, list[dict[str, Any]]] = {
                group: [r.to_dict() for r in r_list]
                for group, r_list in sorted(results.items())
            }
            with open(args.output, "w") as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"\nResults saved to {args.output}")

        conn.close()
        return 0

    except Exception as exc:
        logger.error(f"Benchmark failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
