"""
Database Benchmark — insert, query, and concurrent connection performance.

Measures baseline performance metrics for the production database at 100M+ row
scale. Run against a staging database to capture before/after optimisation
numbers.

Usage::

    # Standard benchmark (creates temp tables)
    python scripts/benchmark_database.py

    # Custom row counts
    python scripts/benchmark_database.py --rows 1000000 --insert-only

    # Query-only (assumes data already loaded)
    python scripts/benchmark_database.py --query-only --rows 10000000

Output: ``reports/database_benchmark_{timestamp}.json``
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.config.settings import config as app_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# ── Configuration ──────────────────────────────────────────

BENCHMARK_CONFIG = {
    "insert_batch_sizes": [100, 500, 1000, 5000, 10000],
    "concurrent_workers": [1, 2, 5, 10],
    "query_warmup_runs": 3,
    "query_measure_runs": 5,
    "default_rows": 100000,
}


@dataclass
class BenchmarkResult:
    """Container for all benchmark measurements."""

    insert: dict[str, Any] = field(default_factory=dict)
    queries: dict[str, Any] = field(default_factory=dict)
    concurrent: dict[str, Any] = field(default_factory=dict)
    system_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════


def _get_engine():
    return create_engine(
        app_config.db.sa_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )


def _rows_in_table(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.scalar(text(f"SELECT count(*) FROM {table}"))


# ═══════════════════════════════════════════════════════════
#  1. Insert Performance
# ═══════════════════════════════════════════════════════════


def _gen_match_row(match_id: int, base_date: date) -> dict:
    d = base_date + timedelta(days=match_id // 5)
    return {
        "id": match_id,
        "competition_id": random.randint(1, 50),
        "season_id": random.randint(1, 100),
        "home_team_id": random.randint(1, 500),
        "away_team_id": random.randint(1, 500),
        "match_date": d,
        "home_goals": random.randint(0, 6),
        "away_goals": random.randint(0, 6),
        "result": random.choice(["H", "D", "A"]),
        "status": "finished",
    }


def benchmark_insert(engine, rows: int) -> dict:
    """Measure insert throughput across batch sizes."""
    log.info("Insert benchmark: %d rows", rows)
    results = {}

    for batch_size in BENCHMARK_CONFIG["insert_batch_sizes"]:
        if batch_size > rows:
            continue

        # Create temp table
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS _bm_insert"))
            conn.execute(text(
                "CREATE TEMPORARY TABLE _bm_insert "
                "(id BIGINT, competition_id INT, season_id INT, "
                "home_team_id INT, away_team_id INT, match_date DATE, "
                "home_goals INT, away_goals INT, result VARCHAR(4), status VARCHAR(16))"
            ))
            conn.commit()

        base_date = date(2020, 1, 1)
        data = [_gen_match_row(i, base_date) for i in range(rows)]

        times = []
        for start in range(0, rows, batch_size):
            batch = data[start : start + batch_size]
            t0 = time.perf_counter()
            with Session(engine) as session:
                for row in batch:
                    session.execute(
                        text(
                            "INSERT INTO _bm_insert "
                            "(id, competition_id, season_id, home_team_id, away_team_id, "
                            "match_date, home_goals, away_goals, result, status) "
                            "VALUES (:id, :competition_id, :season_id, :home_team_id, "
                            ":away_team_id, :match_date, :home_goals, :away_goals, "
                            ":result, :status)"
                        ),
                        row,
                    )
                session.commit()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        total_time = sum(times)
        rows_per_sec = rows / total_time if total_time > 0 else 0
        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "total_rows": rows,
            "total_time_seconds": round(total_time, 4),
            "rows_per_second": round(rows_per_sec, 2),
            "avg_time_per_batch_seconds": round(total_time / len(times), 4),
        }

        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS _bm_insert"))
            conn.commit()

        log.info(
            "  batch_size=%d → %.0f rows/sec",
            batch_size,
            rows_per_sec,
        )

    return results


# ═══════════════════════════════════════════════════════════
#  2. Query Performance
# ═══════════════════════════════════════════════════════════


BENCHMARK_QUERIES: list[tuple[str, str, list[dict]]] = [
    (
        "point_lookup_pk",
        "Point lookup by PK",
        [{"id": 5000}, {"id": 50000}, {"id": 500000}],
    ),
    (
        "date_range_30d",
        "Last 30 days of matches",
        [{}],
    ),
    (
        "team_history",
        "Team match history (home + away)",
        [
            {"team_id": 10, "limit": 20},
            {"team_id": 100, "limit": 20},
            {"team_id": 500, "limit": 20},
        ],
    ),
    (
        "h2h_lookup",
        "Head-to-head between two teams",
        [
            {"home_id": 10, "away_id": 100},
            {"home_id": 50, "away_id": 200},
            {"home_id": 1, "away_id": 500},
        ],
    ),
    (
        "competition_season",
        "Matches in a competition season",
        [{"comp_id": 1, "season_id": 5}],
    ),
    (
        "upcoming_matches",
        "Upcoming scheduled matches",
        [{}],
    ),
    (
        "team_elo_timeline",
        "Team Elo rating timeline",
        [{"team_id": 10}],
    ),
    (
        "value_bets",
        "Value bets (predicted > implied)",
        [{"model": "xgboost", "min_ev": 0.05}],
    ),
    (
        "model_performance",
        "Model accuracy by version",
        [{"model": "logistic_regression"}],
    ),
]


def _resolve_query(sql: str, params: dict) -> tuple[str, dict]:
    """Return (sql, resolved_params) for a benchmark query."""
    if "date_range_30d" in sql:
        return (
            "SELECT count(*) FROM matches "
            "WHERE match_date >= CURRENT_DATE - INTERVAL '30 days'",
            {},
        )
    if "team_history" in sql:
        return (
            "SELECT id, match_date, home_team_id, away_team_id, "
            "home_goals, away_goals, result "
            "FROM matches "
            "WHERE home_team_id = :team_id OR away_team_id = :team_id "
            "ORDER BY match_date DESC LIMIT :limit",
            params,
        )
    if "h2h_lookup" in sql:
        return (
            "SELECT id, match_date, home_goals, away_goals, result "
            "FROM matches "
            "WHERE home_team_id = :home_id AND away_team_id = :away_id "
            "ORDER BY match_date DESC",
            params,
        )
    if "competition_season" in sql:
        return (
            "SELECT count(*) FROM matches "
            "WHERE competition_id = :comp_id AND season_id = :season_id",
            params,
        )
    if "upcoming_matches" in sql:
        return (
            "SELECT count(*) FROM matches "
            "WHERE status = 'scheduled' AND match_date >= CURRENT_DATE",
            {},
        )
    if "team_elo_timeline" in sql:
        return (
            "SELECT match_date, elo_before, elo_after, side "
            "FROM team_elo_history "
            "WHERE team_id = :team_id ORDER BY match_date",
            params,
        )
    if "value_bets" in sql:
        return (
            "SELECT count(*) FROM predictions p "
            "JOIN matches m ON p.match_id = m.id "
            "WHERE p.model_name = :model "
            "AND (CASE WHEN m.result = 'H' THEN p.prob_home "
            "WHEN m.result = 'D' THEN p.prob_draw "
            "ELSE p.prob_away END) > :min_ev",
            params,
        )
    if "model_performance" in sql:
        return (
            "SELECT model_version, count(*) as n, "
            "avg(CASE WHEN predicted_result = m.result THEN 1.0 ELSE 0.0 END) as acc "
            "FROM predictions p JOIN matches m ON p.match_id = m.id "
            "WHERE p.model_name = :model AND m.result IS NOT NULL "
            "GROUP BY model_version",
            params,
        )
    # Default: point lookup
    return (
        "SELECT * FROM matches WHERE id = :id",
        params,
    )


def benchmark_queries(engine) -> dict:
    """Measure query execution times."""
    log.info("Query benchmark")
    results = {}

    for qid, description, param_sets in BENCHMARK_QUERIES:
        sql_template = qid
        times = []
        for params in param_sets:
            sql, resolved = _resolve_query(sql_template, params)

            # Warmup
            for _ in range(BENCHMARK_CONFIG["query_warmup_runs"]):
                with engine.connect() as conn:
                    conn.execute(text(sql), resolved)

            # Measure
            for _ in range(BENCHMARK_CONFIG["query_measure_runs"]):
                t0 = time.perf_counter()
                with engine.connect() as conn:
                    conn.execute(text(sql), resolved)
                elapsed = (time.perf_counter() - t0) * 1000  # ms
                times.append(elapsed)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        results[qid] = {
            "description": description,
            "avg_ms": round(avg_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "n_param_sets": len(param_sets),
            "n_measurements": len(times),
        }
        log.info("  %s: avg=%.1fms min=%.1fms max=%.1fms", qid, avg_ms, min_ms, max_ms)

    return results


# ═══════════════════════════════════════════════════════════
#  3. Concurrent Connection Performance
# ═══════════════════════════════════════════════════════════


def _run_single_query(conn_str: str, query: str) -> float:
    """Execute a single query and return elapsed ms."""
    eng = create_engine(conn_str, pool_size=1, max_overflow=0, pool_pre_ping=True)
    t0 = time.perf_counter()
    with eng.connect() as conn:
        conn.execute(text(query))
    elapsed = (time.perf_counter() - t0) * 1000
    eng.dispose()
    return elapsed


def benchmark_concurrent(engine) -> dict:
    """Measure throughput under concurrent connections."""
    log.info("Concurrent benchmark")
    results = {}

    simple_query = "SELECT count(*) FROM matches"
    conn_str = str(engine.url)

    for n_workers in BENCHMARK_CONFIG["concurrent_workers"]:
        latencies = []
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_run_single_query, conn_str, simple_query)
                for _ in range(n_workers * 5)  # 5 queries per worker
            ]
            for f in as_completed(futures):
                latencies.append(f.result())

        total_time = time.perf_counter() - t0
        queries_per_sec = len(latencies) / total_time if total_time > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        results[f"workers_{n_workers}"] = {
            "n_workers": n_workers,
            "total_queries": len(latencies),
            "total_time_seconds": round(total_time, 4),
            "queries_per_second": round(queries_per_sec, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
        }
        log.info(
            "  workers=%d → %.1f qps, avg=%.1fms",
            n_workers,
            queries_per_sec,
            avg_latency,
        )

    return results


# ═══════════════════════════════════════════════════════════
#  4. System Info
# ═══════════════════════════════════════════════════════════


def collect_system_info(engine) -> dict:
    """Collect database version and configuration info."""
    info = {}
    with engine.connect() as conn:
        info["postgres_version"] = conn.scalar(text("SELECT version()"))
        info["server_settings"] = {
            "max_connections": conn.scalar(text("SHOW max_connections")),
            "shared_buffers": conn.scalar(text("SHOW shared_buffers")),
            "work_mem": conn.scalar(text("SHOW work_mem")),
            "maintenance_work_mem": conn.scalar(text("SHOW maintenance_work_mem")),
            "effective_cache_size": conn.scalar(text("SHOW effective_cache_size")),
        }
        # Table sizes
        tables = [
            "matches", "odds", "predictions", "team_elo_history",
            "team_form", "team_xg_history", "player_match_stats",
        ]
        info["table_sizes"] = {}
        for t in tables:
            row = conn.execute(
                text(
                    "SELECT pg_size_pretty(pg_total_relation_size(:t)) as total, "
                    "pg_size_pretty(pg_relation_size(:t)) as data, "
                    "pg_size_pretty(pg_indexes_size(:t)) as indexes, "
                    "pg_stat_get_live_tuples(:t::regclass) as rows"
                ),
                {"t": t},
            ).first()
            if row:
                info["table_sizes"][t] = {
                    "total_size": row.total,
                    "data_size": row.data,
                    "index_size": row.indexes,
                    "estimated_rows": row.rows,
                }
    return info


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Database benchmark suite")
    parser.add_argument("--rows", type=int, default=BENCHMARK_CONFIG["default_rows"])
    parser.add_argument("--insert-only", action="store_true")
    parser.add_argument("--query-only", action="store_true")
    parser.add_argument("--concurrent-only", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("  DATABASE BENCHMARK")
    print("=" * 72)

    engine = _get_engine()
    result = BenchmarkResult()

    # Verify connection
    try:
        with engine.connect() as conn:
            pg_version = conn.scalar(text("SELECT version()"))
            log.info("Connected: %s", pg_version.split(",")[0])
    except Exception as e:
        log.error("Cannot connect to database: %s", e)
        log.error("Configure DATABASE_URL or DB_* env vars in .env")
        return 1

    result.system_info = collect_system_info(engine)
    result.metadata = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": BENCHMARK_CONFIG,
        "rows": args.rows,
    }

    if not args.query_only and not args.concurrent_only:
        print("\n── Insert Performance ──")
        result.insert = benchmark_insert(engine, args.rows)

    if not args.insert_only and not args.concurrent_only:
        print("\n── Query Performance ──")
        result.queries = benchmark_queries(engine)

    if not args.insert_only and not args.query_only:
        print("\n── Concurrent Performance ──")
        result.concurrent = benchmark_concurrent(engine)

    # Save report
    report_path = REPORTS_DIR / f"database_benchmark_{TIMESTAMP}.json"
    with open(report_path, "w") as f:
        json.dump({
            "insert": result.insert,
            "queries": result.queries,
            "concurrent": result.concurrent,
            "system_info": result.system_info,
            "metadata": result.metadata,
        }, f, indent=2, default=str)

    print(f"\n  Report saved to {report_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
