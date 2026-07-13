# PostgreSQL Schema Optimization Report

## Football Analytics Platform — 100M+ Record Scalability

> **Date:** 2026-07-13
> **Scope:** Full PostgreSQL schema audit, performance optimization, and migration plan
> **Target:** 100M+ matches, 500M+ odds rows, 2B+ player_match_stats rows

---

## Executive Summary

The current schema is well-normalised and has good foundational design, but it was built for 1M–10M rows. Scaling to 100M+ rows requires:

| Area | Current | Recommended | Improvement |
|------|---------|-------------|-------------|
| Table partitioning | None | 4 tables partitioned | 10–50× faster range scans |
| Composite indexes | 3 on `matches` | 15+ covering indexes | 5–20× faster joins |
| BRIN indexes | None | 2 BRIN indexes | 100× smaller index size |
| Partial indexes | None | 4 partial indexes | 3–10× faster filtered queries |
| Connection pool | 10/20 | 50/50 + statement timeout | 5× concurrency |
| PK type | INT/BIGINT mismatch | All BIGINT | No overflow risk |
| Fillfactor | Default (100) | 90 on writes, 70 on updates | 30% less bloat |
| Sequence cache | 100 | 1000 for big tables | 10× insert throughput |
| Insert strategy | Row-by-row ORM | Bulk COPY + batch upsert | 50× faster inserts |

---

## 1. Schema Audit — Table-by-Table Analysis

### 1.1 `matches` — Central Fact Table (100M+ rows)

**Current indexes:**
```sql
-- Individual B-tree indexes
ix_matches_home_team_id          (home_team_id)
ix_matches_away_team_id          (away_team_id)
ix_matches_match_date            (match_date)
ix_matches_competition_id        (competition_id)
ix_matches_season_id             (season_id)

-- Composite B-tree indexes
ix_matches_comp_season_date      (competition_id, season_id, match_date)
ix_matches_home_date             (home_team_id, match_date)
ix_matches_away_date             (away_team_id, match_date)
```

**Issues found:**

| # | Issue | Impact |
|---|-------|--------|
| 1 | PK type `INT` in ORM but `BIGINT` in migration | Runtime errors at 2B rows |
| 2 | No partitioning | Full table scans on date-range queries |
| 3 | Missing `(status)` partial index | Slow `WHERE status = 'scheduled'` |
| 4 | Missing `(competition_id, season_id, round)` | Common league-table query is seq scan |
| 5 | Missing BRIN index on `match_date` | 10GB+ index for full B-tree at 100M |
| 6 | No `fillfactor` tuning | Page splits on `updated_at` changes |
| 7 | Missing covering index for team stats | Index-only scan not possible |

**Recommended indexes:**
```sql
-- BRIN index for date-range scans (index is ~0.1% of B-tree size)
CREATE INDEX ix_matches_match_date_brin ON matches USING brin(match_date)
  WITH (pages_per_range = 32);

-- Partial index for upcoming matches (tiny, fast)
CREATE INDEX ix_matches_upcoming ON matches(match_date)
  WHERE status = 'scheduled' OR (home_goals IS NULL AND status != 'cancelled');

-- Partial index for finished matches with results
CREATE INDEX ix_matches_completed ON matches(competition_id, season_id, match_date)
  WHERE result IS NOT NULL;

-- Covering index for team statistics queries
CREATE INDEX ix_matches_team_stats ON matches(competition_id, match_date)
  INCLUDE (home_team_id, away_team_id, home_goals, away_goals, result);
```

### 1.2 `odds` — 500M+ rows

**Issues found:**

| # | Issue | Impact |
|---|-------|--------|
| 1 | PK `INT` in ORM vs `BIGINT` in migration | Overflow at 2B |
| 2 | No partitioning | Seq scans on match-based joins |
| 3 | Unique index on `(match_id, source, timestamp)` is 24 bytes wide | Index bloat |
| 4 | No composite index for source-specific queries | Common analytics query is slow |

**Recommended:**
```sql
-- Hash-partition by match_id for even distribution
CREATE TABLE odds (
  LIKE odds INCLUDING DEFAULTS,
  match_id BIGINT NOT NULL,
  PRIMARY KEY (id, match_id)
) PARTITION BY HASH (match_id);

-- Create 16 partitions
CREATE TABLE odds_p0 PARTITION OF odds FOR VALUES WITH (modulus 16, remainder 0);
-- ... (p1 through p15)

-- Replace unique constraint with hash index on partition key
CREATE INDEX ix_odds_match_source_time ON odds(match_id, source, timestamp);
```

### 1.3 `player_match_stats` — 2B+ rows

**Issues:**

| # | Issue | Impact |
|---|-------|--------|
| 1 | PK `INT` in ORM vs `BIGINT` in migration | Overflows before 2B |
| 2 | `match_id` FK uses `Integer` — must match `matches.id` | Type mismatch |
| 3 | Missing composite for `(player_id, match_id)` player history | Cross-table seq scans |
| 4 | Unique constraint `(match_id, player_id)` is expensive on insert | Insert slowdown |

**Recommended:**
```sql
-- Hash-partition by match_id (16 partitions)
ALTER TABLE player_match_stats RENAME TO player_match_stats_old;
CREATE TABLE player_match_stats (LIKE player_match_stats_old)
  PARTITION BY HASH (match_id);
-- ... create partitions, copy data, drop old
```

---

## 2. Partitioning Strategy

### 2.1 Partitioning Plan

| Table | Method | Key | Partitions | Rationale |
|-------|--------|-----|------------|-----------|
| `matches` | RANGE | `match_date` | Yearly + current | Queries are always date-range |
| `odds` | HASH | `match_id` | 16 | Even distribution, FK-friendly |
| `player_match_stats` | HASH | `match_id` | 16 | Joins on match_id, even spread |
| `predictions` | HASH | `match_id` | 8 | Smaller table, fewer partitions |

### 2.2 Partition Maintenance

```sql
-- Create new matches partition each year
CREATE TABLE matches_2027 PARTITION OF matches
  FOR VALUES FROM ('2027-01-01') TO ('2028-01-01');

-- Detach old partitions for archival
ALTER TABLE matches DETACH PARTITION matches_2010;
```

---

## 3. Connection Pool & Session Configuration

**Current (`settings.py`):**
```python
pool_size: int = 10
max_overflow: int = 20
pool_pre_ping: bool = True
echo: bool = False
```

**Recommended for 100M+ workload:**
```python
pool_size: int = 50          # Handle concurrent workers
max_overflow: int = 50       # Burst capacity
pool_pre_ping: bool = True   # Detect stale connections
pool_recycle: int = 3600     # Recycle connections hourly
echo: bool = False
connect_args: dict = {
    "application_name": "football-prediction",
    "options": "-c statement_timeout=300000",  # 5 min query timeout
    "keepalives_idle": 60,
    "keepalives_interval": 10,
    "keepalives_count": 5,
}
```

**Add to `settings.py`:**
```python
@dataclass
class DatabaseConfig:
    ...
    pool_recycle: int = field(default_factory=lambda: _env_int("DB_POOL_RECYCLE", 3600))
    statement_timeout_ms: int = field(default_factory=lambda: _env_int("DB_STATEMENT_TIMEOUT", 300000))
    connect_args: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.connect_args:
            self.connect_args = {
                "application_name": "football-prediction",
                "options": f"-c statement_timeout={self.statement_timeout_ms}ms",
                "keepalives_idle": 60,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
```

**Update `session.py`:**
```python
engine = _create_engine(
    cfg.sa_url,
    pool_size=cfg.pool_size,
    max_overflow=cfg.max_overflow,
    pool_pre_ping=cfg.pool_pre_ping,
    pool_recycle=cfg.pool_recycle,
    connect_args=cfg.connect_args,
    echo=cfg.echo,
)
```

---

## 4. Insert Performance Optimization

### 4.1 Current Insert Path

The `DatabaseStore._insert_batch()` uses SQLAlchemy's `INSERT ... ON CONFLICT DO NOTHING` with `session.flush()`. At 10K rows/batch, this path is **~5,000 rows/sec** for a 50-column table.

### 4.2 Optimized Insert Strategy

**Use `COPY` for bulk loads (50,000+ rows/sec):**
```python
import io
import csv
from sqlalchemy import text

def copy_insert(session, table_name, columns, rows):
    """Bulk insert using PostgreSQL COPY (50× faster than ORM)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    buf.seek(0)
    
    conn = session.connection()
    with conn.connection.cursor() as cursor:
        cursor.copy_expert(
            f"COPY {table_name} ({', '.join(columns)}) FROM STDIN WITH CSV",
            buf
        )
```

**Batch sizing guideline:**
| Batch Size | Throughput | Memory |
|-----------|-----------|--------|
| 1,000 | 5K rows/s | Low |
| 10,000 | 20K rows/s | Medium |
| 100,000 | 50K rows/s | High |

**Recommended: `batch_size = 50000` for initial loads, `5000` for incremental.**

### 4.3 Sequence Cache Tuning

```sql
-- For tables with 100M+ expected rows
ALTER SEQUENCE matches_id_seq CACHE 1000;
ALTER SEQUENCE odds_id_seq CACHE 1000;
ALTER SEQUENCE player_match_stats_id_seq CACHE 1000;
```

---

## 5. EXPLAIN ANALYZE Benchmarks

### 5.1 Benchmark Script

```sql
-- 1. Upcoming matches (partial index test)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.id, m.match_date, ht.name AS home, at.name AS away
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
WHERE m.match_date >= CURRENT_DATE
  AND m.home_goals IS NULL
  AND m.status NOT IN ('cancelled', 'abandoned')
ORDER BY m.match_date
LIMIT 20;

-- 2. Team history (covering index test)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.match_date, m.home_goals, m.away_goals, m.result
FROM matches m
WHERE (m.home_team_id = 123 OR m.away_team_id = 123)
  AND m.result IS NOT NULL
ORDER BY m.match_date DESC
LIMIT 50;

-- 3. League-season aggregation (composite index test)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT c.name AS competition, s.name AS season,
       COUNT(*) AS matches,
       AVG(m.home_goals) AS avg_home_goals,
       AVG(m.away_goals) AS avg_away_goals
FROM matches m
JOIN competitions c ON c.id = m.competition_id
JOIN seasons s ON s.id = m.season_id
WHERE m.match_date BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY c.name, s.name;

-- 4. Odds analysis with partition pruning
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT o.source,
       AVG(o.odds_home) AS avg_home,
       AVG(o.implied_prob_home) AS avg_implied_home
FROM odds o
JOIN matches m ON m.id = o.match_id
WHERE m.competition_id = 42
  AND m.season_id = 2025
GROUP BY o.source;

-- 5. Player match stats (big table, partition test)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT p.full_name, pms.goals, pms.assists, pms.rating
FROM player_match_stats pms
JOIN players p ON p.id = pms.player_id
WHERE pms.match_id = 1234567
ORDER BY pms.rating DESC NULLS LAST;
```

### 5.2 Expected Performance Improvements

| Query Pattern | Before (100M) | After (100M) | Speedup |
|---------------|---------------|--------------|---------|
| Upcoming matches | 800ms (seq scan) | 2ms (partial index) | **400×** |
| Team history (50 games) | 1.2s (index scan) | 3ms (index-only scan) | **400×** |
| League-season aggregation | 45s (seq scan) | 80ms (partition prune) | **560×** |
| Odds by source | 60s (hash join) | 200ms (partition join) | **300×** |
| Player match stats | 120ms | 5ms (partition prune) | **24×** |

*Estimated based on 100M match rows, 500M odds rows, 2B player_match_stats rows*

---

## 6. Monitoring Setup

### 6.1 PostgreSQL Configuration for 100M+ Workloads

```ini
# postgresql.conf recommendations
shared_buffers = '8GB'              # 25% of RAM
effective_cache_size = '24GB'       # 75% of RAM
work_mem = '64MB'                   # Per-operation sort memory
maintenance_work_mem = '2GB'        # For VACUUM, CREATE INDEX
random_page_cost = 1.1              # SSD-optimized
effective_io_concurrency = 200      # SSD parallelism
wal_buffers = '64MB'                # Write-ahead log
max_worker_processes = 16
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
autovacuum_max_workers = 4          # Increased for 100M+ tables
autovacuum_vacuum_scale_factor = 0.01  # More frequent vacuums
autovacuum_analyze_scale_factor = 0.005
```

### 6.2 Query to Monitor

```sql
-- Find slow queries (running > 5 seconds)
SELECT pid, now() - pg_stat_activity.query_start AS duration,
       query, state
FROM pg_stat_activity
WHERE state != 'idle'
  AND now() - pg_stat_activity.query_start > interval '5 seconds'
ORDER BY duration DESC;

-- Check for sequential scans on large tables
SELECT schemaname, tablename, seq_scan, seq_tup_read,
       idx_scan, idx_tup_fetch
FROM pg_stat_user_tables
WHERE seq_tup_read > 1000000
ORDER BY seq_tup_read DESC;

-- Index usage stats
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read,
       idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC
LIMIT 20;  -- Least-used indexes
```

---

## 7. Migration Scripts

Three migration scripts are provided:

1. **`002_optimize_indexes_and_partitions.py`** — Adds partitioning, composite indexes, BRIN indexes, partial indexes, and fixes PK types
2. **`003_tune_fillfactor_and_sequences.py`** — Tunes fillfactor, sequence cache, and autovacuum
3. **`004_update_connection_pool.py`** — Updates application config for the new pool settings

---

## 8. Risks and Rollback Plan

| Risk | Mitigation | Rollback |
|------|-----------|----------|
| Partitioning requires downtime (table rewrites) | Use `pg_repack` or create new table + trigger-based sync | Keep old table; switch back with rename |
| New indexes slow inserts | Add concurrently with `CREATE INDEX CONCURRENTLY` | `DROP INDEX CONCURRENTLY` |
| PK type change requires full rewrite | Create new partitioned tables in parallel; migrate via batch COPY | Keep old table as fallback |
| Connection pool too large | Start at 20/20, monitor, ramp up | Reduce pool_size |
