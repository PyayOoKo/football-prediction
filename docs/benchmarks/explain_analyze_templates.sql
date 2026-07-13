-- ============================================================================
-- EXPLAIN ANALYZE Templates & Interpretation Guide
-- ============================================================================
-- Run these to diagnose specific query performance issues.
-- Each template includes what to look for in the output.
-- ============================================================================

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 1: Check if an index is being used
-- ══════════════════════════════════════════════════════════════════════════════
-- Look for: "Seq Scan" vs "Index Scan" vs "Index Only Scan"
-- Seq Scan on matches (cost=0.00..X) = BAD (full table scan)
-- Index Only Scan = GOOD (index covers all needed columns)

EXPLAIN (ANALYZE, BUFFERS, TIMING, COSTS)
SELECT m.id, m.match_date, m.result
FROM matches m
WHERE m.home_team_id = 42
ORDER BY m.match_date DESC
LIMIT 20;

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 2: Check BRIN index effectiveness for date range
-- ══════════════════════════════════════════════════════════════════════════════
-- Look for: "Bitmap Heap Scan" with BRIN recheck
-- Low "lossy" blocks = BRIN is working well
-- High "lossy" blocks = pages_per_range too large

EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT COUNT(*), AVG(m.home_goals)
FROM matches m
WHERE m.match_date BETWEEN '2020-01-01' AND '2020-12-31';

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 3: Check partial index usage
-- ══════════════════════════════════════════════════════════════════════════════
-- Look for: "Index Scan using ix_matches_upcoming" (partial)
-- If it shows "Seq Scan on matches" instead, the WHERE clause
-- doesn't match the partial index condition exactly.

EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.id, m.match_date
FROM matches m
WHERE m.status = 'scheduled'
ORDER BY m.match_date
LIMIT 50;

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 4: Check covering index (index-only scan)
-- ══════════════════════════════════════════════════════════════════════════════
-- Look for: "Index Only Scan using ix_matches_home_covering"
-- If it shows "Heap Fetches: N" — the visibility map is stale
-- Run VACUUM matches to fix.

EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.home_team_id, m.match_date, m.away_team_id,
       m.home_goals, m.away_goals, m.result
FROM matches m
WHERE m.home_team_id = 42
ORDER BY m.match_date DESC
LIMIT 10;

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 5: Check JOIN performance
-- ══════════════════════════════════════════════════════════════════════════════
-- Look for: "Hash Join" vs "Nested Loop" vs "Merge Join"
-- Hash Join with one large table = OK (builds hash on smaller table)
-- Nested Loop on large table = BAD (should use index)
-- "Parallel Seq Scan" = table too large for Hash Join

EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.match_date, ht.name AS home, at.name AS away,
       p.prob_home, p.prob_draw, p.prob_away,
       o.odds_home, o.odds_draw, o.odds_away
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
JOIN predictions p ON p.match_id = m.id AND p.model_name = 'ensemble'
JOIN odds o ON o.match_id = m.id AND o.source = 'Pinnacle'
WHERE m.competition_id = 10 AND m.season_id = 123
  AND m.result IS NULL
ORDER BY m.match_date;

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 6: Check materialized view vs direct query
-- ══════════════════════════════════════════════════════════════════════════════
-- Compare the cost/actual time between:

-- Direct query (slow):
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.home_team_id AS team_id,
       COUNT(*) AS played,
       SUM(CASE WHEN m.result = 'H' THEN 3 WHEN m.result = 'D' THEN 1 ELSE 0 END) AS points
FROM matches m
WHERE m.competition_id = 10 AND m.season_id = 123
  AND m.result IS NOT NULL
GROUP BY m.home_team_id;

-- Materialized view (fast):
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT team_id, SUM(played) AS played, SUM(points) AS points
FROM mv_league_standings
WHERE competition_id = 10 AND season_id = 123
GROUP BY team_id
ORDER BY points DESC;

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 7: Partition pruning check
-- ══════════════════════════════════════════════════════════════════════════════
-- Look for: "Partitions: 1 of N" — good pruning
-- If it shows all partitions, the query predicate isn't partition-key aligned

EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT COUNT(*)
FROM matches
WHERE match_date >= '2025-01-01' AND match_date < '2026-01-01';

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 8: Check for sequential scans on large tables
-- ══════════════════════════════════════════════════════════════════════════════
-- Run this to identify tables with problematic seq scans

SELECT schemaname, tablename, seq_scan, seq_tup_read,
       idx_scan,
       ROUND(seq_tup_read::numeric / NULLIF(seq_scan, 0)) AS avg_rows_per_seq_scan
FROM pg_stat_user_tables
WHERE seq_scan > 100
  AND seq_tup_read > 1000000
ORDER BY seq_tup_read DESC;

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 9: Index bloat estimation
-- ══════════════════════════════════════════════════════════════════════════════

SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size((schemaname||'.'||indexname)::regclass)) AS size,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND idx_scan < 100  -- Rarely used indexes
ORDER BY pg_relation_size((schemaname||'.'||indexname)::regclass) DESC;

-- ══════════════════════════════════════════════════════════════════════════════
-- TEMPLATE 10: Table bloat and vacuum urgency
-- ══════════════════════════════════════════════════════════════════════════════

SELECT
    relname,
    n_live_tup,
    n_dead_tup,
    ROUND(n_dead_tup::numeric / NULLIF(n_live_tup, 0) * 100, 2) AS dead_pct,
    last_autovacuum,
    last_autoanalyze,
    CASE
        WHEN n_dead_tup > 1000000 THEN 'CRITICAL — VACUUM NOW'
        WHEN n_dead_tup > 100000 THEN 'WARNING — schedule VACUUM'
        WHEN n_dead_tup > 10000 THEN 'MONITOR'
        ELSE 'OK'
    END AS vacuum_status
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC;

-- ══════════════════════════════════════════════════════════════════════════════
-- Interpretation Quick Reference
-- ══════════════════════════════════════════════════════════════════════════════
--
-- Metric              Good           Warning         Critical
-- ───────────────────────────────────────────────────────────
-- Seq Scan on large   < 1% of rows   > 5% of rows    > 20% of rows
-- table
-- Heap Fetches        < 1% of rows   > 5% of rows    > 20% of rows
-- (index-only scan)
-- Actual Time vs      < 2×           < 10×           > 10×
-- Estimated Time
-- Shared Hit Ratio    > 99%          > 95%           < 95%
-- (buffer cache)
-- Dead tuple %        < 5%           < 20%           > 20%
-- Rows Removed by     < 10%          < 50%           > 50%
-- Filter (WHERE)
-- Partition Pruning   1 of N (max)   N/2 of N        N of N (none)
-- Workers Planned     > 0            = 0 (tables     = 0 on tables
-- (parallel query)                   < 8GB)          > 8GB)
