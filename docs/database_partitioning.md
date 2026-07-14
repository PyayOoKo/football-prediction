# Database Partitioning Strategy

> PostgreSQL declarative partitioning for 100M+ row tables.

---

## Overview

| Table | Partition Key | Type | Partitions | Status |
|-------|--------------|------|------------|--------|
| `matches` | `match_date` | RANGE (yearly) | 27 (2000-2027) | ✅ Applied (m002) |
| `match_statistics` | — | — | — | ❌ Not partitioned |
| `odds` | — | — | — | ❌ Not partitioned |
| `team_elo_history` | — | — | — | ❌ Not partitioned (recommended at 500M+) |
| `player_match_stats` | — | — | — | ❌ Not partitioned (recommended at 1B+) |

---

## Current: `matches` — RANGE by match_date

Applied in Migration 002. Uses PostgreSQL declarative partitioning.

### Partitioned Table Definition

```sql
CREATE TABLE matches (
    id BIGSERIAL,
    competition_id INTEGER,
    season_id INTEGER,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    stadium_id INTEGER,
    referee_id INTEGER,
    match_date DATE NOT NULL,
    round VARCHAR(32),
    is_neutral_venue BOOLEAN DEFAULT false,
    attendance INTEGER,
    home_goals INTEGER,
    away_goals INTEGER,
    result VARCHAR(4),
    duration VARCHAR(8) DEFAULT 'regular',
    status VARCHAR(16) DEFAULT 'scheduled',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (id, match_date)
) PARTITION BY RANGE (match_date);
```

> **Note:** `match_date` must be part of the primary key for RANGE partitioning.

### Yearly Partitions

```sql
-- Historical (2000-2025, created by migration 002)
CREATE TABLE matches_2000 PARTITION OF matches
    FOR VALUES FROM ('2000-01-01') TO ('2001-01-01');
-- ... one per year ...

-- Current season
CREATE TABLE matches_2026 PARTITION OF matches
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Future catch-all
CREATE TABLE matches_future PARTITION OF matches
    FOR VALUES FROM ('2027-01-01') TO ('2035-01-01');
```

### Automated Partition Maintenance

```sql
-- Created in Migration 004
CREATE OR REPLACE FUNCTION create_next_match_partition()
RETURNS void AS $$
DECLARE
    next_year text;
    partition_name text;
BEGIN
    next_year := to_char(CURRENT_DATE + INTERVAL '1 year', 'YYYY');
    partition_name := 'matches_' || next_year;
    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF matches '
            'FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            next_year || '-01-01',
            to_char(CURRENT_DATE + INTERVAL '2 years', 'YYYY') || '-01-01'
        );
    END IF;
END;
$$ LANGUAGE plpgsql;
```

---

## Recommended: `odds` — HASH by match_id

Odds data is **append-only** and queried primarily by `match_id`. Hash partitioning gives even distribution.

### Partitioned Table Definition

```sql
CREATE TABLE odds (
    id BIGSERIAL,
    match_id INTEGER NOT NULL,
    source VARCHAR(32) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    odds_home FLOAT,
    odds_draw FLOAT,
    odds_away FLOAT,
    implied_prob_home FLOAT,
    implied_prob_draw FLOAT,
    implied_prob_away FLOAT,
    margin FLOAT,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (id, match_id)
) PARTITION BY HASH (match_id);
```

### Partitions (4 partitions for ~100M rows)

```sql
CREATE TABLE odds_p0 PARTITION OF odds FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE odds_p1 PARTITION OF odds FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE odds_p2 PARTITION OF odds FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE odds_p3 PARTITION OF odds FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

---

## Recommended: `match_statistics` — HASH by match_id

1:1 with matches; all queries filtered by `match_id`.

### Partitioned Table Definition

```sql
CREATE TABLE match_statistics (
    id BIGSERIAL,
    match_id INTEGER NOT NULL UNIQUE,
    -- ... stat columns ...
    PRIMARY KEY (id, match_id)
) PARTITION BY HASH (match_id);
```

---

## Recommended at 500M+ Rows

These tables will benefit from partitioning once they exceed 500M rows:

| Table | Key | Type | Partitions | Trigger |
|-------|-----|------|------------|---------|
| `team_elo_history` | `team_id` | HASH | 4 | At 500M+ rows |
| `team_form` | `team_id` | HASH | 4 | At 500M+ rows |
| `team_xg_history` | `team_id` | HASH | 4 | At 500M+ rows |

## Recommended at 1B+ Rows

| Table | Key | Type | Partitions | Trigger |
|-------|-----|------|------------|---------|
| `player_match_stats` | `match_id` | HASH | 8-16 | At 1B+ rows |

---

## Implementation Notes

### Downtime Required
- `matches` — zero-downtime via staging table + migration script
- `odds`, `match_statistics` — requires downtime or `pg_repack`
- HASH partitions on existing tables require:
  1. Create new partitioned table
  2. Migrate data in batches
  3. Rename tables
  4. Recreate FKs

### Indexing on Partitions
- Partitioned indexes are created **per partition** (PostgreSQL 12+ propagates them automatically)
- Each partition gets its own B-tree/BRIN index
- Total index size is ~same as non-partitioned, but per-partition indexes are smaller → faster maintenance

### Autovacuum Tuning (per partition)
- Each partition can have its own autovacuum settings
- Current partition: aggressive autovacuum (scale_factor=0.01)
- Historical partitions: relaxed autovacuum (scale_factor=0.05)
