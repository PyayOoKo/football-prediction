# PgBouncer Connection Pooling Guide

> **Purpose:** Reduce PostgreSQL connection overhead in production.
> **Target:** 100M+ row database with multi-service access.

---

## Why PgBouncer?

PostgreSQL uses one OS process per connection. At 100M+ rows, the database
needs its memory for caching data (shared_buffers), not for managing 100+
connection processes. PgBouncer multiplexes many client connections through
a small pool of database connections.

### Connection Overhead Comparison

| Service | Direct Connections | Via PgBouncer | Memory Saved |
|---------|-------------------|---------------|--------------|
| Streamlit Dashboard | 10 | 2 | ~72 MB |
| Scheduler | 5 | 1 | ~36 MB |
| CLI scripts | 20 (transient) | 2 | ~144 MB |
| REST API | 50 | 8 | ~360 MB |
| **Total** | **85** | **13** | **~612 MB** |

---

## Installation

```bash
# Ubuntu/Debian
sudo apt-get install pgbouncer

# macOS
brew install pgbouncer

# Verify
pgbouncer --version  # Should be >= 1.19
```

---

## Configuration

### `pgbouncer.ini`

```ini
[databases]
; Map application database to a PgBouncer database entry
football_prediction = host=localhost port=5432 dbname=football_prediction

; Optional: separate pool for reporting queries (long-running)
football_prediction_reports = host=localhost port=5432 dbname=football_prediction

[pgbouncer]
; ── Network ──────────────────────────────────────────
listen_addr = 127.0.0.1
listen_port = 6432
; For Docker: listen_addr = 0.0.0.0

; ── Authentication ───────────────────────────────────
auth_type = trust
; For production: auth_type = md5
; auth_file = /etc/pgbouncer/userlist.txt

; ── Pool Configuration ───────────────────────────────
; Transaction pooling: connection returned to pool after each COMMIT/ROLLBACK
; Recommended for web applications and most ORM workloads
pool_mode = transaction

; Per-database pool limits
default_pool_size = 20
; 20 connections to PostgreSQL is enough for 100+ concurrent clients
; Each connection can handle ~5-10 concurrent transactions

; Reserve connections for peak loads
reserve_pool_size = 5
reserve_pool_timeout = 5

; Maximum client connections
max_client_conn = 200

; Maximum database connections across all pools
max_db_connections = 50

; ── Timeouts ─────────────────────────────────────────
server_idle_timeout = 600
; Close backend connections idle for 10 minutes

client_idle_timeout = 0
; No client idle timeout (clients can stay connected)

query_timeout = 300
; Matches PostgreSQL statement_timeout (5 minutes)

query_wait_timeout = 30
; Client waits max 30s for a free connection

; ── Logging ──────────────────────────────────────────
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
stats_period = 60
```

### Databse-Specific Pool Overrides

```ini
; In [databases] section:
football_prediction = host=localhost port=5432 dbname=football_prediction \
    pool_size=20 \
    max_db_connections=50 \
    query_timeout=300

; For the ETL pipeline (batch inserts, fewer but larger transactions):
football_prediction_etl = host=localhost port=5432 dbname=football_prediction \
    pool_size=5 \
    pool_mode=session  ; Session mode for long-running bulk inserts
```

---

## Application Integration

### SQLAlchemy + PgBouncer

```python
# src/config/settings.py — PgBouncer connection
# Instead of connecting directly to PostgreSQL:
#   DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/football_prediction
#
# Connect through PgBouncer:
#   DATABASE_URL=postgresql+psycopg2://user:pass@localhost:6432/football_prediction

# Important: Disable pgbouncer-incompatible features
@dataclass
class DatabaseConfig:
    # ...existing fields...

    @property
    def sa_url(self) -> str:
        """Return the SQLAlchemy URL.
        
        When using PgBouncer in transaction mode, PREPARE statements
        are not supported. Disable them with:
            ?prepared_statement_cache_size=0
        """
        url = self.url
        if "pgbouncer" in url or "6432" in url:
            if "?" not in url:
                url += "?"
            url += "&prepared_statement_cache_size=0&keepalives=1"
        return url
```

### Session Pool Tuning for PgBouncer

```python
# src/database/session.py — when using PgBouncer
# PgBouncer manages the connection pool. Use a smaller SQLAlchemy pool
# to match PgBouncer's max_db_connections.

def create_engine_from_config() -> Engine:
    cfg = config.db
    engine = _create_engine(
        cfg.sa_url,
        pool_size=5,              # Smaller pool — PgBouncer handles queueing
        max_overflow=2,           # Minimal overflow (PgBouncer reserve handles spikes)
        pool_pre_ping=True,       # Still needed for PgBouncer
        pool_use_lifo=True,       # LIFO better for PgBouncer (reuse hot connections)
        pool_recycle=3600,        # Recycle connections hourly
        echo=cfg.echo,
        # Disable connection pool for PgBouncer:
        # poolclass=NullPool,     # Uncomment if PgBouncer is the only pooler
    )
    return engine
```

---

## Monitoring PgBouncer

### Show Stats

```bash
# Connect to PgBouncer's admin console
psql -h localhost -p 6432 pgbouncer -U pgbouncer

# Inside the console:
SHOW STATS;          # Traffic statistics
SHOW POOLS;          # Pool usage
SHOW CLIENTS;        # Active clients
SHOW SERVERS;        # Database connections
SHOW DATABASES;      # Database-level stats
SHOW FDS;            # File descriptors
SHOW CONFIG;         # Current configuration
SHOW VERSION;        # PgBouncer version
```

### Key Metrics to Watch

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| pool utilization | < 70% | > 80% | > 95% |
| avg wait time | < 1ms | < 10ms | > 50ms |
| server connection rate | < 10/s | < 50/s | > 100/s |
| client connection rate | < 100/s | < 500/s | > 1000/s |

### Prometheus Integration

```bash
# pgbouncer_exporter (Prometheus exporter)
pgbouncer_exporter --pgBouncer.connection-string="postgres://pgbouncer:@localhost:6432/pgbouncer?sslmode=disable"
```

---

## Migration from Direct to PgBouncer

### Step 1: Add PgBouncer alongside existing connections

```bash
# Start PgBouncer on port 6432 while PostgreSQL runs on 5432
pgbouncer -d /etc/pgbouncer/pgbouncer.ini
```

### Step 2: Update application config

```bash
# Change DATABASE_URL in .env:
# Old:
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/football_prediction
# New:
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:6432/football_prediction
```

### Step 3: Deploy gradually

1. Migrate non-critical services first (CLI, scheduler)
2. Monitor PgBouncer stats for a day
3. Migrate the REST API
4. Migrate the dashboard last

### Step 4: Rollback

```bash
# Return to direct connection by changing DATABASE_URL back to 5432
# PgBouncer has zero data persistence — no migration needed
```

---

## Performance Impact

| Metric | Direct (5432) | PgBouncer (6432) | Improvement |
|--------|---------------|-------------------|-------------|
| Connection time | ~5ms | ~50μs | **100×** |
| Max concurrent clients | ~50 (limited by PG) | ~200 | **4×** |
| PostgreSQL memory | ~50 MB for connections | ~5 MB | **10×** |
| Connection storm recovery | ~30s | ~100ms | **300×** |
