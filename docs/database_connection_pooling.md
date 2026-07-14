# Database Connection Pooling

> Configuration guide for production PostgreSQL connection management.

---

## Current Configuration

```python
# src/config/settings.py
pool_size = 10          # Base connections kept open
max_overflow = 20       # Additional connections allowed during bursts
pool_pre_ping = True    # Verify connection before each use
echo = False            # Don't log all SQL
```

The engine is created in `src/database/session.py`:

```python
engine = create_engine(
    cfg.sa_url,
    pool_size=cfg.pool_size,
    max_overflow=cfg.max_overflow,
    pool_pre_ping=cfg.pool_pre_ping,
    echo=cfg.echo,
)
```

---

## Pool Size Calculation

| Factor | Value | Notes |
|--------|-------|-------|
| CPU cores | 8 | Typical production server |
| Max concurrent queries | 20 | Peak load estimate |
| Pool size formula | `2-4 × CPU cores` | General rule of thumb |
| **Recommended pool_size** | **20** | 2.5× cores |
| **Recommended max_overflow** | **10** | Keep bursts limited |
| **Total max connections** | **30** | pool_size + max_overflow |

### Per-Workload Allocation

| Workload | Connections | Notes |
|----------|-------------|-------|
| Web app (API) | 5 | Live predictions |
| Feature computation | 5 | Batch historical computation |
| Training pipeline | 3 | Model training data loading |
| Dashboard | 3 | Monitoring & reporting |
| Backtesting | 2 | Strategy evaluation |
| Admin / maintenance | 2 | Migrations, VACUUM |

---

## Advanced Settings

```python
# Production engine configuration
engine = create_engine(
    cfg.sa_url,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,        # Recycle connections after 1 hour
    pool_timeout=30,          # Wait 30s for a connection before error
    max_identifier_length=63, # PostgreSQL default
    connect_args={
        "connect_timeout": 10,     # TCP connection timeout
        "keepalives": 1,           # Enable TCP keepalive
        "keepalives_idle": 60,     # Idle time before keepalive
        "keepalives_interval": 10, # Keepalive interval
        "keepalives_count": 5,     # Max keepalive retries
        "application_name": "football_prediction",
    },
)
```

### Parameter Reference

| Parameter | Current | Recommended | Description |
|-----------|---------|-------------|-------------|
| `pool_size` | 10 | 20 | Base pool connections |
| `max_overflow` | 20 | 10 | Burst capacity |
| `pool_pre_ping` | True | True | Validates connections on checkout |
| `pool_recycle` | — | 3600 | Recycle connections (seconds) |
| `pool_timeout` | — | 30 | Wait timeout (seconds) |
| `pool_use_lifo` | — | False | LIFO vs. FIFO pool behavior |

---

## Session Lifecycle

```python
# src/database/session.py

@contextmanager
def get_session():
    """Auto-commit/rollback session context manager."""
    session = Session()
    try:
        yield session
        session.commit()      # Auto-commit on success
    except Exception:
        session.rollback()    # Auto-rollback on error
        raise
    finally:
        session.close()       # Return to pool
```

### Best Practices

1. **Always use `get_session()` context manager** — never manage sessions manually
2. **Keep transactions short** — avoid holding sessions open for long computations
3. **Use `yield_per()` for large reads** — streams results instead of loading all into memory
4. **Never use `get_all()` without a limit** — will OOM on large tables

---

## PgBouncer Setup (Multi-Process Deployments)

For deployments with multiple processes (gunicorn, celery workers), use PgBouncer for connection pooling at the database layer.

### pgbouncer.ini

```ini
[databases]
football_prediction = host=localhost port=5432 dbname=football_prediction

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = trust
pool_mode = transaction
default_pool_size = 25
max_client_conn = 100
max_db_connections = 50
server_idle_timeout = 600
query_timeout = 300
```

### Application Connection String

```
# Without PgBouncer
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/football_prediction

# With PgBouncer
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:6432/football_prediction
```

### PgBouncer Pool Modes

| Mode | Best For | Notes |
|------|----------|-------|
| `session` | Web apps with short-lived connections | One DB connection per client, held for full session |
| **`transaction`** | **Most applications** ✅ | **Connection returned to pool after each transaction** |
| `statement` | Reporting / analytics | Connection returned after each statement |

---

## PostgreSQL Connection Limits

```ini
# postgresql.conf
max_connections = 100        # Total connections to PostgreSQL
superuser_reserved_connections = 5  # Reserved for admin
```

Ensure `max_connections` in PostgreSQL exceeds the sum of all application pools:
- 30 (application pool, max)
- 25 (PgBouncer default pool, if used)
- 10 (admin / maintenance)
- **Total: 65 → set `max_connections = 100`**

---

## Monitoring Pool Usage

Monitor via the `v_slow_queries` and `v_query_stats` views:

```sql
-- Current pool usage
SELECT count(*) AS active_connections,
       state,
       wait_event_type
FROM pg_stat_activity
WHERE datname = 'football_prediction'
GROUP BY state, wait_event_type;

-- Connections by application
SELECT application_name, count(*)
FROM pg_stat_activity
WHERE datname = 'football_prediction'
GROUP BY application_name;
```

---

## Connection Timeout Configuration

Set at the database level (Migration 004):

```sql
-- PostgreSQL configuration
SET statement_timeout = '300s';           -- Max query execution time
SET idle_in_transaction_session_timeout = '10min';  -- Abandoned transactions
SET lock_timeout = '30s';                 -- Lock wait time
```

Corresponding application timeouts:

```python
connect_args = {
    "connect_timeout": 10,     # Raise on slow network
    "keepalives": 1,
    "keepalives_idle": 60,
}
```
