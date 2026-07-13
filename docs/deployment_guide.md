# Deployment Guide

> Production deployment guide for the football prediction system.

## Docker Deployment

### Prerequisites
- Docker 24+
- Docker Compose 2.20+

### Quick Start

```bash
# Clone and enter the project
git clone https://github.com/yourusername/football-prediction.git
cd football-prediction

# Configure environment
cp .env.example .env
# Edit .env with your production settings

# Build and start all services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Docker Compose Services

```yaml
services:
  app:             # Main application (dashboard + API)
  db:              # PostgreSQL 16
  redis:           # Cache (optional)
  scheduler:       # Cron scheduled tasks
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')"

# Default command
CMD ["streamlit", "run", "src/app/dashboard.py", "--server.port=8501"]
```

## Production Database

### PostgreSQL Configuration

```ini
# postgresql.conf (recommended for 100M+ rows)
shared_buffers = '4GB'          # 25% of RAM
effective_cache_size = '12GB'   # 75% of RAM
work_mem = '64MB'                # Per-operation sort memory
maintenance_work_mem = '1GB'    # For VACUUM, CREATE INDEX
random_page_cost = 1.1          # SSD-optimized
effective_io_concurrency = 200  # SSD-optimized
wal_buffers = '64MB'
max_wal_size = '4GB'
min_wal_size = '1GB'
autovacuum_vacuum_scale_factor = 0.01  # More frequent vacuum
autovacuum_analyze_scale_factor = 0.005
autovacuum_naptime = '30s'
```

### Connection Pooling with PgBouncer

```ini
# pgbouncer.ini
[databases]
football_prediction = host=db port=5432 dbname=football_prediction

[pgbouncer]
pool_mode = transaction
max_client_conn = 200
default_pool_size = 20
reserve_pool_size = 10
prepared_statement_cache_size = 0  # Required for SQLAlchemy
```

### Environment Variables

```bash
# Production .env
APP_ENV=production
APP_DEBUG=false
SECRET_KEY=<generate-a-random-secret-key>

DATABASE_URL=postgresql+psycopg2://user:password@db:5432/football_prediction

DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_PRE_PING=true

LOG_LEVEL=INFO
LOG_FILE=true

THE_ODDS_API_KEY=<your-api-key>
FOOTBALL_DATA_API_KEY=<your-api-key>
```

## Migration Procedure

```bash
# 1. Back up the database
pg_dump --no-owner --no-acl --compress=9 \
    -f backup_$(date +%Y%m%d).sql.gz \
    postgresql://user:password@localhost:5432/football_prediction

# 2. Run pending migrations
alembic upgrade head

# 3. Verify
alembic current
```

## Monitoring

### Health Checks

The Docker container exposes a health check endpoint at `/_stcore/health`.

```bash
curl http://localhost:8501/_stcore/health
# {"status": "healthy"}
```

### Logging

Logs are written to `logs/football_prediction.log` with daily rotation (30-day retention).

```bash
# View live logs
docker-compose logs -f app

# Tail the application log
tail -f logs/football_prediction.log
```

### System Metrics

The monitoring module collects:
- CPU usage
- Memory usage
- Disk usage
- Database size
- Cache hit rate
- Data quality metrics

```python
from src.monitoring import SystemCollector

collector = SystemCollector(data_dir="data", db_path="data/football.db")
metrics = collector.collect()
print(f"CPU: {metrics.cpu_percent}% | RAM: {metrics.memory_used_mb:.0f}MB")
```

## Security Checklist

- [ ] `SECRET_KEY` set to a long random value
- [ ] `APP_DEBUG=false`
- [ ] PostgreSQL password set (not default `postgres`)
- [ ] Docker containers not running as root
- [ ] API keys stored in `.env`, not committed
- [ ] Database port not exposed publicly
- [ ] CORS configured if REST API is exposed
- [ ] Rate limiting configured for API endpoints
- [ ] HTTPS/SSL configured (use reverse proxy)

## Scaling

### Vertical Scaling
- **RAM:** 8GB minimum, 16GB+ recommended
- **CPU:** 4+ cores
- **Disk:** SSD, 50GB+ for data + models

### Horizontal Scaling
- Read replicas for dashboard queries
- Separate scheduler worker process
- Redis cache for feature values
- Partitioned PostgreSQL tables (50M+ rows)

### Performance Tuning

| Parameter | 100K Rows | 10M Rows | 100M Rows |
|---|---|---|---|
| `DB_POOL_SIZE` | 5 | 10 | 20 |
| `DB_MAX_OVERFLOW` | 10 | 20 | 40 |
| `--batch-size` | 1000 | 5000 | 10000 |
| `--parallel` | false | true (4 workers) | true (8 workers) |
| Chunk size (Parquet) | 100K | 500K | 1M |

## Troubleshooting Production Issues

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose ps db

# Test connection
docker-compose exec db pg_isready -U postgres

# View connection count
docker-compose exec db psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Reduce batch sizes
python run_pipeline.py --batch-size 1000

# Increase swap
free -h
```

### Slow Queries

```bash
# Enable slow query logging
docker-compose exec db psql -U postgres -c "
SET log_min_duration_statement = 1000;  # Log queries > 1s
"

# View slow queries
docker-compose logs db | grep "duration:"
```

## Backup and Restore

### Automatic Backups

The scheduler runs daily backups automatically:

```bash
# Backup location
ls -la data/backups/

# Retention: 7 days (configurable)
```

### Manual Backup

```bash
# Full database backup
python -m src.scheduler.cli run --tasks backup_database

# Verify backup
gzip -t data/backups/football_db_*.sql.gz
```

### Restore

```bash
# Restore from latest backup
gunzip -c data/backups/football_db_latest.sql.gz | \
    docker-compose exec -T db psql -U postgres football_prediction
```
