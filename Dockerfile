# ─────────────────────────────────────────────────────────
#  Football Prediction System — Dockerfile
#  Multi-stage build for a small, secure production image.
# ─────────────────────────────────────────────────────────

# ── Stage 1: Build ─────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Install the package itself (creates CLI entry points)
COPY . .
RUN pip install --no-cache-dir --user . && \
    rm -rf /root/.cache/pip

# ── Stage 2: Runtime ──────────────────────────────────
FROM python:3.12-slim AS runtime

# Create a non-root user
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

# Install runtime system deps (libpq for psycopg2, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /usr/local

# Copy application code
COPY . .

# Ensure required directories exist
RUN mkdir -p /app/logs /app/data/raw /app/data/processed /app/data/external \
             /app/models /app/reports /app/reports/predictions_worldcup \
             /app/reports/figures /app/reports/backtest && \
    chown -R app:app /app /app/logs /app/data /app/models /app/reports

USER app

# ── Health check ────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || \
        curl -f http://localhost:8501 || \
        exit 1

# ── Labels ──────────────────────────────────────────────
LABEL org.opencontainers.image.title="Football Prediction System"
LABEL org.opencontainers.image.description="AI-powered football match outcome prediction"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.licenses="MIT"

# ── Default command ─────────────────────────────────────
# Override with: docker run --rm -it myimage football-api
# or:             docker run --rm -it myimage football-dashboard
ENTRYPOINT ["football-predict"]

# Default: show help
CMD ["--help"]
