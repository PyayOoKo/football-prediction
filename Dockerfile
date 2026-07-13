# ─────────────────────────────────────────────────────────
#  Football Prediction — Dockerfile
#  Multi-stage build: small, secure production image.
# ─────────────────────────────────────────────────────────

# ── Stage 1: Build ─────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system build deps (psycopg2 needs libpq, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────
FROM python:3.12-slim AS runtime

# Create a non-root user
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

# Install runtime system deps only (libpq for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /usr/local

# Copy application code
COPY . .

# Ensure logs and data directories exist
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R app:app /app

USER app

# Default command
CMD ["python", "-m", "src"]
