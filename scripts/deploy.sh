#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Football Prediction System — Deployment Script
#  Usage: ./scripts/deploy.sh [command]
#
#  Commands:
#    build        Build Docker images
#    push         Push images to registry
#    up           Start all services
#    down         Stop all services
#    restart      Restart all services
#    logs         View service logs
#    status       Check service status
#    backup       Backup the database
#    restore      Restore the database from latest backup
#    migrate      Run database migrations
#    setup        Full setup: build → migrate → up
#    health       Check health of all services
#    help         Show this help message
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colors ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Config ─────────────────────────────────────────────────
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
IMAGE_NAME="${DOCKER_IMAGE:-football-prediction}"
IMAGE_TAG="${DOCKER_TAG:-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# ── Helpers ────────────────────────────────────────────────
log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        log_warn "No .env file found. Creating from .env.example if available..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_ok "Created .env from .env.example — edit it with your settings."
        else
            log_warn "No .env.example found either. Continuing with defaults."
        fi
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed or too old."
        exit 1
    fi
    log_ok "Docker $(docker --version | cut -d' ' -f3 | tr -d ',')"
    log_ok "Compose $(docker compose version | cut -d' ' -f4 | tr -d ',')"
}

# ── Commands ───────────────────────────────────────────────

cmd_build() {
    log_info "Building Docker images..."
    check_env
    docker compose -f "$COMPOSE_FILE" build \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$(grep 'version =' pyproject.toml | head -1 | cut -d'"' -f2)" \
        "$@"
    log_ok "Build complete: ${FULL_IMAGE}"
}

cmd_push() {
    log_info "Pushing image to registry: ${FULL_IMAGE}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"
    log_ok "Push complete: ${FULL_IMAGE}"
}

cmd_up() {
    log_info "Starting all services..."
    check_env
    docker compose -f "$COMPOSE_FILE" up -d "$@"
    log_ok "Services started. Check status with: $0 status"
}

cmd_down() {
    log_info "Stopping all services..."
    docker compose -f "$COMPOSE_FILE" down "$@"
    log_ok "Services stopped."
}

cmd_restart() {
    cmd_down
    cmd_up
}

cmd_logs() {
    docker compose -f "$COMPOSE_FILE" logs -f "$@"
}

cmd_status() {
    echo -e "\n${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  SERVICE STATUS${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    docker compose -f "$COMPOSE_FILE" ps

    echo -e "\n${BLUE}  DISK USAGE${NC}"
    echo -e "${BLUE}───────────────────────────────────────────${NC}"
    echo "  Docker:  $(docker system df | grep 'Images' | awk '{print $3 " " $4}')"
    echo "  Data:    $(du -sh data/ 2>/dev/null | cut -f1 || echo 'N/A')"
    echo "  Models:  $(du -sh models/ 2>/dev/null | cut -f1 || echo 'N/A')"
    echo "  Logs:    $(du -sh logs/ 2>/dev/null | cut -f1 || echo 'N/A')"
}

cmd_backup() {
    log_info "Backing up database..."
    BACKUP_DIR="${BACKUP_DIR:-./data/backups}"
    mkdir -p "$BACKUP_DIR"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    BACKUP_FILE="${BACKUP_DIR}/football_db_${TIMESTAMP}.sql.gz"

    if docker compose ps --services --filter "status=running" 2>/dev/null | grep -q "db"; then
        docker compose exec -T db pg_dump \
            --no-owner --no-acl --compress=9 \
            -U "${DB_USER:-postgres}" \
            "${DB_NAME:-football_prediction}" \
            > "$BACKUP_FILE"
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log_ok "Backup created: ${BACKUP_FILE} (${SIZE})"
    elif command -v pg_dump &> /dev/null; then
        pg_dump --no-owner --no-acl --compress=9 \
            -f "$BACKUP_FILE" \
            "${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/football_prediction}"
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log_ok "Backup created: ${BACKUP_FILE} (${SIZE})"
    else
        log_error "No database running and pg_dump not found."
        exit 1
    fi

    # Clean old backups (retain last 7)
    ls -t "$BACKUP_DIR"/football_db_*.sql.gz 2>/dev/null | tail -n +8 | xargs -r rm
    log_info "Retention: keeping last 7 backups."

    # Create 'latest' symlink
    ln -sf "$BACKUP_FILE" "${BACKUP_DIR}/football_db_latest.sql.gz"
}

cmd_restore() {
    BACKUP_DIR="${BACKUP_DIR:-./data/backups}"
    BACKUP_FILE="${1:-${BACKUP_DIR}/football_db_latest.sql.gz}"

    if [ ! -f "$BACKUP_FILE" ]; then
        log_error "Backup file not found: ${BACKUP_FILE}"
        echo "  Usage: $0 restore [backup-file]"
        echo "  Available backups:"
        ls -t "${BACKUP_DIR}"/football_db_*.sql.gz 2>/dev/null | head -5 || echo "    No backups found."
        exit 1
    fi

    log_warn "About to restore database from: ${BACKUP_FILE}"
    echo "  This will OVERWRITE the current database!"
    read -p "  Are you sure? (y/N) " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled."
        exit 0
    fi

    gunzip -c "$BACKUP_FILE" | docker compose exec -T db psql \
        -U "${DB_USER:-postgres}" \
        -d "${DB_NAME:-football_prediction}"

    log_ok "Database restored from: ${BACKUP_FILE}"
}

cmd_migrate() {
    log_info "Running database migrations..."
    docker compose run --rm migrate
    log_ok "Migrations complete."
}

cmd_setup() {
    log_info "=== Full Setup ==="
    check_env
    cmd_build
    cmd_up db
    echo "  Waiting for database to be ready..."
    sleep 5
    cmd_migrate
    cmd_up
    cmd_status
    log_ok "=== Setup Complete ==="
}

cmd_health() {
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  HEALTH CHECK${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"

    # Check services
    for service in app db; do
        if docker compose ps --services --filter "status=running" 2>/dev/null | grep -q "$service"; then
            log_ok "${service}: running"
        else
            log_error "${service}: not running"
        fi
    done

    # HTTP health check
    if docker compose ps --services --filter "status=running" 2>/dev/null | grep -q "app"; then
        APP_PORT="${APP_PORT:-8000}"
        if curl -sf "http://localhost:${APP_PORT}/health" > /dev/null 2>&1; then
            log_ok "API health endpoint: OK (port ${APP_PORT})"
        else
            log_warn "API health endpoint: unreachable (port ${APP_PORT})"
        fi
    fi

    # Disk space check
    AVAIL=$(df -h . | tail -1 | awk '{print $4}')
    USAGE=$(df -h . | tail -1 | awk '{print $5}')
    if [[ "${USAGE%\%}" -gt 90 ]]; then
        log_warn "Disk usage: ${USAGE} (${AVAIL} available) — consider cleaning up"
    else
        log_ok "Disk usage: ${USAGE} (${AVAIL} available)"
    fi
}

cmd_help() {
    echo "Football Prediction System — Deployment Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build        Build Docker images"
    echo "  push         Push images to registry"
    echo "  up           Start all services"
    echo "  down         Stop all services"
    echo "  restart      Restart all services"
    echo "  logs         View service logs"
    echo "  status       Check service status"
    echo "  backup       Backup the database"
    echo "  restore      Restore the database from latest backup"
    echo "  migrate      Run database migrations"
    echo "  setup        Full setup: build → migrate → up"
    echo "  health       Check health of all services"
    echo "  help         Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  DOCKER_REGISTRY  Container registry (default: ghcr.io)"
    echo "  DOCKER_IMAGE     Image name (default: football-prediction)"
    echo "  DOCKER_TAG       Image tag (default: latest)"
    echo "  BACKUP_DIR       Backup directory (default: ./data/backups)"
    echo "  DB_USER          Database user (default: postgres)"
    echo "  DB_NAME          Database name (default: football_prediction)"
    echo "  APP_PORT         Application port (default: 8000)"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    local cmd="${1:-help}"
    shift 2>/dev/null || true

    check_docker

    case "$cmd" in
        build)    cmd_build "$@" ;;
        push)     cmd_push "$@" ;;
        up)       cmd_up "$@" ;;
        down)     cmd_down "$@" ;;
        restart)  cmd_restart "$@" ;;
        logs)     cmd_logs "$@" ;;
        status)   cmd_status "$@" ;;
        backup)   cmd_backup "$@" ;;
        restore)  cmd_restore "$@" ;;
        migrate)  cmd_migrate "$@" ;;
        setup)    cmd_setup "$@" ;;
        health)   cmd_health "$@" ;;
        help|--help|-h) cmd_help ;;
        *)
            log_error "Unknown command: ${cmd}"
            echo "  Usage: $0 [command]"
            echo "  Run '$0 help' for available commands."
            exit 1
            ;;
    esac
}

main "$@"
