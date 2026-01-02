#!/bin/bash
#
# Database Migration Script for Docker/Production
#
# Usage:
#   ./migrate.sh                    # Run migrations
#   ./migrate.sh status             # Show migration status
#   ./migrate.sh downgrade -1       # Rollback one migration
#   ./migrate.sh history            # Show migration history
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration from environment or defaults
MIGRATION_WAIT_TIMEOUT="${MIGRATION_WAIT_TIMEOUT:-60}"
MIGRATION_FAIL_ON_ERROR="${MIGRATION_FAIL_ON_ERROR:-true}"
MIGRATION_DRY_RUN="${MIGRATION_DRY_RUN:-false}"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Change to backend directory
cd "$BACKEND_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    log_info "Activating virtual environment..."
    source venv/bin/activate
fi

# Default command is 'auto' (run migrations)
COMMAND="${1:-auto}"

case "$COMMAND" in
    auto|upgrade)
        log_info "Running database migrations..."
        python scripts/run_migrations.py
        ;;
    
    status)
        log_info "Checking migration status..."
        python -m app.db.migration_manager status
        ;;
    
    downgrade)
        REVISION="${2:--1}"
        log_warn "Downgrading database to revision: $REVISION"
        python -m app.db.migration_manager downgrade --revision "$REVISION"
        ;;
    
    history)
        log_info "Showing migration history..."
        python -m app.db.migration_manager history
        ;;
    
    create)
        if [ -z "$2" ]; then
            log_error "Please provide a migration message"
            echo "Usage: $0 create 'migration message'"
            exit 1
        fi
        MESSAGE="$2"
        log_info "Creating new migration: $MESSAGE"
        cd "$BACKEND_DIR"
        alembic revision --autogenerate -m "$MESSAGE"
        ;;
    
    help|--help|-h)
        echo "Database Migration Script"
        echo ""
        echo "Usage:"
        echo "  $0                      Run migrations (default)"
        echo "  $0 status              Show migration status"
        echo "  $0 upgrade             Run migrations (alias for auto)"
        echo "  $0 downgrade [rev]     Rollback migrations (default: -1)"
        echo "  $0 history             Show migration history"
        echo "  $0 create 'message'    Create new migration"
        echo "  $0 help                Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  MIGRATION_WAIT_TIMEOUT       Timeout for database (default: 60s)"
        echo "  MIGRATION_FAIL_ON_ERROR      Fail on error (default: true)"
        echo "  MIGRATION_DRY_RUN            Dry run mode (default: false)"
        echo ""
        ;;
    
    *)
        log_error "Unknown command: $COMMAND"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
