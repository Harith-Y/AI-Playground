#!/bin/bash

################################################################################
# AI Playground Database Restore Script
#
# This script restores PostgreSQL database and Docker volumes from backup
#
# Usage:
#   ./restore.sh [OPTIONS] BACKUP_PATH
#
# Options:
#   -e, --env ENV          Environment (development|production) [default: development]
#   -d, --database-only    Restore database only (skip volumes)
#   -v, --volumes-only     Restore volumes only (skip database)
#   -f, --force            Force restore without confirmation
#   --verbose              Verbose output
#   -h, --help             Show this help message
#
# Examples:
#   ./restore.sh ./backups/development/20260101_120000
#   ./restore.sh -e production -f ./backups/production/20260101_120000.tar.gz
#   ./restore.sh -d ./backups/development/20260101_120000  # Database only
################################################################################

set -euo pipefail

# Default configuration
ENVIRONMENT="development"
DATABASE_ONLY=false
VOLUMES_ONLY=false
FORCE=false
VERBOSE=false
BACKUP_PATH=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--database-only)
            DATABASE_ONLY=true
            shift
            ;;
        -v|--volumes-only)
            VOLUMES_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -n 20 "$0" | tail -n 18
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            BACKUP_PATH="$1"
            shift
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${NC}[DEBUG] $1${NC}"
    fi
}

# Validate backup path
validate_backup() {
    if [ -z "$BACKUP_PATH" ]; then
        log_error "Backup path is required"
        echo ""
        head -n 20 "$0" | tail -n 18
        exit 1
    fi

    if [ ! -e "$BACKUP_PATH" ]; then
        log_error "Backup path does not exist: $BACKUP_PATH"
        exit 1
    fi

    log_success "Backup path validated: $BACKUP_PATH"
}

# Extract compressed backup if needed
extract_backup() {
    if [[ "$BACKUP_PATH" == *.tar.gz ]]; then
        log_info "Extracting compressed backup..."

        local extract_dir="$(dirname "$BACKUP_PATH")/temp_restore_$$"
        mkdir -p "$extract_dir"

        tar xzf "$BACKUP_PATH" -C "$extract_dir"

        # Find the extracted directory
        BACKUP_PATH=$(find "$extract_dir" -maxdepth 1 -type d ! -path "$extract_dir" | head -n 1)

        if [ -z "$BACKUP_PATH" ]; then
            log_error "Failed to extract backup"
            exit 1
        fi

        log_success "Backup extracted to: $BACKUP_PATH"
    fi
}

# Display backup manifest
show_manifest() {
    local manifest="$BACKUP_PATH/MANIFEST.txt"

    if [ -f "$manifest" ]; then
        log_info "Backup Manifest:"
        echo ""
        cat "$manifest"
        echo ""
    else
        log_warning "No manifest file found in backup"
    fi
}

# Confirm restore operation
confirm_restore() {
    if [ "$FORCE" = true ]; then
        return 0
    fi

    echo ""
    log_warning "========================================="
    log_warning "WARNING: Restore Operation"
    log_warning "========================================="
    echo ""
    log_warning "This will:"
    if [ "$DATABASE_ONLY" = false ] && [ "$VOLUMES_ONLY" = false ]; then
        log_warning "  - REPLACE the current database"
        log_warning "  - REPLACE all Docker volumes (uploads, logs, redis)"
    elif [ "$DATABASE_ONLY" = true ]; then
        log_warning "  - REPLACE the current database"
    elif [ "$VOLUMES_ONLY" = true ]; then
        log_warning "  - REPLACE all Docker volumes (uploads, logs, redis)"
    fi
    echo ""
    log_warning "Environment: $ENVIRONMENT"
    log_warning "Backup: $BACKUP_PATH"
    echo ""
    log_warning "This operation CANNOT be undone!"
    echo ""

    read -p "Are you sure you want to continue? (yes/no): " -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Restore cancelled by user"
        exit 0
    fi

    log_success "Restore confirmed by user"
}

# Load environment variables
load_env() {
    if [ "$ENVIRONMENT" = "production" ]; then
        ENV_FILE=".env.docker.production"
    else
        ENV_FILE=".env.docker"
    fi

    if [ -f "$ENV_FILE" ]; then
        log_verbose "Loading environment from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    else
        log_warning "Environment file $ENV_FILE not found"
    fi
}

# Restore PostgreSQL database
restore_database() {
    if [ "$VOLUMES_ONLY" = true ]; then
        log_verbose "Skipping database restore (volumes-only mode)"
        return 0
    fi

    log_info "Restoring PostgreSQL database..."

    local db_file="$BACKUP_PATH/database/database.sql"

    if [ ! -f "$db_file" ]; then
        log_error "Database backup file not found: $db_file"
        return 1
    fi

    log_verbose "Database file found: $(du -h "$db_file" | cut -f1)"

    if [ -z "${DATABASE_URL:-}" ]; then
        log_error "DATABASE_URL not set in environment"
        return 1
    fi

    # For Neon or external database
    if [[ "$DATABASE_URL" == *"neon.tech"* ]] || [[ "$DATABASE_URL" == *"amazonaws.com"* ]]; then
        log_verbose "Restoring to external database (Neon/AWS)"

        # Drop and recreate schema (careful!)
        log_warning "Dropping existing data..."
        psql "$DATABASE_URL" -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;" 2>&1

        # Restore
        psql "$DATABASE_URL" < "$db_file" 2>&1

    else
        # For local Docker database
        log_verbose "Restoring to local Docker database"

        docker-compose exec -T postgres psql -U "${POSTGRES_USER:-postgres}" < "$db_file" 2>&1
    fi

    if [ $? -eq 0 ]; then
        log_success "Database restore completed"
    else
        log_error "Database restore failed"
        return 1
    fi
}

# Restore Docker volumes
restore_volumes() {
    if [ "$DATABASE_ONLY" = true ]; then
        log_verbose "Skipping volume restore (database-only mode)"
        return 0
    fi

    log_info "Restoring Docker volumes..."

    local volumes_dir="$BACKUP_PATH/volumes"

    if [ ! -d "$volumes_dir" ]; then
        log_warning "No volumes directory found in backup"
        return 0
    fi

    # Define volumes to restore
    local volumes=(
        "backend_uploads:uploads"
        "backend_logs:logs"
        "redis_data:redis"
    )

    for volume_spec in "${volumes[@]}"; do
        IFS=':' read -r volume_name volume_type <<< "$volume_spec"
        local full_volume_name="ai-playground_${volume_name}"
        local volume_backup="$volumes_dir/${volume_type}.tar.gz"

        if [ ! -f "$volume_backup" ]; then
            log_verbose "Volume backup not found: ${volume_type}, skipping"
            continue
        fi

        log_verbose "Restoring volume: $full_volume_name"

        # Create volume if it doesn't exist
        docker volume inspect "$full_volume_name" &> /dev/null || docker volume create "$full_volume_name"

        # Restore volume data
        docker run --rm \
            -v "${full_volume_name}:/data" \
            -v "$(pwd)/$volumes_dir:/backup" \
            alpine \
            sh -c "rm -rf /data/* && tar xzf /backup/${volume_type}.tar.gz -C /data" 2>&1

        if [ $? -eq 0 ]; then
            log_success "Volume restored: ${volume_type}"
        else
            log_warning "Volume restore failed: ${volume_type}"
        fi
    done
}

# Verify restore
verify_restore() {
    log_info "Verifying restore..."

    # Check database connection
    if [ "$VOLUMES_ONLY" = false ]; then
        if psql "$DATABASE_URL" -c "SELECT 1;" &> /dev/null; then
            log_success "Database connection verified"

            # Count tables
            local table_count=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ')
            log_info "Database tables: $table_count"
        else
            log_warning "Database connection verification failed"
        fi
    fi

    # Check volumes
    if [ "$DATABASE_ONLY" = false ]; then
        for volume_name in "backend_uploads" "backend_logs" "redis_data"; do
            local full_volume_name="ai-playground_${volume_name}"
            if docker volume inspect "$full_volume_name" &> /dev/null; then
                log_success "Volume exists: $volume_name"
            fi
        done
    fi
}

# Cleanup temporary files
cleanup() {
    if [[ "$BACKUP_PATH" == *"temp_restore_"* ]]; then
        log_verbose "Cleaning up temporary extraction directory"
        rm -rf "$(dirname "$BACKUP_PATH")"
    fi
}

# Main restore process
main() {
    log_info "Starting AI Playground restore process"
    log_info "Environment: $ENVIRONMENT"
    echo ""

    # Validate backup
    validate_backup

    # Extract if compressed
    extract_backup

    # Show manifest
    show_manifest

    # Confirm restore
    confirm_restore

    # Load environment
    load_env

    # Stop services before restore
    log_info "Stopping Docker services..."
    docker-compose down
    log_success "Services stopped"
    echo ""

    # Perform restore
    restore_database || log_error "Database restore had issues"
    echo ""

    restore_volumes
    echo ""

    # Start services after restore
    log_info "Starting Docker services..."
    docker-compose up -d
    log_success "Services started"
    echo ""

    # Verify restore
    verify_restore
    echo ""

    # Cleanup
    cleanup

    # Final summary
    log_success "================================"
    log_success "Restore completed successfully!"
    log_success "================================"
    log_info "Restored from: $BACKUP_PATH"
    echo ""

    log_warning "IMPORTANT: Please verify your application is working correctly"
    log_warning "Check database migrations: alembic current"
    log_warning "Check application logs: docker-compose logs -f"
}

# Trap errors and cleanup
trap cleanup EXIT

# Run main function
main
