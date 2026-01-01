#!/bin/bash

################################################################################
# AI Playground Database Backup Script
#
# This script creates compressed backups of the PostgreSQL database
# and associated data volumes (uploads, logs)
#
# Usage:
#   ./backup.sh [OPTIONS]
#
# Options:
#   -e, --env ENV          Environment (development|production) [default: development]
#   -d, --dir DIR          Backup directory [default: ./backups]
#   -r, --retention DAYS   Retention period in days [default: 30]
#   -c, --compress         Compress backup (gzip)
#   -v, --verbose          Verbose output
#   -h, --help             Show this help message
#
# Examples:
#   ./backup.sh                              # Basic backup
#   ./backup.sh -e production -r 90          # Production backup, 90 day retention
#   ./backup.sh -d /mnt/backups -c           # Custom dir with compression
################################################################################

set -euo pipefail

# Default configuration
ENVIRONMENT="development"
BACKUP_DIR="./backups"
RETENTION_DAYS=30
COMPRESS=false
VERBOSE=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
        -d|--dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -r|--retention)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        -c|--compress)
            COMPRESS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -n 23 "$0" | tail -n 21
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
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

# Load environment variables based on environment
load_env() {
    if [ "$ENVIRONMENT" = "production" ]; then
        ENV_FILE=".env.docker.production"
    else
        ENV_FILE=".env.docker"
    fi

    if [ -f "$ENV_FILE" ]; then
        log_verbose "Loading environment from $ENV_FILE"
        # Export variables without overriding existing environment
        set -a
        source "$ENV_FILE"
        set +a
    else
        log_warning "Environment file $ENV_FILE not found, using defaults"
    fi
}

# Create backup directory structure
setup_backup_directory() {
    local backup_path="$BACKUP_DIR/$ENVIRONMENT/$TIMESTAMP"

    log_verbose "Creating backup directory: $backup_path"
    mkdir -p "$backup_path/database"
    mkdir -p "$backup_path/volumes"
    mkdir -p "$backup_path/config"

    echo "$backup_path"
}

# Backup PostgreSQL database
backup_database() {
    local backup_path=$1
    local db_file="$backup_path/database/database.sql"

    log_info "Backing up PostgreSQL database..."

    if [ -z "${DATABASE_URL:-}" ]; then
        log_error "DATABASE_URL not set in environment"
        return 1
    fi

    # Extract connection details from DATABASE_URL
    # Format: postgresql://user:pass@host:port/dbname
    local db_url="$DATABASE_URL"

    log_verbose "Database URL configured"

    # Use docker-compose to run pg_dump
    if command -v docker-compose &> /dev/null; then
        log_verbose "Using docker-compose for database backup"

        # For Neon or external database
        if [[ "$db_url" == *"neon.tech"* ]] || [[ "$db_url" == *"amazonaws.com"* ]]; then
            log_verbose "Detected external database (Neon/AWS)"
            pg_dump "$DATABASE_URL" > "$db_file" 2>&1
        else
            # For local Docker database
            log_verbose "Using local Docker database"
            docker-compose exec -T postgres pg_dumpall -U "${POSTGRES_USER:-postgres}" > "$db_file" 2>&1
        fi

        if [ $? -eq 0 ]; then
            log_success "Database backup completed: $(du -h "$db_file" | cut -f1)"
        else
            log_error "Database backup failed"
            return 1
        fi
    else
        log_error "docker-compose not found"
        return 1
    fi
}

# Backup Docker volumes
backup_volumes() {
    local backup_path=$1

    log_info "Backing up Docker volumes..."

    # Define volumes to backup
    local volumes=(
        "backend_uploads:uploads"
        "backend_logs:logs"
        "redis_data:redis"
    )

    for volume_spec in "${volumes[@]}"; do
        IFS=':' read -r volume_name volume_type <<< "$volume_spec"
        local full_volume_name="ai-playground_${volume_name}"

        log_verbose "Backing up volume: $full_volume_name"

        # Check if volume exists
        if docker volume inspect "$full_volume_name" &> /dev/null; then
            local volume_backup="$backup_path/volumes/${volume_type}.tar"

            docker run --rm \
                -v "${full_volume_name}:/data" \
                -v "$(pwd)/$backup_path/volumes:/backup" \
                alpine \
                tar czf "/backup/${volume_type}.tar.gz" -C /data . 2>&1

            if [ $? -eq 0 ]; then
                log_success "Volume backup completed: ${volume_type} ($(du -h "$backup_path/volumes/${volume_type}.tar.gz" | cut -f1))"
            else
                log_warning "Volume backup failed: ${volume_type}"
            fi
        else
            log_verbose "Volume $full_volume_name does not exist, skipping"
        fi
    done
}

# Backup configuration files
backup_config() {
    local backup_path=$1

    log_info "Backing up configuration files..."

    # List of config files to backup
    local configs=(
        ".env.docker"
        ".env.docker.production"
        "docker-compose.yml"
        "docker-compose.production.yml"
        "backend/alembic.ini"
        "docker/redis/redis.conf"
    )

    for config in "${configs[@]}"; do
        if [ -f "$config" ]; then
            local config_basename=$(basename "$config")
            log_verbose "Backing up: $config"
            cp "$config" "$backup_path/config/$config_basename"
        fi
    done

    log_success "Configuration backup completed"
}

# Create backup manifest
create_manifest() {
    local backup_path=$1
    local manifest="$backup_path/MANIFEST.txt"

    log_verbose "Creating backup manifest"

    {
        echo "AI Playground Backup Manifest"
        echo "=============================="
        echo ""
        echo "Backup Information:"
        echo "  Environment: $ENVIRONMENT"
        echo "  Timestamp: $TIMESTAMP"
        echo "  Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "  Hostname: $(hostname)"
        echo "  User: $(whoami)"
        echo ""
        echo "Backup Contents:"
        echo ""
        find "$backup_path" -type f -exec ls -lh {} \; | awk '{print "  " $9 " (" $5 ")"}'
        echo ""
        echo "Total Size: $(du -sh "$backup_path" | cut -f1)"
        echo ""
        echo "Checksums (SHA256):"
        echo ""
        find "$backup_path" -type f -name "*.sql" -o -name "*.tar.gz" | while read file; do
            echo "  $(basename "$file"): $(sha256sum "$file" | cut -d' ' -f1)"
        done
    } > "$manifest"

    log_success "Manifest created: $manifest"
}

# Compress entire backup (optional)
compress_backup() {
    local backup_path=$1

    if [ "$COMPRESS" = true ]; then
        log_info "Compressing backup..."

        local backup_archive="${backup_path}.tar.gz"
        tar czf "$backup_archive" -C "$(dirname "$backup_path")" "$(basename "$backup_path")" 2>&1

        if [ $? -eq 0 ]; then
            log_success "Backup compressed: $(du -h "$backup_archive" | cut -f1)"
            log_verbose "Removing uncompressed backup"
            rm -rf "$backup_path"
            echo "$backup_archive"
        else
            log_error "Compression failed"
            echo "$backup_path"
        fi
    else
        echo "$backup_path"
    fi
}

# Cleanup old backups based on retention policy
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."

    local cleanup_path="$BACKUP_DIR/$ENVIRONMENT"

    if [ ! -d "$cleanup_path" ]; then
        log_verbose "Backup directory does not exist: $cleanup_path"
        return 0
    fi

    local deleted_count=0

    # Find and delete old backups
    find "$cleanup_path" -maxdepth 1 -type d -mtime +$RETENTION_DAYS | while read old_backup; do
        if [ "$old_backup" != "$cleanup_path" ]; then
            log_verbose "Deleting old backup: $old_backup"
            rm -rf "$old_backup"
            ((deleted_count++))
        fi
    done

    # Also handle compressed backups
    find "$cleanup_path" -maxdepth 1 -type f -name "*.tar.gz" -mtime +$RETENTION_DAYS | while read old_archive; do
        log_verbose "Deleting old archive: $old_archive"
        rm -f "$old_archive"
        ((deleted_count++))
    done

    if [ $deleted_count -gt 0 ]; then
        log_success "Cleaned up $deleted_count old backup(s)"
    else
        log_verbose "No old backups to clean up"
    fi
}

# Main backup process
main() {
    log_info "Starting AI Playground backup process"
    log_info "Environment: $ENVIRONMENT"
    log_info "Backup directory: $BACKUP_DIR"
    log_info "Retention: $RETENTION_DAYS days"
    echo ""

    # Load environment
    load_env

    # Setup backup directory
    local backup_path=$(setup_backup_directory)
    log_success "Backup directory created: $backup_path"
    echo ""

    # Perform backups
    backup_database "$backup_path" || log_warning "Database backup had issues"
    echo ""

    backup_volumes "$backup_path"
    echo ""

    backup_config "$backup_path"
    echo ""

    # Create manifest
    create_manifest "$backup_path"
    echo ""

    # Compress if requested
    local final_backup=$(compress_backup "$backup_path")
    echo ""

    # Cleanup old backups
    cleanup_old_backups
    echo ""

    # Final summary
    log_success "================================"
    log_success "Backup completed successfully!"
    log_success "================================"
    log_info "Backup location: $final_backup"
    log_info "Backup size: $(du -sh "$final_backup" | cut -f1)"
    echo ""

    # Show recent backups
    log_info "Recent backups:"
    ls -lht "$BACKUP_DIR/$ENVIRONMENT" | head -n 6
}

# Run main function
main
