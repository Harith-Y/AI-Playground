#!/bin/bash

################################################################################
# AI Playground Cloud Backup Script
#
# This script uploads backups to cloud storage (AWS S3, Google Cloud Storage,
# Azure Blob Storage, or Cloudflare R2)
#
# Usage:
#   ./cloud-backup.sh [OPTIONS] BACKUP_PATH
#
# Options:
#   -p, --provider PROVIDER    Cloud provider (s3|gcs|azure|r2) [required]
#   -b, --bucket BUCKET        Bucket/container name [required]
#   -r, --region REGION        Region (for S3/R2)
#   -e, --encrypt              Encrypt backup before upload
#   -d, --delete-local         Delete local backup after successful upload
#   -v, --verbose              Verbose output
#   -h, --help                 Show this help message
#
# Environment Variables:
#   AWS S3:
#     AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
#
#   Cloudflare R2:
#     R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID
#
#   Google Cloud:
#     GOOGLE_APPLICATION_CREDENTIALS
#
#   Azure:
#     AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY
#
# Examples:
#   ./cloud-backup.sh -p s3 -b my-backups ./backups/production/20260101_120000.tar.gz
#   ./cloud-backup.sh -p r2 -b aiplayground-backups -e -d ./backups/production/latest.tar.gz
################################################################################

set -euo pipefail

# Default configuration
PROVIDER=""
BUCKET=""
REGION=""
ENCRYPT=false
DELETE_LOCAL=false
VERBOSE=false
BACKUP_PATH=""
ENCRYPTION_KEY_FILE="$HOME/.aiplayground-backup-key"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--provider)
            PROVIDER="$2"
            shift 2
            ;;
        -b|--bucket)
            BUCKET="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -e|--encrypt)
            ENCRYPT=true
            shift
            ;;
        -d|--delete-local)
            DELETE_LOCAL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -n 35 "$0" | tail -n 33
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

# Validate inputs
validate_inputs() {
    if [ -z "$PROVIDER" ]; then
        log_error "Cloud provider is required (-p)"
        exit 1
    fi

    if [ -z "$BUCKET" ]; then
        log_error "Bucket name is required (-b)"
        exit 1
    fi

    if [ -z "$BACKUP_PATH" ]; then
        log_error "Backup path is required"
        exit 1
    fi

    if [ ! -f "$BACKUP_PATH" ]; then
        log_error "Backup file does not exist: $BACKUP_PATH"
        exit 1
    fi

    log_success "Input validation passed"
}

# Generate or load encryption key
setup_encryption() {
    if [ "$ENCRYPT" = true ]; then
        if [ ! -f "$ENCRYPTION_KEY_FILE" ]; then
            log_warning "Encryption key not found, generating new key"
            openssl rand -base64 32 > "$ENCRYPTION_KEY_FILE"
            chmod 600 "$ENCRYPTION_KEY_FILE"
            log_success "Encryption key generated: $ENCRYPTION_KEY_FILE"
            log_warning "IMPORTANT: Save this key securely! You'll need it to decrypt backups."
        fi

        log_verbose "Using encryption key from: $ENCRYPTION_KEY_FILE"
    fi
}

# Encrypt backup file
encrypt_backup() {
    if [ "$ENCRYPT" = true ]; then
        log_info "Encrypting backup..."

        local encrypted_file="${BACKUP_PATH}.enc"

        openssl enc -aes-256-cbc -salt -pbkdf2 \
            -in "$BACKUP_PATH" \
            -out "$encrypted_file" \
            -pass file:"$ENCRYPTION_KEY_FILE"

        if [ $? -eq 0 ]; then
            log_success "Backup encrypted: $encrypted_file"
            echo "$encrypted_file"
        else
            log_error "Encryption failed"
            exit 1
        fi
    else
        echo "$BACKUP_PATH"
    fi
}

# Upload to AWS S3
upload_to_s3() {
    local file=$1
    local filename=$(basename "$file")
    local s3_path="s3://${BUCKET}/${filename}"

    log_info "Uploading to AWS S3: $s3_path"

    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        return 1
    fi

    # Upload with progress
    aws s3 cp "$file" "$s3_path" \
        ${REGION:+--region "$REGION"} \
        --storage-class STANDARD_IA

    if [ $? -eq 0 ]; then
        log_success "Upload to S3 completed"

        # Set lifecycle policy (optional)
        log_verbose "File uploaded to: $s3_path"
        return 0
    else
        log_error "S3 upload failed"
        return 1
    fi
}

# Upload to Cloudflare R2
upload_to_r2() {
    local file=$1
    local filename=$(basename "$file")

    log_info "Uploading to Cloudflare R2: ${BUCKET}/${filename}"

    # R2 uses S3-compatible API
    if [ -z "${R2_ACCOUNT_ID:-}" ]; then
        log_error "R2_ACCOUNT_ID not set"
        return 1
    fi

    local r2_endpoint="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" \
    AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}" \
    aws s3 cp "$file" "s3://${BUCKET}/${filename}" \
        --endpoint-url "$r2_endpoint"

    if [ $? -eq 0 ]; then
        log_success "Upload to R2 completed"
        return 0
    else
        log_error "R2 upload failed"
        return 1
    fi
}

# Upload to Google Cloud Storage
upload_to_gcs() {
    local file=$1
    local filename=$(basename "$file")

    log_info "Uploading to Google Cloud Storage: gs://${BUCKET}/${filename}"

    # Check if gsutil is installed
    if ! command -v gsutil &> /dev/null; then
        log_error "gsutil is not installed"
        return 1
    fi

    gsutil cp "$file" "gs://${BUCKET}/${filename}"

    if [ $? -eq 0 ]; then
        log_success "Upload to GCS completed"
        return 0
    else
        log_error "GCS upload failed"
        return 1
    fi
}

# Upload to Azure Blob Storage
upload_to_azure() {
    local file=$1
    local filename=$(basename "$file")

    log_info "Uploading to Azure Blob Storage: ${BUCKET}/${filename}"

    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed"
        return 1
    fi

    az storage blob upload \
        --account-name "${AZURE_STORAGE_ACCOUNT}" \
        --container-name "${BUCKET}" \
        --name "${filename}" \
        --file "$file" \
        --tier Cool

    if [ $? -eq 0 ]; then
        log_success "Upload to Azure completed"
        return 0
    else
        log_error "Azure upload failed"
        return 1
    fi
}

# Upload backup based on provider
upload_backup() {
    local file=$1

    case "$PROVIDER" in
        s3)
            upload_to_s3 "$file"
            ;;
        r2)
            upload_to_r2 "$file"
            ;;
        gcs)
            upload_to_gcs "$file"
            ;;
        azure)
            upload_to_azure "$file"
            ;;
        *)
            log_error "Unknown provider: $PROVIDER"
            return 1
            ;;
    esac
}

# Cleanup local files
cleanup() {
    local file=$1

    if [ "$DELETE_LOCAL" = true ]; then
        log_info "Deleting local backup files..."
        rm -f "$file"

        # Also delete original if we encrypted
        if [ "$ENCRYPT" = true ] && [ "$file" != "$BACKUP_PATH" ]; then
            rm -f "$BACKUP_PATH"
        fi

        log_success "Local files deleted"
    fi
}

# Main process
main() {
    log_info "Starting cloud backup process"
    log_info "Provider: $PROVIDER"
    log_info "Bucket: $BUCKET"
    log_info "Backup: $BACKUP_PATH"
    echo ""

    # Validate
    validate_inputs

    # Setup encryption
    setup_encryption

    # Encrypt if needed
    local upload_file=$(encrypt_backup)
    echo ""

    # Upload to cloud
    if upload_backup "$upload_file"; then
        echo ""
        log_success "================================"
        log_success "Cloud backup completed!"
        log_success "================================"
        log_info "Provider: $PROVIDER"
        log_info "Bucket: $BUCKET"
        log_info "File: $(basename "$upload_file")"
        log_info "Size: $(du -h "$upload_file" | cut -f1)"
        echo ""

        # Cleanup
        cleanup "$upload_file"

        exit 0
    else
        log_error "Cloud backup failed"
        exit 1
    fi
}

# Run main
main
