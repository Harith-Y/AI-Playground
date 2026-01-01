#!/bin/bash

################################################################################
# AI Playground Automated Backup Cron Job Script
#
# This script is designed to be run by cron for automated scheduled backups
# It includes logging, error handling, and notifications
#
# Installation:
#   1. Make executable: chmod +x backup-cron.sh
#   2. Add to crontab: crontab -e
#   3. Add line: 0 2 * * * /path/to/backup-cron.sh >> /var/log/aiplayground-backup.log 2>&1
#
# Cron Schedule Examples:
#   0 2 * * *         # Daily at 2 AM
#   0 2 * * 0         # Weekly on Sunday at 2 AM
#   0 2 1 * *         # Monthly on the 1st at 2 AM
#   0 */6 * * *       # Every 6 hours
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/var/log/aiplayground-backup.log"
LOCK_FILE="/tmp/aiplayground-backup.lock"

# Email notification settings (optional)
ENABLE_EMAIL_NOTIFICATIONS=false
EMAIL_TO="admin@example.com"
EMAIL_FROM="backup@aiplayground.local"

# Slack notification settings (optional)
ENABLE_SLACK_NOTIFICATIONS=false
SLACK_WEBHOOK_URL=""

# Logging with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Send email notification
send_email() {
    local subject=$1
    local body=$2

    if [ "$ENABLE_EMAIL_NOTIFICATIONS" = true ]; then
        echo "$body" | mail -s "$subject" -r "$EMAIL_FROM" "$EMAIL_TO"
        log "Email notification sent to $EMAIL_TO"
    fi
}

# Send Slack notification
send_slack() {
    local message=$1
    local color=$2  # good, warning, danger

    if [ "$ENABLE_SLACK_NOTIFICATIONS" = true ] && [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"text\": \"$message\",
                    \"footer\": \"AI Playground Backup\",
                    \"ts\": $(date +%s)
                }]
            }" \
            > /dev/null 2>&1

        log "Slack notification sent"
    fi
}

# Check if another backup is running
check_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")

        if kill -0 "$pid" 2>/dev/null; then
            log "ERROR: Another backup is already running (PID: $pid)"
            exit 1
        else
            log "WARNING: Stale lock file found, removing"
            rm -f "$LOCK_FILE"
        fi
    fi

    echo $$ > "$LOCK_FILE"
    log "Lock acquired"
}

# Remove lock file
remove_lock() {
    rm -f "$LOCK_FILE"
    log "Lock released"
}

# Main backup execution
main() {
    log "========================================="
    log "Starting automated backup process"
    log "========================================="

    # Change to project directory
    cd "$PROJECT_ROOT"

    # Run backup script
    local backup_result=0
    if "$SCRIPT_DIR/backup.sh" -e production -c -r 30 >> "$LOG_FILE" 2>&1; then
        log "Backup completed successfully"

        # Send success notification
        send_email \
            "[AI Playground] Backup Successful" \
            "Automated backup completed successfully at $(date)"

        send_slack \
            "✅ Automated backup completed successfully\nEnvironment: production\nTime: $(date)" \
            "good"

        backup_result=0
    else
        log "ERROR: Backup failed"

        # Send failure notification
        send_email \
            "[AI Playground] Backup FAILED" \
            "Automated backup failed at $(date). Please check logs at $LOG_FILE"

        send_slack \
            "❌ Automated backup FAILED\nEnvironment: production\nTime: $(date)\nCheck logs for details" \
            "danger"

        backup_result=1
    fi

    log "========================================="
    log "Automated backup process finished"
    log "========================================="
    echo ""

    return $backup_result
}

# Trap to ensure lock is always removed
trap remove_lock EXIT

# Check lock and run
check_lock
main
exit_code=$?

exit $exit_code
