# Backup Scripts Quick Reference

Quick reference guide for AI Playground backup and restore scripts.

## Quick Commands

### Backup

```bash
# Development backup
./backup.sh

# Production backup (compressed, 30-day retention)
./backup.sh -e production -c -r 30

# Production backup with cloud upload
./backup.sh -e production -c && \
./cloud-backup.sh -p r2 -b aiplayground-backups -e ./backups/production/latest.tar.gz
```

### Restore

```bash
# Full restore
./restore.sh ./backups/production/20260101_120000

# Database only
./restore.sh -d ./backups/production/20260101_120000

# Force restore (no confirmation)
./restore.sh -f ./backups/production/latest.tar.gz
```

### Automated Backups

```bash
# Setup cron (daily at 2 AM)
echo "0 2 * * * $(pwd)/backup-cron.sh >> /var/log/aiplayground-backup.log 2>&1" | crontab -

# View cron jobs
crontab -l

# View backup logs
tail -f /var/log/aiplayground-backup.log
```

## Script Overview

| Script | Purpose |
|--------|---------|
| [backup.sh](backup.sh) | Create backups (database + volumes) |
| [restore.sh](restore.sh) | Restore from backup |
| [backup-cron.sh](backup-cron.sh) | Automated backup via cron |
| [cloud-backup.sh](cloud-backup.sh) | Upload to cloud storage (S3/R2/GCS/Azure) |

## Environment Variables

### Required

- `DATABASE_URL` - PostgreSQL connection string

### Optional

```bash
# Backup configuration
BACKUP_SCHEDULE="0 2 * * *"      # Cron schedule
BACKUP_RETENTION_DAYS=30         # Days to keep backups
BACKUP_COMPRESS=true             # Compress backups

# Cloud backup
ENABLE_CLOUD_BACKUP=true
CLOUD_PROVIDER=r2                # s3, r2, gcs, azure
CLOUD_BUCKET=aiplayground-backups

# AWS S3
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Cloudflare R2
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...

# Notifications
ENABLE_EMAIL_NOTIFICATIONS=true
EMAIL_TO=admin@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## Common Use Cases

### Daily Production Backup

```bash
# In crontab
0 2 * * * /path/to/backup-cron.sh >> /var/log/aiplayground-backup.log 2>&1
```

### Backup Before Deployment

```bash
# Create backup before deploying
./backup.sh -e production -c
# Deploy application
docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

### Disaster Recovery

```bash
# 1. Get latest backup from cloud
aws s3 cp s3://my-backups/latest.tar.gz ./backups/

# 2. Restore
./restore.sh -f ./backups/latest.tar.gz

# 3. Verify
docker-compose logs -f backend
curl http://localhost:8000/health
```

### Migrate to New Server

```bash
# On old server
./backup.sh -e production -c
./cloud-backup.sh -p s3 -b migration ./backups/production/latest.tar.gz

# On new server
aws s3 cp s3://migration/latest.tar.gz ./backups/
./restore.sh -f ./backups/latest.tar.gz
```

## Troubleshooting

### Check Backup Status

```bash
# View recent backups
ls -lht backups/production/ | head

# Check backup size
du -sh backups/production/*

# View manifest
cat backups/production/latest/MANIFEST.txt
```

### Test Restore

```bash
# Always test restore in non-production first!
./restore.sh -e test ./backups/production/latest.tar.gz
```

### Verify Backup

```bash
# Check database backup
head -n 10 backups/production/latest/database/database.sql

# Check volumes
tar -tzf backups/production/latest/volumes/uploads.tar.gz | head
```

## Getting Help

```bash
# Show help for any script
./backup.sh --help
./restore.sh --help
./cloud-backup.sh --help
```

## Full Documentation

See [BACKUP.md](../../BACKUP.md) for complete documentation.
