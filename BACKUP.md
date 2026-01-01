# AI Playground Backup & Restore Guide

Complete guide for backing up and restoring your AI Playground application data, including database, file uploads, and configuration.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Backup Scripts](#backup-scripts)
- [Automated Backups](#automated-backups)
- [Cloud Backups](#cloud-backups)
- [Restore Process](#restore-process)
- [Docker Backup Service](#docker-backup-service)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

### What Gets Backed Up

✅ **PostgreSQL Database** - All tables, schemas, users, experiments, models
✅ **File Uploads** - User-uploaded datasets and files (`backend_uploads` volume)
✅ **Application Logs** - Backend logs (`backend_logs` volume)
✅ **Redis Data** - Cached data and Celery results (`redis_data` volume)
✅ **Configuration Files** - Environment files, Docker Compose configs

### Backup Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `backup.sh` | Main backup script | [scripts/backup/backup.sh](scripts/backup/backup.sh) |
| `restore.sh` | Restore from backup | [scripts/backup/restore.sh](scripts/backup/restore.sh) |
| `backup-cron.sh` | Automated cron job | [scripts/backup/backup-cron.sh](scripts/backup/backup-cron.sh) |
| `cloud-backup.sh` | Upload to cloud storage | [scripts/backup/cloud-backup.sh](scripts/backup/cloud-backup.sh) |

## Quick Start

### Manual Backup

```bash
cd /path/to/AI-Playground

# Basic backup (development)
./scripts/backup/backup.sh

# Production backup with compression
./scripts/backup/backup.sh -e production -c

# Custom backup location with 90-day retention
./scripts/backup/backup.sh -d /mnt/backups -r 90 -c
```

### Manual Restore

```bash
# Restore from specific backup
./scripts/backup/restore.sh ./backups/production/20260101_120000

# Restore database only
./scripts/backup/restore.sh -d ./backups/production/20260101_120000

# Restore without confirmation
./scripts/backup/restore.sh -f ./backups/production/20260101_120000.tar.gz
```

## Backup Scripts

### backup.sh

Main backup script that creates comprehensive backups.

**Usage:**
```bash
./scripts/backup/backup.sh [OPTIONS]
```

**Options:**
- `-e, --env ENV` - Environment (development|production) [default: development]
- `-d, --dir DIR` - Backup directory [default: ./backups]
- `-r, --retention DAYS` - Retention period [default: 30 days]
- `-c, --compress` - Compress backup with gzip
- `-v, --verbose` - Verbose output
- `-h, --help` - Show help

**Examples:**

```bash
# Development backup
./scripts/backup/backup.sh

# Production backup, compressed, 90-day retention
./scripts/backup/backup.sh -e production -c -r 90

# Backup to external drive
./scripts/backup/backup.sh -d /mnt/external/backups -c -v

# Custom retention for compliance
./scripts/backup/backup.sh -e production -r 365  # Keep for 1 year
```

**Output Structure:**
```
backups/
└── production/
    └── 20260101_120000/
        ├── database/
        │   └── database.sql
        ├── volumes/
        │   ├── uploads.tar.gz
        │   ├── logs.tar.gz
        │   └── redis.tar.gz
        ├── config/
        │   ├── .env.docker.production
        │   ├── docker-compose.yml
        │   └── docker-compose.production.yml
        └── MANIFEST.txt
```

### restore.sh

Restore script to recover from backups.

**Usage:**
```bash
./scripts/backup/restore.sh [OPTIONS] BACKUP_PATH
```

**Options:**
- `-e, --env ENV` - Environment to restore to
- `-d, --database-only` - Restore database only
- `-v, --volumes-only` - Restore volumes only
- `-f, --force` - Skip confirmation prompt
- `--verbose` - Verbose output
- `-h, --help` - Show help

**Examples:**

```bash
# Full restore with confirmation
./scripts/backup/restore.sh ./backups/production/20260101_120000

# Restore from compressed backup
./scripts/backup/restore.sh ./backups/production/20260101_120000.tar.gz

# Database only restore
./scripts/backup/restore.sh -d ./backups/production/20260101_120000

# Force restore without prompt
./scripts/backup/restore.sh -f -e production ./backups/latest.tar.gz
```

**What Happens During Restore:**

1. ⚠️ Stops Docker services
2. 📊 Restores database (drops existing data!)
3. 📁 Restores Docker volumes
4. ✅ Starts Docker services
5. 🔍 Verifies restore

**⚠️ WARNING:** Restore is destructive! It will replace existing data.

## Automated Backups

### Using Cron (Linux/macOS)

#### 1. Make Scripts Executable

```bash
chmod +x scripts/backup/backup.sh
chmod +x scripts/backup/backup-cron.sh
chmod +x scripts/backup/restore.sh
chmod +x scripts/backup/cloud-backup.sh
```

#### 2. Configure Cron Job

```bash
# Edit crontab
crontab -e

# Add one of these schedules:

# Daily at 2 AM
0 2 * * * /path/to/AI-Playground/scripts/backup/backup-cron.sh >> /var/log/aiplayground-backup.log 2>&1

# Weekly on Sunday at 2 AM
0 2 * * 0 /path/to/AI-Playground/scripts/backup/backup-cron.sh >> /var/log/aiplayground-backup.log 2>&1

# Every 6 hours
0 */6 * * * /path/to/AI-Playground/scripts/backup/backup-cron.sh >> /var/log/aiplayground-backup.log 2>&1

# Monthly on the 1st at 2 AM
0 2 1 * * /path/to/AI-Playground/scripts/backup/backup-cron.sh >> /var/log/aiplayground-backup.log 2>&1
```

#### 3. Enable Email Notifications

Edit `scripts/backup/backup-cron.sh`:

```bash
ENABLE_EMAIL_NOTIFICATIONS=true
EMAIL_TO="admin@yourdomain.com"
EMAIL_FROM="backup@aiplayground.local"
```

Install `mailutils`:
```bash
# Ubuntu/Debian
sudo apt-get install mailutils

# RHEL/CentOS
sudo yum install mailx
```

#### 4. Enable Slack Notifications

Edit `scripts/backup/backup-cron.sh`:

```bash
ENABLE_SLACK_NOTIFICATIONS=true
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

#### 5. View Backup Logs

```bash
# View recent backups
tail -f /var/log/aiplayground-backup.log

# Search for errors
grep ERROR /var/log/aiplayground-backup.log

# Count successful backups
grep "Backup completed successfully" /var/log/aiplayground-backup.log | wc -l
```

### Using Windows Task Scheduler

#### 1. Install Git Bash or WSL

Ensure you have a Bash environment on Windows.

#### 2. Create Task

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., Daily at 2 AM)
4. Action: Start a program
   - Program: `C:\Program Files\Git\bin\bash.exe`
   - Arguments: `/path/to/scripts/backup/backup-cron.sh`

## Cloud Backups

Upload backups to cloud storage for off-site redundancy.

### Supported Providers

- ✅ AWS S3
- ✅ Cloudflare R2
- ✅ Google Cloud Storage
- ✅ Azure Blob Storage

### AWS S3

#### Setup

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
```

#### Upload to S3

```bash
./scripts/backup/cloud-backup.sh \
  -p s3 \
  -b my-aiplayground-backups \
  -r us-east-1 \
  -e \
  ./backups/production/20260101_120000.tar.gz
```

#### Options

- `-p s3` - Use AWS S3
- `-b BUCKET` - S3 bucket name
- `-r REGION` - AWS region
- `-e` - Encrypt before upload
- `-d` - Delete local copy after upload

### Cloudflare R2

#### Setup

```bash
# Set environment variables
export R2_ACCOUNT_ID="your-account-id"
export R2_ACCESS_KEY_ID="your-access-key"
export R2_SECRET_ACCESS_KEY="your-secret-key"
```

#### Upload to R2

```bash
./scripts/backup/cloud-backup.sh \
  -p r2 \
  -b aiplayground-backups \
  -e -d \
  ./backups/production/latest.tar.gz
```

### Google Cloud Storage

#### Setup

```bash
# Install gcloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Set service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### Upload to GCS

```bash
./scripts/backup/cloud-backup.sh \
  -p gcs \
  -b gs-aiplayground-backups \
  -e \
  ./backups/production/20260101_120000.tar.gz
```

### Azure Blob Storage

#### Setup

```bash
# Install Azure CLI
# https://docs.microsoft.com/cli/azure/install-azure-cli

# Login
az login

# Set credentials
export AZURE_STORAGE_ACCOUNT="yourstorageaccount"
export AZURE_STORAGE_KEY="your-storage-key"
```

#### Upload to Azure

```bash
./scripts/backup/cloud-backup.sh \
  -p azure \
  -b aiplayground-backups \
  -e -d \
  ./backups/production/20260101_120000.tar.gz
```

### Encryption

Backups can be encrypted before upload using AES-256:

```bash
# First upload generates encryption key
./scripts/backup/cloud-backup.sh -p s3 -b my-backups -e backup.tar.gz

# Key saved to: ~/.aiplayground-backup-key
```

**⚠️ IMPORTANT:** Save the encryption key securely! You'll need it to decrypt backups.

**Decrypt a backup:**
```bash
openssl enc -aes-256-cbc -d -pbkdf2 \
  -in backup.tar.gz.enc \
  -out backup.tar.gz \
  -pass file:~/.aiplayground-backup-key
```

## Docker Backup Service

Run automated backups as a Docker service.

### Setup

#### 1. Configure Environment

Add to [.env.docker.production](.env.docker.production):

```env
# Backup Configuration
BACKUP_SCHEDULE=0 2 * * *           # Daily at 2 AM
BACKUP_ENVIRONMENT=production
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESS=true

# Cloud Backup (optional)
ENABLE_CLOUD_BACKUP=true
CLOUD_PROVIDER=r2
CLOUD_BUCKET=aiplayground-backups

# Cloudflare R2 credentials
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key

# Notifications (optional)
ENABLE_EMAIL_NOTIFICATIONS=false
EMAIL_TO=admin@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
```

#### 2. Start Backup Service

```bash
# Start with backup service
docker-compose -f docker-compose.yml -f docker-compose.backup.yml up -d

# View backup service logs
docker-compose -f docker-compose.backup.yml logs -f backup

# Manual backup via Docker
docker-compose -f docker-compose.backup.yml run --rm backup
```

#### 3. Stop Backup Service

```bash
docker-compose -f docker-compose.backup.yml down
```

### Backup Service Features

✅ Automated scheduled backups via cron
✅ Configurable schedule (cron syntax)
✅ Automatic cleanup of old backups
✅ Optional cloud upload (S3, R2, GCS, Azure)
✅ Email and Slack notifications
✅ Compression to save space
✅ Integrated with Docker ecosystem

## Best Practices

### Backup Strategy

#### 3-2-1 Rule

- **3 copies** of your data
- **2 different storage types** (local + cloud)
- **1 off-site backup**

**Example:**
1. Primary data (running database)
2. Local backup (./backups)
3. Cloud backup (S3/R2)

### Backup Schedule

| Environment | Frequency | Retention | Cloud Sync |
|-------------|-----------|-----------|------------|
| Development | Daily | 7 days | No |
| Staging | Daily | 30 days | Weekly |
| Production | Daily | 90 days | Daily |

**Recommended Cron Schedules:**

```bash
# Development: Daily at 2 AM, 7-day retention
0 2 * * * /path/to/backup-cron.sh -r 7

# Production: Daily at 2 AM, 90-day retention, cloud backup
0 2 * * * /path/to/backup-cron.sh -r 90 && /path/to/cloud-backup.sh
```

### Testing Backups

**Test restore regularly!** A backup you can't restore is useless.

```bash
# Monthly restore test
# 1. Create test database
# 2. Restore latest backup
# 3. Verify data integrity
# 4. Delete test database

./scripts/backup/restore.sh -e test ./backups/latest.tar.gz
```

### Monitoring

Monitor backup health:

```bash
# Check last backup date
ls -lt backups/production | head -n 2

# Verify backup size (should be consistent)
du -sh backups/production/*

# Count recent backups
find backups/production -type d -mtime -7 | wc -l
```

**Set up alerts:**
- No backup in 48 hours → Alert
- Backup size drops >50% → Alert
- Backup fails 2x in a row → Alert

### Security

1. **Encrypt sensitive backups**
   ```bash
   ./scripts/backup/cloud-backup.sh -e -p s3 -b backups backup.tar.gz
   ```

2. **Secure backup files**
   ```bash
   chmod 600 backups/*.tar.gz
   ```

3. **Secure encryption keys**
   ```bash
   chmod 600 ~/.aiplayground-backup-key
   # Store key in password manager
   ```

4. **Restrict access**
   ```bash
   # Only allow backup user
   chown backup:backup backups/
   chmod 700 backups/
   ```

## Restore Process

### Step-by-Step Restore

#### 1. Prepare

```bash
# List available backups
ls -lht backups/production/

# View backup manifest
cat backups/production/20260101_120000/MANIFEST.txt
```

#### 2. Test Restore (Recommended)

```bash
# Restore to test environment first
./scripts/backup/restore.sh -e test ./backups/production/latest.tar.gz

# Verify application works
docker-compose -f docker-compose.test.yml up -d
curl http://localhost:8000/health

# Clean up test
docker-compose -f docker-compose.test.yml down -v
```

#### 3. Production Restore

```bash
# IMPORTANT: This will stop services and replace data!

# Create a backup of current state first
./scripts/backup/backup.sh -e production -c

# Perform restore
./scripts/backup/restore.sh -e production -f ./backups/20260101_120000.tar.gz
```

#### 4. Verify

```bash
# Check services
docker-compose ps

# Check database
docker-compose exec backend psql $DATABASE_URL -c "SELECT COUNT(*) FROM users;"

# Check application
curl http://localhost:8000/health

# Check logs
docker-compose logs -f backend
```

### Partial Restore

#### Database Only

```bash
./scripts/backup/restore.sh -d ./backups/production/20260101_120000
```

#### Volumes Only

```bash
./scripts/backup/restore.sh -v ./backups/production/20260101_120000
```

#### Specific Volume

```bash
# Manually restore uploads only
docker run --rm \
  -v ai-playground_backend_uploads:/data \
  -v $(pwd)/backups/production/20260101_120000/volumes:/backup \
  alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/uploads.tar.gz -C /data"
```

## Troubleshooting

### Common Issues

#### Backup Fails: "DATABASE_URL not set"

**Solution:**
```bash
# Check environment file
cat .env.docker.production | grep DATABASE_URL

# Set manually
export DATABASE_URL="postgresql://user:pass@host/db"
./scripts/backup/backup.sh -e production
```

#### Backup Fails: "docker-compose not found"

**Solution:**
```bash
# Install Docker Compose
pip install docker-compose

# Or use docker compose (v2)
alias docker-compose='docker compose'
```

#### Restore Fails: "Permission denied"

**Solution:**
```bash
# Make scripts executable
chmod +x scripts/backup/*.sh

# Run as root if needed
sudo ./scripts/backup/restore.sh ./backups/latest
```

#### Cloud Upload Fails: "Credentials not found"

**Solution:**
```bash
# AWS S3
aws configure

# Cloudflare R2
export R2_ACCOUNT_ID="..."
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."

# Verify
./scripts/backup/cloud-backup.sh -p r2 -b test --verbose backup.tar.gz
```

#### Backup Too Large

**Solution:**
```bash
# Compress backups
./scripts/backup/backup.sh -c

# Clean old backups
./scripts/backup/backup.sh -r 7  # Keep only 7 days

# Exclude logs from backup (edit backup.sh)
# Comment out: backup_volumes "logs"
```

### Verification

#### Verify Backup Integrity

```bash
# Check backup files exist
ls -lh backups/production/latest/database/database.sql
ls -lh backups/production/latest/volumes/uploads.tar.gz

# Verify checksums
sha256sum backups/production/latest/database/database.sql

# Test database file
head -n 10 backups/production/latest/database/database.sql
```

#### Verify Restore

```bash
# After restore, check:

# 1. Database tables
docker-compose exec backend psql $DATABASE_URL -c "\dt"

# 2. Record counts
docker-compose exec backend psql $DATABASE_URL -c "
  SELECT 'users' as table_name, COUNT(*) FROM users
  UNION ALL
  SELECT 'datasets', COUNT(*) FROM datasets
  UNION ALL
  SELECT 'experiments', COUNT(*) FROM experiments;
"

# 3. File uploads
docker run --rm \
  -v ai-playground_backend_uploads:/data \
  alpine \
  ls -lh /data

# 4. Application health
curl http://localhost:8000/api/v1/auth/login
```

## Additional Resources

- [Docker Compose Backup Best Practices](https://docs.docker.com/storage/volumes/#back-up-restore-or-migrate-data-volumes)
- [PostgreSQL Backup Documentation](https://www.postgresql.org/docs/current/backup.html)
- [AWS S3 Backup Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/backup.html)
- [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) - Docker deployment guide

## Support

For backup and restore issues:
1. Check logs: `/var/log/aiplayground-backup.log`
2. Run with `--verbose` flag
3. Verify environment variables are set
4. Test with small backup first
5. Open an issue with error messages and logs
