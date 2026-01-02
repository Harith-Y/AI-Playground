# Automated Database Migrations

Comprehensive guide to automated database migrations in the AI-Playground ML system.

## Overview

The migration system provides:

- ✅ **Automatic migrations** on application startup
- ✅ **CI/CD integration** for deployment pipelines
- ✅ **Safety checks** before applying migrations
- ✅ **Rollback capabilities** for recovery
- ✅ **Status monitoring** via health endpoints
- ✅ **Cross-platform scripts** (Windows, Linux, macOS)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Application Startup                    │
├─────────────────────────────────────────────────────────┤
│  1. Check database connection                           │
│  2. Get current & head revisions                        │
│  3. Apply pending migrations (if any)                   │
│  4. Verify migration success                            │
│  5. Start application server                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                        │
├─────────────────────────────────────────────────────────┤
│  1. Run migration tests on PostgreSQL                   │
│  2. Validate migration chain                            │
│  3. Test rollback capability                            │
│  4. Deploy to staging (auto)                            │
│  5. Deploy to production (manual approval)              │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Automatic Migrations (Default)

Migrations run automatically when you start the application:

```bash
# Application startup automatically runs migrations
python start.py

# Or via Docker
docker-compose up
```

### Manual Migration Commands

**Check migration status:**

```bash
# Linux/Mac
./backend/scripts/migrate.sh status

# Windows
backend\scripts\migrate.bat status

# Or directly with Python
cd backend
python -m app.db.migration_manager status
```

**Run migrations:**

```bash
# Linux/Mac
./backend/scripts/migrate.sh upgrade

# Windows
backend\scripts\migrate.bat upgrade

# Or via Python script
cd backend
python scripts/run_migrations.py
```

**Create new migration:**

```bash
# Linux/Mac
./backend/scripts/migrate.sh create "Add user preferences table"

# Windows
backend\scripts\migrate.bat create "Add user preferences table"

# Or with Alembic directly
cd backend
alembic revision --autogenerate -m "Add user preferences table"
```

**Rollback migration:**

```bash
# Rollback one migration
./backend/scripts/migrate.sh downgrade -1

# Rollback to specific revision
./backend/scripts/migrate.sh downgrade abc123

# Windows
backend\scripts\migrate.bat downgrade -1
```

**View migration history:**

```bash
./backend/scripts/migrate.sh history

# Windows
backend\scripts\migrate.bat history
```

## Configuration

### Environment Variables

```bash
# Database connection
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Migration behavior
MIGRATION_WAIT_TIMEOUT=60        # Seconds to wait for database
MIGRATION_FAIL_ON_ERROR=true     # Exit on migration failure
MIGRATION_DRY_RUN=false          # Preview changes without applying
MIGRATION_CREATE_BACKUP=false    # Create backup before migration
```

### Python Configuration

```python
from app.db.migration_manager import MigrationManager

# Create manager with custom settings
manager = MigrationManager(
    database_url="postgresql://...",
    auto_upgrade=True,               # Auto-upgrade to head
    backup_before_upgrade=False,     # Create backup
    max_retry_attempts=3,            # Retry on failure
    retry_delay=5                    # Delay between retries (seconds)
)

# Run migrations
success = manager.auto_migrate(wait_for_db=True)
```

## Usage Examples

### Development Workflow

```bash
# 1. Make model changes
# Edit backend/app/models/*.py

# 2. Generate migration
cd backend
alembic revision --autogenerate -m "Add new fields to User model"

# 3. Review generated migration
# Check backend/alembic/versions/xxxx_add_new_fields.py

# 4. Test migration locally
python scripts/run_migrations.py

# 5. Verify changes
python -m app.db.migration_manager status

# 6. Commit migration script
git add alembic/versions/xxxx_add_new_fields.py
git commit -m "Add migration for User model fields"
```

### Production Deployment

```bash
# 1. Deploy new code
git push origin main

# 2. CI/CD automatically:
#    - Tests migrations on fresh database
#    - Validates migration chain
#    - Tests rollback capability
#    - Deploys to staging
#    - Awaits production approval

# 3. Migrations run automatically when containers start
docker-compose up -d

# 4. Verify migration status
curl http://localhost:8000/health/migrations
```

### Docker Deployment

```dockerfile
# Dockerfile already includes migration support
# Migrations run automatically on container startup

# Build and deploy
docker build -t ml-backend ./backend
docker run -e DATABASE_URL=postgresql://... ml-backend

# Migrations will run before server starts
```

### Kubernetes Deployment

```yaml
# init-container runs migrations before app starts
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-backend
spec:
  template:
    spec:
      initContainers:
        - name: run-migrations
          image: ml-backend:latest
          command: ["python", "scripts/run_migrations.py"]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
      containers:
        - name: app
          image: ml-backend:latest
          # App starts after migrations complete
```

## Migration Manager API

### MigrationManager Class

```python
from app.db.migration_manager import MigrationManager

manager = MigrationManager()

# Check connection
is_connected = manager.check_database_connection()

# Wait for database
is_ready = manager.wait_for_database(timeout=30)

# Get current revision
current = manager.get_current_revision()  # "abc123" or None

# Get head revision
head = manager.get_head_revision()  # "def456"

# Check if migration needed
needs_migration = manager.is_migration_needed()  # True/False

# Get pending migrations
pending = manager.get_pending_migrations()  # ["abc123", "def456"]

# Get detailed status
status = manager.get_migration_status()
# {
#   "current_revision": "abc123",
#   "head_revision": "def456",
#   "is_up_to_date": False,
#   "pending_migrations": ["def456"],
#   "pending_count": 1,
#   "timestamp": "2026-01-02T12:00:00"
# }

# Upgrade to head
success = manager.upgrade_to_head(dry_run=False)

# Upgrade with retry
success = manager.upgrade_with_retry()

# Downgrade
success = manager.downgrade("-1")  # One back
success = manager.downgrade("abc123")  # Specific revision

# Create new revision
rev_id = manager.create_revision(
    message="Add new table",
    autogenerate=True
)

# Display status
manager.show_current_revision()
manager.show_history(limit=10)
```

### Convenience Functions

```python
from app.db.migration_manager import (
    run_migrations_on_startup,
    check_migration_status
)

# Run on startup (used in start.py)
success = run_migrations_on_startup(
    auto_upgrade=True,
    wait_for_db=True,
    fail_on_error=False
)

# Check status (used in health endpoint)
status = check_migration_status()
```

## Health Endpoints

### Basic Health Check

```bash
GET /health

Response:
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Migration Status Check

```bash
GET /health/migrations

Response:
{
  "current_revision": "abc123def456",
  "head_revision": "abc123def456",
  "is_up_to_date": true,
  "pending_migrations": [],
  "pending_count": 0,
  "timestamp": "2026-01-02T12:00:00.000Z"
}

# When migrations pending:
{
  "current_revision": "abc123",
  "head_revision": "def456",
  "is_up_to_date": false,
  "pending_migrations": ["def456"],
  "pending_count": 1,
  "timestamp": "2026-01-02T12:00:00.000Z"
}
```

Use these endpoints for:

- Load balancer health checks
- Monitoring systems (Prometheus, Datadog)
- Deployment verification
- Alerting on migration issues

## CI/CD Integration

### GitHub Actions

The `.github/workflows/migrations.yml` workflow:

1. **Check Migrations** (all branches)

   - Validates migration script syntax
   - Checks migration chain integrity

2. **Test Migrations** (all branches)

   - Spins up PostgreSQL service
   - Applies migrations to fresh database
   - Tests rollback capability
   - Re-applies migrations

3. **Deploy Staging** (develop branch)

   - Runs migrations on staging database
   - Automatic deployment

4. **Deploy Production** (main branch)
   - Creates database backup
   - Runs migrations on production
   - Requires manual approval
   - Sends notifications

### GitLab CI

```yaml
# .gitlab-ci.yml
test-migrations:
  stage: test
  services:
    - postgres:15
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_password
    DATABASE_URL: postgresql://test_user:test_password@postgres:5432/test_db
  script:
    - cd backend
    - pip install -r requirements.txt
    - python scripts/run_migrations.py
    - python -m pytest tests/test_migrations.py

deploy-migrations:
  stage: deploy
  only:
    - main
  script:
    - cd backend
    - pip install -r requirements.txt
    - python scripts/run_migrations.py
  environment:
    name: production
```

### Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Test Migrations') {
            steps {
                sh 'cd backend && pip install -r requirements.txt'
                sh 'cd backend && python scripts/run_migrations.py'
            }
        }

        stage('Deploy Migrations') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([string(credentialsId: 'db-url', variable: 'DATABASE_URL')]) {
                    sh 'cd backend && python scripts/run_migrations.py'
                }
            }
        }
    }
}
```

## Safety Features

### 1. Connection Validation

```python
# Waits for database before attempting migration
manager.wait_for_database(timeout=30)

# Validates connection is active
manager.check_database_connection()
```

### 2. Dry Run Mode

```bash
# Preview what would be done
MIGRATION_DRY_RUN=true python scripts/run_migrations.py

# Or programmatically
manager.upgrade_to_head(dry_run=True)
```

### 3. Retry Logic

```python
# Automatic retry on transient failures
manager.upgrade_with_retry()  # 3 attempts by default

# Configure retry behavior
manager = MigrationManager(
    max_retry_attempts=5,
    retry_delay=10  # seconds
)
```

### 4. Rollback Support

```bash
# Rollback one migration
python -m app.db.migration_manager downgrade --revision -1

# Rollback to specific revision
python -m app.db.migration_manager downgrade --revision abc123
```

### 5. Status Verification

```python
# Verify migration succeeded
status = manager.get_migration_status()
assert status['is_up_to_date'], "Migration failed!"
```

## Troubleshooting

### Migration Fails on Startup

```bash
# Check migration status
python -m app.db.migration_manager status

# View pending migrations
python -c "from app.db.migration_manager import MigrationManager; m = MigrationManager(); print(m.get_pending_migrations())"

# Try manual migration with verbose output
cd backend
alembic upgrade head --verbose

# Check for errors in migration scripts
python -m py_compile alembic/versions/*.py
```

### Database Connection Issues

```bash
# Test database connection
python -c "from app.db.migration_manager import MigrationManager; m = MigrationManager(); print(m.check_database_connection())"

# Wait for database with timeout
python -c "from app.db.migration_manager import MigrationManager; m = MigrationManager(); print(m.wait_for_database(timeout=60))"

# Check DATABASE_URL environment variable
echo $DATABASE_URL  # Linux/Mac
echo %DATABASE_URL%  # Windows
```

### Migration Stuck at Old Revision

```bash
# Check current vs head
python -m app.db.migration_manager status

# Manually upgrade
cd backend
alembic upgrade head

# If that fails, check for conflicts
alembic history
alembic current
```

### Rollback Failed Migration

```bash
# Rollback to previous revision
python -m app.db.migration_manager downgrade --revision -1

# Or rollback to specific known-good revision
python -m app.db.migration_manager downgrade --revision abc123

# Then investigate the failed migration
cd backend/alembic/versions
# Edit the problematic migration file

# Re-apply
python scripts/run_migrations.py
```

### Migration Script Has Errors

```bash
# Check syntax
python -m py_compile backend/alembic/versions/xxxx_migration.py

# Validate Alembic config
cd backend
alembic check

# Test on fresh database
# 1. Create test database
# 2. Set DATABASE_URL to test database
# 3. Run migrations
# 4. Verify success
```

## Best Practices

### 1. Always Review Generated Migrations

```bash
# After auto-generating migration
alembic revision --autogenerate -m "Add field"

# Review the generated file
cat alembic/versions/xxxx_add_field.py

# Check for:
# - Correct table/column names
# - Proper indexes
# - Data migrations if needed
# - No unintended changes
```

### 2. Test Migrations Locally

```bash
# Create test database
createdb test_migrations

# Test migration
DATABASE_URL=postgresql://localhost/test_migrations python scripts/run_migrations.py

# Test rollback
python -m app.db.migration_manager downgrade --revision -1

# Test re-apply
python scripts/run_migrations.py

# Drop test database
dropdb test_migrations
```

### 3. Use Descriptive Messages

```bash
# Good
alembic revision -m "Add user_preferences table with JSONB column"
alembic revision -m "Create index on dataset.created_at for performance"
alembic revision -m "Migrate legacy_status to new status_enum"

# Bad
alembic revision -m "Update"
alembic revision -m "Fix"
alembic revision -m "Changes"
```

### 4. Handle Data Migrations Carefully

```python
# In migration file
from alembic import op
import sqlalchemy as sa

def upgrade():
    # 1. Add new column (nullable)
    op.add_column('users', sa.Column('email_verified', sa.Boolean(), nullable=True))

    # 2. Set default values for existing rows
    op.execute("UPDATE users SET email_verified = false WHERE email_verified IS NULL")

    # 3. Make column non-nullable
    op.alter_column('users', 'email_verified', nullable=False)

def downgrade():
    op.drop_column('users', 'email_verified')
```

### 5. Keep Migrations Small

```bash
# Split large changes into multiple migrations

# Migration 1: Add columns
alembic revision -m "Add new user fields"

# Migration 2: Migrate data
alembic revision -m "Populate new user fields from legacy"

# Migration 3: Remove old columns
alembic revision -m "Remove legacy user fields"
```

### 6. Never Edit Applied Migrations

```bash
# If migration was already applied to any environment:
# - Do NOT edit the migration file
# - Create a new migration to fix issues
# - Use downgrade only for development

# Create corrective migration instead
alembic revision -m "Fix index created in previous migration"
```

## Monitoring

### Prometheus Metrics

```python
# Add to monitoring system
from prometheus_client import Gauge, Histogram

migration_status = Gauge(
    'db_migration_up_to_date',
    'Database migration status (1=up to date, 0=pending)'
)

migration_duration = Histogram(
    'db_migration_duration_seconds',
    'Time taken to run migrations'
)

# Update in migration code
status = manager.get_migration_status()
migration_status.set(1 if status['is_up_to_date'] else 0)
```

### Logging

```python
# Migrations automatically log to application logger
# Check logs for:
logger.info("Starting database migration to head...")
logger.info("Database migration completed successfully")
logger.error("Migration failed: [error details]")

# View logs
tail -f logs/app.log | grep migration
```

### Alerting

```bash
# Alert on migration failures
# Check health endpoint
curl http://api.example.com/health/migrations

# If is_up_to_date: false, send alert
# If pending_count > 0, send notification
```

## Advanced Topics

### Custom Migration Logic

```python
# In migration file
from alembic import op
from sqlalchemy.orm import Session

def upgrade():
    # Custom logic with session
    bind = op.get_bind()
    session = Session(bind=bind)

    # Complex data migration
    users = session.execute("SELECT id, old_field FROM users").fetchall()
    for user_id, old_value in users:
        new_value = transform_value(old_value)
        session.execute(
            "UPDATE users SET new_field = :value WHERE id = :id",
            {"value": new_value, "id": user_id}
        )

    session.commit()
```

### Branched Migrations

```bash
# Create branch for feature
alembic revision -m "Add feature X tables" --head head

# Merge branches
alembic merge -m "Merge feature X" rev1 rev2
```

### Multiple Database Support

```python
# Configure different databases
staging_manager = MigrationManager(
    database_url=os.getenv("STAGING_DATABASE_URL")
)

production_manager = MigrationManager(
    database_url=os.getenv("PRODUCTION_DATABASE_URL")
)
```

## Summary

The automated migration system provides:

- ✅ **Zero-downtime deployments** with automatic migrations
- ✅ **Safety checks** before applying changes
- ✅ **CI/CD integration** for continuous deployment
- ✅ **Rollback capabilities** for recovery
- ✅ **Health monitoring** for operations
- ✅ **Cross-platform support** for all environments

**Key Benefits:**

- No manual migration steps required
- Consistent migrations across all environments
- Reduced deployment risk
- Easy rollback if issues occur
- Complete audit trail of changes

## See Also

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Database Migration Best Practices](https://www.prisma.io/dataguide/types/relational/migration-strategies)
