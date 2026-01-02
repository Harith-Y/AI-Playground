# Database Migrations Guide

This directory contains Alembic database migrations for the AI Playground backend.

## Overview

Alembic is a database migration tool for SQLAlchemy. It allows us to version control our database schema and apply incremental changes in a controlled manner.

## Migration History

### 1. `5c1b6116de51_create_initial_tables.py`
**Date:** 2025-12-23
**Description:** Initial database schema creation

**Tables Created:**
- `users` - User accounts
- `datasets` - Uploaded datasets
- `experiments` - ML experiments
- `feature_engineering` - Feature engineering operations
- `preprocessing_steps` - Data preprocessing steps
- `model_runs` - Model training runs
- `tuning_runs` - Hyperparameter tuning runs
- `generated_code` - Generated code artifacts

### 2. `a1b2c3d4e5f6_add_preprocessing_history_table.py`
**Date:** 2025-12-28
**Description:** Add preprocessing history tracking

**Changes:**
- Added `preprocessing_history` table to track all preprocessing operations
- Includes execution time, status, error messages, and statistics
- Links to original and output datasets

### 3. `b2c3d4e5f6g7_add_status_metadata_to_model_runs.py`
**Date:** 2025-12-29
**Description:** Add training status tracking to model runs

**Changes:**
- Added `status` column to `model_runs` table (pending, running, completed, failed, cancelled)
- Added `metadata` JSONB column for storing task_id, feature_importance, errors, etc.
- Created index on `status` for faster filtering

**Usage:**
```python
# Store Celery task_id
model_run.metadata = {'task_id': 'abc123'}

# Store feature importance
model_run.metadata['feature_importance'] = [
    {'feature': 'age', 'importance': 0.35},
    {'feature': 'income', 'importance': 0.28}
]

# Store error information
model_run.metadata['error'] = {
    'type': 'ValueError',
    'message': 'Invalid input data',
    'traceback': '...'
}
```

### 4. `c3d4e5f6g7h8_add_is_active_fitted_transformer_to_preprocessing_steps.py`
**Date:** 2025-12-29
**Description:** Add active status and fitted transformer storage to preprocessing steps

**Changes:**
- Added `is_active` boolean column to enable/disable preprocessing steps
- Added `fitted_transformer` binary column to store serialized transformer objects
- Created index on `is_active` for faster filtering

**Usage:**
```python
from app.ml_engine.preprocessing.serialization import serialize_transformer

# Fit and store transformer
scaler = StandardScaler()
scaler.fit(data)
step.fitted_transformer = serialize_transformer(scaler)
step.is_active = True

# Later, during training
from app.ml_engine.preprocessing.serialization import deserialize_transformer
transformer = deserialize_transformer(step.fitted_transformer)
transformed_data = transformer.transform(new_data)
```

---

## Running Migrations

### Apply All Pending Migrations

```bash
# From backend directory
alembic upgrade head
```

### Check Current Migration Version

```bash
alembic current
```

### View Migration History

```bash
alembic history
```

### View Pending Migrations

```bash
alembic show head
```

### Rollback Last Migration

```bash
alembic downgrade -1
```

### Rollback to Specific Version

```bash
alembic downgrade a1b2c3d4e5f6
```

---

## Creating New Migrations

### Auto-generate Migration from Model Changes

```bash
# After modifying SQLAlchemy models
alembic revision --autogenerate -m "Description of changes"
```

### Create Empty Migration Template

```bash
alembic revision -m "Description of changes"
```

### Important Guidelines

1. **Always review auto-generated migrations** - Alembic's autogenerate is smart but not perfect
2. **Test migrations both ways** - Run upgrade and downgrade to ensure they work
3. **Use meaningful revision messages** - Describe what changed and why
4. **Don't modify existing migrations** - Once applied to production, create a new migration instead
5. **Include data migrations when needed** - Use `op.execute()` for SQL statements

---

## Migration File Structure

Each migration file contains:

```python
"""Description of migration

Revision ID: unique_revision_id
Revises: previous_revision_id
Create Date: timestamp
"""

def upgrade() -> None:
    """Apply schema changes."""
    # Changes to apply
    pass

def downgrade() -> None:
    """Revert schema changes."""
    # Reverse the changes
    pass
```

---

## Common Operations

### Add Column

```python
def upgrade():
    op.add_column('table_name',
        sa.Column('column_name', sa.String(), nullable=True)
    )

def downgrade():
    op.drop_column('table_name', 'column_name')
```

### Add Column with Default

```python
def upgrade():
    op.add_column('table_name',
        sa.Column('status', sa.String(), nullable=False, server_default='pending')
    )

def downgrade():
    op.drop_column('table_name', 'status')
```

### Create Index

```python
def upgrade():
    op.create_index(op.f('ix_table_column'), 'table_name', ['column_name'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_table_column'), table_name='table_name')
```

### Add Foreign Key

```python
def upgrade():
    op.create_foreign_key(
        'fk_table_other_table',
        'table_name', 'other_table',
        ['other_id'], ['id'],
        ondelete='CASCADE'
    )

def downgrade():
    op.drop_constraint('fk_table_other_table', 'table_name', type_='foreignkey')
```

### Modify Column Type

```python
def upgrade():
    op.alter_column('table_name', 'column_name',
        type_=sa.Integer(),
        existing_type=sa.String()
    )

def downgrade():
    op.alter_column('table_name', 'column_name',
        type_=sa.String(),
        existing_type=sa.Integer()
    )
```

---

## Troubleshooting

### Migration Failed Mid-way

If a migration fails partway through:

```bash
# Check current version (may show inconsistent state)
alembic current

# Manually fix the database to match the expected state
# Then mark the migration as complete
alembic stamp head
```

### Database Out of Sync

If your database doesn't match the migration history:

```bash
# Option 1: Reset to a known state
alembic downgrade base  # Removes all tables
alembic upgrade head    # Recreates everything

# Option 2: Stamp to current state (if DB is correct but migrations not tracked)
alembic stamp head
```

### Merge Conflicts in Migrations

If multiple developers create migrations simultaneously:

```bash
# Create a merge migration
alembic merge -m "Merge branch migrations" revision1 revision2
```

---

## Best Practices

1. **Run migrations in development first** - Test thoroughly before production
2. **Backup production database** - Always backup before running migrations
3. **Use transactions** - Migrations run in transactions by default, don't disable
4. **Keep migrations small** - One logical change per migration
5. **Document complex migrations** - Add comments explaining non-obvious changes
6. **Test rollbacks** - Ensure downgrade works before deploying

---

## Database Schema Diagram

```
users
  ├─ datasets
  │   ├─ preprocessing_steps (is_active, fitted_transformer)
  │   ├─ preprocessing_history
  │   └─ feature_engineering
  └─ experiments
      ├─ model_runs (status, metadata)
      │   └─ tuning_runs
      └─ generated_code
```

---

## Environment-Specific Migrations

### Development
```bash
# Apply all migrations
alembic upgrade head
```

### Production
```bash
# Review pending migrations
alembic history

# Apply migrations one at a time
alembic upgrade +1

# Or apply all at once (with caution)
alembic upgrade head
```

### Testing
```bash
# Use separate database
export DATABASE_URL="postgresql://user:pass@localhost/test_db"
alembic upgrade head
```

---

## Useful SQL Queries

### Check Current Schema Version
```sql
SELECT * FROM alembic_version;
```

### View Model Runs with Status
```sql
SELECT id, model_type, status, created_at
FROM model_runs
WHERE status = 'completed'
ORDER BY created_at DESC;
```

### View Active Preprocessing Steps
```sql
SELECT id, dataset_id, step_type, is_active, "order"
FROM preprocessing_steps
WHERE is_active = true
ORDER BY dataset_id, "order";
```

### View Preprocessing History
```sql
SELECT id, dataset_id, operation_type, status, execution_time, created_at
FROM preprocessing_history
ORDER BY created_at DESC
LIMIT 10;
```

---

## Related Documentation

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

## Support

For migration issues:
1. Check the Alembic documentation
2. Review the migration file comments
3. Consult the database schema diagram
4. Check git history for context on changes
