# TuningRun Database Model

## Overview

The `TuningRun` model stores information about hyperparameter tuning operations performed on trained models. It tracks the tuning method used, parameters tested, best results found, and the overall status of the tuning process.

## Table Structure

### Table Name
`tuning_runs`

### Columns

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid4() | Primary key, unique identifier for the tuning run |
| `model_run_id` | UUID | No | - | Foreign key to `model_runs.id`, the model being tuned |
| `tuning_method` | String | No | - | Method used: "grid_search", "random_search", or "bayesian" |
| `best_params` | JSONB | Yes | NULL | Best hyperparameters found during tuning |
| `results` | JSONB | Yes | NULL | Complete tuning results including top N combinations |
| `status` | Enum | No | RUNNING | Current status: RUNNING, COMPLETED, or FAILED |
| `created_at` | DateTime(TZ) | No | now() | Timestamp when tuning was initiated |

### Indexes

- **Primary Key:** `id`
- **Foreign Key:** `model_run_id` → `model_runs.id` (CASCADE on delete)
- **Index:** `id` (for fast lookups)
- **Index:** `model_run_id` (for querying tuning runs by model)
- **Index:** `status` (for filtering by status)

### Enums

#### TuningStatus
```python
class TuningStatus(str, enum.Enum):
    RUNNING = "running"      # Tuning is in progress
    COMPLETED = "completed"  # Tuning finished successfully
    FAILED = "failed"        # Tuning failed with error
```

## Relationships

### Parent Relationships
- **model_run** (Many-to-One): Each tuning run belongs to one model run
  - Foreign Key: `model_run_id` → `model_runs.id`
  - Cascade: DELETE (tuning runs are deleted when model run is deleted)

### Child Relationships
None (TuningRun is a leaf node in the relationship tree)

## JSONB Field Structures

### best_params (JSONB)
Stores the best hyperparameters found during tuning.

**Example:**
```json
{
  "n_estimators": 100,
  "max_depth": 10,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt"
}
```

### results (JSONB)
Stores complete tuning results including metadata and top N parameter combinations.

**Structure:**
```json
{
  "best_score": 0.9567,
  "total_combinations": 36,
  "cv_folds": 5,
  "scoring_metric": "accuracy",
  "tuning_time": 120.5,
  "tuning_method": "grid_search",
  "all_results": [
    {
      "rank": 1,
      "params": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2
      },
      "mean_score": 0.9567,
      "std_score": 0.0123,
      "scores": [0.95, 0.96, 0.97, 0.94, 0.96]
    },
    {
      "rank": 2,
      "params": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2
      },
      "mean_score": 0.9543,
      "std_score": 0.0145,
      "scores": [0.94, 0.96, 0.96, 0.95, 0.96]
    }
  ],
  "error": null
}
```

**Fields in results:**
- `best_score` (float): Best cross-validation score achieved
- `total_combinations` (int): Total number of parameter combinations tested
- `cv_folds` (int): Number of cross-validation folds used
- `scoring_metric` (string): Metric used for optimization (e.g., "accuracy", "f1", "r2")
- `tuning_time` (float): Duration of tuning in seconds
- `tuning_method` (string): Method used (grid_search, random_search, bayesian)
- `all_results` (array): Top N parameter combinations with scores
- `error` (string, optional): Error message if tuning failed

## Usage Examples

### Creating a TuningRun

```python
from app.models.tuning_run import TuningRun, TuningStatus
from uuid import uuid4
from datetime import datetime

tuning_run = TuningRun(
    id=uuid4(),
    model_run_id=model_run.id,
    tuning_method="grid_search",
    status=TuningStatus.RUNNING,
    created_at=datetime.utcnow()
)

db.add(tuning_run)
db.commit()
```

### Updating with Results

```python
tuning_run.best_params = {
    "n_estimators": 100,
    "max_depth": 10
}

tuning_run.results = {
    "best_score": 0.95,
    "total_combinations": 36,
    "all_results": [...]
}

tuning_run.status = TuningStatus.COMPLETED

db.commit()
```

### Querying TuningRuns

```python
# Get all tuning runs for a model
tuning_runs = db.query(TuningRun).filter(
    TuningRun.model_run_id == model_run_id
).all()

# Get completed tuning runs
completed_runs = db.query(TuningRun).filter(
    TuningRun.status == TuningStatus.COMPLETED
).all()

# Get tuning run with best score
best_tuning = db.query(TuningRun).filter(
    TuningRun.model_run_id == model_run_id,
    TuningRun.status == TuningStatus.COMPLETED
).order_by(
    TuningRun.results['best_score'].desc()
).first()
```

### Accessing Relationships

```python
# Get the model run for a tuning run
model_run = tuning_run.model_run

# Get all tuning runs for a model run
tuning_runs = model_run.tuning_runs
```

## Database Migration

The `tuning_runs` table is created in the initial migration:

**Migration File:** `backend/alembic/versions/5c1b6116de51_create_initial_tables.py`

**Migration Command:**
```sql
CREATE TABLE tuning_runs (
    id UUID PRIMARY KEY,
    model_run_id UUID NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    tuning_method VARCHAR NOT NULL,
    best_params JSONB,
    results JSONB,
    status tuningstatus NOT NULL DEFAULT 'running',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

CREATE INDEX ix_tuning_runs_id ON tuning_runs(id);
CREATE INDEX ix_tuning_runs_model_run_id ON tuning_runs(model_run_id);
CREATE INDEX ix_tuning_runs_status ON tuning_runs(status);
```

## Constraints

### Foreign Key Constraints
- `model_run_id` must reference an existing `model_runs.id`
- Cascade delete: When a model run is deleted, all its tuning runs are deleted

### Check Constraints
- `tuning_method` must be one of: "grid_search", "random_search", "bayesian"
- `status` must be one of: RUNNING, COMPLETED, FAILED

### Business Logic Constraints
- A tuning run can only be created for a completed model run
- `best_params` and `results` are NULL until tuning completes
- `status` transitions: RUNNING → COMPLETED or RUNNING → FAILED

## Performance Considerations

### Indexing Strategy
- **Primary Key Index (id):** Fast lookups by tuning run ID
- **Foreign Key Index (model_run_id):** Fast queries for all tuning runs of a model
- **Status Index:** Fast filtering by status (e.g., get all running tuning runs)

### JSONB Performance
- JSONB fields (`best_params`, `results`) are stored in binary format for fast access
- Can query nested JSONB fields using PostgreSQL operators:
  ```sql
  SELECT * FROM tuning_runs 
  WHERE results->>'best_score' > '0.9';
  ```

### Query Optimization
```python
# Good: Use indexes
tuning_runs = db.query(TuningRun).filter(
    TuningRun.model_run_id == model_run_id,
    TuningRun.status == TuningStatus.COMPLETED
).all()

# Bad: Full table scan
tuning_runs = db.query(TuningRun).filter(
    TuningRun.results['best_score'].astext.cast(Float) > 0.9
).all()
```

## Data Lifecycle

### Creation
1. User initiates tuning via API endpoint
2. TuningRun record created with status=RUNNING
3. Celery task triggered for async processing

### Processing
1. Celery task loads model and dataset
2. Performs hyperparameter search
3. Updates TuningRun with progress (optional)

### Completion
1. Task completes successfully
2. TuningRun updated with:
   - `best_params`: Best hyperparameters found
   - `results`: Complete results with top N combinations
   - `status`: COMPLETED

### Failure
1. Task fails with exception
2. TuningRun updated with:
   - `status`: FAILED
   - `results.error`: Error message

### Deletion
1. User deletes model run
2. All associated tuning runs are cascade deleted
3. No orphaned tuning runs remain

## Security Considerations

### Authorization
- Users can only access tuning runs for their own model runs
- Authorization checked through model_run → experiment → user relationship

### Data Privacy
- Tuning results contain only hyperparameters and scores
- No sensitive user data stored in tuning runs
- Model artifacts stored separately with access control

## Best Practices

### 1. Always Check Status Before Accessing Results
```python
if tuning_run.status == TuningStatus.COMPLETED:
    best_params = tuning_run.best_params
    best_score = tuning_run.results['best_score']
else:
    # Handle incomplete tuning
    pass
```

### 2. Store Comprehensive Results
```python
tuning_run.results = {
    'best_score': best_score,
    'total_combinations': n_combinations,
    'all_results': top_results,  # Top 20
    'cv_folds': cv_folds,
    'scoring_metric': scoring_metric,
    'tuning_time': duration,
    'tuning_method': method
}
```

### 3. Handle Errors Gracefully
```python
try:
    # Perform tuning
    pass
except Exception as e:
    tuning_run.status = TuningStatus.FAILED
    tuning_run.results = {
        'error': str(e),
        'error_type': type(e).__name__
    }
    db.commit()
```

### 4. Use Transactions
```python
with db.begin():
    tuning_run.status = TuningStatus.COMPLETED
    tuning_run.best_params = best_params
    tuning_run.results = results
    # All updates committed together
```

## Related Models

### ModelRun
- **Relationship:** Parent (One-to-Many)
- **Purpose:** The model being tuned
- **Fields Used:** model_type, hyperparameters, model_artifact_path

### Experiment
- **Relationship:** Grandparent (through ModelRun)
- **Purpose:** Authorization and context
- **Fields Used:** user_id, dataset_id

## API Integration

### Endpoints Using TuningRun
1. `POST /api/v1/tuning/tune` - Creates TuningRun
2. `GET /api/v1/tuning/tune/{tuning_run_id}/status` - Reads status
3. `GET /api/v1/tuning/tune/{tuning_run_id}/results` - Reads results

### Celery Tasks Using TuningRun
1. `tune_hyperparameters` - Updates TuningRun with results

## Monitoring and Debugging

### Useful Queries

**Count tuning runs by status:**
```sql
SELECT status, COUNT(*) 
FROM tuning_runs 
GROUP BY status;
```

**Average tuning time by method:**
```sql
SELECT 
    tuning_method,
    AVG((results->>'tuning_time')::float) as avg_time
FROM tuning_runs
WHERE status = 'completed'
GROUP BY tuning_method;
```

**Find long-running tuning runs:**
```sql
SELECT id, model_run_id, created_at
FROM tuning_runs
WHERE status = 'running'
AND created_at < NOW() - INTERVAL '1 hour';
```

**Best scores by model type:**
```sql
SELECT 
    mr.model_type,
    MAX((tr.results->>'best_score')::float) as best_score
FROM tuning_runs tr
JOIN model_runs mr ON tr.model_run_id = mr.id
WHERE tr.status = 'completed'
GROUP BY mr.model_type;
```

## Future Enhancements

Potential improvements to the TuningRun model:
- [ ] Add `started_at` and `completed_at` timestamps
- [ ] Add `n_iterations` field for tracking progress
- [ ] Add `config` JSONB field for storing tuning configuration
- [ ] Add `task_id` field for Celery task tracking
- [ ] Add `user_notes` field for user annotations
- [ ] Add `tags` array field for categorization
- [ ] Add `parent_tuning_run_id` for iterative tuning

## References

- **Model File:** `backend/app/models/tuning_run.py`
- **Migration File:** `backend/alembic/versions/5c1b6116de51_create_initial_tables.py`
- **API Endpoints:** `backend/app/api/v1/endpoints/tuning.py`
- **Celery Task:** `backend/app/tasks/tuning_tasks.py`
- **Schemas:** `backend/app/schemas/model.py`
