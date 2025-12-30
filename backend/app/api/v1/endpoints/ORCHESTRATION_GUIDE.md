# Tuning Orchestration Quick Reference

Quick guide for using hyperparameter tuning orchestration features.

## Progressive Search

### Start Progressive Search

```bash
curl -X POST "http://localhost:8000/api/v1/tuning-orchestration/progressive-search" \
  -H "Content-Type: application/json" \
  -d '{
    "model_run_id": "abc-123",
    "initial_param_grid": {
      "n_estimators": [50, 100, 200],
      "max_depth": [5, 10, 20]
    },
    "refinement_factor": 0.3,
    "cv_folds": 5,
    "scoring_metric": "accuracy",
    "n_iter_random": 50,
    "n_iter_bayesian": 30
  }'
```

### Workflow

```
Stage 1: Grid Search (RUNNING)
   ↓ (auto-trigger on completion)
Stage 2: Random Search (PENDING → RUNNING)
   ↓ (auto-trigger on completion)
Stage 3: Bayesian Optimization (PENDING → RUNNING)
   ↓
Complete! ✓
```

### Parameters

| Parameter            | Default | Description                  |
| -------------------- | ------- | ---------------------------- |
| `initial_param_grid` | None    | Initial parameter space      |
| `refinement_factor`  | 0.3     | How much to narrow (0.0-1.0) |
| `cv_folds`           | 5       | Cross-validation folds       |
| `scoring_metric`     | None    | Metric to optimize           |
| `n_iter_random`      | 50      | Random search iterations     |
| `n_iter_bayesian`    | 30      | Bayesian iterations          |

## Multi-Model Comparison

### Start Comparison

```bash
curl -X POST "http://localhost:8000/api/v1/tuning-orchestration/multi-model-comparison" \
  -H "Content-Type: application/json" \
  -d '{
    "model_run_ids": ["abc-123", "def-456", "ghi-789"],
    "tuning_method": "bayesian",
    "cv_folds": 5,
    "scoring_metric": "roc_auc",
    "n_iter": 30,
    "parallel": true
  }'
```

### Parameters

| Parameter        | Default  | Description                          |
| ---------------- | -------- | ------------------------------------ |
| `model_run_ids`  | -        | List of model IDs (min 2)            |
| `tuning_method`  | bayesian | grid_search, random_search, bayesian |
| `param_grids`    | None     | Optional model-specific grids        |
| `cv_folds`       | 5        | Cross-validation folds               |
| `scoring_metric` | None     | Metric to optimize                   |
| `n_iter`         | 30       | Iterations (random/bayesian)         |
| `parallel`       | True     | Parallel vs sequential               |

### Custom Param Grids

```json
{
  "model_run_ids": ["rf-id", "svm-id"],
  "param_grids": {
    "rf-id": {
      "n_estimators": [50, 100, 200],
      "max_depth": [5, 10, 20]
    },
    "svm-id": {
      "C": [0.1, 1.0, 10.0],
      "kernel": ["rbf", "linear"]
    }
  },
  "parallel": true
}
```

## Monitoring

### Check Status

```bash
curl "http://localhost:8000/api/v1/tuning-orchestration/orchestration/{orch_id}/status"
```

**Response**:

```json
{
  "overall_status": "RUNNING",
  "progress": {
    "completed": 1,
    "total": 3,
    "percentage": 33.33
  },
  "statuses": {
    "completed": 1,
    "running": 1,
    "pending": 1
  },
  "stages": [...]
}
```

### Get Best Model

```bash
curl "http://localhost:8000/api/v1/tuning-orchestration/orchestration/{orch_id}/best-model"
```

**Response**:

```json
{
  "best_model": {
    "model_run_id": "def-456",
    "model_type": "random_forest_classifier",
    "best_score": 0.96,
    "best_params": {...}
  },
  "all_models": [...]
}
```

## Python Client

### Progressive Search

```python
import httpx
import time

# 1. Start
response = httpx.post(
    "http://localhost:8000/api/v1/tuning-orchestration/progressive-search",
    json={
        "model_run_id": "abc-123",
        "initial_param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20]
        },
        "cv_folds": 5,
        "scoring_metric": "f1"
    }
)
orch_id = response.json()["orchestration_id"]

# 2. Monitor
while True:
    status = httpx.get(
        f"http://localhost:8000/api/v1/tuning-orchestration/orchestration/{orch_id}/status"
    ).json()

    print(f"Progress: {status['progress']['percentage']}%")

    if status['overall_status'] in ['COMPLETED', 'FAILED']:
        break

    time.sleep(30)

# 3. Get final results
final_stage = status['stages'][-1]
results = httpx.get(
    f"http://localhost:8000/api/v1/tuning/tune/{final_stage['tuning_run_id']}/results"
).json()

print(f"Best score: {results['best_score']}")
print(f"Best params: {results['best_params']}")
```

### Multi-Model Comparison

```python
import httpx

# 1. Start comparison
response = httpx.post(
    "http://localhost:8000/api/v1/tuning-orchestration/multi-model-comparison",
    json={
        "model_run_ids": ["abc-123", "def-456", "ghi-789"],
        "tuning_method": "bayesian",
        "cv_folds": 5,
        "n_iter": 30,
        "parallel": True
    }
)
orch_id = response.json()["orchestration_id"]

# 2. Wait for completion
import time
while True:
    status = httpx.get(
        f"http://localhost:8000/api/v1/tuning-orchestration/orchestration/{orch_id}/status"
    ).json()

    if status['overall_status'] == 'COMPLETED':
        break

    time.sleep(30)

# 3. Get best model
best = httpx.get(
    f"http://localhost:8000/api/v1/tuning-orchestration/orchestration/{orch_id}/best-model"
).json()

print(f"Best model: {best['best_model']['model_type']}")
print(f"Score: {best['best_model']['best_score']}")
```

## Status Values

| Status      | Description           |
| ----------- | --------------------- |
| `PENDING`   | Not yet started       |
| `RUNNING`   | Currently executing   |
| `COMPLETED` | Successfully finished |
| `FAILED`    | Error occurred        |

## Common Patterns

### Pattern 1: Quick Progressive Search

```python
# Fast settings for quick results
response = httpx.post("/progressive-search", json={
    "model_run_id": "abc-123",
    "initial_param_grid": {
        "n_estimators": [50, 100],  # Small grid
        "max_depth": [5, 10]
    },
    "cv_folds": 3,  # Fewer folds
    "n_iter_random": 20,  # Fewer iterations
    "n_iter_bayesian": 15
})
```

### Pattern 2: Thorough Progressive Search

```python
# Comprehensive settings for production
response = httpx.post("/progressive-search", json={
    "model_run_id": "abc-123",
    "initial_param_grid": {
        "n_estimators": [50, 100, 200, 300],  # Large grid
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "cv_folds": 10,  # More folds
    "n_iter_random": 100,  # More iterations
    "n_iter_bayesian": 50
})
```

### Pattern 3: Sequential Multi-Model (Resource Limited)

```python
# Sequential execution for limited resources
response = httpx.post("/multi-model-comparison", json={
    "model_run_ids": ["abc-123", "def-456"],
    "tuning_method": "random_search",
    "cv_folds": 3,
    "n_iter": 30,
    "parallel": False  # Sequential
})
```

### Pattern 4: Parallel Multi-Model (High Performance)

```python
# Parallel execution for speed
response = httpx.post("/multi-model-comparison", json={
    "model_run_ids": ["abc-123", "def-456", "ghi-789", "jkl-012"],
    "tuning_method": "bayesian",
    "cv_folds": 5,
    "n_iter": 30,
    "parallel": True  # Parallel
})
```

## Troubleshooting

### Issue: Stage not progressing

```python
# Check status
status = httpx.get(f"/orchestration/{orch_id}/status").json()

# If stuck, manually trigger next stage
if status['stages'][0]['status'] == 'COMPLETED':
    httpx.post("/trigger-next-stage", json={
        "orchestration_id": orch_id,
        "completed_tuning_run_id": status['stages'][0]['tuning_run_id']
    })
```

### Issue: Out of memory

**Solution**: Use sequential mode

```python
{
  "parallel": False  # One at a time
}
```

### Issue: Taking too long

**Solutions**:

- Reduce `cv_folds` to 3
- Decrease `n_iter` to 20
- Smaller `initial_param_grid`
- Higher `refinement_factor` (0.5)

## Performance Tips

### Fast Results

- cv_folds: 3
- n_iter_random: 20
- n_iter_bayesian: 15
- refinement_factor: 0.5

### Balanced

- cv_folds: 5
- n_iter_random: 50
- n_iter_bayesian: 30
- refinement_factor: 0.3

### Thorough

- cv_folds: 10
- n_iter_random: 100
- n_iter_bayesian: 50
- refinement_factor: 0.2

## Time Estimates

### Progressive Search

- **Fast**: 10-20 minutes
- **Balanced**: 30-65 minutes
- **Thorough**: 60-120 minutes

### Multi-Model Comparison (3 models)

- **Parallel**: 10-30 minutes (same as single)
- **Sequential**: 30-90 minutes (3x single)

## API Endpoints Summary

| Endpoint                         | Method | Purpose                    |
| -------------------------------- | ------ | -------------------------- |
| `/progressive-search`            | POST   | Start progressive workflow |
| `/multi-model-comparison`        | POST   | Start multi-model tuning   |
| `/orchestration/{id}/status`     | GET    | Check workflow status      |
| `/orchestration/{id}/best-model` | GET    | Get best model             |
| `/trigger-next-stage`            | POST   | Manual stage trigger       |

---

**Quick Tip**: Always use `parallel=True` for multi-model comparison unless memory limited! 🚀
