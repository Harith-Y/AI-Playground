# Hyperparameter Tuning - Quick Reference

## Endpoints

### 1. Initiate Tuning
```
POST /api/v1/tuning/tune
```

### 2. Get Status
```
GET /api/v1/tuning/tune/{tuning_run_id}/status
```

### 3. Get Results
```
GET /api/v1/tuning/tune/{tuning_run_id}/results?top_n=10
```

## Quick Examples

### Grid Search (Exhaustive)
```bash
curl -X POST http://localhost:8000/api/v1/tuning/tune \
  -H "Content-Type: application/json" \
  -d '{
    "model_run_id": "abc-123",
    "tuning_method": "grid_search",
    "param_grid": {
      "n_estimators": [50, 100, 200],
      "max_depth": [5, 10, null]
    },
    "cv_folds": 5
  }'
```

### Random Search (Faster)
```bash
curl -X POST http://localhost:8000/api/v1/tuning/tune \
  -H "Content-Type: application/json" \
  -d '{
    "model_run_id": "abc-123",
    "tuning_method": "random_search",
    "param_grid": {
      "n_estimators": [50, 100, 200, 300],
      "max_depth": [3, 5, 7, 9, null],
      "learning_rate": [0.01, 0.05, 0.1]
    },
    "n_iter": 20,
    "cv_folds": 5
  }'
```

### Bayesian Optimization (Most Efficient)
```bash
curl -X POST http://localhost:8000/api/v1/tuning/tune \
  -H "Content-Type: application/json" \
  -d '{
    "model_run_id": "abc-123",
    "tuning_method": "bayesian",
    "param_grid": {
      "n_estimators": [50, 100, 200],
      "max_depth": [5, 10, 15]
    },
    "n_iter": 30,
    "cv_folds": 5
  }'
```

## Tuning Methods Comparison

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Grid Search | Slow | Best | Small parameter spaces (< 50 combinations) |
| Random Search | Fast | Good | Large parameter spaces (> 100 combinations) |
| Bayesian | Medium | Best | Expensive models, continuous parameters |

## Common Parameter Grids

### Random Forest
```json
{
  "n_estimators": [50, 100, 200],
  "max_depth": [5, 10, 15, null],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 4]
}
```

### XGBoost
```json
{
  "n_estimators": [50, 100, 200],
  "max_depth": [3, 5, 7],
  "learning_rate": [0.01, 0.05, 0.1],
  "subsample": [0.6, 0.8, 1.0],
  "colsample_bytree": [0.6, 0.8, 1.0]
}
```

### Logistic Regression
```json
{
  "C": [0.001, 0.01, 0.1, 1, 10, 100],
  "penalty": ["l1", "l2"],
  "solver": ["liblinear", "saga"]
}
```

### SVM
```json
{
  "C": [0.1, 1, 10, 100],
  "kernel": ["linear", "rbf", "poly"],
  "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
}
```

## Scoring Metrics

### Classification
- `accuracy` (default)
- `precision`
- `recall`
- `f1`
- `roc_auc`

### Regression
- `r2` (default)
- `neg_mean_squared_error`
- `neg_mean_absolute_error`

## Response Structure

### Initiate Response
```json
{
  "tuning_run_id": "abc-123",
  "task_id": "xyz-789",
  "status": "PENDING",
  "message": "Hyperparameter tuning initiated successfully",
  "created_at": "2025-12-29T10:00:00Z"
}
```

### Results Response
```json
{
  "best_params": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "best_score": 0.95,
  "total_combinations": 36,
  "top_results": [
    {
      "rank": 1,
      "params": {"n_estimators": 100, "max_depth": 10},
      "mean_score": 0.95,
      "std_score": 0.02,
      "scores": [0.94, 0.96, 0.95, 0.94, 0.96]
    }
  ],
  "tuning_time": 120.5
}
```

## Common Errors

| Code | Reason | Solution |
|------|--------|----------|
| 400 | Model not completed | Wait for training to finish |
| 400 | Empty param_grid | Provide at least one parameter |
| 403 | Unauthorized | Check user permissions |
| 404 | Model run not found | Verify model_run_id |

## Best Practices

### 1. Start Small
```python
# First try: Small grid
{
  "n_estimators": [50, 100],
  "max_depth": [5, 10]
}

# Then expand: Larger grid
{
  "n_estimators": [50, 100, 200, 300],
  "max_depth": [3, 5, 7, 9, 11]
}
```

### 2. Use Appropriate CV Folds
- Small dataset (< 1000): 5-10 folds
- Medium dataset (1000-10000): 5 folds
- Large dataset (> 10000): 3 folds

### 3. Parallel Processing
Always use `"n_jobs": -1` to utilize all CPU cores.

### 4. Choose Right Method
- **< 50 combinations:** Grid Search
- **50-500 combinations:** Random Search
- **> 500 combinations:** Bayesian

## Frontend Integration

```typescript
// React/TypeScript example
const tunedModel = async (modelRunId: string) => {
  // Initiate tuning
  const response = await fetch('/api/v1/tuning/tune', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_run_id: modelRunId,
      tuning_method: 'grid_search',
      param_grid: {
        n_estimators: [50, 100, 200],
        max_depth: [5, 10, null]
      },
      cv_folds: 5
    })
  });
  
  const { tuning_run_id } = await response.json();
  
  // Poll for status
  const checkStatus = async () => {
    const statusRes = await fetch(
      `/api/v1/tuning/tune/${tuning_run_id}/status`
    );
    const status = await statusRes.json();
    
    if (status.status === 'SUCCESS') {
      // Get results
      const resultsRes = await fetch(
        `/api/v1/tuning/tune/${tuning_run_id}/results`
      );
      const results = await resultsRes.json();
      console.log('Best params:', results.best_params);
    } else if (status.status === 'PROGRESS') {
      // Continue polling
      setTimeout(checkStatus, 5000);
    }
  };
  
  checkStatus();
};
```

## Related Endpoints

- `POST /api/v1/models/train` - Train model
- `GET /api/v1/models/train/{model_run_id}/metrics` - Get metrics
- `GET /api/v1/models/train/{model_run_id}/feature-importance` - Feature importance

## Notes

- Tuning can take minutes to hours depending on parameter space size
- Results are cached in database for fast retrieval
- Use Random Search for initial exploration, Grid Search for fine-tuning
- Bayesian Optimization requires `scikit-optimize` package
