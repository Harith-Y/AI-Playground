# Celery Tuning Tasks Quick Reference

Quick guide for using hyperparameter tuning tasks.

## Task 1: tune_hyperparameters

### Purpose

Asynchronous hyperparameter tuning with grid search, random search, or Bayesian optimization.

### Basic Usage

```python
from app.tasks.tuning_tasks import tune_hyperparameters

# Grid Search
task = tune_hyperparameters.delay(
    tuning_run_id=tuning_run_id,
    model_run_id=model_run_id,
    tuning_method='grid_search',
    param_grid={
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    cv_folds=5,
    scoring_metric='accuracy'
)

# Random Search
task = tune_hyperparameters.delay(
    tuning_run_id=tuning_run_id,
    model_run_id=model_run_id,
    tuning_method='random_search',
    param_grid={
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    n_iter=50,
    cv_folds=5
)

# Bayesian Optimization
task = tune_hyperparameters.delay(
    tuning_run_id=tuning_run_id,
    model_run_id=model_run_id,
    tuning_method='bayesian',
    param_grid={
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10]
    },
    n_iter=30,
    cv_folds=5
)
```

### Parameters

| Parameter        | Type | Default | Description                                |
| ---------------- | ---- | ------- | ------------------------------------------ |
| `tuning_run_id`  | str  | -       | UUID of TuningRun record                   |
| `model_run_id`   | str  | -       | UUID of ModelRun to tune                   |
| `tuning_method`  | str  | -       | 'grid_search', 'random_search', 'bayesian' |
| `param_grid`     | dict | None    | Parameter search space                     |
| `cv_folds`       | int  | 5       | Number of CV folds                         |
| `scoring_metric` | str  | None    | Metric to optimize                         |
| `n_iter`         | int  | 10      | Iterations (random/bayesian)               |
| `n_jobs`         | int  | -1      | Parallel jobs (-1 = all cores)             |
| `random_state`   | int  | 42      | Random seed                                |
| `user_id`        | str  | None    | User UUID for logging                      |

### Return Value

```python
{
    'tuning_run_id': str,
    'model_run_id': str,
    'tuning_method': str,
    'best_params': dict,
    'best_score': float,
    'total_combinations': int,
    'tuning_time': float,
    'total_execution_time': float,
    'status': 'completed',
    'method_used': str
}
```

### Progress Tracking

```python
from celery.result import AsyncResult

result = AsyncResult(task.id)

# Check status
print(result.state)  # PENDING, PROGRESS, SUCCESS, FAILURE

# Get progress info
if result.state == 'PROGRESS':
    info = result.info
    print(f"{info['current']}/{info['total']}: {info['status']}")
```

## Task 2: validate_model_cv

### Purpose

Cross-validation for model validation with multiple metrics.

### Basic Usage

```python
from app.tasks.tuning_tasks import validate_model_cv

# Single metric
task = validate_model_cv.delay(
    model_run_id=model_run_id,
    cv_folds=10,
    scoring_metrics=['accuracy']
)

# Multiple metrics
task = validate_model_cv.delay(
    model_run_id=model_run_id,
    cv_folds=10,
    scoring_metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

# Auto-detect metrics (based on task type)
task = validate_model_cv.delay(
    model_run_id=model_run_id,
    cv_folds=10,
    scoring_metrics=None  # Auto-select
)
```

### Parameters

| Parameter         | Type | Default | Description                  |
| ----------------- | ---- | ------- | ---------------------------- |
| `model_run_id`    | str  | -       | UUID of ModelRun to validate |
| `cv_folds`        | int  | 10      | Number of CV folds           |
| `scoring_metrics` | list | None    | Metrics to evaluate          |
| `n_jobs`          | int  | -1      | Parallel jobs                |
| `user_id`         | str  | None    | User UUID for logging        |

### Return Value

```python
{
    'model_run_id': str,
    'cv_folds': int,
    'mean_score': float,
    'std_score': float,
    'median_score': float,
    'min_score': float,
    'max_score': float,
    'scores': [float, ...],
    'confidence_interval_95': {'lower': float, 'upper': float},
    'confidence_interval_99': {'lower': float, 'upper': float},
    'mean_fit_time': float,
    'mean_score_time': float,
    'additional_metrics': {
        'precision': {'mean': float, 'std': float, 'scores': [...]},
        'recall': {...},
        ...
    },
    'mean_train_score': float,
    'train_scores': [float, ...],
    'cv_duration': float,
    'total_execution_time': float
}
```

## Default Search Spaces

When `param_grid=None`, automatic defaults are used:

### Classification Models

- **Logistic Regression**: C, penalty, solver, max_iter
- **Random Forest**: n_estimators, max_depth, min_samples_split, max_features
- **SVM**: C, kernel, gamma
- **Gradient Boosting**: n_estimators, learning_rate, max_depth, subsample
- **KNN**: n_neighbors, weights, p

### Regression Models

- **Linear Regression**: fit_intercept
- **Ridge**: alpha, solver, fit_intercept
- **Lasso**: alpha, max_iter, fit_intercept
- **Random Forest**: n_estimators, max_depth, min_samples_split

### Clustering Models

- **KMeans**: n_clusters, init, n_init
- **DBSCAN**: eps, min_samples, metric
- **Agglomerative**: n_clusters, linkage, metric
- **Gaussian Mixture**: n_components, covariance_type, max_iter

## Scoring Metrics

### Classification

- `accuracy`, `precision`, `recall`, `f1`, `f1_weighted`
- `roc_auc`, `roc_auc_ovr`, `roc_auc_ovo`
- `average_precision`

### Regression

- `r2`, `neg_mean_squared_error`, `neg_mean_absolute_error`
- `neg_root_mean_squared_error`, `neg_median_absolute_error`

### Clustering

- `adjusted_rand_score`, `normalized_mutual_info`
- `silhouette_score`, `calinski_harabasz_score`

## Error Handling

```python
from celery.result import AsyncResult

result = AsyncResult(task_id)

if result.failed():
    # Get error info
    exception = result.result
    print(f"Task failed: {exception}")

    # Check TuningRun status
    tuning_run = db.query(TuningRun).get(tuning_run_id)
    print(tuning_run.status)  # FAILED
    print(tuning_run.results['error'])
    print(tuning_run.results['error_type'])
```

## Monitoring

### Celery Flower

```bash
# Start Flower dashboard
celery -A app.celery_app flower --port=5555

# View at http://localhost:5555
```

### Task Logs

```python
from app.utils.logger import get_logger

logger = get_logger(task_id=task_id, user_id=user_id)

# Logs include:
# - tuning_start: Task started
# - tuning_complete: Task finished successfully
# - tuning_failed: Task failed with error
```

## Performance Tips

### 1. Parallel Execution

```python
# Use all CPU cores
task = tune_hyperparameters.delay(..., n_jobs=-1)

# Use specific number of cores
task = tune_hyperparameters.delay(..., n_jobs=4)
```

### 2. Efficient Search Strategies

```python
# Large space: Use random search
task = tune_hyperparameters.delay(
    tuning_method='random_search',
    n_iter=100,  # Sample 100 combinations
    ...
)

# Small space: Use grid search
task = tune_hyperparameters.delay(
    tuning_method='grid_search',
    param_grid={'C': [0.1, 1.0, 10.0]},  # Only 3 combinations
    ...
)

# Expensive models: Use Bayesian
task = tune_hyperparameters.delay(
    tuning_method='bayesian',
    n_iter=30,  # Efficient exploration
    ...
)
```

### 3. CV Folds

```python
# Small dataset: Fewer folds
task = tune_hyperparameters.delay(..., cv_folds=3)

# Large dataset: More folds for stability
task = tune_hyperparameters.delay(..., cv_folds=10)
```

## Complete Example

```python
from app.tasks.tuning_tasks import tune_hyperparameters, validate_model_cv
from app.models.tuning_run import TuningRun, TuningStatus
from celery.result import AsyncResult
import time

# 1. Create TuningRun record
tuning_run = TuningRun(
    model_run_id=model_run.id,
    tuning_method='bayesian',
    status=TuningStatus.PENDING
)
db.add(tuning_run)
db.commit()

# 2. Start tuning task
task = tune_hyperparameters.delay(
    tuning_run_id=str(tuning_run.id),
    model_run_id=str(model_run.id),
    tuning_method='bayesian',
    param_grid={
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    cv_folds=5,
    n_iter=30,
    user_id=str(current_user.id)
)

# 3. Monitor progress
result = AsyncResult(task.id)
while not result.ready():
    if result.state == 'PROGRESS':
        info = result.info
        print(f"Progress: {info['current']}/{info['total']} - {info['status']}")
    time.sleep(5)

# 4. Get results
if result.successful():
    tuning_results = result.result
    print(f"Best score: {tuning_results['best_score']:.4f}")
    print(f"Best params: {tuning_results['best_params']}")

    # 5. Validate with CV
    cv_task = validate_model_cv.delay(
        model_run_id=str(model_run.id),
        cv_folds=10,
        scoring_metrics=['accuracy', 'f1', 'roc_auc']
    )

    cv_result = cv_task.get(timeout=600)
    print(f"Validation score: {cv_result['mean_score']:.4f} ± {cv_result['std_score']:.4f}")
else:
    print(f"Tuning failed: {result.result}")
```

## Common Patterns

### Pattern 1: Progressive Refinement

```python
# 1. Quick grid search
grid_task = tune_hyperparameters.delay(
    tuning_method='grid_search',
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]}
)
grid_result = grid_task.get()

# 2. Refine with random search
best_n = grid_result['best_params']['n_estimators']
random_task = tune_hyperparameters.delay(
    tuning_method='random_search',
    param_grid={
        'n_estimators': range(best_n-20, best_n+20, 5),
        'max_depth': [3, 5, 7, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    n_iter=50
)

# 3. Fine-tune with Bayesian
bayesian_task = tune_hyperparameters.delay(
    tuning_method='bayesian',
    param_grid=refined_grid,
    n_iter=30
)
```

### Pattern 2: Multi-Model Comparison

```python
# Tune multiple models in parallel
tasks = []
for model_type in ['logistic_regression', 'random_forest', 'svm']:
    task = tune_hyperparameters.delay(
        model_run_id=model_runs[model_type].id,
        tuning_method='bayesian',
        param_grid=None,  # Use defaults
        n_iter=30
    )
    tasks.append((model_type, task))

# Wait for all to complete
results = {model: task.get() for model, task in tasks}

# Pick best model
best_model = max(results.items(), key=lambda x: x[1]['best_score'])
print(f"Best model: {best_model[0]}")
```

---

**Quick Tip**: Always set `random_state` for reproducible results! 🎲
