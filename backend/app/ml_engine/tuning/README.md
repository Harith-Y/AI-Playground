# Hyperparameter Tuning Module

Comprehensive utilities for hyperparameter optimization and model selection.

## Features

- **Default Search Spaces**: Pre-configured hyperparameter grids for common models
- **Grid Search**: Exhaustive search over hyperparameter space
- **Random Search**: Randomized search with efficient sampling
- **Bayesian Optimization**: Advanced optimization using scikit-optimize
- **Cross-Validation**: Comprehensive CV utilities with multiple splitters
- **Model Comparison**: Compare multiple models with statistical analysis

## Modules

### 1. Search Spaces (`search_spaces.py`)

Default hyperparameter search spaces for various model types.

```python
from app.ml_engine.tuning import DEFAULT_SEARCH_SPACES, get_default_search_space

# Get default space for a model
space = get_default_search_space("random_forest_classifier")
# Returns: {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 10, 20, 30, 50], ...}

# Available model IDs:
# - logistic_regression
# - random_forest_classifier
# - svm_classifier
# - gradient_boosting_classifier
# - knn_classifier
# - linear_regression
# - ridge_regression
# - lasso_regression
# - random_forest_regressor
# - kmeans
# - dbscan
# - agglomerative_clustering
# - gaussian_mixture
```

### 2. Grid Search (`grid_search.py`)

Exhaustive search over all hyperparameter combinations.

```python
from app.ml_engine.tuning import run_grid_search
from sklearn.ensemble import RandomForestClassifier

# With explicit search space
result = run_grid_search(
    estimator=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    param_grid={
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5]
    },
    cv=5,
    scoring='accuracy'
)

# With default search space
result = run_grid_search(
    estimator=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    model_id='random_forest_classifier',
    cv=5
)

print(f"Best score: {result.best_score}")
print(f"Best params: {result.best_params}")
print(f"Top 3 combinations: {result.top(3)}")
```

**GridSearchResult attributes:**

- `best_params`: Best hyperparameter combination
- `best_score`: Best cross-validation score
- `scoring`: Scoring metric used
- `cv_folds`: Number of CV folds
- `n_candidates`: Total combinations tested
- `results`: All results sorted by score

### 3. Random Search (`random_search.py`)

Randomized search for efficient hyperparameter optimization.

```python
from app.ml_engine.tuning import run_random_search
from sklearn.ensemble import GradientBoostingClassifier

result = run_random_search(
    estimator=GradientBoostingClassifier(),
    X=X_train,
    y=y_train,
    param_distributions={
        'n_estimators': [50, 100, 150, 200, 250],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10, 20]
    },
    n_iter=50,  # Number of random samples
    cv=5,
    scoring='roc_auc'
)
```

**RandomSearchResult attributes:**

- Same as GridSearchResult, plus:
- `n_iter`: Number of random samples

### 4. Bayesian Optimization (`bayesian.py`)

Advanced optimization using Bayesian methods with scikit-optimize.

```python
from app.ml_engine.tuning import run_bayesian_search
from sklearn.ensemble import RandomForestRegressor

# Requires scikit-optimize: pip install scikit-optimize
result = run_bayesian_search(
    estimator=RandomForestRegressor(),
    X=X_train,
    y=y_train,
    search_spaces={
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8]
    },
    n_iter=32,  # Bayesian optimization iterations
    cv=5,
    scoring='neg_mean_squared_error',
    optimizer_kwargs={'base_estimator': 'GP'}  # Gaussian Process
)

print(f"Optimization method: {result.method}")  # 'bayesian' or 'random_fallback'
print(f"Best score: {result.best_score}")
print(f"Best params: {result.best_params}")
```

**Features:**

- Automatic conversion of parameter lists to skopt space objects (Real, Integer, Categorical)
- Graceful fallback to RandomizedSearchCV if scikit-optimize not installed
- Support for custom optimizer kwargs (base_estimator, acq_func, etc.)
- More efficient than random search for expensive evaluations

**BayesianSearchResult attributes:**

- Same as RandomSearchResult, plus:
- `method`: 'bayesian' or 'random_fallback' (indicates which method was used)

**Using skopt space objects directly:**

```python
from skopt.space import Real, Integer, Categorical

search_spaces = {
    'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 15),
    'criterion': Categorical(['gini', 'entropy'])
}

result = run_bayesian_search(
    estimator=estimator,
    X=X_train,
    y=y_train,
    search_spaces=search_spaces,
    n_iter=50
)
```

### 5. Cross-Validation (`cross_validation.py`)

Comprehensive cross-validation utilities.

```python
from app.ml_engine.tuning import (
    run_cross_validation,
    run_simple_cross_validation,
    create_cv_splitter,
    compare_models_cv
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Multi-metric cross-validation
result = run_cross_validation(
    estimator=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    cv=10,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True,
    n_jobs=-1
)

print(f"Mean accuracy: {result.mean_score:.3f} ± {result.std_score:.3f}")
print(f"95% CI: {result.confidence_interval(0.95)}")
print(f"Additional metrics: {result.additional_metrics}")

# Simple single-metric CV
scores = run_simple_cross_validation(
    estimator=LogisticRegression(),
    X=X_train,
    y=y_train,
    cv=5,
    scoring='accuracy'
)

# Custom CV splitter
splitter = create_cv_splitter(
    cv_type='stratified',
    n_splits=10,
    shuffle=True,
    random_state=42
)

# Compare multiple models
results = compare_models_cv(
    models={
        'rf': RandomForestClassifier(),
        'gb': GradientBoostingClassifier(),
        'lr': LogisticRegression()
    },
    X=X_train,
    y=y_train,
    cv=5,
    scoring='f1_weighted'
)

for name, result in results.items():
    print(f"{name}: {result.mean_score:.3f} ± {result.std_score:.3f}")
```

**CV Splitters:**

- `'kfold'`: Standard K-Fold
- `'stratified'`: Stratified K-Fold (preserves class distribution)
- `'group'`: Group K-Fold (requires groups parameter)
- `'timeseries'`: Time Series Split

**CrossValidationResult attributes:**

- `mean_score`: Mean test score across folds
- `std_score`: Standard deviation of test scores
- `median_score`: Median test score
- `min_score`: Minimum test score
- `max_score`: Maximum test score
- `scores`: All fold scores
- `fit_times`: Model fitting times per fold
- `score_times`: Scoring times per fold
- `train_scores`: Training scores (if `return_train_score=True`)
- `additional_metrics`: Dict of other metrics (if multiple scoring metrics)

## Usage Examples

### End-to-End Hyperparameter Tuning Workflow

```python
from sklearn.ensemble import RandomForestClassifier
from app.ml_engine.tuning import (
    run_grid_search,
    run_random_search,
    run_bayesian_search,
    run_cross_validation,
    compare_models_cv
)

# 1. Quick grid search with defaults
grid_result = run_grid_search(
    estimator=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    model_id='random_forest_classifier',
    cv=5
)

# 2. Refined random search
random_result = run_random_search(
    estimator=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    param_distributions={
        'n_estimators': list(range(100, 501, 50)),
        'max_depth': [10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    },
    n_iter=100,
    cv=5
)

# 3. Fine-tune with Bayesian optimization
bayesian_result = run_bayesian_search(
    estimator=RandomForestClassifier(),
    X=X_train,
    y=y_train,
    search_spaces={
        'n_estimators': list(range(150, 251, 10)),
        'max_depth': list(range(20, 41, 2)),
        'min_samples_split': [2, 3, 4, 5, 6]
    },
    n_iter=50,
    cv=10
)

# 4. Validate best model
best_model = RandomForestClassifier(**bayesian_result.best_params)
cv_result = run_cross_validation(
    estimator=best_model,
    X=X_train,
    y=y_train,
    cv=10,
    scoring=['accuracy', 'f1_weighted', 'roc_auc']
)

print(f"Final model performance: {cv_result.mean_score:.3f} ± {cv_result.std_score:.3f}")
print(f"95% CI: {cv_result.confidence_interval(0.95)}")
```

### Model Selection with Statistical Comparison

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from app.ml_engine.tuning import compare_models_cv

# Compare multiple algorithms
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=30),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5),
    'Logistic Regression': LogisticRegression(C=10, max_iter=1000),
    'SVM': SVC(C=10, kernel='rbf', probability=True)
}

results = compare_models_cv(
    models=models,
    X=X_train,
    y=y_train,
    cv=10,
    scoring='roc_auc',
    n_jobs=-1
)

# Analyze results
print("Model Comparison Results:")
print("-" * 60)
for name, result in sorted(results.items(), key=lambda x: x[1].mean_score, reverse=True):
    ci = result.confidence_interval(0.95)
    print(f"{name:20s}: {result.mean_score:.4f} ± {result.std_score:.4f}")
    print(f"{'':20s}  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"{'':20s}  Range: [{result.min_score:.4f}, {result.max_score:.4f}]")
    print()
```

## Best Practices

1. **Search Strategy Selection**:

   - Grid Search: Small hyperparameter space (<100 combinations)
   - Random Search: Medium space (100-1000 combinations)
   - Bayesian Optimization: Large/continuous space, expensive evaluations

2. **Cross-Validation**:

   - Use stratified CV for classification
   - Use time series split for temporal data
   - Increase folds (10+) for small datasets

3. **Scoring Metrics**:

   - Classification: accuracy, f1, roc_auc, precision, recall
   - Regression: neg_mean_squared_error, neg_mean_absolute_error, r2
   - Use multiple metrics for comprehensive evaluation

4. **Computational Efficiency**:

   - Set `n_jobs=-1` for parallel processing
   - Start with random search before Bayesian optimization
   - Use `return_train_score=False` to save time

5. **Bayesian Optimization**:
   - Use 20-50 iterations for initial exploration
   - Increase to 100+ for fine-tuning
   - Works best with continuous parameters
   - Install scikit-optimize for better performance: `pip install scikit-optimize`

## Dependencies

- **Required**: scikit-learn, numpy, scipy
- **Optional**: scikit-optimize (for Bayesian optimization)

Install optional dependencies:

```bash
pip install scikit-optimize
```

## Testing

Run tests for the tuning module:

```bash
pytest backend/tests/ml_engine/tuning/
```

## Integration with API

All tuning functions return structured result objects that can be easily serialized:

```python
# In API endpoint
from app.ml_engine.tuning import run_bayesian_search

result = run_bayesian_search(...)

return {
    "best_params": result.best_params,
    "best_score": result.best_score,
    "method": result.method,
    "top_results": result.top(10)
}
```
