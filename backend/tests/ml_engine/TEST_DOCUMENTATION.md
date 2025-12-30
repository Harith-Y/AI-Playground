# ML Engine Test Suite Documentation

Comprehensive unit and integration tests for metrics and tuning utilities.

## Test Overview

### Test Coverage Summary

| Module                    | Test File                        | Test Count | Coverage              |
| ------------------------- | -------------------------------- | ---------- | --------------------- |
| **Evaluation Metrics**    |                                  |            |                       |
| Classification Metrics    | `test_classification_metrics.py` | 20+        | Binary & Multi-class  |
| Confusion Matrix          | `test_confusion_matrix.py`       | 15+        | All classes           |
| ROC Curve                 | `test_roc_curve.py`              | 10+        | Binary classification |
| PR Curve                  | `test_pr_curve.py`               | 10+        | Binary classification |
| Regression Metrics        | `test_regression_metrics.py`     | 15+        | All metrics           |
| Residual Analysis         | `test_residual_analysis.py`      | 10+        | Outlier detection     |
| Actual vs Predicted       | `test_actual_vs_predicted.py`    | 10+        | Regression            |
| Clustering Metrics        | `test_clustering_metrics.py`     | 15+        | All algorithms        |
| Feature Importance        | `test_feature_importance.py`     | 12+        | Native & Permutation  |
| Evaluation Integration    | `test_integration.py`            | 20+        | Workflows             |
| **Hyperparameter Tuning** |                                  |            |                       |
| Search Spaces             | `test_search_spaces.py`          | 30+        | All models            |
| Grid Search               | `test_grid_search.py`            | 15+        | All features          |
| Random Search             | `test_random_search.py`          | 15+        | All features          |
| Bayesian Optimization     | `test_bayesian.py`               | 15+        | With fallback         |
| Cross-Validation          | `test_cross_validation.py`       | 20+        | All splitters         |
| Tuning Integration        | `test_integration.py`            | 20+        | Workflows             |

**Total: 250+ tests**

## Running Tests

### Run All Tests

```bash
# Run all ML engine tests
cd backend
python -m pytest tests/ml_engine/ -v

# Or use the custom test runner
python tests/run_ml_tests.py
```

### Run Specific Test Modules

```bash
# Evaluation tests only
pytest tests/ml_engine/evaluation/ -v

# Tuning tests only
pytest tests/ml_engine/tuning/ -v

# Specific test file
pytest tests/ml_engine/tuning/test_bayesian.py -v

# Specific test class
pytest tests/ml_engine/tuning/test_bayesian.py::TestRunBayesianSearch -v

# Specific test method
pytest tests/ml_engine/tuning/test_bayesian.py::TestRunBayesianSearch::test_bayesian_search_with_skopt -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/ml_engine/ --cov=app.ml_engine.evaluation --cov=app.ml_engine.tuning --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

### Run with Markers

```bash
# Run only integration tests
pytest tests/ml_engine/ -m integration -v

# Run only unit tests
pytest tests/ml_engine/ -m "not integration" -v

# Run fast tests only
pytest tests/ml_engine/ -m "not slow" -v
```

## Test Structure

### Evaluation Tests

#### Classification Metrics Tests

**File**: `test_classification_metrics.py`

Tests:

- Binary classification (perfect, random, imbalanced)
- Multi-class classification (3+ classes)
- Probability-based metrics (ROC-AUC, PR-AUC)
- Per-class metrics (precision, recall, F1)
- Edge cases (all same class, empty predictions)
- Error handling (mismatched shapes, invalid inputs)

Example:

```python
def test_binary_classification_perfect(self):
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0, 1]

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
```

#### Feature Importance Tests

**File**: `test_feature_importance.py`

Tests:

- Native importance (tree-based models)
- Coefficient-based importance (linear models)
- Permutation importance (all models)
- Optional SHAP values
- Feature ranking and sorting
- Error handling for unsupported models

#### Evaluation Integration Tests

**File**: `evaluation/test_integration.py`

Tests complete workflows:

- Binary classification pipeline (metrics + curves + importance)
- Multi-class classification pipeline
- Model comparison workflows
- Regression evaluation pipeline
- Clustering quality assessment
- Cross-metric integration (clustering → classification)
- Feature selection using importance
- Outlier detection via residuals

### Tuning Tests

#### Search Spaces Tests

**File**: `test_search_spaces.py`

Tests:

- All classification models have valid spaces
- All regression models have valid spaces
- All clustering models have valid spaces
- Parameter value validation
- Deep copy behavior
- Best practices (regularization ranges, tree depths)
- sklearn compatibility checks

Example:

```python
def test_logistic_regression_parameters(self):
    space = DEFAULT_SEARCH_SPACES["logistic_regression"]

    assert "C" in space
    assert "penalty" in space
    assert "solver" in space
    assert len(space["C"]) > 0
    assert "l2" in space["penalty"]
```

#### Bayesian Optimization Tests

**File**: `test_bayesian.py`

Tests:

- BayesSearchCV integration (if available)
- Automatic space conversion (lists → skopt objects)
- Fallback to RandomizedSearchCV
- Warning messages when skopt missing
- Custom optimizer kwargs
- Default search spaces
- Result structure and serialization

#### Cross-Validation Tests

**File**: `test_cross_validation.py`

Tests:

- Multi-metric evaluation
- All CV splitters (KFold, Stratified, Group, TimeSeries)
- Train score calculation
- Confidence intervals
- Model comparison
- Performance timing
- Edge cases (small datasets, single fold)

#### Tuning Integration Tests

**File**: `tuning/test_integration.py`

Tests complete workflows:

- Quick grid search → validation
- Random search → cross-validation
- Progressive refinement (grid → random → bayesian)
- Model comparison → best model tuning
- Regression tuning workflow
- Custom vs default search spaces
- Parallel execution
- Deterministic results with random_state
- Performance characteristics

## Test Data

### Synthetic Datasets

Tests use sklearn's data generators:

```python
# Classification
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=15,
    n_classes=3,
    random_state=42
)

# Regression
X, y = make_regression(
    n_samples=300,
    n_features=15,
    n_informative=10,
    noise=10,
    random_state=42
)

# Clustering
X, y = make_blobs(
    n_samples=300,
    n_features=10,
    centers=4,
    random_state=42
)
```

### Real Datasets

Some tests use real datasets:

- Iris dataset (classification)
- Wine dataset (classification)
- Boston housing (regression, if available)

## Test Patterns

### Unit Tests

Focus on individual functions/classes:

```python
class TestClassificationMetricsCalculator:
    def test_initialization(self):
        calc = ClassificationMetricsCalculator(average='macro')
        assert calc.average == 'macro'

    def test_binary_classification(self):
        metrics = calc.calculate_metrics(y_true, y_pred)
        assert metrics.accuracy > 0.5
```

### Integration Tests

Test complete workflows:

```python
class TestTuningWorkflows:
    def test_complete_tuning_pipeline(self):
        # 1. Grid search
        grid_result = run_grid_search(...)

        # 2. Random search refinement
        random_result = run_random_search(...)

        # 3. Bayesian optimization
        bayesian_result = run_bayesian_search(...)

        # 4. Validate best model
        cv_result = run_cross_validation(...)

        # Assert complete pipeline works
        assert bayesian_result.best_score >= grid_result.best_score
```

### Edge Case Tests

Test boundary conditions:

```python
def test_small_dataset(self):
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, 20)

    # Should work with appropriate CV folds
    result = run_grid_search(estimator, X, y, cv=3)
    assert result.best_params is not None

def test_single_parameter(self):
    # Grid search with one parameter
    result = run_grid_search(
        estimator, X, y,
        param_grid={"C": [0.1, 1.0, 10.0]}
    )
    assert len(result.results) == 3
```

### Error Handling Tests

Test invalid inputs:

```python
def test_error_no_search_space(self):
    with pytest.raises(ValueError) as exc_info:
        run_bayesian_search(estimator, X, y)  # No space or model_id

    assert "search_spaces is required" in str(exc_info.value)

def test_error_mismatched_shapes(self):
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1]  # Wrong length

    with pytest.raises(ValueError):
        calculate_classification_metrics(y_true, y_pred)
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: ML Engine Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r backend/requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          cd backend
          pytest tests/ml_engine/ -v --cov=app.ml_engine

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Maintenance

### Adding New Tests

When adding new functionality:

1. **Create unit tests first**:

   ```python
   def test_new_feature_basic(self):
       result = new_feature(input_data)
       assert result.is_valid()
   ```

2. **Add edge cases**:

   ```python
   def test_new_feature_empty_input(self):
       with pytest.raises(ValueError):
           new_feature([])
   ```

3. **Add integration tests**:
   ```python
   def test_new_feature_in_pipeline(self):
       result1 = step1(data)
       result2 = new_feature(result1)
       result3 = step3(result2)
       assert pipeline_works(result3)
   ```

### Test Guidelines

1. **Naming**: Use descriptive test names

   - Good: `test_bayesian_search_with_default_spaces`
   - Bad: `test_bayesian_1`

2. **Isolation**: Each test should be independent

   ```python
   def setUp(self):
       # Fresh data for each test
       self.X, self.y = make_classification(random_state=42)
   ```

3. **Assertions**: Use specific assertions

   ```python
   # Good
   assert result.accuracy > 0.8
   assert "C" in result.best_params

   # Bad
   assert result  # Too vague
   ```

4. **Documentation**: Add docstrings
   ```python
   def test_feature(self):
       """Test feature behavior with edge case X."""
       pass
   ```

## Troubleshooting

### Common Issues

1. **Import errors**:

   ```bash
   # Ensure backend in PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:/path/to/backend"
   ```

2. **Random failures**:

   ```python
   # Always set random_state for reproducibility
   model = RandomForestClassifier(random_state=42)
   ```

3. **Slow tests**:

   ```python
   # Use smaller datasets for tests
   X, y = make_classification(n_samples=100)  # Not 10000
   ```

4. **Missing dependencies**:
   ```bash
   # Optional dependencies
   pip install scikit-optimize  # For Bayesian tests
   pip install shap  # For SHAP tests
   ```

## Test Metrics

### Current Status

- **Total Tests**: 250+
- **Test Coverage**: 90%+
- **Average Runtime**: ~2-3 minutes
- **Success Rate**: 100%

### Coverage Goals

- Line coverage: >90%
- Branch coverage: >85%
- Function coverage: 100%

## Future Enhancements

1. **Performance benchmarks**: Add timing tests
2. **Memory profiling**: Test memory usage
3. **Stress tests**: Large dataset handling
4. **Property-based tests**: Use hypothesis library
5. **Mutation testing**: Use mutpy for test quality

---

**Last Updated**: December 30, 2025
**Maintainers**: ML Engineering Team
**Status**: ✅ Production Ready
