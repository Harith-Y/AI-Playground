# Integration Tests Quick Reference

Quick commands and examples for running ML pipeline integration tests.

## Quick Commands

### Run All Tests

```bash
# Interactive mode
python tests/run_integration_tests.py

# All integration tests
pytest -v -m integration tests/integration/

# Fast tests only (skip slow)
pytest -v -m "integration and not slow" tests/integration/
```

### Run Specific Test Files

```bash
# End-to-end tests
pytest -v tests/integration/test_ml_pipeline_end_to_end.py

# ML engine tests
pytest -v tests/integration/test_ml_engine_integration.py

# Orchestration tests
pytest -v tests/integration/test_ml_pipeline_orchestration.py
```

### Run Specific Test Classes

```bash
# Classification end-to-end
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd

# Preprocessing tests
pytest -v tests/integration/test_ml_engine_integration.py::TestPreprocessingPipeline

# Tuning orchestration
pytest -v tests/integration/test_ml_pipeline_orchestration.py::TestTuningOrchestration
```

### Run Specific Tests

```bash
# Single test
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_classification_pipeline_end_to_end

# With output
pytest -v -s tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_classification_pipeline_end_to_end
```

### With Coverage

```bash
# Generate HTML coverage report
pytest -v -m integration tests/integration/ \
  --cov=app.ml_engine \
  --cov=app.api.v1.endpoints \
  --cov-report=html:htmlcov/integration

# View report
open htmlcov/integration/index.html  # macOS/Linux
start htmlcov/integration/index.html  # Windows
```

## Test Markers Reference

| Marker        | Description       | Example                 |
| ------------- | ----------------- | ----------------------- |
| `integration` | Integration test  | `pytest -m integration` |
| `slow`        | Slow test (>10s)  | `pytest -m "not slow"`  |
| `api`         | API endpoint test | `pytest -m api`         |

### Combining Markers

```bash
# Integration but not slow
pytest -m "integration and not slow"

# Integration or API
pytest -m "integration or api"

# All except slow
pytest -m "not slow"
```

## Common Test Scenarios

### 1. Classification Pipeline

```bash
# Full classification workflow
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_classification_pipeline_end_to_end

# With tuning
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineWithTuning::test_pipeline_with_grid_search
```

### 2. Regression Pipeline

```bash
# Full regression workflow
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_regression_pipeline_end_to_end
```

### 3. Preprocessing

```bash
# All preprocessing tests
pytest -v tests/integration/test_ml_engine_integration.py::TestPreprocessingPipeline

# Specific preprocessing test
pytest -v tests/integration/test_ml_engine_integration.py::TestPreprocessingPipeline::test_classification_preprocessing_pipeline
```

### 4. Model Training

```bash
# Classification model training
pytest -v tests/integration/test_ml_engine_integration.py::TestModelTraining::test_classification_model_training

# Regression model training
pytest -v tests/integration/test_ml_engine_integration.py::TestModelTraining::test_regression_model_training
```

### 5. Feature Engineering

```bash
# Feature selection
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineFeatureEngineering

# Feature importance
pytest -v tests/integration/test_ml_pipeline_orchestration.py::TestFeatureEngineeringOrchestration
```

### 6. Model Comparison

```bash
# Compare multiple models
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineModelComparison

# Orchestrated comparison
pytest -v tests/integration/test_ml_pipeline_orchestration.py::TestModelComparisonOrchestration
```

## Debugging Tests

### Run with Verbose Output

```bash
# Show print statements
pytest -v -s tests/integration/test_ml_pipeline_end_to_end.py

# Very verbose
pytest -vv tests/integration/test_ml_pipeline_end_to_end.py
```

### Debug on Failure

```bash
# Drop into debugger on failure
pytest -v --pdb tests/integration/

# Stop at first failure
pytest -v -x tests/integration/

# Show local variables on failure
pytest -v --showlocals tests/integration/
```

### Capture Warnings

```bash
# Show all warnings
pytest -v -W all tests/integration/

# Treat warnings as errors
pytest -v -W error tests/integration/
```

## Performance Testing

### Run Performance Tests Only

```bash
# Performance tests (marked as slow)
pytest -v -m slow tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelinePerformance
```

### Measure Test Duration

```bash
# Show slowest tests
pytest -v --durations=10 tests/integration/

# Show all test durations
pytest -v --durations=0 tests/integration/
```

## CI/CD Integration

### GitHub Actions

```bash
# Run fast tests on every push
pytest -m "integration and not slow" tests/integration/

# Run all tests on pull request
pytest -m integration tests/integration/

# Run with coverage
pytest -m integration tests/integration/ --cov=app --cov-report=xml
```

### Docker

```bash
# Run in Docker container
docker-compose run backend pytest -v -m integration tests/integration/
```

## Fixture Reference

### Common Fixtures

```python
def test_example(
    client: TestClient,           # API test client
    auth_headers: Dict,            # Authentication headers
    classification_dataset: Path,  # Classification CSV
    regression_dataset: Path,      # Regression CSV
    tmp_path: Path                 # Temporary directory
):
    pass
```

### Dataset Fixtures

- `classification_dataset` - Binary classification (500 rows, 4 features)
- `regression_dataset` - Regression (400 rows, 5 features)
- `multiclass_dataset` - Multiclass (3 classes)
- `imbalanced_dataset` - Imbalanced (90/10 split)
- `dataset_with_missing_values` - Contains NaN values
- `dataset_with_outliers` - Contains outliers
- `categorical_dataset` - Categorical features

## Environment Variables

```bash
# Test database URL (optional)
export TEST_DATABASE_URL="sqlite:///./test.db"

# Upload directory (auto-created temp dir)
export UPLOAD_DIR="/tmp/uploads"

# Logging level for tests
export LOG_LEVEL="DEBUG"
```

## Troubleshooting Quick Fixes

### Import Errors

```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest -v tests/integration/
```

### Missing Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio
```

### Database Issues

```bash
# Use in-memory SQLite (default)
unset TEST_DATABASE_URL

# Or specify test database
export TEST_DATABASE_URL="sqlite:///./test.db"
```

### Timeout Errors

```bash
# Skip slow tests
pytest -m "integration and not slow" tests/integration/

# Increase timeout (in test code)
@pytest.mark.timeout(300)  # 5 minutes
```

## Example Test Run Output

### Successful Run

```
tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_classification_pipeline_end_to_end PASSED [100%]

[Step 1] Uploading classification dataset...
✓ Dataset uploaded: abc-123
[Step 2] Creating preprocessing configuration...
✓ Preprocessing pipeline created: xyz-456
...
✓ End-to-end classification pipeline test completed successfully

====== 1 passed in 12.34s ======
```

### With Failures

```
tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_classification_pipeline_end_to_end FAILED [100%]

FAILED - AssertionError: Expected status 200, got 404

====== 1 failed in 5.67s ======
```

### Skipped Tests

```
tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_classification_pipeline_end_to_end SKIPPED [100%]

SKIPPED - Feature not implemented

====== 1 skipped in 0.12s ======
```

## Performance Benchmarks

| Test Suite    | Tests          | Duration (Fast) | Duration (Full) |
| ------------- | -------------- | --------------- | --------------- |
| End-to-End    | 8 classes      | ~45s            | ~5 min          |
| ML Engine     | 6 classes      | ~30s            | ~3 min          |
| Orchestration | 6 classes      | ~60s            | ~7 min          |
| **Total**     | **20 classes** | **~2 min**      | **~15 min**     |

_Note: Times are approximate and depend on hardware_

## VS Code Integration

### Run Tests in VS Code

1. Install Python extension
2. Open test file
3. Click "Run Test" above test function
4. Or use Testing view (flask icon in sidebar)

### Debug Tests in VS Code

1. Set breakpoint in test
2. Right-click test function
3. Select "Debug Test"

### VS Code launch.json

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Integration Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-v", "-m", "integration", "tests/integration/"],
      "console": "integratedTerminal"
    }
  ]
}
```

## Useful Pytest Plugins

```bash
# Install useful plugins
pip install pytest-xdist      # Parallel execution
pip install pytest-timeout    # Test timeouts
pip install pytest-benchmark  # Benchmarking
pip install pytest-html       # HTML reports

# Run tests in parallel (4 workers)
pytest -v -n 4 tests/integration/

# Generate HTML report
pytest -v tests/integration/ --html=report.html

# Benchmark tests
pytest -v tests/integration/ --benchmark-only
```

## Quick Tips

1. **Run fast tests during development:**

   ```bash
   pytest -m "integration and not slow" -v
   ```

2. **Focus on one test file:**

   ```bash
   pytest -v tests/integration/test_ml_pipeline_end_to_end.py
   ```

3. **Use watch mode (with pytest-watch):**

   ```bash
   pip install pytest-watch
   ptw tests/integration/
   ```

4. **Run specific pattern:**

   ```bash
   pytest -v -k "classification" tests/integration/
   ```

5. **Generate coverage badge:**
   ```bash
   pytest --cov=app --cov-report=term-missing tests/integration/
   ```
