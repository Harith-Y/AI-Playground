# ML Pipeline Integration Tests

Comprehensive integration tests for the complete ML pipeline, covering end-to-end workflows, ML engine components, and orchestration.

## Overview

This test suite validates the entire ML workflow from data upload through preprocessing, training, evaluation, and prediction. Tests are organized into three main categories:

1. **End-to-End Pipeline Tests** - Complete workflows from data to predictions
2. **ML Engine Integration Tests** - Core ML components (preprocessing, training, evaluation)
3. **Pipeline Orchestration Tests** - Complex workflows with tuning, feature engineering, and model comparison

## Test Structure

```
tests/integration/
├── test_ml_pipeline_end_to_end.py          # Full pipeline workflows
├── test_ml_engine_integration.py           # ML engine component tests
├── test_ml_pipeline_orchestration.py       # Orchestration and advanced workflows
├── run_integration_tests.py                # Test runner script
└── README.md                               # This file
```

## Test Categories

### 1. End-to-End Pipeline Tests (`test_ml_pipeline_end_to_end.py`)

Tests complete ML workflows from start to finish:

- **TestMLPipelineEndToEnd**

  - `test_classification_pipeline_end_to_end` - Full classification workflow
  - `test_regression_pipeline_end_to_end` - Full regression workflow

- **TestMLPipelineWithTuning**

  - `test_pipeline_with_grid_search` - Pipeline with grid search tuning
  - `test_pipeline_with_bayesian_optimization` - Pipeline with Bayesian optimization

- **TestMLPipelineFeatureEngineering**

  - `test_pipeline_with_feature_selection` - Various feature selection methods
  - `test_pipeline_with_feature_importance` - Feature importance analysis

- **TestMLPipelineDataPreprocessing**

  - `test_preprocessing_with_missing_values` - Missing value handling
  - `test_preprocessing_with_outliers` - Outlier detection
  - `test_preprocessing_with_encoding` - Categorical encoding

- **TestMLPipelineModelComparison**

  - `test_compare_multiple_models` - Train and compare multiple algorithms

- **TestMLPipelineExperimentTracking**

  - `test_create_and_track_experiment` - Experiment lifecycle management

- **TestMLPipelineCodeGeneration**

  - `test_generate_deployment_code` - Code generation from trained models

- **TestMLPipelinePerformance** (slow)
  - `test_pipeline_with_large_dataset` - Performance testing with large data

### 2. ML Engine Integration Tests (`test_ml_engine_integration.py`)

Tests core ML engine components:

- **TestPreprocessingPipeline**

  - `test_classification_preprocessing_pipeline` - Classification preprocessing
  - `test_regression_preprocessing_pipeline` - Regression preprocessing
  - `test_pipeline_serialization` - Save/load pipelines
  - `test_pipeline_inverse_transform` - Inverse transformations

- **TestFeatureSelection**

  - `test_variance_threshold_selection` - Variance-based selection
  - `test_correlation_selection` - Correlation-based selection

- **TestModelTraining**

  - `test_classification_model_training` - Train classification models
  - `test_regression_model_training` - Train regression models
  - `test_model_serialization` - Save/load models

- **TestModelEvaluation**

  - `test_classification_metrics` - Classification metrics computation
  - `test_regression_metrics` - Regression metrics computation
  - `test_cross_validation_evaluation` - CV evaluation

- **TestCompleteMLWorkflow**

  - `test_classification_workflow` - Complete classification workflow
  - `test_regression_workflow` - Complete regression workflow

- **TestMLPipelineStressTests** (slow)
  - `test_pipeline_with_many_steps` - Pipeline with many transformations
  - `test_high_dimensional_data` - High-dimensional dataset handling

### 3. Pipeline Orchestration Tests (`test_ml_pipeline_orchestration.py`)

Tests complex orchestration workflows:

- **TestTuningOrchestration**

  - `test_grid_search_orchestration` - Grid search orchestration
  - `test_random_search_orchestration` - Random search orchestration

- **TestFeatureEngineeringOrchestration**

  - `test_automated_feature_engineering` - Automated feature engineering
  - `test_feature_importance_ranking` - Feature importance ranking

- **TestModelComparisonOrchestration**

  - `test_automated_model_comparison` - Automated model comparison
  - `test_model_comparison_with_cv` - Model comparison with CV

- **TestPipelineExportOrchestration**

  - `test_export_complete_pipeline` - Export pipeline as code

- **TestExperimentOrchestration**

  - `test_full_experiment_lifecycle` - Complete experiment lifecycle

- **TestDataPipelineOrchestration** (slow)
  - `test_data_quality_pipeline` - Data quality checking
  - `test_data_transformation_pipeline` - Complex data transformations

## Running Tests

### Using the Test Runner Script

The easiest way to run tests is using the interactive test runner:

```bash
python tests/run_integration_tests.py
```

This provides an interactive menu to:

- Run all integration tests
- Run specific test suites
- Run specific test classes
- Run single tests
- Run fast tests (skip slow tests)

### Using Pytest Directly

Run all integration tests:

```bash
pytest -v -m integration tests/integration/
```

Run fast tests only (skip slow tests):

```bash
pytest -v -m "integration and not slow" tests/integration/
```

Run specific test file:

```bash
pytest -v tests/integration/test_ml_pipeline_end_to_end.py
```

Run specific test class:

```bash
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd
```

Run specific test:

```bash
pytest -v tests/integration/test_ml_pipeline_end_to_end.py::TestMLPipelineEndToEnd::test_classification_pipeline_end_to_end
```

### With Coverage

Generate coverage report:

```bash
pytest -v -m integration tests/integration/ --cov=app.ml_engine --cov=app.api.v1.endpoints --cov-report=html
```

View coverage report:

```bash
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

## Test Markers

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (large datasets, long training times)
- `@pytest.mark.api` - API endpoint tests

## Fixtures

Common fixtures available for all tests (defined in `conftest.py`):

### Dataset Fixtures

- `classification_dataset` - Binary classification dataset (500 samples)
- `regression_dataset` - Regression dataset (400 samples)
- `multiclass_dataset` - Multiclass classification dataset (3 classes)
- `imbalanced_dataset` - Imbalanced classification dataset (90/10 split)
- `time_series_dataset` - Time series dataset (365 days)
- `dataset_with_missing_values` - Dataset with various missing values
- `dataset_with_outliers` - Dataset with outliers
- `categorical_dataset` - Dataset with categorical features

### Authentication Fixtures

- `auth_headers` - Authentication headers for API requests
- `client` - TestClient with database override
- `db` - Database session

### Utility Fixtures

- `tmp_path` - Temporary directory for file operations

## Prerequisites

### Required Packages

```bash
pip install pytest pytest-cov pytest-asyncio
pip install pandas numpy scikit-learn
pip install fastapi httpx sqlalchemy
```

### Environment Setup

1. Set up test database (optional, uses in-memory by default):

```bash
export TEST_DATABASE_URL="sqlite:///./test.db"
```

2. Ensure ML dependencies are installed:

```bash
pip install -r requirements.txt
```

## Test Data

Tests automatically generate synthetic datasets using:

- `sklearn.datasets.make_classification` for classification
- `sklearn.datasets.make_regression` for regression
- Custom generators for specific scenarios

No external data files are required.

## Expected Behavior

### Successful Tests

- All fixtures properly initialized
- Datasets generated without errors
- API endpoints respond correctly
- Models train successfully
- Metrics computed accurately
- Pipelines serialize/deserialize correctly

### Skipped Tests

Some tests may be skipped if:

- Endpoint not implemented (e.g., prediction API)
- Feature not available (e.g., Bayesian optimization)
- Training timeout (long-running models)

Skipped tests are marked with `pytest.skip()` and don't count as failures.

## Troubleshooting

### Common Issues

**Test failures due to missing dependencies:**

```bash
pip install -r requirements.txt
```

**Database connection errors:**

- Check TEST_DATABASE_URL environment variable
- Ensure database is accessible
- Tests use in-memory SQLite by default

**Timeout errors:**

- Increase timeout in test configuration
- Run with fewer parallel jobs
- Skip slow tests: `-m "integration and not slow"`

**Import errors:**

- Ensure you're running from project root
- Check PYTHONPATH includes project directory
- Verify all dependencies installed

### Debug Mode

Run tests with full output:

```bash
pytest -v -s tests/integration/test_ml_pipeline_end_to_end.py
```

Run with debugging on failure:

```bash
pytest -v --pdb tests/integration/
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: "3.9"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run integration tests
        run: |
          pytest -v -m integration tests/integration/ --cov=app
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Performance Benchmarks

Expected test execution times (approximate):

- Fast integration tests: ~30 seconds
- Full integration suite: ~3-5 minutes
- With slow tests: ~10-15 minutes

Actual times depend on:

- Hardware (CPU, memory)
- Model complexity
- Dataset sizes
- Parallel execution

## Contributing

When adding new integration tests:

1. **Place in appropriate file:**

   - End-to-end workflows → `test_ml_pipeline_end_to_end.py`
   - Component tests → `test_ml_engine_integration.py`
   - Orchestration → `test_ml_pipeline_orchestration.py`

2. **Use descriptive names:**

   ```python
   def test_classification_with_feature_selection_and_tuning(self, ...):
   ```

3. **Add appropriate markers:**

   ```python
   @pytest.mark.integration
   @pytest.mark.slow  # If test takes > 10 seconds
   ```

4. **Document test purpose:**

   ```python
   def test_example(self):
       """
       Test classification pipeline with missing values.

       Steps:
       1. Upload dataset with missing values
       2. Apply mean imputation
       3. Train model
       4. Verify accuracy > 0.7
       """
   ```

5. **Use fixtures:**

   - Use existing fixtures when possible
   - Create new fixtures in `conftest.py` if reusable
   - Keep test data generation consistent

6. **Handle failures gracefully:**
   ```python
   if response.status_code != 200:
       pytest.skip("Feature not implemented")
   ```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing FastAPI Applications](https://fastapi.tiangolo.com/tutorial/testing/)
- [Scikit-learn Testing](https://scikit-learn.org/stable/developers/contributing.html#testing)

## Questions or Issues?

- Open an issue on GitHub
- Check existing test examples
- Review fixture documentation in `conftest.py`
