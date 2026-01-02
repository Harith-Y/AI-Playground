# Backend Tests

This directory contains comprehensive tests for the AI Playground backend API.

## Test Structure

```
tests/
├── conftest.py                      # Pytest fixtures and configuration
├── conftest_ml.py                   # ML-specific fixtures
├── test_datasets.py                 # Dataset endpoint tests
├── test_preprocessing_endpoints.py  # Preprocessing CRUD endpoint tests
├── test_model_training_api.py       # Model training endpoint tests
├── test_feature_importance_endpoint.py  # Feature importance tests
├── test_model_comparison.py         # Model comparison endpoint tests
├── test_tuning_api.py              # ✨ Hyperparameter tuning API tests (NEW)
├── test_evaluation_api.py          # ✨ Evaluation & metrics API tests (NEW)
├── test_cache.py                    # Redis caching tests
├── test_tuning_orchestration.py    # Tuning orchestration service tests
├── test_tuning_tasks.py             # Tuning Celery tasks tests
├── ml_engine/                       # ML engine component tests
│   ├── test_*.py                    # Various ML component tests
└── README.md                        # This file
```

## Running Tests

### All Tests

```bash
# Activate virtual environment
venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html
```

### Specific Test Files

```bash
# Run only dataset tests
pytest tests/test_datasets.py

# Run specific test class
pytest tests/test_datasets.py::TestDatasetUpload

# Run specific test function
pytest tests/test_datasets.py::TestDatasetUpload::test_upload_csv_success
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Coverage

After running tests with coverage, view the HTML report:

```bash
# Open coverage report in browser
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
xdg-open htmlcov/index.html  # Linux
```

## Test Fixtures

### Database Fixtures

- `db`: Fresh database session for each test
- `client`: FastAPI test client with database override

### File Fixtures

- `sample_csv_file`: Sample CSV file for testing
- `sample_excel_file`: Sample Excel file for testing
- `sample_json_file`: Sample JSON file for testing
- `large_csv_file`: Large CSV file for size limit testing (skipped by default)
- `invalid_file`: Invalid file type for error testing
- `temp_upload_dir`: Temporary directory for file uploads

## Writing New Tests

### Example Test

```python
def test_new_endpoint(client: TestClient, sample_csv_file: Path):
    """Test description"""
    # Arrange
    with open(sample_csv_file, "rb") as f:
        # Act
        response = client.post("/api/v1/endpoint", files={"file": f})

    # Assert
    assert response.status_code == 200
    assert "expected_key" in response.json()
```

### Best Practices

1. **Use fixtures** for common setup (database, files, etc.)
2. **Follow AAA pattern** (Arrange, Act, Assert)
3. **Test both success and failure cases**
4. **Use descriptive test names** that explain what is being tested
5. **Keep tests independent** - each test should run in isolation
6. **Clean up resources** - fixtures handle cleanup automatically
7. **Mock external dependencies** when appropriate

## Test Classes

### TestDatasetUpload

Tests for POST /api/v1/datasets/upload

- Upload success cases (CSV, Excel, JSON)
- Upload failure cases (invalid type, corrupted file)
- File size validation
- Missing values handling

### TestDatasetList

Tests for GET /api/v1/datasets/

- Empty list
- List with data
- Pagination

### TestDatasetGet

Tests for GET /api/v1/datasets/{id}

- Get existing dataset
- Get non-existent dataset

### TestDatasetPreview

Tests for GET /api/v1/datasets/{id}/preview

- Preview with default rows
- Preview with custom row limit
- Preview when file is deleted
- Column metadata validation

### TestDatasetStats

Tests for GET /api/v1/datasets/{id}/stats

- Statistics calculation
- Duplicate detection
- Missing values tracking

### TestDatasetDelete

Tests for DELETE /api/v1/datasets/{id}

- Successful deletion
- Delete non-existent dataset
- Delete when file already gone

### TestDatasetIntegration

Integration tests for complete workflows

- Full dataset lifecycle
- Multiple datasets handling

---

## Model Training API Tests

Comprehensive unit tests for the model training API endpoints covering all CRUD operations, validation, error handling, and integration workflows.

### Test Coverage by Endpoint

#### 1. GET /api/v1/models/available
**Class: `TestGetAvailableModels`**

Tests for listing and filtering available models:
- ✅ Get all models grouped by task type
- ✅ Filter by task type (classification, regression, clustering)
- ✅ Filter by category (tree_based, boosting, etc.)
- ✅ Search models by keyword
- ✅ Invalid task type error handling
- ✅ Invalid category error handling

#### 2. GET /api/v1/models/available/{model_id}
**Class: `TestGetModelDetails`**

Tests for retrieving model details:
- ✅ Get details of valid model
- ✅ Handle non-existent model (404)

#### 3. POST /api/v1/models/train
**Class: `TestTrainModel`**

Tests for initiating model training:
- ✅ Successful classification model training
- ✅ Successful clustering model training
- ✅ Missing target column for supervised learning (400)
- ✅ Invalid target column name (400)
- ✅ Invalid feature column names (400)
- ✅ Non-existent experiment (404)
- ✅ Non-existent dataset (404)
- ✅ Invalid model type (400)
- ✅ Training with all optional parameters

#### 4. GET /api/v1/models/train/{model_run_id}/status
**Class: `TestGetTrainingStatus`**

Tests for checking training status:
- ✅ Get status of completed model run
- ✅ Get status of pending model run
- ✅ Get status of failed model run with error details
- ✅ Handle non-existent model run (404)

#### 5. GET /api/v1/models/train/{model_run_id}/result
**Class: `TestGetTrainingResult`**

Tests for retrieving training results:
- ✅ Get results of completed model run
- ✅ Handle non-completed model run (400)
- ✅ Handle non-existent model run (404)

#### 6. DELETE /api/v1/models/train/{model_run_id}
**Class: `TestDeleteModelRun`**

Tests for deleting model runs:
- ✅ Delete completed model run with artifact
- ✅ Delete running model run (revokes Celery task)
- ✅ Delete model run without artifact
- ✅ Handle non-existent model run (404)
- ✅ Handle invalid UUID format (400)

#### 7. GET /api/v1/models/train/{model_run_id}/metrics
**Class: `TestGetModelMetrics`**

Tests for retrieving detailed model metrics:
- ✅ Get metrics for completed run
- ✅ Get metrics with feature importance
- ✅ Get metrics for different task types (classification, regression, clustering)
- ✅ Handle non-existent model run (404)
- ✅ Handle non-completed model run (400)
- ✅ Handle missing metrics (404)
- ✅ Handle invalid UUID format (400)
- ✅ Verify permission checks

#### 8. GET /api/v1/models/train/{model_run_id}/feature-importance
**Class: `TestGetFeatureImportance`**

Tests for retrieving feature importance:
- ✅ Get feature importance with default top_n
- ✅ Get feature importance with custom top_n
- ✅ Handle models without feature importance
- ✅ Handle non-existent model run (404)
- ✅ Handle non-completed model run (400)
- ✅ Verify correct ranking and sorting

#### 9. GET /api/v1/models/categories
**Class: `TestGetModelCategories`**

Tests for listing model categories:
- ✅ Get all categories with descriptions

#### 10. GET /api/v1/models/task-types
**Class: `TestGetTaskTypes`**

Tests for listing task types:
- ✅ Get all task types with model counts

### Integration Tests
**Class: `TestModelTrainingIntegration`**

End-to-end workflow tests:
- ✅ Complete workflow: train → status → result → delete
- ✅ Multiple model runs for same experiment
- ✅ Training with all parameters specified
- ✅ Metrics retrieval workflow
- ✅ Feature importance analysis workflow

### Mocking Strategy

#### Celery Tasks
All tests mock the `train_model.delay()` Celery task to avoid actual training:
```python
@patch('app.api.v1.endpoints.models.train_model')
def test_train_model(mock_train_model, ...):
    mock_task = Mock()
    mock_task.id = "test-task-id"
    mock_train_model.delay.return_value = mock_task
```

#### Model Serialization Service
Tests mock the storage service to avoid file I/O:
```python
@patch('app.api.v1.endpoints.models.get_model_serialization_service')
def test_delete_model(mock_get_service, ...):
    mock_service = Mock()
    mock_service.delete_model.return_value = True
    mock_get_service.return_value = mock_service
```

#### Celery AsyncResult
Tests mock Celery's AsyncResult for task revocation:
```python
@patch('app.api.v1.endpoints.models.AsyncResult')
def test_delete_running(mock_async_result, ...):
    mock_task = Mock()
    mock_task.state = "STARTED"
    mock_async_result.return_value = mock_task
```

### Test Data

#### Sample Dataset (Iris)
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
6.2,2.9,4.3,1.3,versicolor
...
```

#### Sample Training Request
```json
{
  "experiment_id": "uuid",
  "dataset_id": "uuid",
  "model_type": "random_forest_classifier",
  "target_column": "species",
  "feature_columns": ["sepal_length", "sepal_width"],
  "test_size": 0.2,
  "random_state": 42,
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

### Assertions

#### Response Status Codes
- 200: Successful GET/DELETE
- 202: Training initiated (async)
- 400: Bad request (validation error)
- 403: Forbidden (permission denied)
- 404: Not found
- 500: Internal server error

#### Response Structure
Tests verify:
- Required fields present
- Correct data types
- Expected values
- Error messages format

#### Database State
Tests verify:
- Records created/updated/deleted
- Relationships maintained
- Transaction rollback on errors

### Error Scenarios Tested

1. **Validation Errors (400)**
   - Missing required fields
   - Invalid field values
   - Non-existent columns
   - Invalid model types

2. **Not Found Errors (404)**
   - Non-existent experiment
   - Non-existent dataset
   - Non-existent model run

3. **Permission Errors (403)**
   - Accessing other user's resources

4. **State Errors (400)**
   - Getting results before completion
   - Invalid UUID format

### Running Model Training Tests

```bash
# Run all model training tests
pytest tests/test_model_training_api.py -v

# Run specific test class
pytest tests/test_model_training_api.py::TestTrainModel -v

# Run specific test
pytest tests/test_model_training_api.py::TestTrainModel::test_train_classification_model_success -v

# Run with coverage
pytest tests/test_model_training_api.py --cov=app.api.v1.endpoints.models --cov-report=html
```

### Coverage Goals

- **Line Coverage**: > 90%
- **Branch Coverage**: > 85%
- **Function Coverage**: 100%

---

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=app --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Database Issues

If you get database connection errors:

```bash
# Check database is running
# For tests, SQLite is used automatically (no setup needed)
```

### Import Errors

If you get import errors:

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure PYTHONPATH includes backend directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows
```

### File Permission Errors

If you get permission errors with temp files:

```bash
# Tests use temp directories which are cleaned up automatically
# If cleanup fails, manually remove test.db and __pycache__
```

## Performance

Current test suite performance:

- **Total tests**: 140+ tests
- **API Integration Tests**: 59 tests
  - Tuning API: 30 tests
  - Evaluation API: 29 tests
- **ML Engine Tests**: 50+ tests
- **Execution time**: ~30-45 seconds
- **Coverage**: 85%+ of API endpoints

## Recent Additions

### ✨ Tuning & Evaluation API Tests (59 tests)

Comprehensive integration tests for hyperparameter tuning and model evaluation endpoints.

**New Test Files:**

- `test_tuning_api.py` (30 tests) - POST /tune, GET /tune/{id}/status, GET /tune/{id}/results
- `test_evaluation_api.py` (29 tests) - GET /metrics, GET /feature-importance

**Coverage:**

- ✅ Successful operations (happy path)
- ✅ Error handling (404, 400, 422)
- ✅ Authorization & security
- ✅ Cache behavior (hit/miss/bypass)
- ✅ Edge cases & boundary conditions
- ✅ Performance benchmarks

See [TEST_COVERAGE_EVALUATION_TUNING.md](../../TEST_COVERAGE_EVALUATION_TUNING.md) for detailed documentation.

## Future Tests

Planned additions:

- [ ] Model deployment endpoint tests
- [ ] Code generation endpoint tests
- [ ] WebSocket connection tests
- [ ] Performance/load tests
- [ ] Security tests (SQL injection, XSS, etc.)
