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
