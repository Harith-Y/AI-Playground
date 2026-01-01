# End-to-End Test Guide

Complete guide for running and understanding end-to-end tests in AI-Playground.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Coverage](#test-coverage)
- [Running Tests](#running-tests)
- [Test Scenarios](#test-scenarios)
- [Understanding Results](#understanding-results)
- [Troubleshooting](#troubleshooting)

## Overview

End-to-end (E2E) tests validate the complete user workflow from dataset upload through model training, evaluation, and code generation. These tests ensure all components work together correctly.

### What E2E Tests Cover

✅ **Complete Workflows**
- Upload → Train → Evaluate → Code Generation
- Data preprocessing pipelines
- Model training and evaluation
- Hyperparameter tuning
- Model comparison
- Experiment tracking

✅ **Real-World Scenarios**
- Classification tasks
- Regression tasks
- Multiple model comparison
- Hyperparameter optimization
- Error handling

✅ **Integration Points**
- API endpoints
- Database operations
- File storage
- Async task processing
- Code generation

## Test Coverage

### Test Classes

#### 1. TestCompleteWorkflowClassification

Tests complete classification workflow with all steps:

**Steps Tested:**
1. Upload classification dataset
2. Preview and explore data
3. Create preprocessing steps (imputation, scaling)
4. Train Random Forest model
5. Evaluate model (accuracy, precision, recall, F1)
6. Generate production code (Python, Jupyter)
7. Export experiment configuration

**Expected Duration:** ~60 seconds

#### 2. TestCompleteWorkflowRegression

Tests complete regression workflow:

**Steps Tested:**
1. Upload regression dataset
2. Train Linear Regression model
3. Evaluate model (MSE, RMSE, MAE, R²)
4. Generate production code

**Expected Duration:** ~30 seconds

#### 3. TestCompleteWorkflowWithTuning

Tests workflow with hyperparameter tuning:

**Steps Tested:**
1. Upload dataset
2. Train baseline model
3. Perform grid search hyperparameter tuning
4. Compare baseline vs optimized model
5. Generate code with best parameters

**Expected Duration:** ~90 seconds

#### 4. TestCompleteWorkflowMultipleModels

Tests model comparison workflow:

**Steps Tested:**
1. Upload dataset
2. Train multiple models (Logistic Regression, Random Forest, Gradient Boosting)
3. Compare model performance
4. Select best model
5. Generate code for best model

**Expected Duration:** ~120 seconds

#### 5. TestCompleteWorkflowErrorHandling

Tests error handling and edge cases:

**Scenarios Tested:**
- Invalid data (all missing values)
- Missing dataset
- Malformed requests
- Timeout handling

**Expected Duration:** ~20 seconds

## Running Tests

### Run All E2E Tests

```bash
# From backend directory
pytest tests/integration/test_complete_workflow_e2e.py -v -s

# With markers
pytest -m "e2e" -v -s

# With integration marker
pytest -m "integration and e2e" -v -s
```

### Run Specific Test Class

```bash
# Classification workflow
pytest tests/integration/test_complete_workflow_e2e.py::TestCompleteWorkflowClassification -v -s

# Regression workflow
pytest tests/integration/test_complete_workflow_e2e.py::TestCompleteWorkflowRegression -v -s

# With tuning
pytest tests/integration/test_complete_workflow_e2e.py::TestCompleteWorkflowWithTuning -v -s

# Model comparison
pytest tests/integration/test_complete_workflow_e2e.py::TestCompleteWorkflowMultipleModels -v -s

# Error handling
pytest tests/integration/test_complete_workflow_e2e.py::TestCompleteWorkflowErrorHandling -v -s
```

### Run Specific Test Method

```bash
# Full classification workflow
pytest tests/integration/test_complete_workflow_e2e.py::TestCompleteWorkflowClassification::test_full_classification_workflow -v -s

# Full regression workflow
pytest tests/integration/test_complete_workflow_e2e.py::TestCompleteWorkflowRegression::test_full_regression_workflow -v -s
```

### Run with Coverage

```bash
pytest tests/integration/test_complete_workflow_e2e.py --cov=app --cov-report=html -v -s
```

### Run with Detailed Output

```bash
# Show print statements
pytest tests/integration/test_complete_workflow_e2e.py -v -s

# Show full traceback
pytest tests/integration/test_complete_workflow_e2e.py -v -s --tb=long

# Stop on first failure
pytest tests/integration/test_complete_workflow_e2e.py -v -s -x
```

## Test Scenarios

### Scenario 1: Classification Workflow

**Objective:** Test complete classification pipeline

**Dataset:** 200 samples, 6 features (age, income, credit_score, years_employed, debt_ratio, target)

**Steps:**
1. Upload CSV file
2. Get dataset preview (5 rows)
3. Get dataset statistics
4. Create imputation step (mean strategy)
5. Create scaling step (standard scaler)
6. Train Random Forest (50 estimators, max_depth=10)
7. Wait for training completion (max 60s)
8. Get evaluation metrics
9. Get feature importance
10. Generate Python code
11. Generate Jupyter notebook
12. Export experiment configuration

**Success Criteria:**
- All API calls return 200/201/202
- Model training completes successfully
- Metrics include accuracy, precision, recall, F1
- Code generation produces valid Python code
- Experiment configuration exports successfully

### Scenario 2: Regression Workflow

**Objective:** Test complete regression pipeline

**Dataset:** 200 samples, 4 features + target (linear relationship)

**Steps:**
1. Upload CSV file
2. Train Linear Regression model
3. Wait for training completion
4. Get evaluation metrics (MSE, RMSE, MAE, R²)
5. Generate Python code

**Success Criteria:**
- Model trains successfully
- Regression metrics are calculated
- R² score is reasonable (> 0.5 for synthetic data)
- Code generation works

### Scenario 3: Hyperparameter Tuning

**Objective:** Test hyperparameter optimization

**Dataset:** Classification dataset

**Steps:**
1. Upload dataset
2. Train baseline model (n_estimators=10, max_depth=3)
3. Start grid search (n_estimators=[10,50], max_depth=[3,5])
4. Wait for tuning completion
5. Get best parameters and score
6. Generate code with optimized parameters

**Success Criteria:**
- Baseline model trains
- Grid search completes (4 combinations)
- Best parameters are returned
- Best score >= baseline score

### Scenario 4: Model Comparison

**Objective:** Compare multiple models

**Dataset:** Classification dataset

**Steps:**
1. Upload dataset
2. Train Logistic Regression
3. Train Random Forest
4. Train Gradient Boosting
5. Wait for all models to complete
6. Compare accuracy scores
7. Select best model
8. Generate code for best model

**Success Criteria:**
- All 3 models train successfully
- Metrics are retrieved for all models
- Best model is identified
- Code generation works

### Scenario 5: Error Handling

**Objective:** Test graceful error handling

**Test Cases:**
- Invalid data (all missing values)
- Non-existent dataset ID
- Malformed requests

**Success Criteria:**
- Appropriate HTTP status codes (400, 404, 422)
- Error messages are descriptive
- No server crashes
- Graceful degradation

## Understanding Results

### Successful Test Output

```
================================================================================
COMPLETE CLASSIFICATION WORKFLOW TEST
================================================================================

[STEP 1] Uploading classification dataset...
✓ Dataset uploaded successfully
  - Dataset ID: 123e4567-e89b-12d3-a456-426614174000
  - Rows: 200
  - Columns: 6

[STEP 2] Previewing and exploring dataset...
✓ Dataset preview retrieved
  - Preview rows: 5
✓ Dataset statistics retrieved
  - Missing values: {}
  - Duplicates: 0

[STEP 3] Creating preprocessing pipeline...
✓ Imputation step created
✓ Scaling step created
  - Total preprocessing steps: 2

[STEP 4] Training classification model...
✓ Model training initiated
  - Model Run ID: 456e7890-e89b-12d3-a456-426614174001
  - Model Type: random_forest

  Waiting for training to complete...
✓ Training completed successfully

[STEP 5] Evaluating model performance...
✓ Model evaluation results retrieved
  - Test Accuracy: 0.85
  - Test Precision: 0.83
  - Test Recall: 0.87
  - Test F1 Score: 0.85
✓ Feature importance retrieved
  - Features analyzed: 5

[STEP 6] Generating production code...
✓ Python code generated
  - Code length: 2543 characters
  - Lines of code: 87
✓ Jupyter notebook generated

[STEP 7] Exporting experiment configuration...
✓ Experiment configuration exported
  - Configuration version: 1.0.0

================================================================================
✓ COMPLETE CLASSIFICATION WORKFLOW TEST PASSED
================================================================================
```

### Test Metrics

**Key Metrics Tracked:**
- Test execution time
- API response times
- Model training duration
- Number of API calls
- Success/failure rates
- Error types and frequencies

### Performance Benchmarks

| Workflow | Expected Duration | API Calls | Success Rate |
|----------|------------------|-----------|--------------|
| Classification | 60s | 10-15 | >95% |
| Regression | 30s | 5-8 | >95% |
| With Tuning | 90s | 8-12 | >90% |
| Model Comparison | 120s | 15-20 | >90% |
| Error Handling | 20s | 3-5 | 100% |

## Troubleshooting

### Common Issues

#### Issue: "Training did not complete within timeout period"

**Cause:** Model training taking longer than expected

**Solutions:**
- Increase `max_wait_time` in test
- Reduce dataset size
- Use simpler model
- Check server resources

#### Issue: "Dataset upload failed"

**Cause:** File path or permissions issue

**Solutions:**
- Verify test fixtures are created
- Check file permissions
- Ensure tmp_path fixture works
- Verify API endpoint is accessible

#### Issue: "Code generation returned 404"

**Cause:** Code generation endpoint not implemented

**Solutions:**
- Check if endpoint exists
- Verify experiment_id is valid
- Skip test if feature not available

#### Issue: "Authentication failed"

**Cause:** Auth headers not configured

**Solutions:**
- Check `auth_headers` fixture
- Verify JWT token is valid
- Ensure authentication is configured

### Debug Mode

Run tests with maximum verbosity:

```bash
pytest tests/integration/test_complete_workflow_e2e.py \
  -v -s \
  --tb=long \
  --log-cli-level=DEBUG \
  -x
```

### Selective Testing

Skip slow tests:

```bash
pytest tests/integration/test_complete_workflow_e2e.py \
  -v -s \
  -m "not slow"
```

Skip tuning tests:

```bash
pytest tests/integration/test_complete_workflow_e2e.py \
  -v -s \
  -k "not tuning"
```

### Test Data Inspection

Inspect generated test data:

```python
# In test
import json

# Save response for inspection
with open('debug_response.json', 'w') as f:
    json.dump(response.json(), f, indent=2)
```

## Best Practices

### Writing E2E Tests

✅ **DO:**
- Test complete user workflows
- Use realistic test data
- Include error scenarios
- Add descriptive print statements
- Set reasonable timeouts
- Clean up test data
- Use fixtures for common setup

❌ **DON'T:**
- Test individual functions (use unit tests)
- Use production data
- Hardcode IDs or paths
- Skip error handling
- Make tests dependent on each other
- Leave test data behind

### Test Data

✅ **Good Test Data:**
- Realistic distributions
- Appropriate size (100-500 rows)
- Mix of data types
- Some missing values
- Representative of real use cases

❌ **Bad Test Data:**
- Too small (<10 rows)
- Too large (>10,000 rows)
- All perfect data
- Unrealistic patterns

### Assertions

✅ **Good Assertions:**
```python
assert response.status_code in [200, 201, 202]
assert dataset_id is not None
assert "accuracy" in metrics
assert len(generated_code) > 100
```

❌ **Bad Assertions:**
```python
assert response.status_code == 200  # Too strict
assert dataset_id  # Could be empty string
assert metrics  # Could be empty dict
```

## Continuous Integration

### GitHub Actions

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run E2E tests
        run: |
          pytest tests/integration/test_complete_workflow_e2e.py \
            -v \
            --cov=app \
            --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Support

For issues or questions:
- Check test output for detailed error messages
- Review API logs
- Check database state
- Verify all services are running
- Consult integration test documentation

---

**Last Updated**: January 2, 2026  
**Test Suite Version**: 1.0.0  
**Total Tests**: 7 test methods across 5 test classes
