# Quick Test Reference Guide

Fast reference for running ML engine tests.

## Quick Start

```bash
# Navigate to backend
cd backend

# Run all tests
pytest tests/ml_engine/ -v

# Run with custom runner
python tests/run_ml_tests.py
```

## Test Categories

### Evaluation Metrics Tests

```bash
# All evaluation tests
pytest tests/ml_engine/evaluation/ -v

# Specific modules
pytest tests/ml_engine/evaluation/test_classification_metrics.py -v
pytest tests/ml_engine/evaluation/test_regression_metrics.py -v
pytest tests/ml_engine/evaluation/test_feature_importance.py -v
pytest tests/ml_engine/evaluation/test_clustering_metrics.py -v
```

### Hyperparameter Tuning Tests

```bash
# All tuning tests
pytest tests/ml_engine/tuning/ -v

# Specific modules
pytest tests/ml_engine/tuning/test_search_spaces.py -v
pytest tests/ml_engine/tuning/test_grid_search.py -v
pytest tests/ml_engine/tuning/test_random_search.py -v
pytest tests/ml_engine/tuning/test_bayesian.py -v
pytest tests/ml_engine/tuning/test_cross_validation.py -v
```

### Integration Tests

```bash
# All integration tests
pytest tests/ml_engine/evaluation/test_integration.py -v
pytest tests/ml_engine/tuning/test_integration.py -v
```

## Common Options

```bash
# Verbose output
pytest tests/ml_engine/ -v

# Show print statements
pytest tests/ml_engine/ -s

# Stop on first failure
pytest tests/ml_engine/ -x

# Run specific test
pytest tests/ml_engine/tuning/test_bayesian.py::TestRunBayesianSearch::test_bayesian_search_with_skopt

# Run tests matching pattern
pytest tests/ml_engine/ -k "bayesian" -v

# Show slowest tests
pytest tests/ml_engine/ --durations=10
```

## Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ml_engine/ \
  --cov=app.ml_engine.evaluation \
  --cov=app.ml_engine.tuning \
  --cov-report=html

# View report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
xdg-open htmlcov/index.html  # Linux

# Terminal coverage report
pytest tests/ml_engine/ \
  --cov=app.ml_engine \
  --cov-report=term-missing
```

## Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest tests/ml_engine/ -n 4

# Auto-detect CPU count
pytest tests/ml_engine/ -n auto
```

## Test Markers

```bash
# Run only fast tests
pytest tests/ml_engine/ -m "not slow"

# Run only integration tests
pytest tests/ml_engine/ -m integration

# Run unit tests only
pytest tests/ml_engine/ -m "not integration"
```

## Debugging

```bash
# Drop into debugger on failure
pytest tests/ml_engine/ --pdb

# Show local variables on failure
pytest tests/ml_engine/ -l

# Show full traceback
pytest tests/ml_engine/ --tb=long

# Quiet mode (minimal output)
pytest tests/ml_engine/ -q
```

## Output Control

```bash
# Capture output (default)
pytest tests/ml_engine/

# Show all output
pytest tests/ml_engine/ -s

# Show only failures
pytest tests/ml_engine/ --tb=short

# No capture + verbose
pytest tests/ml_engine/ -sv
```

## Test Selection

```bash
# Run single file
pytest tests/ml_engine/tuning/test_bayesian.py

# Run single class
pytest tests/ml_engine/tuning/test_bayesian.py::TestRunBayesianSearch

# Run single test
pytest tests/ml_engine/tuning/test_bayesian.py::TestRunBayesianSearch::test_bayesian_search_with_skopt

# Run multiple files
pytest tests/ml_engine/tuning/test_grid_search.py tests/ml_engine/tuning/test_random_search.py
```

## Watch Mode

```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests on file changes
ptw tests/ml_engine/ -- -v
```

## Custom Test Runner

```bash
# Use the custom runner script
python tests/run_ml_tests.py

# Shows organized output:
# - Evaluation tests
# - Tuning tests
# - Summary statistics
# - Success rate
```

## CI/CD Integration

```bash
# Run tests with JUnit XML output (for CI)
pytest tests/ml_engine/ --junitxml=test-results.xml

# Run with multiple output formats
pytest tests/ml_engine/ \
  --junitxml=test-results.xml \
  --html=test-report.html \
  --cov-report=xml
```

## Environment Setup

```bash
# Activate virtual environment
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install test dependencies
pip install pytest pytest-cov pytest-xdist pytest-watch

# Install optional dependencies for full test coverage
pip install scikit-optimize  # Bayesian optimization tests
pip install shap  # SHAP feature importance tests
```

## Test File Structure

```
backend/tests/ml_engine/
├── evaluation/
│   ├── test_classification_metrics.py  ✅ 20+ tests
│   ├── test_confusion_matrix.py       ✅ 15+ tests
│   ├── test_roc_curve.py              ✅ 10+ tests
│   ├── test_pr_curve.py               ✅ 10+ tests
│   ├── test_regression_metrics.py     ✅ 15+ tests
│   ├── test_residual_analysis.py      ✅ 10+ tests
│   ├── test_actual_vs_predicted.py    ✅ 10+ tests
│   ├── test_clustering_metrics.py     ✅ 15+ tests
│   ├── test_feature_importance.py     ✅ 12+ tests
│   └── test_integration.py            ✅ 20+ tests
│
├── tuning/
│   ├── test_search_spaces.py          ✅ 30+ tests
│   ├── test_grid_search.py            ✅ 15+ tests
│   ├── test_random_search.py          ✅ 15+ tests
│   ├── test_bayesian.py               ✅ 15+ tests
│   ├── test_cross_validation.py       ✅ 20+ tests
│   └── test_integration.py            ✅ 20+ tests
│
├── run_ml_tests.py                    ✅ Custom runner
└── TEST_DOCUMENTATION.md              ✅ Full docs
```

## Expected Results

```
========================= test session starts =========================
platform win32 -- Python 3.11.x
collected 250+ items

tests/ml_engine/evaluation/test_classification_metrics.py ........... [70%]
tests/ml_engine/evaluation/test_integration.py ...................... [80%]
tests/ml_engine/tuning/test_search_spaces.py ........................ [90%]
tests/ml_engine/tuning/test_integration.py ...................... [100%]

========================= 250+ passed in 120.00s ======================
```

## Troubleshooting

### Import Errors

```bash
# Add backend to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;C:\path\to\backend  # Windows
export PYTHONPATH="${PYTHONPATH}:/path/to/backend"  # Mac/Linux
```

### Missing Dependencies

```bash
# Install all test requirements
pip install -r requirements.txt
pip install pytest pytest-cov

# Install optional dependencies
pip install scikit-optimize shap
```

### Slow Tests

```bash
# Run only fast tests
pytest tests/ml_engine/ -m "not slow"

# Use parallel execution
pytest tests/ml_engine/ -n auto

# Profile slow tests
pytest tests/ml_engine/ --durations=20
```

## Quick Commands Cheatsheet

| Command                                 | Description                   |
| --------------------------------------- | ----------------------------- |
| `pytest tests/ml_engine/ -v`            | Run all tests (verbose)       |
| `pytest tests/ml_engine/ -x`            | Stop on first failure         |
| `pytest tests/ml_engine/ -k "bayesian"` | Run tests matching "bayesian" |
| `pytest tests/ml_engine/ --lf`          | Run last failed tests         |
| `pytest tests/ml_engine/ --ff`          | Run failures first            |
| `pytest tests/ml_engine/ -n auto`       | Parallel execution            |
| `pytest tests/ml_engine/ --cov`         | With coverage                 |
| `pytest tests/ml_engine/ -s`            | Show print output             |
| `python tests/run_ml_tests.py`          | Custom runner                 |

---

**Quick Tip**: Bookmark this file for fast test execution! 🚀
