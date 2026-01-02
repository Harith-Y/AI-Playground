# AI-Playground Benchmarking Suite

Comprehensive benchmarking system for validating the ML pipeline with standard datasets.

## Overview

This benchmarking suite tests the AI-Playground ML pipeline across multiple dimensions:

- **Model Performance**: Accuracy, precision, recall, F1, R², RMSE, MAE
- **Training Speed**: Time to train models
- **Memory Efficiency**: Memory usage during training
- **Dataset Variety**: Classification and regression tasks
- **Model Diversity**: Multiple model types per task

## Quick Start

### Option 1: Run Complete Suite

```bash
cd backend/benchmarks
python run_all.py
```

This will:
1. Download and prepare all datasets
2. Run all benchmarks
3. Generate comprehensive reports

### Option 2: Step by Step

```bash
# Step 1: Prepare datasets
python prepare_datasets.py

# Step 2: Run benchmarks
python run_benchmarks.py
```

## Datasets

### Classification Datasets

| Dataset | Samples | Features | Classes | Difficulty |
|---------|---------|----------|---------|------------|
| Iris | 150 | 4 | 3 | Easy |
| Wine | 178 | 13 | 3 | Easy |
| Breast Cancer | 569 | 30 | 2 | Medium |
| Digits | 1,797 | 64 | 10 | Medium |

### Regression Datasets

| Dataset | Samples | Features | Difficulty |
|---------|---------|----------|------------|
| Diabetes | 442 | 10 | Easy |
| California Housing | 20,640 | 8 | Medium |

## Models Tested

### Classification Models

- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier
- Decision Tree Classifier

### Regression Models

- Random Forest Regressor
- Linear Regression
- Gradient Boosting Regressor
- Ridge Regression

## Benchmark Metrics

### Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall

### Regression Metrics

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error

### Performance Metrics

- **Training Time**: Time to fit the model (seconds)
- **Memory Delta**: Memory used during training (MB)
- **Baseline Memory**: Memory before training (MB)
- **Final Memory**: Memory after training (MB)

## Output Files

After running benchmarks, you'll find:

```
backend/benchmarks/
├── datasets/              # Benchmark datasets
│   ├── iris.csv
│   ├── wine.csv
│   ├── breast_cancer.csv
│   ├── digits.csv
│   ├── diabetes.csv
│   ├── california_housing.csv
│   └── README.md
└── results/               # Benchmark results
    ├── benchmark_results.json    # Raw results (JSON)
    ├── benchmark_results.csv     # Raw results (CSV)
    └── BENCHMARK_REPORT.md       # Human-readable report
```

## Example Results

### Classification Example

```
Dataset: Iris
Model: Random Forest Classifier
✓ Accuracy: 0.9667
  Precision: 0.9722
  Recall: 0.9667
  F1 Score: 0.9661
  Training time: 0.15s
  Memory delta: +2.34MB
```

### Regression Example

```
Dataset: Diabetes
Model: Gradient Boosting Regressor
✓ R² Score: 0.4821
  RMSE: 53.2145
  MAE: 42.1834
  Training time: 0.42s
  Memory delta: +3.12MB
```

## Understanding the Report

The generated `BENCHMARK_REPORT.md` includes:

1. **Summary**: Overview of all benchmarks
2. **Best Results**: Top performing model for each dataset
3. **Detailed Tables**: Complete results in tabular format
4. **Performance Insights**: Fastest/slowest models, memory usage patterns

## Customizing Benchmarks

### Adding New Datasets

Edit `prepare_datasets.py`:

```python
def prepare_my_dataset(output_dir: Path):
    # Load your dataset
    df = load_data()

    # Save to CSV
    output_path = output_dir / "my_dataset.csv"
    df.to_csv(output_path, index=False)

    return output_path
```

### Testing Additional Models

Edit `run_benchmarks.py`:

```python
benchmarks = [
    {
        'type': 'classification',
        'dataset': 'Iris',
        'path': datasets_dir / 'iris.csv',
        'target': 'species',
        'models': ['random_forest_classifier', 'my_new_model']  # Add here
    }
]
```

### Adjusting Test Split

In `run_benchmarks.py`:

```python
# Change test_size parameter
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,  # Change from 0.2 to 0.3
    random_state=42
)
```

## Requirements

All dependencies are in `backend/requirements.txt`:

- scikit-learn
- pandas
- numpy

The benchmark suite uses the same dependencies as the main application.

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the `backend/benchmarks` directory:

```bash
cd backend/benchmarks
python run_all.py
```

### Missing Datasets

If datasets aren't found:

```bash
# Re-run dataset preparation
python prepare_datasets.py
```

### Memory Issues

For large datasets or limited RAM:

1. Reduce the number of models tested
2. Use smaller datasets
3. Run benchmarks sequentially (one at a time)

### Slow Benchmarks

To speed up benchmarks:

1. Reduce number of models tested
2. Skip large datasets (California Housing)
3. Use simpler models (exclude SVM, Gradient Boosting)

## Performance Expectations

Typical benchmark times on modern hardware:

- **Iris**: ~1-2 seconds per model
- **Wine**: ~1-2 seconds per model
- **Breast Cancer**: ~2-3 seconds per model
- **Digits**: ~3-5 seconds per model
- **Diabetes**: ~2-4 seconds per model
- **California Housing**: ~10-30 seconds per model

**Total suite time**: ~2-5 minutes

## Integration with CI/CD

To run benchmarks in CI/CD:

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run benchmarks
        run: |
          cd backend/benchmarks
          python run_all.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: backend/benchmarks/results/
```

## Future Enhancements

Potential improvements to the benchmarking suite:

- [ ] Add cross-validation metrics
- [ ] Include hyperparameter tuning benchmarks
- [ ] Add data preprocessing benchmarks
- [ ] Test incremental learning performance
- [ ] Benchmark chunked data loading
- [ ] Add GPU benchmarks
- [ ] Include model serialization/deserialization times
- [ ] Test prediction latency
- [ ] Add batch prediction benchmarks
- [ ] Include feature importance calculation time

## Contributing

To contribute new benchmarks:

1. Add dataset preparation in `prepare_datasets.py`
2. Add benchmark configuration in `run_benchmarks.py`
3. Update this README with dataset info
4. Run the suite to verify
5. Submit PR with results

## License

Part of the AI-Playground project.
