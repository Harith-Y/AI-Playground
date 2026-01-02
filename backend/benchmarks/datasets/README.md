# Benchmark Datasets

This directory contains standard ML datasets for benchmarking the AI-Playground pipeline.

## Datasets

### Classification Datasets

1. **Iris** (iris.csv)
   - Samples: 150
   - Features: 4
   - Target: species (3 classes: setosa, versicolor, virginica)
   - Use case: Multi-class classification
   - Difficulty: Easy

2. **Wine** (wine.csv)
   - Samples: 178
   - Features: 13
   - Target: cultivar (3 classes)
   - Use case: Multi-class classification
   - Difficulty: Easy

3. **Breast Cancer** (breast_cancer.csv)
   - Samples: 569
   - Features: 30
   - Target: diagnosis (2 classes: malignant, benign)
   - Use case: Binary classification
   - Difficulty: Medium

4. **Digits** (digits.csv)
   - Samples: 1,797
   - Features: 64
   - Target: digit (10 classes: 0-9)
   - Use case: Multi-class classification
   - Difficulty: Medium

### Regression Datasets

1. **Diabetes** (diabetes.csv)
   - Samples: 442
   - Features: 10
   - Target: progression (disease progression after one year)
   - Use case: Regression
   - Difficulty: Easy

2. **California Housing** (california_housing.csv)
   - Samples: 20,640
   - Features: 8
   - Target: median_house_value
   - Use case: Regression
   - Difficulty: Medium
   - Note: This is the recommended alternative to the deprecated Boston Housing dataset

## Usage

These datasets are used for:
- Testing the ML training pipeline
- Benchmarking model performance
- Validating memory optimization
- Testing preprocessing operations
- Performance profiling

## Data Sources

All datasets are from scikit-learn's built-in datasets:
- https://scikit-learn.org/stable/datasets/toy_dataset.html
- https://scikit-learn.org/stable/datasets/real_world.html
