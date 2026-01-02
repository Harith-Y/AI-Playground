# ML Engine - Complete Documentation

**Version:** 1.0.0  
**Last Updated:** January 2, 2026  
**Status:** Production Ready

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Modules](#core-modules)
6. [Advanced Features](#advanced-features)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Best Practices](#best-practices)
10. [Performance](#performance)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)

---

## 🎯 Overview

The ML Engine is a comprehensive machine learning framework built for the AI-Playground platform. It provides end-to-end ML capabilities from data preprocessing to model deployment.

### Key Features

✅ **Complete ML Pipeline**
- Data preprocessing and cleaning
- Feature engineering and selection
- Model training and evaluation
- Hyperparameter tuning
- Code generation and deployment

✅ **Production-Ready**
- Memory-optimized for large datasets
- Incremental learning support
- Model serialization and versioning
- Comprehensive error handling
- Extensive logging

✅ **Developer-Friendly**
- Scikit-learn compatible interface
- Intuitive API design
- Comprehensive documentation
- 100+ unit tests
- Type hints throughout

✅ **Flexible & Extensible**
- Modular architecture
- Easy to customize
- Plugin-based model registry
- Custom transformer support

### Supported Tasks

| Task Type | Models | Metrics |
|-----------|--------|---------|
| **Classification** | 6 models | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Regression** | 4 models | R², RMSE, MAE, MSE |
| **Clustering** | 4 models | Silhouette, Davies-Bouldin, Calinski-Harabasz |

### Module Statistics

- **Total Modules:** 50+
- **Lines of Code:** 15,000+
- **Test Coverage:** 85%+
- **Documentation:** 10,000+ lines


---

## 🏗️ Architecture

### Directory Structure

```
ml_engine/
├── preprocessing/           # Data preprocessing (8 modules)
│   ├── base.py             # Base transformer class
│   ├── cleaner.py          # Outlier detection (IQR, Z-score)
│   ├── encoder.py          # Categorical encoding
│   ├── imputer.py          # Missing value imputation
│   ├── scaler.py           # Feature scaling
│   ├── oversampling.py     # SMOTE, ADASYN
│   ├── undersampling.py    # Random, Tomek Links
│   ├── pipeline.py         # Pipeline orchestration
│   ├── serializer.py       # Pipeline serialization
│   ├── config.py           # Configuration management
│   └── column_transformer.py # Column-wise transformations
│
├── feature_selection/      # Feature engineering (5 modules)
│   ├── variance_threshold.py    # Variance-based selection
│   ├── correlation_selector.py  # Correlation-based selection
│   ├── mutual_information_selector.py  # MI-based selection
│   ├── rfe_selector.py          # Recursive Feature Elimination
│   └── univariate_selector.py   # Univariate statistical tests
│
├── models/                 # ML models (6 modules)
│   ├── base.py            # Base model wrapper
│   ├── classification.py  # Classification models
│   ├── regression.py      # Regression models
│   ├── clustering.py      # Clustering models
│   ├── registry.py        # Model factory
│   └── validation.py      # Model validation
│
├── training/              # Training utilities (5 modules)
│   ├── trainer.py         # Generic trainer
│   ├── data_split.py      # Train/test splitting
│   ├── cross_validation.py # K-fold CV
│   └── incremental_trainer.py # Incremental learning
│
├── tuning/                # Hyperparameter optimization (5 modules)
│   ├── grid_search.py     # Grid search
│   ├── random_search.py   # Random search
│   ├── bayesian.py        # Bayesian optimization
│   ├── search_spaces.py   # Parameter spaces
│   └── cross_validation.py # CV for tuning
│
├── evaluation/            # Model evaluation (10 modules)
│   ├── metrics.py         # Metric computation
│   ├── classification_metrics.py  # Classification metrics
│   ├── regression_metrics.py      # Regression metrics
│   ├── clustering_metrics.py      # Clustering metrics
│   ├── confusion_matrix.py        # Confusion matrix
│   ├── roc_curve.py              # ROC curve
│   ├── pr_curve.py               # Precision-Recall curve
│   ├── feature_importance.py     # Feature importance
│   ├── residual_analysis.py      # Residual plots
│   └── visualizations.py         # Visualization utilities
│
├── code_generation/       # Code export (8 modules)
│   ├── generator.py       # Main generator
│   ├── template_engine.py # Jinja2 template engine
│   ├── templates.py       # Code templates
│   ├── preprocessing_generator.py  # Preprocessing code
│   ├── training_generator.py       # Training code
│   ├── prediction_generator.py     # Prediction code
│   ├── evaluation_generator.py     # Evaluation code
│   └── requirements_generator.py   # Requirements.txt
│
├── inference/             # Model inference (2 modules)
│   └── optimized_predictor.py  # Optimized prediction
│
├── utils/                 # Utilities (5 modules)
│   ├── serialization.py   # Model/pipeline serialization
│   ├── column_type_detector.py  # Type detection
│   ├── dataset_optimizer.py     # Memory optimization
│   └── THEORY.md          # Theoretical background
│
├── validation/            # Data validation (2 modules)
│   ├── edge_case_validator.py  # Edge case handling
│   └── edge_case_fixes.py      # Automatic fixes
│
├── eda_statistics.py      # EDA analysis
├── correlation_analysis.py # Correlation analysis
├── class_distribution_analysis.py  # Class balance
└── model_registry.py      # Model registry
```

### Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Consistency**: All modules follow scikit-learn interface
3. **Extensibility**: Easy to add new models and transformers
4. **Performance**: Optimized for memory and speed
5. **Reliability**: Comprehensive error handling and validation

### Data Flow

```
Raw Data
    ↓
[Preprocessing Pipeline]
    ├── Cleaning (outliers, duplicates)
    ├── Imputation (missing values)
    ├── Encoding (categorical variables)
    ├── Scaling (normalization)
    └── Feature Selection
    ↓
Processed Data
    ↓
[Model Training]
    ├── Train/Test Split
    ├── Cross-Validation
    ├── Hyperparameter Tuning
    └── Model Fitting
    ↓
Trained Model
    ↓
[Evaluation]
    ├── Metrics Computation
    ├── Visualization
    └── Feature Importance
    ↓
[Deployment]
    ├── Model Serialization
    ├── Code Generation
    └── API Integration
```


---

## 📦 Installation

### Requirements

- Python 3.11+
- scikit-learn 1.8.0+
- pandas 2.3.3+
- numpy 2.4.0+

### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Verify Installation

```python
from app.ml_engine.models.classification import ClassificationModel
from app.ml_engine.preprocessing.pipeline import Pipeline

print("ML Engine installed successfully!")
```

---

## 🚀 Quick Start

### 1. Basic Classification

```python
from app.ml_engine.models.classification import ClassificationModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = ClassificationModel(model_type='random_forest_classifier')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### 2. Complete Pipeline

```python
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder
import pandas as pd

# Create pipeline
pipeline = Pipeline(steps=[
    ('imputer', MeanImputer()),
    ('scaler', StandardScaler()),
    ('encoder', OneHotEncoder())
])

# Fit and transform
df = pd.read_csv('data.csv')
df_transformed = pipeline.fit_transform(df)

# Save pipeline
pipeline.save('pipeline.pkl')

# Load and use
loaded_pipeline = Pipeline.load('pipeline.pkl')
new_data_transformed = loaded_pipeline.transform(new_data)
```

### 3. Hyperparameter Tuning

```python
from app.ml_engine.tuning.grid_search import run_grid_search
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Run grid search
result = run_grid_search(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    X=X_train,
    y=y_train,
    cv=5,
    scoring='accuracy'
)

print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### 4. Model Evaluation

```python
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics

# Evaluate model
metrics = ClassificationMetrics(y_test, predictions)

print(f"Accuracy: {metrics.accuracy():.4f}")
print(f"Precision: {metrics.precision():.4f}")
print(f"Recall: {metrics.recall():.4f}")
print(f"F1 Score: {metrics.f1_score():.4f}")

# Get confusion matrix
cm = metrics.confusion_matrix()
print(cm)

# Get classification report
report = metrics.classification_report()
print(report)
```

### 5. Code Generation

```python
from app.ml_engine.code_generation.generator import generate_training_code

# Generate training code
code = generate_training_code(
    model_type='random_forest_classifier',
    preprocessing_steps=['imputer', 'scaler'],
    hyperparameters={'n_estimators': 100, 'max_depth': 10}
)

# Save to file
with open('train_model.py', 'w') as f:
    f.write(code)
```

