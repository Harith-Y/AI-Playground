# ML Engine - AI-Playground

Comprehensive machine learning module providing preprocessing, feature selection, EDA analysis, and correlation utilities. Built with scikit-learn interface for seamless integration.

## 📁 Structure

```
ml_engine/
├── preprocessing/              # Data preprocessing modules ✅
│   ├── base.py                # Base transformer class (186 lines)
│   ├── cleaner.py             # IQR & Z-score outlier detection (458 lines)
│   ├── encoder.py             # OneHot, Label, Ordinal encoders (449 lines)
│   ├── imputer.py             # Mean, Median, Mode imputers (300 lines)
│   ├── scaler.py              # Standard, MinMax, Robust scalers (464 lines)
│   └── pipeline.py            # Pipeline orchestration (1044 lines) ✅
├── feature_selection/          # Feature engineering modules ✅
│   ├── variance_threshold.py  # Variance-based selection (319 lines)
│   ├── correlation_selector.py # Correlation-based selection (386 lines)
│   └── mutual_information_selector.py  # MI selection (390 lines)
├── eda_statistics.py           # EDA analysis module ✅ (514 lines)
├── correlation_analysis.py     # Correlation matrices ✅ (535 lines)
├── utils/                      # Utility modules ✅
│   ├── column_type_detector.py # Column type detection (175 lines)
│   └── serialization.py       # Model/pipeline serialization (281 lines) ✅
├── models/                     # ML models ✅
│   ├── base.py                # Base model wrappers (157 lines)
│   ├── classification.py      # Classification models (109 lines)
│   ├── regression.py          # Regression models (130 lines)
│   ├── clustering.py          # Clustering models (55 lines)
│   └── registry.py            # Model factory (24 lines)
├── training/                   # Training utilities ✅
│   └── trainer.py             # Generic training functions (149 lines)
├── tuning/                     # Hyperparameter optimization ✅
│   ├── grid_search.py         # Grid search (42 lines)
│   ├── random_search.py       # Random search (42 lines)
│   └── bayesian.py            # Bayesian optimization (62 lines)
├── evaluation/                 # Model evaluation ✅
│   ├── classification_metrics.py # Classification metrics (136 lines)
│   ├── regression_metrics.py  # Regression metrics (174 lines)
│   ├── clustering_metrics.py  # Clustering metrics (114 lines)
│   └── feature_importance.py  # Feature importance (119 lines)
└── code_generation/            # Code export ✅
    ├── preprocessing_generator.py # Preprocessing code (120 lines)
    ├── training_generator.py  # Training code (83 lines)
    ├── prediction_generator.py # Prediction code (140 lines)
    └── evaluation_generator.py # Evaluation code (157 lines)
```

## ✅ Implemented Modules

### 1. Preprocessing

All preprocessing modules follow the scikit-learn `fit/transform` interface.

#### **IQROutlierDetector** (cleaner.py)
Detects and handles outliers using the Interquartile Range method.

```python
from app.ml_engine.preprocessing.cleaner import IQROutlierDetector
import pandas as pd

df = pd.DataFrame({'age': [25, 30, 35, 100, 28, 32]})

# Detect outliers (returns boolean mask)
detector = IQROutlierDetector(threshold=1.5)
outliers = detector.fit_transform(df)
# Returns: [False, False, False, True, False, False]

# Remove outliers
clean_df = df[~outliers]
```

**Parameters:**
- `threshold` (float, default=1.5): IQR multiplier for outlier bounds
- `columns` (list, optional): Specific columns to check

#### **ZScoreOutlierDetector** (cleaner.py)
Detects outliers using standard deviation method.

```python
from app.ml_engine.preprocessing.cleaner import ZScoreOutlierDetector

detector = ZScoreOutlierDetector(threshold=3.0)
outliers = detector.fit_transform(df)
```

**Parameters:**
- `threshold` (float, default=3.0): Number of standard deviations
- `columns` (list, optional): Columns to check

#### **MeanImputer / MedianImputer** (imputer.py)
Fill missing values with mean or median.

```python
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer

df = pd.DataFrame({'age': [25, None, 35], 'salary': [50000, 60000, None]})

# Mean imputation
imputer = MeanImputer()
df_filled = imputer.fit_transform(df)

# Median imputation
imputer = MedianImputer()
df_filled = imputer.fit_transform(df)
```

#### **ModeImputer** (imputer.py)
Fill missing categorical values with most frequent value.

```python
from app.ml_engine.preprocessing.imputer import ModeImputer

df = pd.DataFrame({'category': ['A', 'B', None, 'A', 'B', None]})

imputer = ModeImputer()
df_filled = imputer.fit_transform(df)
```

#### **StandardScaler / MinMaxScaler / RobustScaler** (scaler.py)
Feature normalization and standardization.

```python
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler, RobustScaler

df = pd.DataFrame({'age': [25, 30, 35], 'salary': [50000, 60000, 70000]})

# Standard scaling (mean=0, std=1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# MinMax scaling (range [0, 1])
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Robust scaling (resistant to outliers)
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)
```

#### **OneHotEncoder / LabelEncoder / OrdinalEncoder** (encoder.py)
Categorical variable encoding.

```python
from app.ml_engine.preprocessing.encoder import OneHotEncoder, LabelEncoder

df = pd.DataFrame({'category': ['A', 'B', 'C', 'A', 'B']})

# One-hot encoding
encoder = OneHotEncoder()
df_encoded = encoder.fit_transform(df)
# Returns: category_A, category_B, category_C columns

# Label encoding
encoder = LabelEncoder()
df_encoded = encoder.fit_transform(df)
# Returns: 0, 1, 2, 0, 1
```

### 2. Feature Selection

#### **VarianceThresholdSelector** (variance_threshold.py)
Remove low-variance features.

```python
from app.ml_engine.feature_selection.variance_threshold import VarianceThresholdSelector

df = pd.DataFrame({
    'constant': [1, 1, 1, 1],      # Will be removed
    'low_var': [1, 1, 1, 2],       # Will be removed
    'high_var': [1, 5, 10, 15]     # Will be kept
})

selector = VarianceThresholdSelector(threshold=0.1)
selected = selector.fit_transform(df)
# Returns only 'high_var' column

# Get selected feature names
print(selector.selected_features_)  # ['high_var']

# Get variance scores
print(selector.variances_)  # {' constant': 0.0, 'low_var': 0.05, 'high_var': 25.0}
```

**Methods:**
- `fit_transform(df)` - Select features
- `get_support()` - Get boolean mask of selected features
- `get_feature_names_out()` - Get selected feature names

#### **CorrelationSelector** (correlation_selector.py)
Select features based on correlation with target variable.

```python
from app.ml_engine.feature_selection.correlation_selector import CorrelationSelector

X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],  # Negatively correlated with target
    'feature3': [1, 1, 1, 1, 1]   # No correlation
})
y = pd.Series([10, 20, 30, 40, 50])

# Select top 2 features by correlation
selector = CorrelationSelector(k=2, method='pearson')
X_selected = selector.fit_transform(X, y)

# Get correlation scores
print(selector.correlation_scores_)  # Correlation with target

# Select features above threshold
selector = CorrelationSelector(threshold=0.5, method='spearman')
X_selected = selector.fit_transform(X, y)
```

**Parameters:**
- `k` (int, optional): Number of top features to select
- `threshold` (float, optional): Minimum absolute correlation
- `method` (str, default='pearson'): 'pearson', 'spearman', or 'kendall'

#### **MutualInformationSelector** (mutual_information_selector.py)
Select features using mutual information scores.

```python
from app.ml_engine.feature_selection.mutual_information_selector import MutualInformationSelector

# Classification task
selector = MutualInformationSelector(k=5, task='classification')
X_selected = selector.fit_transform(X, y)

# Regression task
selector = MutualInformationSelector(threshold=0.1, task='regression')
X_selected = selector.fit_transform(X, y)

# Get MI scores
print(selector.mi_scores_)  # Mutual information scores
```

**Parameters:**
- `k` (int, optional): Number of top features
- `threshold` (float, optional): Minimum MI score
- `task` (str, default='classification'): 'classification' or 'regression'
- `random_state` (int, default=42): Random seed

### 3. EDA Statistics

#### **EDAStatistics** (eda_statistics.py)
Comprehensive exploratory data analysis.

```python
from app.ml_engine.eda_statistics import EDAStatistics

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 70000, 80000, 90000],
    'department': ['IT', 'HR', 'IT', 'Sales', 'HR']
})

eda = EDAStatistics(df)

# Quick summary
summary = eda.quick_summary()
print(summary)

# Detailed statistics
stats = eda.describe_all()
print(stats)

# Distribution analysis
dist = eda.analyze_distribution('age')
print(f"Skewness: {dist['skewness']}")
print(f"Kurtosis: {dist['kurtosis']}")
print(f"Is Normal: {dist['is_normal']}")

# Missing values analysis
missing = eda.analyze_missing_values()
print(missing)

# Correlation analysis
corr = eda.analyze_correlations(method='pearson')
print(corr)

# Outlier detection
outliers = eda.detect_outliers(method='iqr', threshold=1.5)
print(outliers)

# Data quality issues
issues = eda.detect_data_issues()
print(issues)
```

**Methods:**
- `quick_summary()` - High-level overview
- `describe_all()` - Detailed statistics for all columns
- `analyze_distribution(column)` - Distribution metrics (skewness, kurtosis, normality)
- `analyze_missing_values()` - Missing value patterns
- `analyze_correlations(method)` - Correlation matrix
- `detect_outliers(method, threshold)` - Outlier detection
- `detect_duplicates()` - Duplicate row analysis
- `get_data_types_summary()` - Data type breakdown
- `analyze_categorical_columns()` - Categorical variable analysis
- `analyze_numeric_columns()` - Numeric variable analysis
- `detect_data_issues()` - Comprehensive data quality check

### 4. Correlation Analysis

#### **CorrelationMatrix** (correlation_analysis.py)
Advanced correlation analysis with clustering and multicollinearity detection.

```python
from app.ml_engine.correlation_analysis import CorrelationMatrix

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'experience': [2, 5, 8, 12, 15],  # Highly correlated with age
    'salary': [50000, 60000, 70000, 80000, 90000]
})

corr_matrix = CorrelationMatrix(df)

# Compute correlation
corr = corr_matrix.compute_correlation(method='pearson')
print(corr)

# Get high correlation pairs
pairs = corr_matrix.get_correlation_pairs(threshold=0.8, absolute=True)
print(pairs)

# Get correlation clusters (hierarchical clustering)
clusters = corr_matrix.get_correlation_clusters(threshold=0.8)
print(clusters)

# Get heatmap data for visualization
heatmap = corr_matrix.get_heatmap_data()
# Returns: {z, x, y, annotations, colorscale}

# Feature ranking by average correlation
ranking = corr_matrix.get_feature_ranking(ascending=False)
print(ranking)

# Multicollinearity detection
multicollinearity = corr_matrix.get_multicollinearity_stats(threshold=0.9)
print(multicollinearity)
# Returns: {high_correlation_count, high_correlation_pairs, problematic_features, vif_approximation}

# Correlation network (for graph visualization)
network = corr_matrix.get_correlation_network(threshold=0.5)
print(network)
```

**Methods:**
- `compute_correlation(method, columns)` - Compute correlation matrix
- `get_correlation_pairs(threshold)` - Get correlated feature pairs
- `get_correlation_clusters(threshold)` - Hierarchical clustering of features
- `get_heatmap_data()` - Plotly-compatible heatmap data
- `get_feature_ranking()` - Rank features by correlation strength
- `get_multicollinearity_stats()` - VIF approximation, problematic features
- `get_correlation_network()` - Graph structure for network visualization

**Convenience Functions:**
```python
from app.ml_engine.correlation_analysis import (
    compute_correlation,
    find_highly_correlated_features,
    get_correlation_heatmap_data,
    detect_multicollinearity,
    recommend_feature_removal
)

# Quick correlation matrix
corr = compute_correlation(df, method='pearson')

# Find problematic pairs
pairs = find_highly_correlated_features(df, threshold=0.9)

# Get heatmap data
heatmap = get_correlation_heatmap_data(df, method='spearman')

# Check for multicollinearity
stats = detect_multicollinearity(df, threshold=0.9)

# Get feature removal recommendations
recommendations = recommend_feature_removal(df, threshold=0.9)
```

## 🎯 Common Workflows

### Complete Preprocessing Pipeline

```python
from app.ml_engine.preprocessing.cleaner import IQROutlierDetector
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# 1. Handle outliers
outlier_detector = IQROutlierDetector(threshold=1.5)
outlier_mask = outlier_detector.fit_transform(df)
df = df[~outlier_mask]

# 2. Impute missing values
imputer = MeanImputer()
df = imputer.fit_transform(df)

# 3. Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder()
df = encoder.fit_transform(df[categorical_cols])

# 4. Scale numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

### Feature Selection Workflow

```python
from app.ml_engine.feature_selection.variance_threshold import VarianceThresholdSelector
from app.ml_engine.feature_selection.correlation_selector import CorrelationSelector
from app.ml_engine.feature_selection.mutual_information_selector import MutualInformationSelector

# 1. Remove low-variance features
variance_selector = VarianceThresholdSelector(threshold=0.01)
X = variance_selector.fit_transform(X)

# 2. Select by correlation with target
corr_selector = CorrelationSelector(k=10, method='pearson')
X = corr_selector.fit_transform(X, y)

# 3. Final selection with mutual information
mi_selector = MutualInformationSelector(k=5, task='classification')
X_final = mi_selector.fit_transform(X, y)
```

### EDA Workflow

```python
from app.ml_engine.eda_statistics import EDAStatistics
from app.ml_engine.correlation_analysis import CorrelationMatrix

# 1. Basic EDA
eda = EDAStatistics(df)
summary = eda.quick_summary()
issues = eda.detect_data_issues()

# 2. Distribution analysis
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    dist = eda.analyze_distribution(col)
    print(f"{col}: skewness={dist['skewness']:.2f}, is_normal={dist['is_normal']}")

# 3. Correlation analysis
corr_matrix = CorrelationMatrix(df)
corr = corr_matrix.compute_correlation(method='pearson')
multicollinearity = corr_matrix.get_multicollinearity_stats(threshold=0.9)

# 4. Identify problems
outliers = eda.detect_outliers(method='iqr')
missing = eda.analyze_missing_values()
```

## 🧪 Testing

All modules have comprehensive unit tests:

```powershell
# Run all ML engine tests
pytest tests/test_*imputer*.py tests/test_*scaler*.py tests/test_*encoder*.py tests/test_*selector*.py tests/test_*outlier*.py tests/test_eda*.py tests/test_correlation*.py

# Run specific module tests
pytest tests/test_mean_median_imputer.py -v
pytest tests/test_variance_threshold.py -v
pytest tests/test_correlation_analysis.py -v
```

## 📚 API Reference

### Base Transformer Class

All preprocessing modules inherit from `BaseTransformer`:

```python
class BaseTransformer:
    def fit(self, X, y=None):
        """Fit transformer to data"""

    def transform(self, X):
        """Transform data"""

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""

    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
```

### Compatibility

All modules are compatible with:
- ✅ pandas DataFrames
- ✅ numpy arrays (most modules)
- ✅ scikit-learn Pipelines
- ✅ Custom ML pipelines

## 🔍 Module Details

### Preprocessing Modules

| Module | Lines | Features |
|--------|-------|----------|
| base.py | 186 | Base transformer, sklearn interface |
| cleaner.py | 458 | IQR & Z-score outlier detection |
| encoder.py | 449 | OneHot, Label, Ordinal encoding |
| imputer.py | 300 | Mean, Median, Mode imputation |
| scaler.py | 464 | Standard, MinMax, Robust scaling |

### Feature Selection Modules

| Module | Lines | Features |
|--------|-------|----------|
| variance_threshold.py | 319 | Variance-based selection |
| correlation_selector.py | 386 | Correlation with target |
| mutual_information_selector.py | 390 | MI-based selection |

### Analysis Modules

| Module | Lines | Features |
|--------|-------|----------|
| eda_statistics.py | 514 | 10+ analysis methods |
| correlation_analysis.py | 535 | 7 methods + 5 convenience functions |

### Utility Modules

| Module | Lines | Features |
|--------|-------|----------|
| serialization.py | 281 | Model/pipeline/workflow serialization ✅ |
| column_type_detector.py | 175 | Automatic column type detection |

---

## 5. Model & Pipeline Serialization ✅

### **ModelSerializer / PipelineSerializer / WorkflowSerializer** (serialization.py)

Complete serialization utilities for saving and loading ML artifacts.

```python
from app.ml_engine.utils.serialization import (
    save_model, load_model,
    save_pipeline, load_pipeline,
    save_workflow, load_workflow
)

# Save a trained model
save_model(trained_model, 'models/my_model.pkl', compression=True)

# Load the model
model = load_model('models/my_model.pkl')
predictions = model.predict(X_test)

# Save a preprocessing pipeline
save_pipeline(fitted_pipeline, 'pipelines/preprocessing.pkl')

# Load the pipeline
pipeline = load_pipeline('pipelines/preprocessing.pkl')
X_transformed = pipeline.transform(X_test)

# Save complete workflow (preprocessing + model)
save_workflow(
    pipeline=preprocessing_pipeline,
    model=trained_model,
    path='workflows/production.pkl',
    workflow_name='CustomerChurn_v1'
)

# Load workflow
pipeline, model = load_workflow('workflows/production.pkl')

# Get info without loading
from app.ml_engine.utils.serialization import get_model_info
info = get_model_info('models/my_model.pkl')
print(f"Model: {info['model_type']}, Features: {info['n_features']}")
```

**Features:**
- Save/load models with all metadata
- Save/load preprocessing pipelines
- Save/load complete workflows (pipeline + model)
- Compression support (gzip)
- Version tracking and compatibility checks
- Metadata preservation (training info, feature names, configs)
- Overwrite protection
- File integrity checking

**Methods:**
- `save_model(model, path, compression, metadata, overwrite)` - Save model
- `load_model(path, verify_version)` - Load model
- `save_pipeline(pipeline, path, ...)` - Save pipeline
- `load_pipeline(path, ...)` - Load pipeline
- `save_workflow(pipeline, model, path, ...)` - Save workflow
- `load_workflow(path, ...)` - Load workflow
- `get_model_info(path)` - Get model metadata
- `get_pipeline_info(path)` - Get pipeline metadata
- `get_workflow_info(path)` - Get workflow metadata

**Documentation:**
- **[Serialization Guide](utils/SERIALIZATION_GUIDE.md)** - Complete user guide with examples
- **[Implementation Summary](utils/SERIALIZATION_README.md)** - Technical details

---

## 📊 Large Dataset Handling

The ML engine automatically handles large datasets using chunked processing, incremental learning, and memory optimization.

### Automatic Size Detection

The system automatically detects dataset size and routes processing appropriately:

- **Small** (< 100MB, < 100K rows): Standard in-memory processing
- **Medium** (100MB-1GB, 100K-1M rows): Optimized in-memory with dtype optimization
- **Large** (1GB-10GB, 1M-10M rows): Chunked processing
- **Very Large** (> 10GB, > 10M rows): Incremental learning + chunked processing

### Key Components

#### 1. Dataset Size Detection

**Module:** `app/ml_engine/utils/dataset_optimizer.py`

```python
from app.ml_engine.utils.dataset_optimizer import DatasetMetrics, DatasetSize

# Analyze dataset
metrics = DatasetMetrics.from_file('large_dataset.csv')

print(f"Size: {metrics.file_size_mb:.2f}MB")
print(f"Estimated rows: {metrics.estimated_rows:,}")
print(f"Category: {metrics.size_category.value}")
print(f"Memory estimate: {metrics.estimated_memory_mb:.2f}MB")
```

#### 2. Chunked Data Loading

```python
from app.ml_engine.utils.dataset_optimizer import ChunkedDataLoader

loader = ChunkedDataLoader(
    file_path='large_dataset.csv',
    chunk_size=10000,  # Auto-calculated if None
    dtype={'age': 'int32', 'income': 'float32'},  # Optional optimization
    usecols=['age', 'income', 'target']  # Load only needed columns
)

# Process in chunks
for chunk in loader.iter_chunks():
    processed = transform(chunk)
    # Chunk is automatically freed after iteration
```

#### 3. Memory-Efficient Preprocessing

```python
from app.ml_engine.utils.dataset_optimizer import MemoryEfficientPreprocessor

preprocessor = MemoryEfficientPreprocessor(chunk_size=10000)

# Two-pass preprocessing: fit then transform
stats = preprocessor.fit_transform_chunked(
    file_path='input.csv',
    output_path='output.csv',
    steps=[
        ('impute', impute_missing),
        ('scale', standard_scale),
        ('encode', one_hot_encode)
    ],
    target_column='target'
)
```

#### 4. Incremental Learning

For very large datasets, use incremental learning with SGD-based models:

```python
from app.ml_engine.training.incremental_trainer import IncrementalTrainer
from sklearn.linear_model import SGDClassifier

# Create model
model = SGDClassifier(loss='log_loss', warm_start=True)

# Create incremental trainer
trainer = IncrementalTrainer(
    model=model,
    model_type='sgd_classifier',
    chunk_size=10000
)

# Train incrementally
stats = trainer.fit_incremental(
    file_path='very_large_dataset.csv',
    target_column='target',
    validation_split=0.2,
    classes=np.array([0, 1, 2]),
    scale_features=True
)

print(f"Trained on {stats['train_samples']:,} samples")
print(f"Validation score: {stats['validation_score']:.4f}")
```

**Supported Incremental Models:**
- SGDClassifier, SGDRegressor
- PassiveAggressiveClassifier, PassiveAggressiveRegressor
- Perceptron
- MiniBatchKMeans
- MLPClassifier, MLPRegressor (with solver='sgd')

#### 5. Memory Optimization

```python
from app.ml_engine.utils.dataset_optimizer import optimize_dtypes

# Optimize DataFrame dtypes (40-60% memory reduction)
df_optimized = optimize_dtypes(df)
```

### Performance Benchmarks

| Dataset Size | Standard | Optimized | Chunked | Incremental | Recommendation |
|--------------|----------|-----------|---------|-------------|----------------|
| **10MB, 50K rows** | 2s | 1.5s | 3s | N/A | Standard |
| **500MB, 500K rows** | 15s, 1.5GB | 10s, 600MB | 12s, 200MB | N/A | Optimized |
| **5GB, 5M rows** | OOM | 90s, 2.5GB | 120s, 500MB | N/A | Chunked |
| **50GB, 50M rows** | OOM | OOM | Very slow | 600s, 500MB | Incremental |

### Best Practices

#### For Large CSVs (1-10GB)

✅ **DO:**
- Let system auto-detect size and route appropriately
- Use chunked processing for preprocessing
- Consider downsampling for exploration
- Use dtype optimization

❌ **DON'T:**
- Load entire dataset with `pd.read_csv()` without chunks
- Create unnecessary copies of data
- Use default dtypes (int64, float64)

#### For Very Large Datasets (> 10GB)

✅ **DO:**
- Use incremental learning models (SGD-based)
- Process in chunks
- Write intermediate results to disk
- Monitor memory usage

❌ **DON'T:**
- Try to load entire dataset into memory
- Use models without partial_fit support
- Create intermediate DataFrames unnecessarily

### Example: Processing 10GB Dataset

```python
from app.ml_engine.utils.dataset_optimizer import (
    DatasetMetrics,
    MemoryEfficientPreprocessor
)
from app.ml_engine.training.incremental_trainer import IncrementalTrainer
from sklearn.linear_model import SGDClassifier

# 1. Analyze dataset
metrics = DatasetMetrics.from_file('10gb_dataset.csv')
# Output: LARGE, ~10M rows, ~25GB in memory

# 2. Preprocess in chunks
preprocessor = MemoryEfficientPreprocessor(chunk_size=10000)
stats = preprocessor.fit_transform_chunked(
    file_path='10gb_dataset.csv',
    output_path='preprocessed.csv',
    steps=[
        ('impute', impute_fn),
        ('scale', scale_fn)
    ],
    target_column='target'
)

# 3. Train with incremental learning
model = SGDClassifier(warm_start=True)
trainer = IncrementalTrainer(model, 'sgd_classifier')

train_stats = trainer.fit_incremental(
    file_path='preprocessed.csv',
    target_column='target',
    classes=np.array([0, 1])
)

print(f"Trained on {train_stats['train_samples']:,} samples")
print(f"Validation accuracy: {train_stats['validation_score']:.4f}")
```

### Memory Monitoring

```python
import psutil
import os

# Get current process memory
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / (1024 * 1024)
print(f"Memory usage: {memory_mb:.2f}MB")

# Check system memory
mem = psutil.virtual_memory()
print(f"Available RAM: {mem.available / (1024**3):.2f}GB")
print(f"RAM usage: {mem.percent}%")
```

### Troubleshooting

#### Out of Memory Error

**Solutions:**
1. Enable chunked processing (reduce chunk size)
2. Use incremental learning (switch to SGD models)
3. Reduce feature set (load only needed columns)

#### Slow Processing

**Solutions:**
1. Increase chunk size if RAM allows
2. Use faster storage (SSD)
3. Optimize preprocessing steps

#### Inconsistent Results

**Solutions:**
1. Set random seeds
2. Use larger validation sets
3. Ensure global statistics with two-pass preprocessing

### API Integration

The training endpoint automatically handles large datasets:

```python
# POST /api/v1/models/train
{
    "experiment_id": "uuid",
    "dataset_id": "uuid",
    "model_type": "sgd_classifier",  # Auto-detects incremental capability
    "target_column": "target",
    "test_size": 0.2,
    "hyperparameters": {...}
}
```

**Backend automatically:**
1. Detects dataset size
2. Routes to chunked/incremental processing
3. Tracks progress (0-100%)
4. Returns model when complete

---

## �🚧 Coming Soon

- ~~**Models** - Regression, classification, clustering~~ ✅ DONE
- ~~**Training** - Cross-validation, train/test split~~ ✅ DONE
- ~~**Tuning** - Grid search, random search, Bayesian optimization~~ ✅ DONE
- ~~**Evaluation** - Metrics, confusion matrix, ROC curves~~ ✅ DONE
- ~~**Code Generation** - Export pipelines as Python code~~ ✅ DONE
- ~~**Serialization** - Save/load models and pipelines~~ ✅ DONE

## 📖 Documentation

- **[Main README](../../../../README.md)** - Project overview
- **[Backend README](../../../README.md)** - Backend API docs
- **[Correlation Analysis Guide](../../docs/CORRELATION_ANALYSIS_GUIDE.md)** - Detailed correlation docs
- **[Serialization Guide](utils/SERIALIZATION_GUIDE.md)** - Model/pipeline serialization guide

## 📄 License

This project is for educational purposes.
