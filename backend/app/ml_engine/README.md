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
│   └── pipeline.py            # Pipeline orchestration (Placeholder)
├── feature_selection/          # Feature engineering modules ✅
│   ├── variance_threshold.py  # Variance-based selection (319 lines)
│   ├── correlation_selector.py # Correlation-based selection (386 lines)
│   └── mutual_information_selector.py  # MI selection (390 lines)
├── eda_statistics.py           # EDA analysis module ✅ (514 lines)
├── correlation_analysis.py     # Correlation matrices ✅ (535 lines)
├── models/                     # ML models (Placeholder)
├── training/                   # Training utilities (Placeholder)
├── tuning/                     # Hyperparameter optimization (Placeholder)
├── evaluation/                 # Model evaluation (Placeholder)
└── code_generation/            # Code export (Placeholder)
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

## 🚧 Coming Soon

- **Models** - Regression, classification, clustering
- **Training** - Cross-validation, train/test split
- **Tuning** - Grid search, random search, Bayesian optimization
- **Evaluation** - Metrics, confusion matrix, ROC curves
- **Code Generation** - Export pipelines as Python code

## 📖 Documentation

- **[Main README](../../../../README.md)** - Project overview
- **[Backend README](../../../README.md)** - Backend API docs
- **[Correlation Analysis Guide](../../docs/CORRELATION_ANALYSIS_GUIDE.md)** - Detailed correlation docs

## 📄 License

This project is for educational purposes.
