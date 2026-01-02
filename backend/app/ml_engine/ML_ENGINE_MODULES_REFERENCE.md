# ML Engine - Modules Reference

Complete reference for all ML Engine modules.

---

## 📊 Preprocessing Modules

### 1. Imputation

#### MeanImputer
Fill missing numeric values with mean.

```python
from app.ml_engine.preprocessing.imputer import MeanImputer

imputer = MeanImputer()
df_filled = imputer.fit_transform(df)
```

**Parameters:**
- None

**Methods:**
- `fit(X, y=None)` - Compute means
- `transform(X)` - Fill missing values
- `fit_transform(X, y=None)` - Fit and transform

#### MedianImputer
Fill missing numeric values with median.

```python
from app.ml_engine.preprocessing.imputer import MedianImputer

imputer = MedianImputer()
df_filled = imputer.fit_transform(df)
```

#### ModeImputer
Fill missing categorical values with mode.

```python
from app.ml_engine.preprocessing.imputer import ModeImputer

imputer = ModeImputer()
df_filled = imputer.fit_transform(df)
```

---

### 2. Scaling

#### StandardScaler
Standardize features (mean=0, std=1).

```python
from app.ml_engine.preprocessing.scaler import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

**Parameters:**
- `with_mean` (bool, default=True): Center data
- `with_std` (bool, default=True): Scale to unit variance

#### MinMaxScaler
Scale features to a given range.

```python
from app.ml_engine.preprocessing.scaler import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)
```

**Parameters:**
- `feature_range` (tuple, default=(0, 1)): Desired range

#### RobustScaler
Scale using statistics robust to outliers.

```python
from app.ml_engine.preprocessing.scaler import RobustScaler

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)
```

**Parameters:**
- `with_centering` (bool, default=True): Center data
- `with_scaling` (bool, default=True): Scale data
- `quantile_range` (tuple, default=(25.0, 75.0)): Quantile range

---

### 3. Encoding

#### OneHotEncoder
Convert categorical variables to binary columns.

```python
from app.ml_engine.preprocessing.encoder import OneHotEncoder

encoder = OneHotEncoder()
df_encoded = encoder.fit_transform(df)
```

**Parameters:**
- `drop` (str, default=None): Drop strategy ('first', 'if_binary', None)
- `handle_unknown` (str, default='ignore'): How to handle unknown categories

#### LabelEncoder
Convert categorical variables to integers.

```python
from app.ml_engine.preprocessing.encoder import LabelEncoder

encoder = LabelEncoder()
df_encoded = encoder.fit_transform(df)
```

#### OrdinalEncoder
Convert categorical variables to ordinal integers.

```python
from app.ml_engine.preprocessing.encoder import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df_encoded = encoder.fit_transform(df)
```

**Parameters:**
- `categories` (list, default='auto'): Category order

---

### 4. Outlier Detection

#### IQROutlierDetector
Detect outliers using Interquartile Range.

```python
from app.ml_engine.preprocessing.cleaner import IQROutlierDetector

detector = IQROutlierDetector(threshold=1.5)
outlier_mask = detector.fit_transform(df)
df_clean = df[~outlier_mask]
```

**Parameters:**
- `threshold` (float, default=1.5): IQR multiplier
- `columns` (list, optional): Columns to check

**Returns:**
- Boolean mask (True = outlier)

#### ZScoreOutlierDetector
Detect outliers using Z-score method.

```python
from app.ml_engine.preprocessing.cleaner import ZScoreOutlierDetector

detector = ZScoreOutlierDetector(threshold=3.0)
outlier_mask = detector.fit_transform(df)
```

**Parameters:**
- `threshold` (float, default=3.0): Number of standard deviations
- `columns` (list, optional): Columns to check

---

### 5. Sampling

#### SMOTE (Oversampling)
Synthetic Minority Over-sampling Technique.

```python
from app.ml_engine.preprocessing.oversampling import SMOTE

smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Parameters:**
- `sampling_strategy` (str/dict, default='auto'): Sampling strategy
- `k_neighbors` (int, default=5): Number of nearest neighbors
- `random_state` (int, default=42): Random seed

#### RandomUnderSampler
Random under-sampling of majority class.

```python
from app.ml_engine.preprocessing.undersampling import RandomUnderSampler

sampler = RandomUnderSampler(sampling_strategy='auto')
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

---

## 🎯 Feature Selection Modules

### 1. VarianceThresholdSelector
Remove low-variance features.

```python
from app.ml_engine.feature_selection.variance_threshold import VarianceThresholdSelector

selector = VarianceThresholdSelector(threshold=0.01)
X_selected = selector.fit_transform(X)

# Get selected features
print(selector.selected_features_)
print(selector.variances_)
```

**Parameters:**
- `threshold` (float, default=0.0): Variance threshold

**Attributes:**
- `selected_features_` - List of selected feature names
- `variances_` - Dictionary of feature variances

---

### 2. CorrelationSelector
Select features based on correlation with target.

```python
from app.ml_engine.feature_selection.correlation_selector import CorrelationSelector

# Select top k features
selector = CorrelationSelector(k=10, method='pearson')
X_selected = selector.fit_transform(X, y)

# Or use threshold
selector = CorrelationSelector(threshold=0.5, method='spearman')
X_selected = selector.fit_transform(X, y)

# Get correlation scores
print(selector.correlation_scores_)
```

**Parameters:**
- `k` (int, optional): Number of top features
- `threshold` (float, optional): Minimum correlation
- `method` (str, default='pearson'): 'pearson', 'spearman', 'kendall'

**Attributes:**
- `correlation_scores_` - Dictionary of correlation scores
- `selected_features_` - List of selected features

---

### 3. MutualInformationSelector
Select features using mutual information.

```python
from app.ml_engine.feature_selection.mutual_information_selector import MutualInformationSelector

# Classification
selector = MutualInformationSelector(k=10, task='classification')
X_selected = selector.fit_transform(X, y)

# Regression
selector = MutualInformationSelector(threshold=0.1, task='regression')
X_selected = selector.fit_transform(X, y)

# Get MI scores
print(selector.mi_scores_)
```

**Parameters:**
- `k` (int, optional): Number of top features
- `threshold` (float, optional): Minimum MI score
- `task` (str, default='classification'): 'classification' or 'regression'
- `random_state` (int, default=42): Random seed

---

### 4. RFESelector
Recursive Feature Elimination.

```python
from app.ml_engine.feature_selection.rfe_selector import RFESelector
from sklearn.ensemble import RandomForestClassifier

selector = RFESelector(
    estimator=RandomForestClassifier(),
    n_features_to_select=10,
    step=1
)
X_selected = selector.fit_transform(X, y)
```

**Parameters:**
- `estimator` - Estimator with feature_importances_ or coef_
- `n_features_to_select` (int, optional): Number of features
- `step` (int/float, default=1): Features to remove per iteration

---

### 5. UnivariateSelector
Select features using univariate statistical tests.

```python
from app.ml_engine.feature_selection.univariate_selector import UnivariateSelector

# Classification (chi2, f_classif, mutual_info_classif)
selector = UnivariateSelector(score_func='f_classif', k=10)
X_selected = selector.fit_transform(X, y)

# Regression (f_regression, mutual_info_regression)
selector = UnivariateSelector(score_func='f_regression', k=10)
X_selected = selector.fit_transform(X, y)
```

**Parameters:**
- `score_func` (str): Statistical test function
- `k` (int, optional): Number of top features
- `percentile` (float, optional): Percentile of top features

---

## 🤖 Model Modules

### Classification Models

```python
from app.ml_engine.models.classification import ClassificationModel

# Available models:
models = [
    'random_forest_classifier',
    'logistic_regression',
    'svm_classifier',
    'knn_classifier',
    'gradient_boosting_classifier',
    'decision_tree_classifier'
]

# Create and train
model = ClassificationModel(model_type='random_forest_classifier')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
score = model.score(X_test, y_test)
```

**Methods:**
- `fit(X, y)` - Train model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Predict probabilities
- `score(X, y)` - Compute accuracy
- `get_params()` - Get model parameters
- `set_params(**params)` - Set model parameters

---

### Regression Models

```python
from app.ml_engine.models.regression import RegressionModel

# Available models:
models = [
    'random_forest_regressor',
    'linear_regression',
    'gradient_boosting_regressor',
    'ridge_regression'
]

# Create and train
model = RegressionModel(model_type='random_forest_regressor')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
score = model.score(X_test, y_test)  # R² score
```

---

### Clustering Models

```python
from app.ml_engine.models.clustering import ClusteringModel

# Available models:
models = [
    'kmeans',
    'dbscan',
    'hierarchical',
    'gaussian_mixture'
]

# Create and fit
model = ClusteringModel(model_type='kmeans', n_clusters=3)
model.fit(X)

# Predict clusters
labels = model.predict(X)
```

---

## 📈 Evaluation Modules

### Classification Metrics

```python
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics

metrics = ClassificationMetrics(y_true, y_pred)

# Basic metrics
print(f"Accuracy: {metrics.accuracy()}")
print(f"Precision: {metrics.precision()}")
print(f"Recall: {metrics.recall()}")
print(f"F1 Score: {metrics.f1_score()}")

# Advanced metrics
print(f"ROC AUC: {metrics.roc_auc_score(y_proba)}")
print(f"Log Loss: {metrics.log_loss(y_proba)}")

# Confusion matrix
cm = metrics.confusion_matrix()

# Classification report
report = metrics.classification_report()
```

---

### Regression Metrics

```python
from app.ml_engine.evaluation.regression_metrics import RegressionMetrics

metrics = RegressionMetrics(y_true, y_pred)

# Basic metrics
print(f"R² Score: {metrics.r2_score()}")
print(f"RMSE: {metrics.rmse()}")
print(f"MAE: {metrics.mae()}")
print(f"MSE: {metrics.mse()}")

# Advanced metrics
print(f"MAPE: {metrics.mape()}")
print(f"Explained Variance: {metrics.explained_variance()}")
```

---

### Feature Importance

```python
from app.ml_engine.evaluation.feature_importance import FeatureImportance

# Get feature importance
importance = FeatureImportance(model, feature_names)
scores = importance.get_importance_scores()

# Plot importance
importance.plot_importance(top_n=10)

# Get top features
top_features = importance.get_top_features(n=10)
```

---

## 🔧 Tuning Modules

### Grid Search

```python
from app.ml_engine.tuning.grid_search import run_grid_search

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

result = run_grid_search(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    X=X_train,
    y=y_train,
    cv=5,
    scoring='accuracy'
)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score}")
```

---

### Random Search

```python
from app.ml_engine.tuning.random_search import run_random_search
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10)
}

result = run_random_search(
    estimator=RandomForestClassifier(),
    param_distributions=param_distributions,
    X=X_train,
    y=y_train,
    n_iter=50,
    cv=5
)
```

---

### Bayesian Optimization

```python
from app.ml_engine.tuning.bayesian import run_bayesian_optimization

param_space = {
    'n_estimators': (50, 200),
    'max_depth': (5, 20),
    'min_samples_split': (2, 10)
}

result = run_bayesian_optimization(
    estimator=RandomForestClassifier(),
    param_space=param_space,
    X=X_train,
    y=y_train,
    n_iter=30,
    cv=5
)
```

---

## 💾 Serialization

### Save and Load Models

```python
from app.ml_engine.utils.serialization import save_model, load_model

# Save model
save_model(
    model=trained_model,
    path='models/my_model.pkl',
    compression=True,
    metadata={'version': '1.0', 'accuracy': 0.95}
)

# Load model
model = load_model('models/my_model.pkl')
predictions = model.predict(X_test)

# Get model info
from app.ml_engine.utils.serialization import get_model_info
info = get_model_info('models/my_model.pkl')
```

---

### Save and Load Pipelines

```python
from app.ml_engine.utils.serialization import save_pipeline, load_pipeline

# Save pipeline
save_pipeline(
    pipeline=fitted_pipeline,
    path='pipelines/preprocessing.pkl',
    compression=True
)

# Load pipeline
pipeline = load_pipeline('pipelines/preprocessing.pkl')
X_transformed = pipeline.transform(X_test)
```

---

### Save and Load Workflows

```python
from app.ml_engine.utils.serialization import save_workflow, load_workflow

# Save complete workflow
save_workflow(
    pipeline=preprocessing_pipeline,
    model=trained_model,
    path='workflows/production.pkl',
    workflow_name='CustomerChurn_v1',
    metadata={'accuracy': 0.95, 'f1': 0.93}
)

# Load workflow
pipeline, model = load_workflow('workflows/production.pkl')

# Use workflow
X_processed = pipeline.transform(X_new)
predictions = model.predict(X_processed)
```

---

## 📊 Analysis Modules

### EDA Statistics

```python
from app.ml_engine.eda_statistics import EDAStatistics

eda = EDAStatistics(df)

# Quick summary
summary = eda.quick_summary()

# Distribution analysis
dist = eda.analyze_distribution('age')

# Missing values
missing = eda.analyze_missing_values()

# Outliers
outliers = eda.detect_outliers(method='iqr')

# Data quality
issues = eda.detect_data_issues()
```

---

### Correlation Analysis

```python
from app.ml_engine.correlation_analysis import CorrelationMatrix

corr_matrix = CorrelationMatrix(df)

# Compute correlation
corr = corr_matrix.compute_correlation(method='pearson')

# Get high correlation pairs
pairs = corr_matrix.get_correlation_pairs(threshold=0.8)

# Detect multicollinearity
multi = corr_matrix.get_multicollinearity_stats(threshold=0.9)

# Get heatmap data
heatmap = corr_matrix.get_heatmap_data()
```

---

## 🎨 Code Generation

### Generate Training Code

```python
from app.ml_engine.code_generation.training_generator import generate_training_code

code = generate_training_code(
    model_type='random_forest_classifier',
    hyperparameters={'n_estimators': 100, 'max_depth': 10},
    output_format='script'
)

with open('train.py', 'w') as f:
    f.write(code)
```

---

### Generate Prediction Code

```python
from app.ml_engine.code_generation.prediction_generator import generate_prediction_code

code = generate_prediction_code(
    model_type='random_forest_classifier',
    preprocessing_steps=['imputer', 'scaler'],
    output_format='function'
)
```

---

### Generate Complete Pipeline

```python
from app.ml_engine.code_generation.generator import generate_complete_pipeline

code = generate_complete_pipeline(
    preprocessing_steps=['imputer', 'scaler', 'encoder'],
    model_type='random_forest_classifier',
    hyperparameters={'n_estimators': 100},
    include_evaluation=True
)
```

