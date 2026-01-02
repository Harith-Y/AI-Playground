# ML Engine - Practical Examples

Real-world examples and use cases for the ML Engine.

---

## 📊 Example 1: Customer Churn Prediction

Complete pipeline for predicting customer churn.

```python
import pandas as pd
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder
from app.ml_engine.models.classification import ClassificationModel
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('customer_data.csv')

# Separate features and target
X = df.drop('churned', axis=1)
y = df['churned']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create preprocessing pipeline
preprocessing = Pipeline(steps=[
    ('imputer', MeanImputer()),
    ('scaler', StandardScaler()),
    ('encoder', OneHotEncoder())
])

# Fit and transform
X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)

# Train model
model = ClassificationModel(model_type='random_forest_classifier')
model.fit(X_train_processed, y_train)

# Make predictions
y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)

# Evaluate
metrics = ClassificationMetrics(y_test, y_pred)
print(f"Accuracy: {metrics.accuracy():.4f}")
print(f"Precision: {metrics.precision():.4f}")
print(f"Recall: {metrics.recall():.4f}")
print(f"F1 Score: {metrics.f1_score():.4f}")
print(f"ROC AUC: {metrics.roc_auc_score(y_proba):.4f}")

# Save artifacts
preprocessing.save('churn_preprocessing.pkl')
from app.ml_engine.utils.serialization import save_model
save_model(model, 'churn_model.pkl', metadata={'accuracy': metrics.accuracy()})
```

---

## 🏠 Example 2: House Price Prediction

Regression pipeline with feature engineering.

```python
import pandas as pd
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MedianImputer
from app.ml_engine.preprocessing.scaler import RobustScaler
from app.ml_engine.preprocessing.cleaner import IQROutlierDetector
from app.ml_engine.feature_selection.correlation_selector import CorrelationSelector
from app.ml_engine.models.regression import RegressionModel
from app.ml_engine.evaluation.regression_metrics import RegressionMetrics
from app.ml_engine.tuning.grid_search import run_grid_search
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('house_prices.csv')

# Remove outliers
outlier_detector = IQROutlierDetector(threshold=1.5)
outlier_mask = outlier_detector.fit_transform(df)
df_clean = df[~outlier_mask]

# Separate features and target
X = df_clean.drop('price', axis=1)
y = df_clean['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
preprocessing = Pipeline(steps=[
    ('imputer', MedianImputer()),
    ('scaler', RobustScaler())
])

X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)

# Feature selection
selector = CorrelationSelector(k=15, method='pearson')
X_train_selected = selector.fit_transform(X_train_processed, y_train)
X_test_selected = selector.transform(X_test_processed)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

from sklearn.ensemble import RandomForestRegressor
tuning_result = run_grid_search(
    estimator=RandomForestRegressor(),
    param_grid=param_grid,
    X=X_train_selected,
    y=y_train,
    cv=5,
    scoring='r2'
)

print(f"Best parameters: {tuning_result.best_params}")
print(f"Best R² score: {tuning_result.best_score:.4f}")

# Train final model with best parameters
model = RegressionModel(model_type='random_forest_regressor')
model.model.set_params(**tuning_result.best_params)
model.fit(X_train_selected, y_train)

# Evaluate
y_pred = model.predict(X_test_selected)
metrics = RegressionMetrics(y_test, y_pred)

print(f"R² Score: {metrics.r2_score():.4f}")
print(f"RMSE: {metrics.rmse():.2f}")
print(f"MAE: {metrics.mae():.2f}")

# Save complete workflow
from app.ml_engine.utils.serialization import save_workflow
save_workflow(
    pipeline=preprocessing,
    model=model,
    path='house_price_workflow.pkl',
    workflow_name='HousePricePredictor_v1',
    metadata={
        'r2_score': metrics.r2_score(),
        'rmse': metrics.rmse(),
        'selected_features': selector.selected_features_
    }
)
```

---

## 🛒 Example 3: Customer Segmentation

Clustering analysis for customer segmentation.

```python
import pandas as pd
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.models.clustering import ClusteringModel
from app.ml_engine.evaluation.clustering_metrics import ClusteringMetrics
from app.ml_engine.eda_statistics import EDAStatistics

# Load data
df = pd.read_csv('customer_data.csv')

# EDA
eda = EDAStatistics(df)
summary = eda.quick_summary()
print(summary)

# Preprocessing
preprocessing = Pipeline(steps=[
    ('imputer', MeanImputer()),
    ('scaler', StandardScaler())
])

X_processed = preprocessing.fit_transform(df)

# Try different numbers of clusters
from sklearn.metrics import silhouette_score

best_k = 0
best_score = -1

for k in range(2, 11):
    model = ClusteringModel(model_type='kmeans', n_clusters=k)
    labels = model.fit_predict(X_processed)
    score = silhouette_score(X_processed, labels)
    
    print(f"k={k}: Silhouette Score = {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nBest number of clusters: {best_k}")

# Train final model
model = ClusteringModel(model_type='kmeans', n_clusters=best_k)
labels = model.fit_predict(X_processed)

# Evaluate
metrics = ClusteringMetrics(X_processed, labels)
print(f"Silhouette Score: {metrics.silhouette_score():.4f}")
print(f"Davies-Bouldin Index: {metrics.davies_bouldin_score():.4f}")
print(f"Calinski-Harabasz Score: {metrics.calinski_harabasz_score():.2f}")

# Add cluster labels to original data
df['cluster'] = labels

# Analyze clusters
for cluster_id in range(best_k):
    cluster_data = df[df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_data)} customers):")
    print(cluster_data.describe())

# Save model
from app.ml_engine.utils.serialization import save_model
save_model(
    model,
    'customer_segmentation.pkl',
    metadata={
        'n_clusters': best_k,
        'silhouette_score': metrics.silhouette_score()
    }
)
```

---

## 📧 Example 4: Spam Detection

Text classification with preprocessing.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from app.ml_engine.models.classification import ClassificationModel
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics
from app.ml_engine.tuning.random_search import run_random_search
from sklearn.model_selection import train_test_split
from scipy.stats import randint

# Load data
df = pd.read_csv('emails.csv')

# Text preprocessing (using sklearn's TfidfVectorizer)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['is_spam']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning with random search
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10)
}

from sklearn.ensemble import RandomForestClassifier
tuning_result = run_random_search(
    estimator=RandomForestClassifier(),
    param_distributions=param_distributions,
    X=X_train,
    y=y_train,
    n_iter=30,
    cv=5,
    scoring='f1'
)

print(f"Best parameters: {tuning_result.best_params}")
print(f"Best F1 score: {tuning_result.best_score:.4f}")

# Train final model
model = ClassificationModel(model_type='random_forest_classifier')
model.model.set_params(**tuning_result.best_params)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

metrics = ClassificationMetrics(y_test, y_pred)
print(f"\nTest Set Performance:")
print(f"Accuracy: {metrics.accuracy():.4f}")
print(f"Precision: {metrics.precision():.4f}")
print(f"Recall: {metrics.recall():.4f}")
print(f"F1 Score: {metrics.f1_score():.4f}")

# Confusion matrix
cm = metrics.confusion_matrix()
print(f"\nConfusion Matrix:")
print(cm)

# Feature importance (top words)
from app.ml_engine.evaluation.feature_importance import FeatureImportance
feature_names = vectorizer.get_feature_names_out()
importance = FeatureImportance(model.model, feature_names)
top_features = importance.get_top_features(n=20)

print(f"\nTop 20 important words:")
for feature, score in top_features:
    print(f"  {feature}: {score:.4f}")
```

---

## 💳 Example 5: Credit Card Fraud Detection

Imbalanced classification with SMOTE.

```python
import pandas as pd
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.oversampling import SMOTE
from app.ml_engine.models.classification import ClassificationModel
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics
from app.ml_engine.class_distribution_analysis import ClassDistributionAnalysis
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('credit_card_transactions.csv')

# Analyze class distribution
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

dist_analysis = ClassDistributionAnalysis(y)
print(f"Class distribution:")
print(dist_analysis.get_class_counts())
print(f"Imbalance ratio: {dist_analysis.get_imbalance_ratio():.2f}")
print(f"Is imbalanced: {dist_analysis.is_imbalanced()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
preprocessing = Pipeline(steps=[
    ('imputer', MeanImputer()),
    ('scaler', StandardScaler())
])

X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)

# Handle imbalance with SMOTE
smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed, y_train
)

print(f"\nAfter SMOTE:")
print(f"Training samples: {len(X_train_balanced)}")
print(f"Class distribution: {pd.Series(y_train_balanced).value_counts()}")

# Train model
model = ClassificationModel(model_type='gradient_boosting_classifier')
model.fit(X_train_balanced, y_train_balanced)

# Evaluate
y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)

metrics = ClassificationMetrics(y_test, y_pred)
print(f"\nTest Set Performance:")
print(f"Accuracy: {metrics.accuracy():.4f}")
print(f"Precision: {metrics.precision():.4f}")
print(f"Recall: {metrics.recall():.4f}")
print(f"F1 Score: {metrics.f1_score():.4f}")
print(f"ROC AUC: {metrics.roc_auc_score(y_proba):.4f}")

# Confusion matrix
cm = metrics.confusion_matrix()
print(f"\nConfusion Matrix:")
print(cm)

# Classification report
report = metrics.classification_report()
print(f"\nClassification Report:")
print(report)

# Save workflow
from app.ml_engine.utils.serialization import save_workflow
save_workflow(
    pipeline=preprocessing,
    model=model,
    path='fraud_detection_workflow.pkl',
    workflow_name='FraudDetector_v1',
    metadata={
        'f1_score': metrics.f1_score(),
        'roc_auc': metrics.roc_auc_score(y_proba),
        'used_smote': True
    }
)
```

---

## 📈 Example 6: Time Series Forecasting

Sales prediction with feature engineering.

```python
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.models.regression import RegressionModel
from app.ml_engine.evaluation.regression_metrics import RegressionMetrics

# Load data
df = pd.read_csv('sales_data.csv', parse_dates=['date'])
df = df.sort_values('date')

# Feature engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['quarter'] = df['date'].dt.quarter

# Lag features
for lag in [1, 7, 30]:
    df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

# Rolling statistics
df['sales_rolling_mean_7'] = df['sales'].rolling(window=7).mean()
df['sales_rolling_std_7'] = df['sales'].rolling(window=7).std()
df['sales_rolling_mean_30'] = df['sales'].rolling(window=30).mean()

# Drop NaN rows created by lag features
df = df.dropna()

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['date', 'sales']]
X = df[feature_cols]
y = df['sales']

# Time-based split (last 20% for testing)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Preprocessing
preprocessing = Pipeline(steps=[
    ('scaler', StandardScaler())
])

X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)

# Train model
model = RegressionModel(model_type='gradient_boosting_regressor')
model.fit(X_train_processed, y_train)

# Evaluate
y_pred = model.predict(X_test_processed)
metrics = RegressionMetrics(y_test, y_pred)

print(f"R² Score: {metrics.r2_score():.4f}")
print(f"RMSE: {metrics.rmse():.2f}")
print(f"MAE: {metrics.mae():.2f}")
print(f"MAPE: {metrics.mape():.2f}%")

# Feature importance
from app.ml_engine.evaluation.feature_importance import FeatureImportance
importance = FeatureImportance(model.model, feature_cols)
top_features = importance.get_top_features(n=10)

print(f"\nTop 10 important features:")
for feature, score in top_features:
    print(f"  {feature}: {score:.4f}")

# Save model
from app.ml_engine.utils.serialization import save_model
save_model(
    model,
    'sales_forecasting.pkl',
    metadata={
        'r2_score': metrics.r2_score(),
        'rmse': metrics.rmse(),
        'features': feature_cols
    }
)
```

---

## 🎯 Example 7: Multi-Class Classification

Image classification (using pre-extracted features).

```python
import pandas as pd
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.feature_selection.variance_threshold import VarianceThresholdSelector
from app.ml_engine.models.classification import ClassificationModel
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics
from sklearn.model_selection import train_test_split

# Load data (pre-extracted image features)
df = pd.read_csv('image_features.csv')

# Separate features and target
X = df.drop('category', axis=1)
y = df['category']

print(f"Number of classes: {y.nunique()}")
print(f"Class distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
preprocessing = Pipeline(steps=[
    ('variance_selector', VarianceThresholdSelector(threshold=0.01)),
    ('scaler', StandardScaler())
])

X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)

print(f"Features after selection: {X_train_processed.shape[1]}")

# Train model
model = ClassificationModel(model_type='random_forest_classifier')
model.fit(X_train_processed, y_train)

# Evaluate
y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)

metrics = ClassificationMetrics(y_test, y_pred)
print(f"\nOverall Performance:")
print(f"Accuracy: {metrics.accuracy():.4f}")
print(f"Precision (weighted): {metrics.precision():.4f}")
print(f"Recall (weighted): {metrics.recall():.4f}")
print(f"F1 Score (weighted): {metrics.f1_score():.4f}")

# Per-class metrics
report = metrics.classification_report()
print(f"\nPer-Class Metrics:")
print(report)

# Confusion matrix
cm = metrics.confusion_matrix()
print(f"\nConfusion Matrix:")
print(cm)

# Save complete workflow
from app.ml_engine.utils.serialization import save_workflow
save_workflow(
    pipeline=preprocessing,
    model=model,
    path='image_classification_workflow.pkl',
    workflow_name='ImageClassifier_v1',
    metadata={
        'accuracy': metrics.accuracy(),
        'n_classes': y.nunique(),
        'n_features': X_train_processed.shape[1]
    }
)
```

---

## 🔄 Example 8: Incremental Learning

Online learning for streaming data.

```python
import pandas as pd
import numpy as np
from app.ml_engine.training.incremental_trainer import IncrementalTrainer
from app.ml_engine.models.classification import ClassificationModel
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics

# Initialize incremental trainer
trainer = IncrementalTrainer(
    model_type='sgd_classifier',  # Supports partial_fit
    batch_size=1000,
    max_memory_mb=500
)

# Simulate streaming data
def data_stream():
    """Generator that yields batches of data"""
    for i in range(10):  # 10 batches
        # Simulate loading a batch
        batch_size = 1000
        X_batch = np.random.randn(batch_size, 20)
        y_batch = np.random.randint(0, 2, batch_size)
        yield X_batch, y_batch

# Train incrementally
print("Training incrementally...")
for batch_idx, (X_batch, y_batch) in enumerate(data_stream()):
    trainer.partial_fit(X_batch, y_batch)
    print(f"Processed batch {batch_idx + 1}")

# Evaluate on test set
X_test = np.random.randn(500, 20)
y_test = np.random.randint(0, 2, 500)

y_pred = trainer.predict(X_test)
metrics = ClassificationMetrics(y_test, y_pred)

print(f"\nFinal Performance:")
print(f"Accuracy: {metrics.accuracy():.4f}")
print(f"F1 Score: {metrics.f1_score():.4f}")

# Save model
trainer.save_model('incremental_model.pkl')
```

---

## 📊 Example 9: Feature Engineering Pipeline

Advanced feature engineering workflow.

```python
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import RobustScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder
from app.ml_engine.feature_selection.mutual_information_selector import MutualInformationSelector
from app.ml_engine.correlation_analysis import CorrelationMatrix
from app.ml_engine.eda_statistics import EDAStatistics

# Load data
df = pd.read_csv('data.csv')

# Step 1: EDA
print("=== Exploratory Data Analysis ===")
eda = EDAStatistics(df)
summary = eda.quick_summary()
print(summary)

# Check for data quality issues
issues = eda.detect_data_issues()
if issues:
    print(f"\nData Quality Issues Found:")
    for issue_type, details in issues.items():
        print(f"  {issue_type}: {details}")

# Step 2: Correlation Analysis
print("\n=== Correlation Analysis ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = CorrelationMatrix(df[numeric_cols])

# Detect multicollinearity
multi_stats = corr_matrix.get_multicollinearity_stats(threshold=0.9)
if multi_stats['high_correlation_count'] > 0:
    print(f"Found {multi_stats['high_correlation_count']} highly correlated pairs")
    print(f"Problematic features: {multi_stats['problematic_features']}")
    
    # Get removal recommendations
    from app.ml_engine.correlation_analysis import recommend_feature_removal
    recommendations = recommend_feature_removal(df[numeric_cols], threshold=0.9)
    print(f"Recommended features to remove: {recommendations}")

# Step 3: Feature Engineering
print("\n=== Feature Engineering ===")

# Create interaction features
df['feature1_x_feature2'] = df['feature1'] * df['feature2']
df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-10)

# Create polynomial features
df['feature1_squared'] = df['feature1'] ** 2
df['feature1_cubed'] = df['feature1'] ** 3

# Create binned features
df['feature1_binned'] = pd.cut(df['feature1'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])

# Step 4: Preprocessing Pipeline
print("\n=== Preprocessing Pipeline ===")

X = df.drop('target', axis=1)
y = df['target']

preprocessing = Pipeline(steps=[
    ('imputer', MeanImputer()),
    ('scaler', RobustScaler()),
    ('encoder', OneHotEncoder())
])

X_processed = preprocessing.fit_transform(X)
print(f"Shape after preprocessing: {X_processed.shape}")

# Step 5: Feature Selection
print("\n=== Feature Selection ===")

selector = MutualInformationSelector(k=20, task='classification')
X_selected = selector.fit_transform(X_processed, y)

print(f"Selected {len(selector.selected_features_)} features")
print(f"Top 10 features by MI score:")
sorted_features = sorted(
    selector.mi_scores_.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]
for feature, score in sorted_features:
    print(f"  {feature}: {score:.4f}")

# Save preprocessing pipeline
preprocessing.save('feature_engineering_pipeline.pkl')
print("\nPipeline saved successfully!")
```

---

## 🎓 Example 10: Complete ML Workflow

End-to-end workflow with all components.

```python
import pandas as pd
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder
from app.ml_engine.feature_selection.correlation_selector import CorrelationSelector
from app.ml_engine.models.classification import ClassificationModel
from app.ml_engine.tuning.grid_search import run_grid_search
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics
from app.ml_engine.code_generation.generator import generate_complete_pipeline
from app.ml_engine.utils.serialization import save_workflow
from sklearn.model_selection import train_test_split

# 1. Load and explore data
df = pd.read_csv('data.csv')
print(f"Dataset shape: {df.shape}")

# 2. Prepare data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessing
preprocessing = Pipeline(steps=[
    ('imputer', MeanImputer()),
    ('scaler', StandardScaler()),
    ('encoder', OneHotEncoder())
])

X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)

# 4. Feature selection
selector = CorrelationSelector(k=15, method='pearson')
X_train_selected = selector.fit_transform(X_train_processed, y_train)
X_test_selected = selector.transform(X_test_processed)

# 5. Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}

from sklearn.ensemble import RandomForestClassifier
tuning_result = run_grid_search(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    X=X_train_selected,
    y=y_train,
    cv=5
)

# 6. Train final model
model = ClassificationModel(model_type='random_forest_classifier')
model.model.set_params(**tuning_result.best_params)
model.fit(X_train_selected, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_selected)
y_proba = model.predict_proba(X_test_selected)

metrics = ClassificationMetrics(y_test, y_pred)
print(f"\nFinal Model Performance:")
print(f"Accuracy: {metrics.accuracy():.4f}")
print(f"F1 Score: {metrics.f1_score():.4f}")
print(f"ROC AUC: {metrics.roc_auc_score(y_proba):.4f}")

# 8. Save workflow
save_workflow(
    pipeline=preprocessing,
    model=model,
    path='complete_workflow.pkl',
    workflow_name='CompleteMLWorkflow_v1',
    metadata={
        'accuracy': metrics.accuracy(),
        'f1_score': metrics.f1_score(),
        'best_params': tuning_result.best_params,
        'selected_features': selector.selected_features_
    }
)

# 9. Generate deployment code
code = generate_complete_pipeline(
    preprocessing_steps=['imputer', 'scaler', 'encoder'],
    model_type='random_forest_classifier',
    hyperparameters=tuning_result.best_params,
    include_evaluation=True
)

with open('deployment_code.py', 'w') as f:
    f.write(code)

print("\nWorkflow complete! All artifacts saved.")
```

