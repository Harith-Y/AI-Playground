# Evaluation Module

Comprehensive model evaluation metrics and visualizations for machine learning models.

## 📋 Overview

The evaluation module provides tools to assess model performance across different ML tasks:

- ✅ **Classification Metrics** - Accuracy, precision, recall, F1, AUC-ROC, confusion matrix
- 🚧 **Regression Metrics** - MAE, MSE, RMSE, R², MAPE (Coming Soon)
- ✅ **Residual Analysis Utilities** - Residual stats, standardized residuals, outlier detection
- 🚧 **Clustering Metrics** - Silhouette score, inertia, Davies-Bouldin index (Coming Soon)
- 🚧 **Visualizations** - ROC curves, PR curves, confusion matrices, residual plots (Coming Soon)

## 🎯 Classification Metrics (ML-46)

### Features

#### Basic Metrics

- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **Precision**: Positive predictive value (TP / (TP + FP))
- **Recall**: Sensitivity/True positive rate (TP / (TP + FN))
- **F1 Score**: Harmonic mean of precision and recall

#### Probability-Based Metrics

- **AUC-ROC**: Area under Receiver Operating Characteristic curve
- **AUC-PR**: Area under Precision-Recall curve
- **Log Loss**: Logarithmic loss for probability predictions

#### Advanced Metrics

- **Balanced Accuracy**: Average recall per class (handles imbalanced data)
- **Matthews Correlation Coefficient (MCC)**: Correlation between predictions and truth
- **Cohen's Kappa**: Agreement between predictions and truth, accounting for chance

#### Multi-Class Support

- **Averaging Strategies**: binary, micro, macro, weighted
- **Per-Class Metrics**: Individual metrics for each class
- **Confusion Matrix**: With optional normalization

### Quick Start

```python
from app.ml_engine.evaluation import calculate_classification_metrics

# Binary classification
y_true = [0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1, 0]
y_proba = [
    [0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2],
    [0.1, 0.9], [0.4, 0.6], [0.3, 0.7], [0.7, 0.3]
]

metrics = calculate_classification_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_proba=y_proba,
    average='binary'
)

print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall: {metrics.recall:.4f}")
print(f"F1 Score: {metrics.f1_score:.4f}")
print(f"AUC-ROC: {metrics.auc_roc:.4f}")
```

### Usage Examples

#### Example 1: Binary Classification

```python
from app.ml_engine.evaluation import ClassificationMetricsCalculator

# Create calculator
calculator = ClassificationMetricsCalculator(average='binary')

# Calculate metrics
metrics = calculator.calculate_metrics(
    y_true=[0, 1, 1, 0, 1],
    y_pred=[0, 1, 0, 0, 1],
    y_proba=[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]]
)

# Access metrics
print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"AUC-ROC: {metrics.auc_roc:.4f}")
print(f"Confusion Matrix:\n{metrics.confusion_matrix}")
```

#### Example 2: Multi-Class Classification

```python
from app.ml_engine.evaluation import calculate_classification_metrics

y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 2, 0, 2, 2, 1, 1, 2]
y_proba = [
    [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
    [0.7, 0.2, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7],
    [0.4, 0.5, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]
]
class_names = ['cat', 'dog', 'bird']

metrics = calculate_classification_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_proba=y_proba,
    average='weighted',
    class_names=class_names,
    include_per_class=True
)

# Overall metrics
print(f"Overall Accuracy: {metrics.accuracy:.4f}")
print(f"Weighted F1: {metrics.f1_score:.4f}")

# Per-class metrics
for class_name, class_metrics in metrics.per_class_metrics.items():
    print(f"\n{class_name}:")
    print(f"  Precision: {class_metrics['precision']:.4f}")
    print(f"  Recall: {class_metrics['recall']:.4f}")
    print(f"  F1: {class_metrics['f1_score']:.4f}")
    print(f"  Support: {class_metrics['support']}")
```

#### Example 3: Confusion Matrix

```python
from app.ml_engine.evaluation import ClassificationMetricsCalculator

calculator = ClassificationMetricsCalculator()

# Get confusion matrix
cm_result = calculator.calculate_confusion_matrix(
    y_true=[0, 1, 2, 0, 1, 2],
    y_pred=[0, 1, 2, 1, 1, 2],
    normalize='true',  # Normalize over true labels
    class_names=['cat', 'dog', 'bird']
)

print("Confusion Matrix (normalized):")
print(cm_result['matrix'])
print(f"Classes: {cm_result['class_names']}")
```

#### Example 4: Classification Report

```python
from app.ml_engine.evaluation import ClassificationMetricsCalculator

calculator = ClassificationMetricsCalculator()

# Get sklearn-style classification report
report = calculator.get_classification_report(
    y_true=[0, 1, 2, 0, 1, 2],
    y_pred=[0, 1, 2, 1, 1, 2],
    class_names=['cat', 'dog', 'bird'],
    output_dict=True
)

print("Classification Report:")
for class_name, metrics in report.items():
    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"{class_name}: {metrics}")
```

#### Example 5: Handling Imbalanced Data

```python
from app.ml_engine.evaluation import calculate_classification_metrics

# Imbalanced dataset (90% class 0, 10% class 1)
y_true = [0] * 90 + [1] * 10
y_pred = [0] * 85 + [1] * 5 + [0] * 10

metrics = calculate_classification_metrics(
    y_true=y_true,
    y_pred=y_pred,
    average='binary',
    include_advanced=True
)

# Regular accuracy might be misleading
print(f"Accuracy: {metrics.accuracy:.4f}")  # High due to imbalance

# Balanced accuracy is more informative
print(f"Balanced Accuracy: {metrics.balanced_accuracy:.4f}")  # Lower, more realistic

# Per-class metrics show the issue
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall: {metrics.recall:.4f}")
```

### API Reference

#### `ClassificationMetricsCalculator`

Main class for calculating classification metrics.

**Constructor Parameters:**

- `average` (str): Averaging strategy - 'binary', 'micro', 'macro', 'weighted'
- `pos_label` (int/str): Positive class label for binary classification (default: 1)
- `labels` (list): List of class labels (auto-detected if None)
- `zero_division` (str/int): Value to return when division by zero (default: 'warn')

**Methods:**

##### `calculate_metrics()`

Calculate comprehensive classification metrics.

```python
def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    y_proba: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    class_names: Optional[List[str]] = None,
    include_per_class: bool = True,
    include_advanced: bool = True
) -> ClassificationMetrics
```

**Parameters:**

- `y_true`: True labels
- `y_pred`: Predicted labels
- `y_proba`: Predicted probabilities (optional, for AUC metrics)
- `class_names`: Names for each class (for reporting)
- `include_per_class`: Calculate per-class metrics
- `include_advanced`: Calculate advanced metrics (MCC, Kappa, etc.)

**Returns:** `ClassificationMetrics` object

##### `calculate_confusion_matrix()`

Calculate confusion matrix with optional normalization.

```python
def calculate_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    normalize: Optional[Literal['true', 'pred', 'all']] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[np.ndarray, List[str]]]
```

##### `get_classification_report()`

Generate sklearn classification report.

```python
def get_classification_report(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    class_names: Optional[List[str]] = None,
    output_dict: bool = True
) -> Union[str, Dict]
```

#### `ClassificationMetrics`

Dataclass containing all calculated metrics.

**Attributes:**

- `accuracy` (float): Overall accuracy
- `precision` (float): Precision score
- `recall` (float): Recall score
- `f1_score` (float): F1 score
- `auc_roc` (float, optional): AUC-ROC score
- `auc_pr` (float, optional): AUC-PR score
- `balanced_accuracy` (float, optional): Balanced accuracy
- `matthews_corrcoef` (float, optional): Matthews correlation coefficient
- `cohen_kappa` (float, optional): Cohen's kappa score
- `log_loss` (float, optional): Logarithmic loss
- `confusion_matrix` (np.ndarray, optional): Confusion matrix
- `per_class_metrics` (dict, optional): Per-class metrics breakdown
- `support` (dict, optional): Number of samples per class
- `n_samples` (int, optional): Total number of samples
- `n_classes` (int, optional): Number of classes

**Methods:**

- `to_dict()`: Convert to dictionary (JSON-serializable)
- `__repr__()`: String representation

#### `calculate_classification_metrics()`

Convenience function for quick metric calculation.

```python
def calculate_classification_metrics(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    y_proba: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    average: Literal['binary', 'micro', 'macro', 'weighted'] = 'weighted',
    class_names: Optional[List[str]] = None,
    include_per_class: bool = True,
    include_advanced: bool = True
) -> ClassificationMetrics
```

### Averaging Strategies

#### Binary

For binary classification only. Calculates metrics for the positive class.

```python
metrics = calculate_classification_metrics(
    y_true=[0, 1, 1, 0, 1],
    y_pred=[0, 1, 0, 0, 1],
    average='binary'
)
```

#### Micro

Calculate metrics globally by counting total TP, FP, FN.

```python
metrics = calculate_classification_metrics(
    y_true=[0, 1, 2, 0, 1, 2],
    y_pred=[0, 1, 2, 1, 1, 2],
    average='micro'
)
```

#### Macro

Calculate metrics for each class, then take unweighted mean.

```python
metrics = calculate_classification_metrics(
    y_true=[0, 1, 2, 0, 1, 2],
    y_pred=[0, 1, 2, 1, 1, 2],
    average='macro'
)
```

#### Weighted

Calculate metrics for each class, then take weighted mean by support.

```python
metrics = calculate_classification_metrics(
    y_true=[0, 1, 2, 0, 1, 2],
    y_pred=[0, 1, 2, 1, 1, 2],
    average='weighted'  # Recommended for imbalanced data
)
```

### Input Formats

The module accepts multiple input formats:

```python
# NumPy arrays
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

# Pandas Series
y_true = pd.Series([0, 1, 1, 0, 1])
y_pred = pd.Series([0, 1, 0, 0, 1])

# Python lists
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Probabilities as 2D array (binary)
y_proba = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]]

# Probabilities as 1D array (binary, class 1 only)
y_proba = [0.1, 0.8, 0.4, 0.2, 0.9]

# Probabilities as DataFrame (multi-class)
y_proba = pd.DataFrame({
    0: [0.8, 0.1, 0.1],
    1: [0.1, 0.8, 0.1],
    2: [0.1, 0.1, 0.8]
})
```

### Error Handling

The module provides clear error messages for common issues:

```python
# Mismatched lengths
try:
    metrics = calculate_classification_metrics(
        y_true=[0, 1, 1],
        y_pred=[0, 1]  # Different length
    )
except ValueError as e:
    print(f"Error: {e}")  # "y_true and y_pred must have same length"

# Zero division handling
metrics = calculate_classification_metrics(
    y_true=[0, 0, 1, 1],
    y_pred=[0, 0, 0, 0],  # No predictions for class 1
    average='binary'
)
# Returns 0.0 for precision/recall instead of error
```

### Integration with ML Pipeline

```python
from app.ml_engine.evaluation import calculate_classification_metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Calculate metrics
metrics = calculate_classification_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    average='weighted',
    class_names=['class_0', 'class_1', 'class_2']
)

# Export for API response
metrics_dict = metrics.to_dict()
```

## 📉 Residual Analysis (ML-51)

Features:

- Residual statistics: mean, median, std dev, MAE, MSE, RMSE, MAPE (safe when targets contain zeros)
- Distribution insights: min/max, five quantiles (5th/25th/50th/75th/95th percentiles), skewness, kurtosis
- Standardized residuals with configurable z-score threshold for outlier flags
- Optional Shapiro-Wilk normality test and correlation between |residuals| and predictions to hint at heteroscedasticity

### Quick Start

```python
from app.ml_engine.evaluation import analyze_residuals

actual = [3.0, -0.5, 2.0, 7.0]
predicted = [2.5, 0.0, 2.0, 8.0]

result = analyze_residuals(actual, predicted, zscore_threshold=2.0)

print(result.mae)                # 0.5
print(result.outlier_indices)    # []
plot_payload = result.residual_series  # ready for plotting
```

### Returned Fields

- `residuals`, `predicted`, `actual`
- `standardized_residuals` (optional), `outlier_indices`, `outlier_threshold`
- `quantiles`, `mean_error`, `median_error`, `std_error`, `mae`, `mse`, `rmse`, `mape`
- `skewness`, `kurtosis`, `normality_test` (Shapiro statistic and p-value)
- `correlation_abs_residuals_predicted` to flag potential heteroscedasticity patterns

## 🧪 Testing

Comprehensive test suite with 80+ test cases covering:

- Binary and multi-class classification
- All averaging strategies
- Probability-based metrics
- Per-class metrics
- Edge cases (perfect predictions, all wrong, imbalanced data)
- Input validation and error handling

Run tests:

```bash
pytest backend/tests/ml_engine/evaluation/test_classification_metrics.py -v
pytest backend/tests/ml_engine/evaluation/test_classification_metrics.py --cov=app.ml_engine.evaluation
```

## 📚 References

- [scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Confusion Matrix Explained](https://en.wikipedia.org/wiki/Confusion_matrix)
- [ROC and AUC Explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Precision-Recall Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

## 🚀 Coming Soon

- **Regression Metrics** (ML-47-50): MAE, MSE, RMSE, R², MAPE, residual analysis
- **Clustering Metrics** (ML-53): Silhouette score, inertia, Davies-Bouldin index
- **Visualizations** (ML-48-49): ROC curves, PR curves, confusion matrix heatmaps
- **Feature Importance** (ML-54-55): Permutation importance, SHAP values

## 📝 Notes

- All metrics are calculated using scikit-learn's implementations for consistency
- Probability-based metrics (AUC-ROC, AUC-PR, log loss) require `y_proba` parameter
- For imbalanced datasets, use `balanced_accuracy` and `weighted` averaging
- Confusion matrix can be normalized over true labels, predicted labels, or all samples
- Per-class metrics provide detailed breakdown for multi-class problems
