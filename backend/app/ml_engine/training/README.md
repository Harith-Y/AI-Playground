# ML Engine Training Module

## Overview

The training module provides generic, high-level functions for training machine learning models with a consistent interface across all model types (regression, classification, clustering).

## Key Features

- ✅ **Generic Training Function** - Works with any model wrapper
- ✅ **Automatic Evaluation** - Computes scores on train/val/test sets
- ✅ **Input Validation** - Comprehensive validation of training inputs
- ✅ **Training Metadata** - Tracks training time, sample counts, features
- ✅ **Prediction Storage** - Optionally stores predictions for analysis
- ✅ **Error Handling** - Robust error handling with informative messages
- ✅ **Logging** - Detailed logging for debugging and monitoring

## Core Functions

### 1. `train_model()`

The main training function that handles the complete training workflow.

**Signature:**
```python
def train_model(
    model: BaseModelWrapper,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray, None] = None,
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    compute_train_score: bool = True,
    store_predictions: bool = False,
    **fit_params
) -> TrainingResult
```

**Parameters:**
- `model`: Model wrapper to train (must not be fitted yet)
- `X_train`: Training features
- `y_train`: Training target (None for unsupervised)
- `X_val`: Validation features (optional)
- `y_val`: Validation target (optional)
- `X_test`: Test features (optional)
- `y_test`: Test target (optional)
- `compute_train_score`: Whether to compute score on training set
- `store_predictions`: Whether to store predictions in result
- `**fit_params`: Additional parameters passed to model.fit()

**Returns:**
- `TrainingResult`: Object containing fitted model and evaluation metrics

**Example:**
```python
from app.ml_engine.models.classification import RandomForestClassifierWrapper
from app.ml_engine.models.base import ModelConfig
from app.ml_engine.training import train_model

# Create model
config = ModelConfig('random_forest', {'n_estimators': 100, 'max_depth': 10})
model = RandomForestClassifierWrapper(config)

# Train model
result = train_model(
    model,
    X_train, y_train,
    X_test=X_test, y_test=y_test,
    compute_train_score=True
)

# Access results
print(f"Train score: {result.train_score:.4f}")
print(f"Test score: {result.test_score:.4f}")
print(f"Training time: {result.training_time:.2f}s")
```

### 2. `evaluate_model()`

Evaluate a fitted model on given data.

**Signature:**
```python
def evaluate_model(
    model: BaseModelWrapper,
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    return_predictions: bool = False
) -> Union[float, Tuple[float, np.ndarray]]
```

**Example:**
```python
from app.ml_engine.training import evaluate_model

# Evaluate on test set
score = evaluate_model(model, X_test, y_test)
print(f"Test score: {score:.4f}")

# Get predictions along with score
score, predictions = evaluate_model(
    model, X_test, y_test, return_predictions=True
)
```

### 3. `predict_with_model()`

Make predictions with a fitted model.

**Signature:**
```python
def predict_with_model(
    model: BaseModelWrapper,
    X: Union[pd.DataFrame, np.ndarray],
    return_proba: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
```

**Example:**
```python
from app.ml_engine.training import predict_with_model

# Make predictions
predictions = predict_with_model(model, X_new)

# Get probabilities (classification only)
predictions, probabilities = predict_with_model(
    model, X_new, return_proba=True
)
```

### 4. `get_model_info()`

Get comprehensive information about a model.

**Signature:**
```python
def get_model_info(model: BaseModelWrapper) -> Dict[str, Any]
```

**Example:**
```python
from app.ml_engine.training import get_model_info

info = get_model_info(model)
print(f"Model type: {info['model_type']}")
print(f"Is fitted: {info['is_fitted']}")
print(f"Features: {info['n_features']}")
```

## TrainingResult Class

Container for training results with comprehensive information.

**Attributes:**
- `model`: Fitted model wrapper
- `train_score`: Score on training set
- `val_score`: Score on validation set (if provided)
- `test_score`: Score on test set (if provided)
- `training_time`: Training duration in seconds
- `predictions`: Dictionary of predictions (if stored)
- `metadata`: Additional metadata about training

**Methods:**
- `to_dict()`: Convert to dictionary
- `__repr__()`: String representation

**Example:**
```python
result = train_model(model, X_train, y_train, X_test=X_test, y_test=y_test)

# Access attributes
print(result.train_score)
print(result.test_score)
print(result.training_time)

# Convert to dict
result_dict = result.to_dict()
print(result_dict)
```

## Complete Examples

### Example 1: Classification with Validation Set

```python
from sklearn.model_selection import train_test_split
from app.ml_engine.models.classification import RandomForestClassifierWrapper
from app.ml_engine.models.base import ModelConfig
from app.ml_engine.training import train_model

# Load data
X, y = load_data()

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Create model
config = ModelConfig(
    'random_forest_classifier',
    {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
)
model = RandomForestClassifierWrapper(config)

# Train with validation
result = train_model(
    model,
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

# Print results
print(f"Training score: {result.train_score:.4f}")
print(f"Validation score: {result.val_score:.4f}")
print(f"Test score: {result.test_score:.4f}")
print(f"Training time: {result.training_time:.2f}s")
```

### Example 2: Regression with Predictions

```python
from app.ml_engine.models.regression import RandomForestRegressorWrapper
from app.ml_engine.models.base import ModelConfig
from app.ml_engine.training import train_model

# Create model
config = ModelConfig(
    'random_forest_regressor',
    {'n_estimators': 100, 'max_depth': 15}
)
model = RandomForestRegressorWrapper(config)

# Train and store predictions
result = train_model(
    model,
    X_train, y_train,
    X_test=X_test, y_test=y_test,
    store_predictions=True
)

# Access predictions
train_predictions = result.predictions['train']
test_predictions = result.predictions['test']

# Calculate custom metrics
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, test_predictions)
print(f"Test MAE: {mae:.4f}")
```

### Example 3: Clustering (Unsupervised)

```python
from app.ml_engine.models.clustering import KMeansWrapper
from app.ml_engine.models.base import ModelConfig
from app.ml_engine.training import train_model

# Create model
config = ModelConfig('kmeans', {'n_clusters': 3, 'random_state': 42})
model = KMeansWrapper(config)

# Train (no y_train for unsupervised)
result = train_model(
    model,
    X_train,
    store_predictions=True
)

# Access cluster labels
labels = result.predictions['train_labels']
print(f"Cluster distribution: {np.bincount(labels)}")

# Get cluster centers
centers = model.get_cluster_centers()
print(f"Cluster centers shape: {centers.shape}")
```

### Example 4: Model Evaluation

```python
from app.ml_engine.training import train_model, evaluate_model

# Train model
result = train_model(model, X_train, y_train)

# Evaluate on new data
new_score = evaluate_model(result.model, X_new, y_new)
print(f"Score on new data: {new_score:.4f}")

# Get predictions
score, predictions = evaluate_model(
    result.model, X_new, y_new, return_predictions=True
)
```

### Example 5: Making Predictions

```python
from app.ml_engine.training import train_model, predict_with_model

# Train model
result = train_model(model, X_train, y_train)

# Make predictions on new data
predictions = predict_with_model(result.model, X_new)

# For classification, get probabilities
predictions, probabilities = predict_with_model(
    result.model, X_new, return_proba=True
)

print(f"Predictions: {predictions[:5]}")
print(f"Probabilities: {probabilities[:5]}")
```

## Input Validation

The training function performs comprehensive validation:

### Validation Checks:
1. ✅ Model not already fitted
2. ✅ Training data not empty
3. ✅ Target provided for supervised learning
4. ✅ X and y have same length
5. ✅ Validation data consistency (X_val and y_val together)
6. ✅ Test data consistency (X_test and y_test together)
7. ✅ Feature count consistency across train/val/test

### Error Messages:
```python
# Model already fitted
ValueError: Model is already fitted. Create a new model instance for training.

# Empty training data
ValueError: Training features X_train cannot be empty

# Missing target for supervised learning
ValueError: Training target y_train is required for supervised learning

# Length mismatch
ValueError: X_train and y_train must have same length. Got X_train: 100, y_train: 90

# Feature count mismatch
ValueError: X_train and X_test must have same number of features. Got train: 10, test: 8
```

## Error Handling

The training function handles errors gracefully:

```python
try:
    result = train_model(model, X_train, y_train)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Training failed: {e}")
```

## Logging

The module provides detailed logging:

```python
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

# Training logs:
# INFO: Starting training for random_forest_classifier
# INFO: Fitting model on 1000 training samples
# INFO: Model fitted successfully in 2.34 seconds
# INFO: Computing training score
# INFO: Training score: 0.9567
# INFO: Computing test score
# INFO: Test score: 0.9234
# INFO: Training completed successfully: TrainingResult(...)
```

## Integration with Existing Code

The training module integrates seamlessly with existing model wrappers:

```python
# Works with any model wrapper
from app.ml_engine.models.classification import (
    LogisticRegressionWrapper,
    RandomForestClassifierWrapper,
    SVMClassifierWrapper
)
from app.ml_engine.models.regression import (
    LinearRegressionWrapper,
    RidgeRegressionWrapper
)
from app.ml_engine.models.clustering import (
    KMeansWrapper,
    DBSCANWrapper
)

# All use the same training interface
result = train_model(any_model, X_train, y_train)
```

## Best Practices

### 1. Always Use Validation Set

```python
# Good: Use validation set for model selection
result = train_model(
    model, X_train, y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

# Use val_score for model selection
# Use test_score for final evaluation
```

### 2. Store Predictions for Analysis

```python
# Good: Store predictions for detailed analysis
result = train_model(
    model, X_train, y_train,
    X_test=X_test, y_test=y_test,
    store_predictions=True
)

# Analyze predictions
from sklearn.metrics import classification_report
print(classification_report(y_test, result.predictions['test']))
```

### 3. Handle Errors Gracefully

```python
# Good: Handle errors appropriately
try:
    result = train_model(model, X_train, y_train)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    # Fix input and retry
except RuntimeError as e:
    logger.error(f"Training failed: {e}")
    # Try different hyperparameters
```

### 4. Log Training Results

```python
# Good: Log comprehensive results
result = train_model(model, X_train, y_train, X_test=X_test, y_test=y_test)

logger.info(f"Model: {result.model.config.model_type}")
logger.info(f"Train score: {result.train_score:.4f}")
logger.info(f"Test score: {result.test_score:.4f}")
logger.info(f"Training time: {result.training_time:.2f}s")
logger.info(f"Samples: {result.metadata['n_train_samples']}")
logger.info(f"Features: {result.metadata['n_features']}")
```

## Performance Considerations

### Memory Usage

```python
# For large datasets, avoid storing predictions
result = train_model(
    model, X_train, y_train,
    store_predictions=False  # Saves memory
)

# Compute predictions on-demand
predictions = predict_with_model(result.model, X_test)
```

### Training Time

```python
# Skip training score computation for large datasets
result = train_model(
    model, X_train, y_train,
    compute_train_score=False  # Faster training
)
```

## Testing

The training module includes comprehensive tests:

```bash
# Run tests
pytest backend/tests/test_trainer.py -v

# Run with coverage
pytest backend/tests/test_trainer.py --cov=app.ml_engine.training
```

## Future Enhancements

Potential improvements:
- [ ] Early stopping support
- [ ] Learning curves generation
- [ ] Cross-validation integration
- [ ] Parallel training for multiple models
- [ ] Training callbacks/hooks
- [ ] Automatic hyperparameter validation
- [ ] Training history tracking

## Related Modules

- `app.ml_engine.models.base` - Base model wrappers
- `app.ml_engine.models.classification` - Classification models
- `app.ml_engine.models.regression` - Regression models
- `app.ml_engine.models.clustering` - Clustering models
- `app.ml_engine.evaluation.metrics` - Evaluation metrics (coming soon)

## References

- Scikit-learn documentation: https://scikit-learn.org/stable/
- Model wrappers: `backend/app/ml_engine/models/`
- Training tasks: `backend/app/tasks/training_tasks.py`
