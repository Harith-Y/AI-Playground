# Model Parameter Validation

Comprehensive parameter validation for ML model configurations.

## Overview

The validation module provides automatic validation of model hyperparameters to ensure configurations are valid before model creation and training. This helps catch configuration errors early and provides clear error messages.

## Features

- **Type Validation**: Ensures parameters are of the correct type (int, float, bool, string, etc.)
- **Range Validation**: Validates numeric parameters are within allowed ranges
- **Value Validation**: Checks parameters against allowed values
- **Length Validation**: Validates length of lists, tuples, and strings
- **Custom Validation**: Supports custom validation functions
- **Comprehensive Schemas**: Pre-defined schemas for all supported models
- **Clear Error Messages**: Provides detailed error messages for validation failures

## Quick Start

### Automatic Validation (Recommended)

```python
from app.ml_engine.models import ModelConfig, create_model

# Validation happens automatically
config = ModelConfig(
    model_type='random_forest_classifier',
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2
    }
)

# Or use the factory
model = create_model(
    'random_forest_classifier',
    n_estimators=100,
    max_depth=10
)
```

### Manual Validation

```python
from app.ml_engine.models import validate_model_config

# Validate hyperparameters
is_valid, errors = validate_model_config(
    'random_forest_classifier',
    {
        'n_estimators': 100,
        'max_depth': 10,
        'invalid_param': 'value'  # Unknown parameter
    },
    strict=False  # Allow unknown parameters
)

if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

### Disable Validation

```python
# Not recommended, but possible
config = ModelConfig(
    model_type='random_forest_classifier',
    hyperparameters={'n_estimators': 100},
    validate=False  # Skip validation
)
```

## Validation Examples

### Valid Configuration

```python
from app.ml_engine.models import create_model

# All parameters are valid
model = create_model(
    'random_forest_classifier',
    n_estimators=100,        # int, >= 1 ✓
    max_depth=10,            # int, >= 1 ✓
    min_samples_split=2,     # int, >= 2 ✓
    criterion='gini',        # allowed value ✓
    random_state=42          # int, >= 0 ✓
)
```

### Invalid Type

```python
from app.ml_engine.models import create_model, ValidationError

try:
    model = create_model(
        'random_forest_classifier',
        n_estimators='100'  # Should be int, not string ✗
    )
except ValueError as e:
    print(e)
    # Output: Invalid hyperparameters for model 'random_forest_classifier':
    #   - Parameter 'n_estimators' must be of type int, got str
```

### Invalid Range

```python
try:
    model = create_model(
        'random_forest_classifier',
        n_estimators=-10  # Must be >= 1 ✗
    )
except ValueError as e:
    print(e)
    # Output: Invalid hyperparameters for model 'random_forest_classifier':
    #   - Parameter 'n_estimators' must be >= 1, got -10
```

### Invalid Value

```python
try:
    model = create_model(
        'random_forest_classifier',
        criterion='invalid'  # Must be 'gini', 'entropy', or 'log_loss' ✗
    )
except ValueError as e:
    print(e)
    # Output: Invalid hyperparameters for model 'random_forest_classifier':
    #   - Parameter 'criterion' must be one of ['gini', 'entropy', 'log_loss'], got invalid
```

### Strict Mode

```python
from app.ml_engine.models import ModelConfig

try:
    config = ModelConfig(
        model_type='random_forest_classifier',
        hyperparameters={
            'n_estimators': 100,
            'unknown_param': 'value'  # Unknown parameter
        },
        strict=True  # Reject unknown parameters
    )
except ValueError as e:
    print(e)
    # Output: Invalid hyperparameters for model 'random_forest_classifier':
    #   - Unknown parameter 'unknown_param' for model 'random_forest_classifier'
```

## Utility Functions

### Get Model Defaults

```python
from app.ml_engine.models import get_model_defaults

# Get default hyperparameters
defaults = get_model_defaults('random_forest_classifier')
print(defaults)
# Output: {
#     'n_estimators': 100,
#     'criterion': 'gini',
#     'max_depth': None,
#     'min_samples_split': 2,
#     ...
# }
```

### Get Parameter Info

```python
from app.ml_engine.models import get_parameter_info

# Get information about a specific parameter
info = get_parameter_info('random_forest_classifier', 'n_estimators')
print(info)
# Output: {
#     'name': 'n_estimators',
#     'type': ['int'],
#     'required': False,
#     'default': 100,
#     'min_value': 1,
#     'max_value': None,
#     'allowed_values': None,
#     'description': 'Number of trees in the forest'
# }
```

### Get Available Models

```python
from app.ml_engine.models import get_available_models_with_schemas

# Get list of models with validation schemas
models = get_available_models_with_schemas()
print(models)
# Output: [
#     'logistic_regression',
#     'random_forest_classifier',
#     'svm_classifier',
#     ...
# ]
```

## Supported Models

### Classification Models

- **logistic_regression**: Logistic Regression
- **random_forest_classifier**: Random Forest Classifier
- **svm_classifier**: Support Vector Machine Classifier
- **gradient_boosting_classifier**: Gradient Boosting Classifier
- **knn_classifier**: K-Nearest Neighbors Classifier

### Regression Models

- **linear_regression**: Linear Regression
- **ridge_regression**: Ridge Regression (L2)
- **lasso_regression**: Lasso Regression (L1)
- **random_forest_regressor**: Random Forest Regressor

### Clustering Models

- **kmeans**: K-Means Clustering
- **dbscan**: DBSCAN Clustering
- **agglomerative_clustering**: Agglomerative Clustering
- **gaussian_mixture**: Gaussian Mixture Model

## Parameter Types

The validation system supports the following parameter types:

- **INT**: Integer values
- **FLOAT**: Float values (also accepts integers)
- **BOOL**: Boolean values
- **STRING**: String values
- **LIST**: List values
- **TUPLE**: Tuple values
- **DICT**: Dictionary values
- **CALLABLE**: Callable/function values
- **NONE**: None value

Parameters can accept multiple types:

```python
ParameterSpec(
    "max_depth",
    [ParameterType.INT, ParameterType.NONE],  # Can be int or None
    default=None
)
```

## Common Parameters

### random_state

```python
# Type: int or None
# Range: >= 0
# Default: None
# Description: Random seed for reproducibility

model = create_model('random_forest_classifier', random_state=42)
```

### n_jobs

```python
# Type: int or None
# Range: >= -1
# Default: None
# Description: Number of parallel jobs (-1 for all cores)

model = create_model('random_forest_classifier', n_jobs=-1)
```

### verbose

```python
# Type: int or bool
# Range: >= 0
# Default: 0
# Description: Verbosity level

model = create_model('random_forest_classifier', verbose=1)
```

## Model-Specific Parameters

### Random Forest Classifier

```python
model = create_model(
    'random_forest_classifier',
    n_estimators=100,           # int, >= 1, default: 100
    criterion='gini',           # 'gini', 'entropy', 'log_loss', default: 'gini'
    max_depth=None,             # int or None, >= 1, default: None
    min_samples_split=2,        # int or float, >= 2, default: 2
    min_samples_leaf=1,         # int or float, >= 1, default: 1
    max_features='sqrt',        # 'sqrt', 'log2', int, float, or None, default: 'sqrt'
    bootstrap=True,             # bool, default: True
    oob_score=False,            # bool, default: False
    class_weight=None,          # 'balanced', 'balanced_subsample', dict, or None
    random_state=None,          # int or None, >= 0, default: None
    n_jobs=None,                # int or None, >= -1, default: None
    verbose=0                   # int or bool, >= 0, default: 0
)
```

### Logistic Regression

```python
model = create_model(
    'logistic_regression',
    penalty='l2',               # 'l1', 'l2', 'elasticnet', 'none', default: 'l2'
    C=1.0,                      # float, > 0, default: 1.0
    solver='lbfgs',             # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    max_iter=100,               # int, >= 1, default: 100
    tol=1e-4,                   # float, >= 0, default: 1e-4
    fit_intercept=True,         # bool, default: True
    class_weight=None,          # 'balanced', dict, or None
    random_state=None           # int or None, >= 0, default: None
)
```

### K-Means

```python
model = create_model(
    'kmeans',
    n_clusters=8,               # int, >= 1, default: 8
    init='k-means++',           # 'k-means++', 'random', default: 'k-means++'
    n_init='auto',              # int or 'auto', >= 1, default: 'auto'
    max_iter=300,               # int, >= 1, default: 300
    tol=1e-4,                   # float, >= 0, default: 1e-4
    algorithm='lloyd',          # 'lloyd', 'elkan', default: 'lloyd'
    random_state=None           # int or None, >= 0, default: None
)
```

### Ridge Regression

```python
model = create_model(
    'ridge_regression',
    alpha=1.0,                  # float, >= 0, default: 1.0
    fit_intercept=True,         # bool, default: True
    max_iter=None,              # int or None, >= 1, default: None
    tol=1e-4,                   # float, >= 0, default: 1e-4
    solver='auto',              # 'auto', 'svd', 'cholesky', 'lsqr', etc.
    random_state=None           # int or None, >= 0, default: None
)
```

## Integration with Training

```python
from app.ml_engine.models import create_model
from app.ml_engine.training import train_model, train_test_split

# 1. Create model with validated config
model = create_model(
    'random_forest_classifier',
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# 2. Split data
split_result = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = split_result.get_train_data()
X_test, y_test = split_result.get_test_data()

# 3. Train model
train_result = train_model(
    model,
    X_train, y_train,
    X_test=X_test, y_test=y_test
)

print(f"Train score: {train_result.train_score:.4f}")
print(f"Test score: {train_result.test_score:.4f}")
```

## Best Practices

### 1. Always Use Validation

```python
# Good: Validation enabled (default)
model = create_model('random_forest_classifier', n_estimators=100)

# Bad: Validation disabled
model = create_model('random_forest_classifier', n_estimators=100, validate=False)
```

### 2. Use Strict Mode for Production

```python
# Production: Reject unknown parameters
config = ModelConfig(
    model_type='random_forest_classifier',
    hyperparameters={'n_estimators': 100},
    strict=True
)
```

### 3. Check Parameter Info

```python
# Before using a parameter, check its constraints
info = get_parameter_info('random_forest_classifier', 'n_estimators')
print(f"Min value: {info['min_value']}")
print(f"Default: {info['default']}")
```

### 4. Use Defaults

```python
# Get defaults and override specific values
defaults = get_model_defaults('random_forest_classifier')
defaults['n_estimators'] = 200
defaults['max_depth'] = 15

model = create_model('random_forest_classifier', **defaults)
```

### 5. Handle Validation Errors

```python
try:
    model = create_model('random_forest_classifier', n_estimators=-10)
except ValueError as e:
    print(f"Configuration error: {e}")
    # Use defaults or prompt user for correction
    model = create_model('random_forest_classifier')  # Use defaults
```

## Error Messages

The validation system provides clear, actionable error messages:

```python
# Type error
"Parameter 'n_estimators' must be of type int, got str"

# Range error
"Parameter 'n_estimators' must be >= 1, got -10"

# Value error
"Parameter 'criterion' must be one of ['gini', 'entropy', 'log_loss'], got invalid"

# Missing required parameter
"Required parameter 'n_clusters' is missing"

# Unknown parameter (strict mode)
"Unknown parameter 'invalid_param' for model 'random_forest_classifier'"

# Length error
"Parameter 'feature_names' must have length >= 1, got 0"
```

## Custom Validation

For advanced use cases, you can define custom validation functions:

```python
from app.ml_engine.models.validation import ParameterSpec, ParameterType

def validate_positive_odd(value):
    """Validate that value is positive and odd."""
    return value > 0 and value % 2 == 1

custom_spec = ParameterSpec(
    "custom_param",
    ParameterType.INT,
    custom_validator=validate_positive_odd,
    description="Must be positive and odd"
)
```

## Performance

Validation is fast and has minimal overhead:

- Type checking: O(1)
- Range checking: O(1)
- Value checking: O(n) where n is number of allowed values
- Custom validation: Depends on custom function

For most use cases, validation overhead is negligible (<1ms per model creation).

## Conclusion

Parameter validation provides:

- **Early Error Detection**: Catch configuration errors before training
- **Clear Error Messages**: Understand what's wrong and how to fix it
- **Type Safety**: Ensure parameters are of correct types
- **Range Safety**: Ensure numeric parameters are within valid ranges
- **Documentation**: Parameter info serves as inline documentation
- **Production Ready**: Strict mode for production deployments

Always use validation to ensure robust, error-free ML pipelines!
