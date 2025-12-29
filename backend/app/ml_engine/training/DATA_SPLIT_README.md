# Data Splitting Utilities

## Overview

The data splitting module provides flexible functions for splitting datasets into train, validation, and test sets with various options for stratification, shuffling, and custom split ratios.

## Key Features

- ✅ **Train/Test Split** - Simple 2-way split
- ✅ **Train/Val/Test Split** - 3-way split with validation set
- ✅ **Custom Ratios** - Flexible split ratios
- ✅ **Stratified Splitting** - Maintains class distribution
- ✅ **Temporal Splitting** - For time-series data (no shuffling)
- ✅ **Input Validation** - Comprehensive validation
- ✅ **Metadata Tracking** - Tracks split configuration
- ✅ **Logging** - Detailed logging

## Core Functions

### 1. `train_test_split()`

Simple 2-way split into train and test sets.

**Signature:**
```python
def train_test_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> DataSplitResult
```

**Example:**
```python
from app.ml_engine.training import train_test_split

# Simple 80/20 split
result = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, y_train = result.get_train_data()
X_test, y_test = result.get_test_data()

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

### 2. `train_val_test_split()`

3-way split into train, validation, and test sets.

**Signature:**
```python
def train_val_test_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> DataSplitResult
```

**Example:**
```python
from app.ml_engine.training import train_val_test_split

# 70/15/15 split
result = train_val_test_split(
    X, y,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    stratify=True  # Maintain class distribution
)

X_train, y_train = result.get_train_data()
X_val, y_val = result.get_val_data()
X_test, y_test = result.get_test_data()

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

### 3. `split_by_ratio()`

Flexible splitting with custom ratios.

**Signature:**
```python
def split_by_ratio(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    ratios: Tuple[float, ...] = (0.7, 0.15, 0.15),
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> DataSplitResult
```

**Example:**
```python
from app.ml_engine.training import split_by_ratio

# 70/15/15 split
result = split_by_ratio(X, y, ratios=(0.7, 0.15, 0.15))

# 80/20 split (no validation)
result = split_by_ratio(X, y, ratios=(0.8, 0.2))

# 60/20/20 split
result = split_by_ratio(X, y, ratios=(0.6, 0.2, 0.2))
```

### 4. `temporal_split()`

Split time-series data without shuffling.

**Signature:**
```python
def temporal_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> DataSplitResult
```

**Example:**
```python
from app.ml_engine.training import temporal_split

# Split time-series data (preserves order)
result = temporal_split(
    X_timeseries, y_timeseries,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)

# Data is NOT shuffled - temporal order preserved
# Train: oldest data
# Val: middle data
# Test: newest data
```

### 5. `get_split_info()`

Get comprehensive information about a split.

**Signature:**
```python
def get_split_info(result: DataSplitResult) -> dict
```

**Example:**
```python
from app.ml_engine.training import train_val_test_split, get_split_info

result = train_val_test_split(X, y)
info = get_split_info(result)

print(info['sizes'])        # {'train': 700, 'val': 150, 'test': 150}
print(info['percentages'])  # {'train': 70.0, 'val': 15.0, 'test': 15.0}
print(info['stratified'])   # False
print(info['shuffled'])     # True
```

## DataSplitResult Class

Container for split datasets with metadata.

**Attributes:**
- `X_train`, `y_train`: Training data
- `X_val`, `y_val`: Validation data (optional)
- `X_test`, `y_test`: Test data (optional)
- `split_ratios`: Tuple of actual split ratios
- `stratified`: Whether stratification was used
- `shuffled`: Whether data was shuffled
- `random_state`: Random seed used

**Methods:**
- `get_train_data()`: Returns (X_train, y_train)
- `get_val_data()`: Returns (X_val, y_val) or None
- `get_test_data()`: Returns (X_test, y_test) or None
- `get_split_sizes()`: Returns dictionary of split sizes
- `__repr__()`: String representation

## Complete Examples

### Example 1: Basic Train/Test Split

```python
from app.ml_engine.training import train_test_split

# 80/20 split
result = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, y_train = result.get_train_data()
X_test, y_test = result.get_test_data()

print(f"Train: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")
```

### Example 2: Stratified Split for Imbalanced Data

```python
from app.ml_engine.training import train_test_split

# Stratified split maintains class distribution
result = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=True  # Maintains class distribution
)

# Check class distribution
print("Train distribution:", np.bincount(result.y_train))
print("Test distribution:", np.bincount(result.y_test))
```

### Example 3: Train/Val/Test Split

```python
from app.ml_engine.training import train_val_test_split

# 70/15/15 split
result = train_val_test_split(
    X, y,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42
)

X_train, y_train = result.get_train_data()
X_val, y_val = result.get_val_data()
X_test, y_test = result.get_test_data()

print(f"Train: {len(X_train)}")
print(f"Val: {len(X_val)}")
print(f"Test: {len(X_test)}")
```

### Example 4: Custom Ratios

```python
from app.ml_engine.training import split_by_ratio

# 60/20/20 split
result = split_by_ratio(X, y, ratios=(0.6, 0.2, 0.2))

# 80/20 split (no validation)
result = split_by_ratio(X, y, ratios=(0.8, 0.2))

# 50/25/25 split
result = split_by_ratio(X, y, ratios=(0.5, 0.25, 0.25))
```

### Example 5: Time-Series Split

```python
from app.ml_engine.training import temporal_split

# Split time-series data (no shuffling)
result = temporal_split(
    X_timeseries, y_timeseries,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)

# Data order preserved:
# Train: 2020-01-01 to 2021-12-31
# Val:   2022-01-01 to 2022-06-30
# Test:  2022-07-01 to 2022-12-31
```

### Example 6: Complete ML Pipeline

```python
from app.ml_engine.training import train_val_test_split, train_model
from app.ml_engine.models.classification import RandomForestClassifierWrapper
from app.ml_engine.models.base import ModelConfig

# 1. Split data
split_result = train_val_test_split(
    X, y,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    stratify=True
)

X_train, y_train = split_result.get_train_data()
X_val, y_val = split_result.get_val_data()
X_test, y_test = split_result.get_test_data()

# 2. Create model
config = ModelConfig('random_forest', {'n_estimators': 100})
model = RandomForestClassifierWrapper(config)

# 3. Train model
train_result = train_model(
    model,
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

# 4. Print results
print(f"Train score: {train_result.train_score:.4f}")
print(f"Val score: {train_result.val_score:.4f}")
print(f"Test score: {train_result.test_score:.4f}")
```

### Example 7: Unsupervised Learning

```python
from app.ml_engine.training import train_test_split

# Split without target (clustering)
result = train_test_split(X, test_size=0.2, random_state=42)

X_train, _ = result.get_train_data()
X_test, _ = result.get_test_data()

# Train clustering model
from app.ml_engine.models.clustering import KMeansWrapper
from app.ml_engine.models.base import ModelConfig
from app.ml_engine.training import train_model

config = ModelConfig('kmeans', {'n_clusters': 3})
model = KMeansWrapper(config)

train_result = train_model(model, X_train)
```

## Stratified Splitting

### When to Use Stratification

**Use stratification when:**
- ✅ Classification tasks with imbalanced classes
- ✅ Small datasets where class distribution matters
- ✅ You want train/val/test to have same class proportions

**Don't use stratification for:**
- ❌ Regression tasks (continuous targets)
- ❌ Clustering (unsupervised)
- ❌ Time-series data (use temporal_split instead)

### Example: Imbalanced Classification

```python
from app.ml_engine.training import train_test_split
import numpy as np

# Imbalanced dataset: 90% class 0, 10% class 1
print("Original distribution:", np.bincount(y) / len(y))
# Output: [0.9, 0.1]

# Without stratification
result = train_test_split(X, y, test_size=0.2, stratify=False)
print("Train distribution:", np.bincount(result.y_train) / len(result.y_train))
# Output: [0.88, 0.12] - Different!

# With stratification
result = train_test_split(X, y, test_size=0.2, stratify=True)
print("Train distribution:", np.bincount(result.y_train) / len(result.y_train))
# Output: [0.9, 0.1] - Same as original!
```

## Temporal Splitting

### When to Use Temporal Split

**Use temporal split for:**
- ✅ Time-series forecasting
- ✅ Stock price prediction
- ✅ Sales forecasting
- ✅ Any data with temporal dependencies

**Why no shuffling?**
- Prevents data leakage (future data in training)
- Preserves temporal order
- Realistic evaluation (predict future from past)

### Example: Time-Series Forecasting

```python
from app.ml_engine.training import temporal_split

# Time-series data (ordered by date)
# Rows: [2020-01, 2020-02, ..., 2022-12]

result = temporal_split(
    X_timeseries, y_timeseries,
    train_size=0.7,   # 2020-01 to 2021-08
    val_size=0.15,    # 2021-09 to 2022-03
    test_size=0.15    # 2022-04 to 2022-12
)

# Train on past, validate on recent past, test on recent data
```

## Input Validation

### Validation Checks

1. ✅ X not empty
2. ✅ test_size between 0.0 and 1.0
3. ✅ X and y same length
4. ✅ Stratification requires y
5. ✅ Minimum 2 samples
6. ✅ Split sizes sum to 1.0 (for 3-way split)

### Error Messages

```python
# Empty data
ValueError: X cannot be empty

# Invalid test_size
ValueError: test_size must be between 0.0 and 1.0. Got: 1.5

# Length mismatch
ValueError: X and y must have same length. Got X: 100, y: 90

# Stratification without target
ValueError: stratify=True requires y to be provided

# Sizes don't sum to 1.0
ValueError: train_size + val_size + test_size must equal 1.0. Got: 0.9
```

## DataSplitResult Class

### Attributes

```python
result = train_val_test_split(X, y)

# Access data
result.X_train  # Training features
result.y_train  # Training target
result.X_val    # Validation features
result.y_val    # Validation target
result.X_test   # Test features
result.y_test   # Test target

# Metadata
result.split_ratios  # (0.7, 0.15, 0.15)
result.stratified    # False
result.shuffled      # True
result.random_state  # 42
```

### Methods

```python
# Get data tuples
X_train, y_train = result.get_train_data()
X_val, y_val = result.get_val_data()
X_test, y_test = result.get_test_data()

# Get sizes
sizes = result.get_split_sizes()
# {'train': 700, 'val': 150, 'test': 150, 'total': 1000}

# String representation
print(result)
# DataSplitResult(train=700, val=150, test=150, total=1000)
```

## Common Split Ratios

### Small Datasets (< 1000 samples)

```python
# 70/30 split (no validation)
result = train_test_split(X, y, test_size=0.3)

# 60/20/20 split (with validation)
result = train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2)
```

### Medium Datasets (1000-10000 samples)

```python
# 80/20 split
result = train_test_split(X, y, test_size=0.2)

# 70/15/15 split
result = train_val_test_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15)
```

### Large Datasets (> 10000 samples)

```python
# 90/10 split
result = train_test_split(X, y, test_size=0.1)

# 80/10/10 split
result = train_val_test_split(X, y, train_size=0.8, val_size=0.1, test_size=0.1)
```

## Best Practices

### 1. Always Set Random State

```python
# Good: Reproducible splits
result = train_test_split(X, y, test_size=0.2, random_state=42)

# Bad: Non-reproducible
result = train_test_split(X, y, test_size=0.2)  # Different each time
```

### 2. Use Stratification for Imbalanced Data

```python
# Good: Maintain class distribution
result = train_test_split(
    X, y,
    test_size=0.2,
    stratify=True,
    random_state=42
)
```

### 3. Use Validation Set for Model Selection

```python
# Good: Use validation for hyperparameter tuning
result = train_val_test_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15)

# Train multiple models, select best on validation
# Final evaluation on test set
```

### 4. Use Temporal Split for Time-Series

```python
# Good: Preserve temporal order
result = temporal_split(X_timeseries, y_timeseries)

# Bad: Shuffling time-series data
result = train_test_split(X_timeseries, y_timeseries, shuffle=True)  # Data leakage!
```

### 5. Check Split Sizes

```python
# Good: Verify split sizes
result = train_val_test_split(X, y)
info = get_split_info(result)

print(f"Train: {info['percentages']['train']:.1f}%")
print(f"Val: {info['percentages']['val']:.1f}%")
print(f"Test: {info['percentages']['test']:.1f}%")
```

## Integration with Training

### Complete Pipeline

```python
from app.ml_engine.training import (
    train_val_test_split,
    train_model,
    get_split_info
)
from app.ml_engine.models.classification import RandomForestClassifierWrapper
from app.ml_engine.models.base import ModelConfig

# 1. Split data
split_result = train_val_test_split(
    X, y,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    stratify=True
)

# 2. Log split info
info = get_split_info(split_result)
print(f"Split sizes: {info['sizes']}")
print(f"Stratified: {info['stratified']}")

# 3. Get data
X_train, y_train = split_result.get_train_data()
X_val, y_val = split_result.get_val_data()
X_test, y_test = split_result.get_test_data()

# 4. Train model
config = ModelConfig('random_forest', {'n_estimators': 100})
model = RandomForestClassifierWrapper(config)

train_result = train_model(
    model,
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

# 5. Print results
print(f"Train score: {train_result.train_score:.4f}")
print(f"Val score: {train_result.val_score:.4f}")
print(f"Test score: {train_result.test_score:.4f}")
```

## Error Handling

### Handle Split Errors

```python
try:
    result = train_val_test_split(X, y, train_size=0.7, val_size=0.2, test_size=0.2)
except ValueError as e:
    print(f"Invalid split configuration: {e}")
    # Adjust sizes and retry
```

### Handle Empty Data

```python
try:
    result = train_test_split(X_empty, y_empty, test_size=0.2)
except ValueError as e:
    print(f"Cannot split empty data: {e}")
```

## Performance Considerations

### Memory Usage

```python
# For large datasets, split returns views (not copies) when possible
result = train_test_split(X_large, y_large, test_size=0.2)

# Memory efficient - no data duplication
```

### Speed

```python
# Shuffling adds overhead
result = train_test_split(X, y, test_size=0.2, shuffle=False)  # Faster

# Stratification adds overhead
result = train_test_split(X, y, test_size=0.2, stratify=False)  # Faster
```

## Common Patterns

### Pattern 1: Quick Train/Test

```python
from app.ml_engine.training import train_test_split

result = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = result.get_train_data()
X_test, y_test = result.get_test_data()
```

### Pattern 2: Full Pipeline with Validation

```python
from app.ml_engine.training import train_val_test_split

result = train_val_test_split(X, y, random_state=42, stratify=True)
X_train, y_train = result.get_train_data()
X_val, y_val = result.get_val_data()
X_test, y_test = result.get_test_data()
```

### Pattern 3: Time-Series

```python
from app.ml_engine.training import temporal_split

result = temporal_split(X_ts, y_ts, train_size=0.7, val_size=0.15, test_size=0.15)
```

## Related Functions

- `train_model()` - Train models using split data
- `evaluate_model()` - Evaluate on any split
- `cross_val_score()` - Cross-validation (coming soon)

## References

- Scikit-learn train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Stratified sampling: https://en.wikipedia.org/wiki/Stratified_sampling
- Time-series cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
