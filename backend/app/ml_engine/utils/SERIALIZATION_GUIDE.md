# Model and Pipeline Serialization Guide

Complete guide for saving and loading ML models, preprocessing pipelines, and complete workflows in AI-Playground.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Model Serialization](#model-serialization)
- [Pipeline Serialization](#pipeline-serialization)
- [Workflow Serialization](#workflow-serialization)
- [Compression](#compression)
- [Metadata](#metadata)
- [Version Compatibility](#version-compatibility)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The serialization module provides comprehensive utilities for persisting ML artifacts:

- **Models**: Save fitted model wrappers with all metadata
- **Pipelines**: Save preprocessing pipelines with fitted transformers
- **Workflows**: Save complete ML workflows (preprocessing + model)
- **Compression**: Optional gzip compression for large files
- **Metadata**: Preserve training information, feature names, configurations
- **Version Tracking**: Compatibility checks across versions

### Supported Formats

- **Primary**: Joblib (optimized for sklearn models)
- **Compression**: Gzip (.gz extension)
- **Metadata**: JSON-serializable dictionaries

## Quick Start

### Save and Load a Model

```python
from app.ml_engine.utils.serialization import save_model, load_model
from app.ml_engine.models.classification import RandomForestClassifierWrapper
from app.ml_engine.models.base import ModelConfig

# Train a model
config = ModelConfig('random_forest_classifier', {'n_estimators': 100})
model = RandomForestClassifierWrapper(config)
model.fit(X_train, y_train)

# Save the model
save_model(model, 'models/my_model.pkl')

# Load the model
loaded_model = load_model('models/my_model.pkl')
predictions = loaded_model.predict(X_test)
```

### Save and Load a Pipeline

```python
from app.ml_engine.utils.serialization import save_pipeline, load_pipeline
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.imputer import MeanImputer

# Create and fit pipeline
pipeline = Pipeline(steps=[
    MeanImputer(),
    StandardScaler()
])
pipeline.fit(X_train)

# Save the pipeline
save_pipeline(pipeline, 'pipelines/my_pipeline.pkl')

# Load the pipeline
loaded_pipeline = load_pipeline('pipelines/my_pipeline.pkl')
X_transformed = loaded_pipeline.transform(X_test)
```

### Save and Load a Complete Workflow

```python
from app.ml_engine.utils.serialization import save_workflow, load_workflow

# After training
save_workflow(
    pipeline=preprocessing_pipeline,
    model=trained_model,
    path='workflows/customer_churn.pkl',
    workflow_name='Customer_Churn_Predictor'
)

# Load and use
pipeline, model = load_workflow('workflows/customer_churn.pkl')
X_transformed = pipeline.transform(X_new)
predictions = model.predict(X_transformed)
```

## Model Serialization

### Basic Usage

```python
from app.ml_engine.utils.serialization import ModelSerializer

serializer = ModelSerializer()

# Save
serializer.save_model(model, 'models/my_model.pkl')

# Load
loaded_model = serializer.load_model('models/my_model.pkl')
```

### With Compression

```python
# Enable compression for large models
serializer = ModelSerializer(compression=True)
serializer.save_model(model, 'models/large_model.pkl')
# Saves as 'models/large_model.pkl.gz'
```

### With Metadata

```python
metadata = {
    'experiment_id': 'exp_123',
    'dataset_name': 'customer_data',
    'training_date': '2024-01-15',
    'notes': 'Best performing model from grid search'
}

serializer.save_model(model, 'models/my_model.pkl', metadata=metadata)
```

### Overwrite Protection

```python
# First save succeeds
serializer.save_model(model, 'models/my_model.pkl')

# Second save fails
serializer.save_model(model, 'models/my_model.pkl')  # Raises FileExistsError

# Force overwrite
serializer.save_model(model, 'models/my_model.pkl', overwrite=True)
```

### Get Model Info Without Loading

```python
info = serializer.get_model_info('models/my_model.pkl')

print(info)
# {
#     'model_class': 'RandomForestClassifierWrapper',
#     'model_type': 'random_forest_classifier',
#     'is_fitted': True,
#     'n_features': 10,
#     'n_train_samples': 1000,
#     'feature_names': ['age', 'income', ...],
#     'target_name': 'churn',
#     'saved_at': '2024-01-15T10:30:00',
#     'serialization_version': '1.0.0',
#     'sklearn_version': '1.3.0',
#     'file_size_kb': 245.6,
#     'compressed': False
# }
```

## Pipeline Serialization

### Basic Usage

```python
from app.ml_engine.utils.serialization import PipelineSerializer

serializer = PipelineSerializer()

# Save
serializer.save_pipeline(pipeline, 'pipelines/my_pipeline.pkl')

# Load
loaded_pipeline = serializer.load_pipeline('pipelines/my_pipeline.pkl')
```

### Complex Pipeline Example

```python
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer, ModeImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder

# Create complex pipeline
pipeline = Pipeline(steps=[
    MeanImputer(columns=['age', 'income']),
    ModeImputer(columns=['category']),
    OneHotEncoder(columns=['category']),
    StandardScaler(columns=['age', 'income'])
], name='ComplexPreprocessing')

pipeline.fit(X_train)

# Save with metadata
metadata = {
    'dataset_version': 'v2.1',
    'preprocessing_strategy': 'standard'
}

serializer.save_pipeline(pipeline, 'pipelines/complex.pkl', metadata=metadata)

# Load and use
loaded_pipeline = serializer.load_pipeline('pipelines/complex.pkl')
X_test_transformed = loaded_pipeline.transform(X_test)
```

### Get Pipeline Info

```python
info = serializer.get_pipeline_info('pipelines/my_pipeline.pkl')

print(info)
# {
#     'name': 'ComplexPreprocessing',
#     'fitted': True,
#     'num_steps': 4,
#     'step_names': ['MeanImputer', 'ModeImputer', 'OneHotEncoder', 'StandardScaler'],
#     'saved_at': '2024-01-15T10:30:00',
#     'serialization_version': '1.0.0',
#     'file_size_kb': 12.3,
#     'compressed': False
# }
```

## Workflow Serialization

### Complete ML Workflow

```python
from app.ml_engine.utils.serialization import WorkflowSerializer

serializer = WorkflowSerializer(compression=True)  # Recommended for workflows

# Save complete workflow
serializer.save_workflow(
    pipeline=preprocessing_pipeline,
    model=trained_model,
    path='workflows/production_model.pkl',
    workflow_name='Production_Customer_Churn_v1',
    metadata={
        'model_version': '1.0.0',
        'training_accuracy': 0.92,
        'validation_accuracy': 0.89,
        'deployment_date': '2024-01-15'
    }
)

# Load workflow
pipeline, model = serializer.load_workflow('workflows/production_model.pkl')

# Use for inference
def predict_churn(customer_data):
    X_transformed = pipeline.transform(customer_data)
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)
    return predictions, probabilities
```

### Workflow Info

```python
info = serializer.get_workflow_info('workflows/production_model.pkl')

print(info)
# {
#     'workflow_name': 'Production_Customer_Churn_v1',
#     'pipeline_name': 'PreprocessingPipeline',
#     'pipeline_fitted': True,
#     'model_class': 'RandomForestClassifierWrapper',
#     'model_fitted': True,
#     'n_features': 15,
#     'feature_names': [...],
#     'target_name': 'churn',
#     'saved_at': '2024-01-15T14:30:00',
#     'serialization_version': '1.0.0',
#     'sklearn_version': '1.3.0',
#     'file_size_kb': 512.8,
#     'compressed': True
# }
```

## Compression

### When to Use Compression

✅ **Use compression for:**
- Large models (>10 MB)
- Complete workflows
- Long-term storage
- Network transfer

❌ **Skip compression for:**
- Small models (<1 MB)
- Frequent load/save operations
- Development/debugging

### Compression Examples

```python
# Model with compression
save_model(model, 'models/large_model.pkl', compression=True)
# Saves as 'models/large_model.pkl.gz'

# Pipeline with compression
save_pipeline(pipeline, 'pipelines/pipeline.pkl', compression=True)

# Workflow (compression recommended)
save_workflow(pipeline, model, 'workflows/workflow.pkl')
# WorkflowSerializer uses compression by default
```

### Size Comparison

```python
# Without compression
save_model(model, 'models/uncompressed.pkl', compression=False)
uncompressed_size = Path('models/uncompressed.pkl').stat().st_size

# With compression
save_model(model, 'models/compressed.pkl', compression=True)
compressed_size = Path('models/compressed.pkl.gz').stat().st_size

print(f"Uncompressed: {uncompressed_size / 1024:.2f} KB")
print(f"Compressed: {compressed_size / 1024:.2f} KB")
print(f"Compression ratio: {uncompressed_size / compressed_size:.2f}x")
```

## Metadata

### Model Metadata

Automatically saved:
- Model class and module
- Configuration (hyperparameters)
- Training metadata (duration, samples, features)
- Feature names and target name
- Sklearn version
- Serialization version
- Save timestamp

Custom metadata:
```python
metadata = {
    'experiment_id': 'exp_456',
    'dataset_version': 'v2.1',
    'cross_val_score': 0.91,
    'notes': 'Tuned with Bayesian optimization'
}

save_model(model, 'models/my_model.pkl', metadata=metadata)
```

### Pipeline Metadata

Automatically saved:
- Pipeline name
- Number of steps
- Step names and configurations
- Fitted status
- Step statistics
- Serialization version

### Workflow Metadata

Combines model and pipeline metadata plus:
- Workflow name
- Model-pipeline compatibility info
- Complete training history

## Version Compatibility

### Version Checking

```python
# Load with version check (default)
model = load_model('models/old_model.pkl', verify_version=True)
# Warning: Version mismatch: model saved with version 0.9.0, loading with version 1.0.0

# Skip version check
model = load_model('models/old_model.pkl', verify_version=False)
```

### Handling Version Mismatches

```python
import warnings
from app.ml_engine.utils.serialization import VersionMismatchWarning

# Catch version warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model = load_model('models/old_model.pkl')
    
    if w and issubclass(w[-1].category, VersionMismatchWarning):
        print(f"Version mismatch detected: {w[-1].message}")
        # Handle accordingly (retrain, test thoroughly, etc.)
```

## Best Practices

### 1. Organize Your Artifacts

```
project/
├── models/
│   ├── production/
│   │   └── customer_churn_v1.pkl.gz
│   ├── experiments/
│   │   ├── exp_001_rf.pkl
│   │   └── exp_002_xgb.pkl
│   └── archived/
├── pipelines/
│   ├── preprocessing_v1.pkl
│   └── preprocessing_v2.pkl
└── workflows/
    ├── production_workflow.pkl.gz
    └── staging_workflow.pkl.gz
```

### 2. Use Descriptive Names

```python
# Good
save_model(model, 'models/customer_churn_rf_v1_20240115.pkl')

# Bad
save_model(model, 'models/model1.pkl')
```

### 3. Include Metadata

```python
metadata = {
    'experiment_id': 'exp_123',
    'model_version': '1.0.0',
    'training_date': datetime.now().isoformat(),
    'dataset_version': 'v2.1',
    'hyperparameters': model.config.hyperparameters,
    'metrics': {
        'accuracy': 0.92,
        'precision': 0.89,
        'recall': 0.91
    },
    'notes': 'Best model from grid search'
}

save_model(model, 'models/my_model.pkl', metadata=metadata)
```

### 4. Version Your Models

```python
# Use semantic versioning
save_workflow(
    pipeline, model,
    'workflows/customer_churn_v1.0.0.pkl',
    workflow_name='CustomerChurn_v1.0.0'
)
```

### 5. Test After Loading

```python
# Load model
model = load_model('models/production_model.pkl')

# Sanity check
assert model.is_fitted, "Model should be fitted"
assert len(model._feature_names) > 0, "Feature names should be preserved"

# Test prediction
test_sample = X_test.iloc[:1]
prediction = model.predict(test_sample)
assert prediction is not None, "Prediction should work"
```

### 6. Handle Errors Gracefully

```python
from app.ml_engine.utils.serialization import SerializationError

try:
    model = load_model('models/my_model.pkl')
except FileNotFoundError:
    print("Model file not found. Using default model.")
    model = get_default_model()
except SerializationError as e:
    print(f"Failed to load model: {e}")
    # Fallback or alert
```

### 7. Use Compression for Production

```python
# Development (fast load/save)
save_model(model, 'models/dev_model.pkl', compression=False)

# Production (smaller size)
save_model(model, 'models/prod_model.pkl', compression=True)
```

## API Reference

### ModelSerializer

```python
class ModelSerializer:
    def __init__(self, compression: bool = False)
    
    def save_model(
        self,
        model: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Path
    
    def load_model(
        self,
        path: Union[str, Path],
        verify_version: bool = True
    ) -> Any
    
    def get_model_info(self, path: Union[str, Path]) -> Dict[str, Any]
```

### PipelineSerializer

```python
class PipelineSerializer:
    def __init__(self, compression: bool = False)
    
    def save_pipeline(
        self,
        pipeline: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Path
    
    def load_pipeline(
        self,
        path: Union[str, Path],
        verify_version: bool = True
    ) -> Any
    
    def get_pipeline_info(self, path: Union[str, Path]) -> Dict[str, Any]
```

### WorkflowSerializer

```python
class WorkflowSerializer:
    def __init__(self, compression: bool = True)
    
    def save_workflow(
        self,
        pipeline: Any,
        model: Any,
        path: Union[str, Path],
        workflow_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Path
    
    def load_workflow(
        self,
        path: Union[str, Path],
        verify_version: bool = True
    ) -> Tuple[Any, Any]
    
    def get_workflow_info(self, path: Union[str, Path]) -> Dict[str, Any]
```

### Convenience Functions

```python
# Model
save_model(model, path, compression=False, metadata=None, overwrite=False) -> Path
load_model(path, verify_version=True) -> Any
get_model_info(path) -> Dict[str, Any]

# Pipeline
save_pipeline(pipeline, path, compression=False, metadata=None, overwrite=False) -> Path
load_pipeline(path, verify_version=True) -> Any
get_pipeline_info(path) -> Dict[str, Any]

# Workflow
save_workflow(pipeline, model, path, workflow_name=None, metadata=None, overwrite=False) -> Path
load_workflow(path, verify_version=True) -> Tuple[Any, Any]
get_workflow_info(path) -> Dict[str, Any]
```

## Examples

### Example 1: Model Registry

```python
from app.ml_engine.utils.serialization import save_model, load_model, get_model_info
from pathlib import Path

class ModelRegistry:
    def __init__(self, base_path='models'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def register_model(self, model, name, version, metadata=None):
        """Register a model with version."""
        path = self.base_path / f"{name}_v{version}.pkl"
        save_model(model, path, compression=True, metadata=metadata, overwrite=True)
        return path
    
    def get_model(self, name, version):
        """Retrieve a specific model version."""
        path = self.base_path / f"{name}_v{version}.pkl"
        return load_model(path)
    
    def list_models(self):
        """List all registered models."""
        models = []
        for path in self.base_path.glob('*.pkl*'):
            info = get_model_info(path)
            models.append({
                'path': path,
                'name': path.stem,
                **info
            })
        return models

# Usage
registry = ModelRegistry()
registry.register_model(model, 'customer_churn', '1.0.0', metadata={'accuracy': 0.92})
loaded_model = registry.get_model('customer_churn', '1.0.0')
```

### Example 2: Production Deployment

```python
from app.ml_engine.utils.serialization import save_workflow, load_workflow

# Training phase
def train_and_save():
    # Train model
    pipeline = create_preprocessing_pipeline()
    model = train_model(pipeline, X_train, y_train)
    
    # Save for production
    save_workflow(
        pipeline, model,
        'production/customer_churn_v1.pkl',
        workflow_name='CustomerChurn_Production_v1',
        metadata={
            'training_date': datetime.now().isoformat(),
            'accuracy': 0.92,
            'dataset_size': len(X_train)
        }
    )

# Inference phase
def predict_production(customer_data):
    # Load production workflow
    pipeline, model = load_workflow('production/customer_churn_v1.pkl')
    
    # Preprocess and predict
    X_transformed = pipeline.transform(customer_data)
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)
    
    return predictions, probabilities
```

---

**Related Documentation:**
- [ML Engine README](../README.md)
- [Pipeline Documentation](../preprocessing/pipeline.py)
- [Model Base Classes](../models/base.py)
