# Experiment Configuration Serialization Guide

Complete guide for serializing, exporting, and reproducing ML experiments in AI-Playground.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Configuration Structure](#configuration-structure)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The Experiment Configuration Serialization system enables:

- **Reproducibility**: Save complete experiment configurations
- **Sharing**: Export experiments for team collaboration
- **Comparison**: Compare different experiment setups
- **Documentation**: Auto-generate reproduction guides
- **Version Control**: Track experiment configurations

### What Gets Serialized

✅ **Dataset Information**
- Dataset metadata (name, shape, dtypes)
- Missing value statistics
- Upload timestamp

✅ **Preprocessing Pipeline**
- All preprocessing steps in order
- Step parameters and configurations
- Column-specific transformations

✅ **Model Configurations**
- Model types and hyperparameters
- Training status
- Performance metrics (optional)
- Model artifact paths (optional)

✅ **Metadata**
- Experiment name and status
- Creation timestamps
- Serialization version

## Quick Start

### Get Experiment Configuration

```python
from app.services.experiment_config_service import ExperimentConfigSerializer
from app.db.session import get_db

# Initialize serializer
db = next(get_db())
serializer = ExperimentConfigSerializer(db)

# Serialize experiment
config = serializer.serialize_experiment(
    experiment_id,
    include_results=True,
    include_artifacts=False
)

print(config)
```

### Save to File

```python
# Save configuration to JSON file
file_path = serializer.save_to_file(
    experiment_id,
    "experiments/my_experiment.json",
    include_results=True,
    pretty=True
)

print(f"Saved to: {file_path}")
```

### Export Complete Package

```python
# Export reproduction package
exported_files = serializer.export_for_reproduction(
    experiment_id,
    "exports/experiment_package"
)

# Returns paths to:
# - experiment_config.json
# - preprocessing_config.json
# - model_configs.json
# - README.md
```

## API Endpoints

### GET /api/v1/experiments/{experiment_id}/config

Get experiment configuration as JSON.

**Query Parameters:**
- `include_results` (bool): Include training results (default: true)
- `include_artifacts` (bool): Include model artifact paths (default: false)

**Response:**
```json
{
  "version": "1.0.0",
  "experiment": {
    "id": "uuid",
    "name": "Customer Churn Prediction",
    "status": "completed",
    "created_at": "2024-01-15T10:30:00"
  },
  "dataset": {
    "name": "customer_data.csv",
    "shape": {"rows": 10000, "columns": 15}
  },
  "preprocessing": [...],
  "models": [...],
  "metadata": {...}
}
```

### GET /api/v1/experiments/{experiment_id}/config/download

Download configuration as JSON file.

**Response:** JSON file download

### GET /api/v1/experiments/{experiment_id}/export

Export complete reproduction package as ZIP.

**Response:** ZIP file containing:
- `experiment_config.json`
- `preprocessing_config.json`
- `model_configs.json`
- `README.md`

### GET /api/v1/experiments/{experiment_id}/preprocessing-config

Get only preprocessing configuration.

**Response:**
```json
{
  "experiment_id": "uuid",
  "experiment_name": "My Experiment",
  "preprocessing": [
    {
      "step_type": "imputation",
      "parameters": {"strategy": "mean"},
      "column_name": "age",
      "order": 0
    }
  ]
}
```

### GET /api/v1/experiments/{experiment_id}/model-configs

Get only model configurations.

**Query Parameters:**
- `include_results` (bool): Include metrics (default: false)

### POST /api/v1/experiments/compare

Compare two experiments.

**Query Parameters:**
- `experiment_id_1` (UUID): First experiment
- `experiment_id_2` (UUID): Second experiment

**Response:**
```json
{
  "experiments": {
    "experiment_1": "Experiment A",
    "experiment_2": "Experiment B"
  },
  "differences": {
    "preprocessing": {...},
    "models": {...}
  },
  "summary": {
    "same_preprocessing": false,
    "same_models": false
  }
}
```

### GET /api/v1/experiments/{experiment_id}/summary

Get high-level experiment summary.

**Response:**
```json
{
  "experiment_id": "uuid",
  "experiment_name": "My Experiment",
  "status": "completed",
  "dataset": {
    "name": "data.csv",
    "shape": {"rows": 1000, "columns": 10}
  },
  "preprocessing": {
    "num_steps": 3,
    "step_types": ["imputation", "scaling", "encoding"]
  },
  "models": {
    "num_models": 2,
    "model_types": ["random_forest", "logistic_regression"],
    "completed": 2,
    "failed": 0
  },
  "best_model": {
    "model_type": "random_forest",
    "metrics": {"accuracy": 0.95}
  }
}
```

## Configuration Structure

### Complete Configuration

```json
{
  "version": "1.0.0",
  "experiment": {
    "id": "experiment-uuid",
    "name": "Customer Churn Prediction",
    "status": "completed",
    "created_at": "2024-01-15T10:30:00",
    "user_id": "user-uuid",
    "dataset_id": "dataset-uuid"
  },
  "dataset": {
    "id": "dataset-uuid",
    "name": "customer_data.csv",
    "file_path": "/uploads/customer_data.csv",
    "shape": {"rows": 10000, "columns": 15},
    "dtypes": {
      "age": "int64",
      "income": "float64",
      "category": "object"
    },
    "missing_values": {
      "age": 50,
      "income": 120
    },
    "uploaded_at": "2024-01-15T09:00:00"
  },
  "preprocessing": [
    {
      "id": "step-uuid",
      "step_type": "imputation",
      "parameters": {
        "strategy": "mean"
      },
      "column_name": "age",
      "order": 0
    },
    {
      "id": "step-uuid",
      "step_type": "scaling",
      "parameters": {
        "method": "standard"
      },
      "column_name": null,
      "order": 1
    }
  ],
  "models": [
    {
      "id": "model-uuid",
      "model_type": "random_forest",
      "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
      },
      "status": "completed",
      "metrics": {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.94,
        "f1_score": 0.935
      },
      "training_time": 45.2,
      "created_at": "2024-01-15T10:30:00"
    }
  ],
  "metadata": {
    "serialized_at": "2024-01-15T11:00:00",
    "serializer_version": "1.0.0"
  }
}
```

## Use Cases

### 1. Experiment Reproducibility

Save experiment configuration for later reproduction:

```python
# Save configuration
serializer.save_to_file(
    experiment_id,
    "experiments/prod_model_v1.json"
)

# Later: Load and reproduce
config = serializer.load_from_file("experiments/prod_model_v1.json")

# Reconstruct preprocessing pipeline
from app.ml_engine.preprocessing.pipeline import Pipeline
pipeline = Pipeline.from_config(config["preprocessing"])

# Reconstruct models
from app.ml_engine.models.registry import ModelFactory
for model_config in config["models"]:
    model = ModelFactory.create_model(
        model_config["model_type"],
        **model_config["hyperparameters"]
    )
```

### 2. Team Collaboration

Export experiment package for sharing:

```python
# Export complete package
exported_files = serializer.export_for_reproduction(
    experiment_id,
    "shared/experiment_package"
)

# Share the directory with team
# They get:
# - Full configuration
# - Preprocessing steps
# - Model configs
# - Reproduction instructions (README.md)
```

### 3. Experiment Comparison

Compare different experiment setups:

```python
# Compare two experiments
comparison = serializer.compare_experiments(
    experiment_id_1,
    experiment_id_2
)

print(f"Same preprocessing: {comparison['summary']['same_preprocessing']}")
print(f"Same models: {comparison['summary']['same_models']}")

# See differences
print(comparison['differences'])
```

### 4. Version Control

Track experiment configurations in Git:

```python
# Save configuration (JSON is Git-friendly)
serializer.save_to_file(
    experiment_id,
    "configs/experiment_v1.0.0.json",
    include_results=False,  # Don't include results in version control
    include_artifacts=False
)

# Commit to Git
# git add configs/experiment_v1.0.0.json
# git commit -m "Add experiment v1.0.0 configuration"
```

### 5. Production Deployment

Export configuration for production:

```python
# Export with artifacts
config = serializer.serialize_experiment(
    experiment_id,
    include_results=True,
    include_artifacts=True  # Include model file paths
)

# Deploy configuration and models to production
deploy_to_production(config)
```

## Best Practices

### 1. Naming Conventions

✅ **DO**:
```python
# Use descriptive names with versions
"customer_churn_v1.0.0.json"
"fraud_detection_experiment_20240115.json"
"production_model_config.json"
```

❌ **DON'T**:
```python
# Avoid generic names
"config.json"
"experiment.json"
"final.json"
```

### 2. Version Control

✅ **DO**:
```python
# Save configurations without results for Git
serializer.save_to_file(
    experiment_id,
    "configs/experiment.json",
    include_results=False,
    include_artifacts=False
)
```

❌ **DON'T**:
```python
# Don't commit large result files
serializer.save_to_file(
    experiment_id,
    "configs/experiment.json",
    include_results=True,  # Results can be large
    include_artifacts=True  # Paths may be environment-specific
)
```

### 3. Documentation

✅ **DO**:
```python
# Export with README for sharing
exported_files = serializer.export_for_reproduction(
    experiment_id,
    "shared/experiment"
)

# README.md is auto-generated with:
# - Experiment overview
# - Preprocessing steps
# - Model configurations
# - Reproduction instructions
```

### 4. Comparison

✅ **DO**:
```python
# Compare before deploying
comparison = serializer.compare_experiments(
    current_prod_experiment_id,
    new_experiment_id
)

if comparison['summary']['same_preprocessing']:
    print("Can reuse preprocessing pipeline")
```

### 5. Error Handling

✅ **DO**:
```python
try:
    config = serializer.serialize_experiment(experiment_id)
except ValueError as e:
    logger.error(f"Experiment not found: {e}")
    # Handle gracefully
except Exception as e:
    logger.error(f"Serialization failed: {e}")
    # Fallback or retry
```

## Examples

### Example 1: Save and Load

```python
from app.services.experiment_config_service import ExperimentConfigSerializer
from app.db.session import get_db

db = next(get_db())
serializer = ExperimentConfigSerializer(db)

# Save
file_path = serializer.save_to_file(
    experiment_id,
    "experiments/my_experiment.json"
)

# Load
config = serializer.load_from_file(file_path)
print(f"Experiment: {config['experiment']['name']}")
print(f"Models: {len(config['models'])}")
```

### Example 2: Export Package

```python
# Export complete package
exported_files = serializer.export_for_reproduction(
    experiment_id,
    "exports/customer_churn_v1"
)

print("Exported files:")
for file_type, path in exported_files.items():
    print(f"  {file_type}: {path}")
```

### Example 3: Compare Experiments

```python
# Compare two experiments
comparison = serializer.compare_experiments(
    baseline_experiment_id,
    new_experiment_id
)

print(f"Baseline: {comparison['experiments']['experiment_1']}")
print(f"New: {comparison['experiments']['experiment_2']}")
print(f"\nSame preprocessing: {comparison['summary']['same_preprocessing']}")
print(f"Same models: {comparison['summary']['same_models']}")

if not comparison['summary']['same_preprocessing']:
    print("\nPreprocessing differences:")
    print(comparison['differences']['preprocessing'])
```

### Example 4: API Usage

```bash
# Get configuration
curl -X GET "http://localhost:8000/api/v1/experiments/{experiment_id}/config?include_results=true"

# Download configuration
curl -X GET "http://localhost:8000/api/v1/experiments/{experiment_id}/config/download" \
  -o experiment_config.json

# Export package
curl -X GET "http://localhost:8000/api/v1/experiments/{experiment_id}/export" \
  -o experiment_package.zip

# Compare experiments
curl -X POST "http://localhost:8000/api/v1/experiments/compare?experiment_id_1={id1}&experiment_id_2={id2}"

# Get summary
curl -X GET "http://localhost:8000/api/v1/experiments/{experiment_id}/summary"
```

## Troubleshooting

### Issue: "Experiment not found"

**Solution**: Verify experiment ID exists in database

```python
from app.models.experiment import Experiment

experiment = db.query(Experiment).filter(
    Experiment.id == experiment_id
).first()

if not experiment:
    print("Experiment does not exist")
```

### Issue: "Version mismatch warning"

**Solution**: This is usually safe. The serializer is backward compatible.

### Issue: "Large configuration file"

**Solution**: Exclude results and artifacts

```python
config = serializer.serialize_experiment(
    experiment_id,
    include_results=False,
    include_artifacts=False
)
```

---

**Related Documentation:**
- [Model Serialization Guide](../ml_engine/utils/SERIALIZATION_GUIDE.md)
- [Pipeline Serialization](../ml_engine/preprocessing/SERIALIZATION.md)
- [API Documentation](../../docs/api.md)
