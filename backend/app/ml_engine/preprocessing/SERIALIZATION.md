## Pipeline Serialization Guide

Complete guide to serializing and deserializing preprocessing pipelines for production deployment and team collaboration.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Serialization Formats](#serialization-formats)
4. [Compression Options](#compression-options)
5. [API Reference](#api-reference)
6. [Production Deployment](#production-deployment)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline serialization system provides robust tools for saving, loading, and managing preprocessing pipelines across different environments.

### Key Features

- **Multiple Formats**: Pickle, JSON, Joblib, YAML
- **Compression Support**: gzip, bz2, lzma
- **Version Tracking**: Automatic versioning for backward compatibility
- **Metadata Preservation**: Store pipeline metadata and statistics
- **Registry System**: Organize and catalog multiple pipelines
- **Auto-detection**: Automatic format and compression detection

### Use Cases

1. **Production Deployment**: Save fitted pipelines for use in production
2. **Version Control**: Track pipeline configurations in Git
3. **Team Collaboration**: Share pipeline configurations across teams
4. **A/B Testing**: Maintain multiple pipeline versions
5. **Disaster Recovery**: Backup and restore pipeline states

---

## Quick Start

### Basic Usage

```python
from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.serializer import save_pipeline, load_pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.scaler import StandardScaler
import pandas as pd

# Create and fit pipeline
df = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 70000]})

pipeline = Pipeline(
    steps=[
        MeanImputer(columns=["age", "salary"]),
        StandardScaler(columns=["age", "salary"])
    ],
    name="MyPipeline"
)
pipeline.fit(df)

# Save pipeline
save_pipeline(pipeline, "my_pipeline.pkl")

# Load pipeline
loaded = load_pipeline("my_pipeline.pkl")

# Use loaded pipeline
result = loaded.transform(df_new)
```

---

## Serialization Formats

### 1. Pickle Format (Recommended for Production)

**Best for**: Production deployment with fitted parameters

```python
from app.ml_engine.preprocessing.serializer import PipelineSerializer

serializer = PipelineSerializer(default_format="pickle")

# Save fitted pipeline with all parameters
serializer.save(pipeline, "pipeline.pkl")

# Load complete pipeline
loaded = serializer.load("pipeline.pkl")
```

**Pros**:
- Preserves complete pipeline state
- Includes all fitted parameters
- Fast serialization/deserialization
- Binary format (compact)

**Cons**:
- Not human-readable
- Python version dependent
- Binary format (not version-control friendly)

---

### 2. JSON Format (Recommended for Configuration)

**Best for**: Version control, configuration sharing

```python
serializer = PipelineSerializer(default_format="json")

# Save configuration only (no fitted params)
serializer.save(pipeline, "pipeline.json")

# Load and reconstruct pipeline
loaded_data = serializer.load("pipeline.json")
config = loaded_data["pipeline"]  # Configuration dict
pipeline = Pipeline.from_dict(config)  # Reconstruct
```

**Pros**:
- Human-readable
- Version control friendly
- Language agnostic
- Easy to edit manually

**Cons**:
- Configuration only (no fitted parameters)
- Larger file size
- Requires reconstruction

**Example JSON Output**:
```json
{
  "serializer_version": "1.0.0",
  "schema_version": 1,
  "timestamp": "2024-01-15T10:30:00",
  "metadata": {
    "name": "MyPipeline",
    "fitted": true,
    "num_steps": 2,
    "step_names": ["MeanImputer", "StandardScaler"]
  },
  "pipeline_config": {
    "name": "MyPipeline",
    "steps": [
      {
        "class": "MeanImputer",
        "name": "mean_imputer",
        "params": {"columns": ["age", "salary"]}
      },
      {
        "class": "StandardScaler",
        "name": "standard_scaler",
        "params": {"columns": ["age", "salary"]}
      }
    ]
  }
}
```

---

### 3. Joblib Format (Optimized Binary)

**Best for**: Large NumPy arrays, scientific computing

```python
serializer = PipelineSerializer(default_format="joblib")

# Save with joblib
serializer.save(pipeline, "pipeline.joblib")

# Load
loaded = serializer.load("pipeline.joblib")
```

**Requirements**: `pip install joblib`

**Pros**:
- Optimized for NumPy arrays
- Efficient compression
- Fast for large datasets
- Preserves complete state

**Cons**:
- Requires joblib library
- Binary format
- Not human-readable

---

### 4. YAML Format (Human-Readable Configuration)

**Best for**: Documentation, readable configs

```python
serializer = PipelineSerializer(default_format="yaml")

# Save configuration as YAML
serializer.save(pipeline, "pipeline.yml")

# Load
loaded_data = serializer.load("pipeline.yml")
```

**Requirements**: `pip install pyyaml`

**Pros**:
- Very human-readable
- Great for documentation
- Comments supported
- Popular format

**Cons**:
- Configuration only
- Requires pyyaml library
- Slower than binary formats

---

## Compression Options

Reduce file size with compression algorithms.

### Comparison Table

| Compression | Speed | Ratio | Best For |
|-------------|-------|-------|----------|
| none | Fastest | 1.0x | Small pipelines, local development |
| gzip | Fast | ~3-5x | General purpose, good balance |
| bz2 | Medium | ~4-6x | Better compression, slower |
| lzma | Slow | ~5-8x | Best compression, archival |

### Usage

```python
# With gzip compression
serializer = PipelineSerializer(
    default_format="pickle",
    compression="gzip"
)

serializer.save(pipeline, "pipeline.pkl.gz")
loaded = serializer.load("pipeline.pkl.gz")  # Auto-detects compression
```

### Benchmark Example

```python
# No compression
save_pipeline(pipeline, "pipeline.pkl", compression="none")
# Size: 45,234 bytes

# With gzip
save_pipeline(pipeline, "pipeline.pkl.gz", compression="gzip")
# Size: 12,456 bytes (72% reduction)

# With bz2
save_pipeline(pipeline, "pipeline.pkl.bz2", compression="bz2")
# Size: 10,123 bytes (78% reduction)

# With lzma
save_pipeline(pipeline, "pipeline.pkl.xz", compression="lzma")
# Size: 8,934 bytes (80% reduction)
```

---

## API Reference

### PipelineSerializer Class

Main class for pipeline serialization.

```python
class PipelineSerializer:
    def __init__(
        self,
        default_format: str = "pickle",
        compression: str = "none",
        include_metadata: bool = True
    )
```

#### Methods

**save(pipeline, path, format=None, compression=None, metadata=None)**

Save a pipeline to disk.

```python
file_info = serializer.save(
    pipeline,
    "path/to/pipeline.pkl",
    metadata={"version": "1.0.0", "author": "John Doe"}
)
```

Returns:
```python
{
    "path": "/absolute/path/to/pipeline.pkl",
    "format": "pickle",
    "compression": "gzip",
    "size_bytes": 12456,
    "checksum": "a1b2c3d4...",
    "timestamp": "2024-01-15T10:30:00",
    "serializer_version": "1.0.0",
    "schema_version": 1
}
```

**load(path, format=None, compression=None, validate=True)**

Load a pipeline from disk.

```python
pipeline = serializer.load("path/to/pipeline.pkl")
```

---

### PipelineRegistry Class

Manage multiple pipelines with a catalog system.

```python
from app.ml_engine.preprocessing.serializer import PipelineRegistry

registry = PipelineRegistry("path/to/registry.json")
```

#### Methods

**register(pipeline_id, file_info, tags=None, description=None)**

Register a saved pipeline.

```python
registry.register(
    "prod-pipeline-v1",
    file_info,
    tags=["production", "v1"],
    description="Production classification pipeline"
)
```

**list(tags=None)**

List pipelines, optionally filtered by tags.

```python
# All pipelines
all_pipelines = registry.list()

# Filter by tag
ml_pipelines = registry.list(tags=["ml", "classification"])
```

**search(query)**

Search pipelines by keyword.

```python
results = registry.search("classification")
```

**get(pipeline_id)**

Get info for a specific pipeline.

```python
info = registry.get("prod-pipeline-v1")
```

**unregister(pipeline_id)**

Remove a pipeline from registry.

```python
registry.unregister("old-pipeline")
```

---

### Convenience Functions

**save_pipeline(pipeline, path, format="pickle", compression="none", metadata=None)**

Quick save function.

```python
from app.ml_engine.preprocessing.serializer import save_pipeline

file_info = save_pipeline(
    pipeline,
    "pipeline.pkl",
    format="pickle",
    compression="gzip"
)
```

**load_pipeline(path, format=None, compression=None)**

Quick load function.

```python
from app.ml_engine.preprocessing.serializer import load_pipeline

pipeline = load_pipeline("pipeline.pkl")
```

---

## Production Deployment

### Complete Production Workflow

```python
# 1. Development: Create and train pipeline
pipeline = Pipeline(
    steps=[
        MeanImputer(columns=["age", "salary"]),
        StandardScaler(columns=["age", "salary"])
    ],
    name="ProductionPipeline_v1.0.0"
)
pipeline.fit(training_data)

# 2. Save configuration for version control
serializer = PipelineSerializer(default_format="json")
serializer.save(pipeline, "configs/pipeline_v1.0.0.json")
# → Commit to Git

# 3. Save fitted pipeline for deployment
serializer = PipelineSerializer(default_format="pickle", compression="gzip")
file_info = serializer.save(
    pipeline,
    "models/pipeline_v1.0.0.pkl.gz",
    metadata={
        "version": "1.0.0",
        "trained_date": "2024-01-15",
        "dataset_size": len(training_data)
    }
)
# → Deploy to production servers

# 4. Register in production registry
registry = PipelineRegistry("production_registry.json")
registry.register(
    "prod-pipeline-v1.0.0",
    file_info,
    tags=["production", "v1.0.0"],
    description="Production pipeline version 1.0.0"
)

# 5. Production: Load and use
prod_pipeline = load_pipeline("models/pipeline_v1.0.0.pkl.gz")
result = prod_pipeline.transform(new_data)

# 6. Verification
assert prod_pipeline.fitted
assert len(prod_pipeline.steps) == 2
assert file_info["checksum"] == calculate_checksum("models/pipeline_v1.0.0.pkl.gz")
```

### Version Management

```python
# Save multiple versions
versions = ["v1.0.0", "v1.1.0", "v2.0.0"]

for version in versions:
    pipeline = create_pipeline_version(version)
    save_pipeline(pipeline, f"models/pipeline_{version}.pkl.gz")

    registry.register(
        f"pipeline-{version}",
        file_info,
        tags=["versioned", version]
    )

# Rollback to previous version
v1_info = registry.get("pipeline-v1.1.0")
rollback_pipeline = load_pipeline(v1_info["path"])
```

---

## Best Practices

### 1. Format Selection

✅ **DO**:
- Use **pickle + compression** for production deployment
- Use **JSON** for version control and configuration sharing
- Use **joblib** for pipelines with large NumPy arrays

❌ **DON'T**:
- Don't use JSON for fitted pipelines (loses fitted parameters)
- Don't use pickle for version control (binary, not diff-friendly)

### 2. Compression

✅ **DO**:
- Use **gzip** for general purpose (good balance)
- Use **lzma** for archival/backup (best compression)
- Use **none** for development/debugging (fastest)

❌ **DON'T**:
- Don't compress tiny files (< 1KB overhead)
- Don't use slow compression for frequently accessed files

### 3. Metadata

✅ **DO**:
```python
metadata = {
    "version": "1.0.0",
    "author": "Data Science Team",
    "trained_date": "2024-01-15",
    "dataset_size": 10000,
    "accuracy": 0.95,
    "environment": "production"
}
serializer.save(pipeline, path, metadata=metadata)
```

❌ **DON'T**:
- Don't store sensitive data in metadata
- Don't omit version information

### 4. File Naming

✅ **DO**:
```python
# Good naming conventions
"pipeline_v1.0.0.pkl.gz"
"classification_pipeline_20240115.pkl"
"production_pipeline_v2.pkl.gz"
```

❌ **DON'T**:
```python
# Bad naming
"pipeline.pkl"  # No version
"v1.pkl"  # Not descriptive
"final_final_v2.pkl"  # Confusing
```

### 5. Registry Usage

✅ **DO**:
```python
# Organize with tags
registry.register(
    "pipeline-id",
    file_info,
    tags=["production", "v1.0.0", "classification"],
    description="Clear description of purpose"
)

# Search and filter
prod_pipelines = registry.list(tags=["production"])
```

### 6. Error Handling

✅ **DO**:
```python
try:
    pipeline = load_pipeline("pipeline.pkl")
except FileNotFoundError:
    logger.error("Pipeline file not found")
    # Use fallback or default pipeline
except Exception as e:
    logger.error(f"Failed to load pipeline: {e}")
    # Handle gracefully
```

### 7. Testing

✅ **DO**:
```python
# Test save/load roundtrip
original_result = pipeline.transform(test_data)
save_pipeline(pipeline, "test.pkl")
loaded = load_pipeline("test.pkl")
loaded_result = loaded.transform(test_data)

assert original_result.equals(loaded_result)
```

---

## Troubleshooting

### Common Issues

#### Issue: "FileNotFoundError: Pipeline file not found"

**Solution**:
```python
from pathlib import Path

path = Path("pipeline.pkl")
if not path.exists():
    print(f"File not found: {path.absolute()}")
    # Check path is correct
```

#### Issue: "Version mismatch warning"

**Solution**:
- This is usually safe (backward compatible)
- Update serializer if needed
- Re-save pipeline with current version

#### Issue: "ImportError: joblib not available"

**Solution**:
```bash
pip install joblib
```

#### Issue: "Large file size"

**Solution**:
```python
# Use compression
save_pipeline(pipeline, "pipeline.pkl.gz", compression="gzip")

# Or use joblib format
save_pipeline(pipeline, "pipeline.joblib", format="joblib")
```

#### Issue: "Cannot reconstruct pipeline from JSON"

**Solution**:
```python
# JSON saves configuration only
loaded_data = load_pipeline("pipeline.json")

# Need to reconstruct:
if isinstance(loaded_data, dict):
    config = loaded_data["pipeline"]
    pipeline = Pipeline.from_dict(config)

    # Then fit on new data
    pipeline.fit(training_data)
```

#### Issue: "Pickle protocol version mismatch"

**Solution**:
```python
# Save with compatible protocol
import pickle
serializer.save(pipeline, path)  # Uses HIGHEST_PROTOCOL

# If compatibility needed, modify source to use:
# pickle.dumps(data, protocol=4)  # Compatible with Python 3.4+
```

---

## API Endpoints

### REST API for Serialization

#### Export Pipeline

```bash
POST /api/v1/pipelines/{pipeline_id}/export
{
  "format": "pickle",  # pickle, json, joblib, yaml
  "compression": "gzip"  # none, gzip, bz2, lzma
}
```

#### Import Pipeline

```bash
POST /api/v1/pipelines/import
{
  "file_path": "/path/to/pipeline.pkl",
  "name": "Imported Pipeline",
  "description": "Pipeline imported from file"
}
```

#### Export Configuration

```bash
GET /api/v1/pipelines/{pipeline_id}/export-config?format=json
```

#### Import Configuration

```bash
POST /api/v1/pipelines/import-config
{
  "config": {/* pipeline configuration */},
  "name": "Pipeline from Config"
}
```

---

## Examples

See [`backend/examples/pipeline_serialization_example.py`](../../../examples/pipeline_serialization_example.py) for comprehensive examples including:

1. Basic save and load
2. Configuration export/import
3. Compression comparison
4. Multiple formats
5. Pipeline registry usage
6. Production deployment workflow
7. Versioning and rollback

---

## Additional Resources

- [Pipeline Documentation](README.md)
- [Configuration Management](CONFIG.md)
- [API Documentation](../../../docs/api.md)
- [Testing Guide](../../../tests/README.md)

---

## Support

For issues or questions:
1. Check this documentation
2. Review examples in `backend/examples/`
3. Check test files for usage patterns
4. Open an issue on GitHub
