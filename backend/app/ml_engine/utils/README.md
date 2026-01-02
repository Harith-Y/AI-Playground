# ML Engine Utilities

This module contains utility functions and classes for the ML Engine.

## Column Type Detector

Automatic detection of column types in pandas DataFrames beyond basic pandas dtypes.

### Features

- **16 distinct column types** including:
  - Identifiers (ID)
  - Temporal (datetime, date, time)
  - Numeric (continuous, discrete, binary)
  - Categorical (ordinal, nominal, binary)
  - Text (long, short)
  - Special (boolean, constant, mixed, unknown)

- **Smart heuristics** based on:
  - Unique value ratios
  - Data patterns (regex matching)
  - Statistical properties
  - String length analysis
  - Name pattern matching

- **Configurable thresholds** for categorical, ID, and text detection
- **Efficient sampling** for large datasets
- **Detailed column information** with statistics

### Quick Start

```python
from app.ml_engine.utils import detect_column_types, ColumnType
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    'user_id': range(1000),
    'age': [25, 30, 35, ...],
    'country': ['USA', 'UK', 'Canada', ...],
    'description': ['Long text about...', ...],
})

# Detect column types
types = detect_column_types(df)

# Print results
for col, col_type in types.items():
    print(f"{col}: {col_type.value}")

# Output:
# user_id: id
# age: numeric_discrete
# country: categorical_nominal
# description: text_long
```

### Advanced Usage

```python
from app.ml_engine.utils import ColumnTypeDetector

# Create detector with custom thresholds
detector = ColumnTypeDetector(
    categorical_threshold=0.1,  # 10% unique ratio for categorical
    id_threshold=0.90,          # 90% unique ratio for IDs
    text_length_threshold=30,   # Avg length > 30 for long text
    sample_size=5000,           # Sample size for large datasets
)

# Detect types
types = detector.detect(df)

# Get detailed information
column_info = detector.get_column_info(df)
print(column_info)
```

### Column Types Reference

| Type | Description | Example |
|------|-------------|---------|
| `id` | Unique identifiers | user_id, order_id |
| `datetime` | Timestamp columns | created_at, updated_at |
| `date` | Date-only columns | birth_date, order_date |
| `time` | Time-only columns | appointment_time |
| `numeric_continuous` | Continuous numeric | price, temperature |
| `numeric_discrete` | Discrete numeric | age, count |
| `numeric_binary` | Binary 0/1 numeric | is_active (0/1) |
| `categorical_ordinal` | Ordered categories | education level |
| `categorical_nominal` | Unordered categories | country, color |
| `categorical_binary` | Two-category | gender, status |
| `text_long` | Long text | descriptions, reviews |
| `text_short` | Short text | names, titles |
| `boolean` | Boolean values | is_premium, verified |
| `constant` | Single value | platform='Web' |
| `mixed` | Mixed types | problematic data |
| `unknown` | Cannot determine | edge cases |

### Use Cases

1. **Automatic Preprocessing**
   ```python
   types = detect_column_types(df)

   # Apply appropriate preprocessing
   for col, col_type in types.items():
       if col_type == ColumnType.NUMERIC_CONTINUOUS:
           df[col] = StandardScaler().fit_transform(df[[col]])
       elif col_type == ColumnType.CATEGORICAL_NOMINAL:
           df = pd.get_dummies(df, columns=[col])
   ```

2. **Feature Selection**
   ```python
   types = detect_column_types(df)

   # Remove ID and constant columns
   cols_to_drop = [col for col, t in types.items()
                   if t in [ColumnType.ID, ColumnType.CONSTANT]]
   df = df.drop(columns=cols_to_drop)
   ```

3. **Data Quality Analysis**
   ```python
   detector = ColumnTypeDetector()
   info = detector.get_column_info(df)

   # Find problematic columns
   mixed_cols = info[info['detected_type'] == 'mixed']['column'].tolist()
   high_null_cols = info[info['null_percentage'] > 50]['column'].tolist()
   ```

### Configuration

#### Categorical Threshold
- Default: 0.05 (5%)
- Controls when a column is considered categorical vs text
- Lower = more columns classified as categorical

#### ID Threshold
- Default: 0.95 (95%)
- Controls when a column is considered an ID
- Higher = stricter ID detection

#### Text Length Threshold
- Default: 50 characters
- Separates long text from short text
- Higher = more columns classified as short text

### Performance

- For datasets > 10,000 rows, automatic sampling is used
- Detection typically takes < 1 second for 100k rows
- Memory efficient - only samples data for analysis

### Example Output

```
Column                   Type                  Unique%  Null%
user_id                  id                    100.0    0.0
age                      numeric_discrete      62.0     0.0
income                   numeric_continuous    98.5     5.0
country                  categorical_nominal   4.0      0.0
education                categorical_ordinal   4.0      0.0
gender                   categorical_binary    2.0      0.0
is_premium               boolean               2.0      0.0
signup_date              datetime              100.0    0.0
customer_name            text_short            95.0     0.0
product_review           text_long             87.0     3.0
platform                 constant              1.0      0.0
```

See `examples/column_type_detection_example.py` for a complete working example.

---

## Model & Pipeline Serialization

Comprehensive serialization utilities for saving and loading ML models, preprocessing pipelines, and complete workflows.

### Features

- **Model Serialization**: Save/load fitted model wrappers with all metadata
- **Pipeline Serialization**: Save/load preprocessing pipelines with fitted transformers
- **Workflow Serialization**: Save/load complete ML workflows (preprocessing + model)
- **Compression Support**: Optional gzip compression for reduced file size
- **Version Tracking**: Serialization format versioning and compatibility checks
- **Metadata Preservation**: Training info, feature names, hyperparameters, and custom metadata
- **File Integrity**: SHA256 hash checking for corrupted files
- **Overwrite Protection**: Prevents accidental file overwrites

### Core Classes

#### ModelSerializer
- Save/load trained models with complete metadata
- Compression support (gzip)
- Version compatibility checks
- Metadata: training info, feature names, configurations

#### PipelineSerializer
- Save/load preprocessing pipelines
- Preserves fitted transformers and step configurations
- Pipeline statistics tracking

#### WorkflowSerializer
- Save/load complete ML workflows (pipeline + model)
- End-to-end reproducibility
- Combined metadata from both components

### Quick Start

#### Save and Load a Model

```python
from app.ml_engine.utils.serialization import save_model, load_model

# Train a model
from app.ml_engine.models import RandomForestClassifierWrapper
model = RandomForestClassifierWrapper(n_estimators=100)
model.fit(X_train, y_train)

# Save with compression
save_model(model, 'models/my_model.pkl', compression=True)

# Load and use
loaded_model = load_model('models/my_model.pkl')
predictions = loaded_model.predict(X_test)
```

#### Save and Load a Pipeline

```python
from app.ml_engine.utils.serialization import save_pipeline, load_pipeline
from app.ml_engine.preprocessing import Pipeline

# Create and fit pipeline
pipeline = Pipeline([
    ('imputer', MeanImputer()),
    ('scaler', StandardScaler()),
    ('encoder', OneHotEncoder())
])
pipeline.fit(X_train)

# Save
save_pipeline(pipeline, 'pipelines/preprocessing.pkl')

# Load and use
loaded_pipeline = load_pipeline('pipelines/preprocessing.pkl')
X_transformed = loaded_pipeline.transform(X_test)
```

#### Save and Load a Complete Workflow

```python
from app.ml_engine.utils.serialization import save_workflow, load_workflow

# Save complete workflow
save_workflow(
    pipeline=preprocessing_pipeline,
    model=trained_model,
    path='workflows/production.pkl',
    workflow_name='CustomerChurn_v1',
    compression=True
)

# Load and use for inference
pipeline, model = load_workflow('workflows/production.pkl')
X_transformed = pipeline.transform(X_new)
predictions = model.predict(X_transformed)
```

### Get Info Without Loading

```python
from app.ml_engine.utils.serialization import get_model_info, get_workflow_info

# Get model information
info = get_model_info('models/my_model.pkl')
print(f"Model: {info['model_type']}")
print(f"Features: {info['n_features']}")
print(f"Trained: {info['metadata']['trained_at']}")
print(f"Size: {info['file_size_kb']:.2f} KB")

# Get workflow information
workflow_info = get_workflow_info('workflows/production.pkl')
print(f"Pipeline steps: {workflow_info['pipeline_steps']}")
print(f"Model: {workflow_info['model_type']}")
```

### Advanced Usage

#### Custom Metadata

```python
from app.ml_engine.utils.serialization import ModelSerializer

serializer = ModelSerializer()

# Save with custom metadata
serializer.save(
    model=trained_model,
    path='models/production_v2.pkl',
    model_name='ProductionModel_v2',
    custom_metadata={
        'dataset_version': '2.0',
        'author': 'data-science-team',
        'deployment_date': '2026-01-02',
        'performance_metrics': {
            'accuracy': 0.95,
            'f1_score': 0.93
        }
    }
)

# Load and access metadata
loaded_model = serializer.load('models/production_v2.pkl')
metadata = serializer.get_metadata('models/production_v2.pkl')
print(metadata['custom_metadata'])
```

#### Version Compatibility

```python
from app.ml_engine.utils.serialization import load_model

# Load with version checking
try:
    model = load_model('models/old_model.pkl')
except SerializationError as e:
    print(f"Version mismatch: {e}")
    # Handle version incompatibility
```

### Use Cases

1. **Model Deployment**
   ```python
   # Save production model with all metadata
   save_workflow(
       pipeline=pipeline,
       model=model,
       path='production/model_v1.0.pkl',
       workflow_name='Production_CustomerChurn_v1.0',
       compression=True
   )
   
   # Deploy: Load and serve
   pipeline, model = load_workflow('production/model_v1.0.pkl')
   ```

2. **Experiment Tracking**
   ```python
   # Save multiple model versions
   for i, (name, model) in enumerate(trained_models.items()):
       save_model(
           model=model,
           path=f'experiments/experiment_{i}_{name}.pkl',
           model_name=name,
           custom_metadata={'experiment_id': i}
       )
   
   # Compare later
   for file in Path('experiments').glob('*.pkl'):
       info = get_model_info(file)
       print(f"{info['model_name']}: {info['metadata']}")
   ```

3. **Model Versioning**
   ```python
   from datetime import datetime
   
   # Save with version info
   version = datetime.now().strftime('%Y%m%d_%H%M%S')
   save_model(
       model=model,
       path=f'models/production_{version}.pkl',
       custom_metadata={
           'version': version,
           'metrics': training_metrics,
           'data_hash': data_checksum
       }
   )
   ```

4. **Collaboration**
   ```python
   # Team member saves trained model
   save_workflow(
       pipeline=pipeline,
       model=model,
       path='shared/team_model.pkl',
       workflow_name='TeamModel',
       custom_metadata={
           'author': 'alice',
           'notes': 'Best performing model on validation set'
       }
   )
   
   # Another team member loads and uses
   pipeline, model = load_workflow('shared/team_model.pkl')
   info = get_workflow_info('shared/team_model.pkl')
   print(f"Model by: {info['custom_metadata']['author']}")
   ```

### Compression Benefits

| Model Type | Uncompressed | Compressed | Reduction |
|------------|--------------|------------|-----------|
| Small (KNN) | 50 KB | 15 KB | 70% |
| Medium (Random Forest) | 5 MB | 1.5 MB | 70% |
| Large (Deep ensemble) | 50 MB | 12 MB | 76% |

Enable compression for:
- Large models (> 1 MB)
- Storage constraints
- Network transfer
- Long-term archival

### Error Handling

```python
from app.ml_engine.utils.serialization import SerializationError

try:
    model = load_model('models/my_model.pkl')
except SerializationError as e:
    print(f"Failed to load model: {e}")
    # Handle error (use backup, retrain, etc.)
except FileNotFoundError:
    print("Model file not found")
    # Handle missing file
```

### Best Practices

1. **Use Compression for Production**: Reduces storage and transfer time
   ```python
   save_workflow(pipeline, model, 'prod.pkl', compression=True)
   ```

2. **Include Custom Metadata**: Document important information
   ```python
   save_model(model, 'model.pkl', custom_metadata={
       'training_date': datetime.now().isoformat(),
       'dataset_version': '1.0',
       'performance': metrics
   })
   ```

3. **Version Your Models**: Include version in filename or metadata
   ```python
   save_model(model, f'model_v{version}.pkl')
   ```

4. **Check Info Before Loading**: Verify model details
   ```python
   info = get_model_info('model.pkl')
   if info['n_features'] == expected_features:
       model = load_model('model.pkl')
   ```

5. **Use Workflows for Production**: Save complete pipelines
   ```python
   save_workflow(pipeline, model, 'prod_workflow.pkl')
   ```

### Performance

- **Save time**: < 100ms for small models, < 1s for large models
- **Load time**: < 50ms for small models, < 500ms for large models
- **Compression**: 70-80% size reduction with minimal time overhead
- **Memory**: Efficient - only loads necessary components

### Integration

- **Compatible with all model wrappers**: Classification, regression, clustering
- **Works with Pipeline class**: Preserves all fitted transformers
- **Integrates with training module**: Save TrainingResult models
- **Supports code generation**: Export serialized models to code
- **Production-ready**: Reliable persistence for deployment

### File Structure

```
models/
├── model_v1.pkl              # Uncompressed model
├── model_v2.pkl.gz           # Compressed model
├── pipeline.pkl              # Preprocessing pipeline
└── workflow_prod.pkl.gz      # Complete workflow

metadata/
├── model_v1_config.json      # Model configuration
├── model_v1_metadata.json    # Training metadata
└── ...
```

For detailed documentation, see `SERIALIZATION_GUIDE.md`.
