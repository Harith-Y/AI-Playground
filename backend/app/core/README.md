# Training Error Handling System

## Overview

Comprehensive error handling system for model training operations that provides:
- Specific exception types for different failure scenarios
- Centralized error handling and logging
- User-friendly error messages
- Recovery suggestions
- Detailed error reporting in database

## Architecture

### 1. Custom Exception Hierarchy

**File:** `backend/app/core/training_exceptions.py`

Base exception with specialized subclasses:

```
TrainingException (base)
├── DataLoadError
├── DataValidationError
├── PreprocessingError
├── ModelInitializationError
├── ModelTrainingError
├── ModelEvaluationError
├── ModelSerializationError
├── InsufficientDataError
├── MemoryError
├── TimeoutError
├── ConfigurationError
└── ResourceNotFoundError
```

Each exception includes:
- `message`: Human-readable error description
- `error_code`: Machine-readable code (e.g., 'DATA_LOAD_ERROR')
- `recoverable`: Whether error can be recovered from
- `details`: Additional context (file paths, model types, etc.)

### 2. Error Handler Service

**File:** `backend/app/services/training_error_handler.py`

Centralized error handling that:
1. Categorizes exceptions into specific types
2. Logs errors with appropriate context
3. Updates database with error details
4. Generates user-friendly messages
5. Provides recovery suggestions

### 3. Integration with Training Tasks

**File:** `backend/app/tasks/training_tasks.py`

Training task wraps critical sections with specific error handling:
- Data loading → `DataLoadError`
- Preprocessing → `PreprocessingError`
- Model training → `ModelTrainingError`
- Model serialization → `ModelSerializationError`

## Error Codes

| Code | Description | Recoverable | Common Causes |
|------|-------------|-------------|---------------|
| DATA_LOAD_ERROR | Failed to load dataset | No | File not found, corrupted file, permission denied |
| DATA_VALIDATION_ERROR | Invalid data format/content | No | Missing columns, wrong data types, empty dataset |
| PREPROCESSING_ERROR | Preprocessing step failed | Yes | Incompatible transformations, missing values |
| MODEL_INIT_ERROR | Model initialization failed | No | Invalid hyperparameters, unsupported model type |
| MODEL_TRAINING_ERROR | Training failed | Yes | Convergence issues, invalid data, memory issues |
| MODEL_EVALUATION_ERROR | Metrics calculation failed | No | Insufficient test data, invalid predictions |
| MODEL_SERIALIZATION_ERROR | Failed to save model | Yes | Disk full, permission denied, path issues |
| INSUFFICIENT_DATA_ERROR | Not enough data | No | Dataset too small, test_size too large |
| MEMORY_ERROR | Out of memory | Yes | Dataset too large, model too complex |
| TIMEOUT_ERROR | Training exceeded time limit | Yes | Model too complex, dataset too large |
| CONFIGURATION_ERROR | Invalid configuration | No | Wrong parameter types, invalid values |
| RESOURCE_NOT_FOUND_ERROR | Required resource missing | No | Experiment/dataset deleted, invalid IDs |

## Usage

### In Training Tasks

```python
from app.core.training_exceptions import DataLoadError, ModelTrainingError
from app.services.training_error_handler import handle_training_error

# Specific error handling
try:
    df = pd.read_csv(dataset.file_path)
except FileNotFoundError:
    raise DataLoadError(
        message=f"Dataset file not found: {dataset.file_path}",
        file_path=dataset.file_path
    )

# Generic error handling with categorization
try:
    # training code
except Exception as e:
    error_report = handle_training_error(
        db=db,
        model_run_id=model_run_id,
        exception=e,
        context={'phase': 'training', 'model_type': model_type}
    )
    raise
```

### Error Report Structure

When an error occurs, the following information is stored in `ModelRun.run_metadata['error']`:

```json
{
  "type": "ValueError",
  "code": "DATA_VALIDATION_ERROR",
  "message": "Target column 'species' not found in dataset",
  "user_message": "The dataset has invalid or missing data. Please verify the data format and content.",
  "recoverable": false,
  "details": {
    "validation_type": "columns"
  },
  "phase": "data_preparation",
  "timestamp": "2025-12-29T10:00:00Z",
  "traceback": "Traceback (most recent call last)..."
}
```

### API Response

When checking training status for a failed run:

```json
{
  "model_run_id": "abc-123",
  "status": "FAILURE",
  "error": {
    "type": "ValueError",
    "code": "DATA_VALIDATION_ERROR",
    "message": "Target column 'species' not found in dataset",
    "user_message": "The dataset has invalid or missing data...",
    "recoverable": false,
    "phase": "data_preparation"
  }
}
```

## Recovery Suggestions

Each error type includes actionable recovery suggestions:

### DataLoadError
- Verify the dataset file exists and is accessible
- Check file permissions
- Re-upload the dataset if necessary

### DataValidationError
- Check for missing or invalid values in the dataset
- Verify column names match the configuration
- Ensure data types are appropriate for the model

### PreprocessingError
- Try training without preprocessing steps
- Review preprocessing configuration
- Check for incompatible preprocessing operations

### ModelTrainingError
- Try different hyperparameters
- Use a simpler model
- Check for data quality issues
- Increase training time limit

### ModelSerializationError
- Check disk space availability
- Verify write permissions
- Try training again

### InsufficientDataError
- Use a larger dataset
- Reduce test_size to allocate more data for training
- Consider data augmentation techniques

### MemoryError
- Use a smaller dataset
- Reduce model complexity
- Decrease batch size if applicable
- Close other applications to free memory

### TimeoutError
- Use a simpler model
- Reduce dataset size
- Adjust hyperparameters for faster training
- Increase time limit if possible

## Error Logging

All errors are logged with structured logging:

```python
logger.error(
    f"Training failed: {training_exc.message}",
    extra={
        'event': 'training_error',
        'error_code': training_exc.error_code,
        'recoverable': training_exc.recoverable,
        'phase': 'training',
        'model_type': 'random_forest_classifier'
    },
    exc_info=True
)
```

Log levels:
- **WARNING**: Recoverable errors (preprocessing failures, etc.)
- **ERROR**: Non-recoverable errors (data load failures, etc.)

## Database Storage

Error information is stored in `ModelRun.run_metadata['error']` as JSONB:

```python
model_run.run_metadata['error'] = {
    'type': 'ValueError',
    'code': 'DATA_VALIDATION_ERROR',
    'message': 'Target column not found',
    'user_message': 'The dataset has invalid or missing data...',
    'recoverable': False,
    'details': {'validation_type': 'columns'},
    'phase': 'data_preparation',
    'timestamp': '2025-12-29T10:00:00Z',
    'traceback': 'Traceback...'[:1000]  # Truncated
}
```

## Testing

Error handling is tested in `backend/tests/test_model_training_api.py`:

```python
def test_get_status_failed(client, db, test_experiment):
    """Test getting status of failed model run."""
    model_run = ModelRun(
        status="failed",
        run_metadata={
            "error": {
                "type": "ValueError",
                "code": "DATA_VALIDATION_ERROR",
                "message": "Invalid configuration"
            }
        }
    )
    # ... test assertions
```

## Best Practices

1. **Use Specific Exceptions**: Raise specific exception types rather than generic `Exception`
2. **Provide Context**: Include relevant details (file paths, model types, etc.)
3. **User-Friendly Messages**: Generate clear, actionable error messages
4. **Log Appropriately**: Use correct log levels (WARNING for recoverable, ERROR for fatal)
5. **Store Details**: Save comprehensive error information in database
6. **Recovery Suggestions**: Provide actionable steps to resolve the issue

## Future Enhancements

Potential improvements:
- Automatic retry for recoverable errors
- Error rate monitoring and alerting
- Error pattern analysis
- Suggested hyperparameter adjustments
- Integration with monitoring systems (Sentry, etc.)
- Error notification webhooks
- Error analytics dashboard
