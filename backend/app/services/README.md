# Training Configuration Validation

## Overview

The `training_validation_service.py` module provides comprehensive validation for model training configurations before training is initiated. This ensures all prerequisites are met and prevents training failures with clear error messages.

## Features

### 1. Experiment Validation
- Verifies experiment exists in database
- Confirms user ownership
- Returns detailed error if not found or unauthorized

### 2. Dataset Validation
- Verifies dataset exists in database
- Confirms user ownership
- Checks that dataset file exists on disk
- Returns detailed error if not found or unauthorized

### 3. Model Type Validation
- Verifies model type exists in model registry
- Provides list of available models if invalid type specified
- Returns model information for further validation

### 4. Target Column Validation
- For supervised learning (classification/regression):
  - Ensures target column is specified
  - Verifies target column exists in dataset
  - Provides list of available columns if not found
- For unsupervised learning (clustering):
  - Warns if target column specified (will be ignored)

### 5. Feature Columns Validation
- Verifies all specified feature columns exist in dataset
- Ensures target column is not included in features
- Checks for duplicate feature columns
- Provides list of available columns if validation fails

### 6. Test Size Validation
- Ensures test_size is between 0.0 and 1.0
- Warns if test_size is too small (< 0.1) or too large (> 0.5)

### 7. Hyperparameters Validation
- Validates parameter types (int, float, string, etc.)
- Checks parameter ranges (min/max values)
- Warns about unexpected parameters
- Ensures parameters match model's expected configuration

### 8. Dataset Size Validation
- Ensures dataset has minimum number of samples (10+)
- Verifies test set will have at least 2 samples
- Verifies train set will have at least 5 samples
- Provides recommendations if dataset too small

## Usage

### In API Endpoints

```python
from app.services.training_validation_service import get_training_validator, ValidationError

@router.post("/train")
async def train_model_endpoint(request: ModelTrainingRequest, db: Session):
    try:
        validator = get_training_validator(db)
        experiment, dataset, model_info = validator.validate_training_config(
            experiment_id=request.experiment_id,
            dataset_id=request.dataset_id,
            model_type=request.model_type,
            user_id=user_id,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            test_size=request.test_size,
            hyperparameters=request.hyperparameters
        )
        
        # Validation passed, proceed with training
        # ...
        
    except ValidationError as e:
        # Handle validation error
        raise HTTPException(status_code=400, detail=e.message)
```

### Direct Usage

```python
from app.services.training_validation_service import TrainingConfigValidator

validator = TrainingConfigValidator(db)

try:
    exp, ds, model_info = validator.validate_training_config(
        experiment_id=exp_id,
        dataset_id=ds_id,
        model_type="random_forest_classifier",
        user_id=user_id,
        target_column="species",
        feature_columns=["sepal_length", "sepal_width"],
        test_size=0.2
    )
    print("Validation passed!")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Field: {e.field}")
```

## Error Handling

The service raises `ValidationError` exceptions with:
- `message`: Human-readable error description
- `field`: Name of the field that failed validation (optional)

This allows API endpoints to return appropriate HTTP status codes:
- 404 Not Found: Resource doesn't exist
- 403 Forbidden: User doesn't have permission
- 400 Bad Request: Invalid configuration

## Integration

The validation service is integrated into the `/api/v1/models/train` endpoint in `backend/app/api/v1/endpoints/models.py`. All training requests are validated before creating a ModelRun record or triggering the Celery task.

## Benefits

1. **Early Error Detection**: Catches configuration errors before expensive training operations
2. **Clear Error Messages**: Provides specific, actionable error messages
3. **Security**: Ensures users can only train on their own datasets and experiments
4. **Data Integrity**: Validates column names and data compatibility
5. **Resource Efficiency**: Prevents wasted compute on invalid configurations
6. **Better UX**: Users get immediate feedback on configuration issues

## Testing

The validation service should be tested with:
- Valid configurations (should pass)
- Missing resources (should fail with 404)
- Unauthorized access (should fail with 403)
- Invalid model types (should fail with 400)
- Missing target columns (should fail with 400)
- Invalid feature columns (should fail with 400)
- Invalid hyperparameters (should fail with 400)
- Insufficient dataset size (should fail with 400)

## Future Enhancements

Potential improvements:
- Validate data types in columns (numeric vs categorical)
- Check for missing values and suggest preprocessing
- Validate class balance for classification tasks
- Suggest appropriate test_size based on dataset size
- Cache validation results to avoid repeated checks
- Add async validation for large datasets
