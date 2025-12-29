"""
Custom exceptions for training operations.

This module defines specific exception types for different training failure scenarios,
enabling better error handling, recovery, and user feedback.
"""

from typing import Optional, Dict, Any


class TrainingException(Exception):
    """Base exception for all training-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        recoverable: bool = False,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize training exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            recoverable: Whether the error can be recovered from
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for storage/logging."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'recoverable': self.recoverable,
            'details': self.details
        }


class DataLoadError(TrainingException):
    """Error loading or reading dataset."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='DATA_LOAD_ERROR',
            recoverable=False,
            details={'file_path': file_path}
        )


class DataValidationError(TrainingException):
    """Error validating dataset structure or content."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='DATA_VALIDATION_ERROR',
            recoverable=False,
            details={'validation_type': validation_type}
        )


class PreprocessingError(TrainingException):
    """Error during preprocessing step application."""
    
    def __init__(self, message: str, step_type: Optional[str] = None, step_order: Optional[int] = None):
        super().__init__(
            message=message,
            error_code='PREPROCESSING_ERROR',
            recoverable=True,  # Can retry without preprocessing
            details={'step_type': step_type, 'step_order': step_order}
        )


class ModelInitializationError(TrainingException):
    """Error initializing or configuring model."""
    
    def __init__(self, message: str, model_type: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='MODEL_INIT_ERROR',
            recoverable=False,
            details={'model_type': model_type}
        )


class ModelTrainingError(TrainingException):
    """Error during model training/fitting."""
    
    def __init__(self, message: str, model_type: Optional[str] = None, epoch: Optional[int] = None):
        super().__init__(
            message=message,
            error_code='MODEL_TRAINING_ERROR',
            recoverable=True,  # Can retry with different hyperparameters
            details={'model_type': model_type, 'epoch': epoch}
        )


class ModelEvaluationError(TrainingException):
    """Error during model evaluation/metrics calculation."""
    
    def __init__(self, message: str, metric_name: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='MODEL_EVALUATION_ERROR',
            recoverable=False,
            details={'metric_name': metric_name}
        )


class ModelSerializationError(TrainingException):
    """Error saving model artifact."""
    
    def __init__(self, message: str, model_path: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='MODEL_SERIALIZATION_ERROR',
            recoverable=True,  # Can retry save operation
            details={'model_path': model_path}
        )


class InsufficientDataError(TrainingException):
    """Insufficient data for training."""
    
    def __init__(self, message: str, n_samples: Optional[int] = None, min_required: Optional[int] = None):
        super().__init__(
            message=message,
            error_code='INSUFFICIENT_DATA_ERROR',
            recoverable=False,
            details={'n_samples': n_samples, 'min_required': min_required}
        )


class MemoryError(TrainingException):
    """Out of memory during training."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='MEMORY_ERROR',
            recoverable=True,  # Can retry with smaller batch size
            details={'operation': operation}
        )


class TimeoutError(TrainingException):
    """Training exceeded time limit."""
    
    def __init__(self, message: str, elapsed_time: Optional[float] = None, time_limit: Optional[float] = None):
        super().__init__(
            message=message,
            error_code='TIMEOUT_ERROR',
            recoverable=True,  # Can retry with simpler model
            details={'elapsed_time': elapsed_time, 'time_limit': time_limit}
        )


class ConfigurationError(TrainingException):
    """Invalid training configuration."""
    
    def __init__(self, message: str, config_field: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='CONFIGURATION_ERROR',
            recoverable=False,
            details={'config_field': config_field}
        )


class ResourceNotFoundError(TrainingException):
    """Required resource not found."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None):
        super().__init__(
            message=message,
            error_code='RESOURCE_NOT_FOUND_ERROR',
            recoverable=False,
            details={'resource_type': resource_type, 'resource_id': resource_id}
        )


def categorize_exception(exc: Exception) -> TrainingException:
    """
    Categorize a generic exception into a specific TrainingException.
    
    Args:
        exc: The exception to categorize
    
    Returns:
        Appropriate TrainingException subclass
    
    Example:
        >>> try:
        ...     pd.read_csv('missing.csv')
        ... except Exception as e:
        ...     training_exc = categorize_exception(e)
        ...     print(training_exc.error_code)
        'DATA_LOAD_ERROR'
    """
    exc_type = type(exc).__name__
    exc_message = str(exc)
    
    # File/IO errors
    if exc_type in ['FileNotFoundError', 'IOError', 'OSError']:
        return DataLoadError(
            message=f"Failed to load data: {exc_message}",
            file_path=getattr(exc, 'filename', None)
        )
    
    # Pandas errors
    if exc_type in ['EmptyDataError', 'ParserError']:
        return DataValidationError(
            message=f"Invalid data format: {exc_message}",
            validation_type='format'
        )
    
    # Memory errors
    if exc_type in ['MemoryError', 'OutOfMemoryError']:
        return MemoryError(
            message=f"Out of memory: {exc_message}",
            operation='training'
        )
    
    # Value errors (often configuration issues)
    if exc_type == 'ValueError':
        if 'column' in exc_message.lower() or 'feature' in exc_message.lower():
            return DataValidationError(
                message=exc_message,
                validation_type='columns'
            )
        elif 'sample' in exc_message.lower() or 'empty' in exc_message.lower():
            return InsufficientDataError(
                message=exc_message
            )
        else:
            return ConfigurationError(
                message=exc_message
            )
    
    # Key errors (missing columns/keys)
    if exc_type == 'KeyError':
        return DataValidationError(
            message=f"Missing required column: {exc_message}",
            validation_type='missing_column'
        )
    
    # Type errors (wrong data types)
    if exc_type == 'TypeError':
        return ConfigurationError(
            message=f"Invalid data type: {exc_message}"
        )
    
    # Sklearn/ML errors
    if 'sklearn' in exc_type.lower() or 'estimator' in exc_message.lower():
        return ModelTrainingError(
            message=exc_message
        )
    
    # Timeout
    if exc_type in ['TimeoutError', 'TimeoutException']:
        return TimeoutError(
            message=exc_message
        )
    
    # Default: generic training error
    return TrainingException(
        message=f"Training failed: {exc_message}",
        error_code='UNKNOWN_ERROR',
        recoverable=False,
        details={'original_type': exc_type}
    )
