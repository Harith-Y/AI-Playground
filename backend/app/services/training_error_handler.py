"""
Training Error Handler Service

Provides centralized error handling, recovery strategies, and error reporting
for training operations.
"""

import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.training_exceptions import (
    TrainingException,
    categorize_exception,
    DataLoadError,
    PreprocessingError,
    ModelTrainingError,
    ModelSerializationError,
    MemoryError,
    TimeoutError
)
from app.utils.logger import get_logger
from app.models.model_run import ModelRun


logger = get_logger(__name__)


class TrainingErrorHandler:
    """
    Handles errors during training operations.
    
    Provides:
    - Error categorization
    - Error logging with context
    - Database error recording
    - Recovery suggestions
    - User-friendly error messages
    """
    
    def __init__(self, db: Session, model_run_id: str):
        """
        Initialize error handler.
        
        Args:
            db: Database session
            model_run_id: UUID of the model run
        """
        self.db = db
        self.model_run_id = model_run_id
        self.logger = get_logger(model_run_id=model_run_id)
    
    def handle_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[TrainingException, Dict[str, Any]]:
        """
        Handle a training error.
        
        This method:
        1. Categorizes the exception
        2. Logs the error with context
        3. Updates database with error details
        4. Generates recovery suggestions
        5. Creates user-friendly error message
        
        Args:
            exception: The exception that occurred
            context: Additional context about the error
        
        Returns:
            Tuple of (categorized_exception, error_report)
        """
        context = context or {}
        
        # 1. Categorize exception
        if isinstance(exception, TrainingException):
            training_exc = exception
        else:
            training_exc = categorize_exception(exception)
        
        # 2. Build error report
        error_report = self._build_error_report(training_exc, exception, context)
        
        # 3. Log error
        self._log_error(training_exc, error_report)
        
        # 4. Update database
        self._update_database(training_exc, error_report)
        
        # 5. Generate recovery suggestions
        error_report['recovery_suggestions'] = self._get_recovery_suggestions(training_exc)
        
        return training_exc, error_report
    
    def _build_error_report(
        self,
        training_exc: TrainingException,
        original_exc: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive error report."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'model_run_id': self.model_run_id,
            'error_type': type(original_exc).__name__,
            'error_code': training_exc.error_code,
            'error_message': training_exc.message,
            'user_message': self._get_user_friendly_message(training_exc),
            'recoverable': training_exc.recoverable,
            'details': training_exc.details,
            'context': context,
            'traceback': traceback.format_exc(),
            'phase': context.get('phase', 'unknown')
        }
    
    def _log_error(self, training_exc: TrainingException, error_report: Dict[str, Any]):
        """Log error with appropriate level and context."""
        log_extra = {
            'event': 'training_error',
            'error_code': training_exc.error_code,
            'recoverable': training_exc.recoverable,
            'phase': error_report['phase'],
            **training_exc.details
        }
        
        if training_exc.recoverable:
            self.logger.warning(
                f"Recoverable training error: {training_exc.message}",
                extra=log_extra
            )
        else:
            self.logger.error(
                f"Training failed: {training_exc.message}",
                extra=log_extra,
                exc_info=True
            )
    
    def _update_database(self, training_exc: TrainingException, error_report: Dict[str, Any]):
        """Update model run with error information."""
        try:
            from uuid import UUID
            from app.utils.cache import invalidate_model_cache, invalidate_comparison_cache
            import asyncio
            
            model_run = self.db.query(ModelRun).filter(
                ModelRun.id == UUID(self.model_run_id)
            ).first()
            
            if model_run:
                model_run.status = "failed"
                
                if not model_run.run_metadata:
                    model_run.run_metadata = {}
                
                model_run.run_metadata['error'] = {
                    'type': error_report['error_type'],
                    'code': error_report['error_code'],
                    'message': error_report['error_message'],
                    'user_message': error_report['user_message'],
                    'recoverable': error_report['recoverable'],
                    'details': error_report['details'],
                    'phase': error_report['phase'],
                    'timestamp': error_report['timestamp'],
                    'traceback': error_report['traceback'][:1000]  # Truncate for storage
                }
                
                # Mark as modified for JSONB field
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(model_run, 'run_metadata')
                
                self.db.commit()
                
                # Invalidate related caches
                try:
                    asyncio.create_task(invalidate_model_cache(self.model_run_id))
                    asyncio.create_task(invalidate_comparison_cache())
                except RuntimeError:
                    # If no event loop is running, run synchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(invalidate_model_cache(self.model_run_id))
                    loop.run_until_complete(invalidate_comparison_cache())
                    loop.close()
                
                self.logger.info(
                    f"Updated model run with error details",
                    extra={'event': 'error_recorded', 'model_run_id': self.model_run_id}
                )
        except Exception as e:
            self.logger.error(
                f"Failed to update database with error: {e}",
                extra={'event': 'error_update_failed'},
                exc_info=True
            )
            self.db.rollback()
    
    def _get_user_friendly_message(self, training_exc: TrainingException) -> str:
        """Generate user-friendly error message."""
        error_messages = {
            'DATA_LOAD_ERROR': "Unable to load the dataset. Please check that the file exists and is accessible.",
            'DATA_VALIDATION_ERROR': "The dataset has invalid or missing data. Please verify the data format and content.",
            'PREPROCESSING_ERROR': "An error occurred while preprocessing the data. You can try training without preprocessing.",
            'MODEL_INIT_ERROR': "Failed to initialize the model. Please check the model type and hyperparameters.",
            'MODEL_TRAINING_ERROR': "The model failed to train. Try adjusting hyperparameters or using a different model.",
            'MODEL_EVALUATION_ERROR': "Failed to evaluate the model. The training may have completed but metrics couldn't be calculated.",
            'MODEL_SERIALIZATION_ERROR': "Failed to save the trained model. The model was trained successfully but couldn't be saved.",
            'INSUFFICIENT_DATA_ERROR': "Not enough data to train the model. Please use a larger dataset.",
            'MEMORY_ERROR': "Out of memory during training. Try using a smaller dataset or simpler model.",
            'TIMEOUT_ERROR': "Training took too long and was stopped. Try using a simpler model or smaller dataset.",
            'CONFIGURATION_ERROR': "Invalid training configuration. Please check your settings and try again.",
            'RESOURCE_NOT_FOUND_ERROR': "A required resource was not found. Please verify all inputs exist.",
        }
        
        base_message = error_messages.get(
            training_exc.error_code,
            "An unexpected error occurred during training."
        )
        
        # Add specific details if available
        if training_exc.details:
            detail_parts = []
            if 'file_path' in training_exc.details:
                detail_parts.append(f"File: {training_exc.details['file_path']}")
            if 'step_type' in training_exc.details:
                detail_parts.append(f"Step: {training_exc.details['step_type']}")
            if 'model_type' in training_exc.details:
                detail_parts.append(f"Model: {training_exc.details['model_type']}")
            
            if detail_parts:
                base_message += f" ({', '.join(detail_parts)})"
        
        return base_message
    
    def _get_recovery_suggestions(self, training_exc: TrainingException) -> list:
        """Generate recovery suggestions based on error type."""
        suggestions = {
            'DATA_LOAD_ERROR': [
                "Verify the dataset file exists and is accessible",
                "Check file permissions",
                "Re-upload the dataset if necessary"
            ],
            'DATA_VALIDATION_ERROR': [
                "Check for missing or invalid values in the dataset",
                "Verify column names match the configuration",
                "Ensure data types are appropriate for the model"
            ],
            'PREPROCESSING_ERROR': [
                "Try training without preprocessing steps",
                "Review preprocessing configuration",
                "Check for incompatible preprocessing operations"
            ],
            'MODEL_INIT_ERROR': [
                "Verify the model type is correct",
                "Check hyperparameter values are valid",
                "Review model requirements and constraints"
            ],
            'MODEL_TRAINING_ERROR': [
                "Try different hyperparameters",
                "Use a simpler model",
                "Check for data quality issues",
                "Increase training time limit"
            ],
            'MODEL_EVALUATION_ERROR': [
                "Check test set size is sufficient",
                "Verify target column has valid values",
                "Review metric requirements"
            ],
            'MODEL_SERIALIZATION_ERROR': [
                "Check disk space availability",
                "Verify write permissions",
                "Try training again"
            ],
            'INSUFFICIENT_DATA_ERROR': [
                "Use a larger dataset",
                "Reduce test_size to allocate more data for training",
                "Consider data augmentation techniques"
            ],
            'MEMORY_ERROR': [
                "Use a smaller dataset",
                "Reduce model complexity",
                "Decrease batch size if applicable",
                "Close other applications to free memory"
            ],
            'TIMEOUT_ERROR': [
                "Use a simpler model",
                "Reduce dataset size",
                "Adjust hyperparameters for faster training",
                "Increase time limit if possible"
            ],
            'CONFIGURATION_ERROR': [
                "Review all configuration parameters",
                "Check for typos in column names",
                "Verify parameter types and ranges"
            ],
            'RESOURCE_NOT_FOUND_ERROR': [
                "Verify the experiment exists",
                "Check the dataset is available",
                "Ensure all referenced resources exist"
            ]
        }
        
        return suggestions.get(training_exc.error_code, [
            "Review the error message for details",
            "Check the training configuration",
            "Contact support if the issue persists"
        ])


def handle_training_error(
    db: Session,
    model_run_id: str,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to handle training errors.
    
    Args:
        db: Database session
        model_run_id: UUID of the model run
        exception: The exception that occurred
        context: Additional context
    
    Returns:
        Error report dictionary
    
    Example:
        >>> try:
        ...     # training code
        ... except Exception as e:
        ...     error_report = handle_training_error(db, run_id, e, {'phase': 'data_loading'})
    """
    handler = TrainingErrorHandler(db, model_run_id)
    _, error_report = handler.handle_error(exception, context)
    return error_report
