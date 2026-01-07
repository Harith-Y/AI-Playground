"""
Celery tasks for model training operations

This module contains asynchronous tasks for training machine learning models.
Long-running training operations are handled here to prevent API timeouts.
"""

import uuid
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from celery import Task
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split

from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.model_run import ModelRun
from app.models.experiment import Experiment
from app.models.dataset import Dataset
from app.models.preprocessing_step import PreprocessingStep
from app.ml_engine.model_registry import ModelRegistry
from app.ml_engine.preprocessing.serializer import PipelineSerializer
import pickle

def deserialize_transformer(binary_data):
    """
    Deserialize a fitted transformer from binary data.
    
    Args:
        binary_data: Pickled transformer object
        
    Returns:
        Deserialized transformer object
    """
    if binary_data is None:
        return None
    return pickle.loads(binary_data)

from app.ml_engine.evaluation.classification_metrics import calculate_classification_metrics
from app.ml_engine.evaluation.regression_metrics import calculate_regression_metrics
from app.ml_engine.evaluation.clustering_metrics import calculate_clustering_metrics
from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.cache import invalidate_model_cache, invalidate_comparison_cache
from app.services.training_error_handler import handle_training_error
from app.utils.error_recovery import retry, safe_execute, TransactionManager, CircuitBreaker
from app.utils.db_recovery import db_retry, ensure_connection
from app.utils.memory_manager import memory_profiler, get_memory_monitor, MemoryOptimizer
from app.core.training_exceptions import (
    DataLoadError,
    PreprocessingError,
    ModelTrainingError,
    ModelSerializationError
)


class TrainingTask(Task):
    """Base task for model training operations with progress tracking and logging"""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task completes successfully"""
        logger = get_logger(task_id=task_id)
        logger.info(
            f"Training task completed successfully",
            extra={
                'event': 'task_success',
                'duration_seconds': retval.get('training_time', 0),
                'model_type': retval.get('model_type', 'unknown'),
                'task_type': retval.get('task_type', 'unknown'),
            }
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger = get_logger(task_id=task_id)
        logger.error(
            f"Training task failed: {exc}",
            extra={
                'event': 'task_failure',
                'error_type': type(exc).__name__,
                'error_message': str(exc),
                'traceback': str(einfo)
            },
            exc_info=True
        )


# ============================================================================
# Progress Tracking Helper Functions
# ============================================================================

@db_retry(max_attempts=3, delay=0.5)
@ensure_connection
def initialize_progress_tracking(db: Session, model_run_id: str) -> bool:
    """
    Initialize progress tracking structure in ModelRun.run_metadata.
    
    Args:
        db: Database session
        model_run_id: UUID of the model run
        
    Returns:
        bool: True if initialization succeeded, False otherwise
    """
    try:
        model_run = db.query(ModelRun).filter(ModelRun.id == uuid.UUID(model_run_id)).first()
        if not model_run:
            return False
        
        # Initialize run_metadata if it doesn't exist
        if not model_run.run_metadata:
            model_run.run_metadata = {}
        
        # Set initial progress tracking fields
        from datetime import datetime
        now = datetime.utcnow()
        
        model_run.run_metadata['started_at'] = now.isoformat()
        model_run.run_metadata['current_progress'] = 0
        model_run.run_metadata['total_progress'] = 100
        model_run.run_metadata['current_phase'] = "Initializing"
        model_run.run_metadata['last_updated'] = now.isoformat()
        model_run.run_metadata['progress_history'] = []
        model_run.run_metadata['stalled'] = False
        
        # Mark as modified for JSONB field
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(model_run, 'run_metadata')
        
        db.commit()
        return True
        
    except Exception as e:
        db.rollback()
        logger = get_logger()
        logger.error(f"Failed to initialize progress tracking: {e}")
        return False


@db_retry(max_attempts=3, delay=0.5)
@ensure_connection
def update_training_progress(
    db: Session,
    model_run_id: str,
    progress_percentage: int,
    phase_description: str,
    max_retries: int = 3
) -> bool:
    """
    Update training progress in the database with retry logic.
    
    This function updates the ModelRun.run_metadata JSONB field with current
    progress information, appends to progress history, and implements exponential
    backoff retry logic for reliability.
    
    Args:
        db: Database session
        model_run_id: UUID of the model run
        progress_percentage: Progress from 0 to 100 (-1 for failure state)
        phase_description: Description of current phase
        max_retries: Maximum number of retry attempts (default: 3)
        
    Returns:
        bool: True if update succeeded, False otherwise
        
    Example:
        >>> update_training_progress(db, run_id, 50, "Training model...")
        True
    """
    logger = get_logger()
    
    for attempt in range(max_retries):
        try:
            # Fetch model run
            model_run = db.query(ModelRun).filter(ModelRun.id == uuid.UUID(model_run_id)).first()
            if not model_run:
                logger.error(f"ModelRun {model_run_id} not found")
                return False
            
            # Initialize run_metadata if needed
            if not model_run.run_metadata:
                model_run.run_metadata = {}
            
            # Update progress fields
            from datetime import datetime
            now = datetime.utcnow()
            
            model_run.run_metadata['current_progress'] = progress_percentage
            model_run.run_metadata['current_phase'] = phase_description
            model_run.run_metadata['last_updated'] = now.isoformat()
            
            # Initialize started_at on first update if not exists
            if 'started_at' not in model_run.run_metadata:
                model_run.run_metadata['started_at'] = now.isoformat()
            
            # Calculate elapsed time
            started_at = datetime.fromisoformat(model_run.run_metadata['started_at'])
            elapsed_seconds = (now - started_at).total_seconds()
            
            # Initialize progress_history if needed
            if 'progress_history' not in model_run.run_metadata:
                model_run.run_metadata['progress_history'] = []
            
            # Append to progress history
            progress_snapshot = {
                'timestamp': now.isoformat(),
                'percentage': progress_percentage,
                'phase': phase_description,
                'elapsed_seconds': elapsed_seconds
            }
            model_run.run_metadata['progress_history'].append(progress_snapshot)
            
            # Prune history to last 20 entries (keep first and last)
            history = model_run.run_metadata['progress_history']
            if len(history) > 20:
                # Keep first entry, last 19 entries
                model_run.run_metadata['progress_history'] = [
                    history[0],  # Keep first
                    *history[-19:]  # Keep last 19
                ]
            
            # Reset stalled flag on successful update
            model_run.run_metadata['stalled'] = False
            
            # Mark as modified (required for JSONB updates in SQLAlchemy)
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(model_run, 'run_metadata')
            
            # Commit transaction
            db.commit()
            
            logger.info(
                f"Progress updated: {model_run_id} -> {progress_percentage}% ({phase_description})",
                extra={
                    'event': 'progress_update',
                    'model_run_id': model_run_id,
                    'progress': progress_percentage,
                    'phase': phase_description,
                    'elapsed_seconds': elapsed_seconds
                }
            )
            return True
            
        except Exception as e:
            logger.warning(
                f"Progress update attempt {attempt + 1}/{max_retries} failed: {e}",
                extra={
                    'event': 'progress_update_retry',
                    'model_run_id': model_run_id,
                    'attempt': attempt + 1,
                    'error': str(e)
                }
            )
            db.rollback()
            
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                backoff_time = 2 ** attempt
                time.sleep(backoff_time)
            else:
                logger.error(
                    f"Failed to update progress after {max_retries} attempts",
                    extra={
                        'event': 'progress_update_failed',
                        'model_run_id': model_run_id,
                        'error': str(e)
                    }
                )
                return False
    
    return False


# ============================================================================
# Training Task
# ============================================================================


def run_training_logic(
    self,
    model_run_id: str,
    experiment_id: str,
    dataset_id: str,
    model_type: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    target_column: Optional[str] = None,
    feature_columns: Optional[list] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a machine learning model asynchronously.

    Args:
        self: Task instance (bound)
        model_run_id: UUID of the model run record
        experiment_id: UUID of the experiment
        dataset_id: UUID of the dataset to train on
        model_type: Type of model to train (e.g., 'random_forest_classifier')
        hyperparameters: Model hyperparameters (optional)
        target_column: Name of target column (required for supervised learning)
        feature_columns: List of feature column names (optional, uses all if None)
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        user_id: UUID of the user (for logging)

    Returns:
        Dictionary with training results and metrics
    """
    # Setup logging with context
    logger = get_logger(
        task_id=self.request.id,
        user_id=user_id,
        dataset_id=dataset_id
    )

    logger.info(
        f"Starting model training",
        extra={
            'event': 'training_start',
            'model_type': model_type,
            'experiment_id': experiment_id,
            'test_size': test_size,
        }
    )

    task_start_time = time.time()
    db: Session = SessionLocal()

    # Initialize memory monitoring for this training task
    memory_monitor = get_memory_monitor()
    memory_monitor.set_baseline()
    logger.info(f"Training task started - baseline memory: {memory_monitor.get_current_snapshot().rss_mb:.2f}MB")

    try:
        # Initialize progress tracking in database
        initialize_progress_tracking(db, model_run_id)

        # Update progress: Initializing (0%)
        update_training_progress(db, model_run_id, 0, "Initializing training...")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'status': 'Initializing training...'
            }
        )

        # 1. Fetch model run and validate
        model_run = db.query(ModelRun).filter(ModelRun.id == uuid.UUID(model_run_id)).first()
        if not model_run:
            raise ValueError(f"ModelRun with id {model_run_id} not found")

        experiment = db.query(Experiment).filter(Experiment.id == uuid.UUID(experiment_id)).first()
        if not experiment:
            raise ValueError(f"Experiment with id {experiment_id} not found")

        dataset = db.query(Dataset).filter(Dataset.id == uuid.UUID(dataset_id)).first()
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")

        # Update progress: Loading data (10%)
        update_training_progress(db, model_run_id, 10, "Loading dataset...")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 10,
                'total': 100,
                'status': 'Loading dataset...'
            }
        )

        # 2. Load dataset with memory profiling
        logger.info(f"Loading dataset from {dataset.file_path}")
        with memory_profiler("Data Loading") as load_monitor:
            try:
                df = pd.read_csv(dataset.file_path)
            except FileNotFoundError:
                raise DataLoadError(
                    message=f"Dataset file not found: {dataset.file_path}",
                    file_path=dataset.file_path
                )
            except pd.errors.EmptyDataError:
                raise DataLoadError(
                    message="Dataset file is empty",
                    file_path=dataset.file_path
                )
            except Exception as e:
                raise DataLoadError(
                    message=f"Failed to load dataset: {str(e)}",
                    file_path=dataset.file_path
                )

        initial_shape = df.shape
        logger.info(f"Dataset loaded: {initial_shape[0]} rows, {initial_shape[1]} columns")

        # Optimize DataFrame memory usage
        logger.info("Optimizing DataFrame memory usage...")
        df = MemoryOptimizer.optimize_dataframe_memory(df, aggressive=False)
        logger.info(f"DataFrame memory optimized")

        # Edge case validation before preprocessing (15%)
        update_training_progress(db, model_run_id, 15, "Validating dataset for edge cases...")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 15,
                'total': 100,
                'status': 'Validating dataset for edge cases...'
            }
        )

        from app.ml_engine.validation.edge_case_validator import validate_for_training
        from app.ml_engine.validation.edge_case_fixes import auto_fix_edge_cases

        # Determine categorical columns for encoding validation
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)

        # Validate dataset for edge cases
        is_valid, issues = validate_for_training(
            df=df,
            target_column=target_column,
            task_type=None,  # Will be determined from model type
            config={
                'test_size': test_size,
                'use_stratify': False,  # Will check later if needed
                'use_oversampling': False,  # Will check later if needed
                'encoding_columns': categorical_cols
            }
        )

        # Log validation issues
        if issues:
            logger.warning(f"Found {len(issues)} edge case issues in dataset")
            for issue in issues:
                logger.warning(
                    f"[{issue.severity.value.upper()}] {issue.category}: {issue.message}",
                    extra={
                        'event': 'edge_case_detected',
                        'severity': issue.severity.value,
                        'category': issue.category,
                        'details': issue.details
                    }
                )

        # Apply auto-fixes for critical issues
        if not is_valid:
            logger.info("Attempting to auto-fix critical edge case issues...")
            df_fixed, fixes = auto_fix_edge_cases(
                df=df,
                issues=issues,
                target_column=target_column,
                encoding_columns=categorical_cols
            )

            if fixes:
                logger.info(f"Applied {len(fixes)} auto-fixes:")
                for fix in fixes:
                    logger.info(f"  - {fix.description}")
                df = df_fixed

                # Re-validate after fixes
                is_valid_after_fix, issues_after_fix = validate_for_training(
                    df=df,
                    target_column=target_column,
                    config={
                        'test_size': test_size,
                        'encoding_columns': categorical_cols
                    }
                )

                # If still not valid after auto-fix, raise error
                if not is_valid_after_fix:
                    critical_issues = [i for i in issues_after_fix if i.severity.value in ['error', 'critical']]
                    if critical_issues:
                        error_msg = "Dataset has critical issues that could not be auto-fixed:\n"
                        for issue in critical_issues:
                            error_msg += f"  - [{issue.severity.value.upper()}] {issue.message}\n"
                            error_msg += f"    Recommendation: {issue.recommendation}\n"
                        raise InsufficientDataError(
                            message=error_msg,
                            n_samples=len(df),
                            min_required=20
                        )
            else:
                # No auto-fixes available, but issues still exist
                critical_issues = [i for i in issues if i.severity.value in ['error', 'critical']]
                if critical_issues:
                    error_msg = "Dataset has critical issues:\n"
                    for issue in critical_issues:
                        error_msg += f"  - [{issue.severity.value.upper()}] {issue.message}\n"
                        error_msg += f"    Recommendation: {issue.recommendation}\n"
                    raise InsufficientDataError(
                        message=error_msg,
                        n_samples=len(df),
                        min_required=20
                    )

        logger.info("Edge case validation passed")

        # Update progress: Applying preprocessing (20%)
        update_training_progress(db, model_run_id, 20, "Applying preprocessing steps...")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Applying preprocessing steps...'
            }
        )

        # 3. Apply preprocessing steps if any exist
        preprocessing_steps = db.query(PreprocessingStep).filter(
            PreprocessingStep.dataset_id == uuid.UUID(dataset_id),
            PreprocessingStep.is_active == True
        ).order_by(PreprocessingStep.order).all()

        if preprocessing_steps:
            logger.info(f"Applying {len(preprocessing_steps)} preprocessing steps")
            for idx, step in enumerate(preprocessing_steps):
                try:
                    if step.fitted_transformer:
                        transformer = deserialize_transformer(step.fitted_transformer)
                        df = pd.DataFrame(
                            transformer.transform(df),
                            columns=df.columns
                        )
                        logger.info(f"Applied step {idx + 1}/{len(preprocessing_steps)}: {step.step_type}")
                except Exception as e:
                    # Log warning but continue - preprocessing errors are not fatal
                    logger.warning(
                        f"Failed to apply preprocessing step {step.step_type}: {e}",
                        extra={
                            'event': 'preprocessing_step_failed',
                            'step_type': step.step_type,
                            'step_order': step.order
                        }
                    )
                    # Optionally raise if preprocessing is critical
                    # raise PreprocessingError(
                    #     message=f"Failed to apply preprocessing step: {str(e)}",
                    #     step_type=step.step_type,
                    #     step_order=step.order
                    # )

        # Update progress: Preparing features (35%)
        update_training_progress(db, model_run_id, 35, "Preparing features and target...")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 35,
                'total': 100,
                'status': 'Preparing features and target...'
            }
        )

        # 4. Get model info from registry
        registry = ModelRegistry()
        model_info = registry.get_model(model_type)
        if not model_info:
            raise ValueError(f"Model type '{model_type}' not found in registry")

        task_type = model_info.task_type
        logger.info(f"Training {task_type.value} model: {model_info.name}")

        # 5. Prepare features and target
        if task_type.value in ['classification', 'regression']:
            # Supervised learning
            if not target_column:
                raise ValueError("target_column is required for supervised learning")

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            # Separate features and target
            if feature_columns:
                # Use specified feature columns
                missing_cols = set(feature_columns) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Feature columns not found: {missing_cols}")
                X = df[feature_columns]
            else:
                # Use all columns except target
                X = df.drop(columns=[target_column])

            y = df[target_column]

            # Smart stratification: only stratify if feasible
            use_stratify = False
            if task_type.value == 'classification':
                n_classes = len(y.unique())
                min_class_size = y.value_counts().min()
                
                # Only stratify if:
                # 1. Not too many classes (< 50)
                # 2. Each class has at least 2 samples (to allow splitting)
                # 3. Test set will have at least 1 sample per class
                min_test_samples = int(len(df) * test_size)
                
                if n_classes < 50 and min_class_size >= 2 and min_test_samples >= n_classes:
                    use_stratify = True
                    logger.info(f"Using stratified split ({n_classes} classes, min class size: {min_class_size})")
                else:
                    logger.warning(
                        f"Skipping stratification: n_classes={n_classes}, "
                        f"min_class_size={min_class_size}, test_samples={min_test_samples}"
                    )

            # Split into train and test sets
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y if use_stratify else None
                )
            except ValueError as e:
                # If stratified split fails, try without stratification
                if use_stratify:
                    logger.warning(f"Stratified split failed: {e}. Retrying without stratification...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=None
                    )
                else:
                    raise

            logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

            # Validate train/test split results for classification
            if task_type.value == 'classification':
                # Check class distribution in train set
                train_class_counts = y_train.value_counts()
                test_class_counts = y_test.value_counts()
                
                # Warn if some classes missing in test set
                missing_in_test = set(train_class_counts.index) - set(test_class_counts.index)
                if missing_in_test:
                    logger.warning(
                        f"Classes {missing_in_test} are not present in test set. "
                        "Evaluation metrics for these classes will be unavailable."
                    )
                
                # Warn if extreme imbalance exists
                if len(train_class_counts) > 1:
                    max_count = train_class_counts.max()
                    min_count = train_class_counts.min()
                    imbalance_ratio = max_count / min_count
                    
                    if imbalance_ratio > 100:
                        logger.warning(
                            f"Extreme class imbalance detected: {imbalance_ratio:.1f}:1 ratio. "
                            "Consider using class weights or resampling techniques."
                        )
                    elif imbalance_ratio > 10:
                        logger.info(
                            f"Class imbalance detected: {imbalance_ratio:.1f}:1 ratio. "
                            "Model performance on minority classes may be limited."
                        )

        else:
            # Unsupervised learning (clustering)
            if feature_columns:
                X = df[feature_columns]
            else:
                X = df

            X_train = X
            X_test = None
            y_train = None
            y_test = None

        # Check memory before training
        memory_snapshot = memory_monitor.get_current_snapshot()
        if memory_snapshot.percent > 85.0:
            logger.warning(f"High memory usage before training: {memory_snapshot.percent:.1f}%")
            logger.info("Running garbage collection to free memory...")
            MemoryOptimizer.aggressive_gc()

        # Update progress: Training model (50%)
        update_training_progress(db, model_run_id, 50, f"Training {model_info.name}...")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 50,
                'total': 100,
                'status': f'Training {model_info.name}...'
            }
        )

        # 6. Create and configure model
        from app.ml_engine.models.classification import ClassificationModel
        from app.ml_engine.models.regression import RegressionModel
        from app.ml_engine.models.clustering import ClusteringModel

        # Create model instance based on task type
        if task_type.value == 'classification':
            model = ClassificationModel(
                model_type=model_type,
                config=hyperparameters or {}
            )
        elif task_type.value == 'regression':
            model = RegressionModel(
                model_type=model_type,
                config=hyperparameters or {}
            )
        else:  # clustering
            model = ClusteringModel(
                model_type=model_type,
                config=hyperparameters or {}
            )

        # 7. Train the model with memory profiling
        training_start = time.time()

        with memory_profiler(f"Model Training - {model_info.name}"):
            try:
                if task_type.value in ['classification', 'regression']:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train)
            except Exception as e:
                raise ModelTrainingError(
                    message=f"Model training failed: {str(e)}",
                    model_type=model_type
                )

        training_duration = time.time() - training_start
        logger.info(f"Model training completed in {training_duration:.2f} seconds")

        # Update progress: Evaluating model (75%)
        update_training_progress(db, model_run_id, 75, "Evaluating model performance...")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 75,
                'total': 100,
                'status': 'Evaluating model performance...'
            }
        )

        # 8. Evaluate model and calculate metrics
        metrics = {}

        if task_type.value == 'classification':
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            metrics = calculate_classification_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                labels=sorted(y_test.unique())
            )

        elif task_type.value == 'regression':
            y_pred = model.predict(X_test)

            metrics = calculate_regression_metrics(
                y_true=y_test,
                y_pred=y_pred
            )

        else:  # clustering
            labels = model.get_labels()

            metrics = calculate_clustering_metrics(
                X=X_train,
                labels=labels,
                n_clusters=len(np.unique(labels))
            )

        # 9. Get feature importance if available
        feature_importance = None
        try:
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                logger.info(f"Feature importance calculated: {len(feature_importance)} features")
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")

        # Update progress: Saving model (90%)
        update_training_progress(db, model_run_id, 90, "Saving model artifact...")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 90,
                'total': 100,
                'status': 'Saving model artifact...'
            }
        )

        # 10. Save model artifact using ModelSerializationService
        from app.services.storage_service import get_model_serialization_service
        
        serialization_service = get_model_serialization_service()
        
        # Prepare additional metadata to save with model
        save_metadata = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'train_samples': len(X_train),
            'test_samples': len(X_test) if X_test is not None else 0,
            'n_features': X_train.shape[1],
            'target_column': target_column,
            'feature_columns': feature_columns or list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None,
            'test_size': test_size,
            'random_state': random_state,
            'task_type': task_type.value,
            'preprocessing_steps_applied': len(preprocessing_steps)
        }
        
        try:
            model_path = serialization_service.save_model(
                model=model,
                model_run_id=model_run_id,
                experiment_id=experiment_id,
                additional_metadata=save_metadata,
                save_config=True,
                save_metadata=True
            )
        except Exception as e:
            raise ModelSerializationError(
                message=f"Failed to save model: {str(e)}",
                model_path=None
            )
        
        logger.info(
            f"Model serialized successfully",
            extra={
                'event': 'model_serialization_complete',
                'model_path': model_path,
                'model_type': model_type,
                'has_metadata': True
            }
        )

        # 11. Update ModelRun with results
        model_run.metrics = metrics
        model_run.training_time = training_duration
        model_run.model_artifact_path = str(model_path)
        model_run.status = "completed"

        # Store feature importance if available
        if feature_importance is not None:
            if not model_run.run_metadata:
                model_run.run_metadata = {}
            model_run.run_metadata['feature_importance'] = feature_importance

        db.commit()
        db.refresh(model_run)

        logger.info(f"ModelRun updated with results")
        
        # Invalidate related caches
        import asyncio
        try:
            asyncio.create_task(invalidate_model_cache(model_run_id))
            asyncio.create_task(invalidate_comparison_cache())
        except RuntimeError:
            # If no event loop is running, run synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(invalidate_model_cache(model_run_id))
            loop.run_until_complete(invalidate_comparison_cache())
            loop.close()

        # 12. Update experiment status if needed
        if experiment.status == "running":
            # Check if all model runs in this experiment are completed
            all_runs = db.query(ModelRun).filter(ModelRun.experiment_id == uuid.UUID(experiment_id)).all()
            if all(run.status == "completed" for run in all_runs):
                experiment.status = "completed"
                db.commit()

        # Calculate total execution time
        total_execution_time = time.time() - task_start_time

        # Log final memory usage
        final_memory = memory_monitor.get_current_snapshot()
        memory_delta = memory_monitor.get_memory_delta()
        logger.info(
            f"Training completed - Final memory: {final_memory.rss_mb:.2f}MB, "
            f"Delta: {memory_delta:+.2f}MB, Peak: {memory_monitor.get_peak_memory():.2f}MB"
        )

        # Update progress: Complete (100%)
        update_training_progress(db, model_run_id, 100, "Training completed successfully!")
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Training completed successfully!'
            }
        )

        result = {
            'model_run_id': model_run_id,
            'model_type': model_type,
            'task_type': task_type.value,
            'training_time': training_duration,
            'total_execution_time': total_execution_time,
            'metrics': metrics,
            'train_samples': len(X_train),
            'test_samples': len(X_test) if X_test is not None else 0,
            'n_features': X_train.shape[1],
            'feature_importance': feature_importance,
            'model_path': str(model_path),
            'status': 'completed'
        }

        logger.info(
            f"Training completed successfully",
            extra={
                'event': 'training_complete',
                'metrics': metrics,
                'duration': total_execution_time
            }
        )

        return result

    except Exception as e:
        # Use centralized error handler
        error_report = handle_training_error(
            db=db,
            model_run_id=model_run_id,
            exception=e,
            context={
                'phase': 'training',
                'model_type': model_type,
                'experiment_id': experiment_id,
                'dataset_id': dataset_id,
                'user_id': user_id,
                'task_id': self.request.id
            }
        )
        
        # Update progress to indicate failure
        try:
            update_training_progress(
                db,
                model_run_id,
                -1,
                f"Training failed: {error_report['user_message']}"
            )
        except Exception as progress_error:
            logger.error(f"Failed to update progress after error: {progress_error}")
        
        # Re-raise the exception for Celery to handle
        raise

    finally:
        db.close()


@celery_app.task(
    base=TrainingTask,
    bind=True,
    name="app.tasks.training_tasks.train_model"
)
def train_model(
    self,
    model_run_id: str,
    experiment_id: str,
    dataset_id: str,
    model_type: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    target_column: Optional[str] = None,
    feature_columns: Optional[list] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Celery task wrapper for model training. 
    Delegates to run_training_logic to allow reuse in BackgroundTasks.
    """
    return run_training_logic(
        self,
        model_run_id=model_run_id,
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        model_type=model_type,
        hyperparameters=hyperparameters,
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=test_size,
        random_state=random_state,
        user_id=user_id
    )
