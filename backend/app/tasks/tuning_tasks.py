"""
Celery tasks for hyperparameter tuning operations

This module contains asynchronous tasks for hyperparameter optimization.
Long-running tuning operations are handled here to prevent API timeouts.
"""

import uuid
import time
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from celery import Task
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.model_run import ModelRun
from app.models.tuning_run import TuningRun, TuningStatus
from app.models.dataset import Dataset
from app.models.preprocessing_step import PreprocessingStep
from app.ml_engine.model_registry import ModelRegistry
from app.ml_engine.tuning import (
    get_default_search_space,
    run_grid_search,
    run_random_search,
    run_bayesian_search,
    run_cross_validation,
)
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
from app.services.storage_service import get_model_serialization_service
from app.utils.logger import get_logger


class TuningTask(Task):
    """Base task for hyperparameter tuning operations"""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task completes successfully"""
        logger = get_logger(task_id=task_id)
        logger.info(
            f"Tuning task completed successfully",
            extra={
                'event': 'tuning_success',
                'duration_seconds': retval.get('tuning_time', 0),
                'best_score': retval.get('best_score', 0),
            }
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger = get_logger(task_id=task_id)
        logger.error(
            f"Tuning task failed: {exc}",
            extra={
                'event': 'tuning_failure',
                'error_type': type(exc).__name__,
                'error_message': str(exc),
            },
            exc_info=True
        )


@celery_app.task(
    base=TuningTask,
    bind=True,
    name="app.tasks.tuning_tasks.tune_hyperparameters"
)
def tune_hyperparameters(
    self,
    tuning_run_id: str,
    model_run_id: str,
    tuning_method: str,
    param_grid: Optional[Dict[str, List[Any]]],
    cv_folds: int = 5,
    scoring_metric: Optional[str] = None,
    n_iter: Optional[int] = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning asynchronously.
    
    Args:
        self: Task instance (bound)
        tuning_run_id: UUID of the tuning run record
        model_run_id: UUID of the model run to tune
        tuning_method: Method to use (grid_search, random_search, bayesian)
        param_grid: Dictionary of hyperparameters to search
        cv_folds: Number of cross-validation folds
        scoring_metric: Metric to optimize (None = use default for task type)
        n_iter: Number of iterations for random_search/bayesian
        n_jobs: Number of parallel jobs (-1 = use all cores)
        random_state: Random seed for reproducibility
        user_id: UUID of the user (for logging)
    
    Returns:
        Dictionary with tuning results
    """
    logger = get_logger(
        task_id=self.request.id,
        user_id=user_id
    )
    
    logger.info(
        f"Starting hyperparameter tuning",
        extra={
            'event': 'tuning_start',
            'tuning_method': tuning_method,
            'tuning_run_id': tuning_run_id,
            'model_run_id': model_run_id
        }
    )
    
    task_start_time = time.time()
    db: Session = SessionLocal()
    
    try:
        # Update status to running
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': 100,
                'status': 'Initializing tuning...'
            }
        )
        
        # 1. Fetch tuning run and model run
        tuning_run = db.query(TuningRun).filter(
            TuningRun.id == uuid.UUID(tuning_run_id)
        ).first()
        
        if not tuning_run:
            raise ValueError(f"TuningRun with id {tuning_run_id} not found")
        
        model_run = db.query(ModelRun).filter(
            ModelRun.id == uuid.UUID(model_run_id)
        ).first()
        
        if not model_run:
            raise ValueError(f"ModelRun with id {model_run_id} not found")
        
        # 2. Load the trained model
        logger.info(f"Loading trained model from {model_run.model_artifact_path}")
        
        serialization_service = get_model_serialization_service()
        model, model_metadata = serialization_service.load_model(
            model_path=model_run.model_artifact_path,
            load_metadata=True
        )
        
        # 3. Load dataset and prepare data
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 10,
                'total': 100,
                'status': 'Loading dataset...'
            }
        )
        
        # Get dataset info from model metadata
        dataset_id = model_metadata.get('dataset_id')
        if not dataset_id:
            # Fallback: get from model_run's experiment
            from app.models.experiment import Experiment
            experiment = db.query(Experiment).filter(
                Experiment.id == model_run.experiment_id
            ).first()
            dataset_id = str(experiment.dataset_id) if experiment else None
        
        if not dataset_id:
            raise ValueError("Could not determine dataset_id for tuning")
        
        dataset = db.query(Dataset).filter(
            Dataset.id == uuid.UUID(dataset_id)
        ).first()
        
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")
        
        # Load dataset
        df = pd.read_csv(dataset.file_path)
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 4. Apply preprocessing steps
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Applying preprocessing...'
            }
        )
        
        preprocessing_steps = db.query(PreprocessingStep).filter(
            PreprocessingStep.dataset_id == uuid.UUID(dataset_id),
            PreprocessingStep.is_active == True
        ).order_by(PreprocessingStep.order).all()
        
        if preprocessing_steps:
            logger.info(f"Applying {len(preprocessing_steps)} preprocessing steps")
            for step in preprocessing_steps:
                if step.fitted_transformer:
                    transformer = deserialize_transformer(step.fitted_transformer)
                    df = pd.DataFrame(
                        transformer.transform(df),
                        columns=df.columns
                    )
        
        # 5. Prepare features and target
        target_column = model_metadata.get('target_column')
        feature_columns = model_metadata.get('feature_columns')
        
        if not target_column:
            raise ValueError("target_column not found in model metadata")
        
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
        
        y = df[target_column]
        
        logger.info(f"Features: {X.shape}, Target: {y.shape}")
        
        # 6. Get model info and determine scoring metric
        registry = ModelRegistry()
        model_info = registry.get_model(model_run.model_type)
        task_type = model_info.task_type.value
        
        if not scoring_metric:
            # Use default scoring metric based on task type
            if task_type == 'classification':
                scoring_metric = 'accuracy'
            elif task_type == 'regression':
                scoring_metric = 'r2'
            else:
                scoring_metric = 'silhouette'  # For clustering
        
        logger.info(f"Using scoring metric: {scoring_metric}")
        
        # 7. Prepare param grid (fallback to default search space when missing)
        if not param_grid:
            param_grid = get_default_search_space(model_run.model_type)
            if not param_grid:
                raise ValueError("param_grid is required and no default search space is defined for this model")
            logger.info("Using default search space for model %s", model_run.model_type)

        # 8. Perform hyperparameter tuning
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 30,
                'total': 100,
                'status': f'Running {tuning_method}...'
            }
        )
        
        tuning_start = time.time()
        
        # Get the underlying sklearn model
        sklearn_model = model.model
        
        # Use our tuning utilities for cleaner, more maintainable code
        if tuning_method == "grid_search":
            logger.info(f"Starting Grid Search with {len(param_grid)} parameters")
            
            result = run_grid_search(
                estimator=sklearn_model,
                X=X,
                y=y,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring_metric,
                n_jobs=n_jobs,
                return_train_score=True,
                verbose=1
            )
            
        elif tuning_method == "random_search":
            logger.info(f"Starting Random Search with {n_iter} iterations")
            
            result = run_random_search(
                estimator=sklearn_model,
                X=X,
                y=y,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring_metric,
                n_jobs=n_jobs,
                random_state=random_state,
                return_train_score=True,
                verbose=1
            )
            
        elif tuning_method == "bayesian":
            logger.info(f"Starting Bayesian Optimization with {n_iter} iterations")
            
            result = run_bayesian_search(
                estimator=sklearn_model,
                X=X,
                y=y,
                search_spaces=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring_metric,
                n_jobs=n_jobs,
                random_state=random_state,
                return_train_score=True,
                verbose=1
            )
            
        else:
            raise ValueError(f"Unknown tuning method: {tuning_method}")
        
        tuning_duration = time.time() - tuning_start
        logger.info(f"Tuning completed in {tuning_duration:.2f} seconds")
        
        # 9. Extract results from our structured result objects
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 90,
                'total': 100,
                'status': 'Processing results...'
            }
        )
        
        best_params = result.best_params
        best_score = result.best_score
        
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        # Get top 20 results
        top_results = result.top(20)
        
        # 10. Update TuningRun with results
        tuning_run.best_params = best_params
        tuning_run.results = {
            'best_score': float(best_score),
            'total_combinations': result.n_candidates,
            'all_results': top_results,
            'cv_folds': result.cv_folds,
            'scoring_metric': result.scoring,
            'tuning_time': tuning_duration,
            'tuning_method': tuning_method,
            'method_used': getattr(result, 'method', tuning_method)  # For Bayesian fallback tracking
        }
        tuning_run.status = TuningStatus.COMPLETED
        
        db.commit()
        db.refresh(tuning_run)
        
        logger.info(f"TuningRun updated with results")
        
        # Calculate total execution time
        total_execution_time = time.time() - task_start_time
        
        # Update progress: Complete
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Tuning completed successfully!'
            }
        )
        
        return_result = {
            'tuning_run_id': tuning_run_id,
            'model_run_id': model_run_id,
            'tuning_method': tuning_method,
            'best_params': best_params,
            'best_score': float(best_score),
            'total_combinations': result.n_candidates,
            'tuning_time': tuning_duration,
            'total_execution_time': total_execution_time,
            'status': 'completed',
            'method_used': getattr(result, 'method', tuning_method)
        }
        
        logger.info(
            f"Tuning completed successfully",
            extra={
                'event': 'tuning_complete',
                'best_score': best_score,
                'duration': total_execution_time
            }
        )
        
        return return_result
        
    except Exception as e:
        # Update tuning run status to failed
        try:
            tuning_run = db.query(TuningRun).filter(
                TuningRun.id == uuid.UUID(tuning_run_id)
            ).first()
            
            if tuning_run:
                tuning_run.status = TuningStatus.FAILED
                tuning_run.results = {
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update tuning run status: {db_error}")
        
        logger.error(
            f"Tuning failed: {e}",
            extra={
                'event': 'tuning_failed',
                'error': str(e),
                'tuning_run_id': tuning_run_id
            },
            exc_info=True
        )
        
        # Re-raise the exception for Celery to handle
        raise
        
    finally:
        db.close()


@celery_app.task(
    base=TuningTask,
    bind=True,
    name="app.tasks.tuning_tasks.validate_model_cv"
)
def validate_model_cv(
    self,
    model_run_id: str,
    cv_folds: int = 10,
    scoring_metrics: Optional[List[str]] = None,
    n_jobs: int = -1,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate a trained model using cross-validation.
    
    Args:
        self: Task instance (bound)
        model_run_id: UUID of the model run to validate
        cv_folds: Number of cross-validation folds
        scoring_metrics: List of metrics to evaluate (None = use defaults)
        n_jobs: Number of parallel jobs (-1 = use all cores)
        user_id: UUID of the user (for logging)
    
    Returns:
        Dictionary with cross-validation results
    """
    logger = get_logger(task_id=self.request.id, user_id=user_id)
    
    logger.info(
        f"Starting model cross-validation",
        extra={
            'event': 'cv_validation_start',
            'model_run_id': model_run_id,
            'cv_folds': cv_folds
        }
    )
    
    task_start_time = time.time()
    db: Session = SessionLocal()
    
    try:
        # Update state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing validation...'}
        )
        
        # 1. Load model run
        model_run = db.query(ModelRun).filter(
            ModelRun.id == uuid.UUID(model_run_id)
        ).first()
        
        if not model_run:
            raise ValueError(f"ModelRun with id {model_run_id} not found")
        
        # 2. Load the trained model
        serialization_service = get_model_serialization_service()
        model, model_metadata = serialization_service.load_model(
            model_path=model_run.model_artifact_path,
            load_metadata=True
        )
        
        # 3. Load dataset
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Loading dataset...'}
        )
        
        dataset_id = model_metadata.get('dataset_id')
        if not dataset_id:
            from app.models.experiment import Experiment
            experiment = db.query(Experiment).filter(
                Experiment.id == model_run.experiment_id
            ).first()
            dataset_id = str(experiment.dataset_id) if experiment else None
        
        if not dataset_id:
            raise ValueError("Could not determine dataset_id for validation")
        
        dataset = db.query(Dataset).filter(
            Dataset.id == uuid.UUID(dataset_id)
        ).first()
        
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")
        
        df = pd.read_csv(dataset.file_path)
        
        # 4. Apply preprocessing
        self.update_state(
            state='PROGRESS',
            meta={'current': 40, 'total': 100, 'status': 'Applying preprocessing...'}
        )
        
        preprocessing_steps = db.query(PreprocessingStep).filter(
            PreprocessingStep.dataset_id == uuid.UUID(dataset_id),
            PreprocessingStep.is_active == True
        ).order_by(PreprocessingStep.order).all()
        
        if preprocessing_steps:
            for step in preprocessing_steps:
                if step.fitted_transformer:
                    transformer = deserialize_transformer(step.fitted_transformer)
                    df = pd.DataFrame(transformer.transform(df), columns=df.columns)
        
        # 5. Prepare features and target
        target_column = model_metadata.get('target_column')
        feature_columns = model_metadata.get('feature_columns')
        
        if not target_column:
            raise ValueError("target_column not found in model metadata")
        
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
        
        y = df[target_column]
        
        # 6. Determine scoring metrics
        registry = ModelRegistry()
        model_info = registry.get_model(model_run.model_type)
        task_type = model_info.task_type.value
        
        if not scoring_metrics:
            if task_type == 'classification':
                scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            elif task_type == 'regression':
                scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            else:
                scoring_metrics = ['adjusted_rand_score', 'normalized_mutual_info']
        
        # 7. Run cross-validation
        self.update_state(
            state='PROGRESS',
            meta={'current': 60, 'total': 100, 'status': f'Running {cv_folds}-fold CV...'}
        )
        
        cv_start = time.time()
        
        cv_result = run_cross_validation(
            estimator=model.model,
            X=X,
            y=y,
            cv=cv_folds,
            scoring=scoring_metrics,
            return_train_score=True,
            n_jobs=n_jobs
        )
        
        cv_duration = time.time() - cv_start
        
        # 8. Process results
        self.update_state(
            state='PROGRESS',
            meta={'current': 90, 'total': 100, 'status': 'Processing results...'}
        )
        
        # Get confidence intervals
        ci_95 = cv_result.confidence_interval(0.95)
        ci_99 = cv_result.confidence_interval(0.99)
        
        validation_results = {
            'model_run_id': model_run_id,
            'cv_folds': cv_folds,
            'mean_score': float(cv_result.mean_score),
            'std_score': float(cv_result.std_score),
            'median_score': float(cv_result.median_score),
            'min_score': float(cv_result.min_score),
            'max_score': float(cv_result.max_score),
            'scores': [float(s) for s in cv_result.scores],
            'confidence_interval_95': {'lower': float(ci_95[0]), 'upper': float(ci_95[1])},
            'confidence_interval_99': {'lower': float(ci_99[0]), 'upper': float(ci_99[1])},
            'mean_fit_time': float(np.mean(cv_result.fit_times)),
            'mean_score_time': float(np.mean(cv_result.score_times)),
            'additional_metrics': {},
            'cv_duration': cv_duration,
            'total_execution_time': time.time() - task_start_time
        }
        
        # Add additional metrics
        if cv_result.additional_metrics:
            for metric_name, metric_data in cv_result.additional_metrics.items():
                validation_results['additional_metrics'][metric_name] = {
                    'mean': float(metric_data['mean']),
                    'std': float(metric_data['std']),
                    'scores': [float(s) for s in metric_data['scores']]
                }
        
        # Add train scores if available
        if cv_result.train_scores is not None:
            validation_results['mean_train_score'] = float(np.mean(cv_result.train_scores))
            validation_results['train_scores'] = [float(s) for s in cv_result.train_scores]
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 100, 'total': 100, 'status': 'Validation completed!'}
        )
        
        logger.info(
            f"Cross-validation completed successfully",
            extra={
                'event': 'cv_validation_complete',
                'mean_score': cv_result.mean_score,
                'duration': validation_results['total_execution_time']
            }
        )
        
        return validation_results
        
    except Exception as e:
        logger.error(
            f"Cross-validation failed: {e}",
            extra={
                'event': 'cv_validation_failed',
                'error': str(e),
                'model_run_id': model_run_id
            },
            exc_info=True
        )
        raise
        
    finally:
        db.close()

