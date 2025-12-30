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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.model_run import ModelRun
from app.models.tuning_run import TuningRun, TuningStatus
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
    param_grid: Dict[str, List[Any]],
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
        
        # 7. Perform hyperparameter tuning
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
        
        if tuning_method == "grid_search":
            logger.info(f"Starting Grid Search with {len(param_grid)} parameters")
            
            search = GridSearchCV(
                estimator=sklearn_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring_metric,
                n_jobs=n_jobs,
                verbose=1,
                return_train_score=True
            )
            
        elif tuning_method == "random_search":
            logger.info(f"Starting Random Search with {n_iter} iterations")
            
            search = RandomizedSearchCV(
                estimator=sklearn_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring_metric,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=1,
                return_train_score=True
            )
            
        elif tuning_method == "bayesian":
            # Bayesian optimization requires scikit-optimize
            try:
                from skopt import BayesSearchCV
                
                logger.info(f"Starting Bayesian Optimization with {n_iter} iterations")
                
                search = BayesSearchCV(
                    estimator=sklearn_model,
                    search_spaces=param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring=scoring_metric,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbose=1,
                    return_train_score=True
                )
            except ImportError:
                logger.warning("scikit-optimize not installed, falling back to RandomizedSearchCV")
                search = RandomizedSearchCV(
                    estimator=sklearn_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring=scoring_metric,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbose=1,
                    return_train_score=True
                )
        else:
            raise ValueError(f"Unknown tuning method: {tuning_method}")
        
        # Fit the search
        search.fit(X, y)
        
        tuning_duration = time.time() - tuning_start
        logger.info(f"Tuning completed in {tuning_duration:.2f} seconds")
        
        # 8. Extract results
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 90,
                'total': 100,
                'status': 'Processing results...'
            }
        )
        
        # Get best parameters and score
        best_params = search.best_params_
        best_score = search.best_score_
        
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        # Get top N results
        cv_results = search.cv_results_
        n_results = len(cv_results['mean_test_score'])
        
        # Sort by mean test score (descending)
        sorted_indices = np.argsort(cv_results['mean_test_score'])[::-1]
        
        top_results = []
        for rank, idx in enumerate(sorted_indices[:20], start=1):  # Top 20
            result_item = {
                'rank': rank,
                'params': {key: cv_results[f'param_{key}'][idx] for key in param_grid.keys()},
                'mean_score': float(cv_results['mean_test_score'][idx]),
                'std_score': float(cv_results['std_test_score'][idx]),
                'scores': [float(cv_results[f'split{i}_test_score'][idx]) for i in range(cv_folds)]
            }
            top_results.append(result_item)
        
        # 9. Update TuningRun with results
        tuning_run.best_params = best_params
        tuning_run.results = {
            'best_score': float(best_score),
            'total_combinations': n_results,
            'all_results': top_results,
            'cv_folds': cv_folds,
            'scoring_metric': scoring_metric,
            'tuning_time': tuning_duration,
            'tuning_method': tuning_method
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
        
        result = {
            'tuning_run_id': tuning_run_id,
            'model_run_id': model_run_id,
            'tuning_method': tuning_method,
            'best_params': best_params,
            'best_score': float(best_score),
            'total_combinations': n_results,
            'tuning_time': tuning_duration,
            'total_execution_time': total_execution_time,
            'status': 'completed'
        }
        
        logger.info(
            f"Tuning completed successfully",
            extra={
                'event': 'tuning_complete',
                'best_score': best_score,
                'duration': total_execution_time
            }
        )
        
        return result
        
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
