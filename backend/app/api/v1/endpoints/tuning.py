"""
Hyperparameter tuning endpoints.

Provides REST API for hyperparameter optimization including
grid search, random search, and Bayesian optimization.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional
from sqlalchemy.orm import Session
from uuid import uuid4, UUID
from datetime import datetime
import uuid

from app.models.model_run import ModelRun
from app.models.tuning_run import TuningRun, TuningStatus
from app.models.experiment import Experiment
from app.schemas.model import (
    HyperparameterTuningRequest,
    HyperparameterTuningResponse,
    HyperparameterTuningStatus,
    HyperparameterTuningResults
)
from app.tasks.tuning_tasks import tune_hyperparameters
from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.utils.logger import get_logger

router = APIRouter()

logger = get_logger(__name__)


# Helper functions
def get_current_user_id() -> str:
    """
    Get current user ID from authentication context.
    In production: Extract from JWT token or session.
    Example: decode_jwt(request.headers['Authorization'])['user_id']
    """
    # Default user ID for development/testing
    return "00000000-0000-0000-0000-000000000001"


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/tune", response_model=HyperparameterTuningResponse, status_code=status.HTTP_202_ACCEPTED)
async def tune_model_hyperparameters(
    request: HyperparameterTuningRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Initiate asynchronous hyperparameter tuning for a trained model.
    
    This endpoint creates a tuning run record and triggers an asynchronous
    Celery task to perform hyperparameter optimization using the specified method.
    
    **Tuning Methods:**
    - **grid_search**: Exhaustive search over all parameter combinations
    - **random_search**: Random sampling of parameter combinations (faster)
    - **bayesian**: Bayesian optimization using Gaussian processes (most efficient)
    
    **Scoring Metrics:**
    - Classification: accuracy, precision, recall, f1, roc_auc
    - Regression: r2, neg_mean_squared_error, neg_mean_absolute_error
    
    Args:
        request: Tuning configuration
        db: Database session
        user_id: Current user ID
    
    Returns:
        HyperparameterTuningResponse with task_id for status checking
    
    Raises:
        HTTPException 404: If model run not found
        HTTPException 400: If invalid request or model not completed
        HTTPException 403: If user doesn't have permission
    
    Example:
        POST /api/v1/tuning/tune
        {
            "model_run_id": "abc-123",
            "tuning_method": "grid_search",
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None]
            },
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }
    """
    logger.info(
        f"Hyperparameter tuning requested",
        extra={
            'event': 'tuning_request',
            'model_run_id': str(request.model_run_id),
            'tuning_method': request.tuning_method,
            'user_id': user_id
        }
    )
    
    # 1. Fetch and validate model run
    model_run = db.query(ModelRun).filter(
        ModelRun.id == request.model_run_id
    ).first()
    
    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model run with id {request.model_run_id} not found"
        )
    
    # 2. Verify user owns this model run through experiment
    experiment = db.query(Experiment).filter(
        Experiment.id == model_run.experiment_id,
        Experiment.user_id == uuid.UUID(user_id)
    ).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to tune this model"
        )
    
    # 3. Check if model run is completed
    if model_run.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model run must be completed before tuning. Current status: {model_run.status}"
        )
    
    # 4. Validate param_grid is not empty
    if not request.param_grid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="param_grid cannot be empty"
        )
    
    # 5. Validate tuning method
    valid_methods = ["grid_search", "random_search", "bayesian"]
    if request.tuning_method not in valid_methods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tuning_method. Must be one of: {', '.join(valid_methods)}"
        )
    
    # 6. Create TuningRun record
    tuning_run = TuningRun(
        id=uuid4(),
        model_run_id=request.model_run_id,
        tuning_method=request.tuning_method,
        status=TuningStatus.RUNNING,
        created_at=datetime.utcnow()
    )
    
    db.add(tuning_run)
    db.commit()
    db.refresh(tuning_run)
    
    logger.info(
        f"TuningRun created",
        extra={
            'event': 'tuning_run_created',
            'tuning_run_id': str(tuning_run.id),
            'model_run_id': str(model_run.id)
        }
    )
    
    # 7. Trigger Celery task
    task = tune_hyperparameters.delay(
        tuning_run_id=str(tuning_run.id),
        model_run_id=str(request.model_run_id),
        tuning_method=request.tuning_method,
        param_grid=request.param_grid,
        cv_folds=request.cv_folds,
        scoring_metric=request.scoring_metric,
        n_iter=request.n_iter,
        n_jobs=request.n_jobs,
        random_state=request.random_state,
        user_id=user_id
    )
    
    # Store task_id in tuning_run for status tracking
    if not tuning_run.results:
        tuning_run.results = {}
    tuning_run.results['task_id'] = task.id
    
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(tuning_run, 'results')
    db.commit()
    
    logger.info(
        f"Tuning task triggered",
        extra={
            'event': 'tuning_task_triggered',
            'task_id': task.id,
            'tuning_run_id': str(tuning_run.id)
        }
    )
    
    return HyperparameterTuningResponse(
        tuning_run_id=tuning_run.id,
        task_id=task.id,
        status="PENDING",
        message="Hyperparameter tuning initiated successfully",
        created_at=tuning_run.created_at
    )


@router.get("/tune/{tuning_run_id}/status", response_model=HyperparameterTuningStatus)
async def get_tuning_status(
    tuning_run_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get the status of a hyperparameter tuning task.
    
    This endpoint provides real-time status information about a tuning run,
    including progress updates from the Celery task if available.
    
    **Status Values:**
    - **PENDING**: Task is queued but not started
    - **PROGRESS**: Task is actively running
    - **SUCCESS**: Task completed successfully
    - **FAILURE**: Task failed with error
    - **REVOKED**: Task was cancelled
    
    Args:
        tuning_run_id: UUID of the tuning run
        db: Database session
        user_id: Current user ID
    
    Returns:
        HyperparameterTuningStatus with current status and progress
    
    Raises:
        HTTPException 400: If invalid UUID format
        HTTPException 404: If tuning run not found
        HTTPException 403: If user doesn't have permission
    
    Example:
        GET /api/v1/tuning/tune/abc-123/status
        
        Response:
        {
            "tuning_run_id": "abc-123",
            "task_id": "xyz-789",
            "status": "PROGRESS",
            "progress": {
                "current": 15,
                "total": 36,
                "status": "Testing parameter combination 15/36...",
                "percentage": 41.67
            },
            "result": null,
            "error": null
        }
    """
    logger.info(
        f"Tuning status requested",
        extra={
            'event': 'tuning_status_request',
            'tuning_run_id': tuning_run_id,
            'user_id': user_id
        }
    )
    
    # 1. Fetch tuning run
    try:
        tuning_run = db.query(TuningRun).filter(
            TuningRun.id == uuid.UUID(tuning_run_id)
        ).first()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tuning_run_id format: {tuning_run_id}"
        )
    
    if not tuning_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tuning run with id {tuning_run_id} not found"
        )
    
    # 2. Verify user owns this tuning run through model run and experiment
    model_run = db.query(ModelRun).filter(
        ModelRun.id == tuning_run.model_run_id
    ).first()
    
    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated model run not found"
        )
    
    experiment = db.query(Experiment).filter(
        Experiment.id == model_run.experiment_id,
        Experiment.user_id == uuid.UUID(user_id)
    ).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this tuning run"
        )
    
    # 3. Get task_id from tuning_run results
    task_id = None
    if tuning_run.results and isinstance(tuning_run.results, dict):
        task_id = tuning_run.results.get('task_id')
    
    # 4. Check Celery task status if task_id exists
    celery_status = None
    progress = None
    result = None
    error = None
    
    if task_id:
        try:
            from celery.result import AsyncResult
            task_result = AsyncResult(task_id, app=celery_app)
            
            celery_status = task_result.state
            
            logger.debug(
                f"Celery task status: {celery_status}",
                extra={
                    'event': 'celery_status_check',
                    'task_id': task_id,
                    'status': celery_status
                }
            )
            
            # Get progress information if task is running
            if celery_status == 'PROGRESS':
                task_info = task_result.info
                if isinstance(task_info, dict):
                    progress = {
                        'current': task_info.get('current', 0),
                        'total': task_info.get('total', 100),
                        'status': task_info.get('status', 'Processing...'),
                        'percentage': round((task_info.get('current', 0) / task_info.get('total', 100)) * 100, 2) if task_info.get('total', 0) > 0 else 0
                    }
            
            # Get result if task completed
            elif celery_status == 'SUCCESS':
                task_result_data = task_result.result
                if isinstance(task_result_data, dict):
                    result = {
                        'best_params': task_result_data.get('best_params'),
                        'best_score': task_result_data.get('best_score'),
                        'total_combinations': task_result_data.get('total_combinations'),
                        'tuning_time': task_result_data.get('tuning_time')
                    }
            
            # Get error if task failed
            elif celery_status == 'FAILURE':
                error_info = task_result.info
                if isinstance(error_info, Exception):
                    error = {
                        'type': type(error_info).__name__,
                        'message': str(error_info)
                    }
                elif isinstance(error_info, dict):
                    error = error_info
                else:
                    error = {'message': str(error_info)}
        
        except Exception as e:
            logger.warning(
                f"Failed to get Celery task status: {e}",
                extra={
                    'event': 'celery_status_error',
                    'task_id': task_id,
                    'error': str(e)
                }
            )
            # Continue with database status if Celery check fails
    
    # 5. Determine overall status
    # Priority: Celery status > Database status
    if celery_status:
        final_status = celery_status
    else:
        # Map TuningStatus to Celery-like status
        status_map = {
            TuningStatus.RUNNING: "PROGRESS",
            TuningStatus.COMPLETED: "SUCCESS",
            TuningStatus.FAILED: "FAILURE"
        }
        final_status = status_map.get(tuning_run.status, "PENDING")
    
    # 6. If completed in database, include results
    if tuning_run.status == TuningStatus.COMPLETED and not result:
        if tuning_run.best_params:
            result = {
                "best_params": tuning_run.best_params,
                "best_score": tuning_run.results.get("best_score") if tuning_run.results else None,
                "total_combinations": tuning_run.results.get("total_combinations") if tuning_run.results else None,
                "tuning_time": tuning_run.results.get("tuning_time") if tuning_run.results else None
            }
    
    # 7. If failed in database, include error
    if tuning_run.status == TuningStatus.FAILED and not error:
        if tuning_run.results and isinstance(tuning_run.results, dict):
            error_data = tuning_run.results.get('error')
            if error_data:
                if isinstance(error_data, dict):
                    error = error_data
                else:
                    error = {'message': str(error_data)}
    
    logger.info(
        f"Tuning status retrieved",
        extra={
            'event': 'tuning_status_retrieved',
            'tuning_run_id': tuning_run_id,
            'status': final_status,
            'has_progress': progress is not None
        }
    )
    
    return HyperparameterTuningStatus(
        tuning_run_id=tuning_run.id,
        task_id=task_id,
        status=final_status,
        progress=progress,
        result=result,
        error=error
    )


@router.get("/tune/{tuning_run_id}/results", response_model=HyperparameterTuningResults)
async def get_tuning_results(
    tuning_run_id: str,
    top_n: int = 10,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get the complete results of a hyperparameter tuning run.
    
    This endpoint returns comprehensive tuning results including the best parameters
    found, performance scores, and top N parameter combinations with their scores.
    
    **Use Cases:**
    - View best hyperparameters to apply to model
    - Compare top parameter combinations
    - Analyze which parameters had most impact
    - Export results for documentation
    
    **Result Ranking:**
    Results are ranked by cross-validation score (descending), with rank 1 being
    the best performing combination.
    
    Args:
        tuning_run_id: UUID of the tuning run
        top_n: Number of top parameter combinations to return (default: 10, max: 100)
        db: Database session
        user_id: Current user ID
    
    Returns:
        Complete tuning results including:
        - Best parameters and score
        - Top N parameter combinations with CV scores
        - Tuning metadata (method, folds, metric, duration)
    
    Raises:
        HTTPException 400: If invalid UUID or tuning not completed
        HTTPException 404: If tuning run not found or no results available
        HTTPException 403: If user doesn't have permission
    
    Example:
        GET /api/v1/tuning/tune/abc-123/results?top_n=5
        
        Response:
        {
            "tuning_run_id": "abc-123",
            "model_run_id": "def-456",
            "tuning_method": "grid_search",
            "best_params": {
                "n_estimators": 100,
                "max_depth": 10
            },
            "best_score": 0.9567,
            "total_combinations": 36,
            "top_results": [
                {
                    "rank": 1,
                    "params": {"n_estimators": 100, "max_depth": 10},
                    "mean_score": 0.9567,
                    "std_score": 0.0123,
                    "scores": [0.95, 0.96, 0.97, 0.94, 0.96]
                }
            ],
            "cv_folds": 5,
            "scoring_metric": "accuracy",
            "tuning_time": 120.5,
            "created_at": "2025-12-29T10:00:00Z"
        }
    """
    logger.info(
        f"Tuning results requested",
        extra={
            'event': 'tuning_results_request',
            'tuning_run_id': tuning_run_id,
            'top_n': top_n,
            'user_id': user_id
        }
    )
    
    # Validate top_n parameter
    if top_n < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_n must be at least 1"
        )
    
    if top_n > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_n cannot exceed 100"
        )
    
    # 1. Fetch tuning run
    try:
        tuning_run = db.query(TuningRun).filter(
            TuningRun.id == uuid.UUID(tuning_run_id)
        ).first()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tuning_run_id format: {tuning_run_id}"
        )
    
    if not tuning_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tuning run with id {tuning_run_id} not found"
        )
    
    # 2. Verify permissions
    model_run = db.query(ModelRun).filter(
        ModelRun.id == tuning_run.model_run_id
    ).first()
    
    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated model run not found"
        )
    
    experiment = db.query(Experiment).filter(
        Experiment.id == model_run.experiment_id,
        Experiment.user_id == uuid.UUID(user_id)
    ).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this tuning run"
        )
    
    # 3. Check if completed
    if tuning_run.status != TuningStatus.COMPLETED:
        # Provide helpful message based on current status
        status_messages = {
            TuningStatus.RUNNING: "Tuning is still in progress. Please check status endpoint for updates.",
            TuningStatus.FAILED: "Tuning failed. Please check status endpoint for error details."
        }
        message = status_messages.get(
            tuning_run.status,
            f"Tuning run is not completed yet. Current status: {tuning_run.status.value}"
        )
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    # 4. Check if results exist
    if not tuning_run.best_params:
        logger.error(
            f"Tuning run completed but no best_params found",
            extra={
                'event': 'tuning_results_missing',
                'tuning_run_id': tuning_run_id,
                'status': tuning_run.status.value
            }
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No best parameters available for this tuning run"
        )
    
    if not tuning_run.results:
        logger.error(
            f"Tuning run completed but no results found",
            extra={
                'event': 'tuning_results_missing',
                'tuning_run_id': tuning_run_id,
                'status': tuning_run.status.value
            }
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results available for this tuning run"
        )
    
    # 5. Extract and validate results data
    results_data = tuning_run.results
    
    # Get best score
    best_score = results_data.get("best_score")
    if best_score is None:
        logger.warning(
            f"Best score not found in results",
            extra={
                'event': 'best_score_missing',
                'tuning_run_id': tuning_run_id
            }
        )
        best_score = 0.0
    
    # Get total combinations
    total_combinations = results_data.get("total_combinations", 0)
    
    # Get all results and extract top N
    all_results = results_data.get("all_results", [])
    
    if not all_results:
        logger.warning(
            f"No detailed results found, returning summary only",
            extra={
                'event': 'detailed_results_missing',
                'tuning_run_id': tuning_run_id
            }
        )
    
    # Limit to requested top_n
    top_results = all_results[:min(top_n, len(all_results))] if all_results else []
    
    # Get metadata with defaults
    cv_folds = results_data.get("cv_folds", 5)
    scoring_metric = results_data.get("scoring_metric", "accuracy")
    tuning_time = results_data.get("tuning_time")
    tuning_method = results_data.get("tuning_method", tuning_run.tuning_method)
    
    logger.info(
        f"Tuning results retrieved",
        extra={
            'event': 'tuning_results_retrieved',
            'tuning_run_id': tuning_run_id,
            'best_score': best_score,
            'total_combinations': total_combinations,
            'top_n_returned': len(top_results)
        }
    )
    
    return HyperparameterTuningResults(
        tuning_run_id=str(tuning_run.id),
        model_run_id=str(tuning_run.model_run_id),
        tuning_method=tuning_method,
        best_params=tuning_run.best_params,
        best_score=float(best_score),
        total_combinations=total_combinations,
        top_results=top_results,
        cv_folds=cv_folds,
        scoring_metric=scoring_metric,
        tuning_time=tuning_time,
        created_at=tuning_run.created_at.isoformat()
    )
