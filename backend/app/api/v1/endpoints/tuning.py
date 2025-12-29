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
from app.core.logging_config import get_logger

router = APIRouter()

logger = get_logger(__name__)


# Helper functions
def get_current_user_id() -> str:
    """
    Mock function to get current user ID.
    Replace with actual authentication in production.
    """
    # TODO: Replace with actual user authentication
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
    
    Args:
        tuning_run_id: UUID of the tuning run
        db: Database session
        user_id: Current user ID
    
    Returns:
        HyperparameterTuningStatus with current status and progress
    
    Raises:
        HTTPException 404: If tuning run not found
        HTTPException 403: If user doesn't have permission
    
    Example:
        GET /api/v1/tuning/tune/abc-123/status
    """
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
    
    # 3. Get task_id from tuning_run (if stored)
    # For now, we'll use the status from the database
    # In a full implementation, you'd track the Celery task_id
    
    # Map TuningStatus to Celery-like status
    status_map = {
        TuningStatus.RUNNING: "PROGRESS",
        TuningStatus.COMPLETED: "SUCCESS",
        TuningStatus.FAILED: "FAILURE"
    }
    
    celery_status = status_map.get(tuning_run.status, "PENDING")
    
    # Build response
    response = {
        "tuning_run_id": tuning_run.id,
        "task_id": None,  # Would be stored in tuning_run metadata
        "status": celery_status,
        "progress": None,
        "result": None,
        "error": None
    }
    
    # If completed, include results
    if tuning_run.status == TuningStatus.COMPLETED and tuning_run.best_params:
        response["result"] = {
            "best_params": tuning_run.best_params,
            "best_score": tuning_run.results.get("best_score") if tuning_run.results else None
        }
    
    return HyperparameterTuningStatus(**response)


@router.get("/tune/{tuning_run_id}/results", response_model=HyperparameterTuningResults)
async def get_tuning_results(
    tuning_run_id: str,
    top_n: int = 10,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get the complete results of a hyperparameter tuning run.
    
    Args:
        tuning_run_id: UUID of the tuning run
        top_n: Number of top parameter combinations to return (default: 10)
        db: Database session
        user_id: Current user ID
    
    Returns:
        Complete tuning results including best parameters and top N combinations
    
    Raises:
        HTTPException 404: If tuning run not found
        HTTPException 400: If tuning not completed
        HTTPException 403: If user doesn't have permission
    
    Example:
        GET /api/v1/tuning/tune/abc-123/results?top_n=5
    """
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tuning run is not completed yet. Current status: {tuning_run.status.value}"
        )
    
    # 4. Check if results exist
    if not tuning_run.best_params or not tuning_run.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results available for this tuning run"
        )
    
    # 5. Extract top N results
    all_results = tuning_run.results.get("all_results", [])
    top_results = all_results[:top_n] if all_results else []
    
    return HyperparameterTuningResults(
        tuning_run_id=str(tuning_run.id),
        model_run_id=str(tuning_run.model_run_id),
        tuning_method=tuning_run.tuning_method,
        best_params=tuning_run.best_params,
        best_score=tuning_run.results.get("best_score", 0.0),
        total_combinations=tuning_run.results.get("total_combinations", 0),
        top_results=top_results,
        cv_folds=tuning_run.results.get("cv_folds", 5),
        scoring_metric=tuning_run.results.get("scoring_metric", "accuracy"),
        tuning_time=tuning_run.results.get("tuning_time"),
        created_at=tuning_run.created_at.isoformat()
    )
