"""
Model training endpoints.

Provides REST API for machine learning model operations including
listing available models, training, evaluation, and prediction.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from uuid import uuid4, UUID
from datetime import datetime
import uuid

from app.ml_engine.model_registry import model_registry, TaskType, ModelCategory, ModelRegistry
from app.models.model_run import ModelRun
from app.models.experiment import Experiment
from app.models.dataset import Dataset
from app.schemas.model import (
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelTrainingStatus,
    ModelRunDeletionResponse,
    ModelMetricsResponse
)
from app.tasks.training_tasks import train_model
from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.services.training_validation_service import get_training_validator, ValidationError
from app.services.storage_service import get_model_serialization_service
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


@router.get("/available")
async def get_available_models(
    task_type: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get list of available machine learning models.

    Query parameters:
        task_type: Filter by task type (regression, classification, clustering)
        category: Filter by model category (linear, tree_based, boosting, etc.)
        search: Search query for model name, description, or tags

    Returns:
        Dictionary containing available models organized by task type

    Example:
        GET /api/v1/models/available
        GET /api/v1/models/available?task_type=classification
        GET /api/v1/models/available?category=boosting
        GET /api/v1/models/available?search=random forest
    """
    try:
        # Apply filters
        if search:
            # Search across all fields
            models = model_registry.search_models(search)

            # Group by task type
            result = {
                "regression": [m.to_dict() for m in models if m.task_type == TaskType.REGRESSION],
                "classification": [m.to_dict() for m in models if m.task_type == TaskType.CLASSIFICATION],
                "clustering": [m.to_dict() for m in models if m.task_type == TaskType.CLUSTERING],
            }

            return {
                "search_query": search,
                "total_results": len(models),
                "models": result
            }

        elif task_type:
            # Filter by task type
            try:
                task_enum = TaskType(task_type.lower())
                models = model_registry.get_models_by_task(task_enum)

                return {
                    "task_type": task_type,
                    "count": len(models),
                    "models": [model.to_dict() for model in models]
                }
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid task_type '{task_type}'. Must be one of: regression, classification, clustering"
                )

        elif category:
            # Filter by category
            try:
                category_enum = ModelCategory(category.lower())
                models = model_registry.get_models_by_category(category_enum)

                return {
                    "category": category,
                    "count": len(models),
                    "models": [model.to_dict() for model in models]
                }
            except ValueError:
                valid_categories = [c.value for c in ModelCategory]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category '{category}'. Must be one of: {', '.join(valid_categories)}"
                )

        else:
            # Return all models grouped by task type
            all_models = model_registry.get_all_models()

            result = {
                "regression": [model.to_dict() for model in all_models["regression"]],
                "classification": [model.to_dict() for model in all_models["classification"]],
                "clustering": [model.to_dict() for model in all_models["clustering"]],
            }

            total_count = sum(len(models) for models in all_models.values())

            return {
                "total_models": total_count,
                "models_by_task": result,
                "summary": {
                    "regression_models": len(all_models["regression"]),
                    "classification_models": len(all_models["classification"]),
                    "clustering_models": len(all_models["clustering"]),
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving models: {str(e)}"
        )


@router.get("/available/{model_id}")
async def get_model_details(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.

    Path parameters:
        model_id: Unique identifier of the model

    Returns:
        Detailed model information including hyperparameters and capabilities

    Example:
        GET /api/v1/models/available/random_forest_classifier
    """
    try:
        model = model_registry.get_model(model_id)
        return model.to_dict()
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model details: {str(e)}"
        )


@router.get("/categories")
async def get_model_categories() -> Dict[str, Any]:
    """
    Get all available model categories.

    Returns:
        List of model categories with descriptions

    Example:
        GET /api/v1/models/categories
    """
    return {
        "categories": [
            {
                "id": "linear",
                "name": "Linear Models",
                "description": "Models that assume linear relationships between features and target"
            },
            {
                "id": "tree_based",
                "name": "Tree-Based Models",
                "description": "Models based on decision trees and tree ensembles"
            },
            {
                "id": "boosting",
                "name": "Boosting Models",
                "description": "Sequential ensemble models that correct previous errors"
            },
            {
                "id": "support_vector",
                "name": "Support Vector Machines",
                "description": "Models that find optimal hyperplanes for classification/regression"
            },
            {
                "id": "instance_based",
                "name": "Instance-Based Models",
                "description": "Models that make predictions based on nearest training examples"
            },
            {
                "id": "neural_network",
                "name": "Neural Networks",
                "description": "Multi-layer perceptron and deep learning models"
            },
            {
                "id": "probabilistic",
                "name": "Probabilistic Models",
                "description": "Models based on probability theory and Bayes theorem"
            },
            {
                "id": "density_based",
                "name": "Density-Based Clustering",
                "description": "Clustering based on density of data points"
            },
            {
                "id": "hierarchical",
                "name": "Hierarchical Clustering",
                "description": "Clustering that builds a hierarchy of clusters"
            }
        ]
    }


@router.get("/task-types")
async def get_task_types() -> Dict[str, Any]:
    """
    Get all available task types.

    Returns:
        List of task types with descriptions and model counts

    Example:
        GET /api/v1/models/task-types
    """
    all_models = model_registry.get_all_models()

    return {
        "task_types": [
            {
                "id": "regression",
                "name": "Regression",
                "description": "Predict continuous numerical values",
                "model_count": len(all_models["regression"]),
                "examples": ["house prices", "sales forecasting", "temperature prediction"]
            },
            {
                "id": "classification",
                "name": "Classification",
                "description": "Predict discrete categories or classes",
                "model_count": len(all_models["classification"]),
                "examples": ["spam detection", "disease diagnosis", "customer churn"]
            },
            {
                "id": "clustering",
                "name": "Clustering",
                "description": "Group similar data points together (unsupervised)",
                "model_count": len(all_models["clustering"]),
                "examples": ["customer segmentation", "anomaly detection", "document grouping"]
            }
        ]
    }


@router.post("/train", response_model=ModelTrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model_endpoint(
    request: ModelTrainingRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Initiate asynchronous model training.

    This endpoint creates a model run record and triggers an asynchronous
    Celery task to train the model. Returns immediately with task ID.

    Args:
        request: Training configuration
        db: Database session
        user_id: Current user ID

    Returns:
        ModelTrainingResponse with task_id for status checking

    Raises:
        HTTPException 404: If experiment or dataset not found
        HTTPException 400: If invalid request
        HTTPException 403: If user doesn't have permission
    """
    try:
        # Validate training configuration using validation service
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
        
    except ValidationError as e:
        # Return appropriate HTTP error based on validation failure
        if e.field in ['experiment_id', 'dataset_id']:
            if 'not found' in e.message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=e.message
                )
            elif 'does not belong' in e.message:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=e.message
                )
        
        # All other validation errors are bad requests
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )

    # Create ModelRun record
    model_run = ModelRun(
        id=uuid4(),
        experiment_id=request.experiment_id,
        model_type=request.model_type,
        hyperparameters=request.hyperparameters or {},
        status="pending",
        created_at=datetime.utcnow()
    )

    db.add(model_run)
    db.commit()
    db.refresh(model_run)

    # Update experiment status to running
    if experiment.status != "running":
        experiment.status = "running"
        db.commit()

    # Trigger Celery task
    task = train_model.delay(
        model_run_id=str(model_run.id),
        experiment_id=str(request.experiment_id),
        dataset_id=str(request.dataset_id),
        model_type=request.model_type,
        hyperparameters=request.hyperparameters,
        target_column=request.target_column,
        feature_columns=request.feature_columns,
        test_size=request.test_size,
        random_state=request.random_state,
        user_id=user_id
    )

    # Store task_id in model_run run_metadata
    if not model_run.run_metadata:
        model_run.run_metadata = {}
    model_run.run_metadata['task_id'] = task.id
    db.commit()

    return ModelTrainingResponse(
        model_run_id=model_run.id,
        task_id=task.id,
        status="PENDING",
        message="Model training initiated successfully",
        created_at=model_run.created_at
    )


@router.get("/train/{model_run_id}/status", response_model=ModelTrainingStatus)
async def get_training_status(
    model_run_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get the status of a training task.

    Args:
        model_run_id: UUID of the model run
        db: Database session
        user_id: Current user ID

    Returns:
        ModelTrainingStatus with current status and progress

    Raises:
        HTTPException 404: If model run not found
    """
    # 1. Fetch model run
    model_run = db.query(ModelRun).filter(
        ModelRun.id == uuid.UUID(model_run_id)
    ).first()

    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model run with id {model_run_id} not found"
        )

    # 2. Verify user owns this model run through experiment
    experiment = db.query(Experiment).filter(
        Experiment.id == model_run.experiment_id,
        Experiment.user_id == uuid.UUID(user_id)
    ).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this model run"
        )

    # 3. Get task_id from run_metadata
    task_id = model_run.run_metadata.get('task_id') if model_run.run_metadata else None

    # 4. Check Celery task status if task_id exists
    celery_status = None
    progress = None
    result = None
    error = None

    if task_id:
        from celery.result import AsyncResult
        task_result = AsyncResult(task_id, app=celery_app)

        celery_status = task_result.state

        if celery_status == 'PROGRESS':
            progress = task_result.info
        elif celery_status == 'SUCCESS':
            result = task_result.result
        elif celery_status == 'FAILURE':
            error = str(task_result.info)

    # 5. Determine overall status
    # Priority: Celery status > ModelRun status
    if celery_status:
        status_str = celery_status
    else:
        # Map model_run.status to Celery-like status
        status_map = {
            'pending': 'PENDING',
            'running': 'PROGRESS',
            'completed': 'SUCCESS',
            'failed': 'FAILURE',
            'cancelled': 'REVOKED'
        }
        status_str = status_map.get(model_run.status, 'PENDING')

    # 6. If completed, include metrics from model_run
    if model_run.status == "completed" and model_run.metrics:
        result = {
            'metrics': model_run.metrics,
            'training_time': model_run.training_time,
            'model_path': model_run.model_artifact_path,
            'feature_importance': model_run.run_metadata.get('feature_importance') if model_run.run_metadata else None
        }

    # If failed, include error from run_metadata
    if model_run.status == "failed" and model_run.run_metadata:
        error_info = model_run.run_metadata.get('error', {})
        error = {
            'type': error_info.get('type', 'Error'),
            'code': error_info.get('code', 'UNKNOWN_ERROR'),
            'message': error_info.get('message', 'Unknown error'),
            'user_message': error_info.get('user_message', 'Training failed'),
            'recoverable': error_info.get('recoverable', False),
            'phase': error_info.get('phase', 'unknown')
        }

    return ModelTrainingStatus(
        model_run_id=model_run.id,
        task_id=task_id,
        status=status_str,
        progress=progress,
        result=result,
        error=error
    )


@router.get("/train/{model_run_id}/result")
async def get_training_result(
    model_run_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get the full results of a completed training task.

    Args:
        model_run_id: UUID of the model run
        db: Database session
        user_id: Current user ID

    Returns:
        Complete training results including metrics and metadata

    Raises:
        HTTPException 404: If model run not found
        HTTPException 400: If training not completed
    """
    # Fetch model run
    model_run = db.query(ModelRun).filter(
        ModelRun.id == uuid.UUID(model_run_id)
    ).first()

    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model run with id {model_run_id} not found"
        )

    # Verify permissions
    experiment = db.query(Experiment).filter(
        Experiment.id == model_run.experiment_id,
        Experiment.user_id == uuid.UUID(user_id)
    ).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this model run"
        )

    # Check if completed
    if model_run.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model run is not completed yet. Current status: {model_run.status}"
        )

    return {
        "model_run_id": str(model_run.id),
        "model_type": model_run.model_type,
        "hyperparameters": model_run.hyperparameters,
        "metrics": model_run.metrics,
        "training_time": model_run.training_time,
        "model_artifact_path": model_run.model_artifact_path,
        "feature_importance": model_run.run_metadata.get('feature_importance') if model_run.run_metadata else None,
        "created_at": model_run.created_at.isoformat(),
        "experiment_id": str(model_run.experiment_id)
    }


@router.delete("/train/{model_run_id}", response_model=ModelRunDeletionResponse, status_code=status.HTTP_200_OK)
async def delete_model_run(
    model_run_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Delete a model run and its associated artifacts.
    
    This endpoint:
    1. Verifies user owns the model run (through experiment)
    2. Revokes the Celery task if still running
    3. Deletes the model artifact file from storage
    4. Deletes the ModelRun database record
    
    Args:
        model_run_id: UUID of the model run to delete
        db: Database session
        user_id: Current user ID
    
    Returns:
        Success message with deletion details
    
    Raises:
        HTTPException 404: If model run not found
        HTTPException 403: If user doesn't have permission
        HTTPException 500: If deletion fails
    
    Example:
        DELETE /api/v1/models/train/123e4567-e89b-12d3-a456-426614174002
    """
    logger.info(
        f"Delete model run requested",
        extra={
            'event': 'model_run_delete_start',
            'model_run_id': model_run_id,
            'user_id': user_id
        }
    )
    
    # 1. Fetch model run
    try:
        model_run = db.query(ModelRun).filter(
            ModelRun.id == uuid.UUID(model_run_id)
        ).first()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_run_id format: {model_run_id}"
        )
    
    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model run with id {model_run_id} not found"
        )
    
    # 2. Verify user owns this model run through experiment
    experiment = db.query(Experiment).filter(
        Experiment.id == model_run.experiment_id,
        Experiment.user_id == uuid.UUID(user_id)
    ).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this model run"
        )
    
    # Store info for response
    model_type = model_run.model_type
    model_status = model_run.status
    model_artifact_path = model_run.model_artifact_path
    task_id = model_run.run_metadata.get('task_id') if model_run.run_metadata else None
    
    deletion_summary = {
        "model_run_id": model_run_id,
        "model_type": model_type,
        "status": model_status,
        "task_revoked": False,
        "artifact_deleted": False,
        "database_record_deleted": False
    }
    
    # 3. Revoke Celery task if still running
    if task_id and model_status in ['pending', 'running']:
        try:
            from celery.result import AsyncResult
            task_result = AsyncResult(task_id, app=celery_app)
            
            # Check if task is still active
            if task_result.state in ['PENDING', 'STARTED', 'RETRY', 'PROGRESS']:
                task_result.revoke(terminate=True, signal='SIGTERM')
                deletion_summary["task_revoked"] = True
                logger.info(
                    f"Revoked Celery task",
                    extra={
                        'event': 'task_revoked',
                        'task_id': task_id,
                        'model_run_id': model_run_id
                    }
                )
        except Exception as e:
            logger.warning(
                f"Failed to revoke Celery task: {e}",
                extra={
                    'event': 'task_revoke_failed',
                    'task_id': task_id,
                    'model_run_id': model_run_id,
                    'error': str(e)
                }
            )
            # Continue with deletion even if task revocation fails
    
    # 4. Delete model artifact file if it exists
    if model_artifact_path:
        try:
            serialization_service = get_model_serialization_service()
            success = serialization_service.delete_model(
                model_path=model_artifact_path,
                delete_metadata=True
            )
            
            if success:
                deletion_summary["artifact_deleted"] = True
                logger.info(
                    f"Deleted model artifact",
                    extra={
                        'event': 'artifact_deleted',
                        'model_run_id': model_run_id,
                        'artifact_path': model_artifact_path
                    }
                )
            else:
                logger.warning(
                    f"Model artifact not found or already deleted",
                    extra={
                        'event': 'artifact_not_found',
                        'model_run_id': model_run_id,
                        'artifact_path': model_artifact_path
                    }
                )
        except Exception as e:
            logger.error(
                f"Failed to delete model artifact: {e}",
                extra={
                    'event': 'artifact_delete_failed',
                    'model_run_id': model_run_id,
                    'artifact_path': model_artifact_path,
                    'error': str(e)
                },
                exc_info=True
            )
            # Continue with database deletion even if file deletion fails
    
    # 5. Delete ModelRun database record
    try:
        db.delete(model_run)
        db.commit()
        deletion_summary["database_record_deleted"] = True
        
        logger.info(
            f"Deleted model run database record",
            extra={
                'event': 'model_run_deleted',
                'model_run_id': model_run_id,
                'experiment_id': str(experiment.id)
            }
        )
    except Exception as e:
        db.rollback()
        logger.error(
            f"Failed to delete model run from database: {e}",
            extra={
                'event': 'database_delete_failed',
                'model_run_id': model_run_id,
                'error': str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model run: {str(e)}"
        )
    
    # 6. Return success response
    return {
        "message": "Model run deleted successfully",
        "deletion_summary": deletion_summary,
        "timestamp": datetime.utcnow().isoformat()
    }



@router.get("/train/{model_run_id}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_run_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get detailed metrics for a completed model run.
    
    This endpoint returns comprehensive evaluation metrics including:
    - Performance metrics (accuracy, precision, recall, F1, etc.)
    - Training metadata (samples, features, training time)
    - Feature importance (if available)
    - Confusion matrix data (for classification)
    - Error distributions (for regression)
    
    Args:
        model_run_id: UUID of the model run
        db: Database session
        user_id: Current user ID
    
    Returns:
        Detailed metrics and evaluation data
    
    Raises:
        HTTPException 404: If model run not found
        HTTPException 403: If user doesn't have permission
        HTTPException 400: If training not completed
    
    Example:
        GET /api/v1/models/train/abc-123/metrics
    """
    # Fetch model run
    try:
        model_run = db.query(ModelRun).filter(
            ModelRun.id == uuid.UUID(model_run_id)
        ).first()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_run_id format: {model_run_id}"
        )
    
    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model run with id {model_run_id} not found"
        )
    
    # Verify permissions
    experiment = db.query(Experiment).filter(
        Experiment.id == model_run.experiment_id,
        Experiment.user_id == uuid.UUID(user_id)
    ).first()
    
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this model run"
        )
    
    # Check if completed
    if model_run.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model run is not completed yet. Current status: {model_run.status}"
        )
    
    # Check if metrics exist
    if not model_run.metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No metrics available for this model run"
        )
    
    # Build comprehensive metrics response
    response = {
        "model_run_id": str(model_run.id),
        "model_type": model_run.model_type,
        "task_type": _get_task_type_from_model(model_run.model_type),
        "metrics": model_run.metrics,
        "training_metadata": {
            "training_time": model_run.training_time,
            "created_at": model_run.created_at.isoformat(),
            "hyperparameters": model_run.hyperparameters
        }
    }
    
    # Add feature importance if available
    if model_run.run_metadata and 'feature_importance' in model_run.run_metadata:
        response["feature_importance"] = model_run.run_metadata['feature_importance']
    
    # Add training samples info if available in run_metadata
    if model_run.run_metadata:
        if 'train_samples' in model_run.run_metadata:
            response["training_metadata"]["train_samples"] = model_run.run_metadata['train_samples']
        if 'test_samples' in model_run.run_metadata:
            response["training_metadata"]["test_samples"] = model_run.run_metadata['test_samples']
        if 'n_features' in model_run.run_metadata:
            response["training_metadata"]["n_features"] = model_run.run_metadata['n_features']
    
    return response


def _get_task_type_from_model(model_type: str) -> str:
    """
    Determine task type from model type.
    
    Args:
        model_type: Model type identifier
    
    Returns:
        Task type (classification, regression, or clustering)
    """
    # Classification models
    classification_keywords = [
        'classifier', 'classification', 'logistic', 'naive_bayes',
        'svc', 'decision_tree_class', 'random_forest_class', 'gradient_boosting_class',
        'ada_boost_class', 'extra_trees_class', 'xgb_class', 'lgbm_class', 'catboost_class'
    ]
    
    # Regression models
    regression_keywords = [
        'regressor', 'regression', 'linear_reg', 'ridge', 'lasso', 'elastic',
        'svr', 'decision_tree_reg', 'random_forest_reg', 'gradient_boosting_reg',
        'ada_boost_reg', 'extra_trees_reg', 'xgb_reg', 'lgbm_reg', 'catboost_reg'
    ]
    
    # Clustering models
    clustering_keywords = [
        'kmeans', 'dbscan', 'hierarchical', 'agglomerative', 'spectral',
        'mean_shift', 'birch', 'optics', 'gaussian_mixture'
    ]
    
    model_type_lower = model_type.lower()
    
    for keyword in classification_keywords:
        if keyword in model_type_lower:
            return "classification"
    
    for keyword in regression_keywords:
        if keyword in model_type_lower:
            return "regression"
    
    for keyword in clustering_keywords:
        if keyword in model_type_lower:
            return "clustering"
    
    # Default to classification if unknown
    return "unknown"
