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
    ModelTrainingStatus
)
from app.tasks.training_tasks import train_model
from app.celery_app import celery_app
from app.db.session import SessionLocal

router = APIRouter()


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
    """
    # 1. Validate experiment exists and belongs to user
    experiment = db.query(Experiment).filter(
        Experiment.id == request.experiment_id,
        Experiment.user_id == uuid.UUID(user_id)
    ).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {request.experiment_id} not found"
        )

    # 2. Validate dataset exists and belongs to user
    dataset = db.query(Dataset).filter(
        Dataset.id == request.dataset_id,
        Dataset.user_id == uuid.UUID(user_id)
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id {request.dataset_id} not found"
        )

    # 3. Validate model type exists in registry
    registry = ModelRegistry()
    model_info = registry.get_model(request.model_type)
    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model type '{request.model_type}' not found in registry"
        )

    # 4. Validate target column for supervised learning
    if model_info.task_type.value in ['classification', 'regression']:
        if not request.target_column:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="target_column is required for supervised learning tasks"
            )

    # 5. Create ModelRun record
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

    # 6. Update experiment status to running
    if experiment.status != "running":
        experiment.status = "running"
        db.commit()

    # 7. Trigger Celery task
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

    # 8. Store task_id in model_run run_metadata
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

    # 7. If failed, include error from run_metadata
    if model_run.status == "failed" and model_run.run_metadata:
        error_info = model_run.run_metadata.get('error', {})
        error = f"{error_info.get('type', 'Error')}: {error_info.get('message', 'Unknown error')}"

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
