"""
Model training endpoints.

Provides REST API for machine learning model operations including
listing available models, training, evaluation, and prediction.
"""

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, Query
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from uuid import uuid4, UUID
from datetime import datetime
import uuid

from app.ml_engine.model_registry import model_registry, TaskType, ModelCategory, ModelRegistry
from app.models.model_run import ModelRun
from app.models.experiment import Experiment, ExperimentStatus
from app.models.dataset import Dataset
from app.schemas.model import (
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelTrainingStatus,
    ModelRunDeletionResponse,
    ModelMetricsResponse,
    FeatureImportanceResponse,
    FeatureImportanceItem,
    CompareModelsRequest,
    ModelComparisonResponse,
    ModelRankingRequest,
    ModelRankingResponse
)
from app.tasks.training_tasks import train_model, run_training_logic
from app.celery_app import celery_app
from app.db.session import SessionLocal, get_db
from app.services.training_validation_service import get_training_validator, ValidationError
from app.utils.cache import cache_service, CacheKeys, CacheTTL, invalidate_model_cache, invalidate_comparison_cache
from app.services.storage_service import get_model_serialization_service
from app.services.model_comparison_service import ModelComparisonService
from app.utils.logger import get_logger
from app.utils.cache import cache_service, CacheKeys, CacheTTL
from app.core.config import settings

from sqlalchemy.exc import ProgrammingError, OperationalError
from app.models.user import User

router = APIRouter()

logger = get_logger(__name__)


# Helper functions
def get_current_user_id() -> str:
    """
    Get current user ID from authentication context.
    For now, return a fixed ID or guest user ID.
    Note: Ideally this should use the shared get_user_or_guest logic, but for now 
    we'll stick to a hardcoded logic or update to match other endpoints.
    """
    # This function is used when we don't need valid db checking
    return "00000000-0000-0000-0000-000000000001" 

async def get_user_or_guest(
    # request: Request,
    db: Session = Depends(get_db)
) -> str:
    """
    Get the current authenticated user ID, or create/return a guest user ID.
    """
    # 2. Fallback to guest
    try:
        guest_email = "guest@aiplayground.local"
        # Check if table exists by trying to query
        try:
            guest_user = db.query(User).filter(User.email == guest_email).first()
        except (ProgrammingError, OperationalError) as e:
            logger.error(f"Database error (missing table?): {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Database error: Tables might be missing. Please run migrations. Error: {str(e)}"
            )

        if not guest_user:
            logger.info("Creating guest user...")
            
            # Workaround for passlib/bcrypt compatibility issue
            import bcrypt
            hashed_pwd = bcrypt.hashpw(b"guest_password", bcrypt.gensalt()).decode('utf-8')
            
            guest_user = User(
                email=guest_email,
                password_hash=hashed_pwd,
                is_active=True
            )
            db.add(guest_user)
            db.commit()
            db.refresh(guest_user)
        
        return str(guest_user.id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get/create guest user: {e}")
        # Fallback UUID if everything fails (but unlikely to work with FKs)
        return "00000000-0000-0000-0000-000000000001"


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/")
async def list_models(
    skip: int = 0,
    limit: int = 100,
    dataset_id: Optional[str] = None,
    status: Optional[str] = Query(None, description="Filter by status (completed, failed, running)"),
    user_id: str = Depends(get_user_or_guest),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    List trained models (model runs).

    Query parameters:
        skip: Number of records to skip
        limit: Max number of records to return
        dataset_id: Filter by dataset ID
        status: Filter by model status (default: show only completed)

    Returns:
        List of model runs with their status and metrics
    """
    query = db.query(ModelRun).join(Experiment).filter(Experiment.user_id == user_id)
    
    if dataset_id:
        query = query.filter(Experiment.dataset_id == dataset_id)
    
    # Filter by status - default to showing only completed models
    if status is None:
        # Default: only show completed models with valid metrics
        query = query.filter(ModelRun.status == "completed")
    elif status != "all":
        query = query.filter(ModelRun.status == status)
    
    models = query.order_by(ModelRun.created_at.desc()).offset(skip).limit(limit).all()
    
    result = []
    for model in models:
        # Get task_type from metadata if available, otherwise infer from model_type
        task_type = None
        if model.run_metadata and 'task_type' in model.run_metadata:
            task_type = model.run_metadata['task_type']
        else:
            # Fallback: Try to determine from model registry
            try:
                from app.ml_engine.model_registry import model_registry
                model_info = model_registry.get_model(model.model_type)
                if model_info:
                    task_type = model_info.task_type.value
            except Exception:
                pass
        
        # Get metrics, handling None and ensuring valid values
        raw_metrics = model.metrics if model.metrics is not None else {}
        
        # Helper to check if value is valid (not None, not NaN, not inf)
        def is_valid_metric(val):
            if val is None:
                return False
            try:
                import math
                float_val = float(val)
                return not (math.isnan(float_val) or math.isinf(float_val))
            except (ValueError, TypeError):
                return False
        
        # Helper to get safe metric value (return 0.0 if invalid for failed models)
        def safe_metric(val):
            return float(val) if is_valid_metric(val) else 0.0
        
        # Create clean metrics dict with valid values (0.0 for missing/invalid)
        metrics = {}
        accuracy = None
        
        if task_type == "regression":
            # Extract and validate regression metrics - provide 0.0 for invalid values
            # Handle both 'r2' (from dataclass) and 'r2_score' (legacy) key names
            # Use explicit None check to allow 0.0 values
            r2_raw = raw_metrics.get("r2") if "r2" in raw_metrics else raw_metrics.get("r2_score")
            r2_value = safe_metric(r2_raw)
            metrics["r2_score"] = r2_value
            metrics["mae"] = safe_metric(raw_metrics.get("mae"))
            metrics["mse"] = safe_metric(raw_metrics.get("mse"))
            metrics["rmse"] = safe_metric(raw_metrics.get("rmse"))
            metrics["explained_variance"] = safe_metric(raw_metrics.get("explained_variance"))
            
            # Use r2_score as primary accuracy metric for regression
            accuracy = r2_value if r2_value != 0.0 else None
            
        elif task_type == "classification":
            # Extract and validate classification metrics
            metrics["accuracy"] = safe_metric(raw_metrics.get("accuracy"))
            metrics["precision"] = safe_metric(raw_metrics.get("precision"))
            metrics["recall"] = safe_metric(raw_metrics.get("recall"))
            metrics["f1_score"] = safe_metric(raw_metrics.get("f1_score"))
            
            # Use accuracy as primary metric
            accuracy = metrics.get("accuracy") if metrics.get("accuracy") != 0.0 else None
            
        else:
            # For clustering or unknown, copy all valid metrics
            for key, val in raw_metrics.items():
                metrics[key] = safe_metric(val)
            # Try to find any valid metric for accuracy
            accuracy = metrics.get("silhouette_score") or metrics.get("r2_score") or metrics.get("accuracy")
            
        result.append({
            "id": str(model.id),
            "name": f"{model.model_type} ({model.created_at.strftime('%Y-%m-%d %H:%M')})",
            "type": model.model_type,
            "task_type": task_type,
            "status": model.status,
            "accuracy": accuracy,
            "createdAt": model.created_at.isoformat(),
            "updatedAt": model.created_at.isoformat(),
            "hyperparameters": model.hyperparameters or {},
            "metrics": metrics
        })
        
    return result


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



class DummyTaskContext:
    """Mock Celery task context for running tasks synchronously without Redis."""
    def __init__(self, task_id):
        self.request = type('Request', (), {'id': task_id})()
    
    def update_state(self, state, meta=None, **kwargs):
        pass

def run_training_in_background(
    task_id: str,
    model_run_id: str,
    experiment_id: str,
    dataset_id: str,
    model_type: str,
    hyperparameters: Optional[Dict[str, Any]],
    target_column: Optional[str],
    feature_columns: Optional[list],
    test_size: float,
    random_state: int,
    user_id: str
):
    """Wrapper to run training task in FastAPI background tasks."""
    dummy_self = DummyTaskContext(task_id)
    try:
        run_training_logic(
            dummy_self,
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
    except Exception as e:
        logger.error(f"Background training failed: {e}")


@router.post("/train", response_model=ModelTrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model_endpoint(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_user_or_guest)
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
        # Handle playground/dummy experiment ID (00000000-0000-0000-0000-000000000000)
        # Check if the UUID is nil/zero
        experiment_id = request.experiment_id
        if str(experiment_id) == "00000000-0000-0000-0000-000000000000":
            # Check if there is an existing "Playground" experiment for this dataset/user or create new
            # For simplicity, we create a new experiment for this run
            new_experiment_id = uuid4()
            logger.info(f"Creating new playground experiment: {new_experiment_id}")
            
            new_experiment = Experiment(
                id=new_experiment_id,
                user_id=user_id,
                dataset_id=request.dataset_id,
                name=f"Playground Experiment {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                status=ExperimentStatus.RUNNING
            )
            db.add(new_experiment)
            db.commit()
            experiment_id = new_experiment_id
            
        # Validate training configuration using validation service
        validator = get_training_validator(db)
        experiment, dataset, model_info = validator.validate_training_config(
            experiment_id=experiment_id,
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
        experiment_id=experiment_id,
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

    # Trigger training task
    # Use FastAPI BackgroundTasks if configured (e.g. Render Free Tier) or fallback if Celery is disabled
    # Otherwise use Celery for robust queuing
    task_id = str(uuid4())
    
    if settings.USE_BACKGROUND_TASKS:
        logger.info("Triggering training using FastAPI BackgroundTasks")
        background_tasks.add_task(
            run_training_in_background,
            task_id=task_id,
            model_run_id=str(model_run.id),
            experiment_id=str(experiment_id),
            dataset_id=str(request.dataset_id),
            model_type=request.model_type,
            hyperparameters=request.hyperparameters,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            test_size=request.test_size,
            random_state=request.random_state,
            user_id=user_id
        )
    else:
        logger.info("Triggering training using Celery")
        task = train_model.delay(
            model_run_id=str(model_run.id),
            experiment_id=str(experiment_id),
            dataset_id=str(request.dataset_id),
            model_type=request.model_type,
            hyperparameters=request.hyperparameters,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            test_size=request.test_size,
            random_state=request.random_state,
            user_id=user_id
        )
        task_id = task.id

    # Store task_id in model_run run_metadata
    if not model_run.run_metadata:
        model_run.run_metadata = {}
    model_run.run_metadata['task_id'] = task_id
    db.commit()

    return ModelTrainingResponse(
        model_run_id=model_run.id,
        task_id=task_id,
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
        
        # Invalidate related caches
        await invalidate_model_cache(model_run_id)
        await invalidate_comparison_cache()
        
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
    use_cache: bool = True,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get detailed metrics for a completed model run.
    
    Results are cached for 1 hour to improve response times.
    
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
        GET /api/v1/models/train/abc-123/metrics?use_cache=false
    """
    # Try cache first
    if use_cache:
        cache_key = CacheKeys.model_metrics(model_run_id)
        cached_result = cache_service.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached metrics for model {model_run_id}")
            return cached_result
    
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
    
    # Cache the response for 1 hour
    if use_cache:
        cache_key = CacheKeys.model_metrics(model_run_id)
        cache_service.set(cache_key, response, CacheTTL.LONG)
        logger.info(f"Cached metrics for model {model_run_id}")
    
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


def _get_importance_method(model_type: str) -> str:
    """
    Determine the method used to calculate feature importance.
    
    Args:
        model_type: Model type identifier
    
    Returns:
        Method name (e.g., 'feature_importances_', 'coef_', 'permutation')
    """
    model_type_lower = model_type.lower()
    
    # Tree-based models use feature_importances_
    tree_based = ['forest', 'tree', 'xgb', 'lgbm', 'catboost', 'gradient_boosting', 'ada_boost', 'extra_trees']
    if any(keyword in model_type_lower for keyword in tree_based):
        return "feature_importances_"
    
    # Linear models use coef_
    linear_models = ['linear', 'ridge', 'lasso', 'elastic', 'logistic', 'sgd']
    if any(keyword in model_type_lower for keyword in linear_models):
        return "coef_"
    
    # SVM models use coef_ (for linear kernel)
    if 'svm' in model_type_lower or 'svc' in model_type_lower or 'svr' in model_type_lower:
        return "coef_"
    
    return "unknown"


@router.get("/train/{model_run_id}/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    model_run_id: str,
    top_n: Optional[int] = None,
    use_cache: bool = True,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get feature importance for a completed model run.
    
    This endpoint returns feature importance scores that indicate which features
    had the most impact on the model's predictions. Feature importance is available
    for models that support it (tree-based models, linear models, etc.).
    
    Cached for 30 minutes to improve performance.
    
    **Supported Models:**
    - Tree-based: Random Forest, Decision Tree, Gradient Boosting, XGBoost, LightGBM, CatBoost
    - Linear: Linear Regression, Ridge, Lasso, Logistic Regression
    - SVM: Support Vector Machines (with linear kernel)
    
    **Not Supported:**
    - K-Nearest Neighbors
    - Naive Bayes
    - Clustering models (K-Means, DBSCAN, etc.)
    
    Args:
        model_run_id: UUID of the model run
        top_n: Optional number of top features to return (default: all features)
        db: Database session
        user_id: Current user ID
    
    Returns:
        Feature importance data with rankings and scores
    
    Raises:
        HTTPException 404: If model run not found or no feature importance available
        HTTPException 403: If user doesn't have permission
        HTTPException 400: If training not completed or model doesn't support feature importance
    
    Example:
        GET /api/v1/models/train/abc-123/feature-importance
        GET /api/v1/models/train/abc-123/feature-importance?top_n=10
    """
    logger.info(
        f"Feature importance requested",
        extra={
            'event': 'feature_importance_request',
            'model_run_id': model_run_id,
            'top_n': top_n,
            'user_id': user_id
        }
    )
    
    # Check cache first
    if use_cache:
        cache_key = CacheKeys.feature_importance(model_run_id, top_n)
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for feature importance {model_run_id}")
            return cached_result
    
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
    
    # Get task type
    task_type = _get_task_type_from_model(model_run.model_type)
    
    # Check if feature importance exists in run_metadata
    feature_importance_dict = None
    if model_run.run_metadata and 'feature_importance' in model_run.run_metadata:
        feature_importance_dict = model_run.run_metadata['feature_importance']
    
    # If no feature importance available
    if not feature_importance_dict:
        logger.warning(
            f"No feature importance available",
            extra={
                'event': 'feature_importance_not_available',
                'model_run_id': model_run_id,
                'model_type': model_run.model_type
            }
        )
        
        # Determine why feature importance is not available
        unsupported_models = ['knn', 'k_nearest', 'naive_bayes', 'kmeans', 'dbscan', 'hierarchical']
        is_unsupported = any(keyword in model_run.model_type.lower() for keyword in unsupported_models)
        
        if is_unsupported:
            message = f"Model type '{model_run.model_type}' does not support feature importance"
        else:
            message = "Feature importance was not calculated during training"
        
        return FeatureImportanceResponse(
            model_run_id=str(model_run.id),
            model_type=model_run.model_type,
            task_type=task_type,
            has_feature_importance=False,
            feature_importance=None,
            feature_importance_dict=None,
            total_features=0,
            top_features=None,
            importance_method=None,
            message=message
        )
    
    # Convert dict to sorted list of FeatureImportanceItem
    feature_importance_list = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in feature_importance_dict.items()
    ]
    
    # Sort by importance (descending)
    feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)
    
    # Add rank
    for rank, item in enumerate(feature_importance_list, start=1):
        item['rank'] = rank
    
    # Convert to FeatureImportanceItem objects
    feature_importance_items = [
        FeatureImportanceItem(**item) for item in feature_importance_list
    ]
    
    # Get top N features if requested
    top_features = None
    if top_n is not None and top_n > 0:
        top_features = feature_importance_items[:top_n]
    else:
        # Default: top 10 or all if less than 10
        top_features = feature_importance_items[:min(10, len(feature_importance_items))]
    
    # Determine importance method
    importance_method = _get_importance_method(model_run.model_type)
    
    logger.info(
        f"Feature importance retrieved",
        extra={
            'event': 'feature_importance_retrieved',
            'model_run_id': model_run_id,
            'total_features': len(feature_importance_items),
            'top_n_requested': top_n,
            'importance_method': importance_method
        }
    )
    
    response = FeatureImportanceResponse(
        model_run_id=str(model_run.id),
        model_type=model_run.model_type,
        task_type=task_type,
        has_feature_importance=True,
        feature_importance=feature_importance_items,
        feature_importance_dict=feature_importance_dict,
        total_features=len(feature_importance_items),
        top_features=top_features,
        importance_method=importance_method,
        message=None
    )
    
    # Cache the result
    if use_cache:
        cache_key = CacheKeys.feature_importance(model_run_id, top_n)
        await cache_service.set(cache_key, response, ttl=CacheTTL.MEDIUM)
    
    return response


# ============================================================================
# Model Comparison Endpoints
# ============================================================================


@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    request: CompareModelsRequest,
    use_cache: bool = True,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Compare multiple model runs across various metrics.
    
    This endpoint allows you to:
    - Compare 2-10 model runs simultaneously
    - Auto-detect appropriate comparison metrics based on task type
    - Rank models by a primary metric (or auto-detect best metric)
    - Get statistical summaries for each metric
    - Receive recommendations for model selection
    
    Args:
        request: Comparison request with model run IDs and options
        db: Database session
        user_id: Current user ID
    
    Returns:
        Comprehensive comparison report with rankings and recommendations
    
    Raises:
        HTTPException 400: If models are not comparable or invalid request
        HTTPException 404: If model runs not found
    
    Example:
        POST /api/v1/models/compare
        {
            "model_run_ids": ["abc-123", "def-456", "ghi-789"],
            "ranking_criteria": "f1_score"
        }
    """
    # Try cache first
    if use_cache:
        cache_key = CacheKeys.model_comparison([str(mid) for mid in request.model_run_ids])
        cached_result = cache_service.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached comparison for {len(request.model_run_ids)} models")
            return cached_result
    
    try:
        comparison_service = ModelComparisonService(db)
        result = comparison_service.compare_models(request, user_id)
        
        logger.info(
            f"Model comparison completed",
            extra={
                'event': 'models_compared',
                'comparison_id': result.comparison_id,
                'total_models': result.total_models,
                'best_model': result.best_model.model_run_id,
                'ranking_criteria': result.ranking_criteria
            }
        )
        
        # Cache the comparison result for 30 minutes
        if use_cache:
            cache_key = CacheKeys.model_comparison([str(mid) for mid in request.model_run_ids])
            # Convert to dict for caching
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
            cache_service.set(cache_key, result_dict, CacheTTL.MEDIUM)
            logger.info(f"Cached comparison for {len(request.model_run_ids)} models")
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in model comparison: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error comparing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare models: {str(e)}"
        )


@router.post("/rank", response_model=ModelRankingResponse)
async def rank_models(
    request: ModelRankingRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Rank models using custom weighted criteria.
    
    This endpoint allows you to:
    - Rank up to 20 models using custom metric weights
    - Specify which metrics should be maximized vs minimized
    - Get composite scores and individual contributions
    - Understand relative performance differences
    
    The composite score is calculated as a weighted sum of normalized metrics.
    All metrics are normalized to 0-1 range before weighting.
    
    Args:
        request: Ranking request with model IDs and weights
        db: Database session
        user_id: Current user ID
    
    Returns:
        Ranked models with composite scores and contributions
    
    Raises:
        HTTPException 400: If weights invalid or models not found
        HTTPException 404: If model runs not found
    
    Example:
        POST /api/v1/models/rank
        {
            "model_run_ids": ["abc-123", "def-456"],
            "ranking_weights": {
                "f1_score": 0.5,
                "precision": 0.3,
                "recall": 0.2
            }
        }
    """
    try:
        comparison_service = ModelComparisonService(db)
        result = comparison_service.rank_models(request, user_id)
        
        logger.info(
            f"Model ranking completed",
            extra={
                'event': 'models_ranked',
                'ranking_id': result.ranking_id,
                'total_models': len(result.ranked_models),
                'best_model': result.best_model.model_run_id if result.best_model else None,
                'weights': result.ranking_weights
            }
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in model ranking: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error ranking models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rank models: {str(e)}"
        )


@router.get("/runs", response_model=List[Dict[str, Any]])
async def list_model_runs(
    experiment_id: Optional[UUID] = None,
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    List model runs with optional filtering.
    
    This endpoint allows you to:
    - List all model runs for an experiment
    - Filter by status (completed, failed, running, etc.)
    - Filter by model type
    - Paginate results
    
    Args:
        experiment_id: Filter by experiment ID
        status: Filter by status (completed, failed, running, pending, cancelled)
        model_type: Filter by model type
        limit: Maximum number of results (default 50, max 100)
        offset: Offset for pagination
        db: Database session
        user_id: Current user ID
    
    Returns:
        List of model runs with metadata
    
    Example:
        GET /api/v1/models/runs?experiment_id=abc-123&status=completed&limit=10
    """
    try:
        query = db.query(ModelRun)
        
        # Apply filters
        if experiment_id:
            query = query.filter(ModelRun.experiment_id == experiment_id)
        
        if status:
            query = query.filter(ModelRun.status == status)
        
        if model_type:
            query = query.filter(ModelRun.model_type == model_type)
        
        # Apply pagination
        limit = min(limit, 100)  # Cap at 100
        query = query.order_by(ModelRun.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        model_runs = query.all()
        
        # Build response
        results = []
        for mr in model_runs:
            results.append({
                "model_run_id": str(mr.id),
                "experiment_id": str(mr.experiment_id),
                "model_type": mr.model_type,
                "status": mr.status,
                "metrics": mr.metrics,
                "hyperparameters": mr.hyperparameters,
                "training_time": mr.training_time,
                "created_at": mr.created_at.isoformat() if mr.created_at else None
            })
        
        logger.info(
            f"Listed {len(results)} model runs",
            extra={
                'event': 'model_runs_listed',
                'total': len(results),
                'experiment_id': str(experiment_id) if experiment_id else None,
                'status_filter': status,
                'model_type_filter': model_type
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error listing model runs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list model runs: {str(e)}"
        )


@router.get("/cache/stats", status_code=status.HTTP_200_OK)
async def get_cache_stats(
    user_id: str = Depends(get_current_user_id)
):
    """
    Get cache statistics and health information.
    
    This endpoint provides insights into the caching system:
    - Cache availability and connection status
    - Memory usage and key counts
    - Hit/miss statistics if available
    - Cache configuration
    
    Args:
        user_id: Current user ID (for authentication)
    
    Returns:
        Cache statistics and health information
    
    Example:
        GET /api/v1/models/cache/stats
        
    Response:
        {
            "available": true,
            "stats": {
                "used_memory": "1.5M",
                "total_keys": 42,
                "hits": 1245,
                "misses": 156
            },
            "config": {
                "ttl_short": 900,
                "ttl_medium": 1800,
                "ttl_long": 3600
            }
        }
    """
    logger.info(
        f"Cache stats requested",
        extra={'event': 'cache_stats_request', 'user_id': user_id}
    )
    
    if not cache_service.is_available():
        return {
            "available": False,
            "message": "Redis cache is not available",
            "stats": None,
            "config": None
        }
    
    try:
        stats = await cache_service.get_stats()
        
        return {
            "available": True,
            "stats": stats,
            "config": {
                "ttl_very_short": CacheTTL.VERY_SHORT,
                "ttl_short": CacheTTL.SHORT,
                "ttl_medium": CacheTTL.MEDIUM,
                "ttl_long": CacheTTL.LONG,
                "ttl_very_long": CacheTTL.VERY_LONG,
                "ttl_day": CacheTTL.DAY,
                "ttl_week": CacheTTL.WEEK
            },
            "message": "Cache is healthy"
        }
    except Exception as e:
        logger.error(
            f"Failed to get cache stats: {e}",
            extra={'event': 'cache_stats_failed', 'error': str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )


@router.delete("/cache/clear", status_code=status.HTTP_200_OK)
async def clear_cache(
    pattern: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
):
    """
    Clear cache entries (admin operation).
    
    This endpoint allows clearing specific cache patterns or all cache entries.
    Use with caution as it will impact system performance temporarily.
    
    Args:
        pattern: Optional pattern to match keys (e.g., 'model:metrics:*')
                If not provided, clears ALL cache entries
        user_id: Current user ID (for authentication)
    
    Returns:
        Number of keys deleted
    
    Example:
        DELETE /api/v1/models/cache/clear
        DELETE /api/v1/models/cache/clear?pattern=model:metrics:*
        
    Response:
        {
            "message": "Cache cleared successfully",
            "deleted_count": 42,
            "pattern": "model:metrics:*"
        }
    """
    logger.info(
        f"Cache clear requested",
        extra={
            'event': 'cache_clear_request',
            'user_id': user_id,
            'pattern': pattern
        }
    )
    
    if not cache_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis cache is not available"
        )
    
    try:
        if pattern:
            deleted_count = await cache_service.delete_pattern(pattern)
            message = f"Cleared {deleted_count} cache entries matching pattern: {pattern}"
        else:
            deleted_count = await cache_service.clear_all()
            message = f"Cleared all {deleted_count} cache entries"
        
        logger.info(
            message,
            extra={
                'event': 'cache_cleared',
                'deleted_count': deleted_count,
                'pattern': pattern
            }
        )
        
        return {
            "message": message,
            "deleted_count": deleted_count,
            "pattern": pattern or "all"
        }
    except Exception as e:
        logger.error(
            f"Failed to clear cache: {e}",
            extra={'event': 'cache_clear_failed', 'error': str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

