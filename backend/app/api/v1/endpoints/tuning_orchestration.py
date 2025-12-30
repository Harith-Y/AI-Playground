"""
Tuning Orchestration API Endpoints.

Provides REST API for complex hyperparameter tuning workflows:
- Progressive search (grid → random → bayesian)
- Multi-model parallel comparison
- Workflow status and management
"""

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.session import SessionLocal
from app.services.tuning_orchestration_service import (
    TuningOrchestrationService,
    ProgressiveSearchConfig,
    MultiModelConfig
)
from app.schemas.tuning_orchestration import (
    ProgressiveSearchRequest,
    ProgressiveSearchResponse,
    MultiModelComparisonRequest,
    MultiModelComparisonResponse,
    OrchestrationStatusResponse,
    BestModelResponse,
    TriggerNextStageRequest,
    NextStageResponse
)
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# Helper functions
def get_current_user_id() -> str:
    """Mock function to get current user ID."""
    # TODO: Replace with actual authentication
    return "00000000-0000-0000-0000-000000000001"


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post(
    "/progressive-search",
    response_model=ProgressiveSearchResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def start_progressive_search(
    request: ProgressiveSearchRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Initiate progressive hyperparameter search workflow.
    
    **Progressive Search Workflow:**
    1. **Grid Search**: Exhaustive exploration of initial parameter space
    2. **Random Search**: Exploration around promising regions (automatically triggered)
    3. **Bayesian Optimization**: Fine-tuning in best region (automatically triggered)
    
    Each stage automatically refines the parameter space based on previous results,
    progressively narrowing the search for optimal performance.
    
    **Benefits:**
    - Combines thoroughness of grid search with efficiency of Bayesian optimization
    - Automatic parameter space refinement between stages
    - No manual intervention needed between stages
    - Reduces risk of missing optimal parameters
    
    **Use Cases:**
    - Complex models with large parameter spaces
    - When you want both exploration and exploitation
    - Production models requiring careful tuning
    
    Args:
        request: Progressive search configuration
        db: Database session
        user_id: Current user ID
    
    Returns:
        ProgressiveSearchResponse with orchestration ID and stage details
    
    Raises:
        HTTPException 404: If model run not found
        HTTPException 400: If model run not completed
    
    Example:
        POST /api/v1/tuning-orchestration/progressive-search
        {
            "model_run_id": "abc-123",
            "initial_param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20]
            },
            "refinement_factor": 0.3,
            "cv_folds": 5,
            "scoring_metric": "accuracy",
            "n_iter_random": 50,
            "n_iter_bayesian": 30
        }
    """
    logger.info(
        f"Progressive search requested for model_run_id={request.model_run_id}",
        extra={
            'event': 'progressive_search_request',
            'model_run_id': str(request.model_run_id),
            'user_id': user_id
        }
    )
    
    try:
        # Create service
        service = TuningOrchestrationService(db)
        
        # Create config
        config = ProgressiveSearchConfig(
            initial_method="grid_search",
            intermediate_method="random_search",
            final_method="bayesian",
            initial_param_grid=request.initial_param_grid,
            refinement_factor=request.refinement_factor,
            cv_folds=request.cv_folds,
            scoring_metric=request.scoring_metric,
            n_iter_random=request.n_iter_random,
            n_iter_bayesian=request.n_iter_bayesian
        )
        
        # Execute progressive search
        result = service.progressive_search(
            model_run_id=request.model_run_id,
            config=config,
            user_id=user_id
        )
        
        logger.info(
            f"Progressive search initiated",
            extra={
                'event': 'progressive_search_initiated',
                'orchestration_id': result['orchestration_id']
            }
        )
        
        return ProgressiveSearchResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error starting progressive search: {e}",
            extra={
                'event': 'progressive_search_error',
                'error': str(e)
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting progressive search: {str(e)}"
        )


@router.post(
    "/multi-model-comparison",
    response_model=MultiModelComparisonResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def start_multi_model_comparison(
    request: MultiModelComparisonRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Initiate parallel hyperparameter tuning across multiple models.
    
    **Multi-Model Comparison:**
    - Tune multiple models simultaneously (if parallel=true)
    - Use same tuning method and settings for fair comparison
    - Identify best performing model automatically
    - Optional: Specify different parameter grids per model
    
    **Benefits:**
    - Compare multiple algorithms efficiently
    - Parallel execution saves time
    - Fair comparison with consistent settings
    - Automatic best model selection
    
    **Use Cases:**
    - Model selection for a dataset
    - Comparing ensemble methods
    - A/B testing different algorithms
    - Finding optimal model type
    
    Args:
        request: Multi-model comparison configuration
        db: Database session
        user_id: Current user ID
    
    Returns:
        MultiModelComparisonResponse with orchestration ID and task details
    
    Raises:
        HTTPException 404: If any model run not found
        HTTPException 400: If any model run not completed
    
    Example:
        POST /api/v1/tuning-orchestration/multi-model-comparison
        {
            "model_run_ids": [
                "abc-123",
                "def-456",
                "ghi-789"
            ],
            "tuning_method": "bayesian",
            "cv_folds": 5,
            "scoring_metric": "accuracy",
            "n_iter": 30,
            "parallel": true
        }
    """
    logger.info(
        f"Multi-model comparison requested for {len(request.model_run_ids)} models",
        extra={
            'event': 'multi_model_comparison_request',
            'n_models': len(request.model_run_ids),
            'user_id': user_id
        }
    )
    
    try:
        # Create service
        service = TuningOrchestrationService(db)
        
        # Convert param_grids keys from strings to UUIDs if needed
        param_grids = {}
        if request.param_grids:
            for model_id_str, grid in request.param_grids.items():
                param_grids[UUID(model_id_str)] = grid
        
        # Create config
        config = MultiModelConfig(
            model_run_ids=request.model_run_ids,
            tuning_method=request.tuning_method,
            param_grids=param_grids,
            cv_folds=request.cv_folds,
            scoring_metric=request.scoring_metric,
            n_iter=request.n_iter,
            parallel=request.parallel
        )
        
        # Execute multi-model comparison
        result = service.multi_model_comparison(
            config=config,
            user_id=user_id
        )
        
        logger.info(
            f"Multi-model comparison initiated",
            extra={
                'event': 'multi_model_comparison_initiated',
                'orchestration_id': result['orchestration_id'],
                'n_models': result['n_models']
            }
        )
        
        return MultiModelComparisonResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error starting multi-model comparison: {e}",
            extra={
                'event': 'multi_model_comparison_error',
                'error': str(e)
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting multi-model comparison: {str(e)}"
        )


@router.get(
    "/orchestration/{orchestration_id}/status",
    response_model=OrchestrationStatusResponse
)
async def get_orchestration_status(
    orchestration_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get the status of an orchestration workflow.
    
    Returns comprehensive status including:
    - Overall workflow progress
    - Status of each stage/model
    - Completion percentage
    - Best scores found so far
    
    **Status Values:**
    - PENDING: Not yet started
    - RUNNING: Currently executing
    - COMPLETED: Successfully finished
    - FAILED: Error occurred
    
    Args:
        orchestration_id: UUID of the orchestration
        db: Database session
        user_id: Current user ID
    
    Returns:
        OrchestrationStatusResponse with detailed status
    
    Raises:
        HTTPException 404: If orchestration not found
    
    Example:
        GET /api/v1/tuning-orchestration/orchestration/abc-123/status
    """
    logger.info(
        f"Orchestration status requested for {orchestration_id}",
        extra={
            'event': 'orchestration_status_request',
            'orchestration_id': orchestration_id,
            'user_id': user_id
        }
    )
    
    try:
        service = TuningOrchestrationService(db)
        status_info = service.get_orchestration_status(orchestration_id)
        
        return OrchestrationStatusResponse(**status_info)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error getting orchestration status: {e}",
            extra={
                'event': 'orchestration_status_error',
                'error': str(e)
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting orchestration status: {str(e)}"
        )


@router.get(
    "/orchestration/{orchestration_id}/best-model",
    response_model=BestModelResponse
)
async def get_best_model_from_comparison(
    orchestration_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get the best performing model from a multi-model comparison.
    
    Returns:
    - Best model details (ID, type, score, parameters)
    - Ranking of all models
    - Tuning method used
    
    **Note:** Only works for completed multi-model comparison orchestrations.
    
    Args:
        orchestration_id: UUID of the multi-model comparison
        db: Database session
        user_id: Current user ID
    
    Returns:
        BestModelResponse with best model and rankings
    
    Raises:
        HTTPException 404: If orchestration not found or no completed runs
        HTTPException 400: If orchestration not completed yet
    
    Example:
        GET /api/v1/tuning-orchestration/orchestration/abc-123/best-model
    """
    logger.info(
        f"Best model requested for orchestration {orchestration_id}",
        extra={
            'event': 'best_model_request',
            'orchestration_id': orchestration_id,
            'user_id': user_id
        }
    )
    
    try:
        service = TuningOrchestrationService(db)
        result = service.get_best_model_from_comparison(orchestration_id)
        
        logger.info(
            f"Best model identified",
            extra={
                'event': 'best_model_identified',
                'orchestration_id': orchestration_id,
                'best_model_id': result['best_model']['model_run_id'],
                'best_score': result['best_model']['best_score']
            }
        )
        
        return BestModelResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error getting best model: {e}",
            extra={
                'event': 'best_model_error',
                'error': str(e)
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting best model: {str(e)}"
        )


@router.post(
    "/trigger-next-stage",
    response_model=NextStageResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def trigger_next_stage(
    request: TriggerNextStageRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Manually trigger the next stage of a progressive search workflow.
    
    **Note:** This is typically handled automatically, but this endpoint
    allows manual control for debugging or custom workflows.
    
    Args:
        request: Next stage trigger request
        db: Database session
        user_id: Current user ID
    
    Returns:
        NextStageResponse with next stage details
    
    Raises:
        HTTPException 404: If tuning run not found
        HTTPException 400: If tuning run not completed
    
    Example:
        POST /api/v1/tuning-orchestration/trigger-next-stage
        {
            "orchestration_id": "abc-123",
            "completed_tuning_run_id": "def-456"
        }
    """
    logger.info(
        f"Next stage trigger requested",
        extra={
            'event': 'next_stage_trigger_request',
            'orchestration_id': request.orchestration_id,
            'completed_tuning_run_id': str(request.completed_tuning_run_id),
            'user_id': user_id
        }
    )
    
    try:
        service = TuningOrchestrationService(db)
        result = service.trigger_next_stage(
            orchestration_id=request.orchestration_id,
            completed_tuning_run_id=request.completed_tuning_run_id,
            user_id=user_id
        )
        
        if result:
            message = f"Next stage ({result['stage']}) triggered successfully"
            logger.info(
                f"Next stage triggered",
                extra={
                    'event': 'next_stage_triggered',
                    'orchestration_id': request.orchestration_id,
                    'next_stage': result['stage']
                }
            )
            return NextStageResponse(
                **result,
                message=message
            )
        else:
            logger.info(
                f"No next stage found, workflow complete",
                extra={
                    'event': 'workflow_complete',
                    'orchestration_id': request.orchestration_id
                }
            )
            return NextStageResponse(
                message="Workflow complete, no next stage"
            )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error triggering next stage: {e}",
            extra={
                'event': 'next_stage_trigger_error',
                'error': str(e)
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error triggering next stage: {str(e)}"
        )
