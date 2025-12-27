"""
API endpoints for preprocessing pipeline management.

Provides CRUD operations and workflow management for preprocessing pipelines.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from uuid import UUID
import pandas as pd
import time
from pathlib import Path

from app.db.session import get_db
from app.models.preprocessing_pipeline import PreprocessingPipeline
from app.models.dataset import Dataset
from app.models.user import User
from app.schemas.pipeline import (
    PipelineCreate,
    PipelineUpdate,
    PipelineRead,
    PipelineDetail,
    PipelineListResponse,
    PipelineFitRequest,
    PipelineFitResponse,
    PipelineTransformRequest,
    PipelineTransformResponse,
    PipelineFitTransformRequest,
    PipelineFitTransformResponse,
    PipelineExportCodeRequest,
    PipelineExportCodeResponse,
    PipelineCloneRequest,
    PipelineCloneResponse,
    PipelineImportRequest,
    PipelineImportResponse,
    PipelineStepAdd,
    PipelineStepRemove,
    PipelineStepReorder,
    PipelineStatistics,
    PipelineValidationResponse,
)
from app.ml_engine.preprocessing.pipeline import (
    Pipeline,
    export_pipeline_to_sklearn_code,
    export_pipeline_to_standalone_code,
)
from app.utils.logger import get_logger

logger = get_logger("pipeline_api")
router = APIRouter()


# Dependency to get current user (placeholder - implement based on your auth)
def get_current_user(db: Session = Depends(get_db)) -> User:
    """Get current authenticated user."""
    # TODO: Implement proper authentication
    # For now, return first user or create test user
    user = db.query(User).first()
    if not user:
        user = User(email="test@example.com")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def _pipeline_from_db(db_pipeline: PreprocessingPipeline, db: Session) -> Pipeline:
    """
    Reconstruct Pipeline object from database model.

    Args:
        db_pipeline: Database pipeline model
        db: Database session

    Returns:
        Pipeline object

    Raises:
        ValueError: If pipeline configuration is invalid
    """
    config = db_pipeline.config

    # If fitted pipeline exists, load from pickle
    if db_pipeline.fitted and db_pipeline.pickle_path:
        pickle_path = Path(db_pipeline.pickle_path)
        if pickle_path.exists():
            return Pipeline.load(pickle_path)

    # Otherwise, create from configuration
    pipeline = Pipeline.from_dict(config)
    return pipeline


def _save_pipeline_to_db(
    pipeline: Pipeline,
    db_pipeline: PreprocessingPipeline,
    db: Session,
    save_pickle: bool = True
) -> None:
    """
    Save Pipeline object to database model.

    Args:
        pipeline: Pipeline object to save
        db_pipeline: Database pipeline model
        db: Database session
        save_pickle: Whether to save fitted pipeline as pickle
    """
    # Update configuration
    db_pipeline.config = pipeline.to_dict()
    db_pipeline.fitted = pipeline.fitted
    db_pipeline.num_steps = len(pipeline.steps)

    # Update step summary
    db_pipeline.step_summary = [
        {
            "index": idx,
            "name": step.name,
            "class": step.__class__.__name__,
            "fitted": step.fitted,
        }
        for idx, step in enumerate(pipeline.steps)
    ]

    # Update statistics
    if pipeline.fitted:
        db_pipeline.statistics = {
            "step_statistics": pipeline.step_statistics,
            "metadata": pipeline.metadata,
        }
        db_pipeline.fitted_at = pd.Timestamp.now()

    # Save pickle if fitted
    if save_pickle and pipeline.fitted:
        pipelines_dir = Path("data/pipelines")
        pipelines_dir.mkdir(parents=True, exist_ok=True)

        pickle_path = pipelines_dir / f"{db_pipeline.id}.pkl"
        pipeline.save(pickle_path)
        db_pipeline.pickle_path = str(pickle_path)

    db.commit()
    db.refresh(db_pipeline)


# CRUD Endpoints

@router.post("/", response_model=PipelineRead, status_code=status.HTTP_201_CREATED)
def create_pipeline(
    pipeline_data: PipelineCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new preprocessing pipeline."""
    try:
        # Create Pipeline object from configuration
        steps_config = {
            "name": pipeline_data.name,
            "steps": [step.dict() for step in pipeline_data.steps],
            "metadata": {},
            "statistics": [],
        }

        pipeline = Pipeline.from_dict(steps_config)

        # Create database record
        db_pipeline = PreprocessingPipeline(
            user_id=current_user.id,
            dataset_id=pipeline_data.dataset_id,
            name=pipeline_data.name,
            description=pipeline_data.description,
            config=pipeline.to_dict(),
            num_steps=len(pipeline.steps),
            step_summary=[
                {
                    "index": idx,
                    "name": step.name,
                    "class": step.__class__.__name__,
                }
                for idx, step in enumerate(pipeline.steps)
            ],
        )

        db.add(db_pipeline)
        db.commit()
        db.refresh(db_pipeline)

        logger.info(f"Created pipeline {db_pipeline.id} with {len(pipeline.steps)} steps")

        return db_pipeline

    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create pipeline: {str(e)}",
        )


@router.get("/", response_model=PipelineListResponse)
def list_pipelines(
    skip: int = 0,
    limit: int = 100,
    dataset_id: Optional[UUID] = None,
    fitted_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all pipelines for the current user."""
    query = db.query(PreprocessingPipeline).filter(
        PreprocessingPipeline.user_id == current_user.id
    )

    if dataset_id:
        query = query.filter(PreprocessingPipeline.dataset_id == dataset_id)

    if fitted_only:
        query = query.filter(PreprocessingPipeline.fitted == True)

    total = query.count()
    pipelines = query.order_by(desc(PreprocessingPipeline.created_at)).offset(skip).limit(limit).all()

    return PipelineListResponse(total=total, pipelines=pipelines)


@router.get("/{pipeline_id}", response_model=PipelineDetail)
def get_pipeline(
    pipeline_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get detailed information about a specific pipeline."""
    db_pipeline = db.query(PreprocessingPipeline).filter(
        PreprocessingPipeline.id == pipeline_id,
        PreprocessingPipeline.user_id == current_user.id,
    ).first()

    if not db_pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    return db_pipeline


@router.put("/{pipeline_id}", response_model=PipelineRead)
def update_pipeline(
    pipeline_id: UUID,
    pipeline_data: PipelineUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an existing pipeline."""
    db_pipeline = db.query(PreprocessingPipeline).filter(
        PreprocessingPipeline.id == pipeline_id,
        PreprocessingPipeline.user_id == current_user.id,
    ).first()

    if not db_pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    try:
        # Update basic fields
        if pipeline_data.name is not None:
            db_pipeline.name = pipeline_data.name

        if pipeline_data.description is not None:
            db_pipeline.description = pipeline_data.description

        if pipeline_data.dataset_id is not None:
            db_pipeline.dataset_id = pipeline_data.dataset_id

        # Update steps if provided
        if pipeline_data.steps is not None:
            steps_config = {
                "name": db_pipeline.name,
                "steps": [step.dict() for step in pipeline_data.steps],
                "metadata": {},
                "statistics": [],
            }

            pipeline = Pipeline.from_dict(steps_config)
            _save_pipeline_to_db(pipeline, db_pipeline, db, save_pickle=False)

            # Reset fitted state since configuration changed
            db_pipeline.fitted = False
            db_pipeline.fitted_at = None

        db.commit()
        db.refresh(db_pipeline)

        logger.info(f"Updated pipeline {pipeline_id}")

        return db_pipeline

    except Exception as e:
        logger.error(f"Error updating pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update pipeline: {str(e)}",
        )


@router.delete("/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_pipeline(
    pipeline_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a pipeline."""
    db_pipeline = db.query(PreprocessingPipeline).filter(
        PreprocessingPipeline.id == pipeline_id,
        PreprocessingPipeline.user_id == current_user.id,
    ).first()

    if not db_pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    # Delete pickle file if exists
    if db_pipeline.pickle_path:
        pickle_path = Path(db_pipeline.pickle_path)
        if pickle_path.exists():
            pickle_path.unlink()

    db.delete(db_pipeline)
    db.commit()

    logger.info(f"Deleted pipeline {pipeline_id}")


# Workflow Endpoints

@router.post("/{pipeline_id}/fit", response_model=PipelineFitResponse)
def fit_pipeline(
    pipeline_id: UUID,
    request: PipelineFitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fit a pipeline on a dataset."""
    db_pipeline = db.query(PreprocessingPipeline).filter(
        PreprocessingPipeline.id == pipeline_id,
        PreprocessingPipeline.user_id == current_user.id,
    ).first()

    if not db_pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    # Load dataset
    dataset = db.query(Dataset).filter(
        Dataset.id == request.dataset_id,
        Dataset.user_id == current_user.id,
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {request.dataset_id} not found",
        )

    try:
        # Load data
        df = pd.read_csv(dataset.file_path)

        # Create pipeline from config
        pipeline = _pipeline_from_db(db_pipeline, db)

        # Fit pipeline
        start_time = time.time()
        pipeline.fit(df)
        duration = time.time() - start_time

        # Save fitted pipeline
        if request.save_fitted:
            _save_pipeline_to_db(pipeline, db_pipeline, db, save_pickle=True)

        logger.info(f"Fitted pipeline {pipeline_id} in {duration:.2f}s")

        return PipelineFitResponse(
            success=True,
            message=f"Pipeline fitted successfully in {duration:.2f}s",
            pipeline_id=pipeline_id,
            fitted=True,
            statistics=pipeline.get_step_statistics(),
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error(f"Error fitting pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fit pipeline: {str(e)}",
        )


@router.post("/{pipeline_id}/transform", response_model=PipelineTransformResponse)
def transform_with_pipeline(
    pipeline_id: UUID,
    request: PipelineTransformRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Transform a dataset using a fitted pipeline."""
    db_pipeline = db.query(PreprocessingPipeline).filter(
        PreprocessingPipeline.id == pipeline_id,
        PreprocessingPipeline.user_id == current_user.id,
    ).first()

    if not db_pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    if not db_pipeline.fitted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pipeline must be fitted before transform",
        )

    # Load dataset
    dataset = db.query(Dataset).filter(
        Dataset.id == request.dataset_id,
        Dataset.user_id == current_user.id,
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {request.dataset_id} not found",
        )

    try:
        # Load data
        df = pd.read_csv(dataset.file_path)
        original_shape = list(df.shape)

        # Load fitted pipeline
        pipeline = _pipeline_from_db(db_pipeline, db)

        # Transform
        start_time = time.time()
        df_transformed = pipeline.transform(df)
        duration = time.time() - start_time

        transformed_shape = list(df_transformed.shape)

        # Save output if requested
        output_dataset_id = None
        if request.save_output:
            output_name = request.output_name or f"{dataset.name}_transformed"
            output_path = Path(dataset.file_path).parent / f"{output_name}.csv"
            df_transformed.to_csv(output_path, index=False)

            # Create dataset record
            output_dataset = Dataset(
                user_id=current_user.id,
                name=output_name,
                file_path=str(output_path),
                rows=transformed_shape[0],
                cols=transformed_shape[1],
            )
            db.add(output_dataset)
            db.commit()
            db.refresh(output_dataset)
            output_dataset_id = output_dataset.id

        # Generate preview
        preview = df_transformed.head(10).to_dict('records')

        logger.info(f"Transformed dataset with pipeline {pipeline_id} in {duration:.2f}s")

        return PipelineTransformResponse(
            success=True,
            message=f"Dataset transformed successfully in {duration:.2f}s",
            original_shape=original_shape,
            transformed_shape=transformed_shape,
            output_dataset_id=output_dataset_id,
            preview=preview,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error(f"Error transforming with pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transform dataset: {str(e)}",
        )


@router.post("/{pipeline_id}/fit-transform", response_model=PipelineFitTransformResponse)
def fit_transform_pipeline(
    pipeline_id: UUID,
    request: PipelineFitTransformRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fit pipeline and transform dataset in one operation."""
    # This combines fit and transform operations
    # Implementation omitted for brevity - similar to fit + transform above
    pass


# Export Endpoints

@router.post("/{pipeline_id}/export-code", response_model=PipelineExportCodeResponse)
def export_pipeline_code(
    pipeline_id: UUID,
    request: PipelineExportCodeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export pipeline as executable Python code."""
    db_pipeline = db.query(PreprocessingPipeline).filter(
        PreprocessingPipeline.id == pipeline_id,
        PreprocessingPipeline.user_id == current_user.id,
    ).first()

    if not db_pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    try:
        pipeline = _pipeline_from_db(db_pipeline, db)

        if request.format == "sklearn":
            code = export_pipeline_to_sklearn_code(
                pipeline,
                include_imports=request.include_imports,
                include_comments=request.include_comments,
            )
        elif request.format == "standalone":
            code = export_pipeline_to_standalone_code(
                pipeline,
                include_imports=request.include_imports,
                include_comments=request.include_comments,
            )
        else:
            raise ValueError(f"Unknown export format: {request.format}")

        logger.info(f"Exported pipeline {pipeline_id} as {request.format} code")

        return PipelineExportCodeResponse(
            success=True,
            code=code,
            format=request.format,
            pipeline_name=pipeline.name,
        )

    except Exception as e:
        logger.error(f"Error exporting pipeline code: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export pipeline: {str(e)}",
        )


# Additional endpoints (clone, import, step management, etc.) can be added here
# Implementations omitted for brevity but follow similar patterns
