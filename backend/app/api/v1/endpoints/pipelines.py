"""
API endpoints for preprocessing pipeline management.

Provides CRUD operations and workflow management for preprocessing pipelines.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from uuid import UUID
import pandas as pd
import time
from pathlib import Path
import zipfile
import tempfile
import os
from datetime import datetime

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


# Helper functions for file cleanup

def cleanup_old_export_files(export_dir: Path, max_age_hours: int = 24):
    """
    Clean up old export files to prevent disk space issues.

    Args:
        export_dir: Directory containing export files
        max_age_hours: Maximum age of files in hours before deletion
    """
    try:
        if not export_dir.exists():
            return

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for file_path in export_dir.glob("*.zip"):
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old export file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old export file {file_path.name}: {str(e)}")

    except Exception as e:
        logger.error(f"Error during export file cleanup: {str(e)}")


def delete_file_after_delay(file_path: Path, delay_seconds: int = 300):
    """
    Delete a file after a specified delay (background task).

    Args:
        file_path: Path to file to delete
        delay_seconds: Delay before deletion in seconds (default: 5 minutes)
    """
    try:
        time.sleep(delay_seconds)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted export file after download: {file_path.name}")
    except Exception as e:
        logger.warning(f"Failed to delete file {file_path}: {str(e)}")


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


@router.post("/{pipeline_id}/export-code-zip")
def export_pipeline_code_as_zip(
    pipeline_id: UUID,
    request: PipelineExportCodeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Export pipeline as executable Python code in a ZIP file.

    Creates a ZIP archive containing:
    - pipeline.py: The main pipeline code
    - requirements.txt: Python dependencies
    - README.md: Usage instructions
    - config.json: Pipeline configuration (optional)
    """
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

        # Generate code based on format
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

        # Create export directory
        export_dir = Path("data/exports/code")
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate ZIP filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = db_pipeline.name.replace(" ", "_").replace("/", "_")
        zip_filename = f"{safe_name}_{pipeline_id}_{timestamp}.zip"
        zip_path = export_dir / zip_filename

        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add main pipeline code
            zipf.writestr("pipeline.py", code)

            # Add requirements.txt
            if request.format == "sklearn":
                requirements = [
                    "pandas>=1.3.0",
                    "numpy>=1.21.0",
                    "scikit-learn>=1.0.0",
                ]
            else:
                requirements = [
                    "pandas>=1.3.0",
                    "numpy>=1.21.0",
                ]
            zipf.writestr("requirements.txt", "\n".join(requirements))

            # Add README.md
            readme_content = f"""# {db_pipeline.name}

## Description
{db_pipeline.description or 'Exported preprocessing pipeline'}

## Pipeline Information
- **Format**: {request.format}
- **Number of Steps**: {len(pipeline.steps)}
- **Fitted**: {'Yes' if pipeline.fitted else 'No'}
- **Export Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Steps
{chr(10).join(f"{i+1}. {step.name} ({step.__class__.__name__})" for i, step in enumerate(pipeline.steps))}

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from pipeline import pipeline

# Fit the pipeline (if not already fitted)
pipeline.fit(X_train, y_train)

# Transform data
X_transformed = pipeline.transform(X_test)

# Or fit and transform in one step
X_train_transformed = pipeline.fit_transform(X_train, y_train)
```

## Notes
- This pipeline was exported from AI-Playground
- Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}
"""
            zipf.writestr("README.md", readme_content)

            # Add config.json (pipeline configuration)
            import json
            config = pipeline.to_dict()
            zipf.writestr("config.json", json.dumps(config, indent=2, default=str))

        logger.info(f"Exported pipeline {pipeline_id} as ZIP to {zip_path}")

        # Clean up old export files in the background
        background_tasks.add_task(cleanup_old_export_files, export_dir, max_age_hours=24)

        return {
            "success": True,
            "message": "Pipeline code exported as ZIP successfully",
            "filename": zip_filename,
            "download_url": f"/api/v1/preprocessing/{pipeline_id}/download-code-zip/{zip_filename}",
            "file_size": zip_path.stat().st_size,
        }

    except Exception as e:
        logger.error(f"Error exporting pipeline code as ZIP: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export pipeline as ZIP: {str(e)}",
        )


@router.get("/{pipeline_id}/download-code-zip/{filename}")
def download_pipeline_code_zip(
    pipeline_id: UUID,
    filename: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Download a previously exported pipeline code ZIP file.

    The ZIP file is automatically cleaned up 5 minutes after download.
    """
    # Verify pipeline ownership
    db_pipeline = db.query(PreprocessingPipeline).filter(
        PreprocessingPipeline.id == pipeline_id,
        PreprocessingPipeline.user_id == current_user.id,
    ).first()

    if not db_pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    # Construct file path
    export_dir = Path("data/exports/code")
    file_path = export_dir / filename

    # Verify file exists and belongs to this pipeline
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export file not found: {filename}",
        )

    # Verify filename contains pipeline_id for security
    if str(pipeline_id) not in filename:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized access to export file",
        )

    try:
        # Schedule file cleanup after 5 minutes (enough time for download to complete)
        background_tasks.add_task(delete_file_after_delay, file_path, delay_seconds=300)

        # Return file for download
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        logger.error(f"Error downloading ZIP file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download ZIP file: {str(e)}",
        )


@router.get("/{pipeline_id}/export-files")
def list_pipeline_export_files(
    pipeline_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all available export files for a pipeline.

    Useful for debugging or when user needs to re-download a file.
    """
    # Verify pipeline ownership
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
        export_dir = Path("data/exports/code")
        if not export_dir.exists():
            return {
                "success": True,
                "files": [],
                "message": "No export files found"
            }

        # Find all files for this pipeline
        pattern = f"*{pipeline_id}*.zip"
        export_files = []

        for file_path in export_dir.glob(pattern):
            file_stat = file_path.stat()
            export_files.append({
                "filename": file_path.name,
                "size": file_stat.st_size,
                "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "download_url": f"/api/v1/preprocessing/{pipeline_id}/download-code-zip/{file_path.name}"
            })

        # Sort by modification time (newest first)
        export_files.sort(key=lambda x: x["modified_at"], reverse=True)

        return {
            "success": True,
            "pipeline_id": str(pipeline_id),
            "pipeline_name": db_pipeline.name,
            "files": export_files,
            "total_files": len(export_files)
        }

    except Exception as e:
        logger.error(f"Error listing export files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list export files: {str(e)}",
        )


# Additional endpoints (clone, import, step management, etc.) can be added here
# Implementations omitted for brevity but follow similar patterns


# Serialization Endpoints

@router.post("/{pipeline_id}/export", response_model=dict)
def export_pipeline_file(
    pipeline_id: UUID,
    format: str = "pickle",
    compression: str = "none",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Export pipeline to a serialized file.

    Supports multiple formats:
    - pickle: Binary format with full state (fitted parameters)
    - json: Configuration only (can be reconstructed)
    - joblib: Optimized binary format
    - yaml: Human-readable configuration

    Compression options: none, gzip, bz2, lzma
    """
    from app.ml_engine.preprocessing.serializer import PipelineSerializer

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

        # Determine export path
        export_dir = Path("data/exports/pipelines")
        export_dir.mkdir(parents=True, exist_ok=True)

        # Build filename with appropriate extensions
        ext_map = {
            "pickle": "pkl",
            "json": "json",
            "joblib": "joblib",
            "yaml": "yml"
        }
        compress_ext_map = {
            "gzip": ".gz",
            "bz2": ".bz2",
            "lzma": ".xz",
            "none": ""
        }

        filename = f"{db_pipeline.name.replace(' ', '_')}_{pipeline_id}.{ext_map[format]}{compress_ext_map[compression]}"
        export_path = export_dir / filename

        # Serialize
        serializer = PipelineSerializer(default_format=format, compression=compression)
        file_info = serializer.save(
            pipeline,
            export_path,
            metadata={
                "pipeline_id": str(pipeline_id),
                "dataset_id": str(db_pipeline.dataset_id) if db_pipeline.dataset_id else None,
                "user_id": str(current_user.id),
                "description": db_pipeline.description,
            }
        )

        logger.info(f"Exported pipeline {pipeline_id} to {export_path}")

        return {
            "success": True,
            "message": f"Pipeline exported successfully",
            "file_info": file_info,
            "download_path": f"/api/v1/pipelines/{pipeline_id}/download/{filename}"
        }

    except Exception as e:
        logger.error(f"Error exporting pipeline file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export pipeline: {str(e)}",
        )


@router.post("/import", response_model=PipelineRead)
def import_pipeline_file(
    file_path: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    format: Optional[str] = None,
    compression: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Import a pipeline from a serialized file.

    Automatically detects format and compression from file extension.
    Creates a new pipeline entry in the database.
    """
    from app.ml_engine.preprocessing.serializer import PipelineSerializer

    import_path = Path(file_path)

    if not import_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}",
        )

    try:
        # Load pipeline
        serializer = PipelineSerializer()
        loaded_data = serializer.load(import_path, format=format, compression=compression)

        # Extract pipeline (might be wrapped in metadata)
        if isinstance(loaded_data, dict) and "pipeline" in loaded_data:
            pipeline = loaded_data["pipeline"]
            metadata = loaded_data.get("metadata", {})
        else:
            pipeline = loaded_data
            metadata = {}

        # If it's a config dict (from JSON/YAML), reconstruct pipeline
        if isinstance(pipeline, dict):
            pipeline = Pipeline.from_dict(pipeline)

        # Create database entry
        pipeline_name = name or metadata.get("name") or pipeline.name or import_path.stem
        db_pipeline = PreprocessingPipeline(
            user_id=current_user.id,
            name=pipeline_name,
            description=description or metadata.get("description") or f"Imported from {import_path.name}",
            config=pipeline.to_dict(),
            fitted=pipeline.fitted if hasattr(pipeline, "fitted") else False,
            num_steps=len(pipeline.steps) if hasattr(pipeline, "steps") else 0,
        )

        # Save fitted pipeline if applicable
        if db_pipeline.fitted:
            pickle_dir = Path("data/pipelines")
            pickle_dir.mkdir(parents=True, exist_ok=True)
            pickle_path = pickle_dir / f"{db_pipeline.id}.pkl"
            pipeline.save(pickle_path)
            db_pipeline.pickle_path = str(pickle_path)

        db.add(db_pipeline)
        db.commit()
        db.refresh(db_pipeline)

        logger.info(f"Imported pipeline from {import_path} as {db_pipeline.id}")

        return PipelineRead.from_orm(db_pipeline)

    except Exception as e:
        logger.error(f"Error importing pipeline: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import pipeline: {str(e)}",
        )


@router.get("/{pipeline_id}/export-config", response_model=dict)
def export_pipeline_config(
    pipeline_id: UUID,
    format: str = "json",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Export pipeline configuration (without fitted parameters).

    Returns the configuration as JSON or YAML that can be used
    to reconstruct the pipeline structure.
    """
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
        config = db_pipeline.config

        if format.lower() == "yaml":
            try:
                import yaml
                config_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="YAML export requires pyyaml package"
                )
        else:  # JSON
            import json
            config_str = json.dumps(config, indent=2)

        return {
            "success": True,
            "pipeline_id": str(pipeline_id),
            "pipeline_name": db_pipeline.name,
            "format": format,
            "config": config,
            "config_string": config_str
        }

    except Exception as e:
        logger.error(f"Error exporting config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export config: {str(e)}",
        )


@router.post("/import-config", response_model=PipelineRead)
def import_pipeline_config(
    config: dict,
    name: Optional[str] = None,
    description: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Import a pipeline from a configuration dictionary.

    Accepts a configuration dict (from to_dict() format) and
    creates a new unfitted pipeline.
    """
    try:
        # Reconstruct pipeline from config
        pipeline = Pipeline.from_dict(config)

        # Create database entry
        pipeline_name = name or config.get("name") or "Imported Pipeline"
        db_pipeline = PreprocessingPipeline(
            user_id=current_user.id,
            name=pipeline_name,
            description=description or "Imported from configuration",
            config=config,
            fitted=False,  # Config import creates unfitted pipeline
            num_steps=len(config.get("steps", [])),
        )

        db.add(db_pipeline)
        db.commit()
        db.refresh(db_pipeline)

        logger.info(f"Imported pipeline config as {db_pipeline.id}")

        return PipelineRead.from_orm(db_pipeline)

    except Exception as e:
        logger.error(f"Error importing config: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import config: {str(e)}",
        )
