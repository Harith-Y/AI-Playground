"""
Preprocessing steps CRUD endpoints

This module provides REST API endpoints for managing preprocessing pipeline steps.
Users can create, read, update, delete, and reorder preprocessing steps for their datasets.
"""

import uuid
import os
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.db.session import get_db
from app.models.preprocessing_step import PreprocessingStep
from app.models.dataset import Dataset
from app.schemas.preprocessing import (
    PreprocessingStepCreate,
    PreprocessingStepUpdate,
    PreprocessingStepRead,
    PreprocessingApplyRequest,
    PreprocessingApplyResponse,
    PreprocessingAsyncResponse,
    PreprocessingTaskStatus,
)
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler, RobustScaler
from app.ml_engine.preprocessing.cleaner import IQROutlierDetector, ZScoreOutlierDetector
from app.tasks.preprocessing_tasks import apply_preprocessing_pipeline as apply_preprocessing_task
from celery.result import AsyncResult

router = APIRouter()


# Mock authentication - replace with actual auth later
def get_current_user_id() -> str:
    """Mock function to get current user ID. Replace with actual auth."""
    return "00000000-0000-0000-0000-000000000001"


@router.post(
    "/",
    response_model=PreprocessingStepRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a preprocessing step",
    description="Create a new preprocessing step for a dataset. Steps are executed in order.",
    responses={
        201: {
            "description": "Preprocessing step created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "dataset_id": "789e4567-e89b-12d3-a456-426614174000",
                        "step_type": "missing_value_imputation",
                        "parameters": {"strategy": "mean"},
                        "column_name": "age",
                        "order": 1
                    }
                }
            }
        },
        404: {"description": "Dataset not found"},
        403: {"description": "Not authorized to access this dataset"},
    }
)
async def create_preprocessing_step(
    step: PreprocessingStepCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Create a new preprocessing step.

    **Step Types:**
    - `missing_value_imputation`: Fill missing values (mean, median, mode, constant)
    - `scaling`: Scale numeric features (standard, minmax, robust)
    - `encoding`: Encode categorical features (onehot, label, ordinal)
    - `outlier_detection`: Detect/remove outliers (iqr, zscore)
    - `feature_selection`: Select important features (variance, correlation, mutual_info)
    - `transformation`: Transform features (log, sqrt, power, box-cox)

    **Parameters Examples:**
    - Imputation: `{"strategy": "mean"}` or `{"strategy": "constant", "fill_value": 0}`
    - Scaling: `{"method": "standard"}` or `{"method": "minmax", "feature_range": [0, 1]}`
    - Encoding: `{"method": "onehot"}` or `{"method": "label"}`
    - Outlier: `{"method": "iqr", "threshold": 1.5}` or `{"method": "zscore", "threshold": 3}`
    """
    # Verify dataset exists and belongs to user
    dataset = db.query(Dataset).filter(
        and_(Dataset.id == step.dataset_id, Dataset.user_id == user_id)
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {step.dataset_id} not found or access denied"
        )

    # If order not specified, add to end
    if step.order is None:
        max_order = db.query(PreprocessingStep).filter(
            PreprocessingStep.dataset_id == step.dataset_id
        ).count()
        step.order = max_order

    # Create preprocessing step
    db_step = PreprocessingStep(
        id=uuid.uuid4(),
        dataset_id=step.dataset_id,
        step_type=step.step_type,
        parameters=step.parameters,
        column_name=step.column_name,
        order=step.order
    )

    db.add(db_step)
    db.commit()
    db.refresh(db_step)

    return db_step


@router.get(
    "/",
    response_model=List[PreprocessingStepRead],
    summary="List preprocessing steps",
    description="Get all preprocessing steps, optionally filtered by dataset",
    responses={
        200: {
            "description": "List of preprocessing steps",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "123e4567-e89b-12d3-a456-426614174000",
                            "dataset_id": "789e4567-e89b-12d3-a456-426614174000",
                            "step_type": "missing_value_imputation",
                            "parameters": {"strategy": "mean"},
                            "column_name": "age",
                            "order": 0
                        },
                        {
                            "id": "456e4567-e89b-12d3-a456-426614174000",
                            "dataset_id": "789e4567-e89b-12d3-a456-426614174000",
                            "step_type": "scaling",
                            "parameters": {"method": "standard"},
                            "column_name": None,
                            "order": 1
                        }
                    ]
                }
            }
        }
    }
)
async def list_preprocessing_steps(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    List all preprocessing steps.

    If `dataset_id` is provided, returns only steps for that dataset.
    Steps are returned in execution order (sorted by `order` field).
    """
    query = db.query(PreprocessingStep).join(Dataset).filter(Dataset.user_id == user_id)

    if dataset_id:
        query = query.filter(PreprocessingStep.dataset_id == dataset_id)

    steps = query.order_by(PreprocessingStep.order).all()
    return steps


@router.get(
    "/{step_id}",
    response_model=PreprocessingStepRead,
    summary="Get preprocessing step",
    description="Get a specific preprocessing step by ID",
    responses={
        200: {"description": "Preprocessing step details"},
        404: {"description": "Preprocessing step not found"},
        403: {"description": "Not authorized to access this step"},
    }
)
async def get_preprocessing_step(
    step_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """Get a specific preprocessing step by ID."""
    step = db.query(PreprocessingStep).join(Dataset).filter(
        and_(
            PreprocessingStep.id == step_id,
            Dataset.user_id == user_id
        )
    ).first()

    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preprocessing step {step_id} not found or access denied"
        )

    return step


@router.put(
    "/{step_id}",
    response_model=PreprocessingStepRead,
    summary="Update preprocessing step",
    description="Update a preprocessing step's configuration",
    responses={
        200: {"description": "Preprocessing step updated successfully"},
        404: {"description": "Preprocessing step not found"},
        403: {"description": "Not authorized to access this step"},
    }
)
async def update_preprocessing_step(
    step_id: str,
    step_update: PreprocessingStepUpdate,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Update a preprocessing step.

    Only provided fields will be updated. Omitted fields remain unchanged.

    **Examples:**
    - Change imputation strategy: `{"parameters": {"strategy": "median"}}`
    - Change step order: `{"order": 2}`
    - Apply to different column: `{"column_name": "income"}`
    """
    # Get step and verify ownership
    db_step = db.query(PreprocessingStep).join(Dataset).filter(
        and_(
            PreprocessingStep.id == step_id,
            Dataset.user_id == user_id
        )
    ).first()

    if not db_step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preprocessing step {step_id} not found or access denied"
        )

    # Update fields
    update_data = step_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_step, field, value)

    db.commit()
    db.refresh(db_step)

    return db_step


@router.delete(
    "/{step_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete preprocessing step",
    description="Delete a preprocessing step",
    responses={
        204: {"description": "Preprocessing step deleted successfully"},
        404: {"description": "Preprocessing step not found"},
        403: {"description": "Not authorized to access this step"},
    }
)
async def delete_preprocessing_step(
    step_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Delete a preprocessing step.

    The step is permanently removed from the pipeline.
    Consider reordering remaining steps after deletion.
    """
    # Get step and verify ownership
    db_step = db.query(PreprocessingStep).join(Dataset).filter(
        and_(
            PreprocessingStep.id == step_id,
            Dataset.user_id == user_id
        )
    ).first()

    if not db_step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preprocessing step {step_id} not found or access denied"
        )

    db.delete(db_step)
    db.commit()

    return None


@router.post(
    "/reorder",
    response_model=List[PreprocessingStepRead],
    summary="Reorder preprocessing steps",
    description="Update the execution order of preprocessing steps for a dataset",
    responses={
        200: {
            "description": "Steps reordered successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "123e4567-e89b-12d3-a456-426614174000",
                            "dataset_id": "789e4567-e89b-12d3-a456-426614174000",
                            "step_type": "outlier_detection",
                            "parameters": {"method": "iqr"},
                            "column_name": None,
                            "order": 0
                        },
                        {
                            "id": "456e4567-e89b-12d3-a456-426614174000",
                            "dataset_id": "789e4567-e89b-12d3-a456-426614174000",
                            "step_type": "missing_value_imputation",
                            "parameters": {"strategy": "mean"},
                            "column_name": "age",
                            "order": 1
                        }
                    ]
                }
            }
        },
        400: {"description": "Invalid reorder request"},
        404: {"description": "Dataset or steps not found"},
    }
)
async def reorder_preprocessing_steps(
    dataset_id: str = Query(..., description="Dataset ID"),
    step_ids: List[str] = Query(..., description="Ordered list of step IDs"),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Reorder preprocessing steps for a dataset.

    Provide the step IDs in the desired execution order.

    **Example:**
    ```
    POST /preprocessing/reorder?dataset_id=789e4567...&step_ids=456e4567...&step_ids=123e4567...
    ```

    This will set the first step's order to 0, the second to 1, etc.
    """
    # Verify dataset exists and belongs to user
    dataset = db.query(Dataset).filter(
        and_(Dataset.id == dataset_id, Dataset.user_id == user_id)
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found or access denied"
        )

    # Get all steps for the dataset
    steps = db.query(PreprocessingStep).filter(
        PreprocessingStep.dataset_id == dataset_id
    ).all()

    # Verify all step IDs exist
    step_dict = {str(step.id): step for step in steps}
    for step_id in step_ids:
        if step_id not in step_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Step {step_id} not found in dataset {dataset_id}"
            )

    # Update order
    for new_order, step_id in enumerate(step_ids):
        step_dict[step_id].order = new_order

    db.commit()

    # Return reordered steps
    updated_steps = db.query(PreprocessingStep).filter(
        PreprocessingStep.dataset_id == dataset_id
    ).order_by(PreprocessingStep.order).all()

    return updated_steps


@router.post(
    "/apply",
    response_model=PreprocessingApplyResponse,
    summary="Apply preprocessing pipeline",
    description="Execute all preprocessing steps on a dataset in order",
    responses={
        200: {
            "description": "Preprocessing pipeline applied successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Successfully applied 3 preprocessing steps",
                        "steps_applied": 3,
                        "original_shape": [1000, 10],
                        "transformed_shape": [950, 12],
                        "output_dataset_id": "new-dataset-uuid",
                        "preview": [
                            {"age": 25, "income": 50000, "education": 1},
                            {"age": 30, "income": 60000, "education": 2}
                        ],
                        "statistics": {
                            "rows_before": 1000,
                            "rows_after": 950,
                            "columns_before": 10,
                            "columns_after": 12,
                            "missing_values_filled": 50,
                            "outliers_removed": 50
                        }
                    }
                }
            }
        },
        404: {"description": "Dataset not found"},
        400: {"description": "No preprocessing steps configured or invalid configuration"},
    }
)
async def apply_preprocessing_pipeline(
    request: PreprocessingApplyRequest = Body(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Apply the complete preprocessing pipeline to a dataset.

    This endpoint:
    1. Loads the original dataset
    2. Executes all preprocessing steps in order
    3. Optionally saves the transformed dataset
    4. Returns preview and statistics

    **Preprocessing Steps Supported:**
    - `missing_value_imputation`: Fill missing values
    - `scaling`: Normalize/standardize numeric features
    - `encoding`: Encode categorical features
    - `outlier_detection`: Detect and handle outliers
    - `feature_selection`: Select important features
    - `transformation`: Apply mathematical transformations

    **Parameters:**
    - `dataset_id`: UUID of the dataset to preprocess
    - `save_output`: Whether to save the transformed dataset (default: True)
    - `output_name`: Name for the transformed dataset (default: "{original_name}_preprocessed")
    """
    # Verify dataset exists and belongs to user
    dataset = db.query(Dataset).filter(
        and_(Dataset.id == request.dataset_id, Dataset.user_id == user_id)
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {request.dataset_id} not found or access denied"
        )

    # Get all preprocessing steps for the dataset
    steps = db.query(PreprocessingStep).filter(
        PreprocessingStep.dataset_id == request.dataset_id
    ).order_by(PreprocessingStep.order).all()

    if not steps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No preprocessing steps configured for this dataset"
        )

    try:
        # Load the dataset
        df = pd.read_csv(dataset.file_path)
        original_shape = list(df.shape)

        # Track statistics
        stats = {
            "rows_before": df.shape[0],
            "columns_before": df.shape[1],
            "missing_values_filled": 0,
            "outliers_removed": 0,
            "features_scaled": 0,
            "features_encoded": 0,
        }

        # Apply each preprocessing step in order
        for step in steps:
            df, step_stats = _apply_single_step(df, step)

            # Update statistics
            for key, value in step_stats.items():
                if key in stats:
                    stats[key] += value

        transformed_shape = list(df.shape)
        stats["rows_after"] = df.shape[0]
        stats["columns_after"] = df.shape[1]

        # Save transformed dataset if requested
        output_dataset_id = None
        if request.save_output:
            output_name = request.output_name or f"{dataset.name}_preprocessed"
            output_path = _save_transformed_dataset(df, output_name, dataset.file_path)

            # Create new dataset record
            new_dataset = Dataset(
                id=uuid.uuid4(),
                user_id=user_id,
                name=output_name,
                file_path=output_path,
                rows=df.shape[0],
                cols=df.shape[1],
                dtypes={col: str(df[col].dtype) for col in df.columns},
                missing_values={col: int(df[col].isna().sum()) for col in df.columns}
            )
            db.add(new_dataset)
            db.commit()
            db.refresh(new_dataset)
            output_dataset_id = new_dataset.id

        # Generate preview (first 5 rows)
        preview = df.head(5).to_dict('records')

        return PreprocessingApplyResponse(
            success=True,
            message=f"Successfully applied {len(steps)} preprocessing steps",
            steps_applied=len(steps),
            original_shape=original_shape,
            transformed_shape=transformed_shape,
            output_dataset_id=output_dataset_id,
            preview=preview,
            statistics=stats
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying preprocessing pipeline: {str(e)}"
        )


def _apply_single_step(df: pd.DataFrame, step: PreprocessingStep) -> tuple[pd.DataFrame, dict]:
    """
    Apply a single preprocessing step to the dataframe.

    Returns:
        Tuple of (transformed_df, statistics_dict)
    """
    step_type = step.step_type
    params = step.parameters or {}
    column_name = step.column_name
    stats = {}

    if step_type == "missing_value_imputation":
        strategy = params.get("strategy", "mean")

        if strategy == "mean":
            imputer = MeanImputer(columns=[column_name] if column_name else None)
        elif strategy == "median":
            imputer = MedianImputer(columns=[column_name] if column_name else None)
        elif strategy in ["mode", "constant", "ffill", "bfill"]:
            # Fallback to pandas fillna for other strategies
            if column_name:
                missing_count = df[column_name].isna().sum()
                if strategy == "mode":
                    df[column_name] = df[column_name].fillna(df[column_name].mode()[0] if not df[column_name].mode().empty else 0)
                elif strategy == "constant":
                    df[column_name] = df[column_name].fillna(params.get("fill_value", 0))
                elif strategy == "ffill":
                    df[column_name] = df[column_name].fillna(method='ffill')
                elif strategy == "bfill":
                    df[column_name] = df[column_name].fillna(method='bfill')
                stats["missing_values_filled"] = int(missing_count)
            return df, stats
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        missing_before = df[column_name].isna().sum() if column_name else df.isna().sum().sum()
        df = imputer.fit_transform(df)
        stats["missing_values_filled"] = int(missing_before)

    elif step_type == "scaling":
        method = params.get("method", "standard")

        if method == "standard":
            scaler = StandardScaler(columns=[column_name] if column_name else None)
        elif method == "minmax":
            scaler = MinMaxScaler(
                columns=[column_name] if column_name else None,
                feature_range=(
                    params.get("feature_range_min", 0),
                    params.get("feature_range_max", 1)
                )
            )
        elif method == "robust":
            scaler = RobustScaler(columns=[column_name] if column_name else None)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        df = scaler.fit_transform(df)
        stats["features_scaled"] = 1 if column_name else len(df.select_dtypes(include=[np.number]).columns)

    elif step_type == "outlier_detection":
        method = params.get("method", "iqr")
        threshold = params.get("threshold", 1.5)
        action = params.get("action", "flag")

        if method == "iqr":
            detector = IQROutlierDetector(
                columns=[column_name] if column_name else None,
                threshold=threshold
            )
        elif method == "zscore":
            detector = ZScoreOutlierDetector(
                columns=[column_name] if column_name else None,
                threshold=threshold
            )
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        rows_before = len(df)
        outlier_mask = detector.fit_transform(df)

        if action == "remove":
            df = df[~outlier_mask.any(axis=1)]
            stats["outliers_removed"] = rows_before - len(df)
        elif action == "cap":
            # Cap outliers to threshold values
            for col in outlier_mask.columns:
                if outlier_mask[col].any():
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    df[col] = df[col].clip(lower, upper)
        # If action == "flag", just mark them (outlier_mask is created)

    elif step_type == "encoding":
        # Placeholder for encoding - would need encoder implementation
        method = params.get("method", "onehot")
        if column_name and method == "label":
            df[column_name] = pd.factorize(df[column_name])[0]
            stats["features_encoded"] = 1
        elif column_name and method == "onehot":
            dummies = pd.get_dummies(df[column_name], prefix=column_name)
            df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
            stats["features_encoded"] = len(dummies.columns)

    elif step_type == "feature_selection":
        # Placeholder for feature selection
        method = params.get("method", "variance_threshold")
        threshold = params.get("threshold", 0.01)

        if method == "variance_threshold":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            variances = df[numeric_cols].var()
            low_variance_cols = variances[variances < threshold].index.tolist()
            df = df.drop(columns=low_variance_cols)
            stats["features_removed"] = len(low_variance_cols)

    elif step_type == "transformation":
        # Placeholder for transformation
        method = params.get("method", "log")

        if column_name and method == "log":
            df[column_name] = np.log1p(df[column_name])
        elif column_name and method == "sqrt":
            df[column_name] = np.sqrt(df[column_name])

    return df, stats


def _save_transformed_dataset(df: pd.DataFrame, name: str, original_path: str) -> str:
    """
    Save the transformed dataset to disk.

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(original_path).parent / "preprocessed"
    output_dir.mkdir(exist_ok=True)

    # Generate output filename
    output_path = output_dir / f"{name}.csv"

    # Save the dataset
    df.to_csv(output_path, index=False)

    return str(output_path)


# ============================================================================
# ASYNC PREPROCESSING ENDPOINTS (Celery-based)
# ============================================================================


@router.post(
    "/apply/async",
    response_model=PreprocessingAsyncResponse,
    summary="Apply preprocessing pipeline asynchronously",
    description="Start a Celery task to apply preprocessing pipeline in the background",
    responses={
        202: {
            "description": "Preprocessing task started successfully",
            "content": {
                "application/json": {
                    "example": {
                        "task_id": "550e8400-e29b-41d4-a716-446655440000",
                        "message": "Preprocessing task started",
                        "status": "PENDING"
                    }
                }
            }
        },
        404: {"description": "Dataset not found"},
        400: {"description": "No preprocessing steps configured"},
    }
)
async def apply_preprocessing_pipeline_async(
    request: PreprocessingApplyRequest = Body(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Apply preprocessing pipeline asynchronously using Celery.

    This endpoint is recommended for large datasets or complex pipelines
    that may take longer than 30 seconds to process.

    **Workflow:**
    1. Submit preprocessing job → Receive task_id
    2. Poll /preprocessing/task/{task_id} for status
    3. When status is SUCCESS, retrieve results

    **Advantages over /apply:**
    - No timeout issues for long-running operations
    - Progress tracking (0-100%)
    - Can check status from multiple clients
    - Task continues even if client disconnects

    **Parameters:**
    - `dataset_id`: UUID of the dataset to preprocess
    - `save_output`: Whether to save the transformed dataset (default: True)
    - `output_name`: Name for the transformed dataset
    """
    # Verify dataset exists and belongs to user
    dataset = db.query(Dataset).filter(
        and_(Dataset.id == request.dataset_id, Dataset.user_id == user_id)
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {request.dataset_id} not found or access denied"
        )

    # Verify preprocessing steps exist
    steps_count = db.query(PreprocessingStep).filter(
        PreprocessingStep.dataset_id == request.dataset_id
    ).count()

    if steps_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No preprocessing steps configured for this dataset"
        )

    # Start Celery task
    task = apply_preprocessing_task.delay(
        dataset_id=str(request.dataset_id),
        user_id=user_id,
        save_output=request.save_output,
        output_name=request.output_name
    )

    return PreprocessingAsyncResponse(
        task_id=task.id,
        message="Preprocessing task started successfully",
        status="PENDING"
    )


@router.get(
    "/task/{task_id}",
    response_model=PreprocessingTaskStatus,
    summary="Get preprocessing task status",
    description="Check the status and progress of an async preprocessing task",
    responses={
        200: {
            "description": "Task status retrieved successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "pending": {
                            "summary": "Task pending",
                            "value": {
                                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                                "state": "PENDING",
                                "status": "Task is waiting to be processed",
                                "progress": 0
                            }
                        },
                        "in_progress": {
                            "summary": "Task in progress",
                            "value": {
                                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                                "state": "PROGRESS",
                                "status": "Applying step 2/5: scaling",
                                "progress": 45,
                                "current_step": "scaling"
                            }
                        },
                        "success": {
                            "summary": "Task completed",
                            "value": {
                                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                                "state": "SUCCESS",
                                "status": "Preprocessing completed successfully",
                                "progress": 100,
                                "result": {
                                    "success": True,
                                    "message": "Successfully applied 5 preprocessing steps",
                                    "steps_applied": 5,
                                    "original_shape": [1000, 10],
                                    "transformed_shape": [950, 12]
                                }
                            }
                        },
                        "failure": {
                            "summary": "Task failed",
                            "value": {
                                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                                "state": "FAILURE",
                                "error": "ValueError: Invalid imputation strategy",
                                "progress": 0
                            }
                        }
                    }
                }
            }
        },
        404: {"description": "Task not found"},
    }
)
async def get_preprocessing_task_status(task_id: str):
    """
    Get the status of a preprocessing task.

    **Task States:**
    - `PENDING`: Task is waiting to start
    - `STARTED`: Task has started but no progress yet
    - `PROGRESS`: Task is running (check `progress` field for %)
    - `SUCCESS`: Task completed successfully (check `result` field)
    - `FAILURE`: Task failed (check `error` field)

    **Progress Tracking:**
    - 0%: Task queued
    - 5-10%: Loading dataset
    - 10-80%: Applying preprocessing steps (incremental)
    - 85%: Saving transformed dataset
    - 95%: Generating preview
    - 100%: Complete

    **Polling Recommendation:**
    - Poll every 1-2 seconds while state is PENDING/STARTED/PROGRESS
    - Stop polling when state is SUCCESS or FAILURE
    """
    task_result = AsyncResult(task_id)

    if not task_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    response = PreprocessingTaskStatus(
        task_id=task_id,
        state=task_result.state
    )

    if task_result.state == 'PENDING':
        response.status = "Task is waiting to be processed"
        response.progress = 0

    elif task_result.state == 'STARTED':
        response.status = "Task has started"
        response.progress = 5

    elif task_result.state == 'PROGRESS':
        info = task_result.info or {}
        response.status = info.get('status', 'Processing...')
        response.progress = info.get('progress', 0)
        response.current_step = info.get('current_step')

    elif task_result.state == 'SUCCESS':
        response.status = "Preprocessing completed successfully"
        response.progress = 100
        response.result = task_result.result

    elif task_result.state == 'FAILURE':
        info = task_result.info or {}
        response.error = str(info.get('error', task_result.info)) if isinstance(info, dict) else str(task_result.info)
        response.status = "Task failed"
        response.progress = 0

    return response
