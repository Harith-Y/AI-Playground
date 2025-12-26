"""
Preprocessing steps CRUD endpoints

This module provides REST API endpoints for managing preprocessing pipeline steps.
Users can create, read, update, delete, and reorder preprocessing steps for their datasets.
"""

import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.db.session import get_db
from app.models.preprocessing_step import PreprocessingStep
from app.models.dataset import Dataset
from app.schemas.preprocessing import (
    PreprocessingStepCreate,
    PreprocessingStepUpdate,
    PreprocessingStepRead,
)

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
