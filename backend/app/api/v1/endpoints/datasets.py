# Dataset upload, preview, stats endpoints

import os
import uuid
import pandas as pd
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetRead

router = APIRouter()

# Mock authentication - replace with actual auth later
def get_current_user_id() -> str:
    """Mock function to get current user ID. Replace with actual auth."""
    return "00000000-0000-0000-0000-000000000001"


@router.post("/upload", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Upload a dataset file (CSV, XLSX, XLS, JSON).

    - Validates file type and size
    - Extracts metadata (rows, columns, dtypes, missing values)
    - Saves file to uploads/{user_id}/{dataset_id}/
    - Creates database record
    """

    # Validate file extension
    allowed_extensions = {".csv", ".xlsx", ".xls", ".json"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Validate file size
    if file_size > settings.MAX_UPLOAD_SIZE:
        max_size_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {max_size_mb}MB"
        )

    # Generate unique dataset ID
    dataset_id = str(uuid.uuid4())

    # Create user-specific upload directory
    upload_path = Path(settings.UPLOAD_DIR) / user_id / dataset_id
    upload_path.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = upload_path / file.filename
    with open(file_path, "wb") as f:
        f.write(content)

    # Extract metadata using pandas
    try:
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif file_ext == ".json":
            df = pd.read_json(file_path)

        # Calculate statistics
        row_count = len(df)
        column_count = len(df.columns)

        # Data types as dict
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Missing values per column
        missing_values = {col: int(df[col].isna().sum()) for col in df.columns}

    except Exception as e:
        # Clean up file if metadata extraction fails
        file_path.unlink()
        upload_path.rmdir()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process file: {str(e)}"
        )

    # Create database record
    dataset = Dataset(
        id=dataset_id,
        user_id=user_id,
        name=Path(file.filename).stem,  # filename without extension
        file_path=str(file_path),
        rows=row_count,
        cols=column_count,
        dtypes=dtypes,
        missing_values=missing_values
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset


@router.get("/", response_model=List[DatasetRead])
async def list_datasets(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
    skip: int = 0,
    limit: int = 100
):
    """
    List all datasets for the current user.

    - Supports pagination with skip and limit
    - Returns datasets ordered by created_at (newest first)
    """
    datasets = db.query(Dataset).filter(
        Dataset.user_id == user_id
    ).order_by(
        Dataset.created_at.desc()
    ).offset(skip).limit(limit).all()

    return datasets


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get a specific dataset by ID.

    - Returns 404 if dataset not found or doesn't belong to user
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == user_id
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    return dataset


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Delete a dataset.

    - Removes database record
    - Deletes file from storage
    - Returns 404 if dataset not found or doesn't belong to user
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == user_id
    ).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    # Delete file from storage
    try:
        file_path = Path(dataset.file_path)
        if file_path.exists():
            file_path.unlink()
            # Try to remove parent directory if empty
            try:
                file_path.parent.rmdir()
            except OSError:
                pass  # Directory not empty or other OS error
    except Exception as e:
        # Log error but continue with database deletion
        print(f"Warning: Failed to delete file {dataset.file_path}: {str(e)}")

    # Delete database record
    db.delete(dataset)
    db.commit()

    return None
