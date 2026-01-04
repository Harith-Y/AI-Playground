# Dataset upload, preview, stats endpoints

import io
import os
import uuid as uuid_lib
import pandas as pd
from pathlib import Path
from typing import List
from uuid import UUID
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_user_id, verify_resource_ownership, decode_token, get_password_hash
from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.user import User
from app.services.r2_storage_service import get_storage_service
from app.schemas.dataset import (
    DatasetRead,
    DatasetPreviewResponse,
    DatasetStatsResponse,
    ColumnInfo,
)

router = APIRouter()


async def get_user_or_guest(
    request: Request,
    db: Session = Depends(get_db)
) -> UUID:
    """
    Get the current authenticated user ID, or create/return a guest user ID.
    This allows the playground to work without forcing login, while still supporting auth.
    """
    # 1. Try to extract token
    authorization: str = request.headers.get("Authorization")
    if authorization:
        scheme, _, param = authorization.partition(" ")
        if scheme.lower() == "bearer":
            try:
                payload = decode_token(param)
                user_id = payload.get("sub")
                if user_id:
                    return UUID(user_id)
            except Exception:
                pass # Token invalid, fall back to guest

    # 2. Fallback to guest
    guest_email = "guest@aiplayground.local"
    guest_user = db.query(User).filter(User.email == guest_email).first()
    if not guest_user:
        guest_user = User(
            email=guest_email,
            password_hash=get_password_hash("guest_password"),
            is_active=True
        )
        db.add(guest_user)
        db.commit()
        db.refresh(guest_user)
    
    return guest_user.id


@router.post(
    "/upload",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a dataset file",
    description="Upload a dataset file (CSV, XLSX, XLS, JSON) and extract metadata",
    responses={
        201: {
            "description": "Dataset uploaded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "user_id": "00000000-0000-0000-0000-000000000001",
                        "name": "sales_data",
                        "file_path": "/uploads/user123/dataset456/sales_data.csv",
                        "rows": 1000,
                        "cols": 5,
                        "dtypes": {
                            "date": "object",
                            "product": "object",
                            "quantity": "int64",
                            "price": "float64",
                            "total": "float64"
                        },
                        "missing_values": {
                            "date": 0,
                            "product": 2,
                            "quantity": 1,
                            "price": 0,
                            "total": 1
                        },
                        "uploaded_at": "2025-01-15T10:30:00Z"
                    }
                }
            }
        },
        400: {
            "description": "Bad request - Invalid file type or corrupted file",
            "content": {
                "application/json": {
                    "example": {"detail": "File type .txt not supported. Allowed: .csv, .xlsx, .xls, .json"}
                }
            }
        },
        413: {
            "description": "File size exceeds maximum allowed size",
            "content": {
                "application/json": {
                    "example": {"detail": "File size exceeds maximum allowed size of 100MB"}
                }
            }
        }
    }
)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file to upload (CSV, XLSX, XLS, or JSON format)"),
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_user_or_guest)
):
    """
    Upload a dataset file and extract metadata.

    This endpoint accepts dataset files in multiple formats and performs the following operations:

    - **Validates** file type (must be CSV, XLSX, XLS, or JSON)
    - **Validates** file size (must be under configured limit, default 100MB)
    - **Extracts metadata** including row count, column count, data types, and missing values
    - **Saves** file to persistent storage in user-specific directory
    - **Creates** database record with all metadata

    **Supported Formats:**
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - JSON (.json)

    **Returns:**
    - Dataset ID for use in other endpoints
    - Complete metadata including row/column counts
    - Data types for each column
    - Missing value counts per column
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
    dataset_id = str(uuid_lib.uuid4())

    # Get storage service (R2 or local filesystem)
    storage_service = get_storage_service()
    
    # Define file path (user_id/dataset_id/filename)
    file_storage_path = f"{user_id}/{dataset_id}/{file.filename}"
    
    # Upload file to R2 or local storage
    try:
        file_url = storage_service.upload_file(
            file_content=content,
            file_path=file_storage_path,
            content_type=file.content_type
        )
    except Exception as e:
        import traceback
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

    # Extract metadata using pandas (load from bytes)
    try:
        if file_ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(io.BytesIO(content))
        elif file_ext == ".json":
            df = pd.read_json(io.BytesIO(content))

        # Calculate statistics
        row_count = len(df)
        column_count = len(df.columns)

        # Data types as dict
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Missing values per column
        missing_values = {col: int(df[col].isna().sum()) for col in df.columns}

    except Exception as e:
        # Clean up uploaded file on error
        import traceback
        print(f"Processing error: {str(e)}")
        print(traceback.format_exc())
        try:
            storage_service.delete_file(file_storage_path)
        except:
            pass  # Ignore cleanup errors
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process file: {str(e)}"
        )

    # Create database record
    dataset = Dataset(
        id=dataset_id,
        user_id=user_id,
        name=Path(file.filename).stem,  # filename without extension
        file_path=file_storage_path,  # Store R2 path or local filesystem path
        rows=row_count,
        cols=column_count,
        dtypes=dtypes,
        missing_values=missing_values
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # Convert to Pydantic model manually to avoid validation errors
    return DatasetRead(
        id=dataset.id,
        user_id=dataset.user_id,
        name=dataset.name,
        file_path=dataset.file_path,
        shape=None, # Dataset model doesn't have shape dict, it has rows/cols
        dtypes=dataset.dtypes,
        missing_values=dataset.missing_values,
        uploaded_at=dataset.uploaded_at
    )


@router.get(
    "/",
    response_model=List[DatasetRead],
    summary="List all datasets",
    description="Get a paginated list of all datasets for the current user",
    responses={
        200: {
            "description": "List of datasets retrieved successfully",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "123e4567-e89b-12d3-a456-426614174000",
                            "user_id": "00000000-0000-0000-0000-000000000001",
                            "name": "sales_data",
                            "file_path": "/uploads/user123/dataset456/sales_data.csv",
                            "rows": 1000,
                            "cols": 5,
                            "dtypes": {"product": "object", "quantity": "int64"},
                            "missing_values": {"product": 2, "quantity": 1},
                            "uploaded_at": "2025-01-15T10:30:00Z"
                        }
                    ]
                }
            }
        }
    }
)
async def list_datasets(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    skip: int = 0,
    limit: int = 100
):
    """
    List all datasets for the current user with pagination.

    Returns datasets ordered by upload date (newest first).

    **Query Parameters:**
    - **skip**: Number of records to skip (default: 0)
    - **limit**: Maximum number of records to return (default: 100, max: 100)
    """
    datasets = db.query(Dataset).filter(
        Dataset.user_id == user_id
    ).order_by(
        Dataset.created_at.desc()
    ).offset(skip).limit(limit).all()

    return datasets


@router.get(
    "/{dataset_id}",
    response_model=DatasetRead,
    summary="Get dataset by ID",
    description="Retrieve detailed information about a specific dataset",
    responses={
        200: {
            "description": "Dataset retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "user_id": "00000000-0000-0000-0000-000000000001",
                        "name": "sales_data",
                        "file_path": "/uploads/user123/dataset456/sales_data.csv",
                        "rows": 1000,
                        "cols": 5,
                        "dtypes": {"product": "object", "quantity": "int64", "price": "float64"},
                        "missing_values": {"product": 2, "quantity": 1, "price": 0},
                        "uploaded_at": "2025-01-15T10:30:00Z"
                    }
                }
            }
        },
        404: {
            "description": "Dataset not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset 123e4567-e89b-12d3-a456-426614174000 not found"}
                }
            }
        }
    }
)
async def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Get detailed information about a specific dataset.

    **Path Parameters:**
    - **dataset_id**: UUID of the dataset

    **Returns:**
    - Complete dataset metadata including file path, dimensions, data types, and missing values
    - Returns 404 if dataset not found
    - Returns 403 if user doesn't own the dataset (unless admin)
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    # Verify ownership (allows admins to access any dataset)
    verify_resource_ownership(dataset.user_id, user_id, allow_admin=True, db=db)

    return dataset


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a dataset",
    description="Delete a dataset and its associated file from storage",
    responses={
        204: {
            "description": "Dataset deleted successfully"
        },
        404: {
            "description": "Dataset not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset 123e4567-e89b-12d3-a456-426614174000 not found"}
                }
            }
        }
    }
)
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Delete a dataset and remove its file from storage.

    This operation is irreversible and will:
    - Remove the dataset record from the database
    - Delete the associated file from disk storage
    - Clean up empty directories

    **Path Parameters:**
    - **dataset_id**: UUID of the dataset to delete

    **Returns:**
    - 204 No Content on successful deletion
    - 404 if dataset not found
    - 403 if user doesn't own the dataset (unless admin)
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    # Verify ownership (allows admins to delete any dataset)
    verify_resource_ownership(dataset.user_id, user_id, allow_admin=True, db=db)

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


@router.get(
    "/{dataset_id}/preview",
    response_model=DatasetPreviewResponse,
    summary="Get dataset preview",
    description="Get a preview of the dataset with sample rows and column metadata",
    responses={
        200: {
            "description": "Dataset preview retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "preview": [
                            ["John Doe", 30, "New York", 75000],
                            ["Jane Smith", 25, "San Francisco", 85000],
                            ["Bob Johnson", 35, "Chicago", 65000]
                        ],
                        "columns": [
                            {
                                "name": "name",
                                "dataType": "object",
                                "nullCount": 0,
                                "uniqueCount": 3,
                                "sampleValues": ["John Doe", "Jane Smith", "Bob Johnson"]
                            },
                            {
                                "name": "age",
                                "dataType": "int64",
                                "nullCount": 0,
                                "uniqueCount": 3,
                                "sampleValues": [30, 25, 35]
                            }
                        ],
                        "totalRows": 1000,
                        "displayedRows": 3
                    }
                }
            }
        },
        404: {
            "description": "Dataset not found or file not found on disk",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset file not found on disk"}
                }
            }
        },
        500: {
            "description": "Failed to read dataset file",
            "content": {
                "application/json": {
                    "example": {"detail": "Failed to read dataset: UnicodeDecodeError"}
                }
            }
        }
    }
)
async def get_dataset_preview(
    dataset_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    rows: int = 10
):
    """
    Get a preview of the dataset with sample rows and detailed column metadata.

    Returns the first N rows of the dataset along with comprehensive column information
    including data types, null counts, unique values, and sample data.

    **Path Parameters:**
    - **dataset_id**: UUID of the dataset

    **Query Parameters:**
    - **rows**: Number of rows to preview (default: 10, max: 100)

    **Returns:**
    - **preview**: 2D array of values (rows x columns)
    - **columns**: Array of column metadata objects with:
      - name: Column name
      - dataType: Pandas data type (int64, float64, object, etc.)
      - nullCount: Number of missing values
      - uniqueCount: Number of unique values
      - sampleValues: First 5 non-null sample values
    - **totalRows**: Total number of rows in dataset
    - **displayedRows**: Number of rows in preview (may be less than requested if dataset is smaller)
    """
    # Get dataset from database
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    # Verify ownership (allows admins to access any dataset)
    verify_resource_ownership(dataset.user_id, user_id, allow_admin=True, db=db)

    # Load file
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on disk"
        )

    try:
        # Read file based on extension
        file_ext = file_path.suffix.lower()
        if file_ext == ".csv":
            df = pd.read_csv(file_path, nrows=rows)
            df_full = pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, nrows=rows)
            df_full = pd.read_excel(file_path)
        elif file_ext == ".json":
            df = pd.read_json(file_path)
            df_full = df.copy()
            df = df.head(rows)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_ext}"
            )

        # Extract column information
        columns = []
        for col in df_full.columns:
            col_data = df_full[col]
            columns.append(
                ColumnInfo(
                    name=col,
                    dataType=str(col_data.dtype),
                    nullCount=int(col_data.isna().sum()),
                    uniqueCount=int(col_data.nunique()),
                    sampleValues=col_data.dropna().head(5).tolist()
                )
            )

        # Convert dataframe to 2D array
        preview_data = df.values.tolist()

        return DatasetPreviewResponse(
            preview=preview_data,
            columns=columns,
            totalRows=len(df_full),
            displayedRows=len(df)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}"
        )


@router.get(
    "/{dataset_id}/stats",
    response_model=DatasetStatsResponse,
    summary="Get dataset statistics",
    description="Get comprehensive statistical summary of the dataset",
    responses={
        200: {
            "description": "Dataset statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "rowCount": 1000,
                        "columnCount": 5,
                        "numericColumns": 3,
                        "categoricalColumns": 2,
                        "missingValues": 15,
                        "duplicateRows": 3,
                        "memoryUsage": 82400,
                        "columns": [
                            {
                                "name": "age",
                                "dataType": "int64",
                                "nullCount": 2,
                                "uniqueCount": 45,
                                "sampleValues": [30, 25, 35, 28, 42]
                            },
                            {
                                "name": "salary",
                                "dataType": "float64",
                                "nullCount": 5,
                                "uniqueCount": 897,
                                "sampleValues": [75000.0, 85000.0, 65000.0, 72000.0, 78000.0]
                            }
                        ]
                    }
                }
            }
        },
        404: {
            "description": "Dataset not found or file not found on disk",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset file not found on disk"}
                }
            }
        },
        500: {
            "description": "Failed to read dataset file",
            "content": {
                "application/json": {
                    "example": {"detail": "Failed to read dataset: MemoryError"}
                }
            }
        }
    }
)
async def get_dataset_stats(
    dataset_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Get comprehensive statistical summary of the dataset.

    Analyzes the entire dataset and returns detailed statistics including:
    - Dataset dimensions (rows, columns)
    - Column type breakdown (numeric vs categorical)
    - Data quality metrics (missing values, duplicates)
    - Memory usage estimation
    - Per-column metadata

    **Path Parameters:**
    - **dataset_id**: UUID of the dataset

    **Returns:**
    - **rowCount**: Total number of rows
    - **columnCount**: Total number of columns
    - **numericColumns**: Count of numeric columns (int, float)
    - **categoricalColumns**: Count of categorical columns (object, string)
    - **missingValues**: Total number of missing values across all columns
    - **duplicateRows**: Number of duplicate rows
    - **memoryUsage**: Estimated memory usage in bytes
    - **columns**: Array of column metadata (same as preview endpoint)

    **Use Cases:**
    - Data quality assessment
    - Feature engineering decisions
    - Memory optimization planning
    - Dataset understanding before preprocessing
    """
    # Get dataset from database
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    # Verify ownership (allows admins to access any dataset)
    verify_resource_ownership(dataset.user_id, user_id, allow_admin=True, db=db)

    # Load file
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on disk"
        )

    try:
        # Read file based on extension
        file_ext = file_path.suffix.lower()
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif file_ext == ".json":
            df = pd.read_json(file_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_ext}"
            )

        # Calculate statistics
        row_count = len(df)
        column_count = len(df.columns)

        # Count numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_count = len(numeric_cols)
        categorical_count = len(categorical_cols)

        # Missing values
        total_missing = int(df.isna().sum().sum())

        # Duplicate rows
        duplicate_count = int(df.duplicated().sum())

        # Memory usage (in bytes)
        memory_usage = int(df.memory_usage(deep=True).sum())

        # Extract column information
        columns = []
        for col in df.columns:
            col_data = df[col]
            columns.append(
                ColumnInfo(
                    name=col,
                    dataType=str(col_data.dtype),
                    nullCount=int(col_data.isna().sum()),
                    uniqueCount=int(col_data.nunique()),
                    sampleValues=col_data.dropna().head(5).tolist()
                )
            )

        return DatasetStatsResponse(
            rowCount=row_count,
            columnCount=column_count,
            numericColumns=numeric_count,
            categoricalColumns=categorical_count,
            missingValues=total_missing,
            duplicateRows=duplicate_count,
            memoryUsage=memory_usage,
            columns=columns
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}"
        )
