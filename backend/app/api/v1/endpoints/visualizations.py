"""
Dataset visualization endpoints for EDA (Exploratory Data Analysis)
"""

import io
import base64
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from uuid import UUID
import logging

from app.db.session import get_db
from app.models.dataset import Dataset
from app.core.security import get_current_user_id
from app.services.r2_storage_service import get_storage_service
from app.core.config import settings
from urllib.parse import urlparse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# Response Models
class VisualizationResponse(BaseModel):
    """Single visualization data"""
    type: str
    data: Dict[str, Any]
    title: str
    xLabel: Optional[str] = None
    yLabel: Optional[str] = None


class VisualizationsResponse(BaseModel):
    """Multiple visualizations"""
    visualizations: List[VisualizationResponse]


def get_user_or_guest(db: Session = Depends(get_db)) -> UUID:
    """Get authenticated user or guest user"""
    try:
        # For now, return guest user
        guest_email = "guest@aiplayground.local"
        from app.models.user import User
        guest_user = db.query(User).filter(User.email == guest_email).first()
        if guest_user:
            return guest_user.id
    except Exception as e:
        logger.error(f"Failed to get guest user: {e}")
    # Fallback UUID
    return UUID("00000000-0000-0000-0000-000000000001")


def load_dataset_df(dataset: Dataset) -> pd.DataFrame:
    """Load dataset file into pandas DataFrame"""
    try:
        storage_service = get_storage_service()
        file_path = dataset.file_path
        
        # Determine the key/path to retrieve
        key = file_path
        if file_path.startswith("http://") or file_path.startswith("https://"):
            if settings.R2_PUBLIC_URL and file_path.startswith(settings.R2_PUBLIC_URL):
                key = file_path[len(settings.R2_PUBLIC_URL):].lstrip('/')
            else:
                # Attempt to extract key from standard R2 URL
                parsed = urlparse(file_path)
                path_parts = parsed.path.lstrip('/').split('/', 1)
                if len(path_parts) > 1 and path_parts[0] == settings.R2_BUCKET_NAME:
                    key = path_parts[1]
                else:
                    key = parsed.path.lstrip('/')
        
        # Download content
        content = storage_service.download_file(key)
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(io.BytesIO(content))
        elif file_ext == ".json":
            df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_ext}"
            )
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}"
        )


@router.get(
    "/{dataset_id}/histogram/{column_name}",
    response_model=VisualizationResponse,
    summary="Get histogram data for a numeric column"
)
async def get_histogram(
    dataset_id: str,
    column_name: str,
    bins: int = Query(default=30, ge=5, le=100),
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_user_or_guest)
):
    """
    Generate histogram data for a numeric column.
    
    Returns bin edges and counts for frontend visualization.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    df = load_dataset_df(dataset)
    
    if column_name not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{column_name}' not found in dataset"
        )
    
    # Get numeric data
    column_data = df[column_name].dropna()
    if not pd.api.types.is_numeric_dtype(column_data):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{column_name}' is not numeric"
        )
    
    # Calculate histogram
    counts, bin_edges = np.histogram(column_data, bins=bins)
    
    return VisualizationResponse(
        type="histogram",
        title=f"Distribution of {column_name}",
        xLabel=column_name,
        yLabel="Frequency",
        data={
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
            "stats": {
                "mean": float(column_data.mean()),
                "median": float(column_data.median()),
                "std": float(column_data.std()),
                "min": float(column_data.min()),
                "max": float(column_data.max())
            }
        }
    )


@router.get(
    "/{dataset_id}/correlation",
    response_model=VisualizationResponse,
    summary="Get correlation matrix for numeric columns"
)
async def get_correlation_matrix(
    dataset_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_user_or_guest)
):
    """
    Calculate correlation matrix for all numeric columns.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    df = load_dataset_df(dataset)
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset must have at least 2 numeric columns for correlation"
        )
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    return VisualizationResponse(
        type="correlation",
        title="Feature Correlation Matrix",
        data={
            "matrix": corr_matrix.values.tolist(),
            "columns": corr_matrix.columns.tolist(),
            "index": corr_matrix.index.tolist()
        }
    )


@router.get(
    "/{dataset_id}/scatter/{x_column}/{y_column}",
    response_model=VisualizationResponse,
    summary="Get scatter plot data for two columns"
)
async def get_scatter_plot(
    dataset_id: str,
    x_column: str,
    y_column: str,
    sample_size: Optional[int] = Query(default=None, ge=10, le=10000),
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_user_or_guest)
):
    """
    Generate scatter plot data for two numeric columns.
    
    Optionally sample data if dataset is large.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    df = load_dataset_df(dataset)
    
    if x_column not in df.columns or y_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columns '{x_column}' or '{y_column}' not found"
        )
    
    # Sample if needed
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Remove missing values
    plot_df = df[[x_column, y_column]].dropna()
    
    return VisualizationResponse(
        type="scatter",
        title=f"{y_column} vs {x_column}",
        xLabel=x_column,
        yLabel=y_column,
        data={
            "x": plot_df[x_column].tolist(),
            "y": plot_df[y_column].tolist(),
            "count": len(plot_df),
            "correlation": float(plot_df[x_column].corr(plot_df[y_column]))
        }
    )


@router.get(
    "/{dataset_id}/boxplot/{column_name}",
    response_model=VisualizationResponse,
    summary="Get box plot data for a numeric column"
)
async def get_box_plot(
    dataset_id: str,
    column_name: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_user_or_guest)
):
    """
    Generate box plot data (quartiles, outliers) for a numeric column.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    df = load_dataset_df(dataset)
    
    if column_name not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{column_name}' not found"
        )
    
    column_data = df[column_name].dropna()
    if not pd.api.types.is_numeric_dtype(column_data):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{column_name}' is not numeric"
        )
    
    # Calculate quartiles
    q1 = column_data.quantile(0.25)
    q2 = column_data.quantile(0.50)  # median
    q3 = column_data.quantile(0.75)
    iqr = q3 - q1
    
    # Calculate whiskers
    lower_whisker = max(column_data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(column_data.max(), q3 + 1.5 * iqr)
    
    # Find outliers
    outliers = column_data[
        (column_data < lower_whisker) | (column_data > upper_whisker)
    ].tolist()
    
    return VisualizationResponse(
        type="box",
        title=f"Box Plot of {column_name}",
        yLabel=column_name,
        data={
            "min": float(column_data.min()),
            "q1": float(q1),
            "median": float(q2),
            "q3": float(q3),
            "max": float(column_data.max()),
            "lowerWhisker": float(lower_whisker),
            "upperWhisker": float(upper_whisker),
            "outliers": outliers[:100],  # Limit outliers to 100
            "outlierCount": len(outliers)
        }
    )


@router.get(
    "/{dataset_id}/value-counts/{column_name}",
    response_model=VisualizationResponse,
    summary="Get value counts for categorical column (bar chart data)"
)
async def get_value_counts(
    dataset_id: str,
    column_name: str,
    top_n: int = Query(default=10, ge=5, le=50),
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_user_or_guest)
):
    """
    Get value counts for categorical columns (useful for bar charts).
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    df = load_dataset_df(dataset)
    
    if column_name not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{column_name}' not found"
        )
    
    # Get value counts
    value_counts = df[column_name].value_counts().head(top_n)
    
    return VisualizationResponse(
        type="bar",
        title=f"Top {top_n} Values in {column_name}",
        xLabel=column_name,
        yLabel="Count",
        data={
            "categories": value_counts.index.tolist(),
            "counts": value_counts.values.tolist(),
            "total": int(df[column_name].count()),
            "unique": int(df[column_name].nunique())
        }
    )


@router.get(
    "/{dataset_id}/overview",
    response_model=VisualizationsResponse,
    summary="Get overview of all recommended visualizations"
)
async def get_visualization_overview(
    dataset_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_user_or_guest)
):
    """
    Get a curated set of visualizations based on dataset characteristics.
    
    Returns:
    - Histograms for key numeric columns
    - Correlation matrix if multiple numeric columns exist
    - Value counts for categorical columns
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    df = load_dataset_df(dataset)
    visualizations = []
    
    # 1. Correlation matrix (if 2+ numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        visualizations.append(VisualizationResponse(
            type="correlation",
            title="Feature Correlation Matrix",
            data={
                "matrix": corr_matrix.values.tolist(),
                "columns": corr_matrix.columns.tolist(),
                "index": corr_matrix.index.tolist()
            }
        ))
    
    # 2. Histograms for first 3 numeric columns
    for col in numeric_cols[:3]:
        column_data = df[col].dropna()
        counts, bin_edges = np.histogram(column_data, bins=30)
        visualizations.append(VisualizationResponse(
            type="histogram",
            title=f"Distribution of {col}",
            xLabel=col,
            yLabel="Frequency",
            data={
                "bins": bin_edges.tolist(),
                "counts": counts.tolist(),
                "stats": {
                    "mean": float(column_data.mean()),
                    "median": float(column_data.median()),
                    "std": float(column_data.std())
                }
            }
        ))
    
    # 3. Bar charts for first 2 categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols[:2]:
        value_counts = df[col].value_counts().head(10)
        visualizations.append(VisualizationResponse(
            type="bar",
            title=f"Top 10 Values in {col}",
            xLabel=col,
            yLabel="Count",
            data={
                "categories": value_counts.index.tolist(),
                "counts": value_counts.values.tolist()
            }
        ))
    
    return VisualizationsResponse(visualizations=visualizations)
