"""
Admin endpoint to fix dataset metadata
"""
import io
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd
import logging

from app.db.session import get_db
from app.models.dataset import Dataset
from app.services.r2_storage_service import get_storage_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/fix-metadata",
    summary="Fix missing dataset metadata",
    description="Retroactively extract and update metadata for datasets with missing rows/cols/stats"
)
async def fix_dataset_metadata(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Fix metadata for all datasets that have NULL rows, cols, or missing values.
    Downloads each file from R2/local storage and extracts the metadata.
    """
    try:
        storage_service = get_storage_service()
        
        # Get all datasets with missing metadata
        datasets = db.query(Dataset).filter(
            (Dataset.rows == None) | (Dataset.cols == None)
        ).all()
        
        if not datasets:
            return {
                "message": "No datasets need fixing",
                "fixed": 0,
                "failed": 0,
                "details": []
            }
        
        logger.info(f"Found {len(datasets)} datasets with missing metadata")
        
        fixed_count = 0
        failed_count = 0
        details = []
        
        for dataset in datasets:
            try:
                logger.info(f"Processing dataset: {dataset.name} (ID: {dataset.id})")
                
                # Download file from R2 or local storage
                file_content = storage_service.download_file(dataset.file_path)
                file_ext = Path(dataset.name).suffix.lower()
                
                # Read file into pandas
                if file_ext == ".csv":
                    df = pd.read_csv(io.BytesIO(file_content))
                elif file_ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(io.BytesIO(file_content))
                elif file_ext == ".json":
                    df = pd.read_json(io.BytesIO(file_content))
                else:
                    details.append({
                        "dataset": dataset.name,
                        "status": "failed",
                        "error": f"Unsupported file format: {file_ext}"
                    })
                    failed_count += 1
                    continue
                
                # Extract metadata
                rows = len(df)
                cols = len(df.columns)
                dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
                missing_values = int(df.isna().sum().sum())
                
                # Update database
                dataset.rows = rows
                dataset.cols = cols
                dataset.dtypes = dtypes
                dataset.missing_values = missing_values
                
                db.commit()
                
                details.append({
                    "dataset": dataset.name,
                    "status": "fixed",
                    "rows": rows,
                    "cols": cols,
                    "missing_values": missing_values
                })
                
                logger.info(f"✓ Fixed: {dataset.name} - {rows} x {cols}")
                fixed_count += 1
                
            except Exception as e:
                logger.error(f"✗ Failed to fix {dataset.name}: {e}")
                details.append({
                    "dataset": dataset.name,
                    "status": "failed",
                    "error": str(e)
                })
                failed_count += 1
                db.rollback()
                continue
        
        return {
            "message": f"Fixed {fixed_count} datasets, {failed_count} failed",
            "fixed": fixed_count,
            "failed": failed_count,
            "details": details
        }
        
    except Exception as e:
        logger.error(f"Error fixing dataset metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fix dataset metadata: {str(e)}"
        )
