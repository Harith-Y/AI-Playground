#!/usr/bin/env python3
"""
Fix dataset metadata for datasets with missing rows/cols/stats
"""

import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.services.r2_storage_service import get_storage_service
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_dataset_metadata():
    """Fix metadata for all datasets with missing information"""
    db = SessionLocal()
    storage_service = get_storage_service()
    
    try:
        # Get all datasets with missing metadata
        datasets = db.query(Dataset).filter(
            (Dataset.rows == None) | (Dataset.cols == None)
        ).all()
        
        logger.info(f"Found {len(datasets)} datasets with missing metadata")
        
        if not datasets:
            logger.info("No datasets need fixing!")
            return
        
        fixed_count = 0
        failed_count = 0
        
        for dataset in datasets:
            try:
                logger.info(f"\nProcessing dataset: {dataset.name} (ID: {dataset.id})")
                
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
                    logger.warning(f"Unsupported file format: {file_ext}")
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
                
                logger.info(f"✓ Fixed: {dataset.name} - {rows} x {cols} rows/cols")
                fixed_count += 1
                
            except Exception as e:
                logger.error(f"✗ Failed to fix {dataset.name}: {e}")
                failed_count += 1
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY:")
        logger.info(f"  Fixed: {fixed_count} datasets")
        logger.info(f"  Failed: {failed_count} datasets")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    fix_dataset_metadata()
