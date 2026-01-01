"""
Experiment Configuration API Endpoints.

Provides endpoints for serializing, exporting, and comparing experiment configurations.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
from pathlib import Path
import tempfile
import shutil

from app.db.session import get_db
from app.services.experiment_config_service import (
    ExperimentConfigSerializer,
    serialize_experiment_to_file,
    export_experiment_package
)
from app.utils.logger import get_logger

logger = get_logger("experiment_config_api")

router = APIRouter()


@router.get("/{experiment_id}/config")
def get_experiment_config(
    experiment_id: UUID,
    include_results: bool = Query(True, description="Include training results and metrics"),
    include_artifacts: bool = Query(False, description="Include model artifact paths"),
    db: Session = Depends(get_db)
):
    """
    Get experiment configuration as JSON.
    
    Returns complete experiment configuration including:
    - Dataset information
    - Preprocessing pipeline
    - Model configurations
    - Training results (optional)
    - Model artifact paths (optional)
    """
    try:
        serializer = ExperimentConfigSerializer(db)
        config = serializer.serialize_experiment(
            experiment_id,
            include_results=include_results,
            include_artifacts=include_artifacts
        )
        
        return JSONResponse(content=config)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting experiment config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment configuration: {str(e)}"
        )


@router.get("/{experiment_id}/config/download")
def download_experiment_config(
    experiment_id: UUID,
    include_results: bool = Query(True, description="Include training results"),
    include_artifacts: bool = Query(False, description="Include artifact paths"),
    db: Session = Depends(get_db)
):
    """
    Download experiment configuration as JSON file.
    
    Returns a downloadable JSON file with the experiment configuration.
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        # Serialize to file
        serializer = ExperimentConfigSerializer(db)
        serializer.save_to_file(
            experiment_id,
            tmp_path,
            include_results=include_results,
            include_artifacts=include_artifacts,
            pretty=True
        )
        
        # Return file
        return FileResponse(
            path=str(tmp_path),
            media_type='application/json',
            filename=f'experiment_{experiment_id}_config.json',
            background=None  # File will be cleaned up by OS
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error downloading experiment config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download experiment configuration: {str(e)}"
        )


@router.get("/{experiment_id}/export")
def export_experiment_for_reproduction(
    experiment_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Export complete experiment package for reproduction.
    
    Creates a ZIP file containing:
    - experiment_config.json: Full configuration
    - preprocessing_config.json: Preprocessing steps
    - model_configs.json: Model configurations
    - README.md: Reproduction instructions
    """
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Export experiment package
            exported_files = export_experiment_package(
                db,
                experiment_id,
                tmp_path
            )
            
            # Create ZIP file
            zip_path = Path(tempfile.mktemp(suffix='.zip'))
            shutil.make_archive(
                str(zip_path.with_suffix('')),
                'zip',
                tmp_path
            )
            
            # Return ZIP file
            return FileResponse(
                path=str(zip_path),
                media_type='application/zip',
                filename=f'experiment_{experiment_id}_package.zip',
                background=None
            )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error exporting experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export experiment: {str(e)}"
        )


@router.get("/{experiment_id}/preprocessing-config")
def get_preprocessing_config(
    experiment_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get only the preprocessing configuration for an experiment.
    
    Returns the preprocessing pipeline configuration without model information.
    """
    try:
        serializer = ExperimentConfigSerializer(db)
        config = serializer.serialize_experiment(
            experiment_id,
            include_results=False,
            include_artifacts=False
        )
        
        return JSONResponse(content={
            "experiment_id": str(experiment_id),
            "experiment_name": config["experiment"]["name"],
            "preprocessing": config["preprocessing"]
        })
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting preprocessing config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get preprocessing configuration: {str(e)}"
        )


@router.get("/{experiment_id}/model-configs")
def get_model_configs(
    experiment_id: UUID,
    include_results: bool = Query(False, description="Include training results"),
    db: Session = Depends(get_db)
):
    """
    Get only the model configurations for an experiment.
    
    Returns model configurations and optionally their training results.
    """
    try:
        serializer = ExperimentConfigSerializer(db)
        config = serializer.serialize_experiment(
            experiment_id,
            include_results=include_results,
            include_artifacts=False
        )
        
        return JSONResponse(content={
            "experiment_id": str(experiment_id),
            "experiment_name": config["experiment"]["name"],
            "models": config["models"]
        })
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting model configs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model configurations: {str(e)}"
        )


@router.post("/compare")
def compare_experiments(
    experiment_id_1: UUID = Query(..., description="First experiment ID"),
    experiment_id_2: UUID = Query(..., description="Second experiment ID"),
    db: Session = Depends(get_db)
):
    """
    Compare two experiment configurations.
    
    Returns a comparison showing differences in:
    - Preprocessing pipelines
    - Model configurations
    - Training results
    """
    try:
        serializer = ExperimentConfigSerializer(db)
        comparison = serializer.compare_experiments(
            experiment_id_1,
            experiment_id_2
        )
        
        return JSONResponse(content=comparison)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error comparing experiments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare experiments: {str(e)}"
        )


@router.get("/{experiment_id}/summary")
def get_experiment_summary(
    experiment_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a high-level summary of the experiment configuration.
    
    Returns key information without full details:
    - Number of preprocessing steps
    - Number of models trained
    - Model types
    - Best performing model (if results available)
    """
    try:
        serializer = ExperimentConfigSerializer(db)
        config = serializer.serialize_experiment(
            experiment_id,
            include_results=True,
            include_artifacts=False
        )
        
        # Calculate summary
        preprocessing_steps = config["preprocessing"]
        models = config["models"]
        
        # Find best model (if metrics available)
        best_model = None
        if models and any(m.get("metrics") for m in models):
            # Assume first metric is primary (e.g., accuracy, r2_score)
            models_with_metrics = [m for m in models if m.get("metrics")]
            if models_with_metrics:
                # Get first metric key
                first_model_metrics = models_with_metrics[0]["metrics"]
                if first_model_metrics:
                    metric_key = list(first_model_metrics.keys())[0]
                    best_model = max(
                        models_with_metrics,
                        key=lambda m: m["metrics"].get(metric_key, 0)
                    )
        
        summary = {
            "experiment_id": str(experiment_id),
            "experiment_name": config["experiment"]["name"],
            "status": config["experiment"]["status"],
            "created_at": config["experiment"]["created_at"],
            "dataset": {
                "name": config["dataset"]["name"],
                "shape": config["dataset"]["shape"]
            },
            "preprocessing": {
                "num_steps": len(preprocessing_steps),
                "step_types": [s["step_type"] for s in preprocessing_steps]
            },
            "models": {
                "num_models": len(models),
                "model_types": [m["model_type"] for m in models],
                "completed": len([m for m in models if m["status"] == "completed"]),
                "failed": len([m for m in models if m["status"] == "failed"])
            }
        }
        
        if best_model:
            summary["best_model"] = {
                "model_type": best_model["model_type"],
                "metrics": best_model["metrics"]
            }
        
        return JSONResponse(content=summary)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting experiment summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment summary: {str(e)}"
        )
