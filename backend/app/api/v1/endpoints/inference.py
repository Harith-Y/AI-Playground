"""
Optimized Inference Endpoints

Provides high-performance inference REST API with model caching,
batch prediction, and performance monitoring.

Based on: ML-73 - Optimize inference speed
"""

from fastapi import APIRouter, HTTPException, status, File, UploadFile
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from app.ml_engine.inference import get_optimized_predictor
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for single/batch predictions."""
    model_path: str = Field(..., description="Path to the saved model file")
    features: List[List[float]] = Field(..., description="Input features as 2D array")
    return_probabilities: bool = Field(default=False, description="Return class probabilities (classification only)")
    use_cache: bool = Field(default=True, description="Use cached model if available")
    batch_size: Optional[int] = Field(default=None, description="Batch size for processing (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "/uploads/models/exp-123/run-456.joblib",
                "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "return_probabilities": True,
                "use_cache": True
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Any] = Field(..., description="Predicted values/labels")
    n_samples: int = Field(..., description="Number of samples processed")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities (if available)")
    classes: Optional[List[Any]] = Field(None, description="Class names (if available)")
    inference_time_seconds: float = Field(..., description="Total inference time")
    throughput_samples_per_sec: float = Field(..., description="Inference throughput")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [1, 0, 1],
                "n_samples": 3,
                "probabilities": [[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]],
                "classes": [0, 1],
                "inference_time_seconds": 0.023,
                "throughput_samples_per_sec": 130.4
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch file predictions."""
    model_path: str = Field(..., description="Path to the saved model file")
    data_path: str = Field(..., description="Path to input data file (CSV/Parquet/JSON)")
    output_path: Optional[str] = Field(None, description="Path to save predictions (optional)")
    batch_size: int = Field(default=5000, description="Batch size for processing")
    return_probabilities: bool = Field(default=False, description="Include class probabilities")
    use_cache: bool = Field(default=True, description="Use cached model if available")

    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "/uploads/models/exp-123/run-456.joblib",
                "data_path": "/uploads/data/test_data.csv",
                "output_path": "/uploads/predictions/results.csv",
                "batch_size": 5000,
                "return_probabilities": True,
                "use_cache": True
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    n_samples: int = Field(..., description="Total number of samples processed")
    output_path: Optional[str] = Field(None, description="Path where predictions were saved")
    inference_time_seconds: float = Field(..., description="Total inference time")
    throughput_samples_per_sec: float = Field(..., description="Inference throughput")
    preview: Optional[List[Dict[str, Any]]] = Field(None, description="Preview of first 5 predictions")

    class Config:
        json_schema_extra = {
            "example": {
                "n_samples": 10000,
                "output_path": "/uploads/predictions/results.csv",
                "inference_time_seconds": 2.45,
                "throughput_samples_per_sec": 4081.6,
                "preview": [{"prediction": 1}, {"prediction": 0}]
            }
        }


class ModelWarmupRequest(BaseModel):
    """Request model for model warmup."""
    model_path: str = Field(..., description="Path to the model to warmup")

    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "/uploads/models/exp-123/run-456.joblib"
            }
        }


class ModelWarmupResponse(BaseModel):
    """Response model for model warmup."""
    model_path: str = Field(..., description="Path to the warmed up model")
    warmup_time_seconds: float = Field(..., description="Time taken to warmup")
    cached: bool = Field(..., description="Whether model is now cached")


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    size: int = Field(..., description="Number of cached models")
    max_size: int = Field(..., description="Maximum cache capacity")
    cached_models: List[str] = Field(..., description="List of cached model paths")
    utilization: float = Field(..., description="Cache utilization (0.0 to 1.0)")


# Endpoints
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using optimized inference engine.

    This endpoint provides high-performance predictions with:
    - Model caching for repeated requests
    - Optimized batch processing
    - Performance metrics (inference time, throughput)

    Example:
        POST /api/v1/inference/predict
        {
            "model_path": "/uploads/models/exp-123/model.joblib",
            "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "return_probabilities": true,
            "use_cache": true
        }
    """
    try:
        logger.info(f"Prediction request for {len(request.features)} samples")

        # Get predictor instance
        predictor = get_optimized_predictor()

        # Convert features to numpy array
        X = np.array(request.features)

        # Use optimized prediction method based on data size
        if request.batch_size and len(X) > request.batch_size:
            # Use batched prediction for large arrays
            results = predictor.predict_from_array(
                model_path=request.model_path,
                X=X,
                batch_size=request.batch_size,
                return_probabilities=request.return_probabilities,
                use_cache=request.use_cache
            )
        else:
            # Use standard prediction for small arrays
            results = predictor.predict(
                model_path=request.model_path,
                X=X,
                return_probabilities=request.return_probabilities,
                use_cache=request.use_cache
            )

        # Convert numpy arrays to lists for JSON serialization
        response_data = {
            "predictions": results['predictions'].tolist() if isinstance(results['predictions'], np.ndarray) else results['predictions'],
            "n_samples": results['n_samples'],
            "inference_time_seconds": results['inference_time_seconds'],
            "throughput_samples_per_sec": results['throughput_samples_per_sec']
        }

        # Add probabilities if available
        if 'probabilities' in results:
            response_data['probabilities'] = results['probabilities'].tolist() if isinstance(results['probabilities'], np.ndarray) else results['probabilities']

        if 'classes' in results:
            response_data['classes'] = results['classes'].tolist() if isinstance(results['classes'], np.ndarray) else results['classes']

        logger.info(f"Prediction successful: {results['throughput_samples_per_sec']:.0f} samples/sec")

        return PredictionResponse(**response_data)

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found: {request.model_path}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions on large datasets from files.

    This endpoint provides optimized batch prediction with:
    - Chunked file reading for memory efficiency
    - Configurable batch size (default: 5000)
    - Support for CSV, Parquet, and JSON formats
    - Optional probability output
    - Performance metrics

    Example:
        POST /api/v1/inference/predict-batch
        {
            "model_path": "/uploads/models/exp-123/model.joblib",
            "data_path": "/uploads/data/test.csv",
            "output_path": "/uploads/predictions/results.csv",
            "batch_size": 5000,
            "return_probabilities": true
        }
    """
    try:
        logger.info(f"Batch prediction request: {request.data_path}")

        # Get predictor instance
        predictor = get_optimized_predictor()

        # Run batch prediction
        from datetime import datetime
        start_time = datetime.now()

        result_df = predictor.predict_batch(
            model_path=request.model_path,
            data_path=request.data_path,
            output_path=request.output_path,
            batch_size=request.batch_size,
            return_probabilities=request.return_probabilities,
            use_cache=request.use_cache
        )

        inference_time = (datetime.now() - start_time).total_seconds()
        throughput = len(result_df) / inference_time if inference_time > 0 else 0

        # Create preview (first 5 rows)
        preview = result_df.head(5).to_dict('records')

        logger.info(f"Batch prediction successful: {len(result_df)} samples, {throughput:.0f} samples/sec")

        return BatchPredictionResponse(
            n_samples=len(result_df),
            output_path=request.output_path,
            inference_time_seconds=inference_time,
            throughput_samples_per_sec=throughput,
            preview=preview
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/warmup", response_model=ModelWarmupResponse)
async def warmup_model(request: ModelWarmupRequest):
    """
    Warmup (preload) a model into cache.

    This endpoint is useful for:
    - Preloading models before serving requests
    - Reducing first-request latency
    - Ensuring models are ready in cache

    Example:
        POST /api/v1/inference/warmup
        {
            "model_path": "/uploads/models/exp-123/model.joblib"
        }
    """
    try:
        logger.info(f"Warmup request for model: {request.model_path}")

        # Get predictor instance
        predictor = get_optimized_predictor()

        # Warmup model
        warmup_time = predictor.warmup(request.model_path)

        logger.info(f"Model warmup successful in {warmup_time:.3f}s")

        return ModelWarmupResponse(
            model_path=request.model_path,
            warmup_time_seconds=warmup_time,
            cached=True
        )

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found: {request.model_path}"
        )
    except Exception as e:
        logger.error(f"Warmup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model warmup failed: {str(e)}"
        )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get model cache statistics.

    Returns information about:
    - Number of cached models
    - Cache capacity and utilization
    - List of cached model paths

    Example:
        GET /api/v1/inference/cache/stats
    """
    try:
        # Get predictor instance
        predictor = get_optimized_predictor()

        # Get cache stats
        stats = predictor.get_cache_stats()

        logger.info(f"Cache stats retrieved: {stats['size']}/{stats['max_size']} models")

        return CacheStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.delete("/cache/clear")
async def clear_cache():
    """
    Clear model cache.

    Removes all cached models from memory.
    Useful for:
    - Freeing up memory
    - Forcing models to reload
    - Testing cache behavior

    Example:
        DELETE /api/v1/inference/cache/clear
    """
    try:
        # Get predictor instance
        predictor = get_optimized_predictor()

        # Clear cache
        predictor.clear_cache()

        logger.info("Cache cleared successfully")

        return {"message": "Cache cleared successfully"}

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for inference service.

    Example:
        GET /api/v1/inference/health
    """
    return {
        "status": "healthy",
        "service": "optimized-inference",
        "version": "1.0.0"
    }
