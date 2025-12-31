"""
Configuration for Render Free Tier deployment
Handles missing heavy ML libraries gracefully
"""
import os
from typing import Dict, List

# Models available on Render Free Tier (without heavy libraries)
RENDER_AVAILABLE_MODELS = {
    'classification': [
        'logistic_regression',
        'decision_tree',
        'random_forest',  # scikit-learn version (lightweight)
        'svm',
        'knn',
        'naive_bayes',
    ],
    'regression': [
        'linear_regression',
        'ridge',
        'lasso',
        'elastic_net',
        'decision_tree',
        'random_forest',  # scikit-learn version (lightweight)
        'svr',
        'knn',
    ],
    'clustering': [
        'kmeans',
        'dbscan',
        'hierarchical',
        'gaussian_mixture',
    ]
}

# Models that require heavy libraries (not available on free tier)
HEAVY_MODELS = {
    'xgboost',
    'catboost',
    'lightgbm',
}

def is_render_deployment() -> bool:
    """Check if running on Render"""
    return os.getenv('RENDER') == 'true' or os.getenv('ENVIRONMENT') == 'production'

def get_available_models(task_type: str) -> List[str]:
    """Get available models based on deployment environment"""
    if is_render_deployment():
        return RENDER_AVAILABLE_MODELS.get(task_type, [])
    
    # Local development - all models available
    from app.ml_engine.models.registry import get_available_models as get_all_models
    return get_all_models(task_type)

def is_model_available(model_name: str) -> bool:
    """Check if a model is available in current environment"""
    if not is_render_deployment():
        return True
    
    # Check if model is in any category
    for models in RENDER_AVAILABLE_MODELS.values():
        if model_name in models:
            return True
    
    return False

def validate_model_request(model_name: str, task_type: str) -> tuple[bool, str]:
    """
    Validate if a model can be used in current environment
    Returns: (is_valid, error_message)
    """
    if model_name in HEAVY_MODELS and is_render_deployment():
        return False, (
            f"Model '{model_name}' is not available on free tier deployment. "
            f"Available models: {', '.join(RENDER_AVAILABLE_MODELS.get(task_type, []))}"
        )
    
    available = RENDER_AVAILABLE_MODELS.get(task_type, [])
    if is_render_deployment() and model_name not in available:
        return False, (
            f"Model '{model_name}' is not available for {task_type}. "
            f"Available models: {', '.join(available)}"
        )
    
    return True, ""

# Memory optimization settings for Render
RENDER_MEMORY_LIMITS = {
    'max_dataset_rows': 10000,
    'max_dataset_size_mb': 10,
    'max_training_time_seconds': 60,
    'max_workers': 1,
    'enable_caching': True,
    'cache_ttl_seconds': 3600,
}

def get_memory_limits() -> Dict:
    """Get memory limits based on environment"""
    if is_render_deployment():
        return RENDER_MEMORY_LIMITS
    
    # Local development - more relaxed limits
    return {
        'max_dataset_rows': 100000,
        'max_dataset_size_mb': 100,
        'max_training_time_seconds': 300,
        'max_workers': 4,
        'enable_caching': True,
        'cache_ttl_seconds': 7200,
    }
