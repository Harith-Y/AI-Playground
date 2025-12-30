"""
ML Pipeline Package - Customer Churn Prediction

Auto-generated modular ML pipeline.

Modules:
    preprocess: Data preprocessing functions
    train: Model training functions
    evaluate: Model evaluation functions
    main: Pipeline orchestration

Usage:
    # Import individual modules
    from ml_pipeline_modules.preprocess import preprocess_data
    from ml_pipeline_modules.train import train_model
    from ml_pipeline_modules.evaluate import evaluate_model
    
    # Or run the complete pipeline
    from ml_pipeline_modules.main import run_complete_pipeline
    results = run_complete_pipeline('data.csv')
"""

from .preprocess import preprocess_data
from .train import train_model, save_model, load_model, predict, split_data
from .evaluate import evaluate_model, save_results

__all__ = [
    'preprocess_data',
    'train_model',
    'save_model',
    'load_model',
    'predict',
    'split_data',
    'evaluate_model',
    'save_results',
]
