"""
Training module for ML Engine.

This module provides generic training functions and utilities for training
machine learning models with consistent interfaces.
"""

from app.ml_engine.training.trainer import (
    TrainingResult,
    train_model,
    evaluate_model,
    predict_with_model,
    get_model_info
)

__all__ = [
    'TrainingResult',
    'train_model',
    'evaluate_model',
    'predict_with_model',
    'get_model_info'
]
