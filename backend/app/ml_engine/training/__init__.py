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
from app.ml_engine.training.data_split import (
    DataSplitResult,
    train_test_split,
    train_val_test_split,
    split_by_ratio,
    temporal_split,
    get_split_info
)

__all__ = [
    # Training functions
    'TrainingResult',
    'train_model',
    'evaluate_model',
    'predict_with_model',
    'get_model_info',
    # Data splitting functions
    'DataSplitResult',
    'train_test_split',
    'train_val_test_split',
    'split_by_ratio',
    'temporal_split',
    'get_split_info'
]
