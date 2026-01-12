"""
Learning Curve Calculation Module

This module provides functions to calculate learning curves for model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from sklearn.model_selection import learning_curve
from app.ml_engine.models.base import BaseModelWrapper
import logging

logger = logging.getLogger(__name__)

def calculate_learning_curve(
    model: BaseModelWrapper,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 3,
    n_jobs: int = -1,
    train_sizes: Optional[np.ndarray] = None
) -> Optional[Dict[str, Any]]:
    """
    Calculate learning curve data for a given model and dataset.
    
    Args:
        model: The model wrapper instance (must contain the underlying sklearn estimator)
        X: Feature data
        y: Target data
        cv: Number of cross-validation folds (default: 3 to save time)
        n_jobs: Number of parallel jobs (default: -1 for all cores)
        train_sizes: Array of training set sizes (fractions) to use
        
    Returns:
        Dictionary containing learning curve data (train_sizes, train_scores, val_scores)
        or None if calculation fails
    """
    try:
        # Defaults
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
            
        # Ensure model has the underlying estimator created
        if model.model is None:
            # Try to create it if it doesn't exist (though usually it should be created by now)
            try:
                model.model = model._create_model()
            except Exception as e:
                logger.error(f"Failed to create underlying model for learning curve: {e}")
                return None
                
        estimator = model.model
        
        # Handle DataFrames/Series
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        y_data = y.values if isinstance(y, pd.Series) else y
        
        # Determine scoring based on task type
        # For classification: accuracy (default)
        # For regression: r2 (default)
        # We'll use the estimator's default score method which aligns with this
        
        logger.info(f"Calculating learning curve with cv={cv}, train_sizes={train_sizes}")
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator,
            X_data,
            y_data,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            shuffle=True, # Important for ordered datasets
            error_score='raise'
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Prepare result structure matching Recharts/Plotly expectations roughly
        # The frontend expects: { x: [...], y: [...], ... } for traces
        # But here we return the raw statistics, and the API/Frontend transforms it
        
        return {
            "train_sizes": train_sizes_abs.tolist(),
            "train_mean": train_mean.tolist(),
            "train_std": train_std.tolist(),
            "test_mean": test_mean.tolist(),
            "test_std": test_std.tolist(),
            "scoring": "default" # usually accuracy or r2
        }
        
    except Exception as e:
        logger.error(f"Error computing learning curve: {str(e)}")
        # Don't fail the whole training if this optional metric fails
        return None
