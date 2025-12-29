"""
Generic training functions for ML models.

This module provides high-level training functions that handle the complete
training workflow including data validation, model fitting, and evaluation.
"""

from typing import Union, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from app.ml_engine.models.base import BaseModelWrapper, SupervisedModelWrapper, UnsupervisedModelWrapper

logger = logging.getLogger(__name__)


class TrainingResult:
    """
    Container for training results.
    
    Stores all information about a training run including the fitted model,
    training metrics, and metadata.
    """
    
    def __init__(
        self,
        model: BaseModelWrapper,
        train_score: Optional[float] = None,
        val_score: Optional[float] = None,
        test_score: Optional[float] = None,
        training_time: Optional[float] = None,
        predictions: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize training result.
        
        Args:
            model: Fitted model wrapper
            train_score: Score on training set
            val_score: Score on validation set (if provided)
            test_score: Score on test set (if provided)
            training_time: Training duration in seconds
            predictions: Dictionary of predictions (train, val, test)
            metadata: Additional metadata about training
        """
        self.model = model
        self.train_score = train_score
        self.val_score = val_score
        self.test_score = test_score
        self.training_time = training_time
        self.predictions = predictions or {}
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert training result to dictionary."""
        return {
            'model_type': self.model.config.model_type,
            'task_type': self.model.get_task_type(),
            'train_score': self.train_score,
            'val_score': self.val_score,
            'test_score': self.test_score,
            'training_time': self.training_time,
            'is_fitted': self.model.is_fitted,
            'n_train_samples': self.model.metadata.n_train_samples,
            'n_features': self.model.metadata.n_features,
            'feature_names': self.model.metadata.feature_names,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation."""
        scores = []
        if self.train_score is not None:
            scores.append(f"train={self.train_score:.4f}")
        if self.val_score is not None:
            scores.append(f"val={self.val_score:.4f}")
        if self.test_score is not None:
            scores.append(f"test={self.test_score:.4f}")
        
        score_str = ", ".join(scores) if scores else "no scores"
        return f"TrainingResult(model={self.model.config.model_type}, {score_str})"


def train_model(
    model: BaseModelWrapper,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray, None] = None,
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    compute_train_score: bool = True,
    store_predictions: bool = False,
    **fit_params
) -> TrainingResult:
    """
    Generic training function for any model.
    
    This function handles the complete training workflow:
    1. Validates inputs
    2. Fits the model on training data
    3. Evaluates on train/val/test sets (if provided)
    4. Stores predictions (if requested)
    5. Returns comprehensive training results
    
    Args:
        model: Model wrapper to train (not yet fitted)
        X_train: Training features
        y_train: Training target (None for unsupervised)
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        X_test: Test features (optional)
        y_test: Test target (optional)
        compute_train_score: Whether to compute score on training set
        store_predictions: Whether to store predictions in result
        **fit_params: Additional parameters passed to model.fit()
    
    Returns:
        TrainingResult containing fitted model and evaluation metrics
    
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If training fails
    
    Example:
        >>> from app.ml_engine.models.classification import RandomForestClassifierWrapper
        >>> from app.ml_engine.models.base import ModelConfig
        >>> 
        >>> config = ModelConfig('random_forest', {'n_estimators': 100})
        >>> model = RandomForestClassifierWrapper(config)
        >>> 
        >>> result = train_model(
        ...     model, X_train, y_train,
        ...     X_test=X_test, y_test=y_test
        ... )
        >>> 
        >>> print(f"Train score: {result.train_score}")
        >>> print(f"Test score: {result.test_score}")
    """
    logger.info(f"Starting training for {model.config.model_type}")
    
    # Validate inputs
    _validate_training_inputs(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Record start time
    start_time = datetime.now()
    
    try:
        # Fit the model
        logger.info(f"Fitting model on {len(X_train)} training samples")
        model.fit(X_train, y_train, **fit_params)
        
        # Calculate training time
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Model fitted successfully in {training_time:.2f} seconds")
        
        # Evaluate model
        train_score = None
        val_score = None
        test_score = None
        predictions = {}
        
        # For supervised models, compute scores
        if isinstance(model, SupervisedModelWrapper):
            # Training score
            if compute_train_score:
                logger.info("Computing training score")
                train_score = model.score(X_train, y_train)
                logger.info(f"Training score: {train_score:.4f}")
            
            # Validation score
            if X_val is not None and y_val is not None:
                logger.info("Computing validation score")
                val_score = model.score(X_val, y_val)
                logger.info(f"Validation score: {val_score:.4f}")
            
            # Test score
            if X_test is not None and y_test is not None:
                logger.info("Computing test score")
                test_score = model.score(X_test, y_test)
                logger.info(f"Test score: {test_score:.4f}")
            
            # Store predictions if requested
            if store_predictions:
                logger.info("Storing predictions")
                predictions['train'] = model.predict(X_train)
                
                if X_val is not None:
                    predictions['val'] = model.predict(X_val)
                
                if X_test is not None:
                    predictions['test'] = model.predict(X_test)
        
        # For unsupervised models, store cluster labels
        elif isinstance(model, UnsupervisedModelWrapper):
            if store_predictions:
                labels = model.get_labels()
                if labels is not None:
                    predictions['train_labels'] = labels
        
        # Create metadata
        metadata = {
            'model_type': model.config.model_type,
            'task_type': model.get_task_type(),
            'hyperparameters': model.config.hyperparameters,
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val) if X_val is not None else 0,
            'n_test_samples': len(X_test) if X_test is not None else 0,
            'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
            'training_start': start_time.isoformat(),
            'training_end': end_time.isoformat()
        }
        
        # Create and return result
        result = TrainingResult(
            model=model,
            train_score=train_score,
            val_score=val_score,
            test_score=test_score,
            training_time=training_time,
            predictions=predictions,
            metadata=metadata
        )
        
        logger.info(f"Training completed successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Training failed: {str(e)}") from e


def evaluate_model(
    model: BaseModelWrapper,
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    return_predictions: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Evaluate a fitted model on given data.
    
    Args:
        model: Fitted model wrapper
        X: Features to evaluate on
        y: True labels/values (required for supervised models)
        return_predictions: Whether to return predictions along with score
    
    Returns:
        Score (and predictions if return_predictions=True)
    
    Raises:
        RuntimeError: If model is not fitted
        ValueError: If inputs are invalid
    
    Example:
        >>> score = evaluate_model(model, X_test, y_test)
        >>> print(f"Test score: {score:.4f}")
        >>> 
        >>> score, predictions = evaluate_model(
        ...     model, X_test, y_test, return_predictions=True
        ... )
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted before evaluation")
    
    logger.info(f"Evaluating model on {len(X)} samples")
    
    # For supervised models
    if isinstance(model, SupervisedModelWrapper):
        if y is None:
            raise ValueError("Target y is required for supervised model evaluation")
        
        score = model.score(X, y)
        logger.info(f"Evaluation score: {score:.4f}")
        
        if return_predictions:
            predictions = model.predict(X)
            return score, predictions
        return score
    
    # For unsupervised models (clustering)
    elif isinstance(model, UnsupervisedModelWrapper):
        predictions = model.predict(X)
        
        if return_predictions:
            return None, predictions  # No score for unsupervised
        return None
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def predict_with_model(
    model: BaseModelWrapper,
    X: Union[pd.DataFrame, np.ndarray],
    return_proba: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions with a fitted model.
    
    Args:
        model: Fitted model wrapper
        X: Features to predict on
        return_proba: Whether to return class probabilities (classification only)
    
    Returns:
        Predictions (and probabilities if return_proba=True and supported)
    
    Raises:
        RuntimeError: If model is not fitted
    
    Example:
        >>> predictions = predict_with_model(model, X_new)
        >>> 
        >>> # For classification with probabilities
        >>> predictions, probabilities = predict_with_model(
        ...     model, X_new, return_proba=True
        ... )
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted before making predictions")
    
    logger.info(f"Making predictions on {len(X)} samples")
    
    predictions = model.predict(X)
    
    if return_proba:
        try:
            probabilities = model.predict_proba(X)
            logger.info("Returning predictions with probabilities")
            return predictions, probabilities
        except AttributeError:
            logger.warning("Model does not support probability prediction, returning predictions only")
            return predictions, None
    
    return predictions


def _validate_training_inputs(
    model: BaseModelWrapper,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Optional[Union[pd.Series, np.ndarray]],
    X_val: Optional[Union[pd.DataFrame, np.ndarray]],
    y_val: Optional[Union[pd.Series, np.ndarray]],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]],
    y_test: Optional[Union[pd.Series, np.ndarray]]
) -> None:
    """
    Validate training inputs.
    
    Args:
        model: Model wrapper
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features
        y_test: Test target
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Check if model is already fitted
    if model.is_fitted:
        raise ValueError("Model is already fitted. Create a new model instance for training.")
    
    # Check training data
    if X_train is None or len(X_train) == 0:
        raise ValueError("Training features X_train cannot be empty")
    
    # For supervised models, check target
    if isinstance(model, SupervisedModelWrapper):
        if y_train is None:
            raise ValueError("Training target y_train is required for supervised learning")
        
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train must have same length. "
                f"Got X_train: {len(X_train)}, y_train: {len(y_train)}"
            )
    
    # Check validation data consistency
    if X_val is not None or y_val is not None:
        if X_val is None or y_val is None:
            raise ValueError("Both X_val and y_val must be provided together")
        
        if len(X_val) != len(y_val):
            raise ValueError(
                f"X_val and y_val must have same length. "
                f"Got X_val: {len(X_val)}, y_val: {len(y_val)}"
            )
        
        # Check feature consistency
        n_features_train = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        n_features_val = X_val.shape[1] if hasattr(X_val, 'shape') else len(X_val[0])
        
        if n_features_train != n_features_val:
            raise ValueError(
                f"X_train and X_val must have same number of features. "
                f"Got train: {n_features_train}, val: {n_features_val}"
            )
    
    # Check test data consistency
    if X_test is not None or y_test is not None:
        if X_test is None or y_test is None:
            raise ValueError("Both X_test and y_test must be provided together")
        
        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test and y_test must have same length. "
                f"Got X_test: {len(X_test)}, y_test: {len(y_test)}"
            )
        
        # Check feature consistency
        n_features_train = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        n_features_test = X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0])
        
        if n_features_train != n_features_test:
            raise ValueError(
                f"X_train and X_test must have same number of features. "
                f"Got train: {n_features_train}, test: {n_features_test}"
            )
    
    logger.debug("Training inputs validated successfully")


def get_model_info(model: BaseModelWrapper) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model: Model wrapper
    
    Returns:
        Dictionary containing model information
    
    Example:
        >>> info = get_model_info(model)
        >>> print(info['model_type'])
        >>> print(info['is_fitted'])
        >>> print(info['n_features'])
    """
    info = {
        'model_type': model.config.model_type,
        'task_type': model.get_task_type(),
        'is_fitted': model.is_fitted,
        'hyperparameters': model.config.hyperparameters,
    }
    
    if model.is_fitted:
        info.update({
            'n_train_samples': model.metadata.n_train_samples,
            'n_features': model.metadata.n_features,
            'feature_names': model.metadata.feature_names,
            'target_name': model.metadata.target_name,
            'training_duration_seconds': model.metadata.training_duration_seconds,
        })
        
        # Add feature importance if available
        try:
            feature_importance = model.get_feature_importance()
            if feature_importance:
                info['has_feature_importance'] = True
                info['feature_importance'] = feature_importance
            else:
                info['has_feature_importance'] = False
        except:
            info['has_feature_importance'] = False
    
    return info
