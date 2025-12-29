"""
Base classes for ML model wrappers.

This module provides abstract base classes and interfaces for wrapping
scikit-learn and other ML models with consistent training, prediction,
and serialization interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path


class ModelConfig:
    """Base configuration for ML models."""

    def __init__(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any],
        random_state: Optional[int] = None,
        validate: bool = True,
        strict: bool = False,
        **kwargs
    ):
        """
        Initialize model configuration.

        Args:
            model_type: Type/name of the model (e.g., 'random_forest_classifier')
            hyperparameters: Model hyperparameters
            random_state: Random seed for reproducibility
            validate: Whether to validate hyperparameters
            strict: If True, reject unknown parameters during validation
            **kwargs: Additional configuration parameters
        
        Raises:
            ValueError: If validation is enabled and hyperparameters are invalid
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self.additional_config = kwargs
        
        # Validate hyperparameters if requested
        if validate:
            self._validate_hyperparameters(strict=strict)
    
    def _validate_hyperparameters(self, strict: bool = False) -> None:
        """
        Validate hyperparameters against model schema.
        
        Args:
            strict: If True, reject unknown parameters
        
        Raises:
            ValueError: If validation fails
        """
        # Import here to avoid circular dependency
        from app.ml_engine.models.validation import validate_model_config
        
        is_valid, errors = validate_model_config(
            self.model_type,
            self.hyperparameters,
            strict=strict
        )
        
        if not is_valid:
            error_msg = f"Invalid hyperparameters for model '{self.model_type}':\n"
            error_msg += "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "random_state": self.random_state,
            **self.additional_config
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any], validate: bool = True) -> 'ModelConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config: Configuration dictionary
            validate: Whether to validate hyperparameters
        
        Returns:
            ModelConfig instance
        """
        return cls(
            model_type=config["model_type"],
            hyperparameters=config["hyperparameters"],
            random_state=config.get("random_state"),
            validate=validate,
            **{k: v for k, v in config.items() if k not in ["model_type", "hyperparameters", "random_state"]}
        )


class TrainingMetadata:
    """Metadata about model training."""

    def __init__(
        self,
        train_start_time: Optional[datetime] = None,
        train_end_time: Optional[datetime] = None,
        training_duration_seconds: Optional[float] = None,
        n_train_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        sklearn_version: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize training metadata.

        Args:
            train_start_time: When training started
            train_end_time: When training ended
            training_duration_seconds: Training duration
            n_train_samples: Number of training samples
            n_features: Number of features
            feature_names: Names of features
            target_name: Name of target variable
            sklearn_version: Version of scikit-learn used
            **kwargs: Additional metadata
        """
        self.train_start_time = train_start_time
        self.train_end_time = train_end_time
        self.training_duration_seconds = training_duration_seconds
        self.n_train_samples = n_train_samples
        self.n_features = n_features
        self.feature_names = feature_names or []
        self.target_name = target_name
        self.sklearn_version = sklearn_version
        self.additional_metadata = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "train_start_time": self.train_start_time.isoformat() if self.train_start_time else None,
            "train_end_time": self.train_end_time.isoformat() if self.train_end_time else None,
            "training_duration_seconds": self.training_duration_seconds,
            "n_train_samples": self.n_train_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "sklearn_version": self.sklearn_version,
            **self.additional_metadata
        }


class BaseModelWrapper(ABC):
    """
    Abstract base class for ML model wrappers.

    All model wrappers must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize model wrapper.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.is_fitted = False
        self.metadata = TrainingMetadata()
        self._feature_names: List[str] = []
        self._target_name: Optional[str] = None

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create the underlying sklearn model instance.

        Returns:
            Scikit-learn model instance

        This method must be implemented by subclasses to create
        the specific sklearn model with the configured hyperparameters.
        """
        pass

    @abstractmethod
    def get_task_type(self) -> str:
        """
        Get the task type for this model.

        Returns:
            Task type: 'regression', 'classification', or 'clustering'
        """
        pass

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
        **fit_params
    ) -> 'BaseModelWrapper':
        """
        Train the model.

        Args:
            X: Training features
            y: Training target (None for unsupervised)
            **fit_params: Additional parameters for model fitting

        Returns:
            self (fitted model wrapper)
        """
        # Record training start
        self.metadata.train_start_time = datetime.now()

        # Store feature and target information
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if y is not None:
            if isinstance(y, pd.Series):
                self._target_name = y.name or "target"
                y_array = y.values
            else:
                y_array = y
                self._target_name = "target"
        else:
            y_array = None

        # Update metadata
        self.metadata.n_train_samples = len(X)
        self.metadata.n_features = X_array.shape[1]
        self.metadata.feature_names = self._feature_names
        self.metadata.target_name = self._target_name

        # Create model if not already created
        if self.model is None:
            self.model = self._create_model()

        # Fit the model
        if y_array is not None:
            self.model.fit(X_array, y_array, **fit_params)
        else:
            # Unsupervised learning (clustering)
            self.model.fit(X_array, **fit_params)

        self.is_fitted = True

        # Record training end
        self.metadata.train_end_time = datetime.now()
        self.metadata.training_duration_seconds = (
            self.metadata.train_end_time - self.metadata.train_start_time
        ).total_seconds()

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            X: Features to predict on

        Returns:
            Class probabilities

        Raises:
            RuntimeError: If model hasn't been fitted
            AttributeError: If model doesn't support probability prediction
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"Model {self.config.model_type} does not support probability prediction")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_array)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support feature importance
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")

        # Check for feature_importances_ attribute (tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self._feature_names, importances))

        # Check for coef_ attribute (linear models)
        elif hasattr(self.model, 'coef_'):
            # For multi-class, coef_ is 2D, take mean of absolute values
            coef = self.model.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
            return dict(zip(self._feature_names, importances))

        return None

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save model that hasn't been fitted")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model object and metadata
        save_dict = {
            "model": self.model,
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "feature_names": self._feature_names,
            "target_name": self._target_name,
            "is_fitted": self.is_fitted
        }

        joblib.dump(save_dict, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModelWrapper':
        """
        Load model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model wrapper
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load saved data
        save_dict = joblib.load(path)

        # Reconstruct model wrapper
        config = ModelConfig.from_dict(save_dict["config"])
        wrapper = cls(config)
        wrapper.model = save_dict["model"]
        wrapper._feature_names = save_dict["feature_names"]
        wrapper._target_name = save_dict["target_name"]
        wrapper.is_fitted = save_dict["is_fitted"]

        # Reconstruct metadata
        metadata_dict = save_dict["metadata"]
        wrapper.metadata = TrainingMetadata(
            train_start_time=datetime.fromisoformat(metadata_dict["train_start_time"]) if metadata_dict.get("train_start_time") else None,
            train_end_time=datetime.fromisoformat(metadata_dict["train_end_time"]) if metadata_dict.get("train_end_time") else None,
            training_duration_seconds=metadata_dict.get("training_duration_seconds"),
            n_train_samples=metadata_dict.get("n_train_samples"),
            n_features=metadata_dict.get("n_features"),
            feature_names=metadata_dict.get("feature_names"),
            target_name=metadata_dict.get("target_name"),
            sklearn_version=metadata_dict.get("sklearn_version")
        )

        return wrapper

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}

    def set_params(self, **params) -> 'BaseModelWrapper':
        """
        Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            self
        """
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)

        # Also update config
        self.config.hyperparameters.update(params)

        return self

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(model_type='{self.config.model_type}', status='{status}')"


class SupervisedModelWrapper(BaseModelWrapper):
    """Base class for supervised learning models (regression and classification)."""

    @abstractmethod
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate model score.

        Args:
            X: Test features
            y: True labels/values

        Returns:
            Model score (e.g., R² for regression, accuracy for classification)
        """
        pass


class UnsupervisedModelWrapper(BaseModelWrapper):
    """Base class for unsupervised learning models (clustering)."""

    def get_labels(self) -> Optional[np.ndarray]:
        """
        Get cluster labels.

        Returns:
            Cluster labels for training data, or None if not fitted
        """
        if not self.is_fitted:
            return None

        if hasattr(self.model, 'labels_'):
            return self.model.labels_

        return None

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers.

        Returns:
            Cluster centers, or None if model doesn't have cluster centers
        """
        if not self.is_fitted:
            return None

        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_

        return None
