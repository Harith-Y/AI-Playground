from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from app.utils.logger import get_logger

logger = get_logger("preprocessing")


class PreprocessingStep(ABC):
    """
    Abstract base class for all preprocessing steps.
    
    Defines the interface for data preprocessing operations that can be:
    - Fitted on training data
    - Transformed on any dataset
    - Serialized/deserialized for reuse
    - Chained in preprocessing pipelines
    
    Attributes:
        name: Human-readable name for this preprocessing step
        fitted: Whether this step has been fitted on training data
        params: Configuration parameters for this step
    """
    
    def __init__(self, name: Optional[str] = None, **params):
        """
        Initialize preprocessing step.
        
        Args:
            name: Optional name for this step (defaults to class name)
            **params: Step-specific configuration parameters
        """
        self.name = name or self.__class__.__name__
        self.fitted = False
        self.params = params
        logger.debug(f"Initialized {self.name} with params={params}")
    
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> "PreprocessingStep":
        """
        Fit this preprocessing step on training data.
        
        Args:
            X: Training features (DataFrame or array)
            y: Optional training labels (for supervised preprocessing)
            
        Returns:
            Self (for method chaining)
        """
        pass
    
    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data using fitted parameters.
        
        Args:
            X: Data to transform (DataFrame or array)
            
        Returns:
            Transformed data in same format as input
            
        Raises:
            RuntimeError: If step has not been fitted
        """
        pass
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit on data and then transform it (convenience method).

        Args:
            X: Training features
            y: Optional training labels

        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Reverse the transformation (optional, not all steps support this).

        Args:
            X: Transformed data to reverse

        Returns:
            Data in original representation

        Raises:
            NotImplementedError: If this step doesn't support inverse transform
            RuntimeError: If step has not been fitted
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform. "
            "Only invertible transformations (e.g., scalers, some encoders) can be reversed."
        )

    def supports_inverse_transform(self) -> bool:
        """
        Check if this step supports inverse transformation.

        Returns:
            True if inverse_transform is implemented, False otherwise
        """
        # Check if inverse_transform is overridden in subclass
        return (
            self.__class__.inverse_transform is not PreprocessingStep.inverse_transform
        )

    def _check_fitted(self) -> None:
        """
        Check if step has been fitted.
        
        Raises:
            RuntimeError: If step has not been fitted
        """
        if not self.fitted:
            raise RuntimeError(f"{self.name} must be fitted before transform. Call fit() first.")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for this step.
        
        Returns:
            Dictionary of parameter names and values
        """
        return self.params.copy()
    
    def set_params(self, **params) -> "PreprocessingStep":
        """
        Set parameters for this step.
        
        Args:
            **params: Parameters to update
            
        Returns:
            Self (for method chaining)
        """
        self.params.update(params)
        logger.debug(f"Updated {self.name} params={self.params}")
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize step configuration to dictionary.
        
        Returns:
            Dictionary containing step metadata and parameters
        """
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "fitted": self.fitted,
            "params": self.params,
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PreprocessingStep":
        """
        Deserialize step from dictionary configuration.
        
        Args:
            config: Dictionary containing step configuration
            
        Returns:
            New preprocessing step instance
        """
        name = config.get("name")
        params = config.get("params", {})
        return cls(name=name, **params)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted step to disk.
        
        Args:
            path: File path to save to (will use pickle format)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "PreprocessingStep":
        """
        Load fitted step from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded preprocessing step
        """
        with open(path, "rb") as f:
            step = pickle.load(f)
        
        logger.info(f"Loaded {step.name} from {path}")
        return step
    
    def __repr__(self) -> str:
        """String representation of this step."""
        fitted_status = "fitted" if self.fitted else "not fitted"
        return f"{self.name}({fitted_status}, params={self.params})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
