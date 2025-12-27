"""
Preprocessing Pipeline Module.

Provides a Pipeline class for chaining multiple preprocessing steps together.
The pipeline follows the fit/transform pattern and can be serialized for reuse.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import pickle
import json
import time
from datetime import datetime

from app.ml_engine.preprocessing.base import PreprocessingStep
from app.utils.logger import get_logger

logger = get_logger("pipeline")


class Pipeline:
    """
    Sequential pipeline for chaining preprocessing steps.

    A Pipeline chains multiple preprocessing steps together, allowing you to:
    - Fit all steps sequentially on training data
    - Transform data through all fitted steps
    - Serialize/deserialize the entire pipeline
    - Track statistics and metadata for each step
    - Manage steps dynamically (add, remove, reorder)

    The Pipeline follows scikit-learn's fit/transform interface and can be used
    as a single preprocessing unit that encapsulates multiple transformations.

    Example:
        >>> from app.ml_engine.preprocessing.scaler import StandardScaler
        >>> from app.ml_engine.preprocessing.imputer import MeanImputer
        >>>
        >>> pipeline = Pipeline(steps=[
        ...     MeanImputer(columns=['age', 'salary']),
        ...     StandardScaler(columns=['age', 'salary'])
        ... ])
        >>> pipeline.fit(X_train, y_train)
        >>> X_transformed = pipeline.transform(X_test)

    Attributes:
        steps: List of preprocessing steps in execution order
        fitted: Whether the pipeline has been fitted
        step_statistics: Statistics collected during fit/transform
        metadata: Additional pipeline metadata
    """

    def __init__(
        self,
        steps: Optional[List[PreprocessingStep]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            steps: List of preprocessing step instances (in execution order)
            name: Optional name for this pipeline
        """
        self.steps: List[PreprocessingStep] = steps or []
        self.name = name or "PreprocessingPipeline"
        self.fitted = False
        self.step_statistics: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "fitted_at": None,
            "num_steps": len(self.steps)
        }

        logger.info(f"Initialized pipeline '{self.name}' with {len(self.steps)} steps")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "Pipeline":
        """
        Fit all steps in the pipeline sequentially.

        Each step is fitted on the output of the previous step's transformation.
        This means step i is fitted on the data transformed by steps 0 to i-1.

        Args:
            X: Training features (DataFrame or array)
            y: Optional training labels (for supervised preprocessing)

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If no steps in pipeline
            RuntimeError: If any step fails during fitting
        """
        if not self.steps:
            raise ValueError("Pipeline has no steps. Add steps before fitting.")

        logger.info(f"Fitting pipeline '{self.name}' with {len(self.steps)} steps")

        # Reset statistics
        self.step_statistics = []

        # Fit each step sequentially
        X_current = X

        for idx, step in enumerate(self.steps):
            try:
                step_start_time = time.time()
                logger.debug(f"Fitting step {idx + 1}/{len(self.steps)}: {step.name}")

                # Fit step
                step.fit(X_current, y)

                # Transform data for next step
                X_current = step.transform(X_current)

                step_duration = time.time() - step_start_time

                # Collect statistics
                step_stats = {
                    "step_index": idx,
                    "step_name": step.name,
                    "step_class": step.__class__.__name__,
                    "fit_duration_seconds": round(step_duration, 4),
                    "output_shape": X_current.shape if hasattr(X_current, 'shape') else None,
                    "fitted": step.fitted,
                    "params": step.get_params()
                }

                self.step_statistics.append(step_stats)

                logger.info(
                    f"Step {idx + 1}/{len(self.steps)} ({step.name}) fitted "
                    f"in {step_duration:.3f}s"
                )

            except Exception as e:
                logger.error(f"Error fitting step {idx + 1} ({step.name}): {str(e)}")
                raise RuntimeError(
                    f"Pipeline fitting failed at step {idx + 1} ({step.name}): {str(e)}"
                ) from e

        self.fitted = True
        self.metadata["fitted_at"] = datetime.now().isoformat()
        self.metadata["num_samples_fitted"] = len(X)

        logger.info(f"Pipeline '{self.name}' fitted successfully")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data through all fitted steps in the pipeline.

        Args:
            X: Data to transform (DataFrame or array)

        Returns:
            Transformed data in same format as input

        Raises:
            RuntimeError: If pipeline has not been fitted
            RuntimeError: If any step fails during transformation
        """
        if not self.fitted:
            raise RuntimeError(
                f"Pipeline '{self.name}' must be fitted before transform. "
                "Call fit() or fit_transform() first."
            )

        if not self.steps:
            logger.warning("Pipeline has no steps, returning input unchanged")
            return X

        logger.debug(f"Transforming data through {len(self.steps)} steps")

        X_current = X

        for idx, step in enumerate(self.steps):
            try:
                logger.debug(f"Applying step {idx + 1}/{len(self.steps)}: {step.name}")
                X_current = step.transform(X_current)

            except Exception as e:
                logger.error(f"Error transforming step {idx + 1} ({step.name}): {str(e)}")
                raise RuntimeError(
                    f"Pipeline transformation failed at step {idx + 1} ({step.name}): {str(e)}"
                ) from e

        logger.debug(f"Pipeline transformation complete")

        return X_current

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit all steps and transform data (convenience method).

        Args:
            X: Training features
            y: Optional training labels

        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)

    # Step Management Methods

    def add_step(
        self,
        step: PreprocessingStep,
        position: Optional[int] = None
    ) -> "Pipeline":
        """
        Add a preprocessing step to the pipeline.

        Args:
            step: Preprocessing step to add
            position: Optional position to insert step (None = append to end)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If step is not a PreprocessingStep instance
        """
        if not isinstance(step, PreprocessingStep):
            raise TypeError(
                f"Step must be a PreprocessingStep instance, got {type(step)}"
            )

        if position is None:
            self.steps.append(step)
            logger.info(f"Added step '{step.name}' to end of pipeline")
        else:
            if position < 0 or position > len(self.steps):
                raise ValueError(
                    f"Invalid position {position}. Must be between 0 and {len(self.steps)}"
                )
            self.steps.insert(position, step)
            logger.info(f"Inserted step '{step.name}' at position {position}")

        # Reset fitted state since pipeline structure changed
        self.fitted = False
        self.metadata["num_steps"] = len(self.steps)

        return self

    def remove_step(self, index: int) -> PreprocessingStep:
        """
        Remove a step from the pipeline by index.

        Args:
            index: Index of step to remove

        Returns:
            Removed preprocessing step

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.steps):
            raise IndexError(
                f"Step index {index} out of range. Pipeline has {len(self.steps)} steps."
            )

        removed_step = self.steps.pop(index)
        logger.info(f"Removed step '{removed_step.name}' at index {index}")

        # Reset fitted state
        self.fitted = False
        self.metadata["num_steps"] = len(self.steps)

        return removed_step

    def remove_step_by_name(self, name: str) -> PreprocessingStep:
        """
        Remove a step from the pipeline by name.

        Args:
            name: Name of step to remove

        Returns:
            Removed preprocessing step

        Raises:
            ValueError: If no step with given name found
        """
        for idx, step in enumerate(self.steps):
            if step.name == name:
                return self.remove_step(idx)

        raise ValueError(f"No step with name '{name}' found in pipeline")

    def get_step(self, index: int) -> PreprocessingStep:
        """
        Get a step from the pipeline by index.

        Args:
            index: Index of step to retrieve

        Returns:
            Preprocessing step at given index
        """
        return self.steps[index]

    def get_step_by_name(self, name: str) -> Optional[PreprocessingStep]:
        """
        Get a step from the pipeline by name.

        Args:
            name: Name of step to retrieve

        Returns:
            Preprocessing step with given name, or None if not found
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def reorder_steps(self, new_order: List[int]) -> "Pipeline":
        """
        Reorder steps in the pipeline.

        Args:
            new_order: List of indices specifying new order
                      Example: [2, 0, 1] moves step 2 to position 0

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If new_order is invalid
        """
        if len(new_order) != len(self.steps):
            raise ValueError(
                f"new_order must have {len(self.steps)} indices, got {len(new_order)}"
            )

        if sorted(new_order) != list(range(len(self.steps))):
            raise ValueError(
                f"new_order must contain each index 0-{len(self.steps)-1} exactly once"
            )

        self.steps = [self.steps[i] for i in new_order]
        logger.info(f"Reordered pipeline steps: {new_order}")

        # Reset fitted state
        self.fitted = False

        return self

    def clear(self) -> "Pipeline":
        """
        Remove all steps from the pipeline.

        Returns:
            Self (for method chaining)
        """
        self.steps.clear()
        self.fitted = False
        self.step_statistics.clear()
        self.metadata["num_steps"] = 0
        logger.info("Cleared all steps from pipeline")
        return self

    # Information and Statistics Methods

    def get_num_steps(self) -> int:
        """
        Get the number of steps in the pipeline.

        Returns:
            Number of steps
        """
        return len(self.steps)

    def get_step_names(self) -> List[str]:
        """
        Get names of all steps in the pipeline.

        Returns:
            List of step names in execution order
        """
        return [step.name for step in self.steps]

    def get_step_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics collected during pipeline fitting.

        Returns:
            List of statistics dictionaries for each step
        """
        return self.step_statistics.copy()

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the pipeline.

        Returns:
            Dictionary with pipeline information and statistics
        """
        return {
            "name": self.name,
            "num_steps": len(self.steps),
            "fitted": self.fitted,
            "steps": [
                {
                    "index": idx,
                    "name": step.name,
                    "class": step.__class__.__name__,
                    "fitted": step.fitted,
                    "params": step.get_params()
                }
                for idx, step in enumerate(self.steps)
            ],
            "metadata": self.metadata,
            "statistics": self.step_statistics
        }

    # Serialization Methods

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize pipeline configuration to dictionary.

        Note: This saves configuration only, not fitted parameters.
        Use save() to persist fitted pipeline with pickle.

        Returns:
            Dictionary containing pipeline configuration
        """
        return {
            "name": self.name,
            "fitted": self.fitted,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
            "statistics": self.step_statistics
        }

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        step_registry: Optional[Dict[str, type]] = None
    ) -> "Pipeline":
        """
        Deserialize pipeline from dictionary configuration.

        Note: This loads configuration only, not fitted parameters.
        Steps will need to be re-fitted. Use load() to restore fitted pipeline.

        Args:
            config: Dictionary containing pipeline configuration
            step_registry: Optional mapping of class names to step classes
                          If None, uses default registry

        Returns:
            New pipeline instance

        Raises:
            ValueError: If step class not found in registry
        """
        if step_registry is None:
            step_registry = _get_default_step_registry()

        # Reconstruct steps
        steps = []
        for step_config in config.get("steps", []):
            step_class_name = step_config.get("class")

            if step_class_name not in step_registry:
                raise ValueError(
                    f"Unknown preprocessing step class: {step_class_name}. "
                    f"Available: {list(step_registry.keys())}"
                )

            step_class = step_registry[step_class_name]
            step = step_class.from_dict(step_config)
            steps.append(step)

        # Create pipeline
        pipeline = cls(steps=steps, name=config.get("name"))
        pipeline.fitted = config.get("fitted", False)
        pipeline.metadata = config.get("metadata", {})
        pipeline.step_statistics = config.get("statistics", [])

        logger.info(f"Loaded pipeline '{pipeline.name}' from config")

        return pipeline

    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted pipeline to disk using pickle.

        This saves the entire pipeline including all fitted parameters.

        Args:
            path: File path to save to (will use .pkl extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Saved pipeline '{self.name}' to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Pipeline":
        """
        Load fitted pipeline from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded pipeline with fitted parameters
        """
        with open(path, "rb") as f:
            pipeline = pickle.load(f)

        logger.info(f"Loaded pipeline '{pipeline.name}' from {path}")
        return pipeline

    def save_config(self, path: Union[str, Path]) -> None:
        """
        Save pipeline configuration to JSON file.

        This saves only the configuration, not fitted parameters.

        Args:
            path: File path to save to (will use .json extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = self.to_dict()

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved pipeline config to {path}")

    @classmethod
    def load_config(
        cls,
        path: Union[str, Path],
        step_registry: Optional[Dict[str, type]] = None
    ) -> "Pipeline":
        """
        Load pipeline configuration from JSON file.

        Pipeline will need to be re-fitted after loading.

        Args:
            path: File path to load from
            step_registry: Optional mapping of class names to step classes

        Returns:
            New pipeline instance (not fitted)
        """
        with open(path, "r") as f:
            config = json.load(f)

        pipeline = cls.from_dict(config, step_registry=step_registry)
        logger.info(f"Loaded pipeline config from {path}")

        return pipeline

    # Special Methods

    def __len__(self) -> int:
        """Return number of steps in pipeline."""
        return len(self.steps)

    def __getitem__(self, index: int) -> PreprocessingStep:
        """Get step by index using bracket notation."""
        return self.steps[index]

    def __repr__(self) -> str:
        """String representation of pipeline."""
        fitted_status = "fitted" if self.fitted else "not fitted"
        step_names = ", ".join(self.get_step_names())
        return f"Pipeline('{self.name}', {len(self.steps)} steps, {fitted_status})\nSteps: [{step_names}]"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


def _get_default_step_registry() -> Dict[str, type]:
    """
    Get default registry of preprocessing step classes.

    Returns:
        Dictionary mapping class names to step classes
    """
    # Import all preprocessing step classes
    from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler, RobustScaler
    from app.ml_engine.preprocessing.encoder import OneHotEncoder, LabelEncoder, OrdinalEncoder
    from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer, ModeImputer
    from app.ml_engine.preprocessing.cleaner import IQROutlierDetector, ZScoreOutlierDetector
    from app.ml_engine.preprocessing.oversampling import SMOTE, BorderlineSMOTE, ADASYN
    from app.ml_engine.preprocessing.undersampling import (
        RandomUnderSampler,
        NearMissUnderSampler,
        TomekLinksRemover
    )

    return {
        # Scalers
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,

        # Encoders
        "OneHotEncoder": OneHotEncoder,
        "LabelEncoder": LabelEncoder,
        "OrdinalEncoder": OrdinalEncoder,

        # Imputers
        "MeanImputer": MeanImputer,
        "MedianImputer": MedianImputer,
        "ModeImputer": ModeImputer,

        # Outlier Detection
        "IQROutlierDetector": IQROutlierDetector,
        "ZScoreOutlierDetector": ZScoreOutlierDetector,

        # Oversampling
        "SMOTE": SMOTE,
        "BorderlineSMOTE": BorderlineSMOTE,
        "ADASYN": ADASYN,

        # Undersampling
        "RandomUnderSampler": RandomUnderSampler,
        "NearMissUnderSampler": NearMissUnderSampler,
        "TomekLinksRemover": TomekLinksRemover,
    }


# Convenience Functions

def create_pipeline(steps: List[PreprocessingStep], name: Optional[str] = None) -> Pipeline:
    """
    Create a preprocessing pipeline (convenience function).

    Args:
        steps: List of preprocessing steps
        name: Optional pipeline name

    Returns:
        New pipeline instance

    Example:
        >>> pipeline = create_pipeline([
        ...     MeanImputer(columns=['age']),
        ...     StandardScaler(columns=['age', 'salary'])
        ... ])
    """
    return Pipeline(steps=steps, name=name)


def load_pipeline(path: Union[str, Path]) -> Pipeline:
    """
    Load a fitted pipeline from disk (convenience function).

    Args:
        path: File path to load from

    Returns:
        Loaded pipeline

    Example:
        >>> pipeline = load_pipeline('models/preprocessing_pipeline.pkl')
    """
    return Pipeline.load(path)
