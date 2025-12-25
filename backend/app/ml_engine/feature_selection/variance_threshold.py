"""
Variance Threshold Feature Selection

Removes features with low variance (near-constant features).
Features with variance below a threshold are removed as they provide little information.
"""

from typing import List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.base import PreprocessingStep
from app.utils.logger import get_logger

logger = get_logger("feature_selection.variance_threshold")


class VarianceThreshold(PreprocessingStep):
    """
    Feature selector that removes low-variance features.

    Features with variance below the threshold are considered near-constant
    and removed from the dataset. This is useful for:
    - Removing constant or quasi-constant features
    - Reducing dimensionality
    - Improving model training speed
    - Preventing overfitting on noise

    The variance is calculated differently based on feature type:
    - Numeric features: Standard variance (mean of squared deviations)
    - Binary features (0/1): p * (1 - p) where p is proportion of 1s

    Parameters
    ----------
    threshold : float, default=0.0
        Features with variance below this threshold will be removed.
        - threshold=0.0: Remove only constant features (default)
        - threshold=0.01: Remove features with very low variance
        - For binary features: threshold=0.16 removes features with >80% same value

    columns : list of str, optional
        Specific columns to apply variance threshold to.
        If None, applies to all numeric columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from ml_engine.feature_selection import VarianceThreshold
    >>>
    >>> # Example dataset with one constant and one low-variance feature
    >>> df = pd.DataFrame({
    ...     'constant': [1, 1, 1, 1, 1],
    ...     'low_var': [1, 1, 1, 1, 2],
    ...     'high_var': [1, 2, 3, 4, 5],
    ...     'target': [0, 1, 0, 1, 0]
    ... })
    >>>
    >>> # Remove constant features
    >>> selector = VarianceThreshold(threshold=0.0)
    >>> df_transformed = selector.fit_transform(df)
    >>> print(df_transformed.columns)  # ['low_var', 'high_var', 'target']
    >>>
    >>> # Remove low-variance features (variance < 0.2)
    >>> selector = VarianceThreshold(threshold=0.2)
    >>> df_transformed = selector.fit_transform(df)
    >>> print(df_transformed.columns)  # ['high_var', 'target']
    """

    def __init__(
        self,
        threshold: float = 0.0,
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize VarianceThreshold selector.

        Args:
            threshold: Variance threshold below which features are removed
            columns: Specific columns to apply threshold to (None = all numeric)
            name: Optional custom name for this step
        """
        super().__init__(name=name, threshold=threshold, columns=columns)
        self.threshold = threshold
        self.columns = columns

        # Will be set during fit
        self.variances_: Optional[pd.Series] = None
        self.selected_features_: Optional[List[str]] = None
        self.removed_features_: Optional[List[str]] = None

        logger.debug(f"Initialized VarianceThreshold(threshold={threshold}, columns={columns})")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "VarianceThreshold":
        """
        Compute variance for each feature and determine which to keep.

        Args:
            X: Training data (DataFrame or array)
            y: Not used, present for API consistency

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If threshold is negative or X has no numeric columns
        """
        if self.threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {self.threshold}")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Determine which columns to analyze
        if self.columns is not None:
            # Use specified columns
            analyze_cols = [col for col in self.columns if col in X.columns]
            if len(analyze_cols) == 0:
                raise ValueError(f"None of the specified columns found in data: {self.columns}")
        else:
            # Use all numeric columns
            analyze_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(analyze_cols) == 0:
                raise ValueError("No numeric columns found in data")

        logger.info(f"Analyzing variance for {len(analyze_cols)} features")

        # Calculate variance for each column
        self.variances_ = X[analyze_cols].var()

        # Select features with variance above threshold
        mask = self.variances_ >= self.threshold
        self.selected_features_ = self.variances_[mask].index.tolist()
        self.removed_features_ = self.variances_[~mask].index.tolist()

        # Add non-numeric columns to selected features (they pass through)
        non_numeric_cols = [col for col in X.columns if col not in analyze_cols]
        self.selected_features_ = non_numeric_cols + self.selected_features_

        self.fitted = True

        logger.info(
            f"Variance threshold fit complete: "
            f"kept {len(self.selected_features_)} features, "
            f"removed {len(self.removed_features_)} features"
        )

        if self.removed_features_:
            logger.debug(f"Removed features: {self.removed_features_}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Remove low-variance features from data.

        Args:
            X: Data to transform (DataFrame or array)

        Returns:
            Data with low-variance features removed (same type as input)

        Raises:
            RuntimeError: If not fitted yet
            ValueError: If X doesn't have expected columns
        """
        self._check_fitted()

        is_array = isinstance(X, np.ndarray)

        # Convert to DataFrame if needed
        if is_array:
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        # Check that we have the expected columns
        missing_cols = set(self.selected_features_) - set(X_df.columns)
        if missing_cols:
            logger.warning(
                f"Some selected features not found in transform data: {missing_cols}"
            )
            # Use only available columns
            available_features = [col for col in self.selected_features_ if col in X_df.columns]
        else:
            available_features = self.selected_features_

        # Select only the high-variance features
        X_transformed = X_df[available_features]

        logger.debug(
            f"Transformed data from {X_df.shape[1]} to {X_transformed.shape[1]} features"
        )

        # Convert back to array if input was array
        if is_array:
            return X_transformed.values
        else:
            return X_transformed

    def get_feature_variances(self) -> Optional[pd.Series]:
        """
        Get variance values for all features.

        Returns:
            Series with feature names as index and variance values
            None if not fitted yet
        """
        if not self.fitted:
            return None
        return self.variances_.copy()

    def get_selected_features(self) -> Optional[List[str]]:
        """
        Get list of selected (kept) features.

        Returns:
            List of feature names that passed the threshold
            None if not fitted yet
        """
        if not self.fitted:
            return None
        return self.selected_features_.copy()

    def get_removed_features(self) -> Optional[List[str]]:
        """
        Get list of removed features.

        Returns:
            List of feature names that were removed
            None if not fitted yet
        """
        if not self.fitted:
            return None
        return self.removed_features_.copy()

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask or indices of selected features.

        Args:
            indices: If True, return feature indices. If False, return boolean mask.

        Returns:
            Boolean mask or integer indices of selected features

        Raises:
            RuntimeError: If not fitted yet
        """
        self._check_fitted()

        if self.variances_ is None:
            return np.array([])

        mask = self.variances_ >= self.threshold

        if indices:
            return np.where(mask)[0].tolist()
        else:
            return mask.values

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize variance threshold configuration to dictionary.

        Returns:
            Dictionary with configuration and fitted parameters
        """
        config = super().to_dict()

        if self.fitted:
            config["variances"] = self.variances_.to_dict() if self.variances_ is not None else None
            config["selected_features"] = self.selected_features_
            config["removed_features"] = self.removed_features_

        return config

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VarianceThreshold":
        """
        Create VarianceThreshold from dictionary configuration.

        Args:
            config: Dictionary with configuration

        Returns:
            New VarianceThreshold instance
        """
        # Extract init parameters
        name = config.get("name")
        params = config.get("params", {})

        # Create instance
        instance = cls(name=name, **params)

        # Restore fitted state if available
        if config.get("fitted", False):
            instance.fitted = True
            if "variances" in config and config["variances"] is not None:
                instance.variances_ = pd.Series(config["variances"])
            instance.selected_features_ = config.get("selected_features")
            instance.removed_features_ = config.get("removed_features")

        return instance

    def __repr__(self) -> str:
        """String representation."""
        if self.fitted and self.selected_features_ is not None:
            return (
                f"VarianceThreshold(threshold={self.threshold}, "
                f"n_features_in={len(self.variances_)}, "
                f"n_features_out={len(self.selected_features_)})"
            )
        else:
            return f"VarianceThreshold(threshold={self.threshold}, not fitted)"
