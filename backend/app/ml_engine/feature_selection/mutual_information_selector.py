"""
Mutual Information-based feature selection.

This module provides feature selection based on mutual information (MI) scores,
which measure the dependency between features and the target variable.
MI can capture non-linear relationships unlike correlation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Literal
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from app.ml_engine.preprocessing.base import PreprocessingStep


class MutualInformationSelector(PreprocessingStep):
    """
    Selects features based on mutual information with the target variable.

    Mutual Information (MI) measures the dependency between a feature and the target.
    Unlike correlation, MI can detect non-linear relationships and is useful for
    both classification and regression tasks.

    Key advantages:
    - Captures non-linear relationships
    - Works with both continuous and discrete features
    - No assumption about relationship type
    - Robust to outliers

    Parameters
    ----------
    task : {'classification', 'regression'}, default='classification'
        Type of supervised learning task:
        - 'classification': Use mutual_info_classif (discrete target)
        - 'regression': Use mutual_info_regression (continuous target)

    k : int, optional
        Number of top features to select based on MI scores
        If None, uses threshold parameter

    threshold : float, default=0.0
        Minimum MI score threshold for feature selection
        Features with MI < threshold are dropped
        Used when k is None

    percentile : int, optional
        Select features with MI scores in top percentile (0-100)
        Alternative to k and threshold parameters

    discrete_features : {'auto', 'infer'} or array-like, default='auto'
        Indicates which features are discrete:
        - 'auto': All features treated as continuous
        - 'infer': Automatically detect discrete features (integers with few unique values)
        - array-like: Boolean mask or indices of discrete features

    n_neighbors : int, default=3
        Number of neighbors for MI estimation (used for continuous features)
        Higher values = smoother estimates but less local detail

    random_state : int, optional
        Random seed for reproducibility

    columns : list of str, optional
        Specific columns to consider for selection
        If None, uses all numeric columns

    Example
    -------
    >>> # Classification task
    >>> selector = MutualInformationSelector(task='classification', k=10)
    >>> X_selected = selector.fit_transform(X_train, y_train)
    >>>
    >>> # Regression task with threshold
    >>> selector = MutualInformationSelector(task='regression', threshold=0.1)
    >>> X_selected = selector.fit_transform(X_train, y_train)
    """

    def __init__(
        self,
        task: Literal['classification', 'regression'] = 'classification',
        k: Optional[int] = None,
        threshold: float = 0.0,
        percentile: Optional[int] = None,
        discrete_features: Union[str, List[bool], List[int]] = 'auto',
        n_neighbors: int = 3,
        random_state: Optional[int] = None,
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize MutualInformationSelector.

        Args:
            task: Type of supervised learning task
            k: Number of top features to select
            threshold: Minimum MI score threshold
            percentile: Select top percentile of features
            discrete_features: Which features are discrete
            n_neighbors: Number of neighbors for MI estimation
            random_state: Random seed
            columns: Specific columns to consider
            name: Optional custom name

        Raises:
            ValueError: If task is invalid or parameter conflicts exist
        """
        if task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")

        if k is not None and k <= 0:
            raise ValueError("k must be positive")

        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        if percentile is not None and not (0 < percentile <= 100):
            raise ValueError("percentile must be between 0 and 100")

        # Check for conflicting parameters
        param_count = sum([k is not None, percentile is not None])
        if param_count > 1:
            raise ValueError("Only one of k, percentile can be specified")

        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")

        super().__init__(
            name=name,
            task=task,
            k=k,
            threshold=threshold,
            percentile=percentile,
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            random_state=random_state,
            columns=columns
        )

        # Fitted parameters
        self.selected_features_: Optional[List[str]] = None
        self.mi_scores_: Optional[pd.Series] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> "MutualInformationSelector":
        """
        Compute mutual information scores and select features.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Target variable (required)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
            ValueError: If y is not provided
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MutualInformationSelector expects a pandas DataFrame")

        if y is None:
            raise ValueError("y (target) must be provided for mutual information calculation")

        task = self.params["task"]
        k = self.params["k"]
        threshold = self.params["threshold"]
        percentile = self.params["percentile"]
        discrete_features = self.params["discrete_features"]
        n_neighbors = self.params["n_neighbors"]
        random_state = self.params["random_state"]
        columns = self.params["columns"]

        # Determine columns to analyze
        if columns is not None:
            analyze_cols = [col for col in columns if col in X.columns]
        else:
            analyze_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(analyze_cols) == 0:
            raise ValueError("No numeric columns found to analyze")

        X_subset = X[analyze_cols]
        self.feature_names_ = analyze_cols

        # Convert y to numpy array if Series
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Prepare discrete_features parameter
        if discrete_features == 'infer':
            # Auto-detect discrete features (integers with few unique values)
            discrete_mask = self._infer_discrete_features(X_subset)
        elif discrete_features == 'auto':
            discrete_mask = 'auto'
        else:
            discrete_mask = discrete_features

        # Compute mutual information scores
        if task == 'classification':
            mi_scores = mutual_info_classif(
                X_subset,
                y_array,
                discrete_features=discrete_mask,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
        else:  # regression
            mi_scores = mutual_info_regression(
                X_subset,
                y_array,
                discrete_features=discrete_mask,
                n_neighbors=n_neighbors,
                random_state=random_state
            )

        # Store MI scores
        self.mi_scores_ = pd.Series(mi_scores, index=analyze_cols)

        # Select features based on criteria
        if k is not None:
            # Select top k features
            top_indices = np.argsort(mi_scores)[-k:]
            self.selected_features_ = [analyze_cols[i] for i in top_indices]
        elif percentile is not None:
            # Select top percentile
            threshold_value = np.percentile(mi_scores, 100 - percentile)
            self.selected_features_ = [
                col for col, score in zip(analyze_cols, mi_scores)
                if score >= threshold_value
            ]
        else:
            # Use threshold
            self.selected_features_ = [
                col for col, score in zip(analyze_cols, mi_scores)
                if score >= threshold
            ]

        self.fitted = True
        return self

    def _infer_discrete_features(self, X: pd.DataFrame) -> List[bool]:
        """
        Automatically infer which features are discrete.

        A feature is considered discrete if:
        1. It has integer dtype
        2. It has relatively few unique values (< 20 or < 5% of samples)

        Args:
            X: DataFrame to analyze

        Returns:
            Boolean mask indicating discrete features
        """
        discrete_mask = []
        for col in X.columns:
            # Check if integer dtype
            is_integer = X[col].dtype in [np.int32, np.int64, int]

            # Check number of unique values
            n_unique = X[col].nunique()
            n_samples = len(X)

            # Consider discrete if integer with few unique values
            is_discrete = is_integer and (n_unique < 20 or n_unique < 0.05 * n_samples)
            discrete_mask.append(is_discrete)

        return discrete_mask

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data by selecting only features with high MI scores.

        Args:
            X: Data to transform (must be a pandas DataFrame)

        Returns:
            DataFrame with only selected features

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("MutualInformationSelector expects a pandas DataFrame")

        # Check for missing features
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in DataFrame")

        return X[self.selected_features_].copy()

    def get_selected_features(self) -> List[str]:
        """
        Get list of selected feature names.

        Returns:
            List of selected feature names

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()
        return self.selected_features_.copy()

    def get_mi_scores(self) -> pd.Series:
        """
        Get mutual information scores for all features.

        Returns:
            Series with MI scores indexed by feature names

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()
        return self.mi_scores_.copy()

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance summary sorted by MI scores.

        Returns:
            DataFrame with features, MI scores, and selection status

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        df = pd.DataFrame({
            'feature': self.mi_scores_.index,
            'mi_score': self.mi_scores_.values,
            'selected': [f in self.selected_features_ for f in self.mi_scores_.index]
        })

        return df.sort_values('mi_score', ascending=False).reset_index(drop=True)

    def get_support(self, indices: bool = False) -> Union[List[bool], List[int]]:
        """
        Get mask or indices of selected features (scikit-learn compatible).

        Args:
            indices: If True, return feature indices; if False, return boolean mask

        Returns:
            Boolean mask or integer indices of selected features

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        if indices:
            # Return indices of selected features
            return [
                i for i, feat in enumerate(self.feature_names_)
                if feat in self.selected_features_
            ]
        else:
            # Return boolean mask
            return [feat in self.selected_features_ for feat in self.feature_names_]

    def __repr__(self) -> str:
        """String representation."""
        task = self.params["task"]
        k = self.params["k"]

        if self.fitted and self.selected_features_:
            n_selected = len(self.selected_features_)
            n_total = len(self.mi_scores_)
            return (
                f"MutualInformationSelector(task='{task}', k={k}, "
                f"selected={n_selected}/{n_total})"
            )
        else:
            return f"MutualInformationSelector(task='{task}', k={k}, not fitted)"
