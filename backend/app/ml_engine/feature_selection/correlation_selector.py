"""
Correlation-based feature selection.

This module provides feature selection based on correlation analysis:
1. Remove features highly correlated with each other (multicollinearity)
2. Select features most correlated with the target variable
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List, Literal
from app.ml_engine.preprocessing.base import PreprocessingStep


class CorrelationSelector(PreprocessingStep):
    """
    Selects features based on correlation analysis.

    This selector can operate in two modes:
    1. 'multicollinearity': Remove highly correlated features (feature-feature correlation)
    2. 'target': Select features most correlated with target (feature-target correlation)

    Multicollinearity Mode:
    - Identifies pairs of features with correlation above threshold
    - Keeps one feature from each highly correlated pair
    - Useful for reducing redundancy and preventing multicollinearity issues

    Target Mode:
    - Calculates correlation between each feature and the target variable
    - Selects top_k features with highest absolute correlation
    - Useful for supervised learning feature selection

    Parameters
    ----------
    method : {'multicollinearity', 'target'}, default='multicollinearity'
        Feature selection method:
        - 'multicollinearity': Remove redundant features based on feature-feature correlation
        - 'target': Select features based on feature-target correlation

    threshold : float, default=0.9
        For 'multicollinearity' mode: correlation threshold above which features are considered redundant
        For 'target' mode: minimum correlation threshold with target

    top_k : int, optional
        For 'target' mode: number of top features to select (by correlation with target)
        If None, selects all features above threshold

    correlation_method : {'pearson', 'spearman', 'kendall'}, default='pearson'
        Method to compute correlation:
        - 'pearson': Standard correlation (linear relationships)
        - 'spearman': Rank correlation (monotonic relationships)
        - 'kendall': Tau correlation (ordinal associations)

    columns : list of str, optional
        Specific columns to consider for selection
        If None, uses all numeric columns

    Example
    -------
    >>> # Remove multicollinear features
    >>> selector = CorrelationSelector(method='multicollinearity', threshold=0.9)
    >>> X_reduced = selector.fit_transform(X_train)
    >>>
    >>> # Select top features correlated with target
    >>> selector = CorrelationSelector(method='target', top_k=10)
    >>> X_selected = selector.fit_transform(X_train, y_train)
    """

    def __init__(
        self,
        method: Literal['multicollinearity', 'target'] = 'multicollinearity',
        threshold: float = 0.9,
        top_k: Optional[int] = None,
        correlation_method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize CorrelationSelector.

        Args:
            method: Feature selection method ('multicollinearity' or 'target')
            threshold: Correlation threshold
            top_k: Number of top features to select (for 'target' mode)
            correlation_method: Correlation computation method
            columns: Specific columns to consider
            name: Optional custom name for this step

        Raises:
            ValueError: If method is invalid or threshold is out of range
        """
        if method not in ['multicollinearity', 'target']:
            raise ValueError("method must be 'multicollinearity' or 'target'")

        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        if correlation_method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("correlation_method must be 'pearson', 'spearman', or 'kendall'")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        super().__init__(
            name=name,
            method=method,
            threshold=threshold,
            top_k=top_k,
            correlation_method=correlation_method,
            columns=columns
        )

        # Fitted parameters
        self.selected_features_: Optional[List[str]] = None
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.target_correlations_: Optional[pd.Series] = None
        self.dropped_features_: Optional[Dict[str, str]] = None  # feature -> reason

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "CorrelationSelector":
        """
        Learn which features to select based on correlation analysis.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Target variable (required for method='target', ignored for method='multicollinearity')

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
            ValueError: If method='target' and y is not provided
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CorrelationSelector expects a pandas DataFrame")

        method = self.params["method"]
        threshold = self.params["threshold"]
        top_k = self.params["top_k"]
        correlation_method = self.params["correlation_method"]
        columns = self.params["columns"]

        # Determine columns to analyze
        if columns is not None:
            analyze_cols = [col for col in columns if col in X.columns]
        else:
            analyze_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(analyze_cols) == 0:
            raise ValueError("No numeric columns found to analyze")

        X_subset = X[analyze_cols]

        if method == 'multicollinearity':
            self._fit_multicollinearity(X_subset, correlation_method, threshold)
        elif method == 'target':
            if y is None:
                raise ValueError("y (target) must be provided for method='target'")
            self._fit_target(X_subset, y, correlation_method, threshold, top_k)

        self.fitted = True
        return self

    def _fit_multicollinearity(
        self,
        X: pd.DataFrame,
        correlation_method: str,
        threshold: float
    ) -> None:
        """
        Fit for multicollinearity removal.

        Identifies and removes one feature from each highly correlated pair.
        """
        # Compute correlation matrix
        self.correlation_matrix_ = X.corr(method=correlation_method).abs()

        # Find highly correlated pairs
        features_to_drop = set()
        self.dropped_features_ = {}

        # Iterate through upper triangle of correlation matrix
        for i in range(len(self.correlation_matrix_.columns)):
            for j in range(i + 1, len(self.correlation_matrix_.columns)):
                col_i = self.correlation_matrix_.columns[i]
                col_j = self.correlation_matrix_.columns[j]

                if self.correlation_matrix_.iloc[i, j] > threshold:
                    # Drop the second feature (arbitrary choice)
                    if col_j not in features_to_drop:
                        features_to_drop.add(col_j)
                        self.dropped_features_[col_j] = f"Correlated with {col_i} (r={self.correlation_matrix_.iloc[i, j]:.3f})"

        # Selected features are those not dropped
        self.selected_features_ = [col for col in X.columns if col not in features_to_drop]

    def _fit_target(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        correlation_method: str,
        threshold: float,
        top_k: Optional[int]
    ) -> None:
        """
        Fit for target-based feature selection.

        Selects features most correlated with the target variable.
        """
        # Convert y to Series if numpy array
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)

        # Compute correlation with target for each feature
        self.target_correlations_ = pd.Series(index=X.columns, dtype=float)

        for col in X.columns:
            self.target_correlations_[col] = X[col].corr(y, method=correlation_method)

        # Get absolute correlations for ranking
        abs_correlations = self.target_correlations_.abs()

        # Apply threshold
        above_threshold = abs_correlations[abs_correlations >= threshold]

        # Select top_k features
        if top_k is not None:
            # Sort by absolute correlation and take top_k
            selected = above_threshold.nlargest(min(top_k, len(above_threshold)))
            self.selected_features_ = selected.index.tolist()
        else:
            # Select all features above threshold
            self.selected_features_ = above_threshold.index.tolist()

        # Track dropped features
        self.dropped_features_ = {}
        for col in X.columns:
            if col not in self.selected_features_:
                corr_val = self.target_correlations_[col]
                if pd.isna(corr_val):
                    self.dropped_features_[col] = "NaN correlation with target"
                else:
                    self.dropped_features_[col] = f"Low target correlation (r={corr_val:.3f})"

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data by selecting only the chosen features.

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
            raise TypeError("CorrelationSelector expects a pandas DataFrame")

        # Return only selected features
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

    def get_dropped_features(self) -> Dict[str, str]:
        """
        Get dictionary of dropped features and reasons.

        Returns:
            Dictionary mapping feature names to reasons for dropping

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()
        return self.dropped_features_.copy()

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """
        Get the correlation matrix (for multicollinearity mode).

        Returns:
            Correlation matrix if method='multicollinearity', None otherwise

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()
        if self.correlation_matrix_ is not None:
            return self.correlation_matrix_.copy()
        return None

    def get_target_correlations(self) -> Optional[pd.Series]:
        """
        Get feature-target correlations (for target mode).

        Returns:
            Series of correlations with target if method='target', None otherwise

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()
        if self.target_correlations_ is not None:
            return self.target_correlations_.copy()
        return None

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance summary.

        For 'multicollinearity' mode: Returns features and their selection status
        For 'target' mode: Returns features ranked by target correlation

        Returns:
            DataFrame with feature importance information

        Raises:
            RuntimeError: If selector has not been fitted
        """
        self._check_fitted()

        method = self.params["method"]

        if method == 'target' and self.target_correlations_ is not None:
            # Create importance dataframe sorted by absolute correlation
            df = pd.DataFrame({
                'feature': self.target_correlations_.index,
                'correlation': self.target_correlations_.values,
                'abs_correlation': self.target_correlations_.abs().values,
                'selected': [f in self.selected_features_ for f in self.target_correlations_.index]
            })
            return df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)

        elif method == 'multicollinearity':
            # Create summary of selected vs dropped features
            all_features = self.selected_features_ + list(self.dropped_features_.keys())
            df = pd.DataFrame({
                'feature': all_features,
                'selected': [f in self.selected_features_ for f in all_features],
                'reason': [self.dropped_features_.get(f, 'Selected') for f in all_features]
            })
            return df

        return pd.DataFrame()

    def __repr__(self) -> str:
        """String representation."""
        method = self.params["method"]
        threshold = self.params["threshold"]

        if self.fitted and self.selected_features_:
            n_selected = len(self.selected_features_)
            n_dropped = len(self.dropped_features_)
            return (
                f"CorrelationSelector(method='{method}', threshold={threshold}, "
                f"selected={n_selected}, dropped={n_dropped})"
            )
        else:
            return f"CorrelationSelector(method='{method}', threshold={threshold}, not fitted)"
