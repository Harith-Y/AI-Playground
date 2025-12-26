"""
Data cleaning utilities for outlier detection and removal.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List, Literal
from app.ml_engine.preprocessing.base import PreprocessingStep


class IQROutlierDetector(PreprocessingStep):
    """
    Detects and handles outliers using Interquartile Range (IQR) method.

    IQR Method:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Lower bound = Q1 - (threshold * IQR)
    - Upper bound = Q3 + (threshold * IQR)
    - Values outside bounds are outliers

    Parameters
    ----------
    threshold : float, default=1.5
        IQR multiplier for outlier bounds.
        - 1.5: Standard outlier detection (mild outliers)
        - 3.0: Extreme outlier detection

    method : {'remove', 'clip', 'nan'}, default='clip'
        How to handle outliers:
        - 'remove': Remove rows containing outliers
        - 'clip': Cap outliers at bounds (Winsorization)
        - 'nan': Replace outliers with NaN

    columns : list of str, optional
        Columns to check for outliers. If None, uses all numeric columns.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'price': [10, 12, 11, 100, 13, 10],  # 100 is outlier
    ...     'quantity': [5, 6, 5, 7, 200, 5]     # 200 is outlier
    ... })
    >>>
    >>> detector = IQROutlierDetector(threshold=1.5, method='clip')
    >>> df_clean = detector.fit_transform(df)
    >>> # Outliers clipped to bounds
    """

    def __init__(
        self,
        threshold: float = 1.5,
        method: Literal['remove', 'clip', 'nan'] = 'clip',
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name, threshold=threshold, method=method, columns=columns)
        self.threshold = threshold
        self.method = method
        self.columns = columns

        # Fitted parameters
        self.bounds_: Optional[Dict[str, Dict[str, float]]] = None
        self.outlier_counts_: Optional[Dict[str, int]] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "IQROutlierDetector":
        """
        Calculate outlier bounds for each column.

        Args:
            X: Training data (DataFrame or array)
            y: Not used, present for API consistency

        Returns:
            self: Returns self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQROutlierDetector expects a pandas DataFrame")

        # Determine columns to analyze
        if self.columns is not None:
            analyze_cols = [col for col in self.columns if col in X.columns]
        else:
            analyze_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(analyze_cols) == 0:
            raise ValueError("No numeric columns found to analyze")

        # Calculate bounds for each column
        self.bounds_ = {}
        for col in analyze_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - (self.threshold * IQR)
            upper_bound = Q3 + (self.threshold * IQR)

            self.bounds_[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Handle outliers based on fitted bounds and specified method.

        Args:
            X: Data to transform (DataFrame or array)

        Returns:
            DataFrame with outliers handled according to method

        Raises:
            RuntimeError: If not fitted yet
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQROutlierDetector expects a pandas DataFrame")

        X = X.copy()
        self.outlier_counts_ = {}

        if self.method == 'remove':
            # Remove rows with any outliers
            mask = pd.Series([True] * len(X), index=X.index)

            for col, bounds in self.bounds_.items():
                if col in X.columns:
                    outlier_mask = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])
                    self.outlier_counts_[col] = outlier_mask.sum()
                    mask &= ~outlier_mask

            X = X[mask]

        elif self.method == 'clip':
            # Clip outliers to bounds (Winsorization)
            for col, bounds in self.bounds_.items():
                if col in X.columns:
                    outlier_mask = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])
                    self.outlier_counts_[col] = outlier_mask.sum()
                    X[col] = X[col].clip(lower=bounds['lower'], upper=bounds['upper'])

        elif self.method == 'nan':
            # Replace outliers with NaN
            for col, bounds in self.bounds_.items():
                if col in X.columns:
                    outlier_mask = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])
                    self.outlier_counts_[col] = outlier_mask.sum()
                    X.loc[outlier_mask, col] = np.nan

        return X

    def get_bounds(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get calculated outlier bounds for each column.

        Returns:
            Dictionary with bounds info per column, or None if not fitted
        """
        if not self.fitted:
            return None
        return self.bounds_.copy()

    def get_outlier_counts(self) -> Optional[Dict[str, int]]:
        """
        Get count of outliers detected in last transform.

        Returns:
            Dictionary with outlier counts per column, or None if not transformed
        """
        return self.outlier_counts_.copy() if self.outlier_counts_ else None

    def detect_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get boolean mask of outliers without modifying data.

        Args:
            X: Data to check for outliers

        Returns:
            Boolean DataFrame where True indicates outlier
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQROutlierDetector expects a pandas DataFrame")

        outlier_mask = pd.DataFrame(False, index=X.index, columns=X.columns)

        for col, bounds in self.bounds_.items():
            if col in X.columns:
                outlier_mask[col] = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])

        return outlier_mask

    def __repr__(self) -> str:
        """String representation."""
        if self.fitted and self.bounds_:
            return (
                f"IQROutlierDetector(threshold={self.threshold}, method='{self.method}', "
                f"n_columns={len(self.bounds_)})"
            )
        else:
            return f"IQROutlierDetector(threshold={self.threshold}, method='{self.method}', not fitted)"


class ZScoreOutlierDetector(PreprocessingStep):
    """
    Detects and handles outliers using Z-score (standard score) method.

    Z-Score Method:
    - Z-score = (value - mean) / std
    - Values with |Z-score| > threshold are outliers
    - Assumes data follows approximately normal distribution

    Parameters
    ----------
    threshold : float, default=3.0
        Z-score threshold for outlier detection.
        - 2.0: ~95% of data (2 standard deviations)
        - 3.0: ~99.7% of data (3 standard deviations) - recommended
        - 4.0: Very conservative (~99.99% of data)

    method : {'remove', 'clip', 'nan'}, default='clip'
        How to handle outliers:
        - 'remove': Remove rows containing outliers
        - 'clip': Cap outliers at bounds (mean ± threshold * std)
        - 'nan': Replace outliers with NaN

    columns : list of str, optional
        Columns to check for outliers. If None, uses all numeric columns.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'score': [85, 87, 86, 50, 88, 87],  # 50 is outlier
    ...     'age': [25, 26, 24, 27, 100, 25]    # 100 is outlier
    ... })
    >>>
    >>> detector = ZScoreOutlierDetector(threshold=3.0, method='clip')
    >>> df_clean = detector.fit_transform(df)
    >>> # Outliers clipped to bounds
    """

    def __init__(
        self,
        threshold: float = 3.0,
        method: Literal['remove', 'clip', 'nan'] = 'clip',
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name, threshold=threshold, method=method, columns=columns)
        self.threshold = threshold
        self.method = method
        self.columns = columns

        # Fitted parameters
        self.stats_: Optional[Dict[str, Dict[str, float]]] = None
        self.outlier_counts_: Optional[Dict[str, int]] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "ZScoreOutlierDetector":
        """
        Calculate mean and std for each column.

        Args:
            X: Training data (DataFrame or array)
            y: Not used, present for API consistency

        Returns:
            self: Returns self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZScoreOutlierDetector expects a pandas DataFrame")

        # Determine columns to analyze
        if self.columns is not None:
            analyze_cols = [col for col in self.columns if col in X.columns]
        else:
            analyze_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(analyze_cols) == 0:
            raise ValueError("No numeric columns found to analyze")

        # Calculate statistics for each column
        self.stats_ = {}
        for col in analyze_cols:
            mean = X[col].mean()
            std = X[col].std()

            lower_bound = mean - (self.threshold * std)
            upper_bound = mean + (self.threshold * std)

            self.stats_[col] = {
                'mean': mean,
                'std': std,
                'lower': lower_bound,
                'upper': upper_bound
            }

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Handle outliers based on fitted statistics and specified method.

        Args:
            X: Data to transform (DataFrame or array)

        Returns:
            DataFrame with outliers handled according to method

        Raises:
            RuntimeError: If not fitted yet
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZScoreOutlierDetector expects a pandas DataFrame")

        X = X.copy()
        self.outlier_counts_ = {}

        if self.method == 'remove':
            # Remove rows with any outliers
            mask = pd.Series([True] * len(X), index=X.index)

            for col, stats in self.stats_.items():
                if col in X.columns:
                    z_scores = np.abs((X[col] - stats['mean']) / stats['std'])
                    outlier_mask = z_scores > self.threshold
                    self.outlier_counts_[col] = outlier_mask.sum()
                    mask &= ~outlier_mask

            X = X[mask]

        elif self.method == 'clip':
            # Clip outliers to bounds
            for col, stats in self.stats_.items():
                if col in X.columns:
                    z_scores = np.abs((X[col] - stats['mean']) / stats['std'])
                    outlier_mask = z_scores > self.threshold
                    self.outlier_counts_[col] = outlier_mask.sum()
                    X[col] = X[col].clip(lower=stats['lower'], upper=stats['upper'])

        elif self.method == 'nan':
            # Replace outliers with NaN
            for col, stats in self.stats_.items():
                if col in X.columns:
                    z_scores = np.abs((X[col] - stats['mean']) / stats['std'])
                    outlier_mask = z_scores > self.threshold
                    self.outlier_counts_[col] = outlier_mask.sum()
                    X.loc[outlier_mask, col] = np.nan

        return X

    def get_statistics(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get calculated statistics (mean, std, bounds) for each column.

        Returns:
            Dictionary with statistics per column, or None if not fitted
        """
        if not self.fitted:
            return None
        return self.stats_.copy()

    def get_outlier_counts(self) -> Optional[Dict[str, int]]:
        """
        Get count of outliers detected in last transform.

        Returns:
            Dictionary with outlier counts per column, or None if not transformed
        """
        return self.outlier_counts_.copy() if self.outlier_counts_ else None

    def detect_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get boolean mask of outliers without modifying data.

        Args:
            X: Data to check for outliers

        Returns:
            Boolean DataFrame where True indicates outlier
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZScoreOutlierDetector expects a pandas DataFrame")

        outlier_mask = pd.DataFrame(False, index=X.index, columns=X.columns)

        for col, stats in self.stats_.items():
            if col in X.columns:
                z_scores = np.abs((X[col] - stats['mean']) / stats['std'])
                outlier_mask[col] = z_scores > self.threshold

        return outlier_mask

    def get_z_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Z-scores for all values.

        Args:
            X: Data to calculate Z-scores for

        Returns:
            DataFrame with Z-scores for each value

        Raises:
            RuntimeError: If not fitted yet
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZScoreOutlierDetector expects a pandas DataFrame")

        z_scores = pd.DataFrame(index=X.index, columns=X.columns)

        for col, stats in self.stats_.items():
            if col in X.columns:
                z_scores[col] = (X[col] - stats['mean']) / stats['std']

        return z_scores

    def __repr__(self) -> str:
        """String representation."""
        if self.fitted and self.stats_:
            return (
                f"ZScoreOutlierDetector(threshold={self.threshold}, method='{self.method}', "
                f"n_columns={len(self.stats_)})"
            )
        else:
            return f"ZScoreOutlierDetector(threshold={self.threshold}, method='{self.method}', not fitted)"
