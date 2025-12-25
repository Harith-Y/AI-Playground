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
