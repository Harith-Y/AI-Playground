import pandas as pd
import numpy as np
from typing import Optional, Union, Dict

from app.ml_engine.preprocessing.base import PreprocessingStep


class MedianImputer(PreprocessingStep):
    """
    Imputes missing numeric values using column-wise median.

    Example:
        age  -> median(age)
        salary -> median(salary)
    """

    def __init__(self, columns: Optional[list[str]] = None, name: Optional[str] = None):
        """
        Args:
            columns: List of columns to impute.
                     If None, all numeric columns are used.
            name: Optional custom name.
        """
        super().__init__(name=name, columns=columns)
        self.medians: Dict[str, float] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None
    ) -> "MedianImputer":
        """
        Compute median for each column on training data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MedianImputer expects a pandas DataFrame")

        columns = self.params["columns"]

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            self.medians[col] = X[col].median()

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Replace missing values with learned medians.
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("MedianImputer expects a pandas DataFrame")

        X = X.copy()

        for col, median in self.medians.items():
            X[col] = X[col].fillna(median)

        return X


class MeanImputer(PreprocessingStep):
    """
    Imputes missing numeric values using column-wise mean.
    """

    def __init__(self, columns: Optional[list[str]] = None, name: Optional[str] = None):
        """
        Args:
            columns: List of columns to impute.
                     If None, all numeric columns are used.
            name: Optional custom name.
        """
        super().__init__(name=name, columns=columns)
        self.means: Dict[str, float] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None
    ) -> "MeanImputer":
        """
        Compute mean for each column on training data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MeanImputer expects a pandas DataFrame")

        columns = self.params["columns"]

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            self.means[col] = X[col].mean()

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Replace missing values with learned means.
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("MeanImputer expects a pandas DataFrame")

        X = X.copy()

        for col, mean in self.means.items():
            X[col] = X[col].fillna(mean)

        return X

