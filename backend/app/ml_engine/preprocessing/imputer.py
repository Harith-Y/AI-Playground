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


class ModeImputer(PreprocessingStep):
    """
    Imputes missing values using column-wise mode (most frequent value).

    The mode imputer works with both numeric and categorical columns,
    making it versatile for mixed-type datasets. It's particularly useful for:
    - Categorical features (e.g., 'color', 'category')
    - Binary features (e.g., 0/1, True/False)
    - Ordinal features (e.g., 'low', 'medium', 'high')
    - Numeric features with discrete values

    For numeric columns with continuous values, consider MedianImputer instead,
    as mode may not be meaningful.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'color': ['red', 'blue', 'red', None, 'red'],
        ...     'size': ['S', 'M', None, 'M', 'M'],
        ...     'count': [1, 2, 2, None, 2]
        ... })
        >>>
        >>> imputer = ModeImputer()
        >>> df_imputed = imputer.fit_transform(df)
        >>> print(df_imputed)
        # Missing values replaced with: 'red', 'M', 2

    Attributes:
        modes (Dict[str, Any]): Dictionary mapping column names to their mode values
        fitted (bool): Whether the imputer has been fitted
    """

    def __init__(self, columns: Optional[list[str]] = None, name: Optional[str] = None):
        """
        Initialize ModeImputer.

        Args:
            columns: List of column names to impute.
                     If None, all columns with missing values are used.
                     Can include numeric and categorical columns.
            name: Optional custom name for this preprocessing step.

        Example:
            >>> # Impute all columns
            >>> imputer = ModeImputer()
            >>>
            >>> # Impute specific columns only
            >>> imputer = ModeImputer(columns=['color', 'category'])
        """
        super().__init__(name=name, columns=columns)
        self.modes: Dict[str, Union[str, int, float]] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None
    ) -> "ModeImputer":
        """
        Compute mode (most frequent value) for each column on training data.

        For each column:
        1. Finds the most frequent value (excluding NaN)
        2. Stores it for use during transform
        3. If multiple modes exist, uses the first one (pandas default)

        Args:
            X: Training data (pandas DataFrame)
            y: Target values (not used, present for API consistency)

        Returns:
            self: Returns self for method chaining

        Raises:
            TypeError: If X is not a pandas DataFrame
            ValueError: If specified columns don't exist in X

        Example:
            >>> df = pd.DataFrame({
            ...     'color': ['red', 'blue', 'red', None, 'red'],
            ...     'size': [1, 2, 2, None, 2]
            ... })
            >>> imputer = ModeImputer()
            >>> imputer.fit(df)
            >>> print(imputer.modes)
            # {'color': 'red', 'size': 2}
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ModeImputer expects a pandas DataFrame")

        columns = self.params["columns"]

        if columns is None:
            # Use all columns with missing values
            columns = X.columns[X.isnull().any()].tolist()
        else:
            # Validate specified columns exist
            missing_cols = set(columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        # Compute mode for each column
        for col in columns:
            if col in X.columns:
                # Get mode (most frequent value)
                # mode() returns a Series, we take the first value
                mode_values = X[col].mode()

                if len(mode_values) > 0:
                    self.modes[col] = mode_values[0]
                else:
                    # If all values are NaN, we can't compute mode
                    # Store None and skip imputation for this column
                    self.modes[col] = None

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Replace missing values with learned mode values.

        Args:
            X: Data to transform (pandas DataFrame)

        Returns:
            DataFrame with missing values imputed using mode values

        Raises:
            RuntimeError: If imputer has not been fitted yet
            TypeError: If X is not a pandas DataFrame

        Example:
            >>> df_test = pd.DataFrame({
            ...     'color': ['blue', None, 'green'],
            ...     'size': [1, None, 3]
            ... })
            >>> df_transformed = imputer.transform(df_test)
            >>> print(df_transformed)
            # Missing values replaced with fitted mode values
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("ModeImputer expects a pandas DataFrame")

        X = X.copy()

        # Replace missing values with mode for each column
        for col, mode_value in self.modes.items():
            if col in X.columns and mode_value is not None:
                X[col] = X[col].fillna(mode_value)

        return X

    def get_modes(self) -> Optional[Dict[str, Union[str, int, float]]]:
        """
        Get the mode values learned during fit.

        Returns:
            Dictionary mapping column names to their mode values,
            or None if not fitted yet.

        Example:
            >>> imputer.fit(df)
            >>> modes = imputer.get_modes()
            >>> print(modes)
            # {'color': 'red', 'size': 2}
        """
        if not self.fitted:
            return None
        return self.modes.copy()

