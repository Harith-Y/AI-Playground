import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List
from app.utils.logger import get_logger

from app.ml_engine.preprocessing.base import PreprocessingStep

logger = get_logger("scaler")


class StandardScaler(PreprocessingStep):
    """
    Standardizes features by removing the mean and scaling to unit variance.

    The standard score (z-score) of a sample x is calculated as:
        z = (x - mean) / std

    This scaler is essential for algorithms that assume features are centered
    around zero with similar variance (e.g., SVM, Linear Regression, Neural Networks).

    Example:
        data: [1, 2, 3, 4, 5]
        mean: 3, std: 1.414
        scaled: [-1.414, -0.707, 0, 0.707, 1.414]

    Attributes:
        means_: Mean values for each column (learned during fit)
        stds_: Standard deviation values for each column (learned during fit)
        with_mean: Whether to center data by subtracting the mean
        with_std: Whether to scale data to unit variance
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize StandardScaler.

        Args:
            columns: List of columns to scale.
                     If None, all numeric columns are used.
            with_mean: If True, center the data before scaling (subtract mean).
                      Default is True.
            with_std: If True, scale the data to unit variance (divide by std).
                     Default is True.
            name: Optional custom name for this preprocessing step.

        Raises:
            ValueError: If both with_mean and with_std are False
        """
        if not with_mean and not with_std:
            raise ValueError("At least one of with_mean or with_std must be True")

        super().__init__(
            name=name,
            columns=columns,
            with_mean=with_mean,
            with_std=with_std
        )
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "StandardScaler":
        """
        Compute the mean and standard deviation for each column from training data.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Optional training labels (not used, present for API consistency)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StandardScaler expects a pandas DataFrame")

        columns = self.params["columns"]
        with_mean = self.params["with_mean"]
        with_std = self.params["with_std"]

        # Auto-detect numeric columns if not specified
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
            logger.debug(f"Auto-detected numeric columns: {columns}")

        if not columns:
            logger.warning("No numeric columns found or specified for scaling")
            self.fitted = True
            return self

        # Learn mean and standard deviation for each column
        self.means_ = {}
        self.stds_ = {}

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            if with_mean:
                self.means_[col] = X[col].mean()

            if with_std:
                std = X[col].std()
                # Handle constant columns (std = 0)
                if std == 0:
                    logger.warning(f"Column '{col}' has zero standard deviation. Setting std to 1 to avoid division by zero.")
                    std = 1.0
                self.stds_[col] = std

        logger.debug(f"Fitted StandardScaler on {len(columns)} columns")
        if with_mean:
            logger.debug(f"Learned means for columns: {list(self.means_.keys())}")
        if with_std:
            logger.debug(f"Learned standard deviations for columns: {list(self.stds_.keys())}")

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Standardize features by removing mean and scaling to unit variance.

        Args:
            X: Data to transform (must be a pandas DataFrame)

        Returns:
            Transformed DataFrame with standardized numeric columns

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("StandardScaler expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.means_ and not self.stds_:
            return X.copy()

        X = X.copy()
        with_mean = self.params["with_mean"]
        with_std = self.params["with_std"]

        # Determine which columns to transform
        columns = list(self.means_.keys() if with_mean else self.stds_.keys())

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during transform")

            # Apply standardization
            if with_mean and with_std:
                X[col] = (X[col] - self.means_[col]) / self.stds_[col]
            elif with_mean:
                X[col] = X[col] - self.means_[col]
            elif with_std:
                X[col] = X[col] / self.stds_[col]

        return X

    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Scale data back to original representation.

        Args:
            X: Data to inverse transform (must be a pandas DataFrame)

        Returns:
            DataFrame with values scaled back to original range

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("StandardScaler expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.means_ and not self.stds_:
            return X.copy()

        X = X.copy()
        with_mean = self.params["with_mean"]
        with_std = self.params["with_std"]

        # Determine which columns to inverse transform
        columns = list(self.means_.keys() if with_mean else self.stds_.keys())

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during inverse_transform")

            # Apply inverse transformation (reverse order of transform)
            if with_mean and with_std:
                X[col] = (X[col] * self.stds_[col]) + self.means_[col]
            elif with_std:
                X[col] = X[col] * self.stds_[col]
            elif with_mean:
                X[col] = X[col] + self.means_[col]

        return X

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get the learned mean and standard deviation statistics.

        Returns:
            Dictionary with 'means' and 'stds' keys containing column statistics

        Raises:
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()
        return {
            "means": self.means_.copy(),
            "stds": self.stds_.copy()
        }


class MinMaxScaler(PreprocessingStep):
    """
    Scales features to a specified range (default [0, 1]) using min-max normalization.

    The transformation is calculated as:
        X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min

    This scaler is useful when you need features within a bounded range,
    particularly for neural networks and algorithms sensitive to feature magnitude.

    Example:
        data: [1, 2, 3, 4, 5]
        min: 1, max: 5
        scaled to [0, 1]: [0, 0.25, 0.5, 0.75, 1]

    Attributes:
        mins_: Minimum values for each column (learned during fit)
        maxs_: Maximum values for each column (learned during fit)
        feature_range: Desired range of transformed data (min, max)
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0, 1),
        name: Optional[str] = None
    ):
        """
        Initialize MinMaxScaler.

        Args:
            columns: List of columns to scale.
                     If None, all numeric columns are used.
            feature_range: Desired range of transformed data as (min, max).
                          Default is (0, 1).
            name: Optional custom name for this preprocessing step.

        Raises:
            ValueError: If feature_range is invalid (min >= max)
        """
        if len(feature_range) != 2:
            raise ValueError("feature_range must be a tuple of (min, max)")

        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range min must be less than max")

        super().__init__(
            name=name,
            columns=columns,
            feature_range=feature_range
        )
        self.mins_: Dict[str, float] = {}
        self.maxs_: Dict[str, float] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "MinMaxScaler":
        """
        Compute the minimum and maximum for each column from training data.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Optional training labels (not used, present for API consistency)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MinMaxScaler expects a pandas DataFrame")

        columns = self.params["columns"]

        # Auto-detect numeric columns if not specified
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
            logger.debug(f"Auto-detected numeric columns: {columns}")

        if not columns:
            logger.warning("No numeric columns found or specified for scaling")
            self.fitted = True
            return self

        # Learn min and max for each column
        self.mins_ = {}
        self.maxs_ = {}

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            min_val = X[col].min()
            max_val = X[col].max()

            # Handle constant columns (min == max)
            if min_val == max_val:
                logger.warning(
                    f"Column '{col}' has constant value {min_val}. "
                    f"Scaling will result in constant output."
                )

            self.mins_[col] = min_val
            self.maxs_[col] = max_val

        logger.debug(f"Fitted MinMaxScaler on {len(columns)} columns")
        logger.debug(f"Learned min/max ranges for columns: {list(self.mins_.keys())}")

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Scale features to the specified range.

        Args:
            X: Data to transform (must be a pandas DataFrame)

        Returns:
            Transformed DataFrame with scaled numeric columns in feature_range

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("MinMaxScaler expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.mins_:
            return X.copy()

        X = X.copy()
        feature_range = self.params["feature_range"]
        range_min, range_max = feature_range

        for col in self.mins_.keys():
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during transform")

            data_min = self.mins_[col]
            data_max = self.maxs_[col]

            # Handle constant columns
            if data_min == data_max:
                # Set to the middle of the feature range
                X[col] = (range_min + range_max) / 2
            else:
                # Apply min-max scaling
                # Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
                X[col] = ((X[col] - data_min) / (data_max - data_min)) * (range_max - range_min) + range_min

        return X

    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Scale data back to original representation.

        Args:
            X: Data to inverse transform (must be a pandas DataFrame)

        Returns:
            DataFrame with values scaled back to original range

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("MinMaxScaler expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.mins_:
            return X.copy()

        X = X.copy()
        feature_range = self.params["feature_range"]
        range_min, range_max = feature_range

        for col in self.mins_.keys():
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during inverse_transform")

            data_min = self.mins_[col]
            data_max = self.maxs_[col]

            # Handle constant columns
            if data_min == data_max:
                X[col] = data_min
            else:
                # Apply inverse transformation
                # X_original = (X_scaled - min) / (max - min) * (X_max - X_min) + X_min
                X[col] = ((X[col] - range_min) / (range_max - range_min)) * (data_max - data_min) + data_min

        return X

    def get_data_range(self) -> Dict[str, Dict[str, float]]:
        """
        Get the learned minimum and maximum values for each column.

        Returns:
            Dictionary with 'mins' and 'maxs' keys containing column ranges

        Raises:
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()
        return {
            "mins": self.mins_.copy(),
            "maxs": self.maxs_.copy()
        }


class RobustScaler(PreprocessingStep):
    """
    Scales features using statistics that are robust to outliers.

    This scaler removes the median and scales the data according to the
    Interquartile Range (IQR). The IQR is the range between the 1st quartile
    (25th quantile) and the 3rd quartile (75th quantile).

    The transformation is calculated as:
        X_scaled = (X - median) / IQR

    This scaler is particularly useful when your data contains outliers,
    as it uses the median and IQR which are not influenced by extreme values.

    Example:
        data: [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        median: 3.5, IQR: 3
        scaled: [-0.833, -0.5, -0.167, 0.167, 0.5, 32.167]
        (Note: outlier is scaled but doesn't affect the scaling of other values)

    Attributes:
        medians_: Median values for each column (learned during fit)
        iqrs_: Interquartile range values for each column (learned during fit)
        with_centering: Whether to center data by subtracting the median
        with_scaling: Whether to scale data to IQR
        quantile_range: Quantile range used to calculate IQR (default: 25.0, 75.0)
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple = (25.0, 75.0),
        name: Optional[str] = None
    ):
        """
        Initialize RobustScaler.

        Args:
            columns: List of columns to scale.
                     If None, all numeric columns are used.
            with_centering: If True, center the data before scaling (subtract median).
                           Default is True.
            with_scaling: If True, scale the data to IQR.
                         Default is True.
            quantile_range: Quantile range used to calculate scale (IQR).
                           Default is (25.0, 75.0) for standard IQR.
            name: Optional custom name for this preprocessing step.

        Raises:
            ValueError: If both with_centering and with_scaling are False
            ValueError: If quantile_range is invalid
        """
        if not with_centering and not with_scaling:
            raise ValueError("At least one of with_centering or with_scaling must be True")

        if len(quantile_range) != 2:
            raise ValueError("quantile_range must be a tuple of (lower, upper)")

        if quantile_range[0] >= quantile_range[1]:
            raise ValueError("quantile_range lower must be less than upper")

        if quantile_range[0] < 0 or quantile_range[1] > 100:
            raise ValueError("quantile_range values must be between 0 and 100")

        super().__init__(
            name=name,
            columns=columns,
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range
        )
        self.medians_: Dict[str, float] = {}
        self.iqrs_: Dict[str, float] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "RobustScaler":
        """
        Compute the median and IQR for each column from training data.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Optional training labels (not used, present for API consistency)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RobustScaler expects a pandas DataFrame")

        columns = self.params["columns"]
        with_centering = self.params["with_centering"]
        with_scaling = self.params["with_scaling"]
        quantile_range = self.params["quantile_range"]

        # Auto-detect numeric columns if not specified
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
            logger.debug(f"Auto-detected numeric columns: {columns}")

        if not columns:
            logger.warning("No numeric columns found or specified for scaling")
            self.fitted = True
            return self

        # Learn median and IQR for each column
        self.medians_ = {}
        self.iqrs_ = {}

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            if with_centering:
                self.medians_[col] = X[col].median()

            if with_scaling:
                q_lower = X[col].quantile(quantile_range[0] / 100.0)
                q_upper = X[col].quantile(quantile_range[1] / 100.0)
                iqr = q_upper - q_lower

                # Handle constant columns (IQR = 0)
                if iqr == 0:
                    logger.warning(
                        f"Column '{col}' has zero IQR (constant or near-constant values). "
                        f"Setting IQR to 1 to avoid division by zero."
                    )
                    iqr = 1.0

                self.iqrs_[col] = iqr

        logger.debug(f"Fitted RobustScaler on {len(columns)} columns")
        if with_centering:
            logger.debug(f"Learned medians for columns: {list(self.medians_.keys())}")
        if with_scaling:
            logger.debug(f"Learned IQRs for columns: {list(self.iqrs_.keys())}")

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Scale features using robust statistics (median and IQR).

        Args:
            X: Data to transform (must be a pandas DataFrame)

        Returns:
            Transformed DataFrame with robustly scaled numeric columns

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("RobustScaler expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.medians_ and not self.iqrs_:
            return X.copy()

        X = X.copy()
        with_centering = self.params["with_centering"]
        with_scaling = self.params["with_scaling"]

        # Determine which columns to transform
        columns = list(self.medians_.keys() if with_centering else self.iqrs_.keys())

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during transform")

            # Apply robust scaling
            if with_centering and with_scaling:
                X[col] = (X[col] - self.medians_[col]) / self.iqrs_[col]
            elif with_centering:
                X[col] = X[col] - self.medians_[col]
            elif with_scaling:
                X[col] = X[col] / self.iqrs_[col]

        return X

    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Scale data back to original representation.

        Args:
            X: Data to inverse transform (must be a pandas DataFrame)

        Returns:
            DataFrame with values scaled back to original range

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("RobustScaler expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.medians_ and not self.iqrs_:
            return X.copy()

        X = X.copy()
        with_centering = self.params["with_centering"]
        with_scaling = self.params["with_scaling"]

        # Determine which columns to inverse transform
        columns = list(self.medians_.keys() if with_centering else self.iqrs_.keys())

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during inverse_transform")

            # Apply inverse transformation (reverse order of transform)
            if with_centering and with_scaling:
                X[col] = (X[col] * self.iqrs_[col]) + self.medians_[col]
            elif with_scaling:
                X[col] = X[col] * self.iqrs_[col]
            elif with_centering:
                X[col] = X[col] + self.medians_[col]

        return X

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get the learned median and IQR statistics.

        Returns:
            Dictionary with 'medians' and 'iqrs' keys containing column statistics

        Raises:
            RuntimeError: If scaler has not been fitted
        """
        self._check_fitted()
        return {
            "medians": self.medians_.copy(),
            "iqrs": self.iqrs_.copy()
        }
