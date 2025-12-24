import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List
from app.utils.logger import get_logger

from app.ml_engine.preprocessing.base import PreprocessingStep

logger = get_logger("encoder")


class OneHotEncoder(PreprocessingStep):
    """
    One-Hot encodes categorical columns into binary indicator columns.

    Converts categorical variables into a format that works better with ML algorithms
    by creating binary columns for each category.

    Example:
        color: ['red', 'blue', 'red']
        -> color_red: [1, 0, 1], color_blue: [0, 1, 0]

    Attributes:
        categories_: Mapping of column names to their unique categories (learned during fit)
        feature_names_: List of all generated feature names after encoding
        drop_first: Whether to drop the first category to avoid multicollinearity
        handle_unknown: How to handle unknown categories during transform ('error' or 'ignore')
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        drop_first: bool = False,
        handle_unknown: str = "error",
        name: Optional[str] = None
    ):
        """
        Initialize OneHotEncoder.

        Args:
            columns: List of columns to encode.
                     If None, all object/category dtype columns are used.
            drop_first: If True, drop the first category in each column to avoid
                       the dummy variable trap (multicollinearity). Default False.
            handle_unknown: How to handle unknown categories during transform.
                          'error': Raise an error if unknown category is encountered
                          'ignore': Create all-zero row for unknown categories
                          Default is 'error'.
            name: Optional custom name for this preprocessing step.

        Raises:
            ValueError: If handle_unknown is not 'error' or 'ignore'
        """
        if handle_unknown not in ["error", "ignore"]:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")

        super().__init__(
            name=name,
            columns=columns,
            drop_first=drop_first,
            handle_unknown=handle_unknown
        )
        self.categories_: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "OneHotEncoder":
        """
        Learn the unique categories for each column from training data.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Optional training labels (not used, present for API consistency)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OneHotEncoder expects a pandas DataFrame")

        columns = self.params["columns"]
        drop_first = self.params["drop_first"]

        # Auto-detect categorical columns if not specified
        if columns is None:
            columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
            logger.debug(f"Auto-detected categorical columns: {columns}")

        if not columns:
            logger.warning("No categorical columns found or specified for encoding")
            self.fitted = True
            return self

        # Learn unique categories for each column
        self.categories_ = {}
        self.feature_names_ = []

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            # Get unique categories, sorted for consistency
            unique_categories = sorted(X[col].dropna().unique().tolist())
            self.categories_[col] = unique_categories

            # Generate feature names
            categories_to_use = unique_categories[1:] if drop_first else unique_categories
            for category in categories_to_use:
                self.feature_names_.append(f"{col}_{category}")

        logger.debug(f"Learned categories for columns: {list(self.categories_.keys())}")
        logger.debug(f"Generated {len(self.feature_names_)} one-hot encoded features")

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data by one-hot encoding categorical columns.

        Args:
            X: Data to transform (must be a pandas DataFrame)

        Returns:
            Transformed DataFrame with original categorical columns replaced
            by one-hot encoded binary columns

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If encoder has not been fitted
            ValueError: If unknown categories are encountered and handle_unknown='error'
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("OneHotEncoder expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.categories_:
            return X.copy()

        X = X.copy()
        handle_unknown = self.params["handle_unknown"]
        drop_first = self.params["drop_first"]

        # Store columns to drop (original categorical columns)
        columns_to_drop = list(self.categories_.keys())

        # Create one-hot encoded columns
        encoded_data = {}

        for col, categories in self.categories_.items():
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during transform")

            # Check for unknown categories
            data_categories = set(X[col].dropna().unique())
            known_categories = set(categories)
            unknown_categories = data_categories - known_categories

            if unknown_categories:
                if handle_unknown == "error":
                    raise ValueError(
                        f"Found unknown categories {unknown_categories} in column '{col}' "
                        f"during transform. Known categories are {known_categories}"
                    )
                else:  # handle_unknown == "ignore"
                    logger.warning(
                        f"Unknown categories {unknown_categories} in column '{col}' "
                        f"will be encoded as all zeros"
                    )

            # Determine which categories to create columns for
            categories_to_use = categories[1:] if drop_first else categories

            # Create binary columns for each category
            for category in categories_to_use:
                feature_name = f"{col}_{category}"
                encoded_data[feature_name] = (X[col] == category).astype(int)

        # Drop original categorical columns
        X = X.drop(columns=columns_to_drop)

        # Add encoded columns
        encoded_df = pd.DataFrame(encoded_data, index=X.index)
        X = pd.concat([X, encoded_df], axis=1)

        return X

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the one-hot encoded features.

        Returns:
            List of feature names generated by the encoder

        Raises:
            RuntimeError: If encoder has not been fitted
        """
        self._check_fitted()
        return self.feature_names_.copy()
