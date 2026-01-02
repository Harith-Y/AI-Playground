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
            n_unique = len(unique_categories)
            
            # Warn about high cardinality
            if n_unique > 100:
                logger.warning(
                    f"Column '{col}' has very high cardinality ({n_unique} unique values). "
                    f"One-hot encoding will create {n_unique - 1 if drop_first else n_unique} features. "
                    "Consider using alternative encoding methods (target encoding, label encoding) "
                    "or grouping rare categories."
                )
            elif n_unique > 50:
                logger.warning(
                    f"Column '{col}' has high cardinality ({n_unique} unique values). "
                    f"This will create {n_unique - 1 if drop_first else n_unique} features."
                )
            
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


class LabelEncoder(PreprocessingStep):
    """
    Encodes categorical columns into integer labels.

    Converts categorical variables into numeric format by assigning each unique
    category a unique integer label. This is useful for ordinal data or when
    feeding data to tree-based models.

    Example:
        color: ['red', 'blue', 'red', 'green']
        -> color: [2, 0, 2, 1]  (alphabetically: blue=0, green=1, red=2)

    Attributes:
        label_mappings_: Mapping of column names to category-to-label dictionaries
        inverse_mappings_: Mapping of column names to label-to-category dictionaries (for inverse_transform)
        handle_unknown: How to handle unknown categories during transform ('error' or 'use_encoded_value')
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        handle_unknown: str = "error",
        unknown_value: int = -1,
        name: Optional[str] = None
    ):
        """
        Initialize LabelEncoder.

        Args:
            columns: List of columns to encode.
                     If None, all object/category dtype columns are used.
            handle_unknown: How to handle unknown categories during transform.
                          'error': Raise an error if unknown category is encountered
                          'use_encoded_value': Assign unknown_value to unknown categories
                          Default is 'error'.
            unknown_value: The integer value to use for unknown categories when
                          handle_unknown='use_encoded_value'. Default is -1.
            name: Optional custom name for this preprocessing step.

        Raises:
            ValueError: If handle_unknown is not 'error' or 'use_encoded_value'
        """
        if handle_unknown not in ["error", "use_encoded_value"]:
            raise ValueError("handle_unknown must be 'error' or 'use_encoded_value'")

        super().__init__(
            name=name,
            columns=columns,
            handle_unknown=handle_unknown,
            unknown_value=unknown_value
        )
        self.label_mappings_: Dict[str, Dict[str, int]] = {}
        self.inverse_mappings_: Dict[str, Dict[int, str]] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "LabelEncoder":
        """
        Learn the unique categories and their label mappings from training data.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Optional training labels (not used, present for API consistency)

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LabelEncoder expects a pandas DataFrame")

        columns = self.params["columns"]

        # Auto-detect categorical columns if not specified
        if columns is None:
            columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
            logger.debug(f"Auto-detected categorical columns: {columns}")

        if not columns:
            logger.warning("No categorical columns found or specified for encoding")
            self.fitted = True
            return self

        # Learn label mappings for each column
        self.label_mappings_ = {}
        self.inverse_mappings_ = {}

        for col in columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            # Get unique categories, sorted for consistency
            unique_categories = sorted(X[col].dropna().unique().tolist())

            # Create bidirectional mappings
            label_mapping = {category: idx for idx, category in enumerate(unique_categories)}
            inverse_mapping = {idx: category for idx, category in enumerate(unique_categories)}

            self.label_mappings_[col] = label_mapping
            self.inverse_mappings_[col] = inverse_mapping

            logger.debug(f"Column '{col}': {len(unique_categories)} categories mapped to labels 0-{len(unique_categories)-1}")

        logger.debug(f"Learned label mappings for columns: {list(self.label_mappings_.keys())}")

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data by encoding categorical columns to integer labels.

        Args:
            X: Data to transform (must be a pandas DataFrame)

        Returns:
            Transformed DataFrame with categorical columns replaced by integer labels

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If encoder has not been fitted
            ValueError: If unknown categories are encountered and handle_unknown='error'
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("LabelEncoder expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.label_mappings_:
            return X.copy()

        X = X.copy()
        handle_unknown = self.params["handle_unknown"]
        unknown_value = self.params["unknown_value"]

        for col, label_mapping in self.label_mappings_.items():
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during transform")

            # Check for unknown categories
            data_categories = set(X[col].dropna().unique())
            known_categories = set(label_mapping.keys())
            unknown_categories = data_categories - known_categories

            if unknown_categories:
                if handle_unknown == "error":
                    raise ValueError(
                        f"Found unknown categories {unknown_categories} in column '{col}' "
                        f"during transform. Known categories are {known_categories}"
                    )
                else:  # handle_unknown == "use_encoded_value"
                    logger.warning(
                        f"Unknown categories {unknown_categories} in column '{col}' "
                        f"will be encoded as {unknown_value}"
                    )

            # Apply label encoding
            if handle_unknown == "use_encoded_value":
                # Map known categories, use unknown_value for unknown ones
                X[col] = X[col].map(label_mapping).fillna(unknown_value).astype(int)
            else:
                # Only map known categories (will fail earlier if unknown found)
                X[col] = X[col].map(label_mapping).astype(int)

        return X

    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform encoded integer labels back to original categorical values.

        Args:
            X: Data to inverse transform (must be a pandas DataFrame with integer-encoded columns)

        Returns:
            DataFrame with integer labels converted back to original categorical values

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If encoder has not been fitted
            ValueError: If a label is not found in the inverse mapping
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("LabelEncoder expects a pandas DataFrame")

        # If no columns were fitted, return copy unchanged
        if not self.inverse_mappings_:
            return X.copy()

        X = X.copy()
        unknown_value = self.params.get("unknown_value", -1)

        for col, inverse_mapping in self.inverse_mappings_.items():
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame during inverse_transform")

            # Check for unknown labels
            data_labels = set(X[col].dropna().unique())
            known_labels = set(inverse_mapping.keys())

            # If unknown_value exists in data but not in mapping, that's expected
            unknown_labels = data_labels - known_labels - {unknown_value}

            if unknown_labels:
                raise ValueError(
                    f"Found unknown labels {unknown_labels} in column '{col}' "
                    f"during inverse_transform. Known labels are {known_labels}"
                )

            # Apply inverse mapping (unknown_value will become NaN)
            X[col] = X[col].map(inverse_mapping)

        return X

    def get_label_mappings(self) -> Dict[str, Dict[str, int]]:
        """
        Get the category-to-label mappings for all encoded columns.

        Returns:
            Dictionary mapping column names to their category-to-label dictionaries

        Raises:
            RuntimeError: If encoder has not been fitted
        """
        self._check_fitted()
        return {col: mapping.copy() for col, mapping in self.label_mappings_.items()}
