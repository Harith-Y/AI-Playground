"""
ColumnTransformer for applying different transformations to different column types.

Enables parallel processing of numeric and categorical columns with appropriate
transformations for each type.
"""

from typing import List, Tuple, Union, Optional, Dict, Any
import pandas as pd
import numpy as np
from copy import deepcopy

from app.ml_engine.preprocessing.base import PreprocessingStep
from app.ml_engine.preprocessing.column_selector import BaseColumnSelector, ExplicitSelector
from app.utils.logger import get_logger

logger = get_logger("column_transformer")


class ColumnTransformer(PreprocessingStep):
    """
    Apply different preprocessing steps to different column subsets.

    Similar to scikit-learn's ColumnTransformer, this allows you to:
    - Apply different transformations to numeric vs categorical columns
    - Process multiple column subsets in parallel
    - Combine results back into a single DataFrame

    Example:
        >>> from app.ml_engine.preprocessing.scaler import StandardScaler
        >>> from app.ml_engine.preprocessing.encoder import OneHotEncoder
        >>> from app.ml_engine.preprocessing.column_selector import NumericSelector, CategoricalSelector
        >>>
        >>> transformer = ColumnTransformer(transformers=[
        ...     ('numeric', StandardScaler(), NumericSelector()),
        ...     ('categorical', OneHotEncoder(), CategoricalSelector())
        ... ])
        >>> transformer.fit_transform(df)

    Attributes:
        transformers: List of (name, transformer, selector) tuples
        remainder: How to handle columns not selected by any transformer
                  'drop', 'passthrough', or a PreprocessingStep
        fitted_transformers_: Dictionary of fitted transformer instances
        selected_columns_: Dictionary mapping transformer names to their columns
    """

    def __init__(
        self,
        transformers: List[Tuple[str, PreprocessingStep, Union[BaseColumnSelector, List[str]]]],
        remainder: Union[str, PreprocessingStep] = 'passthrough',
        name: Optional[str] = None
    ):
        """
        Initialize ColumnTransformer.

        Args:
            transformers: List of (name, transformer, selector) tuples where:
                        - name: String identifier for this transformer
                        - transformer: PreprocessingStep to apply
                        - selector: BaseColumnSelector or explicit list of columns
            remainder: How to handle remaining columns:
                      - 'drop': Remove columns not selected by any transformer
                      - 'passthrough': Keep unchanged
                      - PreprocessingStep: Apply this transformation
            name: Optional name for this preprocessing step
        """
        super().__init__(name=name, transformers=transformers, remainder=remainder)

        self.transformers = transformers
        self.remainder = remainder

        # Storage for fitted state
        self.fitted_transformers_: Dict[str, PreprocessingStep] = {}
        self.selected_columns_: Dict[str, List[str]] = {}
        self.remainder_columns_: List[str] = []

        # Validate transformers
        self._validate_transformers()

        logger.info(f"Initialized ColumnTransformer with {len(transformers)} transformers")

    def _validate_transformers(self) -> None:
        """Validate transformer configuration."""
        if not self.transformers:
            raise ValueError("transformers list cannot be empty")

        # Check for duplicate names
        names = [t[0] for t in self.transformers]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate transformer names found: {duplicates}")

        # Validate each transformer
        for name, transformer, selector in self.transformers:
            if not isinstance(transformer, PreprocessingStep):
                raise TypeError(
                    f"Transformer '{name}' must be a PreprocessingStep instance, "
                    f"got {type(transformer)}"
                )

            if not isinstance(selector, (BaseColumnSelector, list)):
                raise TypeError(
                    f"Selector for '{name}' must be a BaseColumnSelector or list of strings, "
                    f"got {type(selector)}"
                )

        # Validate remainder
        if isinstance(self.remainder, str):
            if self.remainder not in ['drop', 'passthrough']:
                raise ValueError(
                    f"remainder must be 'drop', 'passthrough', or a PreprocessingStep, "
                    f"got '{self.remainder}'"
                )
        elif not isinstance(self.remainder, PreprocessingStep):
            raise TypeError(
                f"remainder must be 'drop', 'passthrough', or a PreprocessingStep, "
                f"got {type(self.remainder)}"
            )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "ColumnTransformer":
        """
        Fit all transformers on their respective column subsets.

        Args:
            X: Training features (must be a pandas DataFrame)
            y: Optional training labels

        Returns:
            Self (for method chaining)

        Raises:
            TypeError: If X is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnTransformer expects a pandas DataFrame")

        logger.info(f"Fitting ColumnTransformer on {X.shape[0]} samples, {X.shape[1]} columns")

        # Resolve column selections
        selected_all = set()

        for name, transformer, selector in self.transformers:
            # Convert selector to column list
            if isinstance(selector, list):
                columns = selector
            else:
                columns = selector(X)

            self.selected_columns_[name] = columns
            selected_all.update(columns)

            logger.debug(f"Transformer '{name}' will process {len(columns)} columns: {columns}")

            # Fit transformer on selected columns
            if columns:
                X_subset = X[columns].copy()
                self.fitted_transformers_[name] = deepcopy(transformer)
                self.fitted_transformers_[name].fit(X_subset, y)
                logger.info(f"Fitted transformer '{name}' on {len(columns)} columns")
            else:
                logger.warning(f"Transformer '{name}' has no columns to process")

        # Handle remainder columns
        all_columns = set(X.columns)
        self.remainder_columns_ = list(all_columns - selected_all)

        if self.remainder_columns_:
            logger.info(f"Remainder: {len(self.remainder_columns_)} columns: {self.remainder_columns_}")

            if isinstance(self.remainder, PreprocessingStep):
                X_remainder = X[self.remainder_columns_].copy()
                self.remainder.fit(X_remainder, y)
                logger.info(f"Fitted remainder transformer on {len(self.remainder_columns_)} columns")

        self.fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Transform data using fitted transformers.

        Args:
            X: Data to transform (must be a pandas DataFrame)

        Returns:
            Transformed DataFrame with all columns combined

        Raises:
            TypeError: If X is not a pandas DataFrame
            RuntimeError: If transformer has not been fitted
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnTransformer expects a pandas DataFrame")

        logger.debug(f"Transforming data with {len(self.transformers)} transformers")

        # Transform each column subset
        transformed_parts = []

        for name, transformer, selector in self.transformers:
            columns = self.selected_columns_[name]

            if not columns:
                continue

            # Transform subset
            X_subset = X[columns].copy()
            X_transformed = self.fitted_transformers_[name].transform(X_subset)

            # Ensure result is DataFrame
            if not isinstance(X_transformed, pd.DataFrame):
                X_transformed = pd.DataFrame(
                    X_transformed,
                    columns=columns,
                    index=X.index
                )

            transformed_parts.append(X_transformed)
            logger.debug(f"Transformed '{name}': {X_transformed.shape}")

        # Handle remainder columns
        if self.remainder_columns_:
            if self.remainder == 'passthrough':
                X_remainder = X[self.remainder_columns_].copy()
                transformed_parts.append(X_remainder)
                logger.debug(f"Passthrough remainder: {X_remainder.shape}")

            elif self.remainder == 'drop':
                logger.debug(f"Dropping {len(self.remainder_columns_)} remainder columns")

            elif isinstance(self.remainder, PreprocessingStep):
                X_remainder = X[self.remainder_columns_].copy()
                X_remainder_transformed = self.remainder.transform(X_remainder)

                if not isinstance(X_remainder_transformed, pd.DataFrame):
                    X_remainder_transformed = pd.DataFrame(
                        X_remainder_transformed,
                        columns=self.remainder_columns_,
                        index=X.index
                    )

                transformed_parts.append(X_remainder_transformed)
                logger.debug(f"Transformed remainder: {X_remainder_transformed.shape}")

        # Combine all transformed parts
        if not transformed_parts:
            logger.warning("No columns to transform, returning empty DataFrame")
            return pd.DataFrame(index=X.index)

        result = pd.concat(transformed_parts, axis=1)
        logger.info(f"ColumnTransformer output: {result.shape}")

        return result

    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Inverse transform data (if all transformers support it).

        Args:
            X: Transformed data to reverse

        Returns:
            Data in original representation

        Raises:
            RuntimeError: If transformer has not been fitted
            NotImplementedError: If any transformer doesn't support inverse transform
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnTransformer expects a pandas DataFrame")

        logger.debug(f"Inverse transforming data with {len(self.transformers)} transformers")

        # Inverse transform each subset
        inverse_parts = []

        for name in self.fitted_transformers_:
            transformer = self.fitted_transformers_[name]
            columns = self.selected_columns_[name]

            if not columns:
                continue

            # Get columns that exist in X (they might have been encoded)
            existing_cols = [col for col in X.columns if col in columns or col.startswith(f"{columns[0]}_")]

            if not existing_cols:
                logger.warning(f"No columns found for inverse transform of '{name}'")
                continue

            X_subset = X[existing_cols].copy()

            if not transformer.supports_inverse_transform():
                raise NotImplementedError(
                    f"Transformer '{name}' does not support inverse_transform"
                )

            X_inverse = transformer.inverse_transform(X_subset)

            if not isinstance(X_inverse, pd.DataFrame):
                X_inverse = pd.DataFrame(
                    X_inverse,
                    columns=columns,
                    index=X.index
                )

            inverse_parts.append(X_inverse)

        # Handle remainder
        if self.remainder_columns_:
            remainder_cols = [col for col in X.columns if col in self.remainder_columns_]

            if remainder_cols:
                if self.remainder == 'passthrough':
                    X_remainder = X[remainder_cols].copy()
                    inverse_parts.append(X_remainder)

                elif isinstance(self.remainder, PreprocessingStep):
                    X_remainder = X[remainder_cols].copy()
                    X_remainder_inverse = self.remainder.inverse_transform(X_remainder)

                    if not isinstance(X_remainder_inverse, pd.DataFrame):
                        X_remainder_inverse = pd.DataFrame(
                            X_remainder_inverse,
                            columns=remainder_cols,
                            index=X.index
                        )

                    inverse_parts.append(X_remainder_inverse)

        result = pd.concat(inverse_parts, axis=1)
        return result

    def supports_inverse_transform(self) -> bool:
        """
        Check if all transformers support inverse transformation.

        Returns:
            True if all transformers support inverse_transform
        """
        for transformer in self.fitted_transformers_.values():
            if not transformer.supports_inverse_transform():
                return False

        if isinstance(self.remainder, PreprocessingStep):
            if not self.remainder.supports_inverse_transform():
                return False

        return True

    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names after transformation.

        Returns:
            List of output column names

        Raises:
            RuntimeError: If transformer has not been fitted
        """
        self._check_fitted()

        feature_names = []

        for name in self.fitted_transformers_:
            columns = self.selected_columns_[name]
            feature_names.extend(columns)

        if self.remainder_columns_:
            if self.remainder in ['passthrough']:
                feature_names.extend(self.remainder_columns_)

        return feature_names

    def get_transformer(self, name: str) -> PreprocessingStep:
        """
        Get a fitted transformer by name.

        Args:
            name: Name of the transformer

        Returns:
            Fitted transformer

        Raises:
            ValueError: If transformer name not found
            RuntimeError: If transformer has not been fitted
        """
        self._check_fitted()

        if name not in self.fitted_transformers_:
            raise ValueError(f"Transformer '{name}' not found")

        return self.fitted_transformers_[name]

    def get_column_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping of transformer names to their columns.

        Returns:
            Dictionary mapping transformer names to column lists
        """
        return self.selected_columns_.copy()

    def __repr__(self) -> str:
        """String representation."""
        fitted_status = "fitted" if self.fitted else "not fitted"
        return (
            f"ColumnTransformer({fitted_status}, "
            f"{len(self.transformers)} transformers, "
            f"remainder='{self.remainder}')"
        )


# Convenience function

def make_column_transformer(
    *transformers,
    remainder: Union[str, PreprocessingStep] = 'passthrough',
    name: Optional[str] = None
) -> ColumnTransformer:
    """
    Convenience function to create a ColumnTransformer.

    Args:
        *transformers: Variable number of (name, transformer, selector) tuples
        remainder: How to handle remaining columns
        name: Optional name for the transformer

    Returns:
        ColumnTransformer instance

    Example:
        >>> transformer = make_column_transformer(
        ...     ('num', StandardScaler(), NumericSelector()),
        ...     ('cat', OneHotEncoder(), CategoricalSelector()),
        ...     remainder='drop'
        ... )
    """
    return ColumnTransformer(
        transformers=list(transformers),
        remainder=remainder,
        name=name
    )
