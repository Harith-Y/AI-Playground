"""
Column selector utilities for intelligent column type-based selection.

Provides flexible column selection based on types, patterns, and predicates,
enabling automatic routing of transformations to appropriate column types.
"""

from typing import List, Optional, Union, Callable, Pattern
import pandas as pd
import numpy as np
import re
from abc import ABC, abstractmethod

from app.ml_engine.utils.column_type_detector import ColumnTypeDetector, ColumnType
from app.utils.logger import get_logger

logger = get_logger("column_selector")


class BaseColumnSelector(ABC):
    """
    Abstract base class for column selectors.

    Column selectors are callable objects that take a DataFrame and return
    a list of column names matching specific criteria.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize column selector.

        Args:
            name: Optional name for this selector
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> List[str]:
        """
        Select columns from DataFrame.

        Args:
            df: DataFrame to select columns from

        Returns:
            List of selected column names
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class TypeSelector(BaseColumnSelector):
    """
    Select columns based on detected semantic type.

    Uses ColumnTypeDetector to identify column types and select those
    matching the specified types.

    Example:
        >>> # Select all numeric columns
        >>> selector = TypeSelector([ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE])
        >>> numeric_cols = selector(df)
    """

    def __init__(
        self,
        column_types: Union[ColumnType, List[ColumnType]],
        detector: Optional[ColumnTypeDetector] = None,
        name: Optional[str] = None
    ):
        """
        Initialize type-based selector.

        Args:
            column_types: Single type or list of types to select
            detector: Optional custom ColumnTypeDetector
            name: Optional name for this selector
        """
        super().__init__(name)

        if isinstance(column_types, ColumnType):
            column_types = [column_types]

        self.column_types = column_types
        self.detector = detector or ColumnTypeDetector()

    def __call__(self, df: pd.DataFrame) -> List[str]:
        """Select columns matching specified types."""
        detected_types = self.detector.detect(df)

        selected = [
            col for col, col_type in detected_types.items()
            if col_type in self.column_types
        ]

        logger.debug(f"TypeSelector selected {len(selected)} columns: {selected}")
        return selected


class NumericSelector(TypeSelector):
    """
    Convenience selector for all numeric columns.

    Selects columns of any numeric type (continuous, discrete, binary).

    Example:
        >>> selector = NumericSelector()
        >>> numeric_cols = selector(df)
    """

    def __init__(
        self,
        include_binary: bool = True,
        detector: Optional[ColumnTypeDetector] = None,
        name: Optional[str] = None
    ):
        """
        Initialize numeric selector.

        Args:
            include_binary: Whether to include binary numeric columns
            detector: Optional custom ColumnTypeDetector
            name: Optional name for this selector
        """
        types = [
            ColumnType.NUMERIC_CONTINUOUS,
            ColumnType.NUMERIC_DISCRETE
        ]

        if include_binary:
            types.append(ColumnType.NUMERIC_BINARY)

        super().__init__(column_types=types, detector=detector, name=name)


class CategoricalSelector(TypeSelector):
    """
    Convenience selector for all categorical columns.

    Selects columns of any categorical type (nominal, ordinal, binary).

    Example:
        >>> selector = CategoricalSelector()
        >>> cat_cols = selector(df)
    """

    def __init__(
        self,
        include_binary: bool = True,
        include_boolean: bool = True,
        detector: Optional[ColumnTypeDetector] = None,
        name: Optional[str] = None
    ):
        """
        Initialize categorical selector.

        Args:
            include_binary: Whether to include binary categorical columns
            include_boolean: Whether to include boolean columns
            detector: Optional custom ColumnTypeDetector
            name: Optional name for this selector
        """
        types = [
            ColumnType.CATEGORICAL_NOMINAL,
            ColumnType.CATEGORICAL_ORDINAL
        ]

        if include_binary:
            types.append(ColumnType.CATEGORICAL_BINARY)

        if include_boolean:
            types.append(ColumnType.BOOLEAN)

        super().__init__(column_types=types, detector=detector, name=name)


class DtypeSelector(BaseColumnSelector):
    """
    Select columns based on pandas dtype.

    Simpler than TypeSelector, just checks pandas dtype without semantic analysis.

    Example:
        >>> # Select all float columns
        >>> selector = DtypeSelector(include=[np.float64, np.float32])
        >>> float_cols = selector(df)
    """

    def __init__(
        self,
        include: Optional[List] = None,
        exclude: Optional[List] = None,
        name: Optional[str] = None
    ):
        """
        Initialize dtype-based selector.

        Args:
            include: List of dtypes to include (None = all)
            exclude: List of dtypes to exclude
            name: Optional name for this selector
        """
        super().__init__(name)
        self.include = include
        self.exclude = exclude or []

    def __call__(self, df: pd.DataFrame) -> List[str]:
        """Select columns based on dtype."""
        if self.include is not None:
            selected = df.select_dtypes(include=self.include).columns.tolist()
        else:
            selected = df.columns.tolist()

        if self.exclude:
            excluded = df.select_dtypes(include=self.exclude).columns.tolist()
            selected = [col for col in selected if col not in excluded]

        logger.debug(f"DtypeSelector selected {len(selected)} columns")
        return selected


class PatternSelector(BaseColumnSelector):
    """
    Select columns based on name pattern matching.

    Supports regex patterns for flexible column selection.

    Example:
        >>> # Select all columns ending with '_id'
        >>> selector = PatternSelector(pattern=r'.*_id$')
        >>> id_cols = selector(df)
    """

    def __init__(
        self,
        pattern: Union[str, Pattern],
        case_sensitive: bool = False,
        invert: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize pattern-based selector.

        Args:
            pattern: Regex pattern to match against column names
            case_sensitive: Whether to use case-sensitive matching
            invert: If True, select columns that DON'T match pattern
            name: Optional name for this selector
        """
        super().__init__(name)

        if isinstance(pattern, str):
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern, flags)

        self.pattern = pattern
        self.invert = invert

    def __call__(self, df: pd.DataFrame) -> List[str]:
        """Select columns matching pattern."""
        if self.invert:
            selected = [col for col in df.columns if not self.pattern.match(str(col))]
        else:
            selected = [col for col in df.columns if self.pattern.match(str(col))]

        logger.debug(f"PatternSelector selected {len(selected)} columns")
        return selected


class PredicateSelector(BaseColumnSelector):
    """
    Select columns based on custom predicate function.

    Most flexible selector - allows arbitrary logic for column selection.

    Example:
        >>> # Select columns with >50% missing values
        >>> selector = PredicateSelector(lambda df, col: df[col].isnull().sum() / len(df) > 0.5)
        >>> high_missing_cols = selector(df)
    """

    def __init__(
        self,
        predicate: Callable[[pd.DataFrame, str], bool],
        name: Optional[str] = None
    ):
        """
        Initialize predicate-based selector.

        Args:
            predicate: Function that takes (DataFrame, column_name) and returns bool
            name: Optional name for this selector
        """
        super().__init__(name)
        self.predicate = predicate

    def __call__(self, df: pd.DataFrame) -> List[str]:
        """Select columns where predicate returns True."""
        selected = [col for col in df.columns if self.predicate(df, col)]

        logger.debug(f"PredicateSelector selected {len(selected)} columns")
        return selected


class ExplicitSelector(BaseColumnSelector):
    """
    Select explicitly specified columns.

    Simplest selector - just returns the provided column list.

    Example:
        >>> selector = ExplicitSelector(['age', 'salary', 'years_exp'])
        >>> cols = selector(df)  # Returns ['age', 'salary', 'years_exp']
    """

    def __init__(
        self,
        columns: List[str],
        raise_on_missing: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize explicit selector.

        Args:
            columns: List of column names to select
            raise_on_missing: If True, raise error if columns not in DataFrame
            name: Optional name for this selector
        """
        super().__init__(name)
        self.columns = columns
        self.raise_on_missing = raise_on_missing

    def __call__(self, df: pd.DataFrame) -> List[str]:
        """Return explicitly specified columns."""
        if self.raise_on_missing:
            missing = set(self.columns) - set(df.columns)
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")

        # Return columns that exist in DataFrame
        selected = [col for col in self.columns if col in df.columns]

        logger.debug(f"ExplicitSelector selected {len(selected)} columns")
        return selected


class CompositeSelector(BaseColumnSelector):
    """
    Combine multiple selectors with set operations.

    Supports union (OR), intersection (AND), and difference operations.

    Example:
        >>> # Select numeric columns except IDs
        >>> selector = CompositeSelector(
        ...     NumericSelector(),
        ...     PatternSelector(r'.*_id$'),
        ...     operation='difference'
        ... )
    """

    def __init__(
        self,
        selector1: BaseColumnSelector,
        selector2: BaseColumnSelector,
        operation: str = 'union',
        name: Optional[str] = None
    ):
        """
        Initialize composite selector.

        Args:
            selector1: First selector
            selector2: Second selector
            operation: Set operation - 'union', 'intersection', or 'difference'
            name: Optional name for this selector
        """
        super().__init__(name)

        if operation not in ['union', 'intersection', 'difference']:
            raise ValueError(f"Invalid operation: {operation}")

        self.selector1 = selector1
        self.selector2 = selector2
        self.operation = operation

    def __call__(self, df: pd.DataFrame) -> List[str]:
        """Select columns using set operation."""
        cols1 = set(self.selector1(df))
        cols2 = set(self.selector2(df))

        if self.operation == 'union':
            selected = list(cols1 | cols2)
        elif self.operation == 'intersection':
            selected = list(cols1 & cols2)
        else:  # difference
            selected = list(cols1 - cols2)

        # Preserve original column order from DataFrame
        selected = [col for col in df.columns if col in selected]

        logger.debug(f"CompositeSelector ({self.operation}) selected {len(selected)} columns")
        return selected


# Convenience functions

def select_numeric_columns(
    df: pd.DataFrame,
    include_binary: bool = True,
    detector: Optional[ColumnTypeDetector] = None
) -> List[str]:
    """
    Select all numeric columns from DataFrame.

    Args:
        df: DataFrame to select from
        include_binary: Whether to include binary numeric columns
        detector: Optional custom ColumnTypeDetector

    Returns:
        List of numeric column names
    """
    selector = NumericSelector(include_binary=include_binary, detector=detector)
    return selector(df)


def select_categorical_columns(
    df: pd.DataFrame,
    include_binary: bool = True,
    include_boolean: bool = True,
    detector: Optional[ColumnTypeDetector] = None
) -> List[str]:
    """
    Select all categorical columns from DataFrame.

    Args:
        df: DataFrame to select from
        include_binary: Whether to include binary categorical columns
        include_boolean: Whether to include boolean columns
        detector: Optional custom ColumnTypeDetector

    Returns:
        List of categorical column names
    """
    selector = CategoricalSelector(
        include_binary=include_binary,
        include_boolean=include_boolean,
        detector=detector
    )
    return selector(df)


def select_by_type(
    df: pd.DataFrame,
    column_types: Union[ColumnType, List[ColumnType]],
    detector: Optional[ColumnTypeDetector] = None
) -> List[str]:
    """
    Select columns by detected type(s).

    Args:
        df: DataFrame to select from
        column_types: Single type or list of types to select
        detector: Optional custom ColumnTypeDetector

    Returns:
        List of column names matching specified types
    """
    selector = TypeSelector(column_types=column_types, detector=detector)
    return selector(df)


def select_by_pattern(
    df: pd.DataFrame,
    pattern: Union[str, Pattern],
    case_sensitive: bool = False,
    invert: bool = False
) -> List[str]:
    """
    Select columns matching a pattern.

    Args:
        df: DataFrame to select from
        pattern: Regex pattern to match
        case_sensitive: Whether to use case-sensitive matching
        invert: If True, select columns that DON'T match

    Returns:
        List of column names matching pattern
    """
    selector = PatternSelector(pattern=pattern, case_sensitive=case_sensitive, invert=invert)
    return selector(df)
