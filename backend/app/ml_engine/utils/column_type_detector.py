"""
Automatic column type detection for pandas DataFrames.

This module provides intelligent detection of column types beyond basic pandas dtypes,
including categorical, numeric, datetime, text, and ID columns. It uses heuristics
based on unique value counts, data patterns, and statistical properties.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import re
from collections import Counter
from app.utils.logger import get_logger

logger = get_logger("column_type_detector")


class ColumnType(str, Enum):
    """
    Enumeration of detected column types.

    Types are ordered from most specific to least specific to help
    with type resolution when multiple types could apply.
    """
    # Identifiers
    ID = "id"                           # Unique identifiers (IDs, keys)

    # Temporal
    DATETIME = "datetime"               # Datetime/timestamp columns
    DATE = "date"                       # Date-only columns
    TIME = "time"                       # Time-only columns

    # Numeric
    NUMERIC_CONTINUOUS = "numeric_continuous"   # Continuous numeric values
    NUMERIC_DISCRETE = "numeric_discrete"       # Discrete/count numeric values
    NUMERIC_BINARY = "numeric_binary"           # Binary numeric (0/1, True/False)

    # Categorical
    CATEGORICAL_ORDINAL = "categorical_ordinal"     # Ordered categories
    CATEGORICAL_NOMINAL = "categorical_nominal"     # Unordered categories
    CATEGORICAL_BINARY = "categorical_binary"       # Binary categorical

    # Text
    TEXT_LONG = "text_long"             # Long text (descriptions, comments)
    TEXT_SHORT = "text_short"           # Short text (names, titles)

    # Special
    BOOLEAN = "boolean"                 # Boolean columns
    CONSTANT = "constant"               # Columns with single value
    MIXED = "mixed"                     # Mixed types (problematic)
    UNKNOWN = "unknown"                 # Cannot determine type


class ColumnTypeDetector:
    """
    Automatic detection of column types in pandas DataFrames.

    Uses multiple heuristics to determine the semantic type of each column,
    going beyond simple dtype checking to understand the actual nature of the data.

    Detection Heuristics:
    - Unique value ratio (for IDs and categorical)
    - Data patterns (for dates, IDs, etc.)
    - Statistical properties (for continuous vs discrete)
    - Value distributions
    - String length analysis (for text)

    Example:
        >>> detector = ColumnTypeDetector()
        >>> types = detector.detect(df)
        >>> print(types)
        {'user_id': ColumnType.ID, 'age': ColumnType.NUMERIC_DISCRETE, ...}
    """

    def __init__(
        self,
        categorical_threshold: float = 0.05,
        id_threshold: float = 0.95,
        text_length_threshold: int = 50,
        sample_size: Optional[int] = 10000,
    ):
        """
        Initialize the column type detector.

        Args:
            categorical_threshold: Max ratio of unique values to consider categorical (default: 0.05 = 5%)
            id_threshold: Min ratio of unique values to consider ID column (default: 0.95 = 95%)
            text_length_threshold: Min average string length to consider long text (default: 50)
            sample_size: Number of rows to sample for large datasets (None = use all)
        """
        self.categorical_threshold = categorical_threshold
        self.id_threshold = id_threshold
        self.text_length_threshold = text_length_threshold
        self.sample_size = sample_size

        # Common ID column name patterns
        self.id_patterns = [
            r"^id$",
            r"_id$",
            r"^.*_id$",
            r"^id_.*$",
            r"^.*key$",
            r"^.*_key$",
            r"^uuid$",
            r"^guid$",
        ]

        # Common ordinal category patterns
        self.ordinal_patterns = {
            "education": ["elementary", "middle", "high", "bachelor", "master", "phd"],
            "size": ["xs", "s", "m", "l", "xl", "xxl"],
            "rating": ["poor", "fair", "good", "very good", "excellent"],
            "agreement": ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
        }

        logger.info(f"Initialized ColumnTypeDetector with thresholds: categorical={categorical_threshold}, "
                   f"id={id_threshold}, text_length={text_length_threshold}")

    def detect(self, df: pd.DataFrame) -> Dict[str, ColumnType]:
        """
        Detect types for all columns in a DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary mapping column names to detected types
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return {}

        # Sample if necessary
        if self.sample_size and len(df) > self.sample_size:
            df_sample = df.sample(n=self.sample_size, random_state=42)
            logger.info(f"Sampling {self.sample_size} rows from {len(df)} total rows")
        else:
            df_sample = df

        column_types = {}

        for col in df.columns:
            try:
                col_type = self._detect_column_type(df_sample[col], col)
                column_types[col] = col_type
                logger.debug(f"Column '{col}': detected as {col_type}")
            except Exception as e:
                logger.error(f"Error detecting type for column '{col}': {e}")
                column_types[col] = ColumnType.UNKNOWN

        return column_types

    def _detect_column_type(self, series: pd.Series, col_name: str) -> ColumnType:
        """
        Detect the type of a single column.

        Args:
            series: Column data
            col_name: Column name (used for pattern matching)

        Returns:
            Detected column type
        """
        # Handle empty or all-null columns
        if series.isna().all():
            return ColumnType.CONSTANT

        # Drop nulls for analysis
        series_clean = series.dropna()

        if len(series_clean) == 0:
            return ColumnType.CONSTANT

        # Check for constant values
        if series_clean.nunique() == 1:
            return ColumnType.CONSTANT

        # Get unique value ratio
        unique_ratio = series_clean.nunique() / len(series_clean)

        # 1. Check for ID columns
        if self._is_id_column(series_clean, col_name, unique_ratio):
            return ColumnType.ID

        # 2. Check for datetime columns
        datetime_type = self._detect_datetime(series_clean)
        if datetime_type:
            return datetime_type

        # 3. Check for boolean columns
        if self._is_boolean(series_clean):
            return ColumnType.BOOLEAN

        # 4. Check for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            return self._detect_numeric_type(series_clean, unique_ratio)

        # 5. Check for categorical/text columns
        return self._detect_categorical_or_text(series_clean, col_name, unique_ratio)

    def _is_id_column(self, series: pd.Series, col_name: str, unique_ratio: float) -> bool:
        """Check if column is an ID/identifier column."""
        # Check name patterns
        col_lower = col_name.lower()
        if any(re.match(pattern, col_lower, re.IGNORECASE) for pattern in self.id_patterns):
            if unique_ratio >= 0.9:  # Slightly lower threshold if name suggests ID
                return True

        # Check uniqueness
        if unique_ratio >= self.id_threshold:
            # Additional checks for ID-like patterns
            if series.dtype == 'object':
                # Check if values look like IDs (alphanumeric with dashes/underscores)
                sample = series.head(100).astype(str)
                id_pattern = r'^[a-zA-Z0-9_-]+$'
                matching = sample.str.match(id_pattern).sum()
                if matching / len(sample) > 0.8:
                    return True
            return True

        return False

    def _detect_datetime(self, series: pd.Series) -> Optional[ColumnType]:
        """Detect if column is datetime/date/time."""
        # Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            # Check if time component exists
            if hasattr(series.iloc[0], 'hour'):
                times = series.dt.time
                if times.nunique() > 1:
                    return ColumnType.DATETIME
                else:
                    return ColumnType.DATE
            return ColumnType.DATETIME

        # Try to parse as datetime
        if series.dtype == 'object':
            try:
                sample = series.head(100)
                parsed = pd.to_datetime(sample, errors='coerce')
                valid_ratio = parsed.notna().sum() / len(sample)

                if valid_ratio > 0.8:
                    # Check if it's date-only or datetime
                    parsed_full = pd.to_datetime(series, errors='coerce')
                    if parsed_full.dt.time.nunique() > 1:
                        return ColumnType.DATETIME
                    else:
                        return ColumnType.DATE
            except:
                pass

        return None

    def _is_boolean(self, series: pd.Series) -> bool:
        """Check if column is boolean."""
        if pd.api.types.is_bool_dtype(series):
            return True

        # Check for boolean-like values
        unique_vals = series.unique()
        if len(unique_vals) == 2:
            vals_lower = set(str(v).lower() for v in unique_vals)
            boolean_sets = [
                {'true', 'false'},
                {'yes', 'no'},
                {'y', 'n'},
                {'1', '0'},
                {'1.0', '0.0'},
                {'t', 'f'},
            ]
            if vals_lower in boolean_sets:
                return True

        return False

    def _detect_numeric_type(self, series: pd.Series, unique_ratio: float) -> ColumnType:
        """Detect specific numeric type (continuous, discrete, binary)."""
        unique_vals = series.nunique()

        # Binary numeric
        if unique_vals == 2:
            vals = set(series.unique())
            if vals == {0, 1} or vals == {0.0, 1.0}:
                return ColumnType.NUMERIC_BINARY

        # Discrete vs continuous
        # If all values are integers or low unique count, likely discrete
        is_integer = (series == series.astype(int)).all()

        if is_integer and (unique_vals < 50 or unique_ratio < 0.05):
            return ColumnType.NUMERIC_DISCRETE

        # Check if values are counts (non-negative integers)
        if is_integer and (series >= 0).all():
            if unique_vals < 100:
                return ColumnType.NUMERIC_DISCRETE

        return ColumnType.NUMERIC_CONTINUOUS

    def _detect_categorical_or_text(self, series: pd.Series, col_name: str, unique_ratio: float) -> ColumnType:
        """Detect if column is categorical or text, and what subtype."""
        unique_vals = series.nunique()

        # Binary categorical
        if unique_vals == 2:
            return ColumnType.CATEGORICAL_BINARY

        # Check if it's categorical based on unique ratio
        if unique_ratio <= self.categorical_threshold:
            # Check if ordinal
            if self._is_ordinal_categorical(series, col_name):
                return ColumnType.CATEGORICAL_ORDINAL
            return ColumnType.CATEGORICAL_NOMINAL

        # Must be text - check length to classify
        if series.dtype == 'object':
            avg_length = series.astype(str).str.len().mean()
            if avg_length >= self.text_length_threshold:
                return ColumnType.TEXT_LONG
            else:
                # Could be high-cardinality categorical or short text
                if unique_ratio > 0.5:
                    return ColumnType.TEXT_SHORT
                else:
                    return ColumnType.CATEGORICAL_NOMINAL

        return ColumnType.UNKNOWN

    def _is_ordinal_categorical(self, series: pd.Series, col_name: str) -> bool:
        """Check if categorical column is ordinal."""
        col_lower = col_name.lower()

        # Check against known ordinal patterns
        for pattern_name, pattern_values in self.ordinal_patterns.items():
            if pattern_name in col_lower:
                unique_vals = set(str(v).lower() for v in series.unique())
                pattern_set = set(pattern_values)
                if unique_vals.issubset(pattern_set):
                    return True

        # Check for numeric-looking ordinal (e.g., "1st", "2nd", "3rd")
        sample = series.head(100).astype(str)
        ordinal_pattern = r'^\d+(st|nd|rd|th)$'
        matching = sample.str.match(ordinal_pattern, flags=re.IGNORECASE).sum()
        if matching / len(sample) > 0.7:
            return True

        return False

    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed information about all columns.

        Args:
            df: DataFrame to analyze

        Returns:
            DataFrame with column information including detected types
        """
        types = self.detect(df)

        info_data = []
        for col in df.columns:
            series = df[col]
            info = {
                'column': col,
                'detected_type': types.get(col, ColumnType.UNKNOWN).value,
                'pandas_dtype': str(series.dtype),
                'null_count': series.isna().sum(),
                'null_percentage': (series.isna().sum() / len(df)) * 100,
                'unique_count': series.nunique(),
                'unique_ratio': series.nunique() / len(df) if len(df) > 0 else 0,
            }

            # Add type-specific info
            if pd.api.types.is_numeric_dtype(series):
                info.update({
                    'min': series.min(),
                    'max': series.max(),
                    'mean': series.mean(),
                    'median': series.median(),
                })
            elif series.dtype == 'object':
                avg_len = series.dropna().astype(str).str.len().mean()
                info['avg_length'] = avg_len if not pd.isna(avg_len) else 0

            info_data.append(info)

        return pd.DataFrame(info_data)


def detect_column_types(
    df: pd.DataFrame,
    categorical_threshold: float = 0.05,
    id_threshold: float = 0.95,
    text_length_threshold: int = 50,
    sample_size: Optional[int] = 10000,
) -> Dict[str, ColumnType]:
    """
    Convenience function to detect column types.

    Args:
        df: DataFrame to analyze
        categorical_threshold: Max ratio of unique values to consider categorical
        id_threshold: Min ratio of unique values to consider ID column
        text_length_threshold: Min average string length to consider long text
        sample_size: Number of rows to sample for large datasets

    Returns:
        Dictionary mapping column names to detected types

    Example:
        >>> types = detect_column_types(df)
        >>> print(types)
        {'user_id': <ColumnType.ID: 'id'>, 'age': <ColumnType.NUMERIC_DISCRETE: 'numeric_discrete'>}
    """
    detector = ColumnTypeDetector(
        categorical_threshold=categorical_threshold,
        id_threshold=id_threshold,
        text_length_threshold=text_length_threshold,
        sample_size=sample_size,
    )
    return detector.detect(df)
