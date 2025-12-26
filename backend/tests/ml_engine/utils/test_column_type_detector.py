"""
Tests for automatic column type detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from app.ml_engine.utils.column_type_detector import (
    ColumnTypeDetector,
    detect_column_types,
    ColumnType,
)


class TestColumnTypeDetector:
    """Tests for ColumnTypeDetector class."""

    def test_initialization(self):
        """Test detector initialization with default and custom parameters."""
        # Default initialization
        detector = ColumnTypeDetector()
        assert detector.categorical_threshold == 0.05
        assert detector.id_threshold == 0.95
        assert detector.text_length_threshold == 50
        assert detector.sample_size == 10000

        # Custom initialization
        detector = ColumnTypeDetector(
            categorical_threshold=0.1,
            id_threshold=0.9,
            text_length_threshold=100,
            sample_size=5000,
        )
        assert detector.categorical_threshold == 0.1
        assert detector.id_threshold == 0.9
        assert detector.text_length_threshold == 100
        assert detector.sample_size == 5000

    def test_detect_id_column(self):
        """Test detection of ID columns."""
        detector = ColumnTypeDetector()

        # Test with unique sequential IDs
        df = pd.DataFrame({
            'user_id': range(1000),
            'order_id': [f'ORD-{i:05d}' for i in range(1000)],
            'uuid': [f'{i:08x}-{i:04x}-{i:04x}' for i in range(1000)],
        })

        types = detector.detect(df)
        assert types['user_id'] == ColumnType.ID
        assert types['order_id'] == ColumnType.ID
        assert types['uuid'] == ColumnType.ID

    def test_detect_numeric_continuous(self):
        """Test detection of continuous numeric columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'price': np.random.uniform(10, 100, 1000),
            'temperature': np.random.normal(20, 5, 1000),
            'weight_kg': np.random.exponential(5, 1000),
        })

        types = detector.detect(df)
        assert types['price'] == ColumnType.NUMERIC_CONTINUOUS
        assert types['temperature'] == ColumnType.NUMERIC_CONTINUOUS
        assert types['weight_kg'] == ColumnType.NUMERIC_CONTINUOUS

    def test_detect_numeric_discrete(self):
        """Test detection of discrete numeric columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'age': np.random.randint(18, 80, 1000),
            'num_children': np.random.randint(0, 5, 1000),
            'rating': np.random.randint(1, 6, 1000),
        })

        types = detector.detect(df)
        assert types['age'] == ColumnType.NUMERIC_DISCRETE
        assert types['num_children'] == ColumnType.NUMERIC_DISCRETE
        assert types['rating'] == ColumnType.NUMERIC_DISCRETE

    def test_detect_numeric_binary(self):
        """Test detection of binary numeric columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'is_active': np.random.choice([0, 1], 1000),
            'has_discount': np.random.choice([0.0, 1.0], 1000),
        })

        types = detector.detect(df)
        assert types['is_active'] == ColumnType.NUMERIC_BINARY
        assert types['has_discount'] == ColumnType.NUMERIC_BINARY

    def test_detect_boolean(self):
        """Test detection of boolean columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'is_premium': np.random.choice([True, False], 1000),
            'verified': np.random.choice(['yes', 'no'], 1000),
            'active': np.random.choice(['Y', 'N'], 1000),
        })

        types = detector.detect(df)
        assert types['is_premium'] == ColumnType.BOOLEAN
        assert types['verified'] == ColumnType.BOOLEAN
        assert types['active'] == ColumnType.BOOLEAN

    def test_detect_datetime(self):
        """Test detection of datetime columns."""
        detector = ColumnTypeDetector()

        # Create datetime columns
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        datetimes = pd.date_range('2023-01-01 00:00:00', periods=1000, freq='H')

        df = pd.DataFrame({
            'date_col': dates.date,
            'datetime_col': datetimes,
            'timestamp_str': datetimes.astype(str),
        })

        types = detector.detect(df)
        # Date column should be detected as DATE
        assert types['datetime_col'] == ColumnType.DATETIME

    def test_detect_categorical_nominal(self):
        """Test detection of nominal categorical columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'color': np.random.choice(['red', 'blue', 'green', 'yellow'], 1000),
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
        })

        types = detector.detect(df)
        assert types['color'] == ColumnType.CATEGORICAL_NOMINAL
        assert types['city'] == ColumnType.CATEGORICAL_NOMINAL
        assert types['category'] == ColumnType.CATEGORICAL_NOMINAL

    def test_detect_categorical_binary(self):
        """Test detection of binary categorical columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], 1000),
            'status': np.random.choice(['Active', 'Inactive'], 1000),
        })

        types = detector.detect(df)
        assert types['gender'] == ColumnType.CATEGORICAL_BINARY
        assert types['status'] == ColumnType.CATEGORICAL_BINARY

    def test_detect_categorical_ordinal(self):
        """Test detection of ordinal categorical columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'education': np.random.choice(['high', 'bachelor', 'master', 'phd'], 1000),
            'size': np.random.choice(['S', 'M', 'L', 'XL'], 1000),
            'rating': np.random.choice(['poor', 'fair', 'good', 'excellent'], 1000),
        })

        types = detector.detect(df)
        assert types['education'] == ColumnType.CATEGORICAL_ORDINAL
        assert types['rating'] == ColumnType.CATEGORICAL_ORDINAL

    def test_detect_text_short(self):
        """Test detection of short text columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'name': [f'Person {i}' for i in range(1000)],
            'title': [f'Title {i}' for i in range(1000)],
        })

        types = detector.detect(df)
        assert types['name'] == ColumnType.TEXT_SHORT
        assert types['title'] == ColumnType.TEXT_SHORT

    def test_detect_text_long(self):
        """Test detection of long text columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'description': [f'This is a long description for item {i}. ' * 5 for i in range(1000)],
            'comment': [f'A detailed comment about something important for record {i}. ' * 3 for i in range(1000)],
        })

        types = detector.detect(df)
        assert types['description'] == ColumnType.TEXT_LONG
        assert types['comment'] == ColumnType.TEXT_LONG

    def test_detect_constant(self):
        """Test detection of constant columns."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'constant_num': [42] * 1000,
            'constant_str': ['same'] * 1000,
            'all_null': [None] * 1000,
        })

        types = detector.detect(df)
        assert types['constant_num'] == ColumnType.CONSTANT
        assert types['constant_str'] == ColumnType.CONSTANT
        assert types['all_null'] == ColumnType.CONSTANT

    def test_detect_with_nulls(self):
        """Test detection with null values."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'with_nulls_num': [1, 2, None, 4, 5] * 200,
            'with_nulls_cat': ['A', 'B', None, 'C', 'A'] * 200,
        })

        types = detector.detect(df)
        # Should still detect correctly despite nulls
        assert types['with_nulls_num'] == ColumnType.NUMERIC_DISCRETE
        assert types['with_nulls_cat'] == ColumnType.CATEGORICAL_NOMINAL

    def test_detect_mixed_types(self):
        """Test detection of columns with mixed types."""
        detector = ColumnTypeDetector()

        # Create a column with mixed types (problematic data)
        mixed_data = ['text', 123, 'more text', 456, None, 'abc'] * 167
        df = pd.DataFrame({
            'mixed': mixed_data[:1000],
        })

        types = detector.detect(df)
        # Mixed types should be detected as text or mixed
        assert types['mixed'] in [ColumnType.TEXT_SHORT, ColumnType.CATEGORICAL_NOMINAL, ColumnType.MIXED]

    def test_detect_empty_dataframe(self):
        """Test detection on empty DataFrame."""
        detector = ColumnTypeDetector()
        df = pd.DataFrame()

        types = detector.detect(df)
        assert types == {}

    def test_sampling_large_dataset(self):
        """Test that sampling works for large datasets."""
        detector = ColumnTypeDetector(sample_size=100)

        # Create a large dataset
        df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000),
        })

        types = detector.detect(df)
        # Should still detect correctly with sampling
        assert types['id'] == ColumnType.ID
        assert types['value'] == ColumnType.NUMERIC_CONTINUOUS

    def test_get_column_info(self):
        """Test get_column_info method."""
        detector = ColumnTypeDetector()

        df = pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
        })

        info = detector.get_column_info(df)

        # Check that info DataFrame has correct structure
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 3
        assert 'column' in info.columns
        assert 'detected_type' in info.columns
        assert 'pandas_dtype' in info.columns
        assert 'null_count' in info.columns
        assert 'unique_count' in info.columns

        # Check that detected types are present
        assert info[info['column'] == 'id']['detected_type'].iloc[0] == ColumnType.ID.value
        assert info[info['column'] == 'category']['detected_type'].iloc[0] == ColumnType.CATEGORICAL_NOMINAL.value


class TestDetectColumnTypesFunction:
    """Tests for the convenience function detect_column_types."""

    def test_detect_column_types_function(self):
        """Test the convenience function works correctly."""
        df = pd.DataFrame({
            'id': range(1000),
            'amount': np.random.uniform(0, 100, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'is_valid': np.random.choice([True, False], 1000),
        })

        types = detect_column_types(df)

        assert isinstance(types, dict)
        assert len(types) == 4
        assert types['id'] == ColumnType.ID
        assert types['amount'] == ColumnType.NUMERIC_CONTINUOUS
        assert types['category'] == ColumnType.CATEGORICAL_NOMINAL
        assert types['is_valid'] == ColumnType.BOOLEAN

    def test_detect_column_types_with_custom_thresholds(self):
        """Test convenience function with custom thresholds."""
        df = pd.DataFrame({
            'semi_unique': range(500).tolist() * 2,  # 50% unique
        })

        # With default threshold (0.95), should not be ID
        types1 = detect_column_types(df)
        assert types1['semi_unique'] != ColumnType.ID

        # With lower threshold (0.4), should be ID
        types2 = detect_column_types(df, id_threshold=0.4)
        assert types2['semi_unique'] == ColumnType.ID


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_row_dataframe(self):
        """Test detection on single-row DataFrame."""
        detector = ColumnTypeDetector()
        df = pd.DataFrame({
            'col1': [1],
            'col2': ['text'],
        })

        types = detector.detect(df)
        # Single row should be detected as constant
        assert types['col1'] == ColumnType.CONSTANT
        assert types['col2'] == ColumnType.CONSTANT

    def test_all_unique_string_column(self):
        """Test detection of string column with all unique values."""
        detector = ColumnTypeDetector()
        df = pd.DataFrame({
            'unique_strings': [f'unique_{i}' for i in range(1000)],
        })

        types = detector.detect(df)
        # Should be detected as ID or TEXT_SHORT
        assert types['unique_strings'] in [ColumnType.ID, ColumnType.TEXT_SHORT]

    def test_numeric_stored_as_string(self):
        """Test detection of numeric values stored as strings."""
        detector = ColumnTypeDetector()
        df = pd.DataFrame({
            'numeric_strings': [str(i) for i in range(1000)],
        })

        types = detector.detect(df)
        # Could be ID or TEXT depending on uniqueness
        assert types['numeric_strings'] in [ColumnType.ID, ColumnType.TEXT_SHORT, ColumnType.CATEGORICAL_NOMINAL]

    def test_high_cardinality_categorical(self):
        """Test detection of high-cardinality categorical column."""
        detector = ColumnTypeDetector()

        # Create categorical with many unique values but still < threshold
        num_categories = 30
        df = pd.DataFrame({
            'high_card_cat': np.random.choice([f'Cat_{i}' for i in range(num_categories)], 1000),
        })

        types = detector.detect(df)
        # Should be categorical if unique ratio is below threshold
        assert types['high_card_cat'] in [ColumnType.CATEGORICAL_NOMINAL, ColumnType.TEXT_SHORT]
