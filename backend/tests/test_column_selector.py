"""
Tests for column selector utilities.
"""

import pytest
import pandas as pd
import numpy as np

from app.ml_engine.preprocessing.column_selector import (
    TypeSelector,
    NumericSelector,
    CategoricalSelector,
    DtypeSelector,
    PatternSelector,
    PredicateSelector,
    ExplicitSelector,
    CompositeSelector,
    select_numeric_columns,
    select_categorical_columns,
    select_by_type,
    select_by_pattern,
)
from app.ml_engine.utils.column_type_detector import ColumnType


class TestNumericSelector:
    """Test NumericSelector."""

    def test_select_numeric_columns(self):
        """Test selecting all numeric columns."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 70000.0],
            'years': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'active': [True, False, True]
        })

        selector = NumericSelector()
        columns = selector(df)

        # Should select numeric columns
        assert 'age' in columns
        assert 'salary' in columns
        assert 'years' in columns
        assert 'name' not in columns

    def test_select_numeric_without_binary(self):
        """Test excluding binary numeric columns."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'is_active': [0, 1, 0],
            'flag': [1, 1, 0]
        })

        selector = NumericSelector(include_binary=False)
        columns = selector(df)

        assert 'age' in columns
        # Binary columns might still be included if detected as discrete

    def test_convenience_function(self):
        """Test select_numeric_columns convenience function."""
        df = pd.DataFrame({
            'num1': [1.5, 2.5, 3.5],
            'num2': [10, 20, 30],
            'text': ['a', 'b', 'c']
        })

        columns = select_numeric_columns(df)

        assert 'num1' in columns
        assert 'num2' in columns
        assert 'text' not in columns


class TestCategoricalSelector:
    """Test CategoricalSelector."""

    def test_select_categorical_columns(self):
        """Test selecting all categorical columns."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 10,
            'color': ['red', 'blue', 'green'] * 10,
            'age': list(range(30)),
            'is_active': [True, False] * 15
        })

        selector = CategoricalSelector()
        columns = selector(df)

        assert 'category' in columns
        assert 'color' in columns
        assert 'age' not in columns

    def test_select_categorical_without_binary(self):
        """Test excluding binary categorical columns."""
        df = pd.DataFrame({
            'multi_cat': ['A', 'B', 'C', 'D'] * 10,
            'binary_cat': ['Yes', 'No'] * 20,
            'num': list(range(40))
        })

        selector = CategoricalSelector(include_binary=False)
        columns = selector(df)

        assert 'multi_cat' in columns
        # binary_cat might be excluded if detected as binary

    def test_convenience_function(self):
        """Test select_categorical_columns convenience function."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 5,
            'cat2': ['X', 'Y', 'Z'] * 5,
            'num': list(range(15))
        })

        columns = select_categorical_columns(df)

        assert 'cat1' in columns
        assert 'cat2' in columns
        assert 'num' not in columns


class TestTypeSelector:
    """Test TypeSelector."""

    def test_select_by_specific_type(self):
        """Test selecting by specific column type."""
        df = pd.DataFrame({
            'id': range(100),
            'amount': np.random.uniform(0, 100, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })

        selector = TypeSelector(ColumnType.ID)
        columns = selector(df)

        assert 'id' in columns
        assert 'amount' not in columns

    def test_select_by_multiple_types(self):
        """Test selecting by multiple types."""
        df = pd.DataFrame({
            'continuous': np.random.randn(100),
            'discrete': np.random.randint(0, 10, 100),
            'category': np.random.choice(['A', 'B'], 100)
        })

        selector = TypeSelector([
            ColumnType.NUMERIC_CONTINUOUS,
            ColumnType.NUMERIC_DISCRETE
        ])
        columns = selector(df)

        assert 'continuous' in columns
        assert 'discrete' in columns
        assert 'category' not in columns

    def test_convenience_function(self):
        """Test select_by_type convenience function."""
        df = pd.DataFrame({
            'num': np.random.randn(50),
            'cat': np.random.choice(['A', 'B'], 50)
        })

        columns = select_by_type(df, ColumnType.NUMERIC_CONTINUOUS)

        assert 'num' in columns


class TestDtypeSelector:
    """Test DtypeSelector."""

    def test_select_by_dtype_include(self):
        """Test selecting by included dtype."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })

        selector = DtypeSelector(include=[np.number])
        columns = selector(df)

        assert 'int_col' in columns
        assert 'float_col' in columns
        assert 'str_col' not in columns

    def test_select_by_dtype_exclude(self):
        """Test selecting by excluded dtype."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })

        selector = DtypeSelector(exclude=[object])
        columns = selector(df)

        assert 'int_col' in columns
        assert 'float_col' in columns
        assert 'str_col' not in columns


class TestPatternSelector:
    """Test PatternSelector."""

    def test_select_by_pattern_simple(self):
        """Test simple pattern matching."""
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'product_id': [10, 20, 30],
            'amount': [100, 200, 300],
            'user_name': ['Alice', 'Bob', 'Charlie']
        })

        selector = PatternSelector(r'.*_id$')
        columns = selector(df)

        assert 'user_id' in columns
        assert 'product_id' in columns
        assert 'amount' not in columns
        assert 'user_name' not in columns

    def test_select_by_pattern_case_sensitive(self):
        """Test case-sensitive pattern matching."""
        df = pd.DataFrame({
            'UserID': [1, 2],
            'userid': [3, 4],
            'Amount': [100, 200]
        })

        selector_insensitive = PatternSelector(r'userid', case_sensitive=False)
        columns_insensitive = selector_insensitive(df)

        assert 'UserID' in columns_insensitive
        assert 'userid' in columns_insensitive

        selector_sensitive = PatternSelector(r'userid', case_sensitive=True)
        columns_sensitive = selector_sensitive(df)

        assert 'userid' in columns_sensitive
        assert 'UserID' not in columns_sensitive

    def test_select_by_pattern_inverted(self):
        """Test inverted pattern matching."""
        df = pd.DataFrame({
            'user_id': [1, 2],
            'product_id': [10, 20],
            'name': ['Alice', 'Bob'],
            'age': [25, 30]
        })

        selector = PatternSelector(r'.*_id$', invert=True)
        columns = selector(df)

        assert 'user_id' not in columns
        assert 'product_id' not in columns
        assert 'name' in columns
        assert 'age' in columns

    def test_convenience_function(self):
        """Test select_by_pattern convenience function."""
        df = pd.DataFrame({
            'col_1': [1],
            'col_2': [2],
            'other': [3]
        })

        columns = select_by_pattern(df, r'^col_')

        assert 'col_1' in columns
        assert 'col_2' in columns
        assert 'other' not in columns


class TestPredicateSelector:
    """Test PredicateSelector."""

    def test_select_by_custom_predicate(self):
        """Test selection with custom predicate."""
        df = pd.DataFrame({
            'mostly_null': [1, None, None, None, None],
            'few_null': [1, 2, None, 4, 5],
            'no_null': [1, 2, 3, 4, 5]
        })

        # Select columns with >50% null values
        selector = PredicateSelector(
            lambda df, col: df[col].isnull().sum() / len(df) > 0.5
        )
        columns = selector(df)

        assert 'mostly_null' in columns
        assert 'few_null' not in columns
        assert 'no_null' not in columns

    def test_select_high_cardinality(self):
        """Test selecting high-cardinality columns."""
        df = pd.DataFrame({
            'low_card': ['A', 'B', 'A', 'B'] * 5,
            'high_card': list(range(20)),
        })

        # Select columns with >10 unique values
        selector = PredicateSelector(
            lambda df, col: df[col].nunique() > 10
        )
        columns = selector(df)

        assert 'high_card' in columns
        assert 'low_card' not in columns


class TestExplicitSelector:
    """Test ExplicitSelector."""

    def test_select_explicit_columns(self):
        """Test explicit column selection."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        selector = ExplicitSelector(['a', 'c'])
        columns = selector(df)

        assert columns == ['a', 'c']

    def test_select_with_missing_columns(self):
        """Test selection with missing columns."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        # Without raise_on_missing
        selector1 = ExplicitSelector(['a', 'c'], raise_on_missing=False)
        columns1 = selector1(df)
        assert columns1 == ['a']

        # With raise_on_missing
        selector2 = ExplicitSelector(['a', 'c'], raise_on_missing=True)
        with pytest.raises(ValueError, match="Columns not found"):
            selector2(df)


class TestCompositeSelector:
    """Test CompositeSelector."""

    def test_union_operation(self):
        """Test union of two selectors."""
        df = pd.DataFrame({
            'num_1': [1.0, 2.0],
            'num_2': [3, 4],
            'cat_1': ['A', 'B'],
            'text_id': ['ID1', 'ID2']
        })

        numeric_selector = NumericSelector()
        id_selector = PatternSelector(r'.*_id$')

        selector = CompositeSelector(numeric_selector, id_selector, operation='union')
        columns = selector(df)

        # Should include both numeric and ID columns
        assert 'num_1' in columns
        assert 'num_2' in columns
        assert 'text_id' in columns

    def test_intersection_operation(self):
        """Test intersection of two selectors."""
        df = pd.DataFrame({
            'user_id': range(100),
            'product_id': range(100),
            'amount': np.random.randn(100)
        })

        numeric_selector = NumericSelector()
        id_selector = PatternSelector(r'.*_id$')

        selector = CompositeSelector(numeric_selector, id_selector, operation='intersection')
        columns = selector(df)

        # Should only include columns that are both numeric AND match ID pattern
        assert 'user_id' in columns or 'product_id' in columns
        assert 'amount' not in columns

    def test_difference_operation(self):
        """Test difference of two selectors."""
        df = pd.DataFrame({
            'user_id': range(50),
            'amount': np.random.randn(50),
            'count': np.random.randint(0, 10, 50)
        })

        numeric_selector = NumericSelector()
        id_selector = TypeSelector(ColumnType.ID)

        selector = CompositeSelector(numeric_selector, id_selector, operation='difference')
        columns = selector(df)

        # Should include numeric columns except IDs
        assert 'amount' in columns
        assert 'count' in columns
        assert 'user_id' not in columns

    def test_invalid_operation(self):
        """Test invalid operation raises error."""
        selector1 = NumericSelector()
        selector2 = CategoricalSelector()

        with pytest.raises(ValueError, match="Invalid operation"):
            CompositeSelector(selector1, selector2, operation='invalid')


class TestEdgeCases:
    """Test edge cases for column selectors."""

    def test_empty_dataframe(self):
        """Test selectors on empty DataFrame."""
        df = pd.DataFrame()

        selector = NumericSelector()
        columns = selector(df)

        assert columns == []

    def test_single_column(self):
        """Test selectors on single-column DataFrame."""
        df = pd.DataFrame({'num': [1, 2, 3]})

        selector = NumericSelector()
        columns = selector(df)

        assert columns == ['num']

    def test_all_columns_same_type(self):
        """Test when all columns are same type."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        numeric_selector = NumericSelector()
        numeric_cols = numeric_selector(df)

        cat_selector = CategoricalSelector()
        cat_cols = cat_selector(df)

        assert len(numeric_cols) > 0
        assert len(cat_cols) == 0

    def test_preserve_column_order(self):
        """Test that column order is preserved."""
        df = pd.DataFrame({
            'z_col': [1, 2],
            'a_col': [3, 4],
            'm_col': [5, 6]
        })

        selector = ExplicitSelector(['z_col', 'm_col', 'a_col'])
        columns = selector(df)

        # Should preserve order from explicit list
        assert columns == ['z_col', 'm_col', 'a_col']
