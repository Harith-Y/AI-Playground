"""
Tests for ColumnTransformer.
"""

import pytest
import pandas as pd
import numpy as np

from app.ml_engine.preprocessing.column_transformer import ColumnTransformer, make_column_transformer
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler
from app.ml_engine.preprocessing.encoder import LabelEncoder
from app.ml_engine.preprocessing.imputer import MeanImputer
from app.ml_engine.preprocessing.column_selector import (
    NumericSelector,
    CategoricalSelector,
    ExplicitSelector,
    PatternSelector
)


class TestColumnTransformerInitialization:
    """Test ColumnTransformer initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        transformer = ColumnTransformer(transformers=[
            ('numeric', StandardScaler(), NumericSelector()),
            ('categorical', LabelEncoder(column='cat'), CategoricalSelector())
        ])

        assert len(transformer.transformers) == 2
        assert not transformer.fitted

    def test_initialization_with_explicit_columns(self):
        """Test initialization with explicit column lists."""
        transformer = ColumnTransformer(transformers=[
            ('group1', StandardScaler(), ['col1', 'col2']),
            ('group2', MinMaxScaler(), ['col3', 'col4'])
        ])

        assert len(transformer.transformers) == 2

    def test_initialization_with_invalid_transformer(self):
        """Test error on invalid transformer."""
        with pytest.raises(TypeError, match="must be a PreprocessingStep"):
            ColumnTransformer(transformers=[
                ('invalid', "not a transformer", NumericSelector())
            ])

    def test_initialization_with_duplicate_names(self):
        """Test error on duplicate transformer names."""
        with pytest.raises(ValueError, match="Duplicate transformer names"):
            ColumnTransformer(transformers=[
                ('same_name', StandardScaler(), ['col1']),
                ('same_name', MinMaxScaler(), ['col2'])
            ])

    def test_initialization_empty_transformers(self):
        """Test error on empty transformers list."""
        with pytest.raises(ValueError, match="transformers list cannot be empty"):
            ColumnTransformer(transformers=[])


class TestColumnTransformerFit:
    """Test ColumnTransformer fit method."""

    def test_fit_with_numeric_and_categorical(self):
        """Test fitting with numeric and categorical transformers."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['IT', 'HR', 'IT', 'Sales', 'HR']
        })

        transformer = ColumnTransformer(transformers=[
            ('numeric', StandardScaler(), NumericSelector()),
            ('categorical', LabelEncoder(column='department'), CategoricalSelector())
        ])

        transformer.fit(df)

        assert transformer.fitted
        assert 'numeric' in transformer.fitted_transformers_
        assert 'categorical' in transformer.fitted_transformers_

    def test_fit_with_explicit_columns(self):
        """Test fitting with explicit column selection."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        transformer = ColumnTransformer(transformers=[
            ('ab', StandardScaler(), ['a', 'b']),
            ('c', MinMaxScaler(), ['c'])
        ])

        transformer.fit(df)

        assert transformer.selected_columns_['ab'] == ['a', 'b']
        assert transformer.selected_columns_['c'] == ['c']

    def test_fit_with_remainder_passthrough(self):
        """Test fitting with remainder='passthrough'."""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4, 5, 6],
            'text': ['a', 'b', 'c']
        })

        transformer = ColumnTransformer(
            transformers=[('numeric', StandardScaler(), ['num1'])],
            remainder='passthrough'
        )

        transformer.fit(df)

        assert 'num2' in transformer.remainder_columns_
        assert 'text' in transformer.remainder_columns_

    def test_fit_with_remainder_drop(self):
        """Test fitting with remainder='drop'."""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4, 5, 6]
        })

        transformer = ColumnTransformer(
            transformers=[('first', StandardScaler(), ['num1'])],
            remainder='drop'
        )

        transformer.fit(df)

        assert 'num2' in transformer.remainder_columns_


class TestColumnTransformerTransform:
    """Test ColumnTransformer transform method."""

    def test_transform_basic(self):
        """Test basic transformation."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        transformer = ColumnTransformer(transformers=[
            ('all', StandardScaler(), ['x', 'y'])
        ])

        transformer.fit(df)
        result = transformer.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert 'x' in result.columns
        assert 'y' in result.columns
        # Check standardization
        assert abs(result['x'].mean()) < 0.01
        assert abs(result['y'].mean()) < 0.01

    def test_transform_separate_groups(self):
        """Test transforming separate column groups."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [10, 20, 30],
            'c': [100, 200, 300]
        })

        transformer = ColumnTransformer(transformers=[
            ('group1', StandardScaler(), ['a', 'b']),
            ('group2', MinMaxScaler(), ['c'])
        ])

        transformer.fit(df)
        result = transformer.transform(df)

        assert len(result.columns) == 3
        assert 'a' in result.columns
        assert 'b' in result.columns
        assert 'c' in result.columns

    def test_transform_with_remainder_passthrough(self):
        """Test transformation with passthrough remainder."""
        df = pd.DataFrame({
            'scale_me': [1, 2, 3],
            'keep_me': [10, 20, 30]
        })

        transformer = ColumnTransformer(
            transformers=[('scaled', StandardScaler(), ['scale_me'])],
            remainder='passthrough'
        )

        transformer.fit(df)
        result = transformer.transform(df)

        # scale_me should be standardized
        assert abs(result['scale_me'].mean()) < 0.01

        # keep_me should be unchanged
        assert list(result['keep_me']) == [10, 20, 30]

    def test_transform_with_remainder_drop(self):
        """Test transformation with dropped remainder."""
        df = pd.DataFrame({
            'keep': [1, 2, 3],
            'drop1': [10, 20, 30],
            'drop2': ['a', 'b', 'c']
        })

        transformer = ColumnTransformer(
            transformers=[('keeper', StandardScaler(), ['keep'])],
            remainder='drop'
        )

        transformer.fit(df)
        result = transformer.transform(df)

        assert 'keep' in result.columns
        assert 'drop1' not in result.columns
        assert 'drop2' not in result.columns

    def test_transform_not_fitted_error(self):
        """Test error when transforming before fitting."""
        df = pd.DataFrame({'x': [1, 2, 3]})

        transformer = ColumnTransformer(transformers=[
            ('t', StandardScaler(), ['x'])
        ])

        with pytest.raises(RuntimeError, match="must be fitted"):
            transformer.transform(df)


class TestColumnTransformerIntegration:
    """Integration tests for ColumnTransformer."""

    def test_numeric_categorical_pipeline(self):
        """Test complete numeric/categorical processing pipeline."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            'department': ['IT', 'HR', 'IT', 'Sales', 'HR'],
            'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA']
        })

        transformer = ColumnTransformer(transformers=[
            ('numeric', StandardScaler(), NumericSelector()),
            ('categorical', LabelEncoder(column='department'), CategoricalSelector())
        ])

        result = transformer.fit_transform(df)

        assert isinstance(result, pd.DataFrame)
        # Numeric columns should be scaled
        assert 'age' in result.columns
        assert 'salary' in result.columns
        # Categorical should be encoded
        assert 'department' in result.columns or 'city' in result.columns

    def test_mixed_transformers_with_selectors(self):
        """Test different transformers on different column patterns."""
        df = pd.DataFrame({
            'user_id': range(10),
            'age': range(20, 30),
            'income': np.random.uniform(30000, 100000, 10),
            'category': np.random.choice(['A', 'B', 'C'], 10)
        })

        transformer = ColumnTransformer(transformers=[
            ('age_scale', StandardScaler(), ['age']),
            ('income_scale', MinMaxScaler(), ['income']),
            ('cat_encode', LabelEncoder(column='category'), ['category'])
        ], remainder='passthrough')

        result = transformer.fit_transform(df)

        # All columns should be present
        assert len(result.columns) == 4

    def test_fit_transform_workflow(self):
        """Test complete fit-transform workflow."""
        df_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        df_test = pd.DataFrame({
            'x': [6, 7],
            'y': [60, 70]
        })

        transformer = ColumnTransformer(transformers=[
            ('scale', StandardScaler(), ['x', 'y'])
        ])

        # Fit on training data
        transformer.fit(df_train)

        # Transform test data
        result = transformer.transform(df_test)

        assert len(result) == 2
        assert 'x' in result.columns
        assert 'y' in result.columns


class TestColumnTransformerHelpers:
    """Test helper methods of ColumnTransformer."""

    def test_get_transformer(self):
        """Test getting a specific transformer."""
        df = pd.DataFrame({'x': [1, 2, 3]})

        transformer = ColumnTransformer(transformers=[
            ('my_scaler', StandardScaler(), ['x'])
        ])

        transformer.fit(df)

        scaler = transformer.get_transformer('my_scaler')
        assert isinstance(scaler, StandardScaler)
        assert scaler.fitted

    def test_get_transformer_not_found(self):
        """Test error when getting non-existent transformer."""
        df = pd.DataFrame({'x': [1, 2, 3]})

        transformer = ColumnTransformer(transformers=[
            ('exists', StandardScaler(), ['x'])
        ])

        transformer.fit(df)

        with pytest.raises(ValueError, match="not found"):
            transformer.get_transformer('doesnt_exist')

    def test_get_column_mapping(self):
        """Test getting column mapping."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        transformer = ColumnTransformer(transformers=[
            ('ab', StandardScaler(), ['a', 'b']),
            ('c', MinMaxScaler(), ['c'])
        ])

        transformer.fit(df)

        mapping = transformer.get_column_mapping()

        assert mapping['ab'] == ['a', 'b']
        assert mapping['c'] == ['c']

    def test_get_feature_names_out(self):
        """Test getting output feature names."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })

        transformer = ColumnTransformer(
            transformers=[('scale', StandardScaler(), ['x'])],
            remainder='passthrough'
        )

        transformer.fit(df)

        names = transformer.get_feature_names_out()

        assert 'x' in names
        assert 'y' in names


class TestMakeColumnTransformer:
    """Test make_column_transformer convenience function."""

    def test_make_column_transformer(self):
        """Test make_column_transformer function."""
        transformer = make_column_transformer(
            ('num', StandardScaler(), NumericSelector()),
            ('cat', LabelEncoder(column='cat'), CategoricalSelector()),
            remainder='drop'
        )

        assert isinstance(transformer, ColumnTransformer)
        assert len(transformer.transformers) == 2
        assert transformer.remainder == 'drop'


class TestColumnTransformerEdgeCases:
    """Test edge cases for ColumnTransformer."""

    def test_empty_column_selection(self):
        """Test when selector returns no columns."""
        df = pd.DataFrame({
            'a': ['text', 'more', 'text']
        })

        # NumericSelector should find no columns
        transformer = ColumnTransformer(transformers=[
            ('numeric', StandardScaler(), NumericSelector())
        ])

        transformer.fit(df)
        result = transformer.transform(df)

        # Should return empty result or just remainder
        assert isinstance(result, pd.DataFrame)

    def test_single_column_transformation(self):
        """Test transforming single column."""
        df = pd.DataFrame({'x': [1, 2, 3]})

        transformer = ColumnTransformer(transformers=[
            ('scale', StandardScaler(), ['x'])
        ])

        result = transformer.fit_transform(df)

        assert 'x' in result.columns
        assert abs(result['x'].mean()) < 0.01

    def test_preserve_index(self):
        """Test that DataFrame index is preserved."""
        df = pd.DataFrame(
            {'x': [1, 2, 3]},
            index=['a', 'b', 'c']
        )

        transformer = ColumnTransformer(transformers=[
            ('scale', StandardScaler(), ['x'])
        ])

        result = transformer.fit_transform(df)

        assert list(result.index) == ['a', 'b', 'c']
