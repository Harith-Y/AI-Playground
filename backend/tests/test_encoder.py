"""
Tests for OneHotEncoder and LabelEncoder classes.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.encoder import OneHotEncoder, LabelEncoder


class TestOneHotEncoderInitialization:
    """Test OneHotEncoder initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        encoder = OneHotEncoder()
        assert encoder.params["columns"] is None
        assert encoder.params["drop_first"] == False
        assert encoder.params["handle_unknown"] == "error"
        assert not encoder.fitted

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        encoder = OneHotEncoder(
            columns=["color", "size"],
            drop_first=True,
            handle_unknown="ignore"
        )
        assert encoder.params["columns"] == ["color", "size"]
        assert encoder.params["drop_first"] == True
        assert encoder.params["handle_unknown"] == "ignore"

    def test_error_invalid_handle_unknown(self):
        """Test error with invalid handle_unknown parameter."""
        with pytest.raises(ValueError, match="handle_unknown must be 'error' or 'ignore'"):
            OneHotEncoder(handle_unknown="invalid")


class TestOneHotEncoderFit:
    """Test OneHotEncoder fit method."""

    def test_fit_basic(self):
        """Test basic fitting."""
        df = pd.DataFrame({'color': ['red', 'blue', 'red', 'green']})
        encoder = OneHotEncoder()
        encoder.fit(df)

        assert encoder.fitted
        assert 'color' in encoder.categories_
        assert set(encoder.categories_['color']) == {'blue', 'green', 'red'}

    def test_fit_learns_categories(self):
        """Test that fit learns unique categories."""
        df = pd.DataFrame({'size': ['S', 'M', 'L', 'M', 'S']})
        encoder = OneHotEncoder()
        encoder.fit(df)

        assert encoder.categories_['size'] == ['L', 'M', 'S']  # Sorted

    def test_fit_multiple_columns(self):
        """Test fitting with multiple columns."""
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red'],
            'size': ['S', 'M', 'L'],
            'numeric': [1, 2, 3]
        })
        encoder = OneHotEncoder(columns=['color', 'size'])
        encoder.fit(df)

        assert set(encoder.categories_.keys()) == {'color', 'size'}
        assert 'numeric' not in encoder.categories_

    def test_fit_auto_detect_categorical(self):
        """Test auto-detection of categorical columns."""
        df = pd.DataFrame({
            'categorical': ['a', 'b', 'c'],
            'numeric': [1, 2, 3],
            'mixed': ['x', 'y', 'z']
        })
        encoder = OneHotEncoder()  # No columns specified
        encoder.fit(df)

        # Should detect object dtypes
        assert 'categorical' in encoder.categories_
        assert 'mixed' in encoder.categories_
        assert 'numeric' not in encoder.categories_

    def test_fit_with_nan(self):
        """Test fit ignores NaN values."""
        df = pd.DataFrame({'color': ['red', 'blue', np.nan, 'red']})
        encoder = OneHotEncoder()
        encoder.fit(df)

        assert encoder.categories_['color'] == ['blue', 'red']  # NaN excluded

    def test_fit_generates_feature_names(self):
        """Test that fit generates correct feature names."""
        df = pd.DataFrame({'color': ['red', 'blue', 'green']})
        encoder = OneHotEncoder()
        encoder.fit(df)

        assert encoder.feature_names_ == ['color_blue', 'color_green', 'color_red']

    def test_fit_drop_first_feature_names(self):
        """Test feature names when drop_first=True."""
        df = pd.DataFrame({'color': ['red', 'blue', 'green']})
        encoder = OneHotEncoder(drop_first=True)
        encoder.fit(df)

        # Should exclude first category (blue)
        assert encoder.feature_names_ == ['color_green', 'color_red']

    def test_fit_error_column_not_found(self):
        """Test error when specified column doesn't exist."""
        df = pd.DataFrame({'color': ['red', 'blue']})
        encoder = OneHotEncoder(columns=['color', 'nonexistent'])

        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            encoder.fit(df)

    def test_fit_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        encoder = OneHotEncoder()
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            encoder.fit(np.array([[1, 2], [3, 4]]))


class TestOneHotEncoderTransform:
    """Test OneHotEncoder transform method."""

    def test_transform_basic(self):
        """Test basic transformation."""
        df = pd.DataFrame({'color': ['red', 'blue', 'red']})
        encoder = OneHotEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        assert 'color_red' in result.columns
        assert 'color_blue' in result.columns
        assert 'color' not in result.columns  # Original dropped

        # Check values
        assert list(result['color_red']) == [1, 0, 1]
        assert list(result['color_blue']) == [0, 1, 0]

    def test_transform_multiple_columns(self):
        """Test transform with multiple columns."""
        df = pd.DataFrame({
            'color': ['red', 'blue'],
            'size': ['S', 'L']
        })
        encoder = OneHotEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        assert 'color_red' in result.columns
        assert 'color_blue' in result.columns
        assert 'size_S' in result.columns
        assert 'size_L' in result.columns
        assert 'color' not in result.columns
        assert 'size' not in result.columns

    def test_transform_drop_first(self):
        """Test transform with drop_first=True."""
        df = pd.DataFrame({'color': ['red', 'blue', 'green']})
        encoder = OneHotEncoder(drop_first=True)
        encoder.fit(df)
        result = encoder.transform(df)

        # First category (blue) should be dropped
        assert 'color_blue' not in result.columns
        assert 'color_green' in result.columns
        assert 'color_red' in result.columns

    def test_transform_preserves_other_columns(self):
        """Test that transform preserves non-encoded columns."""
        df = pd.DataFrame({
            'color': ['red', 'blue'],
            'price': [100, 200],
            'quantity': [5, 10]
        })
        encoder = OneHotEncoder(columns=['color'])
        encoder.fit(df)
        result = encoder.transform(df)

        assert 'price' in result.columns
        assert 'quantity' in result.columns
        assert list(result['price']) == [100, 200]
        assert list(result['quantity']) == [5, 10]

    def test_transform_unknown_category_error(self):
        """Test error when unknown category encountered with handle_unknown='error'."""
        train = pd.DataFrame({'color': ['red', 'blue']})
        test = pd.DataFrame({'color': ['red', 'green']})  # green is unknown

        encoder = OneHotEncoder(handle_unknown="error")
        encoder.fit(train)

        with pytest.raises(ValueError, match="Found unknown categories"):
            encoder.transform(test)

    def test_transform_unknown_category_ignore(self):
        """Test unknown category handling with handle_unknown='ignore'."""
        train = pd.DataFrame({'color': ['red', 'blue']})
        test = pd.DataFrame({'color': ['red', 'green', 'blue']})

        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(train)
        result = encoder.transform(test)

        # green should be encoded as all zeros
        assert list(result['color_red']) == [1, 0, 0]
        assert list(result['color_blue']) == [0, 0, 1]

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'color': ['red', 'blue']})
        encoder = OneHotEncoder()

        with pytest.raises(RuntimeError, match="must be fitted"):
            encoder.transform(df)

    def test_transform_error_column_missing(self):
        """Test error when column is missing in transform."""
        encoder = OneHotEncoder()
        encoder.fit(pd.DataFrame({'color': ['red', 'blue']}))

        with pytest.raises(ValueError, match="Column 'color' not found"):
            encoder.transform(pd.DataFrame({'size': ['S', 'M']}))


class TestOneHotEncoderUtilities:
    """Test OneHotEncoder utility methods."""

    def test_get_feature_names(self):
        """Test get_feature_names method."""
        df = pd.DataFrame({
            'color': ['red', 'blue'],
            'size': ['S', 'L']
        })
        encoder = OneHotEncoder()
        encoder.fit(df)

        feature_names = encoder.get_feature_names()
        assert 'color_blue' in feature_names
        assert 'color_red' in feature_names
        assert 'size_L' in feature_names
        assert 'size_S' in feature_names

    def test_get_feature_names_before_fit(self):
        """Test get_feature_names before fit raises error."""
        encoder = OneHotEncoder()
        with pytest.raises(RuntimeError):
            encoder.get_feature_names()


class TestLabelEncoderInitialization:
    """Test LabelEncoder initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        encoder = LabelEncoder()
        assert encoder.params["columns"] is None
        assert encoder.params["handle_unknown"] == "error"
        assert encoder.params["unknown_value"] == -1
        assert not encoder.fitted

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        encoder = LabelEncoder(
            columns=["category"],
            handle_unknown="use_encoded_value",
            unknown_value=-99
        )
        assert encoder.params["columns"] == ["category"]
        assert encoder.params["handle_unknown"] == "use_encoded_value"
        assert encoder.params["unknown_value"] == -99

    def test_error_invalid_handle_unknown(self):
        """Test error with invalid handle_unknown parameter."""
        with pytest.raises(ValueError, match="handle_unknown must be 'error' or 'use_encoded_value'"):
            LabelEncoder(handle_unknown="invalid")


class TestLabelEncoderFit:
    """Test LabelEncoder fit method."""

    def test_fit_basic(self):
        """Test basic fitting."""
        df = pd.DataFrame({'color': ['red', 'blue', 'red', 'green']})
        encoder = LabelEncoder()
        encoder.fit(df)

        assert encoder.fitted
        assert 'color' in encoder.label_mappings_
        assert 'color' in encoder.inverse_mappings_

    def test_fit_creates_correct_mappings(self):
        """Test that fit creates correct label mappings."""
        df = pd.DataFrame({'color': ['red', 'blue', 'green']})
        encoder = LabelEncoder()
        encoder.fit(df)

        # Categories should be sorted: blue=0, green=1, red=2
        assert encoder.label_mappings_['color'] == {'blue': 0, 'green': 1, 'red': 2}
        assert encoder.inverse_mappings_['color'] == {0: 'blue', 1: 'green', 2: 'red'}

    def test_fit_multiple_columns(self):
        """Test fitting with multiple columns."""
        df = pd.DataFrame({
            'color': ['red', 'blue'],
            'size': ['S', 'L'],
            'numeric': [1, 2]
        })
        encoder = LabelEncoder(columns=['color', 'size'])
        encoder.fit(df)

        assert set(encoder.label_mappings_.keys()) == {'color', 'size'}
        assert 'numeric' not in encoder.label_mappings_

    def test_fit_with_nan(self):
        """Test fit ignores NaN values."""
        df = pd.DataFrame({'color': ['red', 'blue', np.nan, 'red']})
        encoder = LabelEncoder()
        encoder.fit(df)

        assert set(encoder.label_mappings_['color'].keys()) == {'blue', 'red'}

    def test_fit_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        encoder = LabelEncoder()
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            encoder.fit(np.array([[1, 2], [3, 4]]))


class TestLabelEncoderTransform:
    """Test LabelEncoder transform method."""

    def test_transform_basic(self):
        """Test basic transformation."""
        df = pd.DataFrame({'color': ['red', 'blue', 'red']})
        encoder = LabelEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        # blue=0, red=1
        assert list(result['color']) == [1, 0, 1]

    def test_transform_multiple_columns(self):
        """Test transform with multiple columns."""
        df = pd.DataFrame({
            'color': ['red', 'blue'],
            'size': ['S', 'L']
        })
        encoder = LabelEncoder()
        encoder.fit(df)
        result = encoder.transform(df)

        assert 'color' in result.columns
        assert 'size' in result.columns
        assert result['color'].dtype in [np.int32, np.int64]
        assert result['size'].dtype in [np.int32, np.int64]

    def test_transform_preserves_other_columns(self):
        """Test that transform preserves non-encoded columns."""
        df = pd.DataFrame({
            'color': ['red', 'blue'],
            'price': [100, 200]
        })
        encoder = LabelEncoder(columns=['color'])
        encoder.fit(df)
        result = encoder.transform(df)

        assert 'price' in result.columns
        assert list(result['price']) == [100, 200]

    def test_transform_unknown_category_error(self):
        """Test error when unknown category encountered with handle_unknown='error'."""
        train = pd.DataFrame({'color': ['red', 'blue']})
        test = pd.DataFrame({'color': ['red', 'green']})

        encoder = LabelEncoder(handle_unknown="error")
        encoder.fit(train)

        with pytest.raises(ValueError, match="Found unknown categories"):
            encoder.transform(test)

    def test_transform_unknown_category_use_value(self):
        """Test unknown category handling with handle_unknown='use_encoded_value'."""
        train = pd.DataFrame({'color': ['red', 'blue']})
        test = pd.DataFrame({'color': ['red', 'green', 'blue']})

        encoder = LabelEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(train)
        result = encoder.transform(test)

        # blue=0, red=1, green=-1 (unknown)
        assert list(result['color']) == [1, -1, 0]

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'color': ['red', 'blue']})
        encoder = LabelEncoder()

        with pytest.raises(RuntimeError, match="must be fitted"):
            encoder.transform(df)


class TestLabelEncoderInverseTransform:
    """Test LabelEncoder inverse_transform method."""

    def test_inverse_transform_basic(self):
        """Test basic inverse transformation."""
        df = pd.DataFrame({'color': ['red', 'blue', 'red']})
        encoder = LabelEncoder()
        encoder.fit(df)

        transformed = encoder.transform(df)
        recovered = encoder.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df)

    def test_inverse_transform_round_trip(self):
        """Test transform -> inverse_transform round trip."""
        df = pd.DataFrame({
            'color': ['red', 'blue', 'green'],
            'size': ['S', 'M', 'L']
        })
        encoder = LabelEncoder()
        encoder.fit(df)

        transformed = encoder.transform(df)
        recovered = encoder.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df)

    def test_inverse_transform_with_unknown_value(self):
        """Test inverse transform with unknown_value becomes NaN."""
        train = pd.DataFrame({'color': ['red', 'blue']})
        test = pd.DataFrame({'color': ['red', 'green']})

        encoder = LabelEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(train)
        transformed = encoder.transform(test)
        recovered = encoder.inverse_transform(transformed)

        assert recovered['color'].iloc[0] == 'red'
        assert pd.isna(recovered['color'].iloc[1])  # green was encoded as -1, becomes NaN

    def test_inverse_transform_error_unknown_label(self):
        """Test error when unknown label encountered."""
        df = pd.DataFrame({'color': ['red', 'blue']})
        encoder = LabelEncoder()
        encoder.fit(df)

        # Manually create invalid data with label 99
        invalid = pd.DataFrame({'color': [0, 99]})

        with pytest.raises(ValueError, match="Found unknown labels"):
            encoder.inverse_transform(invalid)

    def test_inverse_transform_requires_fit(self):
        """Test that inverse_transform requires fit."""
        df = pd.DataFrame({'color': [0, 1, 0]})
        encoder = LabelEncoder()

        with pytest.raises(RuntimeError, match="must be fitted"):
            encoder.inverse_transform(df)


class TestLabelEncoderUtilities:
    """Test LabelEncoder utility methods."""

    def test_get_label_mappings(self):
        """Test get_label_mappings method."""
        df = pd.DataFrame({
            'color': ['red', 'blue'],
            'size': ['S', 'L']
        })
        encoder = LabelEncoder()
        encoder.fit(df)

        mappings = encoder.get_label_mappings()
        assert 'color' in mappings
        assert 'size' in mappings
        assert mappings['color']['blue'] == 0
        assert mappings['color']['red'] == 1

    def test_get_label_mappings_before_fit(self):
        """Test get_label_mappings before fit raises error."""
        encoder = LabelEncoder()
        with pytest.raises(RuntimeError):
            encoder.get_label_mappings()

    def test_get_label_mappings_returns_copy(self):
        """Test that get_label_mappings returns a copy."""
        df = pd.DataFrame({'color': ['red', 'blue']})
        encoder = LabelEncoder()
        encoder.fit(df)

        mappings = encoder.get_label_mappings()
        mappings['color']['blue'] = 999

        # Original should be unchanged
        assert encoder.label_mappings_['color']['blue'] == 0


class TestEncodersPractical:
    """Test practical use cases for both encoders."""

    def test_onehot_train_test_split(self):
        """Test OneHotEncoder with train/test split."""
        train = pd.DataFrame({'color': ['red', 'blue', 'green']})
        test = pd.DataFrame({'color': ['red', 'blue']})

        encoder = OneHotEncoder()
        train_encoded = encoder.fit_transform(train)
        test_encoded = encoder.transform(test)

        # Test should have same columns as train
        assert set(test_encoded.columns) == set(train_encoded.columns)

    def test_label_train_test_split(self):
        """Test LabelEncoder with train/test split."""
        train = pd.DataFrame({'color': ['red', 'blue', 'green']})
        test = pd.DataFrame({'color': ['red', 'blue']})

        encoder = LabelEncoder()
        encoder.fit(train)
        test_encoded = encoder.transform(test)

        # Should use same mapping learned from train
        assert list(test_encoded['color']) == [2, 0]  # blue=0, green=1, red=2

    def test_onehot_with_many_categories(self):
        """Test OneHotEncoder with many categories."""
        df = pd.DataFrame({'category': [f'cat_{i}' for i in range(100)]})
        encoder = OneHotEncoder()
        result = encoder.fit_transform(df)

        # Should create 100 columns
        assert len(result.columns) == 100

    def test_label_consistent_encoding(self):
        """Test that LabelEncoder produces consistent encodings."""
        df1 = pd.DataFrame({'color': ['red', 'blue', 'green']})
        df2 = pd.DataFrame({'color': ['green', 'red', 'blue']})

        encoder = LabelEncoder()
        encoder.fit(df1)

        result1 = encoder.transform(df1)
        result2 = encoder.transform(df2)

        # Same categories should get same labels
        assert result1['color'].iloc[0] == result2['color'].iloc[1]  # red
        assert result1['color'].iloc[1] == result2['color'].iloc[2]  # blue
        assert result1['color'].iloc[2] == result2['color'].iloc[0]  # green
