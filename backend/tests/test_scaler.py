"""
Tests for StandardScaler and MinMaxScaler classes.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler


class TestStandardScalerInitialization:
    """Test StandardScaler initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scaler = StandardScaler()
        assert scaler.params["columns"] is None
        assert scaler.params["with_mean"] == True
        assert scaler.params["with_std"] == True
        assert not scaler.fitted

    def test_custom_columns(self):
        """Test initialization with custom columns."""
        scaler = StandardScaler(columns=["age", "salary"])
        assert scaler.params["columns"] == ["age", "salary"]

    def test_with_mean_only(self):
        """Test initialization with only mean centering."""
        scaler = StandardScaler(with_mean=True, with_std=False)
        assert scaler.params["with_mean"] == True
        assert scaler.params["with_std"] == False

    def test_with_std_only(self):
        """Test initialization with only std scaling."""
        scaler = StandardScaler(with_mean=False, with_std=True)
        assert scaler.params["with_mean"] == False
        assert scaler.params["with_std"] == True

    def test_error_both_false(self):
        """Test error when both with_mean and with_std are False."""
        with pytest.raises(ValueError, match="At least one of with_mean or with_std must be True"):
            StandardScaler(with_mean=False, with_std=False)


class TestStandardScalerFit:
    """Test StandardScaler fit method."""

    def test_fit_basic(self):
        """Test basic fitting."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        scaler = StandardScaler()
        scaler.fit(df)

        assert scaler.fitted
        assert 'x' in scaler.means_
        assert 'x' in scaler.stds_

    def test_fit_calculates_correct_statistics(self):
        """Test that fit calculates correct mean and std."""
        df = pd.DataFrame({'x': [10, 20, 30, 40, 50]})
        scaler = StandardScaler()
        scaler.fit(df)

        assert scaler.means_['x'] == 30.0
        assert abs(scaler.stds_['x'] - df['x'].std()) < 0.001

    def test_fit_multiple_columns(self):
        """Test fitting with multiple columns."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30],
            'z': [100, 200, 300]
        })
        scaler = StandardScaler()
        scaler.fit(df)

        assert set(scaler.means_.keys()) == {'x', 'y', 'z'}
        assert set(scaler.stds_.keys()) == {'x', 'y', 'z'}

    def test_fit_specific_columns(self):
        """Test fitting only specific columns."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30],
            'ignore': [100, 200, 300]
        })
        scaler = StandardScaler(columns=['x', 'y'])
        scaler.fit(df)

        assert set(scaler.means_.keys()) == {'x', 'y'}
        assert 'ignore' not in scaler.means_

    def test_fit_with_mean_only(self):
        """Test fit with only mean calculation."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        scaler = StandardScaler(with_mean=True, with_std=False)
        scaler.fit(df)

        assert 'x' in scaler.means_
        assert 'x' not in scaler.stds_

    def test_fit_with_std_only(self):
        """Test fit with only std calculation."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        scaler = StandardScaler(with_mean=False, with_std=True)
        scaler.fit(df)

        assert 'x' not in scaler.means_
        assert 'x' in scaler.stds_

    def test_fit_constant_column(self):
        """Test fit with constant column (std=0)."""
        df = pd.DataFrame({'x': [5, 5, 5, 5]})
        scaler = StandardScaler()
        scaler.fit(df)

        assert scaler.stds_['x'] == 1.0  # Should set to 1.0 to avoid division by zero

    def test_fit_mixed_types(self):
        """Test fit auto-detects numeric columns."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'text': ['a', 'b', 'c'],
            'numeric2': [10, 20, 30]
        })
        scaler = StandardScaler()
        scaler.fit(df)

        assert 'numeric' in scaler.means_
        assert 'numeric2' in scaler.means_
        assert 'text' not in scaler.means_

    def test_fit_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        scaler = StandardScaler()
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            scaler.fit(np.array([[1, 2], [3, 4]]))

    def test_fit_error_column_not_found(self):
        """Test error when specified column doesn't exist."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        scaler = StandardScaler(columns=['x', 'nonexistent'])

        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            scaler.fit(df)


class TestStandardScalerTransform:
    """Test StandardScaler transform method."""

    def test_transform_basic(self):
        """Test basic transformation."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        scaler = StandardScaler()
        scaler.fit(df)
        result = scaler.transform(df)

        # After standardization, mean should be ~0 and std should be ~1
        assert abs(result['x'].mean()) < 0.001
        assert abs(result['x'].std() - 1.0) < 0.001

    def test_transform_preserves_shape(self):
        """Test that transform preserves DataFrame shape."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30]
        })
        scaler = StandardScaler()
        scaler.fit(df)
        result = scaler.transform(df)

        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        scaler = StandardScaler()

        with pytest.raises(RuntimeError, match="must be fitted"):
            scaler.transform(df)

    def test_transform_different_data(self):
        """Test transform on different data using fitted parameters."""
        train = pd.DataFrame({'x': [10, 20, 30]})
        test = pd.DataFrame({'x': [15, 25, 35]})

        scaler = StandardScaler()
        scaler.fit(train)
        result = scaler.transform(test)

        # Should use training mean and std
        expected_mean = train['x'].mean()
        expected_std = train['x'].std()
        expected = (test['x'] - expected_mean) / expected_std

        pd.testing.assert_series_equal(result['x'], expected, check_names=False)

    def test_transform_with_mean_only(self):
        """Test transform with only mean centering."""
        df = pd.DataFrame({'x': [10, 20, 30]})
        scaler = StandardScaler(with_mean=True, with_std=False)
        scaler.fit(df)
        result = scaler.transform(df)

        # Mean should be 0, but std should be unchanged
        assert abs(result['x'].mean()) < 0.001
        assert abs(result['x'].std() - df['x'].std()) < 0.001

    def test_transform_with_std_only(self):
        """Test transform with only std scaling."""
        df = pd.DataFrame({'x': [10, 20, 30]})
        scaler = StandardScaler(with_mean=False, with_std=True)
        scaler.fit(df)
        result = scaler.transform(df)

        # Std should be 1, but mean should be scaled, not centered at 0
        assert abs(result['x'].std() - 1.0) < 0.001
        assert abs(result['x'].mean()) > 0.1  # Not centered

    def test_transform_error_not_dataframe(self):
        """Test error when transform input is not DataFrame."""
        scaler = StandardScaler()
        scaler.fit(pd.DataFrame({'x': [1, 2, 3]}))

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            scaler.transform(np.array([[1, 2], [3, 4]]))

    def test_transform_error_column_missing(self):
        """Test error when column is missing in transform."""
        scaler = StandardScaler()
        scaler.fit(pd.DataFrame({'x': [1, 2, 3]}))

        with pytest.raises(ValueError, match="Column 'x' not found"):
            scaler.transform(pd.DataFrame({'y': [1, 2, 3]}))


class TestStandardScalerInverseTransform:
    """Test StandardScaler inverse_transform method."""

    def test_inverse_transform_basic(self):
        """Test basic inverse transformation."""
        df = pd.DataFrame({'x': [10, 20, 30, 40, 50]})
        scaler = StandardScaler()
        scaler.fit(df)

        transformed = scaler.transform(df)
        recovered = scaler.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df, check_dtype=False)

    def test_inverse_transform_round_trip(self):
        """Test transform -> inverse_transform round trip."""
        df = pd.DataFrame({
            'x': [1.5, 2.7, 3.2, 4.8],
            'y': [100.5, 200.3, 300.7, 400.1]
        })
        scaler = StandardScaler()
        scaler.fit(df)

        transformed = scaler.transform(df)
        recovered = scaler.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df, atol=0.001)

    def test_inverse_transform_with_mean_only(self):
        """Test inverse transform with only mean."""
        df = pd.DataFrame({'x': [10, 20, 30]})
        scaler = StandardScaler(with_mean=True, with_std=False)
        scaler.fit(df)

        transformed = scaler.transform(df)
        recovered = scaler.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df, check_dtype=False)

    def test_inverse_transform_with_std_only(self):
        """Test inverse transform with only std."""
        df = pd.DataFrame({'x': [10, 20, 30]})
        scaler = StandardScaler(with_mean=False, with_std=True)
        scaler.fit(df)

        transformed = scaler.transform(df)
        recovered = scaler.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df, atol=0.001)


class TestStandardScalerUtilities:
    """Test StandardScaler utility methods."""

    def test_get_statistics(self):
        """Test get_statistics method."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30]
        })
        scaler = StandardScaler()
        scaler.fit(df)

        stats = scaler.get_statistics()
        assert 'means' in stats
        assert 'stds' in stats
        assert stats['means']['x'] == 2.0
        assert stats['means']['y'] == 20.0

    def test_get_statistics_before_fit(self):
        """Test get_statistics before fit raises error."""
        scaler = StandardScaler()
        with pytest.raises(RuntimeError):
            scaler.get_statistics()


class TestMinMaxScalerInitialization:
    """Test MinMaxScaler initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scaler = MinMaxScaler()
        assert scaler.params["columns"] is None
        assert scaler.params["feature_range"] == (0, 1)
        assert not scaler.fitted

    def test_custom_feature_range(self):
        """Test initialization with custom feature range."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        assert scaler.params["feature_range"] == (-1, 1)

    def test_error_invalid_feature_range_length(self):
        """Test error with invalid feature_range length."""
        with pytest.raises(ValueError, match="must be a tuple"):
            MinMaxScaler(feature_range=(0, 1, 2))

    def test_error_invalid_feature_range_order(self):
        """Test error when feature_range min >= max."""
        with pytest.raises(ValueError, match="min must be less than max"):
            MinMaxScaler(feature_range=(1, 0))

        with pytest.raises(ValueError, match="min must be less than max"):
            MinMaxScaler(feature_range=(5, 5))


class TestMinMaxScalerFit:
    """Test MinMaxScaler fit method."""

    def test_fit_basic(self):
        """Test basic fitting."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        scaler = MinMaxScaler()
        scaler.fit(df)

        assert scaler.fitted
        assert scaler.mins_['x'] == 1
        assert scaler.maxs_['x'] == 5

    def test_fit_multiple_columns(self):
        """Test fitting with multiple columns."""
        df = pd.DataFrame({
            'x': [1, 5, 10],
            'y': [100, 200, 300]
        })
        scaler = MinMaxScaler()
        scaler.fit(df)

        assert scaler.mins_['x'] == 1
        assert scaler.maxs_['x'] == 10
        assert scaler.mins_['y'] == 100
        assert scaler.maxs_['y'] == 300

    def test_fit_constant_column(self):
        """Test fit with constant column."""
        df = pd.DataFrame({'x': [5, 5, 5, 5]})
        scaler = MinMaxScaler()
        scaler.fit(df)

        assert scaler.mins_['x'] == 5
        assert scaler.maxs_['x'] == 5

    def test_fit_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        scaler = MinMaxScaler()
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            scaler.fit(np.array([[1, 2], [3, 4]]))


class TestMinMaxScalerTransform:
    """Test MinMaxScaler transform method."""

    def test_transform_basic(self):
        """Test basic transformation to [0, 1]."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        scaler = MinMaxScaler()
        scaler.fit(df)
        result = scaler.transform(df)

        assert result['x'].min() == 0.0
        assert result['x'].max() == 1.0
        assert result['x'].iloc[2] == 0.5  # Middle value

    def test_transform_custom_range(self):
        """Test transformation to custom range."""
        df = pd.DataFrame({'x': [0, 50, 100]})
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df)
        result = scaler.transform(df)

        assert result['x'].min() == -1.0
        assert result['x'].max() == 1.0
        assert result['x'].iloc[1] == 0.0  # Middle value

    def test_transform_constant_column(self):
        """Test transform with constant column."""
        df = pd.DataFrame({'x': [5, 5, 5]})
        scaler = MinMaxScaler()
        scaler.fit(df)
        result = scaler.transform(df)

        # Constant values should map to middle of feature range
        assert all(result['x'] == 0.5)

    def test_transform_different_data(self):
        """Test transform on different data."""
        train = pd.DataFrame({'x': [0, 10]})
        test = pd.DataFrame({'x': [5, 15]})  # 15 is outside training range

        scaler = MinMaxScaler()
        scaler.fit(train)
        result = scaler.transform(test)

        assert result['x'].iloc[0] == 0.5  # 5 maps to 0.5
        assert result['x'].iloc[1] == 1.5  # 15 maps to 1.5 (outside [0,1])

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        scaler = MinMaxScaler()

        with pytest.raises(RuntimeError, match="must be fitted"):
            scaler.transform(df)


class TestMinMaxScalerInverseTransform:
    """Test MinMaxScaler inverse_transform method."""

    def test_inverse_transform_basic(self):
        """Test basic inverse transformation."""
        df = pd.DataFrame({'x': [10, 20, 30, 40, 50]})
        scaler = MinMaxScaler()
        scaler.fit(df)

        transformed = scaler.transform(df)
        recovered = scaler.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df, check_dtype=False)

    def test_inverse_transform_custom_range(self):
        """Test inverse transform with custom range."""
        df = pd.DataFrame({'x': [100, 200, 300]})
        scaler = MinMaxScaler(feature_range=(-10, 10))
        scaler.fit(df)

        transformed = scaler.transform(df)
        recovered = scaler.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df, atol=0.001)

    def test_inverse_transform_constant_column(self):
        """Test inverse transform with constant column."""
        df = pd.DataFrame({'x': [42, 42, 42]})
        scaler = MinMaxScaler()
        scaler.fit(df)

        transformed = scaler.transform(df)
        recovered = scaler.inverse_transform(transformed)

        pd.testing.assert_frame_equal(recovered, df, check_dtype=False)


class TestMinMaxScalerUtilities:
    """Test MinMaxScaler utility methods."""

    def test_get_data_range(self):
        """Test get_data_range method."""
        df = pd.DataFrame({
            'x': [1, 5, 10],
            'y': [100, 200, 300]
        })
        scaler = MinMaxScaler()
        scaler.fit(df)

        data_range = scaler.get_data_range()
        assert data_range['mins']['x'] == 1
        assert data_range['maxs']['x'] == 10
        assert data_range['mins']['y'] == 100
        assert data_range['maxs']['y'] == 300

    def test_get_data_range_before_fit(self):
        """Test get_data_range before fit raises error."""
        scaler = MinMaxScaler()
        with pytest.raises(RuntimeError):
            scaler.get_data_range()


class TestScalersPractical:
    """Test practical use cases for both scalers."""

    def test_standard_scaler_pipeline_usage(self):
        """Test StandardScaler in typical ML pipeline."""
        # Simulate train/test split
        train = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        test = pd.DataFrame({'feature': [2.5, 3.5]})

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

        # Both should use same statistics
        assert scaler.means_['feature'] == 3.0
        assert isinstance(test_scaled, pd.DataFrame)

    def test_minmax_scaler_pipeline_usage(self):
        """Test MinMaxScaler in typical ML pipeline."""
        train = pd.DataFrame({'feature': [0, 10, 20, 30, 40]})
        test = pd.DataFrame({'feature': [15, 25]})

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

        assert train_scaled['feature'].min() == 0.0
        assert train_scaled['feature'].max() == 1.0
        assert 0 < test_scaled['feature'].iloc[0] < 1

    def test_scaler_with_missing_values(self):
        """Test scaler behavior with NaN values."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0, 5.0]})
        scaler = StandardScaler()
        scaler.fit(df)

        # Mean and std should ignore NaN
        assert not np.isnan(scaler.means_['x'])
        assert not np.isnan(scaler.stds_['x'])
