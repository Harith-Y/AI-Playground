"""
Tests for MeanImputer and MedianImputer classes.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer


class TestMeanImputerInitialization:
    """Test MeanImputer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        imputer = MeanImputer()
        assert imputer.params["columns"] is None
        assert not imputer.fitted
        assert imputer.means == {}

    def test_custom_columns(self):
        """Test initialization with custom columns."""
        imputer = MeanImputer(columns=["age", "salary"])
        assert imputer.params["columns"] == ["age", "salary"]


class TestMeanImputerFit:
    """Test MeanImputer fit method."""

    def test_fit_basic(self):
        """Test basic fitting."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = MeanImputer()
        imputer.fit(df)

        assert imputer.fitted
        assert 'x' in imputer.means
        assert imputer.means['x'] == 3.0  # (1+2+4+5)/4

    def test_fit_multiple_columns(self):
        """Test fitting with multiple columns."""
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [10.0, 20.0, 30.0],
            'z': [100.0, 200.0, 300.0]
        })
        imputer = MeanImputer()
        imputer.fit(df)

        assert imputer.means['x'] == 2.0
        assert imputer.means['y'] == 20.0
        assert imputer.means['z'] == 200.0

    def test_fit_specific_columns(self):
        """Test fitting only specific columns."""
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [10.0, 20.0, 30.0],
            'ignore': [100.0, 200.0, 300.0]
        })
        imputer = MeanImputer(columns=['x', 'y'])
        imputer.fit(df)

        assert 'x' in imputer.means
        assert 'y' in imputer.means
        assert 'ignore' not in imputer.means

    def test_fit_auto_detect_numeric(self):
        """Test auto-detection of numeric columns."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0],
            'text': ['a', 'b', 'c'],
            'numeric2': [10.0, 20.0, 30.0]
        })
        imputer = MeanImputer()
        imputer.fit(df)

        assert 'numeric' in imputer.means
        assert 'numeric2' in imputer.means
        assert 'text' not in imputer.means

    def test_fit_with_all_nan(self):
        """Test fit with column that has all NaN."""
        df = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})
        imputer = MeanImputer()
        imputer.fit(df)

        # Mean of all NaN is NaN
        assert pd.isna(imputer.means['x'])

    def test_fit_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        imputer = MeanImputer()
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            imputer.fit(np.array([[1, 2], [3, 4]]))


class TestMeanImputerTransform:
    """Test MeanImputer transform method."""

    def test_transform_basic(self):
        """Test basic transformation."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0, np.nan]})
        imputer = MeanImputer()
        imputer.fit(df)
        result = imputer.transform(df)

        # Mean is 2.33... (1+2+4)/3
        mean_val = (1.0 + 2.0 + 4.0) / 3.0
        assert result['x'].isna().sum() == 0
        assert result['x'].iloc[2] == mean_val
        assert result['x'].iloc[4] == mean_val

    def test_transform_multiple_columns(self):
        """Test transform with multiple columns."""
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0],
            'y': [10.0, 20.0, np.nan]
        })
        imputer = MeanImputer()
        imputer.fit(df)
        result = imputer.transform(df)

        assert result['x'].iloc[1] == 2.0  # (1+3)/2
        assert result['y'].iloc[2] == 15.0  # (10+20)/2

    def test_transform_preserves_non_nan(self):
        """Test that transform preserves non-NaN values."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0]})
        imputer = MeanImputer()
        imputer.fit(df)
        result = imputer.transform(df)

        assert result['x'].iloc[0] == 1.0
        assert result['x'].iloc[1] == 2.0
        assert result['x'].iloc[3] == 4.0

    def test_transform_different_data(self):
        """Test transform on different data."""
        train = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        test = pd.DataFrame({'x': [np.nan, np.nan]})

        imputer = MeanImputer()
        imputer.fit(train)
        result = imputer.transform(test)

        # Should use training mean (2.0)
        assert all(result['x'] == 2.0)

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan]})
        imputer = MeanImputer()

        with pytest.raises(RuntimeError, match="must be fitted"):
            imputer.transform(df)

    def test_transform_error_not_dataframe(self):
        """Test error when transform input is not DataFrame."""
        imputer = MeanImputer()
        imputer.fit(pd.DataFrame({'x': [1.0, 2.0, 3.0]}))

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            imputer.transform(np.array([[1, 2], [3, 4]]))


class TestMedianImputerInitialization:
    """Test MedianImputer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        imputer = MedianImputer()
        assert imputer.params["columns"] is None
        assert not imputer.fitted
        assert imputer.medians == {}

    def test_custom_columns(self):
        """Test initialization with custom columns."""
        imputer = MedianImputer(columns=["age", "income"])
        assert imputer.params["columns"] == ["age", "income"]


class TestMedianImputerFit:
    """Test MedianImputer fit method."""

    def test_fit_basic(self):
        """Test basic fitting."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = MedianImputer()
        imputer.fit(df)

        assert imputer.fitted
        assert 'x' in imputer.medians
        assert imputer.medians['x'] == 3.0  # median of [1,2,4,5]

    def test_fit_calculates_correct_median(self):
        """Test that fit calculates correct median."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 100.0]})
        imputer = MedianImputer()
        imputer.fit(df)

        # Median is 3.0 (not affected by outlier 100)
        assert imputer.medians['x'] == 3.0

    def test_fit_even_count(self):
        """Test median calculation with even number of values."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0]})
        imputer = MedianImputer()
        imputer.fit(df)

        # Median of even count is average of middle two
        assert imputer.medians['x'] == 2.5

    def test_fit_multiple_columns(self):
        """Test fitting with multiple columns."""
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [10.0, 20.0, 30.0]
        })
        imputer = MedianImputer()
        imputer.fit(df)

        assert imputer.medians['x'] == 2.0
        assert imputer.medians['y'] == 20.0

    def test_fit_auto_detect_numeric(self):
        """Test auto-detection of numeric columns."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0],
            'text': ['a', 'b', 'c'],
            'numeric2': [10.0, 20.0, 30.0]
        })
        imputer = MedianImputer()
        imputer.fit(df)

        assert 'numeric' in imputer.medians
        assert 'numeric2' in imputer.medians
        assert 'text' not in imputer.medians

    def test_fit_with_all_nan(self):
        """Test fit with column that has all NaN."""
        df = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})
        imputer = MedianImputer()
        imputer.fit(df)

        # Median of all NaN is NaN
        assert pd.isna(imputer.medians['x'])

    def test_fit_error_not_dataframe(self):
        """Test error when input is not DataFrame."""
        imputer = MedianImputer()
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            imputer.fit(np.array([[1, 2], [3, 4]]))


class TestMedianImputerTransform:
    """Test MedianImputer transform method."""

    def test_transform_basic(self):
        """Test basic transformation."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0, np.nan]})
        imputer = MedianImputer()
        imputer.fit(df)
        result = imputer.transform(df)

        # Median is 2.0 from [1,2,4]
        assert result['x'].isna().sum() == 0
        assert result['x'].iloc[2] == 2.0
        assert result['x'].iloc[4] == 2.0

    def test_transform_with_outliers(self):
        """Test that median imputation is robust to outliers."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, np.nan, 1000.0]})
        imputer = MedianImputer()
        imputer.fit(df)
        result = imputer.transform(df)

        # Median is 2.5 (not affected by 1000 outlier)
        assert result['x'].iloc[3] == 2.5

    def test_transform_multiple_columns(self):
        """Test transform with multiple columns."""
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0],
            'y': [10.0, 20.0, np.nan]
        })
        imputer = MedianImputer()
        imputer.fit(df)
        result = imputer.transform(df)

        assert result['x'].iloc[1] == 2.0
        assert result['y'].iloc[2] == 15.0

    def test_transform_preserves_non_nan(self):
        """Test that transform preserves non-NaN values."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0]})
        imputer = MedianImputer()
        imputer.fit(df)
        result = imputer.transform(df)

        assert result['x'].iloc[0] == 1.0
        assert result['x'].iloc[1] == 2.0
        assert result['x'].iloc[3] == 4.0

    def test_transform_different_data(self):
        """Test transform on different data."""
        train = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        test = pd.DataFrame({'x': [np.nan, np.nan]})

        imputer = MedianImputer()
        imputer.fit(train)
        result = imputer.transform(test)

        # Should use training median (2.0)
        assert all(result['x'] == 2.0)

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan]})
        imputer = MedianImputer()

        with pytest.raises(RuntimeError, match="must be fitted"):
            imputer.transform(df)

    def test_transform_error_not_dataframe(self):
        """Test error when transform input is not DataFrame."""
        imputer = MedianImputer()
        imputer.fit(pd.DataFrame({'x': [1.0, 2.0, 3.0]}))

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            imputer.transform(np.array([[1, 2], [3, 4]]))


class TestImputersPractical:
    """Test practical use cases for both imputers."""

    def test_mean_vs_median_with_outliers(self):
        """Test that median is more robust to outliers than mean."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, np.nan, 1000.0]})

        mean_imputer = MeanImputer()
        mean_imputer.fit(df)
        mean_result = mean_imputer.transform(df)

        median_imputer = MedianImputer()
        median_imputer.fit(df)
        median_result = median_imputer.transform(df)

        # Mean is heavily affected by outlier
        assert mean_result['x'].iloc[3] > 100

        # Median is not affected
        assert median_result['x'].iloc[3] == 2.5

    def test_train_test_split(self):
        """Test imputers with train/test split."""
        train = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
        test = pd.DataFrame({'x': [np.nan, np.nan, np.nan]})

        mean_imputer = MeanImputer()
        mean_imputer.fit(train)
        test_result = mean_imputer.transform(test)

        # All NaN in test should be filled with training mean (3.0)
        assert all(test_result['x'] == 3.0)

    def test_no_missing_values(self):
        """Test imputers when there are no missing values."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})

        mean_imputer = MeanImputer()
        mean_result = mean_imputer.fit_transform(df)

        median_imputer = MedianImputer()
        median_result = median_imputer.fit_transform(df)

        # Data should be unchanged
        pd.testing.assert_frame_equal(mean_result, df)
        pd.testing.assert_frame_equal(median_result, df)

    def test_partial_missing_values(self):
        """Test imputers with partial missing values."""
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0],
            'y': [10.0, 20.0, 30.0],  # No missing
            'z': [np.nan, np.nan, 100.0]  # Mostly missing
        })

        imputer = MedianImputer()
        result = imputer.fit_transform(df)

        assert result['x'].isna().sum() == 0
        assert result['y'].isna().sum() == 0
        assert result['z'].isna().sum() == 0
        assert result['y'].iloc[0] == 10.0  # Unchanged
        assert result['z'].iloc[0] == 100.0  # Filled with median (100)

    def test_fit_transform_consistency(self):
        """Test that fit_transform gives same result as fit then transform."""
        df = pd.DataFrame({'x': [1.0, 2.0, np.nan, 4.0]})

        # Method 1: fit_transform
        imputer1 = MeanImputer()
        result1 = imputer1.fit_transform(df)

        # Method 2: fit then transform
        imputer2 = MeanImputer()
        imputer2.fit(df)
        result2 = imputer2.transform(df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_integer_data(self):
        """Test imputers with integer data."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]}, dtype='Int64')
        df.loc[2, 'x'] = pd.NA

        imputer = MeanImputer()
        result = imputer.fit_transform(df)

        # Mean of [1,2,4,5] is 3.0
        assert result['x'].iloc[2] == 3.0

    def test_preserves_other_columns(self):
        """Test that imputers preserve non-imputed columns."""
        df = pd.DataFrame({
            'impute_me': [1.0, np.nan, 3.0],
            'keep_me': [10.0, 20.0, 30.0],
            'text': ['a', 'b', 'c']
        })

        imputer = MeanImputer(columns=['impute_me'])
        result = imputer.fit_transform(df)

        # Non-imputed columns should be unchanged
        pd.testing.assert_series_equal(result['keep_me'], df['keep_me'])
        pd.testing.assert_series_equal(result['text'], df['text'])

        # Only impute_me should be changed
        assert result['impute_me'].isna().sum() == 0
        assert df['impute_me'].isna().sum() == 1
