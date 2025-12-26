"""
Tests for ZScoreOutlierDetector class.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.cleaner import ZScoreOutlierDetector


class TestZScoreOutlierDetectorBasic:
    """Test basic functionality of ZScoreOutlierDetector."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        detector = ZScoreOutlierDetector()
        assert detector.threshold == 3.0
        assert detector.method == 'clip'
        assert detector.columns is None
        assert not detector.fitted

    def test_initialization_custom(self):
        """Test custom initialization."""
        detector = ZScoreOutlierDetector(
            threshold=2.0,
            method='remove',
            columns=['col1', 'col2']
        )
        assert detector.threshold == 2.0
        assert detector.method == 'remove'
        assert detector.columns == ['col1', 'col2']

    def test_fit_basic(self):
        """Test basic fitting."""
        df = pd.DataFrame({
            'x': [10, 12, 11, 13, 12, 11],
            'y': [20, 22, 21, 23, 22, 21]
        })
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        assert detector.fitted
        assert detector.stats_ is not None
        assert 'x' in detector.stats_
        assert 'y' in detector.stats_

    def test_fit_calculates_correct_statistics(self):
        """Test that fit calculates correct mean and std."""
        df = pd.DataFrame({'x': [10, 20, 30, 40, 50]})
        detector = ZScoreOutlierDetector(threshold=2.0)
        detector.fit(df)

        stats = detector.get_statistics()
        assert stats['x']['mean'] == 30.0
        assert abs(stats['x']['std'] - df['x'].std()) < 0.001
        assert abs(stats['x']['lower'] - (30 - 2 * df['x'].std())) < 0.001
        assert abs(stats['x']['upper'] - (30 + 2 * df['x'].std())) < 0.001

    def test_transform_requires_fit(self):
        """Test that transform raises error if not fitted."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        detector = ZScoreOutlierDetector()

        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.transform(df)

    def test_fit_transform(self):
        """Test fit_transform combination."""
        df = pd.DataFrame({'x': [10, 12, 11, 100, 13, 12]})
        detector = ZScoreOutlierDetector(threshold=2.0, method='clip')
        result = detector.fit_transform(df)

        assert detector.fitted
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_only_numeric_columns(self):
        """Test that only numeric columns are processed."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e'],
            'numeric2': [10, 20, 30, 40, 50]
        })
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        stats = detector.get_statistics()
        assert 'numeric' in stats
        assert 'numeric2' in stats
        assert 'text' not in stats


class TestZScoreOutlierDetectorMethods:
    """Test different outlier handling methods."""

    def test_clip_method(self):
        """Test clip method caps outliers at bounds."""
        df = pd.DataFrame({'x': [10, 12, 11, 100, 13, 12]})
        detector = ZScoreOutlierDetector(threshold=2.0, method='clip')
        result = detector.fit_transform(df)

        # 100 should be clipped
        assert result['x'].max() < 100
        assert len(result) == len(df)  # Same number of rows
        assert result['x'].isna().sum() == 0  # No NaN values

    def test_remove_method(self):
        """Test remove method deletes outlier rows."""
        df = pd.DataFrame({'x': [10, 12, 11, 100, 13, 12]})
        detector = ZScoreOutlierDetector(threshold=2.0, method='remove')
        result = detector.fit_transform(df)

        # Row with 100 should be removed
        assert len(result) < len(df)
        assert 100 not in result['x'].values

    def test_nan_method(self):
        """Test nan method replaces outliers with NaN."""
        df = pd.DataFrame({'x': [10, 12, 11, 100, 13, 12]})
        detector = ZScoreOutlierDetector(threshold=2.0, method='nan')
        result = detector.fit_transform(df)

        # 100 should become NaN
        assert len(result) == len(df)  # Same number of rows
        assert result['x'].isna().sum() > 0  # Has NaN values
        assert 100 not in result['x'].dropna().values

    def test_remove_multiple_outliers(self):
        """Test remove method with multiple outliers."""
        df = pd.DataFrame({'x': [10, 12, 11, 100, 13, 200, 12]})
        detector = ZScoreOutlierDetector(threshold=2.0, method='remove')
        result = detector.fit_transform(df)

        assert len(result) < len(df)
        assert 100 not in result['x'].values
        assert 200 not in result['x'].values

    def test_clip_both_bounds(self):
        """Test clip method handles both upper and lower outliers."""
        df = pd.DataFrame({'x': [1, 10, 12, 11, 13, 12, 100]})
        detector = ZScoreOutlierDetector(threshold=2.0, method='clip')
        result = detector.fit_transform(df)

        stats = detector.get_statistics()
        assert result['x'].min() >= stats['x']['lower']
        assert result['x'].max() <= stats['x']['upper']


class TestZScoreOutlierDetectorEdgeCases:
    """Test edge cases and error handling."""

    def test_no_numeric_columns(self):
        """Test error when no numeric columns exist."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        detector = ZScoreOutlierDetector()

        with pytest.raises(ValueError, match="No numeric columns"):
            detector.fit(df)

    def test_single_value_column(self):
        """Test handling of column with single unique value."""
        df = pd.DataFrame({'x': [10, 10, 10, 10]})
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        # std = 0, so lower and upper bounds should equal mean
        stats = detector.get_statistics()
        assert stats['x']['std'] == 0.0
        assert stats['x']['lower'] == 10.0
        assert stats['x']['upper'] == 10.0

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({'x': []})
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        # Should fit with NaN statistics
        stats = detector.get_statistics()
        assert pd.isna(stats['x']['mean'])

    def test_nan_values_in_fit(self):
        """Test that NaN values are ignored during fit."""
        df = pd.DataFrame({'x': [10, 12, np.nan, 11, 13, 12]})
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        stats = detector.get_statistics()
        # Mean should be calculated without NaN
        assert not pd.isna(stats['x']['mean'])

    def test_nan_values_in_transform(self):
        """Test that NaN values are preserved during transform."""
        df_train = pd.DataFrame({'x': [10, 12, 11, 13, 12]})
        df_test = pd.DataFrame({'x': [10, np.nan, 11, 100]})

        detector = ZScoreOutlierDetector(threshold=2.0, method='clip')
        detector.fit(df_train)
        result = detector.transform(df_test)

        # Original NaN should be preserved
        assert pd.isna(result['x'].iloc[1])

    def test_not_dataframe_error(self):
        """Test error when input is not DataFrame."""
        detector = ZScoreOutlierDetector()
        arr = np.array([[1, 2], [3, 4]])

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            detector.fit(arr)

    def test_column_not_in_dataframe(self):
        """Test handling when specified column doesn't exist."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        detector = ZScoreOutlierDetector(columns=['y'])
        detector.fit(df)

        # Should fit with empty stats
        stats = detector.get_statistics()
        assert len(stats) == 0


class TestZScoreOutlierDetectorThresholds:
    """Test different threshold values."""

    def test_threshold_2_sigma(self):
        """Test threshold of 2 (95% of data)."""
        # Generate normally distributed data
        np.random.seed(42)
        df = pd.DataFrame({'x': np.random.normal(0, 1, 1000)})

        detector = ZScoreOutlierDetector(threshold=2.0, method='remove')
        result = detector.fit_transform(df)

        # Approximately 95% of data should remain
        assert len(result) > 900  # Allow some variance
        assert len(result) < len(df)

    def test_threshold_3_sigma(self):
        """Test threshold of 3 (99.7% of data)."""
        np.random.seed(42)
        df = pd.DataFrame({'x': np.random.normal(0, 1, 1000)})

        detector = ZScoreOutlierDetector(threshold=3.0, method='remove')
        result = detector.fit_transform(df)

        # Approximately 99.7% of data should remain
        assert len(result) > 990

    def test_low_threshold_removes_more(self):
        """Test that lower threshold removes more outliers."""
        df = pd.DataFrame({'x': [10, 12, 11, 13, 12, 11, 20, 25]})

        detector_strict = ZScoreOutlierDetector(threshold=1.5, method='remove')
        result_strict = detector_strict.fit_transform(df)

        detector_lenient = ZScoreOutlierDetector(threshold=3.0, method='remove')
        result_lenient = detector_lenient.fit_transform(df)

        assert len(result_strict) <= len(result_lenient)

    def test_custom_threshold(self):
        """Test custom threshold value."""
        df = pd.DataFrame({'x': [10, 12, 11, 13, 12, 11]})
        detector = ZScoreOutlierDetector(threshold=1.0)
        detector.fit(df)

        stats = detector.get_statistics()
        mean = stats['x']['mean']
        std = stats['x']['std']

        assert abs(stats['x']['lower'] - (mean - std)) < 0.001
        assert abs(stats['x']['upper'] - (mean + std)) < 0.001


class TestZScoreOutlierDetectorUtilities:
    """Test utility methods."""

    def test_get_statistics(self):
        """Test get_statistics method."""
        df = pd.DataFrame({
            'x': [10, 12, 11, 13, 12],
            'y': [20, 22, 21, 23, 22]
        })
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        stats = detector.get_statistics()
        assert isinstance(stats, dict)
        assert 'x' in stats
        assert 'y' in stats
        assert 'mean' in stats['x']
        assert 'std' in stats['x']
        assert 'lower' in stats['x']
        assert 'upper' in stats['x']

    def test_get_statistics_before_fit(self):
        """Test get_statistics before fitting."""
        detector = ZScoreOutlierDetector()
        assert detector.get_statistics() is None

    def test_get_outlier_counts(self):
        """Test get_outlier_counts method."""
        df = pd.DataFrame({
            'x': [10, 12, 11, 100, 13],
            'y': [20, 22, 21, 23, 200]
        })
        detector = ZScoreOutlierDetector(threshold=2.0, method='clip')
        detector.fit_transform(df)

        counts = detector.get_outlier_counts()
        assert isinstance(counts, dict)
        assert 'x' in counts
        assert 'y' in counts
        assert counts['x'] >= 1  # At least 100 is an outlier
        assert counts['y'] >= 1  # At least 200 is an outlier

    def test_get_outlier_counts_before_transform(self):
        """Test get_outlier_counts before transform."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        assert detector.get_outlier_counts() is None

    def test_detect_outliers(self):
        """Test detect_outliers method."""
        df = pd.DataFrame({'x': [10, 12, 11, 100, 13, 12]})
        detector = ZScoreOutlierDetector(threshold=2.0)
        detector.fit(df)

        mask = detector.detect_outliers(df)
        assert isinstance(mask, pd.DataFrame)
        assert mask.shape == df.shape
        assert mask.dtypes['x'] == bool
        assert mask['x'].sum() >= 1  # At least 100 should be marked

    def test_get_z_scores(self):
        """Test get_z_scores method."""
        df = pd.DataFrame({'x': [10, 20, 30, 40, 50]})
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        z_scores = detector.get_z_scores(df)
        assert isinstance(z_scores, pd.DataFrame)
        assert z_scores.shape == df.shape

        # Z-score of mean should be ~0
        mean_idx = 2  # 30 is the mean
        assert abs(z_scores['x'].iloc[mean_idx]) < 0.001

    def test_get_z_scores_properties(self):
        """Test Z-score properties (mean=0, std=1)."""
        df = pd.DataFrame({'x': [10, 20, 30, 40, 50]})
        detector = ZScoreOutlierDetector()
        detector.fit(df)

        z_scores = detector.get_z_scores(df)

        # Z-scores should have mean ~0 and std ~1
        assert abs(z_scores['x'].mean()) < 0.001
        assert abs(z_scores['x'].std() - 1.0) < 0.001


class TestZScoreOutlierDetectorPractical:
    """Test practical use cases."""

    def test_train_test_split(self):
        """Test proper train/test split usage."""
        train = pd.DataFrame({'x': [10, 12, 11, 13, 12, 11]})
        test = pd.DataFrame({'x': [10, 100, 12]})  # 100 is outlier

        # Fit only on training data
        detector = ZScoreOutlierDetector(threshold=2.0, method='clip')
        detector.fit(train)

        # Apply to test data
        result = detector.transform(test)

        # 100 should be clipped based on training stats
        assert result['x'].iloc[1] < 100

    def test_multiple_columns(self):
        """Test handling multiple columns independently."""
        df = pd.DataFrame({
            'price': [100, 110, 105, 1000, 108],  # 1000 is outlier
            'quantity': [10, 12, 11, 13, 100]      # 100 is outlier
        })
        detector = ZScoreOutlierDetector(threshold=2.0, method='clip')
        result = detector.fit_transform(df)

        # Both outliers should be handled
        assert result['price'].max() < 1000
        assert result['quantity'].max() < 100

    def test_specific_columns_only(self):
        """Test processing only specified columns."""
        df = pd.DataFrame({
            'process': [10, 12, 11, 100, 13],
            'ignore': [10, 12, 11, 100, 13]
        })
        detector = ZScoreOutlierDetector(
            threshold=2.0,
            method='clip',
            columns=['process']
        )
        result = detector.fit_transform(df)

        # Only 'process' should be clipped
        assert result['process'].max() < 100
        assert result['ignore'].max() == 100  # Unchanged

    def test_pipeline_integration(self):
        """Test integration with preprocessing pipeline."""
        from app.ml_engine.preprocessing.pipeline import PreprocessingPipeline
        from app.ml_engine.preprocessing.scaler import StandardScaler

        df = pd.DataFrame({'x': [10, 12, 11, 100, 13, 12]})

        pipeline = PreprocessingPipeline(steps=[
            ZScoreOutlierDetector(threshold=2.0, method='clip', name="outlier_handler"),
            StandardScaler(name="scaler")
        ])

        result = pipeline.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result['x'].max() < 100  # Outlier handled

    def test_real_world_scenario(self):
        """Test with real-world-like data."""
        # Simulate sensor data with occasional spikes
        np.random.seed(42)
        normal_data = np.random.normal(25, 2, 95)  # Temperature readings
        outliers = np.array([50, 60, 0, -10, 55])  # Sensor errors

        df = pd.DataFrame({'temperature': np.concatenate([normal_data, outliers])})

        detector = ZScoreOutlierDetector(threshold=3.0, method='clip')
        result = detector.fit_transform(df)

        # Outliers should be clipped
        assert result['temperature'].max() < 50
        assert result['temperature'].min() > 0

    def test_repr_not_fitted(self):
        """Test string representation before fitting."""
        detector = ZScoreOutlierDetector(threshold=2.5, method='remove')
        repr_str = repr(detector)
        assert 'not fitted' in repr_str
        assert '2.5' in repr_str
        assert 'remove' in repr_str

    def test_repr_fitted(self):
        """Test string representation after fitting."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        detector = ZScoreOutlierDetector(threshold=2.0, method='clip')
        detector.fit(df)

        repr_str = repr(detector)
        assert 'n_columns=2' in repr_str
        assert '2.0' in repr_str
        assert 'clip' in repr_str
