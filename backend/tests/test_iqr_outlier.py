"""
Unit tests for IQROutlierDetector.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.cleaner import IQROutlierDetector


class TestIQROutlierDetectorBasic:
    """Basic functionality tests"""

    def test_initialization(self):
        """Test initialization with different parameters"""
        detector = IQROutlierDetector()
        assert detector.threshold == 1.5
        assert detector.method == 'clip'
        assert not detector.fitted

        detector_custom = IQROutlierDetector(threshold=3.0, method='remove')
        assert detector_custom.threshold == 3.0
        assert detector_custom.method == 'remove'

    def test_fit_calculates_bounds(self):
        """Test that fit calculates correct IQR bounds"""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        detector = IQROutlierDetector(threshold=1.5)
        detector.fit(df)

        assert detector.fitted
        bounds = detector.get_bounds()
        assert 'values' in bounds
        assert 'lower' in bounds['values']
        assert 'upper' in bounds['values']
        assert 'Q1' in bounds['values']
        assert 'Q3' in bounds['values']
        assert 'IQR' in bounds['values']

    def test_clip_method(self):
        """Test clipping outliers to bounds"""
        df = pd.DataFrame({
            'price': [10, 12, 11, 13, 100, 10, 12]  # 100 is outlier
        })

        detector = IQROutlierDetector(threshold=1.5, method='clip')
        df_clean = detector.fit_transform(df)

        # Outlier should be clipped, not removed
        assert len(df_clean) == len(df)
        assert df_clean['price'].max() < 100  # Outlier clipped

    def test_remove_method(self):
        """Test removing rows with outliers"""
        df = pd.DataFrame({
            'price': [10, 12, 11, 13, 100, 10, 12]
        })

        detector = IQROutlierDetector(threshold=1.5, method='remove')
        df_clean = detector.fit_transform(df)

        # Row with outlier should be removed
        assert len(df_clean) < len(df)
        assert 100 not in df_clean['price'].values

    def test_nan_method(self):
        """Test replacing outliers with NaN"""
        df = pd.DataFrame({
            'price': [10, 12, 11, 13, 100, 10, 12]
        })

        detector = IQROutlierDetector(threshold=1.5, method='nan')
        df_clean = detector.fit_transform(df)

        # Row count same, but outlier is NaN
        assert len(df_clean) == len(df)
        assert df_clean['price'].isna().sum() > 0


class TestIQROutlierDetectorMethods:
    """Test utility methods"""

    def test_get_bounds_before_fit(self):
        """Test get_bounds before fitting"""
        detector = IQROutlierDetector()
        assert detector.get_bounds() is None

    def test_get_outlier_counts(self):
        """Test getting outlier counts after transform"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 100],
            'col2': [5, 6, 7, 8]
        })

        detector = IQROutlierDetector(method='clip')
        detector.fit_transform(df)

        counts = detector.get_outlier_counts()
        assert counts is not None
        assert 'col1' in counts

    def test_detect_outliers_mask(self):
        """Test detect_outliers returns boolean mask"""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]
        })

        detector = IQROutlierDetector(threshold=1.5)
        detector.fit(df)

        mask = detector.detect_outliers(df)
        assert isinstance(mask, pd.DataFrame)
        assert mask.shape == df.shape
        assert mask['values'].iloc[-1] == True  # 100 is outlier

    def test_specific_columns(self):
        """Test detecting outliers in specific columns only"""
        df = pd.DataFrame({
            'check_this': [1, 2, 3, 100],
            'ignore_this': [1, 2, 3, 100]
        })

        detector = IQROutlierDetector(columns=['check_this'])
        detector.fit(df)

        bounds = detector.get_bounds()
        assert 'check_this' in bounds
        assert 'ignore_this' not in bounds


class TestIQROutlierDetectorEdgeCases:
    """Test edge cases"""

    def test_no_outliers(self):
        """Test when no outliers exist"""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5]
        })

        detector = IQROutlierDetector(threshold=1.5, method='remove')
        df_clean = detector.fit_transform(df)

        assert len(df_clean) == len(df)

    def test_all_same_values(self):
        """Test with constant column (IQR=0)"""
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5]
        })

        detector = IQROutlierDetector()
        detector.fit(df)

        bounds = detector.get_bounds()
        # IQR is 0, so bounds are Q1 and Q3
        assert bounds['constant']['IQR'] == 0

    def test_multiple_columns(self):
        """Test with multiple numeric columns"""
        df = pd.DataFrame({
            'price': [10, 12, 11, 100, 13],
            'quantity': [5, 6, 5, 7, 200],
            'rating': [4, 5, 4, 5, 4]
        })

        detector = IQROutlierDetector(threshold=1.5, method='clip')
        df_clean = detector.fit_transform(df)

        bounds = detector.get_bounds()
        assert len(bounds) == 3
        assert all(col in bounds for col in ['price', 'quantity', 'rating'])

    def test_transform_before_fit_error(self):
        """Test that transform before fit raises error"""
        df = pd.DataFrame({'values': [1, 2, 3]})
        detector = IQROutlierDetector()

        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.transform(df)

    def test_non_dataframe_input(self):
        """Test error with non-DataFrame input"""
        arr = np.array([[1, 2], [3, 4]])
        detector = IQROutlierDetector()

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            detector.fit(arr)


class TestIQROutlierDetectorThresholds:
    """Test different threshold values"""

    def test_threshold_15_standard(self):
        """Test standard threshold=1.5 (mild outliers)"""
        df = pd.DataFrame({
            'values': list(range(1, 11)) + [50]  # 50 is outlier
        })

        detector = IQROutlierDetector(threshold=1.5, method='remove')
        df_clean = detector.fit_transform(df)

        assert len(df_clean) < len(df)

    def test_threshold_30_extreme(self):
        """Test threshold=3.0 (extreme outliers only)"""
        df = pd.DataFrame({
            'values': list(range(1, 11)) + [20]  # 20 might not be extreme outlier
        })

        detector = IQROutlierDetector(threshold=3.0, method='remove')
        df_clean = detector.fit_transform(df)

        # With higher threshold, fewer outliers detected
        assert len(df_clean) >= len(df) - 1


class TestIQROutlierDetectorPractical:
    """Practical use case tests"""

    def test_sales_data(self):
        """Test with realistic sales data"""
        df = pd.DataFrame({
            'daily_sales': [100, 110, 105, 115, 108, 1000, 112, 107],  # 1000 is outlier
            'customers': [20, 22, 21, 23, 21, 25, 22, 21]
        })

        detector = IQROutlierDetector(threshold=1.5, method='clip')
        df_clean = detector.fit_transform(df)

        # Outlier should be clipped
        assert df_clean['daily_sales'].max() < 1000
        counts = detector.get_outlier_counts()
        assert counts['daily_sales'] > 0

    def test_pipeline_compatibility(self):
        """Test with preprocessing pipeline"""
        from app.ml_engine.preprocessing.pipeline import PreprocessingPipeline
        from app.ml_engine.preprocessing.scaler import StandardScaler

        df = pd.DataFrame({
            'values': [10, 12, 11, 100, 13, 10, 12]
        })

        pipeline = PreprocessingPipeline(steps=[
            IQROutlierDetector(threshold=1.5, method='clip', name="outlier_removal"),
            StandardScaler(name="scaling")
        ])

        df_processed = pipeline.fit_transform(df)

        # Should have outliers clipped then scaled
        assert df_processed is not None

    def test_separate_train_test(self):
        """Test fitting on train and transforming test"""
        train = pd.DataFrame({
            'price': [10, 12, 11, 13, 10, 12, 14, 11, 13]
        })

        test = pd.DataFrame({
            'price': [11, 150, 12]  # 150 is outlier based on train bounds
        })

        detector = IQROutlierDetector(threshold=1.5, method='clip')
        detector.fit(train)

        test_clean = detector.transform(test)

        # 150 should be clipped based on training bounds
        assert test_clean['price'].max() < 150
