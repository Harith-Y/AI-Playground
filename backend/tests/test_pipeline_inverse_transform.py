"""
Comprehensive tests for Pipeline inverse_transform functionality.

Tests cover:
- Inverse transform for individual steps
- Pipeline inverse transform with invertible steps
- Pipeline inverse transform with mixed (invertible + non-invertible) steps
- Error handling for non-invertible steps
- Accuracy of roundtrip transformations
"""

import pytest
import pandas as pd
import numpy as np

from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler, RobustScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder, LabelEncoder
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer


class TestPreprocessingStepInverseTransform:
    """Test inverse_transform for individual preprocessing steps."""

    def test_standard_scaler_inverse_transform(self):
        """Test StandardScaler inverse transform."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]})

        scaler = StandardScaler(columns=['x', 'y'])
        scaler.fit(df)

        # Transform
        df_scaled = scaler.transform(df)

        # Inverse transform
        df_original = scaler.inverse_transform(df_scaled)

        # Should be close to original
        pd.testing.assert_frame_equal(df, df_original, rtol=1e-5)

    def test_minmax_scaler_inverse_transform(self):
        """Test MinMaxScaler inverse transform."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]})

        scaler = MinMaxScaler(columns=['x', 'y'])
        scaler.fit(df)

        df_scaled = scaler.transform(df)
        df_original = scaler.inverse_transform(df_scaled)

        pd.testing.assert_frame_equal(df, df_original, rtol=1e-5)

    def test_robust_scaler_inverse_transform(self):
        """Test RobustScaler inverse transform."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]})

        scaler = RobustScaler(columns=['x', 'y'])
        scaler.fit(df)

        df_scaled = scaler.transform(df)
        df_original = scaler.inverse_transform(df_scaled)

        pd.testing.assert_frame_equal(df, df_original, rtol=1e-5)

    def test_label_encoder_inverse_transform(self):
        """Test LabelEncoder inverse transform."""
        df = pd.DataFrame({'category': ['A', 'B', 'C', 'A', 'B']})

        encoder = LabelEncoder(column='category')
        encoder.fit(df)

        df_encoded = encoder.transform(df)
        df_original = encoder.inverse_transform(df_encoded)

        pd.testing.assert_frame_equal(df, df_original)

    def test_imputer_does_not_support_inverse_transform(self):
        """Test that imputer raises NotImplementedError for inverse_transform."""
        df = pd.DataFrame({'x': [1, 2, np.nan, 4, 5]})

        imputer = MeanImputer(columns=['x'])
        imputer.fit(df)

        df_imputed = imputer.transform(df)

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="does not support inverse_transform"):
            imputer.inverse_transform(df_imputed)


class TestPreprocessingStepSupportsInverseTransform:
    """Test supports_inverse_transform method."""

    def test_scaler_supports_inverse_transform(self):
        """Test that scalers support inverse transform."""
        assert StandardScaler(columns=['x']).supports_inverse_transform()
        assert MinMaxScaler(columns=['x']).supports_inverse_transform()
        assert RobustScaler(columns=['x']).supports_inverse_transform()

    def test_label_encoder_supports_inverse_transform(self):
        """Test that LabelEncoder supports inverse transform."""
        assert LabelEncoder(column='x').supports_inverse_transform()

    def test_imputer_does_not_support_inverse_transform(self):
        """Test that imputers don't support inverse transform."""
        assert not MeanImputer(columns=['x']).supports_inverse_transform()
        assert not MedianImputer(columns=['x']).supports_inverse_transform()


class TestPipelineInverseTransform:
    """Test Pipeline inverse_transform method."""

    def test_inverse_transform_single_step(self):
        """Test inverse transform with single step pipeline."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})

        pipeline = Pipeline(steps=[StandardScaler(columns=['x'])])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_original = pipeline.inverse_transform(df_transformed)

        pd.testing.assert_frame_equal(df, df_original, rtol=1e-5)

    def test_inverse_transform_multiple_invertible_steps(self):
        """Test inverse transform with multiple invertible steps."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]})

        pipeline = Pipeline(steps=[
            StandardScaler(columns=['x', 'y']),
            MinMaxScaler(columns=['x', 'y'])
        ])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_original = pipeline.inverse_transform(df_transformed)

        # Should be close to original (within floating point precision)
        pd.testing.assert_frame_equal(df, df_original, rtol=1e-4)

    def test_inverse_transform_three_steps(self):
        """Test inverse transform with three invertible steps."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'z': [100, 200, 300, 400, 500]
        })

        pipeline = Pipeline(steps=[
            StandardScaler(columns=['x', 'y']),
            MinMaxScaler(columns=['z']),
            RobustScaler(columns=['x'])
        ])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_original = pipeline.inverse_transform(df_transformed)

        pd.testing.assert_frame_equal(df, df_original, rtol=1e-4)

    def test_inverse_transform_with_non_invertible_skip(self):
        """Test inverse transform with non-invertible step (skip mode)."""
        df = pd.DataFrame({'x': [1, 2, np.nan, 4, 5], 'y': [10, 20, 30, 40, 50]})

        pipeline = Pipeline(steps=[
            MeanImputer(columns=['x']),  # Non-invertible
            StandardScaler(columns=['x', 'y'])  # Invertible
        ])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)

        # Should skip imputer and only reverse scaler
        df_partial = pipeline.inverse_transform(df_transformed, skip_non_invertible=True)

        # x and y should be unscaled, but x won't have NaN restored
        assert df_partial['x'].isnull().sum() == 0  # No NaN (imputer not reversed)

    def test_inverse_transform_with_non_invertible_error(self):
        """Test inverse transform with non-invertible step (error mode)."""
        df = pd.DataFrame({'x': [1, 2, np.nan, 4, 5]})

        pipeline = Pipeline(steps=[
            MeanImputer(columns=['x']),
            StandardScaler(columns=['x'])
        ])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="does not support inverse_transform"):
            pipeline.inverse_transform(df_transformed, skip_non_invertible=False)

    def test_inverse_transform_not_fitted_error(self):
        """Test error when inverse transforming unfitted pipeline."""
        df = pd.DataFrame({'x': [1, 2, 3]})

        pipeline = Pipeline(steps=[StandardScaler(columns=['x'])])

        with pytest.raises(RuntimeError, match="must be fitted before inverse_transform"):
            pipeline.inverse_transform(df)

    def test_inverse_transform_empty_pipeline(self):
        """Test inverse transform with empty pipeline."""
        df = pd.DataFrame({'x': [1, 2, 3]})

        pipeline = Pipeline()
        pipeline.fitted = True  # Manually set fitted to True

        result = pipeline.inverse_transform(df)

        pd.testing.assert_frame_equal(df, result)


class TestPipelineInverseTransformHelpers:
    """Test Pipeline helper methods for inverse transform support."""

    def test_supports_full_inverse_transform_all_invertible(self):
        """Test supports_full_inverse_transform with all invertible steps."""
        pipeline = Pipeline(steps=[
            StandardScaler(columns=['x']),
            MinMaxScaler(columns=['y'])
        ])

        assert pipeline.supports_full_inverse_transform()

    def test_supports_full_inverse_transform_with_non_invertible(self):
        """Test supports_full_inverse_transform with non-invertible step."""
        pipeline = Pipeline(steps=[
            MeanImputer(columns=['x']),
            StandardScaler(columns=['x'])
        ])

        assert not pipeline.supports_full_inverse_transform()

    def test_supports_full_inverse_transform_empty_pipeline(self):
        """Test supports_full_inverse_transform with empty pipeline."""
        pipeline = Pipeline()

        assert pipeline.supports_full_inverse_transform()

    def test_get_invertible_steps(self):
        """Test get_invertible_steps method."""
        pipeline = Pipeline(steps=[
            MeanImputer(columns=['x']),  # Non-invertible (index 0)
            StandardScaler(columns=['x']),  # Invertible (index 1)
            MedianImputer(columns=['y']),  # Non-invertible (index 2)
            MinMaxScaler(columns=['y'])  # Invertible (index 3)
        ])

        invertible = pipeline.get_invertible_steps()

        assert invertible == [1, 3]

    def test_get_non_invertible_steps(self):
        """Test get_non_invertible_steps method."""
        pipeline = Pipeline(steps=[
            MeanImputer(columns=['x']),  # Non-invertible (index 0)
            StandardScaler(columns=['x']),  # Invertible (index 1)
            MedianImputer(columns=['y']),  # Non-invertible (index 2)
            MinMaxScaler(columns=['y'])  # Invertible (index 3)
        ])

        non_invertible = pipeline.get_non_invertible_steps()

        assert non_invertible == [0, 2]


class TestInverseTransformAccuracy:
    """Test accuracy of roundtrip transformations."""

    def test_roundtrip_standard_scaler(self):
        """Test roundtrip accuracy for StandardScaler."""
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100) * 10 + 50
        })

        pipeline = Pipeline(steps=[StandardScaler(columns=['x', 'y'])])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_roundtrip = pipeline.inverse_transform(df_transformed)

        # Check very close to original
        np.testing.assert_allclose(df.values, df_roundtrip.values, rtol=1e-10)

    def test_roundtrip_minmax_scaler(self):
        """Test roundtrip accuracy for MinMaxScaler."""
        df = pd.DataFrame({
            'x': np.random.rand(100) * 100,
            'y': np.random.rand(100) * 1000
        })

        pipeline = Pipeline(steps=[MinMaxScaler(columns=['x', 'y'])])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_roundtrip = pipeline.inverse_transform(df_transformed)

        np.testing.assert_allclose(df.values, df_roundtrip.values, rtol=1e-10)

    def test_roundtrip_complex_pipeline(self):
        """Test roundtrip accuracy for complex pipeline."""
        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.rand(100) * 100,
            'c': np.random.randn(100) * 5 + 20
        })

        pipeline = Pipeline(steps=[
            StandardScaler(columns=['a', 'c']),
            MinMaxScaler(columns=['b']),
            RobustScaler(columns=['a'])
        ])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_roundtrip = pipeline.inverse_transform(df_transformed)

        # Should be very close to original
        np.testing.assert_allclose(df.values, df_roundtrip.values, rtol=1e-8)

    def test_roundtrip_with_different_feature_ranges(self):
        """Test roundtrip with different feature ranges."""
        df = pd.DataFrame({
            'small': np.random.rand(50) * 0.01,  # Very small values
            'medium': np.random.rand(50) * 100,
            'large': np.random.rand(50) * 1000000  # Very large values
        })

        pipeline = Pipeline(steps=[
            StandardScaler(columns=['small', 'medium', 'large'])
        ])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_roundtrip = pipeline.inverse_transform(df_transformed)

        # Use relative tolerance for different scales
        for col in df.columns:
            np.testing.assert_allclose(
                df[col].values,
                df_roundtrip[col].values,
                rtol=1e-8,
                atol=1e-10
            )


class TestInverseTransformEdgeCases:
    """Test edge cases for inverse transform."""

    def test_inverse_transform_single_value(self):
        """Test inverse transform with single value."""
        df_train = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        df_test = pd.DataFrame({'x': [3]})

        pipeline = Pipeline(steps=[StandardScaler(columns=['x'])])
        pipeline.fit(df_train)

        df_transformed = pipeline.transform(df_test)
        df_original = pipeline.inverse_transform(df_transformed)

        pd.testing.assert_frame_equal(df_test, df_original, rtol=1e-5)

    def test_inverse_transform_preserves_column_names(self):
        """Test that inverse transform preserves column names."""
        df = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [10, 20, 30]
        })

        pipeline = Pipeline(steps=[StandardScaler(columns=['feature_1', 'feature_2'])])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_original = pipeline.inverse_transform(df_transformed)

        assert list(df_original.columns) == list(df.columns)

    def test_inverse_transform_preserves_index(self):
        """Test that inverse transform preserves DataFrame index."""
        df = pd.DataFrame(
            {'x': [1, 2, 3, 4, 5]},
            index=['a', 'b', 'c', 'd', 'e']
        )

        pipeline = Pipeline(steps=[StandardScaler(columns=['x'])])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_original = pipeline.inverse_transform(df_transformed)

        assert list(df_original.index) == list(df.index)

    def test_inverse_transform_with_zero_variance_column(self):
        """Test inverse transform with zero variance column."""
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 5]
        })

        pipeline = Pipeline(steps=[StandardScaler(columns=['constant', 'variable'])])
        pipeline.fit(df)

        df_transformed = pipeline.transform(df)
        df_original = pipeline.inverse_transform(df_transformed)

        # Constant column should remain constant
        assert all(df_original['constant'] == 5)
        pd.testing.assert_series_equal(df['variable'], df_original['variable'], rtol=1e-5)


class TestInverseTransformIntegration:
    """Integration tests for inverse transform."""

    def test_fit_transform_inverse_workflow(self):
        """Test complete fit-transform-inverse workflow."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'z': [100, 200, 300, 400, 500]
        })

        pipeline = Pipeline(steps=[
            StandardScaler(columns=['x', 'y']),
            MinMaxScaler(columns=['z'])
        ])

        # Fit and transform
        df_transformed = pipeline.fit_transform(df)

        # Check transformed values are different
        assert not df.equals(df_transformed)

        # Inverse transform
        df_recovered = pipeline.inverse_transform(df_transformed)

        # Should match original
        pd.testing.assert_frame_equal(df, df_recovered, rtol=1e-5)

    def test_partial_inverse_transform_with_mixed_steps(self):
        """Test partial inverse with mixed invertible/non-invertible steps."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        pipeline = Pipeline(steps=[
            MeanImputer(columns=['x']),  # Non-invertible
            StandardScaler(columns=['x', 'y']),  # Invertible
            MinMaxScaler(columns=['y'])  # Invertible
        ])

        df_transformed = pipeline.fit_transform(df)
        df_partial = pipeline.inverse_transform(df_transformed)

        # y should be fully recovered
        # x should be unscaled but imputation not reversed
        assert df_partial['x'].isnull().sum() == 0  # Imputation not reversed
        pd.testing.assert_series_equal(df['y'], df_partial['y'], rtol=1e-5)

    def test_inverse_transform_info_logging(self):
        """Test that inverse transform provides helpful information."""
        df = pd.DataFrame({'x': [1, 2, 3]})

        pipeline = Pipeline(steps=[
            MeanImputer(columns=['x']),
            StandardScaler(columns=['x'])
        ])
        pipeline.fit(df)

        # Get invertible/non-invertible steps
        invertible = pipeline.get_invertible_steps()
        non_invertible = pipeline.get_non_invertible_steps()

        assert len(invertible) == 1  # StandardScaler
        assert len(non_invertible) == 1  # MeanImputer
        assert not pipeline.supports_full_inverse_transform()
