"""
Unit tests for regression metrics module.

Tests cover:
- Basic metrics (MAE, MSE, RMSE, R²)
- Adjusted R²
- Percentage metrics (MAPE, SMAPE, RMSPE)
- Advanced metrics
- Residual analysis
- Model comparison
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from app.ml_engine.evaluation.regression_metrics import (
    RegressionMetricsCalculator,
    RegressionMetrics,
    ResidualAnalysis,
    calculate_regression_metrics
)


class TestRegressionMetricsCalculator:
    """Test suite for RegressionMetricsCalculator."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        calc = RegressionMetricsCalculator()
        assert calc is not None
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        # Perfect predictions should have zero error and R² = 1
        assert metrics.mae == 0.0
        assert metrics.mse == 0.0
        assert metrics.rmse == 0.0
        assert metrics.r2 == 1.0
        assert metrics.n_samples == 5
    
    def test_basic_metrics(self):
        """Test basic regression metrics calculation."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        # Check that metrics are calculated
        assert metrics.mae > 0
        assert metrics.mse > 0
        assert metrics.rmse > 0
        assert 0 <= metrics.r2 <= 1
        
        # RMSE should be sqrt of MSE
        assert abs(metrics.rmse - np.sqrt(metrics.mse)) < 1e-10
    
    def test_adjusted_r2(self):
        """Test adjusted R² calculation."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(y_true, y_pred, n_features=2)
        
        assert metrics.adjusted_r2 is not None
        # Adjusted R² should be less than or equal to R²
        assert metrics.adjusted_r2 <= metrics.r2
    
    def test_percentage_metrics(self):
        """Test percentage-based metrics (MAPE, SMAPE, RMSPE)."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(
            y_true, y_pred,
            include_percentage_metrics=True
        )
        
        assert metrics.mape is not None
        assert metrics.smape is not None
        assert metrics.rmspe is not None
        
        # Percentage metrics should be positive
        assert metrics.mape >= 0
        assert metrics.smape >= 0
        assert metrics.rmspe >= 0
    
    def test_without_percentage_metrics(self):
        """Test calculation without percentage metrics."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(
            y_true, y_pred,
            include_percentage_metrics=False
        )
        
        assert metrics.mape is None
        assert metrics.smape is None
        assert metrics.rmspe is None
    
    def test_advanced_metrics(self):
        """Test advanced metrics calculation."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(
            y_true, y_pred,
            include_advanced=True
        )
        
        assert metrics.explained_variance is not None
        assert metrics.max_error is not None
        assert metrics.median_absolute_error is not None
        assert metrics.mean_residual is not None
        assert metrics.std_residual is not None
        
        # Mean residual should be close to 0
        assert abs(metrics.mean_residual) < 1.0
    
    def test_residual_analysis(self):
        """Test residual analysis functionality."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        analysis = calc.analyze_residuals(y_true, y_pred)
        
        assert isinstance(analysis, ResidualAnalysis)
        assert len(analysis.residuals) == 5
        assert len(analysis.standardized_residuals) == 5
        assert len(analysis.absolute_residuals) == 5
        
        # Check statistics
        assert analysis.mean_residual is not None
        assert analysis.std_residual is not None
        assert analysis.min_residual is not None
        assert analysis.max_residual is not None
        assert analysis.median_residual is not None
        
        # Check outlier detection
        assert analysis.n_outliers is not None
        assert analysis.outlier_indices is not None
    
    def test_residual_analysis_with_outliers(self):
        """Test residual analysis with outliers."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1, 10.0]  # Last one is outlier
        
        calc = RegressionMetricsCalculator()
        analysis = calc.analyze_residuals(y_true, y_pred, outlier_threshold=2.0)
        
        # Should detect at least one outlier
        assert analysis.n_outliers > 0
        assert len(analysis.outlier_indices) > 0
    
    def test_residual_to_dataframe(self):
        """Test conversion of residual analysis to DataFrame."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        analysis = calc.analyze_residuals(y_true, y_pred)
        
        df = analysis.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'residual' in df.columns
        assert 'standardized_residual' in df.columns
        assert 'absolute_residual' in df.columns
        assert len(df) == 5
    
    def test_compare_models(self):
        """Test comparing multiple models."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_preds = {
            'model_a': [1.1, 2.1, 2.9, 4.1, 4.9],
            'model_b': [1.2, 2.2, 3.1, 3.9, 5.1]
        }
        
        calc = RegressionMetricsCalculator()
        results = calc.compare_models(y_true, y_preds)
        
        assert len(results) == 2
        assert 'model_a' in results
        assert 'model_b' in results
        assert isinstance(results['model_a'], RegressionMetrics)
        assert isinstance(results['model_b'], RegressionMetrics)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'mae' in metrics_dict
        assert 'mse' in metrics_dict
        assert 'rmse' in metrics_dict
        assert 'r2' in metrics_dict
    
    def test_repr(self):
        """Test string representation."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        repr_str = repr(metrics)
        
        assert 'RegressionMetrics' in repr_str
        assert 'MAE' in repr_str
        assert 'RMSE' in repr_str
        assert 'R²' in repr_str
    
    def test_mismatched_shapes(self):
        """Test error handling for mismatched shapes."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1]  # Different length
        
        calc = RegressionMetricsCalculator()
        
        with pytest.raises(ValueError, match="must have same shape"):
            calc.calculate_metrics(y_true, y_pred)
    
    def test_empty_arrays(self):
        """Test error handling for empty arrays."""
        y_true = []
        y_pred = []
        
        calc = RegressionMetricsCalculator()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            calc.calculate_metrics(y_true, y_pred)
    
    def test_pandas_input(self):
        """Test with pandas Series input."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.1, 2.2, 2.9, 4.1, 4.8])
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        assert metrics.n_samples == 5
        assert metrics.mae > 0
    
    def test_negative_r2(self):
        """Test with predictions worse than mean baseline."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [5.0, 4.0, 3.0, 2.0, 1.0]  # Inverted predictions
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(y_true, y_pred)
        
        # R² can be negative when model is worse than baseline
        assert metrics.r2 < 0


class TestConvenienceFunction:
    """Test suite for convenience function."""
    
    def test_calculate_regression_metrics_function(self):
        """Test the convenience function."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert isinstance(metrics, RegressionMetrics)
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert 0 <= metrics.r2 <= 1
    
    def test_convenience_function_with_all_params(self):
        """Test convenience function with all parameters."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        
        metrics = calculate_regression_metrics(
            y_true=y_true,
            y_pred=y_pred,
            n_features=2,
            include_percentage_metrics=True,
            include_advanced=True
        )
        
        assert metrics.adjusted_r2 is not None
        assert metrics.mape is not None
        assert metrics.explained_variance is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_constant_predictions(self):
        """Test with constant predictions."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [3.0, 3.0, 3.0, 3.0, 3.0]  # All same
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        # R² should be 0 for constant predictions at mean
        assert metrics.r2 == 0.0
    
    def test_large_errors(self):
        """Test with large prediction errors."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        # Should have large errors
        assert metrics.mae > 10
        assert metrics.rmse > 10
        assert metrics.r2 < 0  # Worse than baseline
    
    def test_small_sample_size(self):
        """Test with small sample size."""
        y_true = [1.0, 2.0]
        y_pred = [1.1, 2.1]
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert metrics.n_samples == 2
        assert metrics.mae > 0
    
    def test_zeros_in_y_true(self):
        """Test with zeros in y_true (affects percentage metrics)."""
        y_true = [0.0, 1.0, 2.0, 3.0, 4.0]
        y_pred = [0.1, 1.1, 2.1, 3.1, 4.1]
        
        calc = RegressionMetricsCalculator()
        metrics = calc.calculate_metrics(
            y_true, y_pred,
            include_percentage_metrics=True
        )
        
        # Should still calculate metrics (with warnings)
        assert metrics.mae > 0
        assert metrics.rmse > 0
    
    def test_large_dataset(self):
        """Test with large dataset."""
        np.random.seed(42)
        n_samples = 10000
        y_true = np.random.randn(n_samples)
        y_pred = y_true + np.random.randn(n_samples) * 0.1
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert metrics.n_samples == n_samples
        assert 0 < metrics.r2 < 1
