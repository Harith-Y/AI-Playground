"""Tests for residual analysis utilities."""

import numpy as np
import pytest

from app.ml_engine.evaluation.residual_analysis import (
    ResidualAnalysisCalculator,
    analyze_residuals,
)


def test_residual_analysis_basic_stats():
    actual = np.array([3.0, -0.5, 2.0, 7.0])
    predicted = np.array([2.5, 0.0, 2.0, 8.0])

    result = analyze_residuals(actual, predicted, compute_normality=False)

    np.testing.assert_allclose(result.residuals, np.array([0.5, -0.5, 0.0, -1.0]))
    assert pytest.approx(result.mae, rel=1e-3) == 0.5
    assert pytest.approx(result.rmse, rel=1e-3) == np.sqrt(0.375)
    assert result.n_samples == 4
    assert result.standardized_residuals is not None
    assert set(result.quantiles.keys()) == {"q05", "q25", "q50", "q75", "q95"}


def test_outlier_detection_with_standardized_residuals():
    actual = np.array([10, 10, 10, 10, 10])
    predicted = np.array([10, 10, 10, 10, -5])

    calculator = ResidualAnalysisCalculator(zscore_threshold=1.5, include_standardized=True)
    result = calculator.analyze(actual, predicted, compute_normality=False)

    assert result.outlier_indices == [4]
    assert result.outlier_threshold == 1.5


def test_mape_handles_zero_actual_values():
    actual = np.array([0, 2, 4])
    predicted = np.array([0, 2, 5])

    result = analyze_residuals(actual, predicted, compute_normality=False)

    assert result.mape is not None
    assert pytest.approx(result.mape, rel=1e-3) == 12.5


def test_validation_errors_for_mismatched_shapes():
    actual = np.array([1, 2, 3])
    predicted = np.array([1, 2])

    calculator = ResidualAnalysisCalculator()

    with pytest.raises(ValueError):
        calculator.analyze(actual, predicted)


def test_normality_test_optional():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([0.9, 2.1, 3.0, 3.9])

    result = analyze_residuals(actual, predicted, compute_normality=True)

    assert result.normality_test is not None
    assert 0.0 <= result.normality_test["p_value"] <= 1.0


def test_correlation_abs_residuals_predicted():
    actual = np.array([1, 2, 3, 4, 5])
    predicted = np.array([1, 1.5, 2.5, 3.5, 4.5])

    result = analyze_residuals(actual, predicted, compute_normality=False)

    assert result.correlation_abs_residuals_predicted is not None
    assert -1.0 <= result.correlation_abs_residuals_predicted <= 1.0
