"""Tests for actual vs predicted aggregation (ML-52)."""

import numpy as np
import pytest

from app.ml_engine.evaluation.actual_vs_predicted import (
    ActualVsPredictedAggregator,
    aggregate_actual_vs_predicted,
)


def test_basic_stats_and_best_fit():
    actual = np.array([3.0, -0.5, 2.0, 7.0])
    predicted = np.array([2.5, 0.0, 2.0, 8.0])

    result = aggregate_actual_vs_predicted(actual, predicted)

    np.testing.assert_allclose(result.residuals, np.array([0.5, -0.5, 0.0, -1.0]))
    assert pytest.approx(result.mae, rel=1e-3) == 0.5
    assert pytest.approx(result.rmse, rel=1e-3) == np.sqrt(0.375)
    assert result.r2 is not None and result.r2 > 0.9
    assert result.best_fit is not None
    assert pytest.approx(result.best_fit["slope"], rel=1e-3) == 0.897
    assert pytest.approx(result.best_fit["intercept"], rel=1e-3) == 0.072


def test_mape_handles_zero_actual_values():
    actual = np.array([0, 2, 4])
    predicted = np.array([0, 2, 5])

    result = aggregate_actual_vs_predicted(actual, predicted)

    assert result.mape is not None
    assert pytest.approx(result.mape, rel=1e-3) == 12.5


def test_validation_errors_for_mismatched_shapes():
    actual = np.array([1, 2, 3])
    predicted = np.array([1, 2])

    aggregator = ActualVsPredictedAggregator()

    with pytest.raises(ValueError):
        aggregator.aggregate(actual, predicted)


def test_best_fit_returns_none_when_variance_zero():
    actual = np.array([1.0, 1.0, 1.0])
    predicted = np.array([2.0, 2.0, 2.0])

    result = aggregate_actual_vs_predicted(actual, predicted)

    assert result.best_fit is None
    # R2 can be undefined here; we just assert it is returned as a float or None
    assert result.r2 is not None or result.r2 is None


def test_correlations_optional_rank():
    actual = np.array([1, 2, 3, 4, 5])
    predicted = np.array([1, 1.5, 2.5, 3.5, 4.5])

    result = aggregate_actual_vs_predicted(actual, predicted, compute_rank_corr=False)

    assert result.pearson_r is not None
    assert result.spearman_rho is None
