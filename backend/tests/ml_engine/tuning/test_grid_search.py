"""Tests for Grid Search wrapper."""

import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from app.ml_engine.tuning import run_grid_search, get_default_search_space


def test_run_grid_search_with_explicit_grid():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    result = run_grid_search(
        estimator=estimator,
        X=X,
        y=y,
        param_grid={"C": [0.1, 1.0], "solver": ["lbfgs"], "penalty": ["l2"]},
        scoring="accuracy",
        cv=3,
    )

    assert result.best_params
    assert isinstance(result.best_score, float)
    assert result.n_candidates == 2
    assert len(result.results) == result.n_candidates
    # Ensure ranks are contiguous starting at 1
    assert [item["rank"] for item in result.results][:3] == [1, 2]


def test_run_grid_search_with_default_search_space():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=200)

    defaults = get_default_search_space("logistic_regression")
    assert defaults  # sanity check

    result = run_grid_search(
        estimator=estimator,
        X=X,
        y=y,
        model_id="logistic_regression",
        scoring="accuracy",
        cv=3,
    )

    assert result.best_params
    # Best params should be a subset of default keys
    assert set(result.best_params.keys()).issubset(set(defaults.keys()))


def test_run_grid_search_requires_grid_or_model_id():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=100)

    with pytest.raises(ValueError):
        run_grid_search(estimator=estimator, X=X, y=y, param_grid=None, model_id=None)
