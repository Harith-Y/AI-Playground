"""Tests for Random Search wrapper."""

import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from app.ml_engine.tuning import run_random_search, get_default_search_space


def test_run_random_search_with_explicit_distributions():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    result = run_random_search(
        estimator=estimator,
        X=X,
        y=y,
        param_distributions={"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"], "penalty": ["l2"]},
        n_iter=3,
        scoring="accuracy",
        cv=3,
        random_state=42,
    )

    assert result.best_params
    assert isinstance(result.best_score, float)
    assert result.n_iter == 3
    assert result.n_candidates == 3
    assert len(result.results) == 3
    # Ensure ranks are contiguous starting at 1
    assert [item["rank"] for item in result.results] == [1, 2, 3]


def test_run_random_search_with_default_search_space():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=200)

    defaults = get_default_search_space("logistic_regression")
    assert defaults  # sanity check

    result = run_random_search(
        estimator=estimator,
        X=X,
        y=y,
        model_id="logistic_regression",
        n_iter=5,
        scoring="accuracy",
        cv=3,
        random_state=123,
    )

    assert result.best_params
    assert result.n_iter == 5
    # Best params should be a subset of default keys
    assert set(result.best_params.keys()).issubset(set(defaults.keys()))


def test_run_random_search_requires_distributions_or_model_id():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=100)

    with pytest.raises(ValueError):
        run_random_search(
            estimator=estimator, X=X, y=y, param_distributions=None, model_id=None
        )


def test_random_search_result_to_dict():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=200)

    result = run_random_search(
        estimator=estimator,
        X=X,
        y=y,
        param_distributions={"C": [0.5, 1.0], "penalty": ["l2"]},
        n_iter=2,
        scoring="accuracy",
        cv=2,
        random_state=7,
    )

    data = result.to_dict()
    assert isinstance(data, dict)
    assert "best_params" in data
    assert "best_score" in data
    assert isinstance(data["best_score"], float)
    assert len(data["results"]) == 2


def test_random_search_top_n():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=200)

    result = run_random_search(
        estimator=estimator,
        X=X,
        y=y,
        param_distributions={"C": [0.1, 1.0, 10.0, 100.0], "penalty": ["l2"]},
        n_iter=4,
        scoring="accuracy",
        cv=3,
        random_state=0,
    )

    top_2 = result.top(n=2)
    assert len(top_2) == 2
    # Top results should be sorted by mean_score descending
    assert top_2[0]["mean_score"] >= top_2[1]["mean_score"]
