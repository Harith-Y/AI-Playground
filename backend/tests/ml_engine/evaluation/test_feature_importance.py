"""Tests for feature importance calculations (ML-54/ML-55)."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from app.ml_engine.evaluation import (
    FeatureImportanceCalculator,
    calculate_feature_importance,
    calculate_permutation_importance,
)


def test_native_feature_importances_tree():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=42, n_estimators=20).fit(X, y)

    result = calculate_feature_importance(model, X, y)

    assert result.method == "feature_importances_"
    assert result.n_features == X.shape[1]
    ranked = result.to_ranked_list()
    assert ranked[0]["rank"] == 1
    assert ranked[-1]["rank"] == X.shape[1]


def test_native_feature_importances_coef():
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=123,
    )
    model = LogisticRegression(max_iter=300, multi_class="auto", solver="lbfgs").fit(X, y)

    result = calculate_feature_importance(model, X, y)

    assert result.method == "coef_"
    assert set(result.importances.keys()) == {f"feature_{i}" for i in range(X.shape[1])}
    assert all(val >= 0 for val in result.importances.values())


def test_permutation_importance_explicit():
    X, y = load_iris(return_X_y=True)
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    calculator = FeatureImportanceCalculator(scoring="accuracy", n_repeats=3, random_state=0)

    result = calculator.calculate(model, X, y, method="permutation")

    assert result.method == "permutation"
    assert result.metadata["n_repeats"] == 3
    assert len(result.importances) == X.shape[1]


def test_permutation_importance_helper_function():
    X, y = load_iris(return_X_y=True)
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)

    result = calculate_permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring="accuracy",
        n_repeats=4,
        random_state=123,
    )

    assert result.method == "permutation"
    assert result.metadata["n_repeats"] == 4
    assert result.n_features == X.shape[1]


def test_auto_fallback_to_permutation_when_no_native_importance():
    X, y = make_regression(n_samples=80, n_features=4, noise=0.1, random_state=21)
    model = KNeighborsRegressor(n_neighbors=4).fit(X, y)
    calculator = FeatureImportanceCalculator(n_repeats=2, random_state=1)

    result = calculator.calculate(model, X, y, method="auto")

    assert result.method == "permutation"
    assert len(result.importances) == X.shape[1]


def test_feature_names_validation_errors():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=7, n_estimators=10).fit(X, y)

    with pytest.raises(ValueError):
        calculate_feature_importance(model, X, y, feature_names=["a", "b"])  # wrong length

    with pytest.raises(ValueError):
        calculator = FeatureImportanceCalculator()
        calculator.calculate(model, X, method="permutation")  # missing y


def test_ranked_list_sorting_descending():
    importances = {"a": 0.1, "b": 0.4, "c": 0.2}
    result = FeatureImportanceCalculator()._build_result(
        np.array(list(importances.values())),
        ["a", "b", "c"],
        "feature_importances_",
        {},
    )

    ranked = result.to_ranked_list()
    assert [item["feature"] for item in ranked] == ["b", "c", "a"]
    assert [item["importance"] for item in ranked] == sorted(importances.values(), reverse=True)
