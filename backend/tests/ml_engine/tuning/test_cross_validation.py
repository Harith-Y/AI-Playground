"""Tests for cross-validation utilities."""

import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from app.ml_engine.tuning import (
    run_cross_validation,
    run_simple_cross_validation,
    create_cv_splitter,
    compare_models_cv,
)


def test_run_cross_validation_basic():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    result = run_cross_validation(
        estimator=estimator,
        X=X,
        y=y,
        scoring="accuracy",
        cv=3,
    )

    assert result.cv_folds == 3
    assert len(result.scores) == 3
    assert isinstance(result.mean_score, float)
    assert isinstance(result.std_score, float)
    assert result.scoring == "accuracy"
    assert result.fit_times is not None
    assert len(result.fit_times) == 3


def test_run_cross_validation_with_train_scores():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    result = run_cross_validation(
        estimator=estimator,
        X=X,
        y=y,
        scoring="accuracy",
        cv=3,
        return_train_score=True,
    )

    assert result.train_scores is not None
    assert len(result.train_scores) == 3
    # Train scores should typically be higher than test scores
    assert np.mean(result.train_scores) >= result.mean_score - 0.3


def test_run_cross_validation_multiple_metrics():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    result = run_cross_validation(
        estimator=estimator,
        X=X,
        y=y,
        scoring=["accuracy", "f1_weighted", "precision_weighted"],
        cv=3,
    )

    assert result.scoring == "accuracy"
    assert result.additional_metrics is not None
    assert "f1_weighted" in result.additional_metrics
    assert "precision_weighted" in result.additional_metrics
    assert len(result.additional_metrics["f1_weighted"]) == 3


def test_run_cross_validation_to_dict():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    result = run_cross_validation(
        estimator=estimator,
        X=X,
        y=y,
        scoring="accuracy",
        cv=3,
        return_train_score=True,
    )

    data = result.to_dict()
    assert isinstance(data, dict)
    assert "mean_score" in data
    assert "std_score" in data
    assert "mean_fit_time" in data
    assert "mean_train_score" in data
    assert isinstance(data["mean_score"], float)


def test_run_simple_cross_validation():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    scores = run_simple_cross_validation(
        estimator=estimator,
        X=X,
        y=y,
        scoring="accuracy",
        cv=5,
    )

    assert isinstance(scores, list)
    assert len(scores) == 5
    assert all(isinstance(s, float) for s in scores)


def test_create_cv_splitter_kfold():
    splitter = create_cv_splitter(cv_type="kfold", n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import KFold
    assert isinstance(splitter, KFold)


def test_create_cv_splitter_stratified():
    splitter = create_cv_splitter(cv_type="stratified", n_splits=3, shuffle=True, random_state=0)
    from sklearn.model_selection import StratifiedKFold
    assert isinstance(splitter, StratifiedKFold)


def test_create_cv_splitter_group():
    splitter = create_cv_splitter(cv_type="group", n_splits=4)
    from sklearn.model_selection import GroupKFold
    assert isinstance(splitter, GroupKFold)


def test_create_cv_splitter_timeseries():
    splitter = create_cv_splitter(cv_type="timeseries", n_splits=3, gap=2)
    from sklearn.model_selection import TimeSeriesSplit
    assert isinstance(splitter, TimeSeriesSplit)


def test_create_cv_splitter_invalid():
    with pytest.raises(ValueError):
        create_cv_splitter(cv_type="invalid", n_splits=5)


def test_compare_models_cv():
    X, y = load_iris(return_X_y=True)

    estimators = {
        "logistic": LogisticRegression(max_iter=300),
        "tree": DecisionTreeClassifier(random_state=42),
        "forest": RandomForestClassifier(n_estimators=10, random_state=42),
    }

    results = compare_models_cv(
        estimators=estimators,
        X=X,
        y=y,
        scoring="accuracy",
        cv=3,
    )

    assert len(results) == 3
    assert "logistic" in results
    assert "tree" in results
    assert "forest" in results

    for name, result in results.items():
        assert result.cv_folds == 3
        assert len(result.scores) == 3
        assert isinstance(result.mean_score, float)


def test_cross_validation_with_custom_splitter():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    splitter = create_cv_splitter(cv_type="stratified", n_splits=4, shuffle=True, random_state=123)

    result = run_cross_validation(
        estimator=estimator,
        X=X,
        y=y,
        scoring="accuracy",
        cv=splitter,
    )

    assert result.cv_folds == 4
    assert len(result.scores) == 4


def test_cross_validation_result_statistics():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    result = run_cross_validation(
        estimator=estimator,
        X=X,
        y=y,
        scoring="accuracy",
        cv=5,
    )

    # Verify statistical properties
    assert result.min_score <= result.mean_score <= result.max_score
    assert result.std_score >= 0
    assert result.median_score >= result.min_score
    assert result.median_score <= result.max_score


def test_cross_validation_multiple_metrics_to_dict():
    X, y = load_iris(return_X_y=True)
    estimator = LogisticRegression(max_iter=300)

    result = run_cross_validation(
        estimator=estimator,
        X=X,
        y=y,
        scoring=["accuracy", "precision_weighted"],
        cv=3,
    )

    data = result.to_dict()
    assert "additional_metrics" in data
    assert "precision_weighted" in data["additional_metrics"]
    assert "mean" in data["additional_metrics"]["precision_weighted"]
    assert "std" in data["additional_metrics"]["precision_weighted"]
    assert "scores" in data["additional_metrics"]["precision_weighted"]
