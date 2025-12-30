"""Tests for Bayesian optimization wrapper."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import warnings

from app.ml_engine.tuning.bayesian import (
    BayesianSearchResult,
    run_bayesian_search,
)


class TestBayesianSearchResult(unittest.TestCase):
    """Test BayesianSearchResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BayesianSearchResult(
            best_params={"n_estimators": 100, "max_depth": 5},
            best_score=0.85,
            scoring="accuracy",
            cv_folds=5,
            n_iter=10,
            n_candidates=10,
            method="bayesian",
            results=[
                {
                    "rank": 1,
                    "params": {"n_estimators": 100, "max_depth": 5},
                    "mean_score": 0.85,
                    "std_score": 0.02,
                    "scores": [0.83, 0.84, 0.85, 0.86, 0.87],
                }
            ],
        )

        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["best_params"], {"n_estimators": 100, "max_depth": 5})
        self.assertIsInstance(result_dict["best_score"], float)
        self.assertEqual(result_dict["scoring"], "accuracy")
        self.assertEqual(result_dict["cv_folds"], 5)
        self.assertEqual(result_dict["n_iter"], 10)
        self.assertEqual(result_dict["n_candidates"], 10)
        self.assertEqual(result_dict["method"], "bayesian")
        self.assertEqual(len(result_dict["results"]), 1)

    def test_top(self):
        """Test getting top-n results."""
        results_list = [
            {
                "rank": i + 1,
                "params": {"n_estimators": 100 - i * 10},
                "mean_score": 0.9 - i * 0.01,
                "std_score": 0.02,
                "scores": [0.88 + j * 0.01 for j in range(5)],
            }
            for i in range(20)
        ]

        result = BayesianSearchResult(
            best_params={"n_estimators": 100},
            best_score=0.9,
            scoring="accuracy",
            cv_folds=5,
            n_iter=20,
            n_candidates=20,
            method="bayesian",
            results=results_list,
        )

        top_5 = result.top(5)
        self.assertEqual(len(top_5), 5)
        self.assertEqual(top_5[0]["rank"], 1)
        self.assertEqual(top_5[4]["rank"], 5)

        # Default should return up to 10
        top_default = result.top()
        self.assertEqual(len(top_default), 10)


class TestRunBayesianSearch(unittest.TestCase):
    """Test run_bayesian_search function."""

    def setUp(self):
        """Set up test data."""
        self.X_clf, self.y_clf = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )

    def test_bayesian_search_with_skopt(self):
        """Test Bayesian search with scikit-optimize (if available)."""
        try:
            from skopt import BayesSearchCV

            estimator = RandomForestClassifier(random_state=42)
            search_spaces = {
                "n_estimators": [10, 50, 100],
                "max_depth": [3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
            }

            result = run_bayesian_search(
                estimator=estimator,
                X=self.X_clf,
                y=self.y_clf,
                search_spaces=search_spaces,
                n_iter=5,
                cv=3,
                random_state=42,
                verbose=0,
            )

            self.assertIsInstance(result, BayesianSearchResult)
            self.assertEqual(result.method, "bayesian")
            self.assertIsInstance(result.best_params, dict)
            self.assertIsInstance(result.best_score, float)
            self.assertEqual(result.cv_folds, 3)
            self.assertEqual(result.n_iter, 5)
            self.assertGreater(result.n_candidates, 0)
            self.assertEqual(len(result.results), result.n_candidates)

            # Check results are sorted by score
            scores = [r["mean_score"] for r in result.results]
            self.assertEqual(scores, sorted(scores, reverse=True))

        except ImportError:
            self.skipTest("scikit-optimize not installed")

    def test_bayesian_search_fallback(self):
        """Test fallback to RandomizedSearchCV when skopt not available."""
        with patch.dict("sys.modules", {"skopt": None}):
            # Force ImportError for skopt
            estimator = LogisticRegression(random_state=42, max_iter=200)
            search_spaces = {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            }

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_bayesian_search(
                    estimator=estimator,
                    X=self.X_clf,
                    y=self.y_clf,
                    search_spaces=search_spaces,
                    n_iter=3,
                    cv=3,
                    random_state=42,
                    verbose=0,
                )

                # Check that fallback warning was issued
                self.assertTrue(
                    any("scikit-optimize not installed" in str(warning.message) for warning in w)
                )

            self.assertIsInstance(result, BayesianSearchResult)
            self.assertEqual(result.method, "random_fallback")
            self.assertIsInstance(result.best_params, dict)
            self.assertIsInstance(result.best_score, float)

    def test_with_default_search_space(self):
        """Test using default search space from model_id."""
        estimator = RandomForestClassifier(random_state=42)

        result = run_bayesian_search(
            estimator=estimator,
            X=self.X_clf,
            y=self.y_clf,
            model_id="random_forest_classifier",
            n_iter=3,
            cv=3,
            random_state=42,
            verbose=0,
        )

        self.assertIsInstance(result, BayesianSearchResult)
        self.assertIsInstance(result.best_params, dict)
        self.assertGreater(len(result.best_params), 0)

    def test_with_scoring_metric(self):
        """Test with custom scoring metric."""
        estimator = RandomForestClassifier(random_state=42)
        search_spaces = {
            "n_estimators": [10, 50],
            "max_depth": [3, 5],
        }

        result = run_bayesian_search(
            estimator=estimator,
            X=self.X_clf,
            y=self.y_clf,
            search_spaces=search_spaces,
            n_iter=2,
            scoring="f1_weighted",
            cv=3,
            random_state=42,
            verbose=0,
        )

        self.assertEqual(result.scoring, "f1_weighted")

    def test_with_regression(self):
        """Test Bayesian search with regression."""
        estimator = RandomForestRegressor(random_state=42)
        search_spaces = {
            "n_estimators": [10, 50],
            "max_depth": [3, 5, 7],
        }

        result = run_bayesian_search(
            estimator=estimator,
            X=self.X_reg,
            y=self.y_reg,
            search_spaces=search_spaces,
            n_iter=3,
            scoring="neg_mean_squared_error",
            cv=3,
            random_state=42,
            verbose=0,
        )

        self.assertIsInstance(result, BayesianSearchResult)
        self.assertEqual(result.scoring, "neg_mean_squared_error")

    def test_error_no_search_space_or_model_id(self):
        """Test error when neither search_spaces nor model_id provided."""
        estimator = RandomForestClassifier(random_state=42)

        with self.assertRaises(ValueError) as context:
            run_bayesian_search(
                estimator=estimator,
                X=self.X_clf,
                y=self.y_clf,
                n_iter=3,
                cv=3,
            )

        self.assertIn("search_spaces is required", str(context.exception))

    def test_error_invalid_model_id(self):
        """Test error with invalid model_id."""
        estimator = RandomForestClassifier(random_state=42)

        with self.assertRaises(ValueError) as context:
            run_bayesian_search(
                estimator=estimator,
                X=self.X_clf,
                y=self.y_clf,
                model_id="nonexistent_model",
                n_iter=3,
                cv=3,
            )

        self.assertIn("No default search space", str(context.exception))

    def test_result_structure(self):
        """Test detailed structure of results."""
        estimator = LogisticRegression(random_state=42, max_iter=200)
        search_spaces = {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"],
        }

        result = run_bayesian_search(
            estimator=estimator,
            X=self.X_clf,
            y=self.y_clf,
            search_spaces=search_spaces,
            n_iter=3,
            cv=3,
            random_state=42,
            verbose=0,
        )

        # Check first result structure
        first_result = result.results[0]
        self.assertIn("rank", first_result)
        self.assertIn("params", first_result)
        self.assertIn("mean_score", first_result)
        self.assertIn("std_score", first_result)
        self.assertIn("scores", first_result)

        self.assertEqual(first_result["rank"], 1)
        self.assertIsInstance(first_result["params"], dict)
        self.assertIsInstance(first_result["mean_score"], float)
        self.assertIsInstance(first_result["std_score"], float)
        self.assertIsInstance(first_result["scores"], list)
        self.assertEqual(len(first_result["scores"]), 3)  # cv=3

    def test_optimizer_kwargs(self):
        """Test passing additional optimizer kwargs."""
        try:
            from skopt import BayesSearchCV

            estimator = RandomForestClassifier(random_state=42)
            search_spaces = {
                "n_estimators": [10, 50],
                "max_depth": [3, 5],
            }

            # Test with additional optimizer kwargs
            result = run_bayesian_search(
                estimator=estimator,
                X=self.X_clf,
                y=self.y_clf,
                search_spaces=search_spaces,
                n_iter=3,
                cv=3,
                random_state=42,
                optimizer_kwargs={"base_estimator": "GP"},
                verbose=0,
            )

            self.assertIsInstance(result, BayesianSearchResult)
            self.assertEqual(result.method, "bayesian")

        except ImportError:
            self.skipTest("scikit-optimize not installed")

    def test_n_jobs_parallel(self):
        """Test parallel execution with n_jobs."""
        estimator = RandomForestClassifier(random_state=42)
        search_spaces = {
            "n_estimators": [10, 50, 100],
            "max_depth": [3, 5, 7],
        }

        result = run_bayesian_search(
            estimator=estimator,
            X=self.X_clf,
            y=self.y_clf,
            search_spaces=search_spaces,
            n_iter=3,
            cv=3,
            n_jobs=2,
            random_state=42,
            verbose=0,
        )

        self.assertIsInstance(result, BayesianSearchResult)


if __name__ == "__main__":
    unittest.main()
