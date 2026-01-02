"""Integration tests for hyperparameter tuning utilities.

Tests cover end-to-end workflows combining multiple tuning components.
"""

import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from app.ml_engine.tuning import (
    DEFAULT_SEARCH_SPACES,
    get_default_search_space,
    run_grid_search,
    run_random_search,
    run_bayesian_search,
    run_cross_validation,
    compare_models_cv,
)


class TestTuningWorkflows(unittest.TestCase):
    """Test complete tuning workflows."""

    @classmethod
    def setUpClass(cls):
        """Set up test data for all tests."""
        # Classification dataset
        cls.X_clf, cls.y_clf = make_classification(
            n_samples=200,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            random_state=42,
        )
        cls.X_clf_train, cls.X_clf_test, cls.y_clf_train, cls.y_clf_test = train_test_split(
            cls.X_clf, cls.y_clf, test_size=0.2, random_state=42
        )

        # Regression dataset
        cls.X_reg, cls.y_reg = make_regression(
            n_samples=200, n_features=15, n_informative=10, random_state=42
        )
        cls.X_reg_train, cls.X_reg_test, cls.y_reg_train, cls.y_reg_test = train_test_split(
            cls.X_reg, cls.y_reg, test_size=0.2, random_state=42
        )

    def test_quick_grid_search_workflow(self):
        """Test quick grid search with default spaces."""
        # Quick grid search
        result = run_grid_search(
            estimator=LogisticRegression(max_iter=200),
            X=self.X_clf_train,
            y=self.y_clf_train,
            model_id="logistic_regression",
            cv=3,
            scoring="accuracy",
        )

        self.assertIsNotNone(result.best_params)
        self.assertGreater(result.best_score, 0.5)
        self.assertIn("C", result.best_params)

        # Train final model
        best_model = LogisticRegression(**result.best_params, max_iter=200)
        best_model.fit(self.X_clf_train, self.y_clf_train)
        test_score = best_model.score(self.X_clf_test, self.y_clf_test)

        self.assertGreater(test_score, 0.5)

    def test_random_search_then_validation_workflow(self):
        """Test random search followed by cross-validation."""
        # Random search
        search_result = run_random_search(
            estimator=RandomForestClassifier(random_state=42),
            X=self.X_clf_train,
            y=self.y_clf_train,
            model_id="random_forest_classifier",
            n_iter=10,
            cv=3,
            scoring="f1",
            random_state=42,
        )

        self.assertGreater(search_result.best_score, 0)

        # Validate with more thorough CV
        best_model = RandomForestClassifier(**search_result.best_params, random_state=42)
        cv_result = run_cross_validation(
            estimator=best_model,
            X=self.X_clf,
            y=self.y_clf,
            cv=5,
            scoring="f1",
        )

        self.assertGreater(cv_result.mean_score, 0)
        self.assertGreater(len(cv_result.scores), 0)

    def test_progressive_search_workflow(self):
        """Test progressive refinement: grid -> random -> bayesian."""
        # 1. Coarse grid search
        grid_result = run_grid_search(
            estimator=RandomForestClassifier(random_state=42),
            X=self.X_clf_train,
            y=self.y_clf_train,
            param_grid={
                "n_estimators": [50, 100],
                "max_depth": [5, 10],
            },
            cv=3,
            scoring="accuracy",
        )

        # 2. Random search around best grid params
        best_n_est = grid_result.best_params["n_estimators"]
        random_result = run_random_search(
            estimator=RandomForestClassifier(random_state=42),
            X=self.X_clf_train,
            y=self.y_clf_train,
            param_distributions={
                "n_estimators": [best_n_est - 10, best_n_est, best_n_est + 10],
                "max_depth": [3, 5, 7, 10, 15],
                "min_samples_split": [2, 5, 10],
            },
            n_iter=10,
            cv=3,
            random_state=42,
        )

        # 3. Bayesian optimization for final tuning
        bayesian_result = run_bayesian_search(
            estimator=RandomForestClassifier(random_state=42),
            X=self.X_clf_train,
            y=self.y_clf_train,
            search_spaces={
                "n_estimators": [best_n_est - 20, best_n_est, best_n_est + 20],
                "max_depth": [5, 7, 10, 12, 15],
                "min_samples_split": [2, 3, 5, 7],
            },
            n_iter=8,
            cv=3,
            random_state=42,
        )

        # All searches should produce valid results
        self.assertIsNotNone(grid_result.best_params)
        self.assertIsNotNone(random_result.best_params)
        self.assertIsNotNone(bayesian_result.best_params)

        # Bayesian result should be competitive or better
        self.assertGreater(bayesian_result.best_score, 0)

    def test_model_comparison_workflow(self):
        """Test comparing multiple models with different search strategies."""
        # Define models
        models = {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=50, random_state=42
            ),
        }

        # Compare models
        comparison = compare_models_cv(
            models=models,
            X=self.X_clf,
            y=self.y_clf,
            cv=5,
            scoring="f1",
        )

        # Check all models evaluated
        self.assertEqual(len(comparison), len(models))

        # Find best model
        best_model_name = max(comparison.items(), key=lambda x: x[1].mean_score)[0]
        best_result = comparison[best_model_name]

        self.assertGreater(best_result.mean_score, 0)

        # Tune best model
        if best_model_name == "RandomForest":
            tuning_result = run_random_search(
                estimator=RandomForestClassifier(random_state=42),
                X=self.X_clf_train,
                y=self.y_clf_train,
                model_id="random_forest_classifier",
                n_iter=10,
                cv=3,
                scoring="f1",
                random_state=42,
            )
            self.assertIsNotNone(tuning_result.best_params)

    def test_regression_workflow(self):
        """Test complete workflow for regression."""
        # Grid search
        grid_result = run_grid_search(
            estimator=Ridge(),
            X=self.X_reg_train,
            y=self.y_reg_train,
            model_id="ridge_regression",
            cv=3,
            scoring="neg_mean_squared_error",
        )

        self.assertIn("alpha", grid_result.best_params)

        # Validate
        best_model = Ridge(**grid_result.best_params)
        cv_result = run_cross_validation(
            estimator=best_model,
            X=self.X_reg,
            y=self.y_reg,
            cv=5,
            scoring="r2",
        )

        self.assertIsNotNone(cv_result.mean_score)
        self.assertEqual(len(cv_result.scores), 5)

    def test_search_with_custom_and_default_spaces(self):
        """Test mixing custom and default search spaces."""
        # Get default space
        default_space = get_default_search_space("random_forest_classifier")

        # Modify it
        custom_space = default_space.copy()
        custom_space["n_estimators"] = [25, 50, 75]  # Smaller for faster testing

        # Run search
        result = run_grid_search(
            estimator=RandomForestClassifier(random_state=42),
            X=self.X_clf_train,
            y=self.y_clf_train,
            param_grid=custom_space,
            cv=3,
            scoring="accuracy",
        )

        self.assertIn(result.best_params["n_estimators"], [25, 50, 75])

    def test_parallel_execution(self):
        """Test that n_jobs parameter works."""
        # This should execute faster with parallelization
        result = run_grid_search(
            estimator=RandomForestClassifier(random_state=42),
            X=self.X_clf_train,
            y=self.y_clf_train,
            param_grid={
                "n_estimators": [10, 20],
                "max_depth": [3, 5],
            },
            cv=3,
            n_jobs=2,  # Use 2 parallel jobs
        )

        self.assertIsNotNone(result.best_params)
        self.assertGreater(result.n_candidates, 0)

    def test_multiple_scoring_metrics(self):
        """Test using multiple scoring metrics in CV."""
        cv_result = run_cross_validation(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            X=self.X_clf,
            y=self.y_clf,
            cv=5,
            scoring=["accuracy", "precision", "recall", "f1"],
        )

        # Main score is first metric
        self.assertIsNotNone(cv_result.mean_score)

        # Additional metrics available
        self.assertIn("precision", cv_result.additional_metrics)
        self.assertIn("recall", cv_result.additional_metrics)
        self.assertIn("f1", cv_result.additional_metrics)

        # Each metric should have mean/std
        for metric_data in cv_result.additional_metrics.values():
            self.assertIn("mean", metric_data)
            self.assertIn("std", metric_data)

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        cv_result = run_cross_validation(
            estimator=LogisticRegression(max_iter=200),
            X=self.X_clf,
            y=self.y_clf,
            cv=10,
            scoring="accuracy",
        )

        # Get 95% confidence interval
        ci_95 = cv_result.confidence_interval(0.95)
        self.assertIsInstance(ci_95, tuple)
        self.assertEqual(len(ci_95), 2)
        self.assertLess(ci_95[0], ci_95[1])
        self.assertLessEqual(ci_95[0], cv_result.mean_score)
        self.assertGreaterEqual(ci_95[1], cv_result.mean_score)

        # Get 99% confidence interval (should be wider)
        ci_99 = cv_result.confidence_interval(0.99)
        self.assertLess(ci_99[0], ci_95[0])
        self.assertGreater(ci_99[1], ci_95[1])


class TestTuningEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_small_dataset(self):
        """Test tuning with very small dataset."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        # Should still work but with appropriate CV folds
        result = run_grid_search(
            estimator=LogisticRegression(max_iter=200),
            X=X,
            y=y,
            param_grid={"C": [0.1, 1.0]},
            cv=3,
        )

        self.assertIsNotNone(result.best_params)

    def test_single_parameter_search(self):
        """Test search with only one parameter."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        result = run_grid_search(
            estimator=LogisticRegression(max_iter=200),
            X=X,
            y=y,
            param_grid={"C": [0.1, 1.0, 10.0]},
            cv=3,
        )

        self.assertEqual(len(result.results), 3)
        self.assertIn("C", result.best_params)

    def test_search_all_params_same(self):
        """Test when all parameter combinations give similar results."""
        X, y = make_classification(
            n_samples=100, n_features=2, n_informative=2, random_state=42
        )

        # Very similar C values
        result = run_grid_search(
            estimator=LogisticRegression(max_iter=200),
            X=X,
            y=y,
            param_grid={"C": [0.99, 1.0, 1.01]},
            cv=3,
        )

        # Should still pick a best one
        self.assertIsNotNone(result.best_params)
        self.assertIn(result.best_params["C"], [0.99, 1.0, 1.01])

    def test_deterministic_results_with_random_state(self):
        """Test that random_state makes results reproducible."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        result1 = run_random_search(
            estimator=RandomForestClassifier(),
            X=X,
            y=y,
            param_distributions={
                "n_estimators": [10, 20, 30, 40, 50],
                "max_depth": [3, 5, 7, 10],
            },
            n_iter=5,
            cv=3,
            random_state=42,
        )

        result2 = run_random_search(
            estimator=RandomForestClassifier(),
            X=X,
            y=y,
            param_distributions={
                "n_estimators": [10, 20, 30, 40, 50],
                "max_depth": [3, 5, 7, 10],
            },
            n_iter=5,
            cv=3,
            random_state=42,
        )

        # Results should be identical
        self.assertEqual(result1.best_params, result2.best_params)
        self.assertAlmostEqual(result1.best_score, result2.best_score, places=10)


class TestTuningPerformance(unittest.TestCase):
    """Test performance characteristics of tuning methods."""

    def test_random_faster_than_grid(self):
        """Verify random search evaluates fewer combinations than grid."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        param_space = {
            "n_estimators": [10, 20, 30, 40, 50],
            "max_depth": [3, 5, 7, 10, 15],
            "min_samples_split": [2, 5, 10],
        }

        # Grid search would evaluate 5 * 5 * 3 = 75 combinations
        grid_total = 75

        # Random search with n_iter=20
        random_result = run_random_search(
            estimator=RandomForestClassifier(random_state=42),
            X=X,
            y=y,
            param_distributions=param_space,
            n_iter=20,
            cv=3,
            random_state=42,
        )

        # Random search evaluated fewer
        self.assertLess(random_result.n_candidates, grid_total)
        self.assertEqual(random_result.n_candidates, 20)

    def test_cv_fold_count(self):
        """Test that CV evaluates correct number of folds."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        for n_folds in [3, 5, 10]:
            result = run_cross_validation(
                estimator=LogisticRegression(max_iter=200),
                X=X,
                y=y,
                cv=n_folds,
            )

            self.assertEqual(len(result.scores), n_folds)
            self.assertEqual(len(result.fit_times), n_folds)


if __name__ == "__main__":
    unittest.main()
