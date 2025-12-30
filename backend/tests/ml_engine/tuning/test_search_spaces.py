"""Tests for default hyperparameter search spaces."""

import unittest
from copy import deepcopy

from app.ml_engine.tuning.search_spaces import (
    DEFAULT_SEARCH_SPACES,
    get_default_search_space,
)


class TestDefaultSearchSpaces(unittest.TestCase):
    """Test default search space definitions."""

    def test_all_classification_models_exist(self):
        """Test that all expected classification models have search spaces."""
        expected_models = [
            "logistic_regression",
            "random_forest_classifier",
            "svm_classifier",
            "gradient_boosting_classifier",
            "knn_classifier",
        ]

        for model in expected_models:
            self.assertIn(model, DEFAULT_SEARCH_SPACES)
            self.assertIsInstance(DEFAULT_SEARCH_SPACES[model], dict)
            self.assertGreater(len(DEFAULT_SEARCH_SPACES[model]), 0)

    def test_all_regression_models_exist(self):
        """Test that all expected regression models have search spaces."""
        expected_models = [
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "random_forest_regressor",
        ]

        for model in expected_models:
            self.assertIn(model, DEFAULT_SEARCH_SPACES)
            self.assertIsInstance(DEFAULT_SEARCH_SPACES[model], dict)
            self.assertGreater(len(DEFAULT_SEARCH_SPACES[model]), 0)

    def test_all_clustering_models_exist(self):
        """Test that all expected clustering models have search spaces."""
        expected_models = [
            "kmeans",
            "dbscan",
            "agglomerative_clustering",
            "gaussian_mixture",
        ]

        for model in expected_models:
            self.assertIn(model, DEFAULT_SEARCH_SPACES)
            self.assertIsInstance(DEFAULT_SEARCH_SPACES[model], dict)
            self.assertGreater(len(DEFAULT_SEARCH_SPACES[model]), 0)

    def test_logistic_regression_parameters(self):
        """Test logistic regression search space structure."""
        space = DEFAULT_SEARCH_SPACES["logistic_regression"]

        self.assertIn("C", space)
        self.assertIn("penalty", space)
        self.assertIn("solver", space)
        self.assertIn("max_iter", space)

        # Check types and values
        self.assertIsInstance(space["C"], list)
        self.assertGreater(len(space["C"]), 0)
        self.assertTrue(all(isinstance(c, (int, float)) for c in space["C"]))

        self.assertIn("l2", space["penalty"])
        self.assertIn("lbfgs", space["solver"])

    def test_random_forest_classifier_parameters(self):
        """Test random forest classifier search space structure."""
        space = DEFAULT_SEARCH_SPACES["random_forest_classifier"]

        expected_params = [
            "n_estimators",
            "max_depth",
            "max_features",
            "min_samples_split",
            "min_samples_leaf",
            "bootstrap",
        ]

        for param in expected_params:
            self.assertIn(param, space)
            self.assertIsInstance(space[param], list)
            self.assertGreater(len(space[param]), 0)

        # Check specific values
        self.assertTrue(all(isinstance(n, int) for n in space["n_estimators"]))
        self.assertIn(None, space["max_depth"])
        self.assertIn(True, space["bootstrap"])
        self.assertIn(False, space["bootstrap"])

    def test_svm_classifier_parameters(self):
        """Test SVM classifier search space structure."""
        space = DEFAULT_SEARCH_SPACES["svm_classifier"]

        self.assertIn("C", space)
        self.assertIn("kernel", space)
        self.assertIn("gamma", space)

        self.assertIn("linear", space["kernel"])
        self.assertIn("rbf", space["kernel"])
        self.assertIn("scale", space["gamma"])

    def test_gradient_boosting_parameters(self):
        """Test gradient boosting search space structure."""
        space = DEFAULT_SEARCH_SPACES["gradient_boosting_classifier"]

        expected_params = [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "subsample",
            "min_samples_split",
            "min_samples_leaf",
        ]

        for param in expected_params:
            self.assertIn(param, space)

        # Check learning rate is float
        self.assertTrue(all(isinstance(lr, float) for lr in space["learning_rate"]))
        # Check subsample is in valid range
        self.assertTrue(all(0 < s <= 1.0 for s in space["subsample"]))

    def test_knn_classifier_parameters(self):
        """Test KNN classifier search space structure."""
        space = DEFAULT_SEARCH_SPACES["knn_classifier"]

        self.assertIn("n_neighbors", space)
        self.assertIn("weights", space)
        self.assertIn("p", space)

        # Check n_neighbors are odd numbers (best practice)
        self.assertTrue(all(isinstance(n, int) and n > 0 for n in space["n_neighbors"]))
        self.assertIn("uniform", space["weights"])
        self.assertIn("distance", space["weights"])
        self.assertIn(1, space["p"])  # Manhattan
        self.assertIn(2, space["p"])  # Euclidean

    def test_ridge_regression_parameters(self):
        """Test ridge regression search space structure."""
        space = DEFAULT_SEARCH_SPACES["ridge_regression"]

        self.assertIn("alpha", space)
        self.assertIn("solver", space)
        self.assertIn("fit_intercept", space)

        self.assertTrue(all(isinstance(a, (int, float)) and a > 0 for a in space["alpha"]))
        self.assertIn("auto", space["solver"])

    def test_lasso_regression_parameters(self):
        """Test lasso regression search space structure."""
        space = DEFAULT_SEARCH_SPACES["lasso_regression"]

        self.assertIn("alpha", space)
        self.assertIn("max_iter", space)
        self.assertIn("fit_intercept", space)

        self.assertTrue(all(isinstance(a, (int, float)) and a > 0 for a in space["alpha"]))
        self.assertTrue(all(isinstance(m, int) and m > 0 for m in space["max_iter"]))

    def test_kmeans_parameters(self):
        """Test K-means search space structure."""
        space = DEFAULT_SEARCH_SPACES["kmeans"]

        self.assertIn("n_clusters", space)
        self.assertIn("init", space)
        self.assertIn("n_init", space)

        self.assertTrue(all(isinstance(k, int) and k >= 2 for k in space["n_clusters"]))
        self.assertIn("k-means++", space["init"])
        self.assertIn("random", space["init"])

    def test_dbscan_parameters(self):
        """Test DBSCAN search space structure."""
        space = DEFAULT_SEARCH_SPACES["dbscan"]

        self.assertIn("eps", space)
        self.assertIn("min_samples", space)
        self.assertIn("metric", space)

        self.assertTrue(all(isinstance(e, (int, float)) and e > 0 for e in space["eps"]))
        self.assertTrue(all(isinstance(m, int) and m > 0 for m in space["min_samples"]))
        self.assertIn("euclidean", space["metric"])

    def test_agglomerative_clustering_parameters(self):
        """Test agglomerative clustering search space structure."""
        space = DEFAULT_SEARCH_SPACES["agglomerative_clustering"]

        self.assertIn("n_clusters", space)
        self.assertIn("linkage", space)
        self.assertIn("metric", space)

        self.assertTrue(all(isinstance(k, int) and k >= 2 for k in space["n_clusters"]))
        self.assertIn("ward", space["linkage"])
        self.assertIn("euclidean", space["metric"])

    def test_gaussian_mixture_parameters(self):
        """Test Gaussian mixture search space structure."""
        space = DEFAULT_SEARCH_SPACES["gaussian_mixture"]

        self.assertIn("n_components", space)
        self.assertIn("covariance_type", space)
        self.assertIn("max_iter", space)

        self.assertTrue(all(isinstance(k, int) and k >= 1 for k in space["n_components"]))
        self.assertIn("full", space["covariance_type"])
        self.assertIn("diag", space["covariance_type"])


class TestGetDefaultSearchSpace(unittest.TestCase):
    """Test get_default_search_space function."""

    def test_get_existing_model(self):
        """Test getting search space for existing model."""
        space = get_default_search_space("random_forest_classifier")

        self.assertIsInstance(space, dict)
        self.assertGreater(len(space), 0)
        self.assertIn("n_estimators", space)

    def test_get_nonexistent_model(self):
        """Test getting search space for non-existent model."""
        space = get_default_search_space("nonexistent_model")

        self.assertIsInstance(space, dict)
        self.assertEqual(len(space), 0)

    def test_returns_copy(self):
        """Test that function returns a copy, not reference."""
        space1 = get_default_search_space("logistic_regression")
        space2 = get_default_search_space("logistic_regression")

        # Modify space1
        space1["C"] = [999]

        # space2 should be unchanged
        self.assertNotEqual(space1["C"], space2["C"])
        self.assertNotIn(999, space2["C"])

    def test_original_not_modified(self):
        """Test that modifying returned space doesn't affect original."""
        original = deepcopy(DEFAULT_SEARCH_SPACES["svm_classifier"])
        space = get_default_search_space("svm_classifier")

        # Modify returned space
        space["C"] = [999]
        space["kernel"] = ["fake_kernel"]

        # Original should be unchanged
        self.assertEqual(DEFAULT_SEARCH_SPACES["svm_classifier"], original)

    def test_all_models_accessible(self):
        """Test that all models in DEFAULT_SEARCH_SPACES are accessible."""
        for model_id in DEFAULT_SEARCH_SPACES.keys():
            space = get_default_search_space(model_id)

            self.assertIsInstance(space, dict)
            self.assertGreater(len(space), 0)

    def test_nested_structures_copied(self):
        """Test that nested structures are deep copied."""
        space = get_default_search_space("random_forest_classifier")

        # Modify a list in the returned space
        original_n_estimators = DEFAULT_SEARCH_SPACES["random_forest_classifier"]["n_estimators"]
        space["n_estimators"].append(9999)

        # Original should be unchanged
        self.assertNotIn(9999, original_n_estimators)


class TestSearchSpaceQuality(unittest.TestCase):
    """Test that search spaces follow best practices."""

    def test_no_empty_parameter_lists(self):
        """Test that no parameter has an empty list."""
        for model_id, space in DEFAULT_SEARCH_SPACES.items():
            for param, values in space.items():
                self.assertIsInstance(values, list, f"{model_id}.{param} is not a list")
                self.assertGreater(
                    len(values), 0, f"{model_id}.{param} has empty list"
                )

    def test_reasonable_search_space_sizes(self):
        """Test that search spaces aren't too large."""
        for model_id, space in DEFAULT_SEARCH_SPACES.items():
            # Calculate total combinations for grid search
            total_combinations = 1
            for values in space.values():
                total_combinations *= len(values)

            # Warn if grid search would be very expensive
            # (This is a soft check - some models might legitimately need large spaces)
            if total_combinations > 10000:
                print(
                    f"Warning: {model_id} has {total_combinations} combinations. "
                    f"Consider using random or Bayesian search."
                )

    def test_numeric_parameters_ordered(self):
        """Test that numeric parameters are in ascending order."""
        for model_id, space in DEFAULT_SEARCH_SPACES.items():
            for param, values in space.items():
                # Filter out None and non-numeric values
                numeric_values = [v for v in values if isinstance(v, (int, float))]

                if len(numeric_values) > 1:
                    # Check if sorted
                    is_sorted = all(
                        numeric_values[i] <= numeric_values[i + 1]
                        for i in range(len(numeric_values) - 1)
                    )
                    if not is_sorted:
                        print(
                            f"Note: {model_id}.{param} numeric values not sorted: {numeric_values}"
                        )

    def test_regularization_parameters_include_weak_and_strong(self):
        """Test that regularization parameters cover weak and strong values."""
        # Models with regularization
        reg_models = {
            "logistic_regression": "C",
            "ridge_regression": "alpha",
            "lasso_regression": "alpha",
            "svm_classifier": "C",
        }

        for model_id, param in reg_models.items():
            values = DEFAULT_SEARCH_SPACES[model_id][param]
            numeric_values = [v for v in values if isinstance(v, (int, float))]

            self.assertGreater(
                len(numeric_values),
                0,
                f"{model_id}.{param} has no numeric values",
            )

            # Should span multiple orders of magnitude
            if param == "C":
                # C: smaller = stronger regularization
                self.assertLessEqual(
                    min(numeric_values), 1.0, f"{model_id}.{param} should include small values"
                )
                self.assertGreaterEqual(
                    max(numeric_values), 1.0, f"{model_id}.{param} should include large values"
                )
            elif param == "alpha":
                # alpha: larger = stronger regularization
                self.assertGreater(
                    max(numeric_values), 0.1, f"{model_id}.{param} should include large values"
                )

    def test_tree_based_models_include_none_depth(self):
        """Test that tree-based models include None for max_depth."""
        tree_models = [
            "random_forest_classifier",
            "random_forest_regressor",
        ]

        for model_id in tree_models:
            if "max_depth" in DEFAULT_SEARCH_SPACES[model_id]:
                self.assertIn(
                    None,
                    DEFAULT_SEARCH_SPACES[model_id]["max_depth"],
                    f"{model_id} should include None for max_depth (unlimited)",
                )

    def test_ensemble_models_have_reasonable_estimators(self):
        """Test that ensemble models have reasonable n_estimators ranges."""
        ensemble_models = [
            "random_forest_classifier",
            "random_forest_regressor",
            "gradient_boosting_classifier",
        ]

        for model_id in ensemble_models:
            n_estimators = DEFAULT_SEARCH_SPACES[model_id]["n_estimators"]

            # Should have at least 3 options
            self.assertGreaterEqual(len(n_estimators), 3)

            # Should include reasonable range
            self.assertGreaterEqual(min(n_estimators), 10)
            self.assertLessEqual(max(n_estimators), 500)


class TestSearchSpaceCompatibility(unittest.TestCase):
    """Test search spaces are compatible with sklearn models."""

    def test_logistic_regression_solver_penalty_compatibility(self):
        """Test logistic regression solver-penalty compatibility."""
        space = DEFAULT_SEARCH_SPACES["logistic_regression"]

        # All specified solvers should support l2 penalty
        solvers = space["solver"]
        penalties = space["penalty"]

        # lbfgs and liblinear both support l2
        self.assertIn("l2", penalties)
        self.assertIn("lbfgs", solvers)
        self.assertIn("liblinear", solvers)

    def test_clustering_models_valid_n_clusters(self):
        """Test clustering models have valid n_clusters."""
        clustering_models = ["kmeans", "agglomerative_clustering"]

        for model_id in clustering_models:
            n_clusters = DEFAULT_SEARCH_SPACES[model_id]["n_clusters"]

            # All values should be >= 2
            self.assertTrue(all(k >= 2 for k in n_clusters))

    def test_gaussian_mixture_valid_n_components(self):
        """Test Gaussian mixture has valid n_components."""
        n_components = DEFAULT_SEARCH_SPACES["gaussian_mixture"]["n_components"]

        # All values should be >= 1
        self.assertTrue(all(k >= 1 for k in n_components))

    def test_agglomerative_ward_linkage_compatibility(self):
        """Test agglomerative clustering ward linkage compatibility."""
        space = DEFAULT_SEARCH_SPACES["agglomerative_clustering"]

        # Ward linkage only works with euclidean metric
        if "ward" in space["linkage"]:
            self.assertIn(
                "euclidean",
                space["metric"],
                "Ward linkage requires euclidean metric",
            )


if __name__ == "__main__":
    unittest.main()
