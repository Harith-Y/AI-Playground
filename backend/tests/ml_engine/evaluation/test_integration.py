"""Integration tests for evaluation metrics utilities.

Tests cover end-to-end workflows combining multiple evaluation components.
"""

import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from app.ml_engine.evaluation import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_roc_curve,
    calculate_pr_curve,
    calculate_regression_metrics,
    calculate_residual_analysis,
    calculate_actual_vs_predicted,
    calculate_clustering_metrics,
    calculate_feature_importance,
    calculate_permutation_importance,
)


class TestClassificationEvaluationWorkflow(unittest.TestCase):
    """Test complete evaluation workflow for classification."""

    @classmethod
    def setUpClass(cls):
        """Set up classification test data."""
        cls.X, cls.y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_classes=3,
            random_state=42,
        )
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=42
        )

    def test_complete_binary_classification_evaluation(self):
        """Test complete evaluation pipeline for binary classification."""
        # Create binary dataset
        X_bin, y_bin = make_classification(
            n_samples=300, n_features=10, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_bin, y_bin, test_size=0.3, random_state=42
        )

        # Train model
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # 1. Classification metrics
        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        self.assertGreater(metrics.accuracy, 0.5)
        self.assertGreater(metrics.precision, 0)
        self.assertGreater(metrics.recall, 0)
        self.assertGreater(metrics.f1_score, 0)
        self.assertIsNotNone(metrics.roc_auc)

        # 2. Confusion matrix
        cm = calculate_confusion_matrix(y_test, y_pred)
        self.assertEqual(cm.matrix.shape, (2, 2))
        self.assertEqual(cm.matrix.sum(), len(y_test))

        # 3. ROC curve
        roc = calculate_roc_curve(y_test, y_pred_proba[:, 1])
        self.assertGreater(len(roc.fpr), 0)
        self.assertGreater(len(roc.tpr), 0)
        self.assertGreaterEqual(roc.auc, 0.5)

        # 4. PR curve
        pr = calculate_pr_curve(y_test, y_pred_proba[:, 1])
        self.assertGreater(len(pr.precision), 0)
        self.assertGreater(len(pr.recall), 0)
        self.assertGreater(pr.average_precision, 0)

        # 5. Feature importance
        importance = calculate_feature_importance(model, X_test, y_test)
        self.assertEqual(importance.n_features, X_test.shape[1])
        self.assertGreater(len(importance.importances), 0)

    def test_complete_multiclass_classification_evaluation(self):
        """Test complete evaluation pipeline for multi-class classification."""
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)

        # 1. Classification metrics
        metrics = calculate_classification_metrics(self.y_test, y_pred, y_pred_proba)
        self.assertGreater(metrics.accuracy, 0.3)  # 3 classes, so > random
        self.assertIsNotNone(metrics.precision)
        self.assertIsNotNone(metrics.recall)
        self.assertIsNotNone(metrics.f1_score)

        # 2. Confusion matrix (3x3)
        cm = calculate_confusion_matrix(self.y_test, y_pred)
        self.assertEqual(cm.matrix.shape, (3, 3))
        self.assertEqual(cm.matrix.sum(), len(self.y_test))

        # Diagonal should have most values
        diagonal_sum = np.trace(cm.matrix)
        off_diagonal_sum = cm.matrix.sum() - diagonal_sum
        self.assertGreater(diagonal_sum, off_diagonal_sum * 0.5)

        # 3. Per-class metrics
        per_class = cm.per_class_metrics()
        self.assertEqual(len(per_class), 3)
        for class_metrics in per_class:
            self.assertIn("precision", class_metrics)
            self.assertIn("recall", class_metrics)
            self.assertIn("f1_score", class_metrics)

        # 4. Feature importance (native)
        importance = calculate_feature_importance(model, self.X_test, self.y_test)
        self.assertEqual(importance.method, "feature_importances_")
        self.assertEqual(importance.n_features, self.X_test.shape[1])

        # 5. Permutation importance
        perm_importance = calculate_permutation_importance(
            model, self.X_test, self.y_test, n_repeats=5, random_state=42
        )
        self.assertEqual(perm_importance.method, "permutation")
        self.assertGreater(len(perm_importance.importances), 0)

    def test_model_comparison_evaluation(self):
        """Test comparing evaluations of multiple models."""
        models = {
            "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=30, random_state=42),
        }

        results = {}

        for name, model in models.items():
            # Train
            model.fit(self.X_train, self.y_train)

            # Predict
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)

            # Evaluate
            metrics = calculate_classification_metrics(
                self.y_test, y_pred, y_pred_proba
            )
            cm = calculate_confusion_matrix(self.y_test, y_pred)

            results[name] = {
                "accuracy": metrics.accuracy,
                "f1_score": metrics.f1_score,
                "confusion_matrix": cm.matrix,
            }

        # Both models should have valid results
        for name, result in results.items():
            self.assertGreater(result["accuracy"], 0)
            self.assertGreater(result["f1_score"], 0)
            self.assertEqual(result["confusion_matrix"].shape, (3, 3))

        # Can compare accuracies
        lr_acc = results["LogisticRegression"]["accuracy"]
        rf_acc = results["RandomForest"]["accuracy"]
        self.assertIsNotNone(lr_acc)
        self.assertIsNotNone(rf_acc)


class TestRegressionEvaluationWorkflow(unittest.TestCase):
    """Test complete evaluation workflow for regression."""

    @classmethod
    def setUpClass(cls):
        """Set up regression test data."""
        cls.X, cls.y = make_regression(
            n_samples=300, n_features=15, n_informative=10, noise=10, random_state=42
        )
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=42
        )

    def test_complete_regression_evaluation(self):
        """Test complete evaluation pipeline for regression."""
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)

        # 1. Regression metrics
        metrics = calculate_regression_metrics(self.y_test, y_pred)
        self.assertGreater(metrics.r2_score, 0.5)  # Should explain >50% variance
        self.assertGreater(metrics.mae, 0)
        self.assertGreater(metrics.mse, 0)
        self.assertGreater(metrics.rmse, 0)
        self.assertAlmostEqual(metrics.rmse, np.sqrt(metrics.mse), places=5)

        # 2. Residual analysis
        residuals = calculate_residual_analysis(self.y_test, y_pred)
        self.assertEqual(len(residuals.residuals), len(self.y_test))
        self.assertIsNotNone(residuals.mean)
        self.assertIsNotNone(residuals.std)

        # Mean residual should be close to 0
        self.assertAlmostEqual(residuals.mean, 0, delta=5)

        # 3. Actual vs predicted
        avp = calculate_actual_vs_predicted(self.y_test, y_pred)
        self.assertEqual(len(avp.actual), len(self.y_test))
        self.assertEqual(len(avp.predicted), len(self.y_test))
        self.assertGreater(avp.r2_score, 0)

        # 4. Feature importance
        importance = calculate_feature_importance(model, self.X_test, self.y_test)
        self.assertEqual(importance.n_features, self.X_test.shape[1])
        self.assertEqual(importance.method, "feature_importances_")

        # Top features should have higher importance
        ranked = importance.to_ranked_list()
        self.assertGreater(ranked[0]["importance"], ranked[-1]["importance"])

    def test_regression_with_poor_model(self):
        """Test evaluation with intentionally poor predictions."""
        # Train model
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # Make predictions worse
        y_pred_bad = y_pred + np.random.randn(len(y_pred)) * 50

        # Calculate metrics
        metrics = calculate_regression_metrics(self.y_test, y_pred_bad)

        # Poor model should have low R2, high errors
        self.assertLess(metrics.r2_score, 0.9)  # Worse than good model
        self.assertGreater(metrics.mae, 0)

    def test_regression_model_comparison(self):
        """Test comparing multiple regression models."""
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=30, random_state=42),
        }

        results = {}

        for name, model in models.items():
            # Train
            model.fit(self.X_train, self.y_train)

            # Predict
            y_pred = model.predict(self.X_test)

            # Evaluate
            metrics = calculate_regression_metrics(self.y_test, y_pred)

            results[name] = {
                "r2_score": metrics.r2_score,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
            }

        # Both models should have valid results
        for name, result in results.items():
            self.assertGreater(result["r2_score"], 0)
            self.assertGreater(result["rmse"], 0)
            self.assertGreater(result["mae"], 0)


class TestClusteringEvaluationWorkflow(unittest.TestCase):
    """Test complete evaluation workflow for clustering."""

    @classmethod
    def setUpClass(cls):
        """Set up clustering test data."""
        cls.X, cls.y_true = make_blobs(
            n_samples=300, n_features=10, centers=4, random_state=42
        )

    def test_complete_clustering_evaluation(self):
        """Test complete evaluation pipeline for clustering."""
        # Train model
        model = KMeans(n_clusters=4, random_state=42)
        labels = model.fit_predict(self.X)

        # 1. Clustering metrics (with ground truth)
        metrics_with_truth = calculate_clustering_metrics(
            self.X, labels, ground_truth=self.y_true
        )

        self.assertIsNotNone(metrics_with_truth.silhouette_score)
        self.assertIsNotNone(metrics_with_truth.calinski_harabasz_score)
        self.assertIsNotNone(metrics_with_truth.davies_bouldin_score)
        self.assertIsNotNone(metrics_with_truth.adjusted_rand_score)
        self.assertIsNotNone(metrics_with_truth.normalized_mutual_info)

        # Silhouette should be positive for well-separated clusters
        self.assertGreater(metrics_with_truth.silhouette_score, 0)

        # 2. Clustering metrics (without ground truth)
        metrics_no_truth = calculate_clustering_metrics(self.X, labels)

        self.assertIsNotNone(metrics_no_truth.silhouette_score)
        self.assertIsNotNone(metrics_no_truth.calinski_harabasz_score)
        self.assertIsNone(metrics_no_truth.adjusted_rand_score)  # No ground truth

    def test_clustering_quality_comparison(self):
        """Test comparing different cluster counts."""
        results = {}

        for n_clusters in [2, 3, 4, 5]:
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(self.X)

            metrics = calculate_clustering_metrics(self.X, labels, ground_truth=self.y_true)

            results[n_clusters] = {
                "silhouette": metrics.silhouette_score,
                "calinski_harabasz": metrics.calinski_harabasz_score,
                "davies_bouldin": metrics.davies_bouldin_score,
                "ari": metrics.adjusted_rand_score,
            }

        # True number of clusters (4) should have best ARI
        ari_4 = results[4]["ari"]
        for n_clusters, metrics in results.items():
            if n_clusters != 4:
                # k=4 should have higher or equal ARI
                self.assertGreaterEqual(ari_4, metrics["ari"] * 0.8)


class TestCrossMetricIntegration(unittest.TestCase):
    """Test integration between different metric types."""

    def test_classification_to_clustering_pipeline(self):
        """Test using clustering for feature engineering before classification."""
        # Generate data
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5, random_state=42
        )

        # 1. Cluster to find data structure
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X)

        # Evaluate clustering
        cluster_metrics = calculate_clustering_metrics(X, cluster_labels)
        self.assertGreater(cluster_metrics.silhouette_score, -1)

        # 2. Use cluster labels as features
        X_with_clusters = np.column_stack([X, cluster_labels])

        # 3. Train classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_clusters, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate classification
        y_pred = model.predict(X_test)
        clf_metrics = calculate_classification_metrics(y_test, y_pred)

        self.assertGreater(clf_metrics.accuracy, 0.5)

    def test_feature_importance_guides_feature_selection(self):
        """Test using feature importance for feature selection."""
        # Generate data with some irrelevant features
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=5,
            n_redundant=5,
            n_repeated=5,
            random_state=42,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 1. Train initial model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Get baseline performance
        y_pred_base = model.predict(X_test)
        metrics_base = calculate_classification_metrics(y_test, y_pred_base)

        # 2. Get feature importance
        importance = calculate_feature_importance(model, X_train, y_train)
        ranked = importance.to_ranked_list()

        # 3. Select top features
        top_k = 10
        top_features = [item["feature_index"] for item in ranked[:top_k]]

        X_train_selected = X_train[:, top_features]
        X_test_selected = X_test[:, top_features]

        # 4. Train with selected features
        model_selected = RandomForestClassifier(n_estimators=50, random_state=42)
        model_selected.fit(X_train_selected, y_train)

        y_pred_selected = model_selected.predict(X_test_selected)
        metrics_selected = calculate_classification_metrics(y_test, y_pred_selected)

        # Selected features should maintain performance
        self.assertGreater(metrics_selected.accuracy, metrics_base.accuracy * 0.8)

    def test_residual_analysis_identifies_outliers(self):
        """Test using residual analysis to identify outliers."""
        # Generate regression data
        X, y = make_regression(n_samples=200, n_features=10, noise=5, random_state=42)

        # Add some outliers
        outlier_indices = [0, 1, 2]
        y[outlier_indices] += 100

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Analyze residuals
        residuals = calculate_residual_analysis(y_test, y_pred)

        # Should be able to identify outliers using residuals
        outlier_threshold = residuals.mean + 3 * residuals.std
        large_residuals = np.abs(residuals.residuals) > outlier_threshold

        # There should be some outliers detected
        self.assertGreater(large_residuals.sum(), 0)


class TestMetricSerializability(unittest.TestCase):
    """Test that all metric results are serializable (for API responses)."""

    def test_classification_metrics_to_dict(self):
        """Test classification metrics can be converted to dict."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1]

        metrics = calculate_classification_metrics(y_true, y_pred)
        metrics_dict = metrics.to_dict()

        self.assertIsInstance(metrics_dict, dict)
        self.assertIn("accuracy", metrics_dict)
        self.assertIn("precision", metrics_dict)

    def test_regression_metrics_to_dict(self):
        """Test regression metrics can be converted to dict."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = calculate_regression_metrics(y_true, y_pred)
        metrics_dict = metrics.to_dict()

        self.assertIsInstance(metrics_dict, dict)
        self.assertIn("r2_score", metrics_dict)
        self.assertIn("mse", metrics_dict)

    def test_confusion_matrix_to_dict(self):
        """Test confusion matrix can be converted to dict."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1]

        cm = calculate_confusion_matrix(y_true, y_pred)
        cm_dict = cm.to_dict()

        self.assertIsInstance(cm_dict, dict)
        self.assertIn("matrix", cm_dict)

    def test_feature_importance_to_dict(self):
        """Test feature importance can be converted to dict."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

        importance = calculate_feature_importance(model, X, y)
        importance_dict = importance.to_dict()

        self.assertIsInstance(importance_dict, dict)
        self.assertIn("importances", importance_dict)
        self.assertIn("method", importance_dict)


if __name__ == "__main__":
    unittest.main()
