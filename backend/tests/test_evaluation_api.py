"""
Integration tests for Model Evaluation API endpoints.

Tests cover:
- GET /api/v1/models/train/{id}/metrics - Get evaluation metrics
- GET /api/v1/models/train/{id}/feature-importance - Get feature importance
- Metrics for classification, regression models
- Authorization and validation
- Cache behavior
- Error handling
"""

import pytest
import uuid
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime

from app.models.dataset import Dataset
from app.models.user import User
from app.models.experiment import Experiment
from app.models.model_run import ModelRun


@pytest.fixture
def test_user(db: Session):
    """Create a test user."""
    user = User(
        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        email="test@example.com"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_dataset(db: Session, test_user: User):
    """Create a test dataset."""
    dataset = Dataset(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="test_dataset",
        file_path="/fake/path/test.csv",
        rows=1000,
        cols=10,
        dtypes={f"col{i}": "float64" for i in range(10)},
        missing_values={f"col{i}": 0 for i in range(10)}
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@pytest.fixture
def test_experiment(db: Session, test_user: User, test_dataset: Dataset):
    """Create a test experiment."""
    experiment = Experiment(
        id=uuid.uuid4(),
        user_id=test_user.id,
        dataset_id=test_dataset.id,
        name="test_experiment",
        description="Test experiment for evaluation",
        status="running"
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


@pytest.fixture
def test_classification_model(db: Session, test_experiment: Experiment):
    """Create a completed classification model run."""
    model_run = ModelRun(
        id=uuid.uuid4(),
        experiment_id=test_experiment.id,
        model_type="random_forest_classifier",
        status="completed",
        hyperparameters={"n_estimators": 100, "max_depth": 10},
        metrics={
            "accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.89,
            "f1_score": 0.895,
            "auc_roc": 0.95,
            "confusion_matrix": [[450, 50], [55, 445]]
        },
        training_time=15.2,
        model_artifact_path="/fake/path/model.pkl",
        run_metadata={
            "feature_importance": {
                "feature_1": 0.25,
                "feature_2": 0.20,
                "feature_3": 0.15,
                "feature_4": 0.12,
                "feature_5": 0.10,
                "feature_6": 0.08,
                "feature_7": 0.05,
                "feature_8": 0.03,
                "feature_9": 0.01,
                "feature_10": 0.01
            }
        }
    )
    db.add(model_run)
    db.commit()
    db.refresh(model_run)
    return model_run


@pytest.fixture
def test_regression_model(db: Session, test_experiment: Experiment):
    """Create a completed regression model run."""
    model_run = ModelRun(
        id=uuid.uuid4(),
        experiment_id=test_experiment.id,
        model_type="linear_regression",
        status="completed",
        hyperparameters={"fit_intercept": True},
        metrics={
            "r2_score": 0.85,
            "mean_squared_error": 12.5,
            "mean_absolute_error": 2.8,
            "root_mean_squared_error": 3.54
        },
        training_time=5.1,
        model_artifact_path="/fake/path/regression_model.pkl",
        run_metadata={
            "feature_importance": {
                "feature_1": 0.35,
                "feature_2": 0.28,
                "feature_3": 0.20,
                "feature_4": 0.10,
                "feature_5": 0.07
            }
        }
    )
    db.add(model_run)
    db.commit()
    db.refresh(model_run)
    return model_run


class TestGetModelMetrics:
    """Tests for GET /api/v1/models/train/{id}/metrics endpoint"""

    def test_get_metrics_classification_success(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test getting metrics for classification model"""
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["model_run_id"] == str(test_classification_model.id)
        assert data["model_type"] == "random_forest_classifier"
        assert data["task_type"] == "classification"
        assert data["has_metrics"] is True
        
        metrics = data["metrics"]
        assert metrics["accuracy"] == 0.92
        assert metrics["precision"] == 0.90
        assert metrics["recall"] == 0.89
        assert metrics["f1_score"] == 0.895
        assert metrics["auc_roc"] == 0.95
        assert "confusion_matrix" in metrics

    def test_get_metrics_regression_success(
        self, 
        client: TestClient, 
        test_regression_model: ModelRun
    ):
        """Test getting metrics for regression model"""
        response = client.get(
            f"/api/v1/models/train/{test_regression_model.id}/metrics"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["model_run_id"] == str(test_regression_model.id)
        assert data["task_type"] == "regression"
        assert data["has_metrics"] is True
        
        metrics = data["metrics"]
        assert metrics["r2_score"] == 0.85
        assert metrics["mean_squared_error"] == 12.5
        assert metrics["mean_absolute_error"] == 2.8

    def test_get_metrics_not_found(self, client: TestClient):
        """Test getting metrics for non-existent model"""
        fake_id = uuid.uuid4()
        response = client.get(f"/api/v1/models/train/{fake_id}/metrics")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_metrics_invalid_uuid(self, client: TestClient):
        """Test getting metrics with invalid UUID"""
        response = client.get("/api/v1/models/train/not-a-uuid/metrics")

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_get_metrics_not_completed(
        self, 
        client: TestClient, 
        db: Session, 
        test_classification_model: ModelRun
    ):
        """Test getting metrics for incomplete model"""
        test_classification_model.status = "running"
        db.commit()

        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics"
        )

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"].lower()

    def test_get_metrics_no_metrics_available(
        self, 
        client: TestClient, 
        db: Session, 
        test_classification_model: ModelRun
    ):
        """Test getting metrics when no metrics data available"""
        test_classification_model.metrics = None
        db.commit()

        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics"
        )

        # Should return response indicating no metrics
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert data["has_metrics"] is False

    @patch('app.utils.cache.cache_service.get')
    @patch('app.utils.cache.cache_service.set')
    def test_get_metrics_cache_hit(
        self, 
        mock_cache_set,
        mock_cache_get, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test cache hit when retrieving metrics"""
        cached_response = {
            "model_run_id": str(test_classification_model.id),
            "model_type": "random_forest_classifier",
            "task_type": "classification",
            "has_metrics": True,
            "metrics": {"accuracy": 0.92}
        }
        mock_cache_get.return_value = cached_response

        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics"
        )

        assert response.status_code == 200
        mock_cache_get.assert_called_once()
        # Cache set should not be called on cache hit
        mock_cache_set.assert_not_called()

    @patch('app.utils.cache.cache_service.get')
    @patch('app.utils.cache.cache_service.set')
    def test_get_metrics_cache_miss(
        self, 
        mock_cache_set,
        mock_cache_get, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test cache miss when retrieving metrics"""
        mock_cache_get.return_value = None

        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics"
        )

        assert response.status_code == 200
        mock_cache_get.assert_called_once()
        # Cache set should be called on cache miss
        mock_cache_set.assert_called_once()

    def test_get_metrics_bypass_cache(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test bypassing cache with use_cache=false"""
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics?use_cache=false"
        )

        assert response.status_code == 200

    def test_get_metrics_unauthorized(
        self, 
        client: TestClient, 
        db: Session
    ):
        """Test getting metrics for model belonging to different user"""
        # Create model for different user
        other_user = User(id=uuid.uuid4(), email="other@example.com")
        db.add(other_user)
        
        other_dataset = Dataset(
            id=uuid.uuid4(),
            user_id=other_user.id,
            name="other_dataset",
            file_path="/other/path.csv",
            rows=100,
            cols=5
        )
        db.add(other_dataset)
        
        other_experiment = Experiment(
            id=uuid.uuid4(),
            user_id=other_user.id,
            dataset_id=other_dataset.id,
            name="other_experiment",
            status="running"
        )
        db.add(other_experiment)
        
        other_model = ModelRun(
            id=uuid.uuid4(),
            experiment_id=other_experiment.id,
            model_type="logistic_regression",
            status="completed",
            metrics={"accuracy": 0.8}
        )
        db.add(other_model)
        db.commit()

        response = client.get(f"/api/v1/models/train/{other_model.id}/metrics")

        assert response.status_code in [403, 404]


class TestGetFeatureImportance:
    """Tests for GET /api/v1/models/train/{id}/feature-importance endpoint"""

    def test_get_feature_importance_success(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test getting feature importance for model with feature importance"""
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["model_run_id"] == str(test_classification_model.id)
        assert data["has_feature_importance"] is True
        assert data["total_features"] == 10
        assert len(data["feature_importance"]) == 10
        
        # Check first feature has highest importance
        assert data["feature_importance"][0]["feature"] == "feature_1"
        assert data["feature_importance"][0]["importance"] == 0.25
        assert data["feature_importance"][0]["rank"] == 1
        
        # Check top features
        assert len(data["top_features"]) == 10  # Default is min(10, total)

    def test_get_feature_importance_with_top_n(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test getting top N features"""
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance?top_n=5"
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["top_features"]) == 5
        assert data["total_features"] == 10

    def test_get_feature_importance_not_available(
        self, 
        client: TestClient, 
        db: Session, 
        test_experiment: Experiment
    ):
        """Test getting feature importance for model without it"""
        # Create model without feature importance
        model_without_fi = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="k_nearest_neighbors",
            status="completed",
            hyperparameters={"n_neighbors": 5},
            metrics={"accuracy": 0.85},
            run_metadata={}
        )
        db.add(model_without_fi)
        db.commit()

        response = client.get(
            f"/api/v1/models/train/{model_without_fi.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["has_feature_importance"] is False
        assert data["feature_importance"] is None
        assert "not support" in data["message"].lower() or "not calculated" in data["message"].lower()

    def test_get_feature_importance_not_found(self, client: TestClient):
        """Test getting feature importance for non-existent model"""
        fake_id = uuid.uuid4()
        response = client.get(
            f"/api/v1/models/train/{fake_id}/feature-importance"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_feature_importance_invalid_uuid(self, client: TestClient):
        """Test getting feature importance with invalid UUID"""
        response = client.get(
            "/api/v1/models/train/not-a-uuid/feature-importance"
        )

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_get_feature_importance_not_completed(
        self, 
        client: TestClient, 
        db: Session, 
        test_classification_model: ModelRun
    ):
        """Test getting feature importance for incomplete model"""
        test_classification_model.status = "running"
        db.commit()

        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance"
        )

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"].lower()

    @patch('app.utils.cache.cache_service.get')
    @patch('app.utils.cache.cache_service.set')
    def test_get_feature_importance_cache_hit(
        self, 
        mock_cache_set,
        mock_cache_get, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test cache hit for feature importance"""
        cached_response = {
            "model_run_id": str(test_classification_model.id),
            "has_feature_importance": True,
            "feature_importance": []
        }
        mock_cache_get.return_value = cached_response

        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance"
        )

        assert response.status_code == 200
        mock_cache_get.assert_called_once()
        mock_cache_set.assert_not_called()

    def test_get_feature_importance_bypass_cache(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test bypassing cache"""
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance?use_cache=false"
        )

        assert response.status_code == 200

    def test_get_feature_importance_different_top_n_cached_separately(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test that different top_n values are cached separately"""
        # First request with top_n=5
        response1 = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance?top_n=5"
        )
        assert response1.status_code == 200
        
        # Second request with top_n=3
        response2 = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance?top_n=3"
        )
        assert response2.status_code == 200
        
        # Should have different number of top features
        assert len(response1.json()["top_features"]) == 5
        assert len(response2.json()["top_features"]) == 3


class TestEvaluationMetricsDetails:
    """Tests for detailed metrics verification"""

    def test_classification_metrics_structure(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test structure of classification metrics response"""
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics"
        )

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        required_fields = [
            "model_run_id",
            "model_type",
            "task_type",
            "has_metrics",
            "metrics"
        ]
        for field in required_fields:
            assert field in data

        # Verify classification-specific metrics
        if data["has_metrics"]:
            metrics = data["metrics"]
            assert "accuracy" in metrics
            assert isinstance(metrics["accuracy"], (int, float))

    def test_regression_metrics_structure(
        self, 
        client: TestClient, 
        test_regression_model: ModelRun
    ):
        """Test structure of regression metrics response"""
        response = client.get(
            f"/api/v1/models/train/{test_regression_model.id}/metrics"
        )

        assert response.status_code == 200
        data = response.json()

        # Verify regression-specific metrics
        if data["has_metrics"]:
            metrics = data["metrics"]
            regression_metrics = ["r2_score", "mean_squared_error", "mean_absolute_error"]
            # At least one regression metric should be present
            assert any(metric in metrics for metric in regression_metrics)

    def test_feature_importance_ranking(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test that feature importance is properly ranked"""
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()

        if data["has_feature_importance"]:
            features = data["feature_importance"]
            
            # Check features are sorted by importance (descending)
            importances = [f["importance"] for f in features]
            assert importances == sorted(importances, reverse=True)
            
            # Check ranks are sequential
            ranks = [f["rank"] for f in features]
            assert ranks == list(range(1, len(ranks) + 1))


class TestEvaluationEdgeCases:
    """Tests for edge cases in evaluation endpoints"""

    def test_get_metrics_empty_metrics_dict(
        self, 
        client: TestClient, 
        db: Session, 
        test_classification_model: ModelRun
    ):
        """Test getting metrics when metrics dict is empty"""
        test_classification_model.metrics = {}
        db.commit()

        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics"
        )

        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert data["has_metrics"] is False or data["metrics"] == {}

    def test_get_feature_importance_empty_dict(
        self, 
        client: TestClient, 
        db: Session, 
        test_classification_model: ModelRun
    ):
        """Test getting feature importance when dict is empty"""
        test_classification_model.run_metadata = {"feature_importance": {}}
        db.commit()

        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["has_feature_importance"] is False
        assert data["total_features"] == 0

    def test_get_feature_importance_top_n_exceeds_total(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test requesting more features than available"""
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/feature-importance?top_n=100"
        )

        assert response.status_code == 200
        data = response.json()

        # Should return all available features
        assert len(data["top_features"]) == data["total_features"]

    def test_get_metrics_failed_model(
        self, 
        client: TestClient, 
        db: Session, 
        test_experiment: Experiment
    ):
        """Test getting metrics for failed model"""
        failed_model = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest_classifier",
            status="failed",
            run_metadata={"error": "Training failed"}
        )
        db.add(failed_model)
        db.commit()

        response = client.get(f"/api/v1/models/train/{failed_model.id}/metrics")

        # Should return appropriate error
        assert response.status_code in [200, 400, 404]


class TestEvaluationPerformance:
    """Tests for performance characteristics of evaluation endpoints"""

    def test_get_metrics_response_time(
        self, 
        client: TestClient, 
        test_classification_model: ModelRun
    ):
        """Test that metrics endpoint responds quickly"""
        import time
        
        start = time.time()
        response = client.get(
            f"/api/v1/models/train/{test_classification_model.id}/metrics"
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should respond within 2 seconds (generous for test environment)
        assert elapsed < 2.0

    def test_get_feature_importance_with_many_features(
        self, 
        client: TestClient, 
        db: Session, 
        test_experiment: Experiment
    ):
        """Test feature importance with large number of features"""
        # Create model with 1000 features
        large_fi = {f"feature_{i}": 1.0 / (i + 1) for i in range(1000)}
        
        large_model = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest_classifier",
            status="completed",
            metrics={"accuracy": 0.9},
            run_metadata={"feature_importance": large_fi}
        )
        db.add(large_model)
        db.commit()

        response = client.get(
            f"/api/v1/models/train/{large_model.id}/feature-importance?top_n=20"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_features"] == 1000
        assert len(data["top_features"]) == 20
