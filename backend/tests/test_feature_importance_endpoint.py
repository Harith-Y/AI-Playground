"""
Tests for feature importance endpoint

Tests the /models/train/{model_run_id}/feature-importance endpoint
"""

import pytest
import uuid
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.models.user import User
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.model_run import ModelRun


client = TestClient(app)


@pytest.fixture
def test_user(db_session: Session):
    """Create a test user"""
    user = User(
        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        email="test@example.com",
        hashed_password="hashed_password",
        created_at=datetime.utcnow()
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_dataset(db_session: Session, test_user: User):
    """Create a test dataset"""
    dataset = Dataset(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="test_dataset.csv",
        file_path="/uploads/test_dataset.csv",
        file_size=1024,
        row_count=100,
        column_count=5,
        columns=["feature1", "feature2", "feature3", "feature4", "target"],
        dtypes={"feature1": "float64", "feature2": "float64", "feature3": "float64", "feature4": "float64", "target": "int64"},
        missing_values={},
        created_at=datetime.utcnow()
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


@pytest.fixture
def test_experiment(db_session: Session, test_user: User, test_dataset: Dataset):
    """Create a test experiment"""
    experiment = Experiment(
        id=uuid.uuid4(),
        user_id=test_user.id,
        dataset_id=test_dataset.id,
        name="Test Experiment",
        description="Test experiment for feature importance",
        status="running",
        created_at=datetime.utcnow()
    )
    db_session.add(experiment)
    db_session.commit()
    db_session.refresh(experiment)
    return experiment


@pytest.fixture
def completed_model_run_with_importance(db_session: Session, test_experiment: Experiment):
    """Create a completed model run with feature importance"""
    model_run = ModelRun(
        id=uuid.uuid4(),
        experiment_id=test_experiment.id,
        model_type="random_forest_classifier",
        hyperparameters={"n_estimators": 100, "max_depth": 10},
        status="completed",
        metrics={
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935
        },
        training_time=45.5,
        model_artifact_path="/models/model_123.pkl",
        run_metadata={
            "feature_importance": {
                "feature1": 0.35,
                "feature2": 0.30,
                "feature3": 0.25,
                "feature4": 0.10
            },
            "train_samples": 80,
            "test_samples": 20,
            "n_features": 4
        },
        created_at=datetime.utcnow()
    )
    db_session.add(model_run)
    db_session.commit()
    db_session.refresh(model_run)
    return model_run


@pytest.fixture
def completed_model_run_without_importance(db_session: Session, test_experiment: Experiment):
    """Create a completed model run without feature importance"""
    model_run = ModelRun(
        id=uuid.uuid4(),
        experiment_id=test_experiment.id,
        model_type="knn_classifier",
        hyperparameters={"n_neighbors": 5},
        status="completed",
        metrics={
            "accuracy": 0.90,
            "precision": 0.89,
            "recall": 0.88,
            "f1_score": 0.885
        },
        training_time=10.2,
        model_artifact_path="/models/model_456.pkl",
        run_metadata={
            "train_samples": 80,
            "test_samples": 20,
            "n_features": 4
        },
        created_at=datetime.utcnow()
    )
    db_session.add(model_run)
    db_session.commit()
    db_session.refresh(model_run)
    return model_run


@pytest.fixture
def pending_model_run(db_session: Session, test_experiment: Experiment):
    """Create a pending model run"""
    model_run = ModelRun(
        id=uuid.uuid4(),
        experiment_id=test_experiment.id,
        model_type="random_forest_classifier",
        hyperparameters={"n_estimators": 100},
        status="pending",
        created_at=datetime.utcnow()
    )
    db_session.add(model_run)
    db_session.commit()
    db_session.refresh(model_run)
    return model_run


class TestFeatureImportanceEndpoint:
    """Test suite for feature importance endpoint"""

    def test_get_feature_importance_success(self, completed_model_run_with_importance: ModelRun):
        """Test successful retrieval of feature importance"""
        response = client.get(
            f"/api/v1/models/train/{completed_model_run_with_importance.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert data["model_run_id"] == str(completed_model_run_with_importance.id)
        assert data["model_type"] == "random_forest_classifier"
        assert data["task_type"] == "classification"
        assert data["has_feature_importance"] is True
        assert data["total_features"] == 4
        assert data["importance_method"] == "feature_importances_"
        assert data["message"] is None

        # Check feature importance list
        assert len(data["feature_importance"]) == 4
        assert all("feature" in item for item in data["feature_importance"])
        assert all("importance" in item for item in data["feature_importance"])
        assert all("rank" in item for item in data["feature_importance"])

        # Check sorting (descending by importance)
        importances = [item["importance"] for item in data["feature_importance"]]
        assert importances == sorted(importances, reverse=True)

        # Check ranks
        ranks = [item["rank"] for item in data["feature_importance"]]
        assert ranks == [1, 2, 3, 4]

        # Check feature_importance_dict
        assert data["feature_importance_dict"] == {
            "feature1": 0.35,
            "feature2": 0.30,
            "feature3": 0.25,
            "feature4": 0.10
        }

        # Check top_features (default top 10, but only 4 features)
        assert len(data["top_features"]) == 4

    def test_get_feature_importance_with_top_n(self, completed_model_run_with_importance: ModelRun):
        """Test feature importance with top_n parameter"""
        response = client.get(
            f"/api/v1/models/train/{completed_model_run_with_importance.id}/feature-importance?top_n=2"
        )

        assert response.status_code == 200
        data = response.json()

        # Check top_features contains only 2 items
        assert len(data["top_features"]) == 2
        assert data["top_features"][0]["feature"] == "feature1"
        assert data["top_features"][0]["importance"] == 0.35
        assert data["top_features"][1]["feature"] == "feature2"
        assert data["top_features"][1]["importance"] == 0.30

        # Full list should still have all 4
        assert len(data["feature_importance"]) == 4

    def test_get_feature_importance_not_available(self, completed_model_run_without_importance: ModelRun):
        """Test feature importance when not available"""
        response = client.get(
            f"/api/v1/models/train/{completed_model_run_without_importance.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["has_feature_importance"] is False
        assert data["feature_importance"] is None
        assert data["feature_importance_dict"] is None
        assert data["total_features"] == 0
        assert data["top_features"] is None
        assert data["message"] is not None
        assert "does not support feature importance" in data["message"]

    def test_get_feature_importance_model_not_completed(self, pending_model_run: ModelRun):
        """Test feature importance for non-completed model run"""
        response = client.get(
            f"/api/v1/models/train/{pending_model_run.id}/feature-importance"
        )

        assert response.status_code == 400
        data = response.json()
        assert "not completed yet" in data["detail"]

    def test_get_feature_importance_invalid_uuid(self):
        """Test feature importance with invalid UUID"""
        response = client.get(
            "/api/v1/models/train/invalid-uuid/feature-importance"
        )

        assert response.status_code == 400
        data = response.json()
        assert "Invalid model_run_id format" in data["detail"]

    def test_get_feature_importance_not_found(self):
        """Test feature importance for non-existent model run"""
        non_existent_id = uuid.uuid4()
        response = client.get(
            f"/api/v1/models/train/{non_existent_id}/feature-importance"
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_get_feature_importance_unauthorized(
        self,
        db_session: Session,
        completed_model_run_with_importance: ModelRun
    ):
        """Test feature importance access by unauthorized user"""
        # Create another user
        other_user = User(
            id=uuid.uuid4(),
            email="other@example.com",
            hashed_password="hashed_password",
            created_at=datetime.utcnow()
        )
        db_session.add(other_user)
        db_session.commit()

        # Try to access with different user (would need to mock auth)
        # For now, this test is a placeholder
        # In production, you'd mock the get_current_user_id dependency
        pass

    def test_feature_importance_linear_model(self, db_session: Session, test_experiment: Experiment):
        """Test feature importance for linear model (uses coef_)"""
        model_run = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="logistic_regression",
            hyperparameters={"C": 1.0},
            status="completed",
            metrics={"accuracy": 0.92},
            training_time=5.0,
            model_artifact_path="/models/model_linear.pkl",
            run_metadata={
                "feature_importance": {
                    "feature1": 0.45,
                    "feature2": 0.35,
                    "feature3": 0.15,
                    "feature4": 0.05
                }
            },
            created_at=datetime.utcnow()
        )
        db_session.add(model_run)
        db_session.commit()

        response = client.get(
            f"/api/v1/models/train/{model_run.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["importance_method"] == "coef_"
        assert data["has_feature_importance"] is True

    def test_feature_importance_regression_model(self, db_session: Session, test_experiment: Experiment):
        """Test feature importance for regression model"""
        model_run = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest_regressor",
            hyperparameters={"n_estimators": 100},
            status="completed",
            metrics={"r2_score": 0.85, "rmse": 2.5},
            training_time=30.0,
            model_artifact_path="/models/model_reg.pkl",
            run_metadata={
                "feature_importance": {
                    "feature1": 0.40,
                    "feature2": 0.30,
                    "feature3": 0.20,
                    "feature4": 0.10
                }
            },
            created_at=datetime.utcnow()
        )
        db_session.add(model_run)
        db_session.commit()

        response = client.get(
            f"/api/v1/models/train/{model_run.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "regression"
        assert data["has_feature_importance"] is True

    def test_feature_importance_top_n_zero(self, completed_model_run_with_importance: ModelRun):
        """Test feature importance with top_n=0 (should return default top 10)"""
        response = client.get(
            f"/api/v1/models/train/{completed_model_run_with_importance.id}/feature-importance?top_n=0"
        )

        assert response.status_code == 200
        data = response.json()
        # Should return all 4 features since less than 10
        assert len(data["top_features"]) == 4

    def test_feature_importance_top_n_exceeds_total(self, completed_model_run_with_importance: ModelRun):
        """Test feature importance with top_n greater than total features"""
        response = client.get(
            f"/api/v1/models/train/{completed_model_run_with_importance.id}/feature-importance?top_n=100"
        )

        assert response.status_code == 200
        data = response.json()
        # Should return all 4 features
        assert len(data["top_features"]) == 4

    def test_feature_importance_response_schema(self, completed_model_run_with_importance: ModelRun):
        """Test that response matches expected schema"""
        response = client.get(
            f"/api/v1/models/train/{completed_model_run_with_importance.id}/feature-importance"
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields
        required_fields = [
            "model_run_id",
            "model_type",
            "task_type",
            "has_feature_importance",
            "total_features"
        ]
        for field in required_fields:
            assert field in data

        # Optional fields (should be present when has_feature_importance=True)
        if data["has_feature_importance"]:
            assert "feature_importance" in data
            assert "feature_importance_dict" in data
            assert "top_features" in data
            assert "importance_method" in data

            # Check feature importance item structure
            for item in data["feature_importance"]:
                assert "feature" in item
                assert "importance" in item
                assert "rank" in item
                assert isinstance(item["feature"], str)
                assert isinstance(item["importance"], (int, float))
                assert isinstance(item["rank"], int)
