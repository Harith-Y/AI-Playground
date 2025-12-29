"""
Unit tests for model training API endpoints.

Tests cover:
- GET /api/v1/models/available - List available models
- GET /api/v1/models/available/{model_id} - Get model details
- POST /api/v1/models/train - Initiate training
- GET /api/v1/models/train/{model_run_id}/status - Get training status
- GET /api/v1/models/train/{model_run_id}/result - Get training results
- DELETE /api/v1/models/train/{model_run_id} - Delete model run
"""

import pytest
import uuid
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.model_run import ModelRun


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_user(db: Session) -> User:
    """Create a test user."""
    user = User(
        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_dataset(db: Session, test_user: User, tmp_path: Path) -> Dataset:
    """Create a test dataset with CSV file."""
    # Create a sample CSV file
    csv_content = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
6.2,2.9,4.3,1.3,versicolor
5.9,3.0,4.2,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
5.1,3.8,1.5,0.3,setosa
6.4,2.8,5.6,2.1,virginica
5.7,2.8,4.5,1.3,versicolor
6.0,2.2,5.0,1.5,virginica
"""
    
    csv_file = tmp_path / "iris_test.csv"
    csv_file.write_text(csv_content)
    
    dataset = Dataset(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="Iris Test Dataset",
        file_path=str(csv_file),
        file_size=len(csv_content),
        file_type="csv",
        row_count=10,
        column_count=5,
        created_at=datetime.utcnow()
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@pytest.fixture
def test_experiment(db: Session, test_user: User) -> Experiment:
    """Create a test experiment."""
    experiment = Experiment(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="Test Experiment",
        description="Test experiment for model training",
        status="pending",
        created_at=datetime.utcnow()
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment



@pytest.fixture
def test_model_run(db: Session, test_experiment: Experiment) -> ModelRun:
    """Create a test model run."""
    model_run = ModelRun(
        id=uuid.uuid4(),
        experiment_id=test_experiment.id,
        model_type="random_forest_classifier",
        hyperparameters={"n_estimators": 100, "max_depth": 10},
        status="completed",
        metrics={"accuracy": 0.95, "f1_score": 0.93},
        training_time=45.5,
        model_artifact_path="/path/to/model.joblib",
        created_at=datetime.utcnow()
    )
    db.add(model_run)
    db.commit()
    db.refresh(model_run)
    return model_run


# ============================================================================
# Test GET /api/v1/models/available
# ============================================================================

class TestGetAvailableModels:
    """Tests for listing available models."""
    
    def test_get_all_models(self, client: TestClient):
        """Test getting all available models."""
        response = client.get("/api/v1/models/available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_models" in data
        assert "models_by_task" in data
        assert "summary" in data
        
        # Check task types exist
        assert "regression" in data["models_by_task"]
        assert "classification" in data["models_by_task"]
        assert "clustering" in data["models_by_task"]
        
        # Check summary counts
        assert data["summary"]["regression_models"] > 0
        assert data["summary"]["classification_models"] > 0
        assert data["summary"]["clustering_models"] > 0

    
    def test_filter_by_task_type(self, client: TestClient):
        """Test filtering models by task type."""
        response = client.get("/api/v1/models/available?task_type=classification")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_type" in data
        assert data["task_type"] == "classification"
        assert "models" in data
        assert len(data["models"]) > 0
        
        # All models should be classification
        for model in data["models"]:
            assert model["task_type"] == "classification"
    
    def test_filter_by_category(self, client: TestClient):
        """Test filtering models by category."""
        response = client.get("/api/v1/models/available?category=tree_based")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "category" in data
        assert data["category"] == "tree_based"
        assert "models" in data
    
    def test_search_models(self, client: TestClient):
        """Test searching models by keyword."""
        response = client.get("/api/v1/models/available?search=random forest")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "search_query" in data
        assert data["search_query"] == "random forest"
        assert "total_results" in data
        assert "models" in data
    
    def test_invalid_task_type(self, client: TestClient):
        """Test with invalid task type."""
        response = client.get("/api/v1/models/available?task_type=invalid")
        
        assert response.status_code == 400
        assert "Invalid task_type" in response.json()["detail"]
    
    def test_invalid_category(self, client: TestClient):
        """Test with invalid category."""
        response = client.get("/api/v1/models/available?category=invalid")
        
        assert response.status_code == 400
        assert "Invalid category" in response.json()["detail"]



# ============================================================================
# Test GET /api/v1/models/available/{model_id}
# ============================================================================

class TestGetModelDetails:
    """Tests for getting model details."""
    
    def test_get_valid_model(self, client: TestClient):
        """Test getting details of a valid model."""
        response = client.get("/api/v1/models/available/random_forest_classifier")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_id"] == "random_forest_classifier"
        assert "name" in data
        assert "description" in data
        assert "task_type" in data
        assert "category" in data
        assert "hyperparameters" in data
    
    def test_get_invalid_model(self, client: TestClient):
        """Test getting details of non-existent model."""
        response = client.get("/api/v1/models/available/invalid_model")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ============================================================================
# Test POST /api/v1/models/train
# ============================================================================

class TestTrainModel:
    """Tests for initiating model training."""
    
    @patch('app.api.v1.endpoints.models.train_model')
    def test_train_classification_model_success(
        self,
        mock_train_model,
        client: TestClient,
        db: Session,
        test_user: User,
        test_dataset: Dataset,
        test_experiment: Experiment
    ):
        """Test successful training initiation for classification."""
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = "test-task-id-123"
        mock_train_model.delay.return_value = mock_task

        
        request_data = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": str(test_dataset.id),
            "model_type": "random_forest_classifier",
            "target_column": "species",
            "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "test_size": 0.2,
            "random_state": 42,
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10
            }
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 202
        data = response.json()
        
        assert "model_run_id" in data
        assert "task_id" in data
        assert data["task_id"] == "test-task-id-123"
        assert data["status"] == "PENDING"
        assert "message" in data
        
        # Verify model run was created in database
        model_run = db.query(ModelRun).filter(
            ModelRun.id == uuid.UUID(data["model_run_id"])
        ).first()
        
        assert model_run is not None
        assert model_run.model_type == "random_forest_classifier"
        assert model_run.status == "pending"
        assert model_run.hyperparameters["n_estimators"] == 100
    
    def test_train_without_target_column(
        self,
        client: TestClient,
        test_dataset: Dataset,
        test_experiment: Experiment
    ):
        """Test training classification without target column (should fail)."""
        request_data = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": str(test_dataset.id),
            "model_type": "random_forest_classifier",
            "test_size": 0.2
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 400
        assert "target_column is required" in response.json()["detail"]

    
    def test_train_with_invalid_target_column(
        self,
        client: TestClient,
        test_dataset: Dataset,
        test_experiment: Experiment
    ):
        """Test training with non-existent target column."""
        request_data = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": str(test_dataset.id),
            "model_type": "random_forest_classifier",
            "target_column": "nonexistent_column",
            "test_size": 0.2
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 400
        assert "not found in dataset" in response.json()["detail"]
    
    def test_train_with_invalid_feature_columns(
        self,
        client: TestClient,
        test_dataset: Dataset,
        test_experiment: Experiment
    ):
        """Test training with non-existent feature columns."""
        request_data = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": str(test_dataset.id),
            "model_type": "random_forest_classifier",
            "target_column": "species",
            "feature_columns": ["invalid_col1", "invalid_col2"],
            "test_size": 0.2
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 400
        assert "not found in dataset" in response.json()["detail"]
    
    def test_train_with_nonexistent_experiment(
        self,
        client: TestClient,
        test_dataset: Dataset
    ):
        """Test training with non-existent experiment."""
        fake_experiment_id = str(uuid.uuid4())
        
        request_data = {
            "experiment_id": fake_experiment_id,
            "dataset_id": str(test_dataset.id),
            "model_type": "random_forest_classifier",
            "target_column": "species",
            "test_size": 0.2
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 404
        assert "Experiment" in response.json()["detail"]

    
    def test_train_with_nonexistent_dataset(
        self,
        client: TestClient,
        test_experiment: Experiment
    ):
        """Test training with non-existent dataset."""
        fake_dataset_id = str(uuid.uuid4())
        
        request_data = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": fake_dataset_id,
            "model_type": "random_forest_classifier",
            "target_column": "species",
            "test_size": 0.2
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 404
        assert "Dataset" in response.json()["detail"]
    
    def test_train_with_invalid_model_type(
        self,
        client: TestClient,
        test_dataset: Dataset,
        test_experiment: Experiment
    ):
        """Test training with invalid model type."""
        request_data = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": str(test_dataset.id),
            "model_type": "invalid_model_type",
            "target_column": "species",
            "test_size": 0.2
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 400
        assert "not found in registry" in response.json()["detail"]
    
    @patch('app.api.v1.endpoints.models.train_model')
    def test_train_clustering_model(
        self,
        mock_train_model,
        client: TestClient,
        test_dataset: Dataset,
        test_experiment: Experiment
    ):
        """Test training clustering model (no target column needed)."""
        mock_task = Mock()
        mock_task.id = "test-task-id-456"
        mock_train_model.delay.return_value = mock_task
        
        request_data = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": str(test_dataset.id),
            "model_type": "kmeans",
            "feature_columns": ["sepal_length", "sepal_width"],
            "test_size": 0.2,
            "hyperparameters": {
                "n_clusters": 3
            }
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "PENDING"



# ============================================================================
# Test GET /api/v1/models/train/{model_run_id}/status
# ============================================================================

class TestGetTrainingStatus:
    """Tests for getting training status."""
    
    def test_get_status_completed(
        self,
        client: TestClient,
        test_model_run: ModelRun
    ):
        """Test getting status of completed model run."""
        response = client.get(f"/api/v1/models/train/{test_model_run.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_run_id"] == str(test_model_run.id)
        assert data["status"] == "SUCCESS"
        assert "result" in data
        assert data["result"]["metrics"]["accuracy"] == 0.95
    
    def test_get_status_nonexistent(self, client: TestClient):
        """Test getting status of non-existent model run."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/models/train/{fake_id}/status")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_status_pending(
        self,
        client: TestClient,
        db: Session,
        test_experiment: Experiment
    ):
        """Test getting status of pending model run."""
        model_run = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest_classifier",
            hyperparameters={},
            status="pending",
            created_at=datetime.utcnow()
        )
        db.add(model_run)
        db.commit()
        
        response = client.get(f"/api/v1/models/train/{model_run.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PENDING"
    
    def test_get_status_failed(
        self,
        client: TestClient,
        db: Session,
        test_experiment: Experiment
    ):
        """Test getting status of failed model run."""
        model_run = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest_classifier",
            hyperparameters={},
            status="failed",
            run_metadata={
                "error": {
                    "type": "ValueError",
                    "message": "Invalid configuration"
                }
            },
            created_at=datetime.utcnow()
        )
        db.add(model_run)
        db.commit()
        
        response = client.get(f"/api/v1/models/train/{model_run.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILURE"
        assert "error" in data
        assert "ValueError" in data["error"]



# ============================================================================
# Test GET /api/v1/models/train/{model_run_id}/result
# ============================================================================

class TestGetTrainingResult:
    """Tests for getting training results."""
    
    def test_get_result_success(
        self,
        client: TestClient,
        test_model_run: ModelRun
    ):
        """Test getting results of completed model run."""
        response = client.get(f"/api/v1/models/train/{test_model_run.id}/result")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_run_id"] == str(test_model_run.id)
        assert data["model_type"] == "random_forest_classifier"
        assert data["metrics"]["accuracy"] == 0.95
        assert data["training_time"] == 45.5
        assert "hyperparameters" in data
    
    def test_get_result_not_completed(
        self,
        client: TestClient,
        db: Session,
        test_experiment: Experiment
    ):
        """Test getting results of non-completed model run."""
        model_run = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest_classifier",
            hyperparameters={},
            status="running",
            created_at=datetime.utcnow()
        )
        db.add(model_run)
        db.commit()
        
        response = client.get(f"/api/v1/models/train/{model_run.id}/result")
        
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]
    
    def test_get_result_nonexistent(self, client: TestClient):
        """Test getting results of non-existent model run."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/models/train/{fake_id}/result")
        
        assert response.status_code == 404


# ============================================================================
# Test DELETE /api/v1/models/train/{model_run_id}
# ============================================================================

class TestDeleteModelRun:
    """Tests for deleting model runs."""
    
    @patch('app.api.v1.endpoints.models.get_model_serialization_service')
    def test_delete_completed_model_run(
        self,
        mock_get_service,
        client: TestClient,
        db: Session,
        test_model_run: ModelRun
    ):
        """Test deleting a completed model run."""
        # Mock serialization service
        mock_service = Mock()
        mock_service.delete_model.return_value = True
        mock_get_service.return_value = mock_service
        
        response = client.delete(f"/api/v1/models/train/{test_model_run.id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Model run deleted successfully"
        assert data["deletion_summary"]["model_run_id"] == str(test_model_run.id)
        assert data["deletion_summary"]["database_record_deleted"] is True
        assert data["deletion_summary"]["artifact_deleted"] is True
        
        # Verify model run was deleted from database
        deleted_run = db.query(ModelRun).filter(
            ModelRun.id == test_model_run.id
        ).first()
        assert deleted_run is None

    
    @patch('app.api.v1.endpoints.models.get_model_serialization_service')
    @patch('app.api.v1.endpoints.models.AsyncResult')
    def test_delete_running_model_run(
        self,
        mock_async_result,
        mock_get_service,
        client: TestClient,
        db: Session,
        test_experiment: Experiment
    ):
        """Test deleting a running model run (should revoke task)."""
        # Create running model run
        model_run = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest_classifier",
            hyperparameters={},
            status="running",
            run_metadata={"task_id": "test-task-123"},
            created_at=datetime.utcnow()
        )
        db.add(model_run)
        db.commit()
        
        # Mock Celery task
        mock_task = Mock()
        mock_task.state = "STARTED"
        mock_task.revoke = Mock()
        mock_async_result.return_value = mock_task
        
        # Mock serialization service
        mock_service = Mock()
        mock_service.delete_model.return_value = False  # No artifact yet
        mock_get_service.return_value = mock_service
        
        response = client.delete(f"/api/v1/models/train/{model_run.id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["deletion_summary"]["task_revoked"] is True
        assert data["deletion_summary"]["database_record_deleted"] is True
        
        # Verify task.revoke was called
        mock_task.revoke.assert_called_once()
    
    def test_delete_nonexistent_model_run(self, client: TestClient):
        """Test deleting non-existent model run."""
        fake_id = str(uuid.uuid4())
        response = client.delete(f"/api/v1/models/train/{fake_id}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_delete_with_invalid_uuid(self, client: TestClient):
        """Test deleting with invalid UUID format."""
        response = client.delete("/api/v1/models/train/invalid-uuid")
        
        assert response.status_code == 400
        assert "Invalid model_run_id format" in response.json()["detail"]
    
    @patch('app.api.v1.endpoints.models.get_model_serialization_service')
    def test_delete_without_artifact(
        self,
        mock_get_service,
        client: TestClient,
        db: Session,
        test_experiment: Experiment
    ):
        """Test deleting model run without artifact file."""
        model_run = ModelRun(
            id=uuid.uuid4(),
            experiment_id=test_experiment.id,
            model_type="random_forest_classifier",
            hyperparameters={},
            status="failed",
            model_artifact_path=None,  # No artifact
            created_at=datetime.utcnow()
        )
        db.add(model_run)
        db.commit()
        
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        
        response = client.delete(f"/api/v1/models/train/{model_run.id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should not attempt to delete artifact
        assert data["deletion_summary"]["artifact_deleted"] is False
        assert data["deletion_summary"]["database_record_deleted"] is True



# ============================================================================
# Test GET /api/v1/models/categories
# ============================================================================

class TestGetModelCategories:
    """Tests for getting model categories."""
    
    def test_get_categories(self, client: TestClient):
        """Test getting all model categories."""
        response = client.get("/api/v1/models/categories")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "categories" in data
        assert len(data["categories"]) > 0
        
        # Check structure of first category
        first_category = data["categories"][0]
        assert "id" in first_category
        assert "name" in first_category
        assert "description" in first_category


# ============================================================================
# Test GET /api/v1/models/task-types
# ============================================================================

class TestGetTaskTypes:
    """Tests for getting task types."""
    
    def test_get_task_types(self, client: TestClient):
        """Test getting all task types."""
        response = client.get("/api/v1/models/task-types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_types" in data
        assert len(data["task_types"]) == 3  # regression, classification, clustering
        
        # Check each task type has required fields
        for task_type in data["task_types"]:
            assert "id" in task_type
            assert "name" in task_type
            assert "description" in task_type
            assert "model_count" in task_type
            assert "examples" in task_type
            assert task_type["model_count"] > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestModelTrainingIntegration:
    """Integration tests for complete training workflows."""
    
    @patch('app.api.v1.endpoints.models.train_model')
    def test_complete_training_workflow(
        self,
        mock_train_model,
        client: TestClient,
        db: Session,
        test_user: User,
        test_dataset: Dataset,
        test_experiment: Experiment
    ):
        """Test complete workflow: train -> status -> result -> delete."""
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = "workflow-task-123"
        mock_train_model.delay.return_value = mock_task
        
        # 1. Initiate training
        train_request = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": str(test_dataset.id),
            "model_type": "random_forest_classifier",
            "target_column": "species",
            "test_size": 0.2
        }
        
        train_response = client.post("/api/v1/models/train", json=train_request)
        assert train_response.status_code == 202
        model_run_id = train_response.json()["model_run_id"]
        
        # 2. Check status
        status_response = client.get(f"/api/v1/models/train/{model_run_id}/status")
        assert status_response.status_code == 200
        
        # 3. Simulate completion and get results
        model_run = db.query(ModelRun).filter(
            ModelRun.id == uuid.UUID(model_run_id)
        ).first()
        model_run.status = "completed"
        model_run.metrics = {"accuracy": 0.92}
        model_run.training_time = 30.0
        db.commit()
        
        result_response = client.get(f"/api/v1/models/train/{model_run_id}/result")
        assert result_response.status_code == 200
        assert result_response.json()["metrics"]["accuracy"] == 0.92
        
        # 4. Delete model run
        with patch('app.api.v1.endpoints.models.get_model_serialization_service') as mock_service:
            mock_service.return_value.delete_model.return_value = True
            delete_response = client.delete(f"/api/v1/models/train/{model_run_id}")
            assert delete_response.status_code == 200

    
    def test_multiple_model_runs_same_experiment(
        self,
        client: TestClient,
        db: Session,
        test_experiment: Experiment
    ):
        """Test creating multiple model runs for same experiment."""
        # Create multiple completed model runs
        model_runs = []
        for i in range(3):
            model_run = ModelRun(
                id=uuid.uuid4(),
                experiment_id=test_experiment.id,
                model_type=f"model_type_{i}",
                hyperparameters={},
                status="completed",
                metrics={"accuracy": 0.8 + i * 0.05},
                created_at=datetime.utcnow()
            )
            db.add(model_run)
            model_runs.append(model_run)
        
        db.commit()
        
        # Verify all can be queried
        for model_run in model_runs:
            response = client.get(f"/api/v1/models/train/{model_run.id}/status")
            assert response.status_code == 200
    
    @patch('app.api.v1.endpoints.models.train_model')
    def test_train_with_all_parameters(
        self,
        mock_train_model,
        client: TestClient,
        test_dataset: Dataset,
        test_experiment: Experiment
    ):
        """Test training with all optional parameters specified."""
        mock_task = Mock()
        mock_task.id = "full-params-task"
        mock_train_model.delay.return_value = mock_task
        
        request_data = {
            "experiment_id": str(test_experiment.id),
            "dataset_id": str(test_dataset.id),
            "model_type": "gradient_boosting_classifier",
            "target_column": "species",
            "feature_columns": ["sepal_length", "sepal_width"],
            "test_size": 0.3,
            "random_state": 123,
            "hyperparameters": {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 5
            }
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        
        assert response.status_code == 202
        
        # Verify Celery task was called with correct parameters
        mock_train_model.delay.assert_called_once()
        call_kwargs = mock_train_model.delay.call_args[1]
        assert call_kwargs["test_size"] == 0.3
        assert call_kwargs["random_state"] == 123
        assert call_kwargs["hyperparameters"]["n_estimators"] == 200
