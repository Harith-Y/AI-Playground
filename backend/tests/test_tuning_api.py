"""
Integration tests for Hyperparameter Tuning API endpoints.

Tests cover:
- POST /api/v1/tuning/tune - Initiate tuning
- GET /api/v1/tuning/tune/{id}/status - Get tuning status
- GET /api/v1/tuning/tune/{id}/results - Get tuning results
- Authorization and validation
- Error handling
"""

import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime

from app.models.dataset import Dataset
from app.models.user import User
from app.models.experiment import Experiment
from app.models.model_run import ModelRun
from app.models.tuning_run import TuningRun, TuningStatus


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
        rows=100,
        cols=5,
        dtypes={"col1": "int64", "col2": "float64", "col3": "object"},
        missing_values={"col1": 0, "col2": 5, "col3": 2}
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
        description="Test experiment for tuning",
        status="running"
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


@pytest.fixture
def test_model_run(db: Session, test_experiment: Experiment):
    """Create a completed test model run."""
    model_run = ModelRun(
        id=uuid.uuid4(),
        experiment_id=test_experiment.id,
        model_type="random_forest_classifier",
        status="completed",
        hyperparameters={"n_estimators": 100, "max_depth": 10},
        metrics={"accuracy": 0.85, "f1_score": 0.83},
        training_time=12.5,
        model_artifact_path="/fake/path/model.pkl"
    )
    db.add(model_run)
    db.commit()
    db.refresh(model_run)
    return model_run


@pytest.fixture
def test_tuning_run(db: Session, test_model_run: ModelRun):
    """Create a test tuning run."""
    tuning_run = TuningRun(
        id=uuid.uuid4(),
        model_run_id=test_model_run.id,
        tuning_method="grid_search",
        param_grid={"n_estimators": [50, 100], "max_depth": [5, 10]},
        status=TuningStatus.PENDING,
        cv_folds=5,
        scoring_metric="accuracy"
    )
    db.add(tuning_run)
    db.commit()
    db.refresh(tuning_run)
    return tuning_run


class TestTuneModelHyperparameters:
    """Tests for POST /api/v1/tuning/tune endpoint"""

    @patch('app.tasks.tuning_tasks.tune_hyperparameters.apply_async')
    def test_tune_success_grid_search(
        self, 
        mock_task, 
        client: TestClient, 
        db: Session, 
        test_model_run: ModelRun
    ):
        """Test successful tuning initiation with grid search"""
        mock_task.return_value = Mock(id="task-123")
        
        payload = {
            "model_run_id": str(test_model_run.id),
            "tuning_method": "grid_search",
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None]
            },
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 202
        data = response.json()

        assert "tuning_run_id" in data
        assert data["task_id"] == "task-123"
        assert data["status"] == "PENDING"
        assert "message" in data
        assert "created_at" in data

        # Verify tuning run created in database
        tuning_run = db.query(TuningRun).filter(
            TuningRun.id == uuid.UUID(data["tuning_run_id"])
        ).first()
        
        assert tuning_run is not None
        assert tuning_run.model_run_id == test_model_run.id
        assert tuning_run.tuning_method == "grid_search"
        assert tuning_run.cv_folds == 5
        assert tuning_run.scoring_metric == "accuracy"

    @patch('app.tasks.tuning_tasks.tune_hyperparameters.apply_async')
    def test_tune_success_random_search(
        self, 
        mock_task, 
        client: TestClient, 
        test_model_run: ModelRun
    ):
        """Test successful tuning with random search"""
        mock_task.return_value = Mock(id="task-456")
        
        payload = {
            "model_run_id": str(test_model_run.id),
            "tuning_method": "random_search",
            "param_distributions": {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10]
            },
            "n_iter": 20,
            "cv_folds": 3,
            "scoring_metric": "f1"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == "task-456"

    @patch('app.tasks.tuning_tasks.tune_hyperparameters.apply_async')
    def test_tune_success_bayesian(
        self, 
        mock_task, 
        client: TestClient, 
        test_model_run: ModelRun
    ):
        """Test successful tuning with Bayesian optimization"""
        mock_task.return_value = Mock(id="task-789")
        
        payload = {
            "model_run_id": str(test_model_run.id),
            "tuning_method": "bayesian",
            "search_space": {
                "n_estimators": {"type": "int", "low": 50, "high": 200},
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3}
            },
            "n_trials": 30,
            "cv_folds": 5,
            "scoring_metric": "roc_auc"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == "task-789"

    def test_tune_model_not_found(self, client: TestClient):
        """Test tuning with non-existent model run"""
        payload = {
            "model_run_id": str(uuid.uuid4()),
            "tuning_method": "grid_search",
            "param_grid": {"n_estimators": [50, 100]},
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_tune_model_not_completed(
        self, 
        client: TestClient, 
        db: Session, 
        test_model_run: ModelRun
    ):
        """Test tuning with incomplete model run"""
        # Update model run to pending
        test_model_run.status = "running"
        db.commit()

        payload = {
            "model_run_id": str(test_model_run.id),
            "tuning_method": "grid_search",
            "param_grid": {"n_estimators": [50, 100]},
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"].lower()

    def test_tune_invalid_method(self, client: TestClient, test_model_run: ModelRun):
        """Test tuning with invalid tuning method"""
        payload = {
            "model_run_id": str(test_model_run.id),
            "tuning_method": "invalid_method",
            "param_grid": {"n_estimators": [50, 100]},
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 422  # Validation error

    def test_tune_invalid_uuid(self, client: TestClient):
        """Test tuning with invalid UUID format"""
        payload = {
            "model_run_id": "not-a-uuid",
            "tuning_method": "grid_search",
            "param_grid": {"n_estimators": [50, 100]},
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code in [400, 422]

    def test_tune_missing_required_fields(self, client: TestClient):
        """Test tuning with missing required fields"""
        payload = {
            "model_run_id": str(uuid.uuid4()),
            "tuning_method": "grid_search"
            # Missing param_grid, cv_folds, scoring_metric
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 422

    def test_tune_empty_param_grid(self, client: TestClient, test_model_run: ModelRun):
        """Test tuning with empty parameter grid"""
        payload = {
            "model_run_id": str(test_model_run.id),
            "tuning_method": "grid_search",
            "param_grid": {},
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower() or "required" in response.json()["detail"].lower()


class TestGetTuningStatus:
    """Tests for GET /api/v1/tuning/tune/{id}/status endpoint"""

    def test_get_status_pending(
        self, 
        client: TestClient, 
        test_tuning_run: TuningRun
    ):
        """Test getting status of pending tuning run"""
        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/status")

        assert response.status_code == 200
        data = response.json()

        assert data["tuning_run_id"] == str(test_tuning_run.id)
        assert data["status"] in ["PENDING", "pending"]
        assert data["task_id"] is not None or data["task_id"] == ""
        assert "progress" in data or data.get("progress") is None

    @patch('app.celery_app.celery_app.AsyncResult')
    def test_get_status_running(
        self, 
        mock_result, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting status of running tuning run"""
        # Update tuning run to running
        test_tuning_run.status = TuningStatus.RUNNING
        test_tuning_run.task_id = "task-123"
        db.commit()

        # Mock Celery task status
        mock_task = Mock()
        mock_task.state = "PROGRESS"
        mock_task.info = {
            "current": 5,
            "total": 10,
            "status": "Testing parameter combination 5/10..."
        }
        mock_result.return_value = mock_task

        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/status")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] in ["PROGRESS", "RUNNING", "running"]
        if "progress" in data and data["progress"]:
            assert data["progress"]["current"] == 5
            assert data["progress"]["total"] == 10

    def test_get_status_completed(
        self, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting status of completed tuning run"""
        # Update tuning run to completed
        test_tuning_run.status = TuningStatus.COMPLETED
        test_tuning_run.best_params = {"n_estimators": 100, "max_depth": 10}
        test_tuning_run.best_score = 0.92
        db.commit()

        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/status")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] in ["SUCCESS", "COMPLETED", "completed"]
        # Result may or may not be included in status endpoint

    def test_get_status_failed(
        self, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting status of failed tuning run"""
        # Update tuning run to failed
        test_tuning_run.status = TuningStatus.FAILED
        test_tuning_run.error_message = "Insufficient memory"
        db.commit()

        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/status")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] in ["FAILURE", "FAILED", "failed"]
        if "error" in data:
            assert data["error"] is not None

    def test_get_status_not_found(self, client: TestClient):
        """Test getting status of non-existent tuning run"""
        fake_id = uuid.uuid4()
        response = client.get(f"/api/v1/tuning/tune/{fake_id}/status")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_status_invalid_uuid(self, client: TestClient):
        """Test getting status with invalid UUID"""
        response = client.get("/api/v1/tuning/tune/not-a-uuid/status")

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()


class TestGetTuningResults:
    """Tests for GET /api/v1/tuning/tune/{id}/results endpoint"""

    def test_get_results_success(
        self, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting results of completed tuning run"""
        # Update tuning run with results
        test_tuning_run.status = TuningStatus.COMPLETED
        test_tuning_run.best_params = {"n_estimators": 150, "max_depth": 8}
        test_tuning_run.best_score = 0.91
        test_tuning_run.results = {
            "best_score": 0.91,
            "total_combinations": 12,
            "all_results": [
                {"params": {"n_estimators": 150, "max_depth": 8}, "score": 0.91},
                {"params": {"n_estimators": 100, "max_depth": 10}, "score": 0.89},
                {"params": {"n_estimators": 200, "max_depth": 5}, "score": 0.87}
            ],
            "cv_folds": 5,
            "scoring_metric": "accuracy",
            "tuning_time": 45.2,
            "tuning_method": "grid_search"
        }
        db.commit()

        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/results")

        assert response.status_code == 200
        data = response.json()

        assert data["tuning_run_id"] == str(test_tuning_run.id)
        assert data["best_params"] == {"n_estimators": 150, "max_depth": 8}
        assert data["best_score"] == 0.91
        assert data["total_combinations"] == 12
        assert len(data["top_results"]) <= 10  # Default top_n
        assert data["cv_folds"] == 5
        assert data["scoring_metric"] == "accuracy"
        assert data["tuning_time"] == 45.2

    def test_get_results_with_top_n(
        self, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting results with custom top_n"""
        # Set up completed tuning run with many results
        test_tuning_run.status = TuningStatus.COMPLETED
        test_tuning_run.best_params = {"n_estimators": 100}
        test_tuning_run.best_score = 0.95
        test_tuning_run.results = {
            "best_score": 0.95,
            "total_combinations": 20,
            "all_results": [
                {"params": {"n_estimators": i}, "score": 0.95 - i*0.01}
                for i in range(20)
            ],
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }
        db.commit()

        response = client.get(
            f"/api/v1/tuning/tune/{test_tuning_run.id}/results?top_n=5"
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["top_results"]) == 5

    def test_get_results_not_completed(
        self, 
        client: TestClient, 
        test_tuning_run: TuningRun
    ):
        """Test getting results of incomplete tuning run"""
        # tuning_run is PENDING by default
        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/results")

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"].lower()

    def test_get_results_no_results_data(
        self, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting results when no results data available"""
        # Set to completed but with no results
        test_tuning_run.status = TuningStatus.COMPLETED
        test_tuning_run.best_params = None
        test_tuning_run.results = None
        db.commit()

        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/results")

        assert response.status_code == 404
        assert "no" in response.json()["detail"].lower()

    def test_get_results_not_found(self, client: TestClient):
        """Test getting results of non-existent tuning run"""
        fake_id = uuid.uuid4()
        response = client.get(f"/api/v1/tuning/tune/{fake_id}/results")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_results_invalid_uuid(self, client: TestClient):
        """Test getting results with invalid UUID"""
        response = client.get("/api/v1/tuning/tune/not-a-uuid/results")

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_get_results_invalid_top_n(
        self, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting results with invalid top_n parameter"""
        test_tuning_run.status = TuningStatus.COMPLETED
        test_tuning_run.best_params = {"n_estimators": 100}
        test_tuning_run.results = {
            "best_score": 0.9,
            "all_results": [],
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }
        db.commit()

        response = client.get(
            f"/api/v1/tuning/tune/{test_tuning_run.id}/results?top_n=-5"
        )

        # Should either reject or default to valid value
        assert response.status_code in [200, 400, 422]


class TestTuningAuthorization:
    """Tests for authorization in tuning endpoints"""

    def test_tune_unauthorized_model(
        self, 
        client: TestClient, 
        db: Session, 
        test_model_run: ModelRun
    ):
        """Test tuning model belonging to different user"""
        # Create a different user and experiment
        other_user = User(
            id=uuid.uuid4(),
            email="other@example.com"
        )
        db.add(other_user)
        
        other_dataset = Dataset(
            id=uuid.uuid4(),
            user_id=other_user.id,
            name="other_dataset",
            file_path="/other/path.csv",
            rows=50,
            cols=3
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
            hyperparameters={},
            metrics={"accuracy": 0.8}
        )
        db.add(other_model)
        db.commit()

        payload = {
            "model_run_id": str(other_model.id),
            "tuning_method": "grid_search",
            "param_grid": {"C": [0.1, 1.0, 10.0]},
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        # Should be forbidden (403) or not found (404) depending on implementation
        assert response.status_code in [403, 404]

    def test_status_unauthorized_tuning_run(
        self, 
        client: TestClient, 
        db: Session
    ):
        """Test getting status of tuning run belonging to different user"""
        # Create tuning run for different user
        other_user = User(id=uuid.uuid4(), email="other@example.com")
        db.add(other_user)
        
        other_dataset = Dataset(
            id=uuid.uuid4(),
            user_id=other_user.id,
            name="other_dataset",
            file_path="/other/path.csv",
            rows=50,
            cols=3
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
            status="completed"
        )
        db.add(other_model)
        
        other_tuning = TuningRun(
            id=uuid.uuid4(),
            model_run_id=other_model.id,
            tuning_method="grid_search",
            status=TuningStatus.RUNNING
        )
        db.add(other_tuning)
        db.commit()

        response = client.get(f"/api/v1/tuning/tune/{other_tuning.id}/status")

        assert response.status_code in [403, 404]


class TestTuningEdgeCases:
    """Tests for edge cases in tuning endpoints"""

    @patch('app.tasks.tuning_tasks.tune_hyperparameters.apply_async')
    def test_tune_with_all_optional_params(
        self, 
        mock_task, 
        client: TestClient, 
        test_model_run: ModelRun
    ):
        """Test tuning with all optional parameters specified"""
        mock_task.return_value = Mock(id="task-999")
        
        payload = {
            "model_run_id": str(test_model_run.id),
            "tuning_method": "random_search",
            "param_distributions": {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, 15]
            },
            "n_iter": 15,
            "cv_folds": 10,
            "scoring_metric": "precision",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 1
        }

        response = client.post("/api/v1/tuning/tune", json=payload)

        assert response.status_code == 202

    def test_get_results_empty_all_results(
        self, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting results when all_results is empty"""
        test_tuning_run.status = TuningStatus.COMPLETED
        test_tuning_run.best_params = {"n_estimators": 100}
        test_tuning_run.results = {
            "best_score": 0.9,
            "total_combinations": 0,
            "all_results": [],
            "cv_folds": 5,
            "scoring_metric": "accuracy"
        }
        db.commit()

        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/results")

        assert response.status_code == 200
        data = response.json()
        assert data["top_results"] == []

    @patch('app.celery_app.celery_app.AsyncResult')
    def test_get_status_celery_task_not_found(
        self, 
        mock_result, 
        client: TestClient, 
        db: Session, 
        test_tuning_run: TuningRun
    ):
        """Test getting status when Celery task doesn't exist"""
        test_tuning_run.task_id = "nonexistent-task"
        db.commit()

        mock_task = Mock()
        mock_task.state = "PENDING"
        mock_task.info = None
        mock_result.return_value = mock_task

        response = client.get(f"/api/v1/tuning/tune/{test_tuning_run.id}/status")

        # Should still return valid response
        assert response.status_code == 200
