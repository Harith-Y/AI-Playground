"""
Unit tests for preprocessing steps CRUD endpoints.

Tests cover:
- Creating preprocessing steps
- Listing preprocessing steps
- Getting individual steps
- Updating steps
- Deleting steps
- Reordering steps
- Authorization and validation
"""

import pytest
import uuid
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.user import User
from app.models.preprocessing_step import PreprocessingStep


# Fixtures for test data
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
def test_preprocessing_step(db: Session, test_dataset: Dataset):
    """Create a test preprocessing step."""
    step = PreprocessingStep(
        id=uuid.uuid4(),
        dataset_id=test_dataset.id,
        step_type="missing_value_imputation",
        parameters={"strategy": "mean"},
        column_name="col2",
        order=0
    )
    db.add(step)
    db.commit()
    db.refresh(step)
    return step


class TestCreatePreprocessingStep:
    """Tests for POST /api/v1/preprocessing/ endpoint"""

    def test_create_step_success(self, client: TestClient, db: Session, test_dataset: Dataset):
        """Test successful preprocessing step creation"""
        payload = {
            "dataset_id": str(test_dataset.id),
            "step_type": "missing_value_imputation",
            "parameters": {"strategy": "mean"},
            "column_name": "col2",
            "order": 0
        }

        response = client.post("/api/v1/preprocessing/", json=payload)

        assert response.status_code == 201
        data = response.json()

        assert "id" in data
        assert data["dataset_id"] == str(test_dataset.id)
        assert data["step_type"] == "missing_value_imputation"
        assert data["parameters"] == {"strategy": "mean"}
        assert data["column_name"] == "col2"
        assert data["order"] == 0

        # Verify it's in database
        step = db.query(PreprocessingStep).filter(
            PreprocessingStep.id == data["id"]
        ).first()
        assert step is not None
        assert step.step_type == "missing_value_imputation"

    def test_create_step_auto_order(self, client: TestClient, db: Session, test_dataset: Dataset):
        """Test that order is auto-assigned if not specified"""
        # Create first step
        payload1 = {
            "dataset_id": str(test_dataset.id),
            "step_type": "outlier_detection",
            "parameters": {"method": "iqr", "threshold": 1.5}
        }
        response1 = client.post("/api/v1/preprocessing/", json=payload1)
        assert response1.status_code == 201
        assert response1.json()["order"] == 0

        # Create second step without specifying order
        payload2 = {
            "dataset_id": str(test_dataset.id),
            "step_type": "scaling",
            "parameters": {"method": "standard"}
        }
        response2 = client.post("/api/v1/preprocessing/", json=payload2)
        assert response2.status_code == 201
        assert response2.json()["order"] == 1

    def test_create_step_scaling(self, client: TestClient, test_dataset: Dataset):
        """Test creating a scaling step"""
        payload = {
            "dataset_id": str(test_dataset.id),
            "step_type": "scaling",
            "parameters": {"method": "minmax", "feature_range": [0, 1]},
            "column_name": None
        }

        response = client.post("/api/v1/preprocessing/", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["step_type"] == "scaling"
        assert data["parameters"]["method"] == "minmax"
        assert data["column_name"] is None

    def test_create_step_encoding(self, client: TestClient, test_dataset: Dataset):
        """Test creating an encoding step"""
        payload = {
            "dataset_id": str(test_dataset.id),
            "step_type": "encoding",
            "parameters": {"method": "onehot"},
            "column_name": "col3"
        }

        response = client.post("/api/v1/preprocessing/", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["step_type"] == "encoding"
        assert data["column_name"] == "col3"

    def test_create_step_outlier_detection(self, client: TestClient, test_dataset: Dataset):
        """Test creating an outlier detection step"""
        payload = {
            "dataset_id": str(test_dataset.id),
            "step_type": "outlier_detection",
            "parameters": {"method": "zscore", "threshold": 3},
            "column_name": "col2"
        }

        response = client.post("/api/v1/preprocessing/", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["step_type"] == "outlier_detection"
        assert data["parameters"]["method"] == "zscore"

    def test_create_step_dataset_not_found(self, client: TestClient):
        """Test creating step for non-existent dataset"""
        payload = {
            "dataset_id": str(uuid.uuid4()),
            "step_type": "scaling",
            "parameters": {"method": "standard"}
        }

        response = client.post("/api/v1/preprocessing/", json=payload)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_create_step_invalid_payload(self, client: TestClient, test_dataset: Dataset):
        """Test creating step with invalid payload"""
        payload = {
            "dataset_id": str(test_dataset.id),
            # Missing step_type
            "parameters": {"strategy": "mean"}
        }

        response = client.post("/api/v1/preprocessing/", json=payload)

        assert response.status_code == 422  # Validation error


class TestListPreprocessingSteps:
    """Tests for GET /api/v1/preprocessing/ endpoint"""

    def test_list_steps_empty(self, client: TestClient, test_dataset: Dataset):
        """Test listing when no steps exist"""
        response = client.get("/api/v1/preprocessing/")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_steps_multiple(self, client: TestClient, db: Session, test_dataset: Dataset):
        """Test listing multiple preprocessing steps"""
        # Create multiple steps
        steps_data = [
            {"step_type": "outlier_detection", "order": 0},
            {"step_type": "missing_value_imputation", "order": 1},
            {"step_type": "scaling", "order": 2}
        ]

        for step_data in steps_data:
            step = PreprocessingStep(
                id=uuid.uuid4(),
                dataset_id=test_dataset.id,
                **step_data
            )
            db.add(step)
        db.commit()

        response = client.get("/api/v1/preprocessing/")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

        # Verify ordering
        assert data[0]["step_type"] == "outlier_detection"
        assert data[1]["step_type"] == "missing_value_imputation"
        assert data[2]["step_type"] == "scaling"

    def test_list_steps_filter_by_dataset(
        self, client: TestClient, db: Session, test_user: User, test_dataset: Dataset
    ):
        """Test filtering steps by dataset_id"""
        # Create second dataset
        dataset2 = Dataset(
            id=uuid.uuid4(),
            user_id=test_user.id,
            name="dataset2",
            file_path="/fake/path/test2.csv",
            rows=50,
            cols=3
        )
        db.add(dataset2)
        db.commit()

        # Create steps for different datasets
        step1 = PreprocessingStep(
            id=uuid.uuid4(),
            dataset_id=test_dataset.id,
            step_type="scaling",
            order=0
        )
        step2 = PreprocessingStep(
            id=uuid.uuid4(),
            dataset_id=dataset2.id,
            step_type="encoding",
            order=0
        )
        db.add_all([step1, step2])
        db.commit()

        # Filter by first dataset
        response = client.get(f"/api/v1/preprocessing/?dataset_id={test_dataset.id}")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["step_type"] == "scaling"
        assert data[0]["dataset_id"] == str(test_dataset.id)


class TestGetPreprocessingStep:
    """Tests for GET /api/v1/preprocessing/{step_id} endpoint"""

    def test_get_step_success(self, client: TestClient, test_preprocessing_step: PreprocessingStep):
        """Test getting a specific step by ID"""
        response = client.get(f"/api/v1/preprocessing/{test_preprocessing_step.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_preprocessing_step.id)
        assert data["step_type"] == "missing_value_imputation"
        assert data["parameters"] == {"strategy": "mean"}

    def test_get_step_not_found(self, client: TestClient):
        """Test getting non-existent step"""
        fake_id = uuid.uuid4()
        response = client.get(f"/api/v1/preprocessing/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestUpdatePreprocessingStep:
    """Tests for PUT /api/v1/preprocessing/{step_id} endpoint"""

    def test_update_step_parameters(
        self, client: TestClient, db: Session, test_preprocessing_step: PreprocessingStep
    ):
        """Test updating step parameters"""
        update_payload = {
            "parameters": {"strategy": "median"}
        }

        response = client.put(
            f"/api/v1/preprocessing/{test_preprocessing_step.id}",
            json=update_payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["parameters"]["strategy"] == "median"

        # Verify in database
        db.refresh(test_preprocessing_step)
        assert test_preprocessing_step.parameters["strategy"] == "median"

    def test_update_step_type(
        self, client: TestClient, db: Session, test_preprocessing_step: PreprocessingStep
    ):
        """Test updating step type"""
        update_payload = {
            "step_type": "scaling",
            "parameters": {"method": "standard"}
        }

        response = client.put(
            f"/api/v1/preprocessing/{test_preprocessing_step.id}",
            json=update_payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["step_type"] == "scaling"
        assert data["parameters"]["method"] == "standard"

    def test_update_step_order(
        self, client: TestClient, db: Session, test_preprocessing_step: PreprocessingStep
    ):
        """Test updating step order"""
        update_payload = {
            "order": 5
        }

        response = client.put(
            f"/api/v1/preprocessing/{test_preprocessing_step.id}",
            json=update_payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["order"] == 5

        # Verify in database
        db.refresh(test_preprocessing_step)
        assert test_preprocessing_step.order == 5

    def test_update_step_column_name(
        self, client: TestClient, db: Session, test_preprocessing_step: PreprocessingStep
    ):
        """Test updating column_name"""
        update_payload = {
            "column_name": "col3"
        }

        response = client.put(
            f"/api/v1/preprocessing/{test_preprocessing_step.id}",
            json=update_payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["column_name"] == "col3"

    def test_update_step_not_found(self, client: TestClient):
        """Test updating non-existent step"""
        fake_id = uuid.uuid4()
        update_payload = {"order": 10}

        response = client.put(
            f"/api/v1/preprocessing/{fake_id}",
            json=update_payload
        )

        assert response.status_code == 404


class TestDeletePreprocessingStep:
    """Tests for DELETE /api/v1/preprocessing/{step_id} endpoint"""

    def test_delete_step_success(
        self, client: TestClient, db: Session, test_preprocessing_step: PreprocessingStep
    ):
        """Test successful step deletion"""
        step_id = test_preprocessing_step.id

        response = client.delete(f"/api/v1/preprocessing/{step_id}")

        assert response.status_code == 204

        # Verify it's deleted from database
        deleted_step = db.query(PreprocessingStep).filter(
            PreprocessingStep.id == step_id
        ).first()
        assert deleted_step is None

    def test_delete_step_not_found(self, client: TestClient):
        """Test deleting non-existent step"""
        fake_id = uuid.uuid4()

        response = client.delete(f"/api/v1/preprocessing/{fake_id}")

        assert response.status_code == 404


class TestReorderPreprocessingSteps:
    """Tests for POST /api/v1/preprocessing/reorder endpoint"""

    def test_reorder_steps_success(self, client: TestClient, db: Session, test_dataset: Dataset):
        """Test successful reordering of steps"""
        # Create multiple steps
        step1 = PreprocessingStep(
            id=uuid.uuid4(),
            dataset_id=test_dataset.id,
            step_type="outlier_detection",
            order=0
        )
        step2 = PreprocessingStep(
            id=uuid.uuid4(),
            dataset_id=test_dataset.id,
            step_type="missing_value_imputation",
            order=1
        )
        step3 = PreprocessingStep(
            id=uuid.uuid4(),
            dataset_id=test_dataset.id,
            step_type="scaling",
            order=2
        )
        db.add_all([step1, step2, step3])
        db.commit()

        # Reorder: reverse the order
        new_order = [str(step3.id), str(step2.id), str(step1.id)]

        response = client.post(
            f"/api/v1/preprocessing/reorder?dataset_id={test_dataset.id}",
            params={"step_ids": new_order}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

        # Verify new order
        assert data[0]["id"] == str(step3.id)
        assert data[0]["order"] == 0
        assert data[1]["id"] == str(step2.id)
        assert data[1]["order"] == 1
        assert data[2]["id"] == str(step1.id)
        assert data[2]["order"] == 2

        # Verify in database
        db.refresh(step1)
        db.refresh(step2)
        db.refresh(step3)
        assert step1.order == 2
        assert step2.order == 1
        assert step3.order == 0

    def test_reorder_steps_dataset_not_found(self, client: TestClient):
        """Test reordering with non-existent dataset"""
        fake_dataset_id = uuid.uuid4()

        response = client.post(
            f"/api/v1/preprocessing/reorder?dataset_id={fake_dataset_id}",
            params={"step_ids": [str(uuid.uuid4())]}
        )

        assert response.status_code == 404

    def test_reorder_steps_invalid_step_id(
        self, client: TestClient, db: Session, test_dataset: Dataset
    ):
        """Test reordering with invalid step ID"""
        # Create one step
        step1 = PreprocessingStep(
            id=uuid.uuid4(),
            dataset_id=test_dataset.id,
            step_type="scaling",
            order=0
        )
        db.add(step1)
        db.commit()

        # Try to reorder with non-existent step
        fake_step_id = str(uuid.uuid4())

        response = client.post(
            f"/api/v1/preprocessing/reorder?dataset_id={test_dataset.id}",
            params={"step_ids": [str(step1.id), fake_step_id]}
        )

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()


class TestPreprocessingStepIntegration:
    """Integration tests for full preprocessing pipeline workflow"""

    def test_full_pipeline_workflow(self, client: TestClient, db: Session, test_dataset: Dataset):
        """Test creating a complete preprocessing pipeline"""
        # Step 1: Outlier detection
        step1_payload = {
            "dataset_id": str(test_dataset.id),
            "step_type": "outlier_detection",
            "parameters": {"method": "iqr", "threshold": 1.5}
        }
        response1 = client.post("/api/v1/preprocessing/", json=step1_payload)
        assert response1.status_code == 201
        step1_id = response1.json()["id"]

        # Step 2: Missing value imputation
        step2_payload = {
            "dataset_id": str(test_dataset.id),
            "step_type": "missing_value_imputation",
            "parameters": {"strategy": "mean"},
            "column_name": "col2"
        }
        response2 = client.post("/api/v1/preprocessing/", json=step2_payload)
        assert response2.status_code == 201
        step2_id = response2.json()["id"]

        # Step 3: Scaling
        step3_payload = {
            "dataset_id": str(test_dataset.id),
            "step_type": "scaling",
            "parameters": {"method": "standard"}
        }
        response3 = client.post("/api/v1/preprocessing/", json=step3_payload)
        assert response3.status_code == 201
        step3_id = response3.json()["id"]

        # List all steps
        response = client.get(f"/api/v1/preprocessing/?dataset_id={test_dataset.id}")
        assert response.status_code == 200
        steps = response.json()
        assert len(steps) == 3

        # Update one step
        update_response = client.put(
            f"/api/v1/preprocessing/{step2_id}",
            json={"parameters": {"strategy": "median"}}
        )
        assert update_response.status_code == 200

        # Reorder steps
        new_order = [step2_id, step1_id, step3_id]
        reorder_response = client.post(
            f"/api/v1/preprocessing/reorder?dataset_id={test_dataset.id}",
            params={"step_ids": new_order}
        )
        assert reorder_response.status_code == 200

        # Delete one step
        delete_response = client.delete(f"/api/v1/preprocessing/{step3_id}")
        assert delete_response.status_code == 204

        # Verify final state
        final_response = client.get(f"/api/v1/preprocessing/?dataset_id={test_dataset.id}")
        final_steps = final_response.json()
        assert len(final_steps) == 2
        assert final_steps[0]["id"] == step2_id
        assert final_steps[1]["id"] == step1_id
