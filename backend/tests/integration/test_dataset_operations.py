"""
Integration tests for dataset operations

Tests the complete dataset upload, validation, and management workflow
"""

import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient
import uuid


class TestDatasetUpload:
    """Test dataset upload functionality"""

    def test_upload_classification_dataset_success(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test successful upload of classification dataset"""
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("classification_data.csv", f, "text/csv")},
                data={
                    "name": "Test Classification Dataset",
                    "description": "Classification test data"
                }
            )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["name"] == "Test Classification Dataset"
        assert data["description"] == "Classification test data"
        assert data["file_name"] == "classification_data.csv"
        assert data["row_count"] == 500
        assert data["column_count"] == 5
        assert "uploaded_at" in data
        assert data["status"] == "uploaded"

    def test_upload_regression_dataset_success(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        regression_dataset: Path
    ):
        """Test successful upload of regression dataset"""
        with open(regression_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("regression_data.csv", f, "text/csv")},
                data={
                    "name": "Test Regression Dataset",
                    "description": "Regression test data"
                }
            )

        assert response.status_code == 200
        data = response.json()

        assert data["row_count"] == 400
        assert data["column_count"] == 6

    def test_upload_without_authentication(
        self,
        client: TestClient,
        classification_dataset: Path
    ):
        """Test that upload requires authentication"""
        with open(classification_dataset, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "Test Dataset"}
            )

        assert response.status_code == 401

    def test_upload_invalid_file_format(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test upload with invalid file format"""
        # Create a non-CSV file
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("This is not a CSV file")

        with open(invalid_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.txt", f, "text/plain")},
                data={"name": "Invalid Dataset"}
            )

        assert response.status_code == 400
        assert "CSV" in response.json()["detail"]

    def test_upload_empty_dataset(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test upload of empty CSV file"""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("col1,col2\n")  # Header only

        with open(empty_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("empty.csv", f, "text/csv")},
                data={"name": "Empty Dataset"}
            )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_malformed_csv(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test upload of malformed CSV file"""
        malformed_file = tmp_path / "malformed.csv"
        malformed_file.write_text("col1,col2\nval1\nval2,val3,val4\n")

        with open(malformed_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("malformed.csv", f, "text/csv")},
                data={"name": "Malformed Dataset"}
            )

        # Should either reject or handle gracefully
        assert response.status_code in [400, 200]

    def test_upload_large_dataset(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        tmp_path: Path
    ):
        """Test upload of large dataset"""
        # Create a dataset with 10,000 rows
        large_data = {
            'feature1': range(10000),
            'feature2': range(10000, 20000),
            'target': [i % 2 for i in range(10000)]
        }
        df = pd.DataFrame(large_data)
        large_file = tmp_path / "large.csv"
        df.to_csv(large_file, index=False)

        with open(large_file, 'rb') as f:
            response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("large.csv", f, "text/csv")},
                data={"name": "Large Dataset"}
            )

        assert response.status_code == 200
        assert response.json()["row_count"] == 10000


class TestDatasetRetrieval:
    """Test dataset retrieval and listing"""

    def test_get_dataset_by_id(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test retrieving a dataset by ID"""
        # First upload
        with open(classification_dataset, 'rb') as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "Test Dataset"}
            )
        dataset_id = upload_response.json()["id"]

        # Then retrieve
        response = client.get(
            f"/api/v1/datasets/{dataset_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == "Test Dataset"

    def test_get_nonexistent_dataset(
        self,
        client: TestClient,
        auth_headers: Dict[str, str]
    ):
        """Test retrieving a dataset that doesn't exist"""
        fake_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/datasets/{fake_id}",
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_get_dataset_without_authentication(
        self,
        client: TestClient
    ):
        """Test that retrieval requires authentication"""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/datasets/{fake_id}")

        assert response.status_code == 401

    def test_get_dataset_owned_by_another_user(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path,
        db
    ):
        """Test that users cannot access datasets owned by others"""
        # Upload as first user
        with open(classification_dataset, 'rb') as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "User1 Dataset"}
            )
        dataset_id = upload_response.json()["id"]

        # Create second user and get their token
        from app.models.user import User
        from app.core.security import get_password_hash

        user2 = User(
            id=uuid.uuid4(),
            email="user2@example.com",
            password_hash=get_password_hash("password123"),
            is_active=True
        )
        db.add(user2)
        db.commit()

        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "user2@example.com", "password": "password123"}
        )
        user2_headers = {
            "Authorization": f"Bearer {login_response.json()['access_token']}"
        }

        # Try to access first user's dataset
        response = client.get(
            f"/api/v1/datasets/{dataset_id}",
            headers=user2_headers
        )

        assert response.status_code == 403

    def test_list_user_datasets(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path,
        regression_dataset: Path
    ):
        """Test listing all datasets for current user"""
        # Upload multiple datasets
        with open(classification_dataset, 'rb') as f:
            client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("classification.csv", f, "text/csv")},
                data={"name": "Classification Dataset"}
            )

        with open(regression_dataset, 'rb') as f:
            client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("regression.csv", f, "text/csv")},
                data={"name": "Regression Dataset"}
            )

        # List datasets
        response = client.get(
            "/api/v1/datasets/",
            headers=auth_headers
        )

        assert response.status_code == 200
        datasets = response.json()
        assert len(datasets) >= 2
        names = [d["name"] for d in datasets]
        assert "Classification Dataset" in names
        assert "Regression Dataset" in names

    def test_list_datasets_pagination(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test dataset listing with pagination"""
        # Upload 5 datasets
        for i in range(5):
            with open(classification_dataset, 'rb') as f:
                client.post(
                    "/api/v1/datasets/upload",
                    headers=auth_headers,
                    files={"file": (f"dataset{i}.csv", f, "text/csv")},
                    data={"name": f"Dataset {i}"}
                )

        # Get first page
        response = client.get(
            "/api/v1/datasets/?skip=0&limit=3",
            headers=auth_headers
        )

        assert response.status_code == 200
        page1 = response.json()
        assert len(page1) <= 3

        # Get second page
        response = client.get(
            "/api/v1/datasets/?skip=3&limit=3",
            headers=auth_headers
        )

        assert response.status_code == 200
        page2 = response.json()
        assert len(page2) >= 2


class TestDatasetDeletion:
    """Test dataset deletion"""

    def test_delete_dataset_success(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test successful dataset deletion"""
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "To Delete"}
            )
        dataset_id = upload_response.json()["id"]

        # Delete dataset
        response = client.delete(
            f"/api/v1/datasets/{dataset_id}",
            headers=auth_headers
        )

        assert response.status_code == 200

        # Verify it's gone
        get_response = client.get(
            f"/api/v1/datasets/{dataset_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404

    def test_delete_dataset_without_authentication(
        self,
        client: TestClient
    ):
        """Test that deletion requires authentication"""
        fake_id = str(uuid.uuid4())
        response = client.delete(f"/api/v1/datasets/{fake_id}")

        assert response.status_code == 401

    def test_delete_dataset_owned_by_another_user(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path,
        db
    ):
        """Test that users cannot delete datasets owned by others"""
        # Upload as first user
        with open(classification_dataset, 'rb') as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "User1 Dataset"}
            )
        dataset_id = upload_response.json()["id"]

        # Create second user
        from app.models.user import User
        from app.core.security import get_password_hash

        user2 = User(
            id=uuid.uuid4(),
            email="user2_delete@example.com",
            password_hash=get_password_hash("password123"),
            is_active=True
        )
        db.add(user2)
        db.commit()

        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": "user2_delete@example.com", "password": "password123"}
        )
        user2_headers = {
            "Authorization": f"Bearer {login_response.json()['access_token']}"
        }

        # Try to delete first user's dataset
        response = client.delete(
            f"/api/v1/datasets/{dataset_id}",
            headers=user2_headers
        )

        assert response.status_code == 403

    def test_delete_nonexistent_dataset(
        self,
        client: TestClient,
        auth_headers: Dict[str, str]
    ):
        """Test deleting a dataset that doesn't exist"""
        fake_id = str(uuid.uuid4())
        response = client.delete(
            f"/api/v1/datasets/{fake_id}",
            headers=auth_headers
        )

        assert response.status_code == 404


class TestDatasetPreview:
    """Test dataset preview functionality"""

    def test_preview_dataset(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test previewing dataset contents"""
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "Preview Test"}
            )
        dataset_id = upload_response.json()["id"]

        # Get preview
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/preview?rows=10",
            headers=auth_headers
        )

        assert response.status_code == 200
        preview = response.json()

        assert "columns" in preview
        assert "data" in preview
        assert "total_rows" in preview
        assert len(preview["data"]) <= 10
        assert preview["total_rows"] == 500

    def test_preview_with_custom_row_limit(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test preview with custom row limit"""
        with open(classification_dataset, 'rb') as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "Preview Test"}
            )
        dataset_id = upload_response.json()["id"]

        response = client.get(
            f"/api/v1/datasets/{dataset_id}/preview?rows=5",
            headers=auth_headers
        )

        assert response.status_code == 200
        assert len(response.json()["data"]) <= 5


class TestDatasetStatistics:
    """Test dataset statistics and profiling"""

    def test_get_dataset_statistics(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test retrieving dataset statistics"""
        # Upload dataset
        with open(classification_dataset, 'rb') as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "Stats Test"}
            )
        dataset_id = upload_response.json()["id"]

        # Get statistics
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/statistics",
            headers=auth_headers
        )

        assert response.status_code == 200
        stats = response.json()

        assert "numeric_columns" in stats
        assert "categorical_columns" in stats
        assert "missing_values" in stats
        assert "correlation_matrix" in stats or "correlations" in stats

    def test_statistics_include_missing_values(
        self,
        client: TestClient,
        auth_headers: Dict[str, str],
        classification_dataset: Path
    ):
        """Test that statistics correctly identify missing values"""
        with open(classification_dataset, 'rb') as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                headers=auth_headers,
                files={"file": ("test.csv", f, "text/csv")},
                data={"name": "Missing Values Test"}
            )
        dataset_id = upload_response.json()["id"]

        response = client.get(
            f"/api/v1/datasets/{dataset_id}/statistics",
            headers=auth_headers
        )

        assert response.status_code == 200
        stats = response.json()

        # Classification dataset fixture has missing values
        assert stats["missing_values"]["sepal_length"] > 0
        assert stats["missing_values"]["petal_width"] > 0
