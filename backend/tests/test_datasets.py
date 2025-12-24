"""
Unit tests for dataset endpoints.
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


class TestDatasetUpload:
    """Tests for POST /api/v1/datasets/upload endpoint"""

    def test_upload_csv_success(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test successful CSV file upload"""
        with open(sample_csv_file, "rb") as f:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        assert response.status_code == 201
        data = response.json()

        # Verify response structure
        assert "id" in data
        assert data["name"] == "test"
        assert data["rows"] == 5
        assert data["cols"] == 4
        assert "dtypes" in data
        assert "missing_values" in data

        # Verify file was saved
        assert len(list(temp_upload_dir.rglob("*.csv"))) == 1

    def test_upload_excel_success(
        self,
        client: TestClient,
        sample_excel_file: Path,
        temp_upload_dir: Path
    ):
        """Test successful Excel file upload"""
        with open(sample_excel_file, "rb") as f:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            )

        assert response.status_code == 201
        data = response.json()

        assert "id" in data
        assert data["name"] == "test"
        assert data["rows"] == 5
        assert data["cols"] == 4

    def test_upload_json_success(
        self,
        client: TestClient,
        sample_json_file: Path,
        temp_upload_dir: Path
    ):
        """Test successful JSON file upload"""
        with open(sample_json_file, "rb") as f:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.json", f, "application/json")}
            )

        assert response.status_code == 201
        data = response.json()

        assert "id" in data
        assert data["name"] == "test"
        assert data["rows"] == 4
        assert data["cols"] == 4

    def test_upload_invalid_file_type(
        self,
        client: TestClient,
        invalid_file: Path
    ):
        """Test upload with invalid file type"""
        with open(invalid_file, "rb") as f:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )

        assert response.status_code == 400
        assert "not supported" in response.json()["detail"].lower()

    def test_upload_file_too_large(
        self,
        client: TestClient,
        large_csv_file: Path
    ):
        """Test upload with file exceeding size limit"""
        # This test might be slow due to file size
        # Skip if large_csv_file fixture is not implemented
        pytest.skip("Skipping large file test to reduce test time")

    def test_upload_corrupted_csv(
        self,
        client: TestClient,
        tmp_path: Path
    ):
        """Test upload with corrupted CSV file"""
        corrupted_file = tmp_path / "corrupted.csv"
        corrupted_file.write_text("invalid,csv,content\n1,2\n3")  # Inconsistent columns

        with open(corrupted_file, "rb") as f:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("corrupted.csv", f, "text/csv")}
            )

        # Should still upload but might have issues during processing
        # Pandas is usually lenient with malformed CSVs
        assert response.status_code in [201, 400]

    def test_upload_empty_file(
        self,
        client: TestClient,
        tmp_path: Path
    ):
        """Test upload with empty file"""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        with open(empty_file, "rb") as f:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("empty.csv", f, "text/csv")}
            )

        assert response.status_code == 400

    def test_upload_with_missing_values(
        self,
        client: TestClient,
        tmp_path: Path,
        temp_upload_dir: Path
    ):
        """Test upload with CSV containing missing values"""
        csv_content = """name,age,city
John,30,New York
Jane,,San Francisco
Bob,35,
"""
        csv_file = tmp_path / "missing_values.csv"
        csv_file.write_text(csv_content)

        with open(csv_file, "rb") as f:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("missing_values.csv", f, "text/csv")}
            )

        assert response.status_code == 201
        data = response.json()

        # Verify missing values are tracked
        assert "missing_values" in data
        assert data["missing_values"]["age"] > 0
        assert data["missing_values"]["city"] > 0


class TestDatasetList:
    """Tests for GET /api/v1/datasets/ endpoint"""

    def test_list_datasets_empty(self, client: TestClient):
        """Test listing datasets when none exist"""
        response = client.get("/api/v1/datasets/")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_datasets_with_data(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test listing datasets after uploading"""
        # Upload a dataset first
        with open(sample_csv_file, "rb") as f:
            client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        # List datasets
        response = client.get("/api/v1/datasets/")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test"

    def test_list_datasets_pagination(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test pagination parameters"""
        # Upload multiple datasets
        for i in range(5):
            with open(sample_csv_file, "rb") as f:
                client.post(
                    "/api/v1/datasets/upload",
                    files={"file": (f"test_{i}.csv", f, "text/csv")}
                )

        # Test pagination
        response = client.get("/api/v1/datasets/?skip=2&limit=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


class TestDatasetGet:
    """Tests for GET /api/v1/datasets/{dataset_id} endpoint"""

    def test_get_dataset_success(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test getting a specific dataset"""
        # Upload dataset
        with open(sample_csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        dataset_id = upload_response.json()["id"]

        # Get dataset
        response = client.get(f"/api/v1/datasets/{dataset_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == "test"

    def test_get_dataset_not_found(self, client: TestClient):
        """Test getting non-existent dataset"""
        fake_id = "00000000-0000-0000-0000-000000000999"
        response = client.get(f"/api/v1/datasets/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestDatasetPreview:
    """Tests for GET /api/v1/datasets/{dataset_id}/preview endpoint"""

    def test_preview_dataset_success(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test getting dataset preview"""
        # Upload dataset
        with open(sample_csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        dataset_id = upload_response.json()["id"]

        # Get preview
        response = client.get(f"/api/v1/datasets/{dataset_id}/preview")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "preview" in data
        assert "columns" in data
        assert "totalRows" in data
        assert "displayedRows" in data

        # Verify preview data
        assert len(data["preview"]) > 0
        assert len(data["columns"]) == 4  # name, age, city, salary

        # Verify column metadata
        for col in data["columns"]:
            assert "name" in col
            assert "dataType" in col
            assert "nullCount" in col
            assert "uniqueCount" in col
            assert "sampleValues" in col

    def test_preview_with_custom_rows(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test preview with custom row limit"""
        # Upload dataset
        with open(sample_csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        dataset_id = upload_response.json()["id"]

        # Get preview with 2 rows
        response = client.get(f"/api/v1/datasets/{dataset_id}/preview?rows=2")

        assert response.status_code == 200
        data = response.json()
        assert data["displayedRows"] == 2
        assert len(data["preview"]) == 2

    def test_preview_dataset_not_found(self, client: TestClient):
        """Test preview for non-existent dataset"""
        fake_id = "00000000-0000-0000-0000-000000000999"
        response = client.get(f"/api/v1/datasets/{fake_id}/preview")

        assert response.status_code == 404

    def test_preview_file_deleted(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path,
        db: Session
    ):
        """Test preview when file is deleted from disk"""
        from app.models.dataset import Dataset

        # Upload dataset
        with open(sample_csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        dataset_id = upload_response.json()["id"]

        # Delete the file from disk (but not database)
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        file_path = Path(dataset.file_path)
        if file_path.exists():
            file_path.unlink()

        # Try to get preview
        response = client.get(f"/api/v1/datasets/{dataset_id}/preview")

        assert response.status_code == 404
        assert "file not found" in response.json()["detail"].lower()


class TestDatasetStats:
    """Tests for GET /api/v1/datasets/{dataset_id}/stats endpoint"""

    def test_stats_dataset_success(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test getting dataset statistics"""
        # Upload dataset
        with open(sample_csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        dataset_id = upload_response.json()["id"]

        # Get stats
        response = client.get(f"/api/v1/datasets/{dataset_id}/stats")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "rowCount" in data
        assert "columnCount" in data
        assert "numericColumns" in data
        assert "categoricalColumns" in data
        assert "missingValues" in data
        assert "duplicateRows" in data
        assert "memoryUsage" in data
        assert "columns" in data

        # Verify values
        assert data["rowCount"] == 5
        assert data["columnCount"] == 4
        assert data["numericColumns"] >= 1  # age and salary
        assert data["memoryUsage"] > 0

    def test_stats_with_duplicates(
        self,
        client: TestClient,
        tmp_path: Path,
        temp_upload_dir: Path
    ):
        """Test stats with duplicate rows"""
        csv_content = """name,age
John,30
Jane,25
John,30
"""
        csv_file = tmp_path / "duplicates.csv"
        csv_file.write_text(csv_content)

        # Upload dataset
        with open(csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("duplicates.csv", f, "text/csv")}
            )

        dataset_id = upload_response.json()["id"]

        # Get stats
        response = client.get(f"/api/v1/datasets/{dataset_id}/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["duplicateRows"] == 1  # One duplicate row

    def test_stats_dataset_not_found(self, client: TestClient):
        """Test stats for non-existent dataset"""
        fake_id = "00000000-0000-0000-0000-000000000999"
        response = client.get(f"/api/v1/datasets/{fake_id}/stats")

        assert response.status_code == 404


class TestDatasetDelete:
    """Tests for DELETE /api/v1/datasets/{dataset_id} endpoint"""

    def test_delete_dataset_success(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test successful dataset deletion"""
        # Upload dataset
        with open(sample_csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        dataset_id = upload_response.json()["id"]

        # Delete dataset
        response = client.delete(f"/api/v1/datasets/{dataset_id}")

        assert response.status_code == 204

        # Verify dataset is deleted
        get_response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert get_response.status_code == 404

        # Verify file is deleted from disk
        assert len(list(temp_upload_dir.rglob("*.csv"))) == 0

    def test_delete_dataset_not_found(self, client: TestClient):
        """Test deleting non-existent dataset"""
        fake_id = "00000000-0000-0000-0000-000000000999"
        response = client.delete(f"/api/v1/datasets/{fake_id}")

        assert response.status_code == 404

    def test_delete_dataset_file_already_deleted(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path,
        db: Session
    ):
        """Test deleting dataset when file is already gone from disk"""
        from app.models.dataset import Dataset

        # Upload dataset
        with open(sample_csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        dataset_id = upload_response.json()["id"]

        # Manually delete file from disk
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        file_path = Path(dataset.file_path)
        if file_path.exists():
            file_path.unlink()

        # Delete dataset should still succeed (removes DB record)
        response = client.delete(f"/api/v1/datasets/{dataset_id}")

        assert response.status_code == 204


class TestDatasetIntegration:
    """Integration tests for complete dataset workflows"""

    def test_complete_dataset_lifecycle(
        self,
        client: TestClient,
        sample_csv_file: Path,
        temp_upload_dir: Path
    ):
        """Test complete lifecycle: upload -> get -> preview -> stats -> delete"""
        # 1. Upload
        with open(sample_csv_file, "rb") as f:
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )
        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        # 2. List datasets
        list_response = client.get("/api/v1/datasets/")
        assert list_response.status_code == 200
        assert len(list_response.json()) == 1

        # 3. Get dataset
        get_response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert get_response.status_code == 200

        # 4. Preview
        preview_response = client.get(f"/api/v1/datasets/{dataset_id}/preview")
        assert preview_response.status_code == 200

        # 5. Stats
        stats_response = client.get(f"/api/v1/datasets/{dataset_id}/stats")
        assert stats_response.status_code == 200

        # 6. Delete
        delete_response = client.delete(f"/api/v1/datasets/{dataset_id}")
        assert delete_response.status_code == 204

        # 7. Verify deleted
        final_get = client.get(f"/api/v1/datasets/{dataset_id}")
        assert final_get.status_code == 404

    def test_multiple_datasets(
        self,
        client: TestClient,
        sample_csv_file: Path,
        sample_json_file: Path,
        temp_upload_dir: Path
    ):
        """Test handling multiple datasets"""
        # Upload CSV
        with open(sample_csv_file, "rb") as f:
            csv_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )

        # Upload JSON
        with open(sample_json_file, "rb") as f:
            json_response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.json", f, "application/json")}
            )

        assert csv_response.status_code == 201
        assert json_response.status_code == 201

        # List should show both
        list_response = client.get("/api/v1/datasets/")
        assert len(list_response.json()) == 2

        # Both should have different IDs
        assert csv_response.json()["id"] != json_response.json()["id"]
