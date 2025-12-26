"""
Test configuration and fixtures for pytest.
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.core.config import settings


# Use test database - defaults to NeonDB from settings, can be overridden
test_db_url = os.getenv("TEST_DATABASE_URL", settings.DATABASE_URL)

# Create engine with appropriate settings
engine = create_engine(
    test_db_url,
    pool_pre_ping=True,
    pool_size=2,  # Small pool for testing
    max_overflow=5
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """
    Create a fresh database for each test.
    """
    # Create tables
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Drop tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db: Session) -> Generator[TestClient, None, None]:
    """
    Create a test client with database dependency override.
    """
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def temp_upload_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for file uploads during testing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Override the upload directory setting
        original_upload_dir = settings.UPLOAD_DIR
        settings.UPLOAD_DIR = str(temp_path)

        yield temp_path

        # Restore original setting
        settings.UPLOAD_DIR = original_upload_dir


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """
    Create a sample CSV file for testing.
    """
    csv_content = """name,age,city,salary
John Doe,30,New York,75000
Jane Smith,25,San Francisco,85000
Bob Johnson,35,Chicago,65000
Alice Williams,28,Boston,72000
Charlie Brown,32,Seattle,78000
"""

    csv_file = tmp_path / "test_dataset.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_excel_file(tmp_path: Path) -> Path:
    """
    Create a sample Excel file for testing.
    """
    import pandas as pd

    data = {
        'product': ['A', 'B', 'C', 'D', 'E'],
        'price': [10.5, 20.0, 15.75, 8.99, 12.50],
        'quantity': [100, 50, 75, 200, 150],
        'category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Clothing']
    }

    df = pd.DataFrame(data)
    excel_file = tmp_path / "test_dataset.xlsx"
    df.to_excel(excel_file, index=False)
    return excel_file


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    """
    Create a sample JSON file for testing.
    """
    import json

    data = [
        {"id": 1, "name": "Item 1", "value": 100, "active": True},
        {"id": 2, "name": "Item 2", "value": 200, "active": False},
        {"id": 3, "name": "Item 3", "value": 150, "active": True},
        {"id": 4, "name": "Item 4", "value": 300, "active": True},
    ]

    json_file = tmp_path / "test_dataset.json"
    json_file.write_text(json.dumps(data))
    return json_file


@pytest.fixture
def large_csv_file(tmp_path: Path) -> Path:
    """
    Create a large CSV file for testing file size limits.
    """
    import pandas as pd
    import numpy as np

    # Create a large dataset (simulate ~150MB file)
    rows = 1000000
    data = {
        'col1': np.random.randint(0, 1000, rows),
        'col2': np.random.rand(rows),
        'col3': [f'text_{i}' for i in range(rows)],
    }

    df = pd.DataFrame(data)
    large_file = tmp_path / "large_dataset.csv"
    df.to_csv(large_file, index=False)
    return large_file


@pytest.fixture
def invalid_file(tmp_path: Path) -> Path:
    """
    Create an invalid file (not CSV/Excel/JSON).
    """
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("This is not a valid dataset file")
    return invalid_file
